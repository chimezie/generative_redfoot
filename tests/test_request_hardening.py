'''Untrusted request data: form fields, `read` without input, and raw PDF sources (Stage 1).'''

import asyncio
import logging
from pathlib import Path

import httpx
import pytest
import yaml

from generative_redfoot.cli import build_app, make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend
from generative_redfoot.object_pdl_model import PDFRead

TEST_PDF = Path(__file__).parent / 'Test.pdf'

ECHO_VARIABLE = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- text: "${ doc_path }"
  contribute: [context]
- model: some/model
'''

ECHO_BODY = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- text: "${ document_text }"
  contribute: [context]
- model: some/model
'''


def client_for(pdl, base_context=None, backend=None):
    backend = backend or InMemoryBackend(['ok'])
    app = build_app(
        yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(), {}, base_context or {}
    )
    # The transport is injected, so the app runs in-process with no server or sockets
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test'), backend


def post(client, **kwargs):
    async def go():
        async with client:
            return await client.post('/gen', **kwargs)

    return asyncio.run(go())


def program_from(pdl, ctx=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), InMemoryBackend(), GenerationDefaults(), {},
        ctx if ctx is not None else {},
    )


# --- 1A: a form field never overrides operator variables or internal state -------------------------------------------

@pytest.mark.parametrize('form_kind', ['urlencoded', 'multipart'])
def test_operator_variable_wins_over_a_form_field(form_kind, caplog):
    client, backend = client_for(ECHO_VARIABLE, base_context={'doc_path': 'safe'})
    with caplog.at_level(logging.WARNING):
        if form_kind == 'urlencoded':
            response = post(client, data={'doc_path': 'evil'})
        else:
            response = post(client, files={'doc_path': (None, 'evil')})

    assert response.status_code == 200
    tokens = bytes(backend.calls[0]['tokens'])
    assert b'safe' in tokens and b'evil' not in tokens
    assert 'doc_path' in caplog.text and 'evil' not in caplog.text  # names are logged, values are not


def test_form_field_named_underscore_is_ignored(caplog):
    client, backend = client_for(ECHO_BODY)
    with caplog.at_level(logging.WARNING):
        response = post(client, data={'_': 'abc', 'document_text': 'body'})

    assert response.status_code == 200
    assert b'body' in bytes(backend.calls[0]['tokens'])
    assert 'Ignoring form fields' in caplog.text and '_' in caplog.text


def test_marker_field_still_carries_the_body():
    client, backend = client_for(ECHO_BODY)
    response = post(client, data={'document_text': 'hello'})

    assert response.status_code == 200
    assert b'hello' in bytes(backend.calls[0]['tokens'])


# --- 1B: a written `read` value is a path; only `read:` with no value may prompt ---------------------------------------

def test_written_read_value_that_evaluates_empty_is_an_author_error(monkeypatch):
    def must_not_prompt(prompt):
        raise AssertionError('an empty written value must not fall back to input')

    monkeypatch.setattr('builtins.input', must_not_prompt)
    program = program_from('description: t\ntext:\n- read: "${ p }"\n', ctx={'p': ''})
    with pytest.raises(ValueError) as excinfo:
        program.execute()

    message = str(excinfo.value)
    assert '${ p }' in message and 'empty/invalid path' in message


def test_written_read_value_that_is_not_a_path_is_an_author_error():
    program = program_from('description: t\ntext:\n- read: "${ n }"\n', ctx={'n': 5})
    with pytest.raises(ValueError, match='empty/invalid path'):
        program.execute()


def test_read_without_a_value_still_prompts_locally(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda prompt: 'typed')
    program = program_from('description: t\ntext:\n- read:\n')
    assert program.execute() == 'typed'


def test_service_read_without_a_value_fails_instead_of_prompting(monkeypatch):
    def must_not_prompt(prompt):
        raise AssertionError('the service must not wait for keyboard input')

    monkeypatch.setattr('builtins.input', must_not_prompt)
    pdl = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- read:
  contribute: [context]
- model: some/model
'''
    client, _ = client_for(pdl)
    response = post(client, content=b'body', headers={'content-type': 'text/plain'})

    assert response.status_code == 500
    assert 'standard input' in response.text and 'not available in a service' in response.text


# --- 1C: raw PDF reads never open a path -----------------------------------------------------------------------------

def test_pdf_bytes_never_treats_a_string_as_a_path(tmp_path):
    f = tmp_path / 'x.pdf'
    f.write_bytes(b'%PDF-1.4 data')

    assert PDFRead._pdf_bytes(str(f)) == str(f).encode('utf-8')
    assert PDFRead._pdf_bytes(b'raw') == b'raw'
    assert PDFRead._pdf_bytes(bytearray(b'raw')) == b'raw'
    assert PDFRead._pdf_bytes('%PDF-1.4 inline') == b'%PDF-1.4 inline'
    assert PDFRead._pdf_bytes(12) == b'12'


def test_service_cannot_make_a_raw_pdf_read_open_a_file():
    pdl = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- PDF_raw_read_txt: "${ document_text }"
  contribute: [context]
- model: some/model
'''
    client, _ = client_for(pdl)
    response = post(client, content=str(TEST_PDF).encode('utf-8'), headers={'content-type': 'text/plain'})

    assert response.status_code == 500
    assert 'invalid or corrupted' in response.text
    assert 'quick brown fox' not in response.text
