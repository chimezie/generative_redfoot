import asyncio

import httpx
import yaml

from generative_redfoot.cli import build_app, make_dispatcher
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend

LOOM = '''lang = "en"

[doc]
_ = "Summarize: {document_text}"
_m = ["document_text"]
'''


def make_client(tmp_path, body_program=None, backend=None):
    loom = tmp_path / 'svc.loom.toml'
    loom.write_text(LOOM)
    pdl = body_program or f'''
server:
  host: 127.0.0.1
  port: 1
  request_body_marker: document_text
  path: /gen
  content_type: text/plain
description: svc
text:
- role: user
  read_from_wordloom: {loom}
  items: doc
  contribute: [context]
- model: some/model
'''
    backend = backend or InMemoryBackend(['the summary'])
    app = build_app(yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(), {})
    # Callers inject the transport, so the app runs in-process with no server or sockets
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test'), backend


def test_text_body_is_bound_to_marker(tmp_path):
    client, backend = make_client(tmp_path)

    async def go():
        async with client:
            return await client.post('/gen', content=b'abc', headers={'content-type': 'text/plain'})

    response = asyncio.run(go())
    assert response.status_code == 200 and response.text == 'the summary'
    assert b'Summarize: abc' in bytes(backend.calls[0]['tokens'])


def test_wrong_content_type_is_415(tmp_path):
    client, _ = make_client(tmp_path)

    async def go():
        async with client:
            return await client.post('/gen', content=b'{}', headers={'content-type': 'application/json'})

    assert asyncio.run(go()).status_code == 415


def test_multipart_file_field_is_bound(tmp_path):
    client, backend = make_client(tmp_path)

    async def go():
        async with client:
            return await client.post('/gen', files={'document_text': ('x.txt', b'from file', 'text/plain')})

    assert asyncio.run(go()).status_code == 200
    assert b'Summarize: from file' in bytes(backend.calls[0]['tokens'])


def test_program_error_is_500(tmp_path):
    pdl = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- model: some/model
'''
    client, _ = make_client(tmp_path, body_program=pdl)

    async def go():
        async with client:
            return await client.post('/gen', content=b'abc')

    response = asyncio.run(go())
    assert response.status_code == 500 and 'No context' in response.text
