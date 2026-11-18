'''`read` file paths, stdin prompts and multiline input (Stage 3).'''

import asyncio
import io
import sys

import httpx
import pytest
import yaml
from click.testing import CliRunner

from generative_redfoot.cli import build_app, main, make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def program_from(pdl, backend=None, ctx=None, base_dir=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend or InMemoryBackend(), GenerationDefaults(), {},
        ctx if ctx is not None else {}, base_dir=base_dir,
    )


def test_read_relative_path_uses_base_dir(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'note.txt').write_text('from note', encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pdl = '''
description: t
text:
- read: ./note.txt
'''
    assert program_from(pdl, base_dir=base).execute() == 'from note'


def test_read_expression_path_uses_base_dir(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'note.txt').write_text('expression note', encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pdl = '''
description: t
text:
- read: "${ fname }"
'''
    assert program_from(pdl, ctx={'fname': 'note.txt'}, base_dir=base).execute() == 'expression note'


def test_read_without_value_uses_default_message(monkeypatch):
    seen = []

    def fake_input(prompt):
        seen.append(prompt)
        return 'hello'

    monkeypatch.setattr('builtins.input', fake_input)
    pdl = '''
description: t
text:
- read:
'''
    assert program_from(pdl).execute() == 'hello'
    assert seen == ['How can I help you?: ']


def test_read_multiline_reads_stdin_to_eof(monkeypatch):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('first\nsecond\n'))
    pdl = '''
description: t
text:
- read:
  multiline: true
  message: "Paste: "
'''
    assert program_from(pdl).execute() == 'first\nsecond\n'


def test_read_missing_file_names_the_resolved_path(tmp_path):
    pdl = '''
description: t
text:
- read: ./missing.txt
'''
    with pytest.raises(FileNotFoundError) as excinfo:
        program_from(pdl, base_dir=tmp_path).execute()
    assert str(tmp_path / 'missing.txt') in str(excinfo.value)


def test_legacy_read_reference_keeps_its_given_path(tmp_path, monkeypatch):
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    (cwd / 'note.txt').write_text('cwd text', encoding='utf-8')
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'note.txt').write_text('base text', encoding='utf-8')
    monkeypatch.chdir(cwd)
    pdl = '''
description: t
text:
- read: { $name }
'''
    assert program_from(pdl, ctx={'name': 'note.txt'}, base_dir=base).execute() == 'cwd text'


def test_service_program_receives_base_dir(tmp_path):
    (tmp_path / 'note.txt').write_text('service note', encoding='utf-8')
    pdl = '''
server: {host: 127.0.0.1, port: 1, request_body_marker: document_text, path: /gen}
description: svc
text:
- read: ./note.txt
  contribute: [context]
- model: some/model
'''
    backend = InMemoryBackend(['ok'])
    app = build_app(yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(), {}, base_dir=tmp_path)

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            return await client.post('/gen', content=b'body', headers={'content-type': 'text/plain'})

    response = asyncio.run(go())
    assert response.status_code == 200 and response.text == 'ok'
    assert b'service note' in bytes(backend.calls[0]['tokens'])


def test_cli_passes_pdl_directory_as_base_dir(tmp_path, monkeypatch):
    (tmp_path / 'note.txt').write_text('cli note', encoding='utf-8')
    pdl_file = tmp_path / 'prog.pdl'
    pdl_file.write_text('description: t\ntext:\n- read: ./note.txt\n', encoding='utf-8')
    monkeypatch.setattr('generative_redfoot.cli.MLXBackend', lambda **kwargs: InMemoryBackend())

    result = CliRunner().invoke(main, [str(pdl_file)])

    assert result.exit_code == 0
    assert 'cli note' in result.output
