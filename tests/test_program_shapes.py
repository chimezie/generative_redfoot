'''
Snapshot tests for the program shapes the interpreter must keep working.

These record today's exact behavior (tokens sent to the backend and the final message list) so later changes to
expressions, results, roles and repeat cannot silently break the prefix-cache and chat flows.
'''

from pathlib import Path

import pytest
import yaml

from generative_redfoot import prompts
from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend
from generative_redfoot.pdf_text import extract_pdf_text

TEST_PDF = Path(__file__).parent / 'Test.pdf'


def b(text):
    # The fake tokenizer is byte-level, so expected token ids are just the UTF-8 bytes
    return list(text.encode('utf-8'))


def program_from(pdl, backend, cache_lookup=None, ctx=None, **defaults):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(**defaults), cache_lookup or {},
        ctx if ctx is not None else {},
    )


# --- named prefix cache fed by a PDF block ---------------------------------------------------------------------------

def pdf_cache_pdl(file):
    return f'''
cache:
- text:
  - role: system
    text: You code.
    contribute: [context]
  - role: user
    text: "Instructions. {{document_text}} Trailing."
    contribute: [context]
  - content_model: some/model
    name: pc
    file: {file}
    prefix_marker: document_text
description: t
text:
- role: user
  text:
  - PDF_raw_read_txt: {{ $document_text }}
  contribute: [context]
- model: pc
'''


PREFIX = '<|system|>You code.<|end|><|user|>Instructions. '


def prepared_pdf(tmp_path, backend, pdf_bytes):
    '''
    Prepare the cache locally, then bind the request document. Cache preparation maps the prefix marker name to
    itself so it survives rendering, so the document variable must be set afterwards (as a CLI run with -v would).
    '''
    file = str(tmp_path / 'pc.json')
    ctx = {}
    p = program_from(pdf_cache_pdl(file), backend, ctx=ctx)
    prepare_caches(p, ctx, verbose=False)
    ctx['document_text'] = pdf_bytes
    p.evaluation_environment = ctx
    return p, file


def test_pdf_fed_named_cache_local_flow(tmp_path):
    pytest.importorskip('pymupdf')
    pdf_bytes = TEST_PDF.read_bytes()
    pdf_text = extract_pdf_text(pdf_bytes, ocr=False)
    backend = InMemoryBackend(['done'])
    p, _ = prepared_pdf(tmp_path, backend, pdf_bytes)

    p.execute()

    assert backend.calls[0]['tokens'] == b(pdf_text + '<|end|><|assistant|>')
    assert backend.calls[0]['cache_len_before'] == len(b(PREFIX))
    assert p.evaluation_environment['_'] == [
        {'role': 'system', 'content': 'You code.'},
        {'role': 'user', 'content': 'Instructions. ' + pdf_text},
        {'role': 'assistant', 'content': 'done'},
    ]


def test_pdf_fed_named_cache_per_request_flow(tmp_path):
    pytest.importorskip('pymupdf')
    pdf_bytes = TEST_PDF.read_bytes()
    pdf_text = extract_pdf_text(pdf_bytes, ocr=False)
    backend = InMemoryBackend(['done'])
    _, file = prepared_pdf(tmp_path, InMemoryBackend(), pdf_bytes)
    # A per-request program starts with an empty context; the cached messages supply the prefix
    p = program_from(pdf_cache_pdl(file), backend, cache_lookup={'pc': file}, ctx={'document_text': pdf_bytes})

    p.execute()

    assert backend.calls[0]['tokens'] == b(pdf_text + '<|end|><|assistant|>')
    assert backend.calls[0]['cache_len_before'] == len(b(PREFIX))
    assert p.evaluation_environment['_'] == [
        {'role': 'user', 'content': pdf_text},
        {'role': 'assistant', 'content': 'done'},
    ]


# --- chat loop -------------------------------------------------------------------------------------------------------

CHAT = '''
cache: "*"
description: t
text:
- read:
  message: "Prompt: "
- repeat:
    text:
    - model: some/model
    - read:
      def: question
      message: "Say: "
  until: "${ question == 'quit' }"
'''


def test_chat_loop_with_internal_cache(monkeypatch):
    backend = InMemoryBackend(['hi there'])
    p = program_from(CHAT, backend)
    inputs = iter(['Hello', 'quit'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(inputs))

    p.execute()

    assert len(backend.calls) == 1
    assert backend.calls[0]['tokens'] == b('<|user|>Hello<|end|><|assistant|>')
    assert backend.calls[0]['cache_len_before'] == 0
    assert p.evaluation_environment['question'] == 'quit'
    assert p.evaluation_environment['_'] == [
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'hi there'},
        {'role': 'user', 'content': 'quit'},
    ]


# --- explicit roles --------------------------------------------------------------------------------------------------

EXPLICIT_ROLES = '''
description: t
text:
- role: system
  text: Be brief.
  contribute: [context]
- role: user
  text: Hello
  contribute: [context]
- model: some/model
'''


def test_explicit_roles_render_one_message_each():
    backend = InMemoryBackend(['hi there'])
    p = program_from(EXPLICIT_ROLES, backend)

    p.execute()

    assert backend.calls[0]['tokens'] == b('<|system|>Be brief.<|end|><|user|>Hello<|end|><|assistant|>')
    assert p.evaluation_environment['_'] == [
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'hi there'},
    ]


# --- Wordloom markers ------------------------------------------------------------------------------------------------

LOOM = '''lang = "en"

[greeting]
_ = "Hello {name}"
_m = ["name"]
'''


def test_wordloom_marker_from_initial_context(tmp_path, monkeypatch):
    monkeypatch.setenv(prompts.ROOT_ENV, str(tmp_path))
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'p.loom.toml').write_text(LOOM)
    pdl = '''
description: t
text:
- read_from_wordloom: p.loom.toml
  items: greeting
  contribute: [context]
- model: some/model
'''
    backend = InMemoryBackend(['hi Ada'])
    p = program_from(pdl, backend, ctx={'name': 'Ada'})

    p.execute()

    assert backend.calls[0]['tokens'] == b('<|user|>Hello Ada<|end|><|assistant|>')
    assert p.evaluation_environment['_'] == [
        {'role': 'user', 'content': 'Hello Ada'},
        {'role': 'assistant', 'content': 'hi Ada'},
    ]
