'''
Integration tests: the real MLXBackend against a small model. Apple Silicon only; downloads the model on first run.
Run with `pytest -m integration`. Override the model with GENERATIVE_REDFOOT_TEST_MODEL.
'''
import os

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, MLXBackend

pytest.importorskip('mlx_lm')
pytestmark = pytest.mark.integration

MODEL = os.environ.get('GENERATIVE_REDFOOT_TEST_MODEL', 'mlx-community/Qwen3-0.6B-4bit')
DEFAULTS = GenerationDefaults(temperature=0.0, max_tokens=24)


@pytest.fixture(scope='module')
def backend():
    return MLXBackend()


def run(pdl, backend, ctx=None, cache_lookup=None):
    program = make_program(yaml.safe_load(pdl), make_dispatcher(), backend, DEFAULTS, cache_lookup or {}, ctx or {})
    program.execute()
    return program


def test_load_and_generate(backend):
    model, tokenizer = backend.load(MODEL)
    tokens = tokenizer.encode('The capital of France is')
    text, ids = backend.generate(model, tokenizer, tokens, {'temperature': 0.0, 'top_p': 0.0, 'min_p': 0.0,
                                                            'top_k': 0, 'repetition_penalty': 0.0, 'max_tokens': 8})
    assert text and ids


def test_plain_program(backend):
    p = run(f'description: t\ntext:\n- role: user\n  text: Say hello.\n  contribute: [context]\n- model: {MODEL}\n',
            backend)
    assert p.evaluation_environment['_'][-1]['role'] == 'assistant'


def test_internal_cache_over_two_turns(backend):
    pdl = f'cache: "*"\ndescription: t\ntext:\n- role: user\n  text: Say hello.\n  contribute: [context]\n- model: {MODEL}\n'
    p = run(pdl, backend)
    ctx = p.evaluation_environment
    ctx['_'].append({'role': 'user', 'content': 'Now say goodbye.'})
    p.text[1].execute(ctx)
    assert ctx['_'][-1]['role'] == 'assistant'
    assert p.kv_state['tokens']  # the ledger tracks what the KV cache holds


def test_prefix_cache_round_trip(backend, tmp_path):
    file = str(tmp_path / 'pc.safetensors')
    pdl = f'''
cache:
- text:
  - role: user
    text: "Answer in one word. {{document_text}}"
    contribute: [context]
  - content_model: {MODEL}
    name: pc
    file: {file}
    prefix_marker: document_text
description: t
text:
- role: user
  text: What colour is the sky?
  contribute: [context]
- model: pc
'''
    ctx = {}
    p = make_program(yaml.safe_load(pdl), make_dispatcher(), backend, DEFAULTS, {}, ctx)
    prepare_caches(p, ctx, verbose=True)
    p.evaluation_environment = ctx
    p.execute()
    assert ctx['_'][-1]['role'] == 'assistant'
    assert os.path.exists(file)
