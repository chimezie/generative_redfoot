'''Keep the programs in examples/ loading and preparing their caches (models are faked; no mlx or OCR needed).'''

from pathlib import Path

import yaml

from generative_redfoot import prompts
from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend

EXAMPLES = Path(__file__).parent.parent / 'examples'


def test_pdf_service_example_builds_its_prefix_cache(tmp_path, monkeypatch):
    example = EXAMPLES / 'pdf_service'
    monkeypatch.chdir(example)  # the example is run from its own directory, so ./prompts/ holds its loom file
    monkeypatch.delenv(prompts.ROOT_ENV, raising=False)
    program_yaml = yaml.safe_load((example / 'service.pdl').read_text())
    program_yaml['cache'][0]['text'][-1]['file'] = str(tmp_path / 'cache.json')  # keep the example directory clean
    backend, ctx = InMemoryBackend(), {}

    program = make_program(program_yaml, make_dispatcher(), backend, GenerationDefaults(), {}, ctx, base_dir=example)
    lookup = prepare_caches(program, ctx, verbose=False)

    prefix = bytes(backend.prefills[0]['tokens']).decode()
    assert lookup == {'briefing_cache': str(tmp_path / 'cache.json')}
    assert prefix.endswith('Document:\n')  # cut at the {document_text} marker
    assert 'Style guidelines' in prefix  # the WordLoom file inclusion was rendered
    assert backend.prefills[0]['kv_bits'] == 4
