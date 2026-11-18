'''Block results, `def` and `defs` (Stage 2).'''

from pathlib import Path

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend
from generative_redfoot.pdf_text import extract_pdf_text

TEST_PDF = Path(__file__).parent / 'Test.pdf'


def program_from(pdl, backend, ctx=None, **defaults):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(**defaults), {},
        ctx if ctx is not None else {},
    )


def test_acceptance_defs_def_and_results(capsys):
    pdl = '''
description: t
defs:
  greeting: Hello
  shout:
    text: "${ greeting }!"
text:
- text: ["${ shout } ", "Ada"]
  def: line
- model: some/model
'''
    backend = InMemoryBackend(['R'])
    p = program_from(pdl, backend)

    result = p.execute()

    context = p.evaluation_environment
    assert context['greeting'] == 'Hello'
    assert context['shout'] == 'Hello!'
    assert context['line'] == 'Hello! Ada'
    assert backend.calls[0]['tokens'] == list(b'<|user|>Hello! Ada<|end|><|assistant|>')
    assert result == 'Hello! AdaR'
    assert context['_'] == [
        {'role': 'user', 'content': 'Hello! Ada'},
        {'role': 'assistant', 'content': 'R'},
    ]
    # The `shout` defs block ran with contribute: []: it printed nothing (the line above prints "Hello! " once)
    assert capsys.readouterr().out.count('Hello!') == 1


def test_list_form_text_prints_plain_text(capsys):
    pdl = '''
description: t
text: ["one", "two"]
'''
    p = program_from(pdl, InMemoryBackend())

    result = p.execute()

    assert result == 'onetwo'
    assert capsys.readouterr().out == 'one\ntwo\n'


def test_string_items_honor_contribute(capsys):
    pdl = '''
description: t
text:
- text: silent
  contribute: [result]
- text: quiet
  contribute: [context]
'''
    p = program_from(pdl, InMemoryBackend())

    result = p.execute()

    out = capsys.readouterr().out
    assert 'silent' in out and 'quiet' not in out
    assert p.evaluation_environment['_'] == [{'role': 'user', 'content': 'quiet'}]
    assert result == 'silent'  # only the child that contributes to `result` joins the parent's result


def test_child_without_result_contribution_is_left_out_of_the_join():
    pdl = '''
description: t
text:
- text: child
  contribute: [context]
- "-end"
'''
    p = program_from(pdl, InMemoryBackend())
    assert p.execute() == '-end'


def test_read_returns_its_result_and_def_binds(monkeypatch):
    pdl = '''
description: t
text:
- read:
  message: "Q: "
  def: answer
'''
    monkeypatch.setattr('builtins.input', lambda prompt: 'answer text')
    p = program_from(pdl, InMemoryBackend())
    assert p.execute() == 'answer text'
    assert p.evaluation_environment['answer'] == 'answer text'


def test_repeat_returns_joined_iterations(monkeypatch):
    pdl = '''
description: t
text:
- repeat:
    text:
    - "iter"
    - read:
      def: done
      message: "Done? "
      contribute: [context]
  until: "${ done == 'yes' }"
'''
    monkeypatch.setattr('builtins.input', lambda prompt: 'yes')
    p = program_from(pdl, InMemoryBackend())
    assert p.execute() == 'iter'


def test_wordloom_returns_rendered_content(tmp_path, monkeypatch):
    monkeypatch.setenv('GENERATIVE_REDFOOT_ROOT', str(tmp_path))
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'p.loom.toml').write_text(
        'lang = "en"\n\n[greeting]\n_ = "Hello {name}"\n_m = ["name"]\n'
    )
    pdl = '''
description: t
text:
- read_from_wordloom: p.loom.toml
  items: greeting
'''
    p = program_from(pdl, InMemoryBackend(), ctx={'name': 'Ada'})
    assert p.execute() == 'Hello Ada'


def test_pdf_block_returns_extracted_text():
    pytest.importorskip('pymupdf')
    pdf_bytes = TEST_PDF.read_bytes()
    expected = extract_pdf_text(pdf_bytes, ocr=False)
    pdl = '''
description: t
text:
- PDF_raw_read_txt: { $document_text }
'''
    p = program_from(pdl, InMemoryBackend(), ctx={'document_text': pdf_bytes})
    assert p.execute() == expected


def test_defs_block_is_isolated_from_the_real_context():
    pdl = '''
description: t
defs:
  summary:
    text:
    - "inside"
    - text: private
      contribute: [context]
text: done
'''
    p = program_from(pdl, InMemoryBackend())

    p.execute()

    assert p.evaluation_environment['summary'] == 'inside'
    # `private` was added only to the defs sandbox, never to the real message list
    assert p.evaluation_environment['_'] == [{'role': 'user', 'content': 'done'}]


def test_defs_values_are_expressions_and_evaluated_in_order():
    pdl = '''
description: t
defs:
  base: 2
  total: "${ base + 1 }"
text: "${ total }"
'''
    p = program_from(pdl, InMemoryBackend())
    p.execute()
    assert p.evaluation_environment['total'] == 3


def test_top_level_defs_and_def_take_effect_once():
    pdl = '''
description: t
defs:
  answer:
    model: some/model
text: done
contribute: [result]
'''
    backend = InMemoryBackend(['R'])
    ctx = {'_': [{'role': 'user', 'content': 'hi'}]}
    p = program_from(pdl, backend, ctx=ctx)

    p.execute()

    assert len(backend.calls) == 1  # not once by PDLProgram and again by its inner PDLText
    assert ctx['answer'] == 'R'
    assert ctx['_'] == [{'role': 'user', 'content': 'hi'}]
