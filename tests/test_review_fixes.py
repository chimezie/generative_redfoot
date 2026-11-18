'''Regression tests for the fixes from the review of the PDL-alignment work (messages, typed text, defs, index).'''

import pytest
import yaml
from click.testing import CliRunner

from generative_redfoot.cli import main, make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def b(text):
    # The fake tokenizer is byte-level, so expected token ids are just the UTF-8 bytes
    return list(text.encode('utf-8'))


def program_from(pdl, backend=None, ctx=None, base_dir=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend or InMemoryBackend(['A']), GenerationDefaults(), {},
        ctx if ctx is not None else {}, base_dir=base_dir,
    )


def test_consecutive_blocks_reach_a_model_as_separate_messages(tmp_path):
    (tmp_path / 'a.txt').write_text('first')
    (tmp_path / 'b.txt').write_text('second')
    pdl = '''
description: t
text:
- read: a.txt
  contribute: [context]
- read: b.txt
  contribute: [context]
- model: some/model
'''
    backend = InMemoryBackend(['A'])
    program = program_from(pdl, backend, base_dir=tmp_path)
    program.execute()
    assert backend.calls[0]['tokens'] == b('<|user|>first<|end|><|user|>second<|end|><|assistant|>')
    # Once the list finishes, neighbouring same-role messages are merged (the shape later blocks see)
    assert program.evaluation_environment['_'][0] == {'role': 'user', 'content': 'firstsecond'}


def test_whole_expression_text_item_is_contributed_as_text():
    pdl = '''
description: t
text:
- text: "${ n }"
  contribute: [context]
- text: "${ items }"
  contribute: [context]
- model: some/model
'''
    backend = InMemoryBackend(['A'])
    program = program_from(pdl, backend, ctx={'n': 7, 'items': {'a': [1, 2]}})
    program.execute()
    contents = [m['content'] for m in program.evaluation_environment['_']]
    assert all(isinstance(c, str) for c in contents)
    assert contents[0] == '7{"a": [1, 2]}'


def test_typed_model_input_is_contributed_as_text():
    pdl = '''
description: t
text:
- model: some/model
  input: "${ n }"
'''
    backend = InMemoryBackend(['A'])
    program_from(pdl, backend, ctx={'n': 7}).execute()
    assert backend.calls[0]['tokens'] == b('<|user|>7<|end|><|assistant|>')


def test_typo_inside_a_defs_block_is_an_error():
    pdl = '''
description: t
defs:
  s:
    text:
    - modle: x
text: "${ s }"
'''
    with pytest.raises(RuntimeError, match='Unknown block type'):
        program_from(pdl).execute()


def test_plain_mapping_in_defs_is_data():
    pdl = '''
description: t
defs:
  person: {name: Ada, age: 36}
text: "${ person.name } is ${ person.age }"
'''
    assert program_from(pdl).execute() == 'Ada is 36'


def test_legacy_reference_in_defs_binds_the_variable():
    pdl = '''
description: t
defs:
  x: { $y }
text: "${ x }"
'''
    assert program_from(pdl, ctx={'y': 'Y'}).execute() == 'Y'


def test_defs_overrides_an_initial_variable():
    pdl = '''
description: t
defs:
  name: from defs
text: "${ name }"
'''
    assert program_from(pdl, ctx={'name': 'from -v'}).execute() == 'from defs'


def test_while_stop_leaves_the_index_of_the_iteration_that_did_not_run():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  index: i
  while: "${ i < 2 }"
'''
    program = program_from(pdl)
    assert program.execute() == '01'
    assert program.evaluation_environment['i'] == 2  # bound before `while` is checked, as in PDL


def cache_program(extra):
    return f'''
cache:
- text:
  - role: user
    text: "Instructions. {{document_text}}"
    contribute: [context]
  - content_model: some/model
    name: pc
    file: pc.json
    prefix_marker: document_text{extra}
description: t
text:
- model: pc
'''


def test_content_model_rejects_kv_bits_with_max_kv_size(tmp_path):
    ctx = {}
    program = program_from(cache_program('\n    kv_bits: 4\n    max_kv_size: 1000'), ctx=ctx, base_dir=tmp_path)
    with pytest.raises(ValueError, match='kv_bits and max_kv_size'):
        prepare_caches(program, ctx, verbose=False)


def test_cli_reports_program_mistakes_without_a_traceback(tmp_path, monkeypatch):
    pdl_file = tmp_path / 'typo.pdl'
    pdl_file.write_text('description: t\ntext:\n- model: some/model\n  paramters: {max_tokens: 5}\n')
    monkeypatch.setattr('generative_redfoot.cli.MLXBackend', lambda **kwargs: InMemoryBackend())

    result = CliRunner().invoke(main, [str(pdl_file)])

    assert result.exit_code == 1
    assert "did you mean 'parameters'?" in result.output
    assert 'Traceback' not in result.output
