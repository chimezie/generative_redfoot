'''Author-facing errors: expression failures, unknown blocks, cache shape, and CLI reporting (Stage 2).'''

import pytest
import yaml
from click.testing import CliRunner

from generative_redfoot.cli import main, make_dispatcher, make_program, prepare_caches
from generative_redfoot.expressions import evaluate
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def program_from(pdl, backend=None, ctx=None, base_dir=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend or InMemoryBackend(['A']), GenerationDefaults(), {},
        ctx if ctx is not None else {}, base_dir=base_dir,
    )


def run_cli(tmp_path, monkeypatch, pdl, *options, name='prog.pdl'):
    pdl_file = tmp_path / name
    pdl_file.write_text(pdl)
    monkeypatch.setattr('generative_redfoot.cli.MLXBackend', lambda **kwargs: InMemoryBackend())
    return CliRunner().invoke(main, [*options, str(pdl_file)])


def cache_pdl(block):
    return f'''cache:
- text:
  - role: user
    text: "Instructions."
    contribute: [context]
  - {block}
description: t
text:
- model: pc
'''


# --- 2A: any expression failure is a ValueError naming the expression and the original exception ----------------------

def test_type_error_names_the_expression_and_the_original_exception():
    with pytest.raises(ValueError) as excinfo:
        evaluate('${ x + 1 }', {'x': 'a'})

    message = str(excinfo.value)
    assert 'x + 1' in message and 'TypeError' in message
    assert isinstance(excinfo.value.__cause__, TypeError)


def test_zero_division_names_the_expression_and_the_original_exception():
    with pytest.raises(ValueError, match='ZeroDivisionError'):
        evaluate('${ 1 / 0 }', {})


def test_sandbox_violation_is_a_value_error():
    with pytest.raises(ValueError) as excinfo:
        evaluate('${ x.__class__ }', {'x': 'a'})

    assert '__class__' in str(excinfo.value) and 'SecurityError' in str(excinfo.value)


# --- 2B: an unknown block is a ValueError ----------------------------------------------------------------------------

def test_unknown_block_is_a_value_error():
    pdl = '''
description: t
text:
- modle: x
'''
    with pytest.raises(ValueError, match='Unknown block type'):
        program_from(pdl)


# --- 2C: cache section mistakes -------------------------------------------------------------------------------------

def test_cache_content_model_requires_a_name(tmp_path):
    program = program_from(cache_pdl('content_model: some/model\n    file: pc.json'), base_dir=tmp_path)
    with pytest.raises(ValueError, match="'name'"):
        prepare_caches(program, {}, verbose=False)


def test_cache_content_model_requires_a_file(tmp_path):
    program = program_from(cache_pdl('content_model: some/model\n    name: pc'), base_dir=tmp_path)
    with pytest.raises(ValueError, match="'file'"):
        prepare_caches(program, {}, verbose=False)


def test_cache_key_typo_gets_the_unknown_key_message(tmp_path):
    block = 'content_model: some/model\n    nmae: pc\n    file: pc.json'
    program = program_from(cache_pdl(block), base_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown key 'nmae'"):
        prepare_caches(program, {}, verbose=False)


@pytest.mark.parametrize(
    'cache_yaml',
    [
        'cache: 5',
        'cache:\n- text: not-a-list',
        'cache:\n- text:\n  - text: no content model here',
    ],
)
def test_malformed_cache_section_is_a_value_error(tmp_path, cache_yaml):
    pdl = f'{cache_yaml}\ndescription: t\ntext:\n- model: some/model\n'
    program = program_from(pdl, base_dir=tmp_path)
    with pytest.raises(ValueError, match='content_model'):
        prepare_caches(program, {}, verbose=False)


# --- 2D: the CLI reports author mistakes without a traceback ---------------------------------------------------------

def test_cli_reports_a_cache_key_typo_without_a_traceback(tmp_path, monkeypatch):
    pdl = cache_pdl('content_model: some/model\n    nmae: pc\n    file: pc.json\n    prefix_marker: document_text')
    result = run_cli(tmp_path, monkeypatch, pdl)

    assert result.exit_code == 1
    assert "Unknown key 'nmae'" in result.output
    assert 'Traceback' not in result.output


def test_cli_reports_an_expression_error_without_a_traceback(tmp_path, monkeypatch):
    pdl = 'description: t\ntext:\n- text: "${ 1 / 0 }"\n'
    result = run_cli(tmp_path, monkeypatch, pdl)

    assert result.exit_code == 1
    assert '1 / 0' in result.output and 'ZeroDivisionError' in result.output
    assert 'Traceback' not in result.output


def test_cli_reports_a_missing_read_file_without_a_traceback(tmp_path, monkeypatch):
    pdl = 'description: t\ntext:\n- read: ./missing.txt\n'
    result = run_cli(tmp_path, monkeypatch, pdl)

    assert result.exit_code == 1
    assert 'missing.txt' in result.output
    assert 'Traceback' not in result.output


def test_cli_verbose_keeps_the_original_exception(tmp_path, monkeypatch):
    pdl = 'description: t\ntext:\n- text: "${ 1 / 0 }"\n'
    result = run_cli(tmp_path, monkeypatch, pdl, '--verbose')

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)


def test_cli_no_longer_treats_runtime_errors_as_author_mistakes(tmp_path, monkeypatch):
    def broken_program(*args, **kwargs):
        raise RuntimeError('a real runtime failure')

    monkeypatch.setattr('generative_redfoot.cli.make_program', broken_program)
    (tmp_path / 'prog.pdl').write_text('description: t\ntext: "x"\n')
    result = CliRunner().invoke(main, [str(tmp_path / 'prog.pdl')])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
