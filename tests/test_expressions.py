'''`${ ... }` expression evaluation and the legacy `{ $name }` reference (Stage 1).'''

from pathlib import Path

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program
from generative_redfoot.expressions import evaluate, resolve_variable_reference
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend
from generative_redfoot.pdf_text import extract_pdf_text

TEST_PDF = Path(__file__).parent / 'Test.pdf'


def program_from(pdl, backend, cache_lookup=None, ctx=None, **defaults):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(**defaults), cache_lookup or {},
        ctx if ctx is not None else {},
    )


# --- evaluate() acceptance table -------------------------------------------------------------------------------------

def test_text_with_expression_renders_and_keeps_trailing_newline():
    assert evaluate('Hello ${ name }!\n', {'name': 'Ada'}) == 'Hello Ada!\n'


def test_whole_string_expression_returns_typed_value():
    result = evaluate('${ n + 1 }', {'n': 2})
    assert result == 3 and isinstance(result, int)


def test_bytes_variable_round_trips():
    document = b'%PDF-1.4 bytes'
    assert evaluate('${ doc }', {'doc': document}) is document


def test_plain_braces_are_never_jinja():
    assert evaluate('plain {marker} {# text', {}) == 'plain {marker} {# text'


def test_dicts_and_lists_are_evaluated_recursively():
    assert evaluate({'max_tokens': '${ budget }', 'stop': ['${ s }']}, {'budget': 9, 's': 'x'}) == {
        'max_tokens': 9,
        'stop': ['x'],
    }


def test_undefined_variable_raises_value_error_naming_it():
    with pytest.raises(ValueError, match='nme'):
        evaluate('${ nme }', {'name': 'Ada'})


def test_syntax_error_raises_value_error_naming_expression():
    with pytest.raises(ValueError, match='boom'):
        evaluate('${ boom', {})


def test_message_list_is_exposed_as_pdl_context():
    scope = {'_': [{'role': 'user', 'content': 'hi'}]}
    assert evaluate('${ pdl_context | length }', scope) == 1


def test_legacy_variable_reference_resolves_only_the_dollar_form():
    document = b'%PDF-1.4 bytes'
    assert resolve_variable_reference({'$document_text': None}, {'document_text': document}) is document
    assert resolve_variable_reference({'name': 'Ada'}, {}) == {'name': 'Ada'}


# --- program-level use ----------------------------------------------------------------------------------------------

def test_model_parameters_are_evaluated():
    pdl = '''
description: t
text:
- role: user
  text: Hello
  contribute: [context]
- model: some/model
  parameters:
    max_tokens: "${ budget }"
'''
    backend = InMemoryBackend()
    program_from(pdl, backend, ctx={'budget': 9}).execute()
    assert backend.calls[0]['sampling']['max_tokens'] == 9


def test_pdf_raw_expression_matches_legacy_reference():
    pytest.importorskip('pymupdf')
    pdf_bytes = TEST_PDF.read_bytes()
    expected = extract_pdf_text(pdf_bytes, ocr=False)
    expression_pdl = '''
description: t
text:
- PDF_raw_read_txt: "${ document_text }"
'''
    legacy_pdl = '''
description: t
text:
- PDF_raw_read_txt: { $document_text }
'''
    contents = []
    for pdl in (expression_pdl, legacy_pdl):
        p = program_from(pdl, InMemoryBackend(), ctx={'document_text': pdf_bytes})
        p.execute()
        contents.append(p.evaluation_environment['_'][-1]['content'])
    assert contents == [expected, expected]


def test_variable_value_containing_expression_is_contributed_literally():
    pdl = '''
description: t
text:
- "${ doc }"
'''
    p = program_from(pdl, InMemoryBackend(), ctx={'doc': '${ boom }'})
    p.execute()
    assert p.evaluation_environment['_'] == [{'role': 'user', 'content': '${ boom }'}]
