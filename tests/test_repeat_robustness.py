'''`repeat` robustness: whitespace around conditions and bare string/list bodies (Stage 3).'''

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def program_from(pdl, backend=None, ctx=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend or InMemoryBackend(), GenerationDefaults(), {},
        ctx if ctx is not None else {},
    )


# --- 3A: a YAML block scalar's trailing newline must not turn a condition into mixed text -----------------------------

def test_multiline_until_expression_is_accepted():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  until: >
    ${ i == 1 }
  index: i
'''
    program = program_from(pdl)
    assert program.execute() == '01'  # runs with i == 0 and i == 1, then stops
    assert program.evaluation_environment['i'] == 1


def test_multiline_while_expression_is_accepted():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  while: >
    ${ i < 2 }
  maxIterations: 5
  index: i
'''
    assert program_from(pdl).execute() == '01'


def test_multiline_max_iterations_expression_is_accepted():
    pdl = '''
description: t
text:
- repeat:
    text: x
  maxIterations: >
    ${ n }
'''
    assert program_from(pdl, ctx={'n': 2}).execute() == 'xx'


def test_condition_values_that_are_not_strings_are_unchanged():
    pdl = '''
description: t
text:
- repeat:
    text: x
  maxIterations: 3
'''
    assert program_from(pdl).execute() == 'xxx'


# --- 3B: string and list bodies are text blocks; other values fail when built -----------------------------------------

def test_string_body_is_a_text_block():
    pdl = '''
description: t
text:
- repeat: "x"
  maxIterations: 2
'''
    assert program_from(pdl).execute() == 'xx'


def test_list_body_is_a_text_block():
    pdl = '''
description: t
text:
- repeat: ['a', 'b']
  maxIterations: 2
'''
    assert program_from(pdl).execute() == 'abab'


def test_mapping_body_is_built_as_before():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  maxIterations: 2
  index: i
'''
    assert program_from(pdl).execute() == '01'


def test_non_text_body_is_a_value_error_when_built():
    pdl = '''
description: t
text:
- repeat: 5
  maxIterations: 1
'''
    with pytest.raises(ValueError, match='repeat body'):
        program_from(pdl)
