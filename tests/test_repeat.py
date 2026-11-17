'''Tests for repeat: the until / while / for / maxIterations stopping rules and the index binding.'''

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def program_from(pdl, backend=None, ctx=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend or InMemoryBackend(), GenerationDefaults(), {},
        ctx if ctx is not None else {},
    )


def test_for_binds_each_name_and_index():
    pdl = '''
description: t
text:
- repeat:
    text: "${ word }-"
  for:
    word: [a, b, c]
  index: i
'''
    program = program_from(pdl)
    assert program.execute() == 'a-b-c-'
    assert program.evaluation_environment['i'] == 2
    assert program.evaluation_environment['word'] == 'c'


def test_for_values_may_be_expressions():
    pdl = '''
description: t
text:
- repeat:
    text: "${ word }-"
  for:
    word: "${ words }"
'''
    program = program_from(pdl, ctx={'words': ['x', 'y']})
    assert program.execute() == 'x-y-'


def test_for_lists_of_different_lengths_raise():
    pdl = '''
description: t
text:
- repeat:
    text: x
  for:
    a: [1, 2]
    b: [3]
'''
    program = program_from(pdl)
    with pytest.raises(ValueError, match='same length'):
        program.execute()


def test_until_is_evaluated_after_each_iteration():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  until: "${ i >= 1 }"
  index: i
'''
    program = program_from(pdl)
    assert program.execute() == '01'  # iterations with i == 0 and i == 1
    assert program.evaluation_environment['i'] == 1


def test_until_runs_the_first_iteration_before_checking():
    pdl = '''
description: t
text:
- repeat:
    text: x
  until: "${ true }"
'''
    assert program_from(pdl).execute() == 'x'


def test_until_must_be_a_boolean():
    pdl = '''
description: t
text:
- repeat:
    text: x
  until: "${ 'yes' }"
'''
    with pytest.raises(ValueError, match='boolean'):
        program_from(pdl).execute()


def test_while_is_checked_before_each_iteration():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  while: "${ i < 2 }"
  maxIterations: 5
  index: i
'''
    program = program_from(pdl)
    assert program.execute() == '01'
    assert program.evaluation_environment['i'] == 2


def test_max_iterations_stops_the_loop():
    pdl = '''
description: t
text:
- repeat:
    text: "${ i }"
  until: "${ false }"
  maxIterations: 3
  index: i
'''
    program = program_from(pdl)
    assert program.execute() == '012'
    assert program.evaluation_environment['i'] == 2


def test_repeat_without_a_stopping_rule_raises_when_built():
    pdl = '''
description: t
text:
- repeat:
    text: x
'''
    with pytest.raises(ValueError, match='maxIterations'):
        program_from(pdl)


def test_empty_for_raises():
    pdl = '''
description: t
text:
- repeat:
    text: x
  for: {}
'''
    with pytest.raises(ValueError, match='at least one list'):
        program_from(pdl).execute()


def test_for_stops_without_other_rules():
    pdl = '''
description: t
text:
- repeat:
    text: "${ n }"
  for:
    n: [1, 2]
'''
    assert program_from(pdl).execute() == '12'
