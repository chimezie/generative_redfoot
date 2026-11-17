'''Tests for role inheritance and the single rule that adds contributions to the message list.'''

import yaml

from generative_redfoot.cli import make_dispatcher, make_program
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend


def b(text):
    # The fake tokenizer is byte-level, so expected token ids are just the UTF-8 bytes
    return list(text.encode('utf-8'))


def program_from(pdl, backend, ctx=None):
    return make_program(
        yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(), {}, ctx if ctx is not None else {},
    )


ACCEPTANCE = '''
description: t
text:
- role: system
  text:
  - "Be brief. "
  - text: "Be kind."
- "Question one"
- model: some/model
- "Question two"
- model: some/model
'''


def test_acceptance_example_roles_and_messages():
    backend = InMemoryBackend(['A1', 'A2'])
    program = program_from(ACCEPTANCE, backend)

    program.execute()

    assert program.evaluation_environment['_'] == [
        {'role': 'system', 'content': 'Be brief. Be kind.'},
        {'role': 'user', 'content': 'Question one'},
        {'role': 'assistant', 'content': 'A1'},
        {'role': 'user', 'content': 'Question two'},
        {'role': 'assistant', 'content': 'A2'},
    ]
    assert backend.calls[0]['tokens'] == b(
        '<|system|>Be brief. Be kind.<|end|><|user|>Question one<|end|><|assistant|>'
    )
    assert backend.calls[1]['tokens'] == b(
        '<|system|>Be brief. Be kind.<|end|><|user|>Question one<|end|>'
        '<|assistant|>A1<|end|><|user|>Question two<|end|><|assistant|>'
    )


def test_nested_text_inherits_the_container_role():
    pdl = '''
description: t
text:
- role: system
  text:
  - text: A
  - text: B
'''
    program = program_from(pdl, InMemoryBackend())

    program.execute()

    assert program.evaluation_environment['_'] == [{'role': 'system', 'content': 'AB'}]


def test_explicit_role_beats_inheritance():
    pdl = '''
description: t
text:
- role: system
  text:
  - text: A
    role: user
'''
    program = program_from(pdl, InMemoryBackend())

    program.execute()

    assert program.evaluation_environment['_'] == [{'role': 'user', 'content': 'A'}]


def test_model_inside_system_text_stays_assistant():
    pdl = '''
description: t
text:
- role: system
  text:
  - text: Be brief.
    contribute: [context]
  - model: some/model
'''
    backend = InMemoryBackend(['A'])
    program = program_from(pdl, backend)

    program.execute()

    assert program.evaluation_environment['_'] == [
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'assistant', 'content': 'A'},
    ]


def test_model_input_starts_from_user():
    pdl = '''
description: t
text:
- role: system
  text:
  - model: some/model
    input:
      text: Hi
      contribute: [context]
'''
    backend = InMemoryBackend(['A'])
    program = program_from(pdl, backend)

    program.execute()

    assert program.evaluation_environment['_'] == [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'A'},
    ]
    assert backend.calls[0]['tokens'] == b('<|user|>Hi<|end|><|assistant|>')


def test_repeat_body_inherits_the_repeat_role():
    pdl = '''
description: t
text:
- role: system
  text:
  - repeat:
      text: x
    maxIterations: 2
'''
    program = program_from(pdl, InMemoryBackend())

    program.execute()

    assert program.evaluation_environment['_'] == [{'role': 'system', 'content': 'xx'}]


def test_same_role_contributions_merge_different_roles_do_not():
    pdl = '''
description: t
text:
- role: user
  text: a
  contribute: [context]
- role: user
  text: b
  contribute: [context]
- role: assistant
  text: c
  contribute: [context]
- role: user
  text: d
  contribute: [context]
'''
    program = program_from(pdl, InMemoryBackend())

    program.execute()

    assert program.evaluation_environment['_'] == [
        {'role': 'user', 'content': 'ab'},
        {'role': 'assistant', 'content': 'c'},
        {'role': 'user', 'content': 'd'},
    ]


def test_string_model_input_merges_with_a_previous_user_message():
    pdl = '''
description: t
text:
- text: Hello
  contribute: [context]
- model: some/model
  input: " there"
'''
    backend = InMemoryBackend(['A'])
    program = program_from(pdl, backend)

    program.execute()

    assert program.evaluation_environment['_'] == [
        {'role': 'user', 'content': 'Hello there'},
        {'role': 'assistant', 'content': 'A'},
    ]
    assert backend.calls[0]['tokens'] == b('<|user|>Hello there<|end|><|assistant|>')


def test_list_contents_are_never_concatenated():
    pdl = '''
description: t
text:
- text: a
  contribute: [context]
'''
    program = program_from(pdl, InMemoryBackend())

    program.execute()
    program.evaluation_environment['_'][0]['content'] = ['a']  # e.g. an APSModel proposition list
    program.text[0].execute(program.evaluation_environment)

    assert program.evaluation_environment['_'][-1] == {'role': 'user', 'content': 'a'}
