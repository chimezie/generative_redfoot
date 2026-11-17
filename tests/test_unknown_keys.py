'''Tests for rejecting keys a block class does not declare.'''

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend
from generative_redfoot.object_pdl_model import PDLObject, PDLStructuredBlock


def program_from(pdl, backend=None, dispatcher=None):
    return make_program(
        yaml.safe_load(pdl), dispatcher or make_dispatcher(), backend or InMemoryBackend(), GenerationDefaults(), {},
    )


def test_content_model_kv_bit_suggests_kv_bits():
    pdl = '''
cache:
- text:
  - content_model: some/model
    name: pc
    file: pc.json
    kv_bit: 4
description: t
text:
- model: pc
'''
    program = program_from(pdl)
    with pytest.raises(ValueError) as excinfo:
        prepare_caches(program, {}, verbose=False)
    message = str(excinfo.value)
    assert 'kv_bit' in message and 'CachePrep' in message and 'kv_bits' in message


def test_read_mesage_suggests_message():
    pdl = '''
description: t
text:
- read: ./note.txt
  mesage: hi
'''
    with pytest.raises(ValueError) as excinfo:
        program_from(pdl)
    message = str(excinfo.value)
    assert 'mesage' in message and 'PDLRead' in message and 'message' in message


def test_text_typo_suggests_role():
    pdl = '''
description: t
text:
- text: hi
  rol: system
'''
    with pytest.raises(ValueError, match="'rol'.*'role'"):
        program_from(pdl)


def test_top_level_typo_suggests_description():
    pdl = '''
descripton: t
text: hi
'''
    with pytest.raises(ValueError, match="'descripton'.*'description'"):
        program_from(pdl)


def test_model_typo_suggests_parameters():
    pdl = '''
description: t
text:
- text: hi
  contribute: [context]
- model: some/model
  paramters:
    max_tokens: 5
'''
    with pytest.raises(ValueError, match="'paramters'.*'parameters'"):
        program_from(pdl)


def test_parameters_contents_are_not_validated():
    pdl = '''
description: t
text:
- text: hi
  contribute: [context]
- model: some/model
  parameters:
    something_unknown: 5
'''
    program_from(pdl)  # builds (and, for unknown sampling keys, the backend just ignores them)


def test_block_class_without_keys_is_not_checked():
    class OutsideBlock(PDLObject, PDLStructuredBlock):
        def __init__(self, item, program):
            self.program = program
            self._get_common_attributes(item)

        def execute(self, context, verbose=False):
            self._handle_execution_contribution('outside', context)
            return 'outside'

        @staticmethod
        def dispatch_check(item, program):
            if 'outside' in item:
                return OutsideBlock(item, program)

    dispatcher = make_dispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [OutsideBlock, *dispatcher.DISPATCH_RESOLUTION_ORDER]
    pdl = '''
description: t
text:
- outside: 1
  anything_goes: true
'''
    program = program_from(pdl, dispatcher=dispatcher)
    assert program.execute() == 'outside'
