import pytest
import yaml

from generative_redfoot import prompts
from generative_redfoot.extensions.loom import WorldLoomRead
from generative_redfoot.object_pdl_model import ParseDispatcher, PDLProgram, PDLText

LOOM = '''lang = "en"

[greeting]
_ = "Hello {name}"
_m = ["name"]

[plain]
_ = "Literal {braces} stay"

[prefixed]
_ = "Instructions. {document_text}"
_m = ["document_text"]

[with_file]
_ = "Notes: {notes}"
notes = "file:prompts/notes.md"
'''


@pytest.fixture
def program_for(tmp_path, monkeypatch):
    monkeypatch.setenv(prompts.ROOT_ENV, str(tmp_path))
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'p.loom.toml').write_text(LOOM)
    (tmp_path / 'prompts' / 'notes.md').write_text('NOTE TEXT')

    def make(items, context=None):
        pdl = f'''
description: t
text:
- read_from_wordloom: p.loom.toml
  items: "{items}"
  contribute: [context]
'''
        dispatcher = ParseDispatcher()
        dispatcher.DISPATCH_RESOLUTION_ORDER = [WorldLoomRead, PDLText]
        return PDLProgram(yaml.safe_load(pdl), dispatcher=dispatcher, initial_context=context or {})

    return make


def test_markers_come_from_context(program_for):
    p = program_for('greeting', {'name': 'Ada'})
    p.execute()
    assert p.evaluation_environment['_'] == [{'role': 'user', 'content': 'Hello Ada'}]


def test_no_markers_leaves_braces_alone(program_for):
    p = program_for('plain')
    p.execute()
    assert p.evaluation_environment['_'][0]['content'] == 'Literal {braces} stay'


def test_missing_context_variable_is_named(program_for):
    p = program_for('greeting')
    with pytest.raises(KeyError, match='name'):
        p.execute()


def test_file_binding_is_rendered(program_for):
    p = program_for('with_file')
    p.execute()
    assert p.evaluation_environment['_'][0]['content'] == 'Notes: NOTE TEXT'


def test_multiple_items_joined_by_newline(program_for):
    p = program_for('plain with_file')
    p.execute()
    assert p.evaluation_environment['_'][0]['content'] == 'Literal {braces} stay\nNotes: NOTE TEXT'


def test_prefix_marker_passthrough(program_for):
    # The CLI maps a prefix marker to itself so it survives rendering and the prompt can be cut there
    p = program_for('prefixed', {'document_text': '{document_text}'})
    p.execute()
    assert p.evaluation_environment['_'][0]['content'] == 'Instructions. {document_text}'
