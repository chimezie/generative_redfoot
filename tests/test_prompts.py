import os

import pytest

from generative_redfoot import prompts

LOOM = '''lang = "en"

[greeting]
_ = "Hello {name}"
_m = ["name"]

[plain]
_ = "Literal {braces} stay"

[with_file]
_ = "Notes: {notes}"
notes = "file:prompts/notes.md"
'''


@pytest.fixture
def root(tmp_path, monkeypatch):
    # A project root with prompts/ holding one loom and the asset it includes
    monkeypatch.setenv(prompts.ROOT_ENV, str(tmp_path))
    monkeypatch.delenv(prompts.PROMPTS_DIR_ENV, raising=False)
    (tmp_path / 'prompts').mkdir()
    (tmp_path / 'prompts' / 'a.loom.toml').write_text(LOOM)
    (tmp_path / 'prompts' / 'notes.md').write_text('NOTE TEXT')
    return tmp_path


def test_resolve_bare_filename_uses_prompts_dir(root):
    abs_path, base = prompts._resolve('a.loom.toml')
    assert abs_path == root / 'prompts' / 'a.loom.toml'
    assert base == root


def test_resolve_absolute_path(root, tmp_path):
    abs_path, base = prompts._resolve(str(root / 'prompts' / 'a.loom.toml'))
    assert abs_path == root / 'prompts' / 'a.loom.toml'
    assert base == root / 'prompts'


def test_resolve_dot_slash_is_cwd_relative_not_bare(root, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    abs_path, base = prompts._resolve('./x.toml')
    assert abs_path == tmp_path / 'x.toml'
    assert base == tmp_path


def test_resolve_multi_segment_is_cwd_relative(root, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    abs_path, _ = prompts._resolve(os.path.join('prompts', 'a.loom.toml'))
    assert abs_path == tmp_path / 'prompts' / 'a.loom.toml'


def test_prompts_dir_env_override(root, monkeypatch, tmp_path):
    other = tmp_path / 'elsewhere'
    other.mkdir()
    monkeypatch.setenv(prompts.PROMPTS_DIR_ENV, str(other))
    abs_path, _ = prompts._resolve('a.loom.toml')
    assert abs_path == other / 'a.loom.toml'


def test_load_renders_markers(root):
    assert prompts.load('a.loom.toml', 'greeting').render(name='World') == 'Hello World'


def test_missing_key_lists_only_section_names(root):
    with pytest.raises(KeyError) as exc:
        prompts.load('a.loom.toml', 'nope')
    message = str(exc.value)
    assert 'greeting' in message and 'plain' in message
    assert 'Hello {name}' not in message  # item bodies are indexed too, but must not be listed


def test_try_load(root):
    assert prompts.try_load('a.loom.toml', 'nope') is None
    assert prompts.try_load('a.loom.toml', 'plain') is not None


def test_file_inclusion_binding(root):
    assert prompts.load('a.loom.toml', 'with_file').render() == 'Notes: NOTE TEXT'


def test_cache_picks_up_edits(root):
    path = root / 'prompts' / 'a.loom.toml'
    assert prompts.load('a.loom.toml', 'plain')
    path.write_text(LOOM.replace('Literal', 'Changed'))
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10_000_000))  # make the mtime change unambiguous
    assert str(prompts.load('a.loom.toml', 'plain')).startswith('Changed')


def test_edit_to_included_file_invalidates_cache(root):
    assert prompts.load('a.loom.toml', 'with_file').render() == 'Notes: NOTE TEXT'
    note = root / 'prompts' / 'notes.md'
    note.write_text('CHANGED')
    os.utime(note, ns=(1, note.stat().st_mtime_ns + 10_000_000))
    assert prompts.load('a.loom.toml', 'with_file').render() == 'Notes: CHANGED'
