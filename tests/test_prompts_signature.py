'''`prompts._signature` walking: pruning, the file cap, one-time warning and vanished files (Stage 4).'''

import logging

from generative_redfoot import prompts


def make_loom(tmp_path):
    loom = tmp_path / 'a.loom.toml'
    loom.write_text('lang = "en"\n')
    return loom


def names_in(signature):
    return {name for name, _ in signature}


def test_hidden_directories_are_never_entered(tmp_path):
    loom = make_loom(tmp_path)
    venv = tmp_path / '.venv' / 'lib' / 'python3'
    venv.mkdir(parents=True)
    (venv / 'site.py').write_text('x')

    assert names_in(prompts._signature(loom)) == {'a.loom.toml'}


def test_dependency_and_bytecode_directories_are_pruned(tmp_path):
    loom = make_loom(tmp_path)
    for dirname in ('__pycache__', 'node_modules'):
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / 'x.txt').write_text('x')

    assert names_in(prompts._signature(loom)) == {'a.loom.toml'}


def test_ordinary_files_beside_the_loom_are_tracked(tmp_path):
    loom = make_loom(tmp_path)
    (tmp_path / 'notes.md').write_text('n')

    assert names_in(prompts._signature(loom)) == {'a.loom.toml', 'notes.md'}


def test_signature_caps_the_walk_and_warns_once(tmp_path, caplog):
    loom = make_loom(tmp_path)
    for index in range(600):
        (tmp_path / f'file-{index:03d}.txt').write_text('x')

    with caplog.at_level(logging.WARNING, logger='generative_redfoot.prompts'):
        signature = prompts._signature(loom)
        prompts._signature(loom)  # the warning is per process, so this second walk must stay quiet

    assert len(signature) <= 501
    assert signature[0][0] == 'a.loom.toml'  # the loom itself is always first, so its edits are always seen
    assert caplog.text.count('clear_cache') == 1


def test_vanished_file_is_skipped(tmp_path):
    loom = make_loom(tmp_path)
    (tmp_path / 'gone.txt').symlink_to(tmp_path / 'missing-target.txt')  # listed, but stat will fail

    assert 'gone.txt' not in names_in(prompts._signature(loom))
