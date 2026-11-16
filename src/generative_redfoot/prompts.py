'''Centralized WordLoom prompt loading.

Loom files named by bare filename live in ``prompts/`` under the project root; file-inclusion is always enabled,
with ``base_dir`` anchored at the project root (bare names) or at the loom's own directory (explicit paths).
'''

import os
from functools import lru_cache
from pathlib import Path

import wordloom

PROMPTS_DIR_ENV = 'GENERATIVE_REDFOOT_PROMPTS_DIR'
ROOT_ENV = 'GENERATIVE_REDFOOT_ROOT'
DEFAULT_DIRNAME = 'prompts'


def project_root() -> Path:
    env = os.environ.get(ROOT_ENV)
    return Path(env).resolve() if env else Path.cwd().resolve()


def prompts_dir() -> Path:
    env = os.environ.get(PROMPTS_DIR_ENV)
    return Path(env).resolve() if env else project_root() / DEFAULT_DIRNAME


def _resolve(loom_path) -> tuple[Path, Path]:
    '''Lexical resolution, no filesystem checks. Returns (abs_path, base_dir).'''
    s = os.fspath(loom_path)
    p = Path(s)
    if p.is_absolute():
        abs_path = p.resolve()
        return abs_path, abs_path.parent
    # Bareness is decided from the original string: Path('./x.toml').parts == ('x.toml',) would look bare
    if '/' not in s and os.sep not in s:  # bare filename -> prompts dir, inclusions anchored at project root
        return (prompts_dir() / p).resolve(), project_root()
    abs_path = p.resolve()  # relative with a directory component (incl. './x') -> cwd
    return abs_path, abs_path.parent


@lru_cache(maxsize=16)
def _cached_load(path_str: str, mtime_ns: int, base_dir_str: str) -> dict:  # mtime_ns only keys the cache
    return wordloom.load(Path(path_str), lang='en', features={'file-inclusion'}, base_dir=Path(base_dir_str))


def load_file(loom_path, *, base_dir=None) -> dict:
    abs_path, default_base = _resolve(loom_path)
    bdir = Path(base_dir).resolve() if base_dir else default_base
    return _cached_load(str(abs_path), abs_path.stat().st_mtime_ns, str(bdir))


def _section_keys(items: dict) -> list[str]:
    '''WordLoom indexes each item by section name AND body text; surface only section names.'''
    return sorted(k for k, v in items.items() if k != str(v))


def load(loom_path, key: str, *, base_dir=None):
    items = load_file(loom_path, base_dir=base_dir)
    if key not in items:
        raise KeyError(f'WordLoom key {key!r} not found in {loom_path!s}. Available: {_section_keys(items)}')
    return items[key]


def try_load(loom_path, key: str, *, base_dir=None):
    try:
        return load(loom_path, key, base_dir=base_dir)
    except KeyError:
        return None


def clear_cache() -> None:
    _cached_load.cache_clear()
