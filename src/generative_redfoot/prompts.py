'''Centralized WordLoom prompt loading.

Loom files named by bare filename live in ``prompts/`` under the project root; file-inclusion is always enabled,
with ``base_dir`` anchored at the project root (bare names) or at the loom's own directory (explicit paths).
'''

import logging
import os
from functools import lru_cache
from pathlib import Path

import wordloom

PROMPTS_DIR_ENV = 'GENERATIVE_REDFOOT_PROMPTS_DIR'
ROOT_ENV = 'GENERATIVE_REDFOOT_ROOT'
DEFAULT_DIRNAME = 'prompts'
MAX_SIGNATURE_FILES = 500  # cap the cache-key walk: a huge directory must not stat every file on every load

logger = logging.getLogger(__name__)
_cap_warned = False  # the cap warning is per process, not per request


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
def _cached_load(path_str: str, signature: tuple, base_dir_str: str) -> dict:  # signature only keys the cache
    return wordloom.load(Path(path_str), lang='en', features={'file-inclusion'}, base_dir=Path(base_dir_str))


def _skip_directory(name: str) -> bool:
    '''Directories never worth walking: hidden ones and the usual dependency / bytecode trees'''
    return name.startswith('.') or name in {'__pycache__', 'node_modules'}


def _signature(abs_path: Path) -> tuple:
    '''
    Cache key part: mtimes of the loom file and of files beside or below it, since `file:` / `glob:` / `dir:`
    inclusions usually live there and must invalidate the parsed result when edited. Hidden directories,
    `__pycache__` and `node_modules` are never entered. The loom's own mtime always comes first, so edits to the
    loom itself are always seen; after MAX_SIGNATURE_FILES, edits to included files may need `clear_cache()`.
    (Inclusions elsewhere under the base directory are not tracked at all.)
    '''
    stamps = [(abs_path.name, abs_path.stat().st_mtime_ns)]  # raises FileNotFoundError if the loom is missing
    root = abs_path.parent
    global _cap_warned
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not _skip_directory(name))  # prune before descending
        for filename in sorted(filenames):
            if filename.startswith('.'):  # keep the old behaviour of ignoring hidden files too
                continue
            if len(stamps) > MAX_SIGNATURE_FILES:
                if not _cap_warned:
                    logger.warning(
                        f'More than {MAX_SIGNATURE_FILES} files beside {abs_path}; edits to included files may need '
                        'prompts.clear_cache()'
                    )
                    _cap_warned = True
                return tuple(stamps)
            entry = Path(dirpath, filename)
            try:
                stamps.append((entry.relative_to(root).as_posix(), entry.stat().st_mtime_ns))
            except FileNotFoundError:
                continue  # the file vanished between listing and stat
    return tuple(stamps)


def load_file(loom_path, *, base_dir=None) -> dict:
    abs_path, default_base = _resolve(loom_path)
    bdir = Path(base_dir).resolve() if base_dir else default_base
    return _cached_load(str(abs_path), _signature(abs_path), str(bdir))


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
