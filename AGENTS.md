# AGENTS.md

## Project

Generative Redfoot interprets a minimal subset of PDL (Prompt Declaration Language, YAML) to run local LLMs with
mlx-lm, loads prompts from WordLoom TOML files, and can serve a PDL program as a FastAPI service.

## Layout

- `src/generative_redfoot/object_pdl_model.py`: PDL block classes, `ParseDispatcher`, `PDLProgram`
- `src/generative_redfoot/prompts.py`: shared WordLoom loader (the only place loom files are opened)
- `src/generative_redfoot/generation.py`: `MLXBackend` (the only module importing `mlx`/`mlx_lm`), `InMemoryBackend`
  (real fake for tests), `GenerationDefaults`, and pure helpers
- `src/generative_redfoot/pdf_text.py`: PyMuPDF text/OCR extraction (needs system Tesseract for OCR) and its cache
- `src/generative_redfoot/extensions/loom.py`: `read_from_wordloom` block
- `src/generative_redfoot/extensions/mlx.py`: `content_model` (cache prep), `model`, `APSModel` blocks
- `src/generative_redfoot/cli.py`: click CLI, cache bootstrap, FastAPI service (`build_app`)
- `tests/`: unit tests (default); `tests/integration/` needs Apple Silicon

## Dev loop

```
uv pip install '.[dev,pdf]'
ruff check .
pytest                   # unit only (default)
pytest -m integration    # real mlx-lm (Apple Silicon only) and OCR (needs Tesseract; skips without it)
```

The working tree usually lives on a Linux box where mlx cannot run: run lint and unit tests there, and ask the owner
to run `pytest -m integration` and CLI smoke tests on the Apple Studio. The owner makes all git commits and pushes;
suggest commands instead of running them.

## Releases

- The version lives only in `src/generative_redfoot/__init__.py` (`pyproject.toml` reads it via hatch).
- Every user-visible change gets a `CHANGELOG.md` line; breaking changes also go in README "Upgrading from ...".
- Keep README "PDL compatibility" and "Roadmap" in step with what the interpreter actually supports; do not document
  planned features as present.
- A `v*` tag triggers the PyPI publish workflow.

## Platform note

MLX only runs on Apple Silicon. All `mlx` / `mlx_lm` imports are confined to `MLXBackend` and are lazy; unit tests
use `InMemoryBackend` and run anywhere.

## Prompts

Loom files named by bare filename resolve to `<root>/prompts/` (root = `$GENERATIVE_REDFOOT_ROOT` or cwd);
`./name.toml` is cwd-relative; absolute paths are used as given. No inline LLM prompts in Python source.

## Conventions

Use ruff and the conventions described in `pyproject.toml`

Additional context for coding tools & agents

- Python 3.12+ code, unless otherwise specified
- Python code uses single outer quotes, including triple single quotes for e.g. docstrings
- Use a decent amount of comments
  - not *too* many, just enough that anybody familiar with the code can use them as a reference point. Not meant to teach somebody new every intricacy of the code, just help keep the savvy reader oriented.
- if it saves a line, put a comment after a line rather than above it
  - use the standard two spaces before the comment character, eg. `CODE # COMMENT`
- Try to stick to 120 characters per line
  - if one of those comments would break this guideline, just put that comment above the line instead, as is standard convention
- If there is a pyproject.toml in place, use it as a reference for coding convention, builds, installs, etc. The basic packaging and dev preference, including if you have to supply your own pyproject.toml, is as follows:
  - Prefer hatchling build system over setuptools, poetry, etc. Avoid setuptools as much as possible. No setup.py.
  - Reusable Python code modules are developed in the `src` folder, and installed using e.g. `uv pip install .`, which includes proper mapping to Python library package namespace via `tool.hatch.build.sources`.
  - If it's a CLI provided as part of a library, it should still use proper installation via `[project.scripts]` entry points and live in `src/cli/` , and installed via `uv` and exposes a `main()` function, using the `click` module if complex options or command-line arguments are needed .
- **Debugging package issues**: When modules aren't importing correctly after installation, check the virtual environment:
  - That you are in the correct virtualenv (you may have to ask the developer)
  - Package structure in site-packages (e.g., `ls -la /path/to/site-packages/package_name/`)
- Use uv, but pay attention to the above
    - **uv venv gotcha**: `uv run` uses the project's `.venv/` (auto-created), ignoring `$VIRTUAL_ENV`. `uv pip install` does the opposite — it follows `$VIRTUAL_ENV`, NOT the project `.venv/`. So `uv pip install .` run from inside an outer venv will *not* update what `uv run` sees. Either deactivate the outer venv before installing, or let `uv run` manage things.
    - **`uv.lock` is not part of this dev loop**: `uv pip install` ignores `uv.lock` entirely, so nothing above ever consults a lock — a committed `uv.lock` is inert here, not a reproducibility guarantee. Only `uv sync` reads/writes it, and `uv sync` makes the target environment an *exact mirror* of the lock: it prunes anything not in that lockfile. That's actively wrong for a venv shared by more than one project (it strips out the other project's packages) and it tailors the environment to a single lockfile. We deliberately do **not** rely on `uv.lock` for reproducibility — prefer testing the actual built distribution via `uv pip install .`. If a specific repo genuinely needs locked, reproducible installs, adopt `uv sync` wholesale for that repo and give it its own dedicated (non-shared) venv.
- Use async (e.g. asyncio) wherever it makes sense. Avoid multithreading, though multiprocessing is OK. Multiprocess for CPU-bound concurrency, and asyncIO for I/O bound, cooperative etc.
- Be pythonic. Avoid e.g. complex abstract class hierarchies for the sake of them, though classes are also fine in many usage patterns: dictionaries, dynamic dispatch, etc. Don't use Pydantic, but use dataclasses instead.
- Type hints are OK in moderation, but avoid absolutely littering the code with them.
  - No excess imports & symbols, e.g. Use type | None rather than Optional[type]
- use iterator patterns as much as practical. Also functional programming approaches, including partials (currying) and decorators
- Testing: keep the default run fast, cheap, and deterministic (unit only). Tests that hit unowned code or live services (DB, HTTP, browser) are integration tests — mark them `@pytest.mark.integration`, register the marker, and exclude them by default via `addopts = '-m "not integration"'` in `pyproject.toml`. Don't mock third-party internals; own the boundary and ship a real in-memory implementation alongside the production one. Let callers inject the `httpx.AsyncClient` so tests can use `MockTransport`. See the `testing` skill for the full pattern.
- Prompts sent to LLMs (Claude, OpenAI, OpenRouter, etc.) are content, not code. Keep them in [WordLoom](https://github.com/OoriData/WordLoom/) `.loom.toml` files under `prompts/`, loaded through a shared `prompts.py` module. Use WordLoom's [file-inclusion](https://github.com/OoriData/WordLoom/blob/main/implementation.md#extension-file-inclusion) (`file:` / `glob:` / `dir:`) to compose prompts from external assets rather than pasting them into TOML. No inline `system = 'You are...'` strings in Python source. See the `python-prompting` skill for the full pattern.

Note: `ruff format` rewrites `'''` docstrings to `"""`, contrary to the convention above, so use `ruff check` (with
`--fix`) rather than `ruff format` on whole files.
