# Changelog

## [0.3.0] - 20261116

Brings the project current with mlx-lm and WordLoom after a year, makes it testable off Apple hardware, and aligns
its core blocks with PDL (`${ ... }` expressions, `defs`, `read`, `repeat`, roles, key validation). Verified
on Apple Silicon (unit, integration, OCR, CLI and service runs). **Breaking changes** are marked; see README
"Upgrading from 0.2".

### Added
- `generation.py`: `MLXBackend` (the only module that touches mlx / mlx-lm) and an `InMemoryBackend` test double
- `prompts.py`: shared WordLoom loader (lexical path resolution, file-inclusion enabled, cached by the mtimes of the
  loom file and the files beside / below it)
- `pdf_text.py`: one bytes-based PDF extraction function shared by all `PDF_*` blocks, with a timing log,
  whitespace-normalized OCR, and a SHA-256-keyed in-memory result cache
- `--trust-remote-code` option (default off), passed through to mlx-lm
- `enable_thinking` model parameter, passed to the chat template
- Unit tests (loader, loom block, caching / generation logic, PDF extraction, service) and integration tests (mlx, OCR)
- `AGENTS.md`, `.gitignore`; README sections for installation, sampling parameters, block reference, PDL
  compatibility, cache modes, request handling, Tesseract setup, environment variables, development, upgrading and
  roadmap; `design_patterns.md` section on owned boundaries / testability
- PDL `${ ... }` expressions (`expressions.py`): a string that is exactly one expression returns its typed value,
  mixed strings render to text, lists and dicts are evaluated recursively, `pdl_context` exposes the message list, and
  bad expressions raise `ValueError`s naming them.  Applied to values written in the PDL file (`text`, `read`,
  `message`, model `parameters` and string `input`, `PDF_*`); the legacy `{ $name }` reference still resolves context
  values
- `defs` is evaluated at the start of the block that declares it (later entries can use earlier ones; a block's result,
  a legacy `{ $name }` value or an expression can be bound), and `def` binds the result of any block.  Every block
  returns its result.  A text item or string `input` that is exactly one `${ ... }` is contributed as text.  A `defs` entry overwrites a variable of the same name
  (including `-v` and request variables), as in PDL
- `repeat` accepts `while`, `for` (one iteration per position, binding each name), `maxIterations` and `index`;
  `until` is now a full expression
- `read` of a filename, a default standard-input prompt, and `multiline` (read to end of file); relative file paths
  resolve from the PDL file's directory
- Role inheritance: nested blocks take their container's role unless they set one; model blocks stay `assistant`, and
  a model's `input` starts from `user`
- Unknown block keys are rejected when the program is built, suggesting the closest valid spelling; extension classes
  that declare no keys are unchecked
- `examples/pdf_service/`: a public PDF briefing service exercising the server, WordLoom prompts with file
  inclusion, a quantized prefix cache, and OCR

### Changed
- **Breaking:** requires Python 3.12+, mlx-lm >= 0.32, WordLoom >= 0.10.1; build backend is hatchling
- **Breaking:** prefix caching follows the mlx-lm convention (`continue_final_message`; only the suffix after the
  cached prefix is sent). Prefix cache files from earlier versions must be regenerated
- **Breaking:** `read_from_wordloom` bare filenames resolve under `prompts/` (use `./name.toml` for the old
  behavior); items with markers are rendered through WordLoom
- **Breaking:** `kv_bits` defaults to no quantization, as documented (was 4); `kv-bits` is accepted with a warning
- Internal cache (`cache: '*'`) tracks the tokens it holds and sends only new ones, trimming or rebuilding as needed
- MLX blocks moved out of the CLI closure into `extensions/mlx.py`; the CLI is thin and `build_app` is testable
- Models are loaded once per process instead of for every block / request
- The service runs requests one at a time on one dedicated worker thread (MLX streams are thread-local)
- Clear error when Tesseract (needed by the `*_ocr` modes) is missing
- `--top-k` default is 0 (disabled); the version is defined once, in `generative_redfoot/__init__.py`
- **Breaking:** a literal `${` in a PDL string is now an expression
- **Breaking:** `read: <string>` now reads that file (it used to fall through to standard input); `read:` without a
  `message` uses a default prompt instead of failing; relative `read` paths resolve from the PDL file's directory
- **Breaking:** `repeat`'s `until` must evaluate to a boolean, and a `repeat` with none of `until`, `while`, `for`,
  `maxIterations` is an error.  The usual `until: "${ question == 'quit' }"` form is unchanged
- **Breaking:** nested blocks inherit their container's `role`, and a text item joins the previous message only when
  the role matches.  Each block's contribution still reaches a later model as its own message, and a finished
  list-form `text` still merges neighbouring same-role text messages.  String items in a `text` block honour its
  `contribute`
- **Breaking:** stray block keys now fail when the program is built instead of being ignored
- `def` now binds `text` and `repeat` results too; `defs` is no longer ignored, and `PDLProgram.execute` returns its
  text's result
- `jinja2` is a dependency
- **Breaking:** `content_model`'s `max_kv_size` defaults to none (the cache is not capped; it was 10000), and
  combining it with `kv_bits` is an error, because mlx-lm cannot quantize a size-capped cache
- **Breaking:** form fields never override operator `-v` variables, and a form field named `_` is ignored; the
  ignored field names are logged at WARNING (never their values)
- **Breaking:** `PDF_raw_read_*` accept an upload, `bytes` or `bytearray` as PDF content and never open a path; a
  `str` is UTF-8 PDF content.  `PDF_filename_*` remains the way to read a file
- **Breaking:** `read` with a written `${ ... }` value must evaluate to a non-empty path (an empty or non-path value
  raises a `ValueError` naming the expression); `read:` with no value is the only standard-input form, and it is
  unavailable in a service
- **Breaking:** an unrecognised block raises `ValueError` (was `RuntimeError`)
- `until`, `while` and `maxIterations` ignore surrounding whitespace, so YAML block scalars (`>` / `|`) work; a
  `repeat` body may be a string or list, treated as a `text` block
- The CLI reports run-phase author mistakes (`ValueError`, `KeyError`, `FileNotFoundError`) as
  `<pdl file>: <message>` without a traceback, no longer treats `RuntimeError` as an author mistake, and re-raises
  the original exception with `--verbose`

### Removed
- **Breaking:** Toolio integration (`structured_output` blocks)
- Dependencies on `ogbujipt`, `transformers` (transitive via mlx-lm), `pytest`, `tqdm`
- Temp-file fallback for opening PDFs (the byte stream is opened with an explicit filetype)
- The `alpha_one` extra: alpha-one-mlx is not on PyPI, so the extra made the project unresolvable for `uv run` /
  `uv sync`; install alpha-one-mlx separately to use `alpha_one` blocks

### Fixed
- `top_p` was ignored outside Alpha-One generation
- A plain-string `input:` on a model block raised an error; it is now a user message
- Internal cache state is discarded when a generation fails, instead of being trusted on the next call
- PDF pages are no longer glued together (`patient` + `was` -> `patientwas`)
- `items:` tolerates repeated or trailing whitespace; the APS block fetches nltk data once per process
- `draft_model` on a `content_model` block is ignored with a warning; the draft-cache error explains the alternative
- Service: crash on a default `log_level`; final-message `assert` replaced by a 500 response; elapsed time logged in
  seconds; debug prints removed
- Undeclared dependencies on `fastapi`, `uvicorn`, `python-multipart`; wrong `pyaml` package name
- A user turn after an assistant reply is no longer appended to that reply
- A list-form `text` prints its strings as plain text instead of pretty-printing an empty string
- Every failure while evaluating a `${ ... }` expression is a `ValueError` containing the expression, the original
  exception type and its message
- A malformed `cache` section, a missing `content_model` `name` / `file`, or a misspelled cache key (e.g. `nmae:`)
  raises a clear `ValueError` instead of a bare `KeyError`
- The WordLoom cache signature prunes hidden directories, `__pycache__` and `node_modules`, stops after 500 files
  with one process-wide warning, and skips files that vanish mid-walk
- Prefix caches store the render date as `date_string` in their metadata, so a cache built before midnight still
  matches a Llama 3.x-style template that embeds today's date

### Known limitations
- Alpha One reasoning is carried over unchanged and unverified against current alpha-one-mlx / mlx-lm
- PDL coverage is a subset (README "PDL compatibility"): no `parser`, `if`, `include`, code blocks or remote models
- A stuck request blocks the one-at-a-time service; there is no per-request timeout
- There is no limit on request size or prompt length (the default `content_model` cache is uncapped)
- Loaded models stay in memory for the life of the process
- The `eos_token` model parameter is ignored when a model block uses a named prefix cache
- A model block whose `contribute` omits `context` clears the whole message list (inherited from earlier versions)
- `PDF_filename_*` relative paths resolve from the current directory, while `read` paths resolve from the PDL
  file's directory
- Named prefix caches are re-read from disk on every request

## [0.2] - 20251119

### Added
- Detailed documentation on Generative Redfoot's design patterns and architecture
- Prompt caching capabilities with mlx-lm support for efficient inference
- FastAPI service deployment for PDL programs with RESTful API endpoints
- Four new PDF reading modes: PDF_raw_read_ocr, PDF_raw_read_txt, PDF_filename_ocr, and PDF_filename_txt
- Support for draft models with speculative decoding for faster inference
- Advanced cache preparation using content_model directive with quantization parameters
- Form data processing support in web service endpoints
- Loom markers display in verbose info output
- Multi-modal input processing capabilities
- Service orchestration with request body marker binding and content type handling

### Changed
- Enhanced PDL model with contextual state management and multi-modal input support
- Refactored CLI with improved context management and caching mechanisms
- Migrated wordloom extension to separate loom package to avoid import conflicts
- Replaced PyPDF2 with PyMuPDF for more robust PDF processing with OCR support
- Enhanced PDFRead class with file path validation and UploadFile integration
- Improved CLI documentation with caching and service deployment details
- Added type conversion for sampling parameters (temperature, min_p, top_k) to ensure proper data types
- Enhanced form data processing to handle UploadFile objects properly
- Improved error handling for PDF content processing with temporary file fallback

### Fixed
- Fixed PDF text extraction logic and removed unused dispatch_check abstract method
- Fixed content type validation in FastAPI service endpoints to make it optional
- Fixed cache loading to extract messages from cache metadata when loading from cache
- Fixed context initialization in CLI when no variables are provided
- Fixed time logging in service requests to show minutes properly

