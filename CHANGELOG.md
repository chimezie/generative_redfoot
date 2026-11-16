# Changelog

## [0.3.0] - 20261116

Brings the project current with mlx-lm and WordLoom after a year, and makes it testable off Apple hardware. Verified
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

### Removed
- **Breaking:** Toolio integration (`structured_output` blocks)
- Dependencies on `ogbujipt`, `transformers` (transitive via mlx-lm), `pytest`, `tqdm`
- Temp-file fallback for opening PDFs (the byte stream is opened with an explicit filetype)

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

### Known limitations
- Alpha One reasoning is carried over unchanged and unverified against current alpha-one-mlx / mlx-lm
- PDL coverage is a small subset (README "PDL compatibility")

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

