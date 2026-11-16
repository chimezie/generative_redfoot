# Changelog

## [0.3.0] - 20261116

### Added
- `AGENTS.md` with project conventions; `.gitignore`
- `prompts.py`: shared WordLoom loader (cached by file mtime, file-inclusion enabled, lexical path resolution)
- `generation.py`: `MLXBackend` (the only module that touches mlx / mlx-lm) and an `InMemoryBackend` test double
- `--trust-remote-code` option (default off), passed through to mlx-lm
- `enable_thinking` model parameter, passed to the chat template
- README: installation, sampling parameters, block reference, cache modes, request handling, environment variables,
  development and upgrade notes; `design_patterns.md` section on owned boundaries / testability
- Unit tests for the loader, loom block, caching / generation logic, and the service; integration tests for mlx

### Changed
- PDF extraction is one bytes-based function (`pdf_text.py`) shared by all `PDF_*` blocks, modeled on another project's
  `_extract_document_text`: timing log, whitespace-normalized OCR, SHA-256-keyed in-memory result cache, and a clear
  message when Tesseract (needed by the `*_ocr` modes) is missing; install instructions (Homebrew, apt, pixi) in the README
- Synced with current mlx-lm (>=0.32) and WordLoom (>=0.10.1); requires Python 3.12+; hatchling build
- MLX blocks moved out of the CLI closure into `extensions/mlx.py`; the CLI is now thin and `build_app` is testable
- Prefix caching follows the mlx-lm convention (`continue_final_message`, only the suffix is sent). **Prefix cache
  files from earlier versions must be regenerated.**
- Internal cache (`cache: '*'`) tracks the tokens it holds and sends only new ones, trimming or rebuilding as needed
- `read_from_wordloom`: bare filenames now resolve under `prompts/` (use `./name.toml` for the old behavior); items
  with markers are rendered through WordLoom (file-inclusion supported)
- `kv_bits` now defaults to no quantization (as documented); `kv-bits` is accepted with a warning
- `--top-k` default is 0 (disabled)
- Models are cached by the backend instead of being reloaded for every block / request; the service runs requests
  one at a time in a worker thread

### Removed
- Temp-file fallback for opening PDFs (the byte stream is now opened with an explicit filetype)
- Toolio integration (`structured_output` blocks)
- Dependencies on `ogbujipt`, `transformers` (transitive via mlx-lm), `pytest`, `tqdm`

### Fixed
- Internal cache ledger is discarded when a generation fails, so the next call rebuilds instead of trusting a stale cache
- Wordloom cache also invalidates when files beside / below the loom (`file:` inclusions) change
- PDF pages are no longer glued together (`patient` + `was` -> `patientwas`)
- `items:` tolerates repeated or trailing whitespace; the APS block downloads nltk data once per process
- The service runs generation on one dedicated worker thread (MLX streams are thread-local) and logs seconds, not minutes
- `draft_model` on a `content_model` block is ignored with a warning; the draft-cache error explains the alternative
- A plain-string `input:` on a model block (used in documented examples) raised an error; it is now a user message
- `top_p` was ignored outside Alpha-One generation
- Service crashed on a default `log_level`; final-message `assert` replaced by a 500 response; debug prints removed
- Undeclared dependencies on `fastapi`, `uvicorn`, `python-multipart`; wrong `pyaml` package name

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

