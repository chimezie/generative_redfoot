# Changelog

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

