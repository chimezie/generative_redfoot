# Design Patterns in Generative Redfoot

Generative Redfoot enables several powerful design patterns that support the creation of flexible, composable AI workflows as described in the main documentation. These patterns demonstrate how the core concepts, extensions, caching mechanisms, and service deployment capabilities work together to build sophisticated AI applications.

## 1. Contextual State Management
The system maintains conversational context through an accumulated context that can be updated and shared across PDL 
blocks. This enables complex multi-step workflows where each step builds on previous results.  It allows evaluation
of prompts engineered using shared libraries to serve as interconnected components in a lego system of workflows described
by PDL programs. This pattern is fundamental to how [Core Concepts](README.md#core-concepts) allow PDL programs to maintain conversational state using the `_` context structure.

## 2. Declarative Composition
PDL blocks can be composed declaratively, allowing complex workflows to be defined in YAML without complex programming logic. This includes:
- Model chaining with different LLMs (as shown in [Usage](README.md#usage) examples)
- Loop control through `repeat` blocks (`until` / `while` expressions, `for`, `maxIterations`, `index`)
- Context contribution controls (`contribute: [context, result]`), with nested blocks inheriting their container's role; each block reaches a later model as its own message, and same-role text messages are merged when a `text` list finishes
- Variable expressions using PDL's `${ ... }` syntax (Jinja), with the legacy `{ $variable_name }` form still resolving values from the context (as described in [Variables and Protocol Binding](README.md#variables-and-protocol-binding))

## 3. Extension-Based Architecture
The ParseDispatcher system allows for extensions to be registered and resolved based on content. This enables the capabilities described in the [Extensions](README.md#extensions) section, including:
- File reading with `read` blocks (documented in [Usage](README.md#usage))
- PDF processing with the four PDF reading modes: `PDF_raw_read_ocr`, `PDF_raw_read_txt`, `PDF_filename_ocr`, and `PDF_filename_txt` (as described in [PDF Reading](README.md#pdf-reading))
- Prompt templates with `read_from_wordloom` (documented in [Prompt Management via Wordloom](README.md#prompt-management-via-wordloom)), loaded through one shared loader (`prompts.py`) that resolves paths lexically, caches by file mtime, fills `_m` markers from the context, and composes prompts from external files via WordLoom file inclusion
- Custom model types through the PDLModel base class, supporting various LLM evaluation approaches like [Draft Model Support](README.md#draft-model-support), and [Alpha One Reasoning](README.md#alpha-one-reasoning)

## 4. Advanced Caching and Optimization
This pattern encompasses the [Caching](README.md#caching) capabilities described in the main documentation:
- **Internal Caching**: Uses mlx-lm's KV cache across model blocks; the cache records the tokens it holds, so each call sends only tokens beyond the shared prefix (trimming or rebuilding on divergence)
- **External Caching**: Creates and reuses persistent prompt cache files for faster subsequent executions
- **Prompt Caching**: Supports prefix caching with the `content_model` directive for expensive-to-process prompts (as detailed in the [Advanced Cache Preparation](README.md#advanced-cache-preparation-with-content_model) section)
- **Prefix Markers**: Uses `prefix_marker` to indicate the end of common prefixes for caching, as specified in the `content_model` parameters. The prefix is stored open (mlx-lm's `continue_final_message` convention) and a model block that names the cache sends only the text after it
- **Quantization Parameters**: Supports advanced KV cache quantization with `kv_group_size`, `quantized_kv_start`, and `kv_bits` parameters, providing the optimization capabilities described in the caching section

## 5. Service Orchestration
This pattern implements the [Service Deployment](README.md#service-deployment) capabilities:
- **REST API Deployment**: PDL programs can be deployed as web services using FastAPI (as documented in the Service Deployment section)
- **Request Processing**: Incoming requests are mapped to context variables for use in PDL execution; requests run one at a time in a worker thread, with models loaded once and reused
- **Protocol Parameter Binding**: Supports variable binding for request body content using `request_body_marker` (as detailed in [Variables and Protocol Binding](README.md#variables-and-protocol-binding))
- **Multi-format Support**: Handles various content types including text, PDF uploads, and structured data (leveraging the [PDF Reading](README.md#pdf-reading) extension)

## 6. Multi-Modal Input Processing
This pattern demonstrates how the various [Extensions](README.md#extensions) work together:
- **File Upload Support**: Handles PDFs, images, and other document types (using [PDF Reading](README.md#pdf-reading) capabilities)
- **OCR Processing**: Extracts text from scanned documents (using `PDF_raw_read_ocr` and `PDF_filename_ocr` as described in [PDF Reading](README.md#pdf-reading))
- **Variable Binding**: Supports dynamic references to context variables with expressions like `${ file }` (the legacy `{ $file }` form still works) for parameter injection, as shown in examples throughout the [Extensions](README.md#extensions) section

## 7. Model Enhancement Features
This pattern leverages the advanced model capabilities described in the [Extensions](README.md#extensions) section:
- **Draft Model Support**: Uses speculative decoding with draft models for faster inference (as documented in [Draft Model Support](README.md#draft-model-support))
- **Alpha One Reasoning**: Supports advanced reasoning with configurable thinking parameters (described in [Alpha One Reasoning](README.md#alpha-one-reasoning) with configurable `thinking_token_length`, `alpha`, and `wait_words` parameters)
- **Chain of Thought (CoT) Processing**: Enables few-shot learning through the `cot_prefix` parameter (documented in [Chain of Thought (CoT) prefix](README.md#chain-of-thought-cot-prefix))

These design patterns work together to create the sophisticated AI workflows demonstrated in the [Examples](README.md#examples) section, where multiple patterns combine to implement complex applications like document processing with cached prompts, web service orchestration, and multi-modal input processing.

## 8. Owned Boundaries and Testability
All use of mlx / mlx-lm sits behind a small backend interface (`MLXBackend` in `generation.py`: load, cache, prefill,
generate, trim). An `InMemoryBackend` implements the same interface deterministically, so the block logic (prefix
cache merging, suffix-only prompts, internal cache ledger, draft-model checks) and the FastAPI service (`build_app`)
are unit tested without Apple hardware, while `pytest -m integration` exercises the real backend. Sampling defaults
travel as a `GenerationDefaults` dataclass rather than being captured from the CLI, and a block's `parameters` override
them. See [Development and testing](README.md#development-and-testing).
