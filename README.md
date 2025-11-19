# Generative Redfoot
A generative, conversational workflow and multi-agent system using PDL and [mlx](https://github.com/ml-explore/mlx-examples/tree/main/llms)

## Table of Contents
- [Introduction](#generative-redfoot)
- [Design Patterns](design_patterns.md)
- [Core Concepts](#core-concepts)
- [Usage](#usage)
- [Caching](#caching)
- [Service Deployment](#service-deployment)
- [Practical Applications](#practical-applications)
- [Extensions](#extensions)
  - [PDF Reading](#pdf-reading)
  - [Toolio](#toolio)
  - [Draft Model Support](#draft-model-support)
  - [Alpha One Reasoning](#alpha-one-reasoning)
  - [Prompt Management via Wordloom](#prompt-management-via-wordloom)
  - [Chain of Thought (CoT) prefix](#chain-of-thought-cot-prefix)
- [Examples](#examples)

For a detailed explanation of design patterns enabled by Generative Redfoot, see the [Design Patterns](design_patterns.md) documentation.

## Core Concepts

Generative Redfoot is built around several core concepts that enable declarative, composable AI workflows:

### Prompt Declaration Language (PDL)
The Prompt Declaration Language (PDL) is a [YAML-based declarative approach to prompt programming](https://www.ibm.com/granite/docs/use-cases/prompt-declaration-language/), where 
prompts are at the forefront. It facilitates model evaluation chaining and tool use, abstracting away the plumbing needed for composition and hyperparameter setting and management.  Generative Redfoot is a PDL-based
declarative approach to creating orchestrated, generative AI workflows and services, conceived as prompt programming composition. 

### Relationship to Original Redfoot
Generative Redfoot originally was meant as a modern replacement 
for the original RDFLIb Redfoot, an extensible RDF service architecture for building a distributed, Semantic Web of peer-to-peer nodes with modern generative AI systems and technologies in mind.  

Generative Redfoot recasts this paradigm further with firm roots in declarative and RESTful principles in how generative AI actions and capabilities can be composed and orchestrated. 
Beyond RDFLib Redfoot, the framework it facilitates was mainly motivated by supporting this use case from the PDL documentation/paper:

<img src="animated_chatbot.gif" alt="Animated GIF of PDL chatbot."/>

### Object Model
The system takes a [Prompt Declaration Language (PDL)](https://github.com/IBM/prompt-declaration-language) file and generates a finite state generative machine
as Python objects for a subset of the PDL language.  These objects (the "programs" in particular) can be executed and served as web services that
take parameters and return results. 

The Model class can be extended and incorporated into how a dispatcher creates the PDL Python objects from a PDL file to incorporate the functionality for evaluating 
the prompts against the models specified in PDL.  This is done using any accumulated conversational context, prompts, and generation parameters (sampling parameters, for example), 
(optionally) updating the context as the program execution continues and how mlx is used to implement the model loading and inference.

The language of the PDL file can be extended with additional custom functionality, and 
other LLM systems can handle the evaluation.

## Usage

It depends on the PyYaml and click third-party Python libraries as well as mlx and can be run this way, where `document.pdl` is a PDL file.
```commandline
% Usage: generative_redfoot [OPTIONS] PDL_FILE

Options:
  -t, --temperature FLOAT
  -rp, --repetition-penalty FLOAT
                                  The penalty factor for repeating tokens
                                  (none if not used)
  --top-k INTEGER                 Sampling top-k
  --top-p INTEGER                 Sampling top-p  
  --max-tokens INTEGER            Max tokens
  --min-p FLOAT                   Sampling min-p
  --verbose / --no-verbose
  -v, --variables <TEXT TEXT>...
  --help                          Show this message and exit.

generative_redfoot.py document.pdl
```

The main argument is a PDL program (a YAML file), possibly with extensions of the language implemented by generative_redfoot.

You can also specify default values for sampling parameters for the LLM calls during the execution of the programs
using mlx.

The model _parameters_ directive in PDL can be used to specify the following mlx generation parameters: **temperature**, **top_k**, **min_p**, **max_tokens**, and **top_p**:

```yaml
description: ...
text:
  - read:
    message: |
      What is your query?
    contribute: [context]
- model: .. model ..
  parameters:
    temperature: 0.6
    min_p: .03
    max_tokens: 200
```

When a PDL file contains a `server` section, generative_redfoot will automatically start a FastAPI web service in which to execute the program.

## Caching

Generative Redfoot includes advanced caching capabilities for efficient inference:

### Internal vs External Caching

Generative Redfoot supports two types of prompt caching:

1. **Internal Caching**: Uses mlx-lm's built-in KV cache for efficiency during program execution
2. **External Caching**: Creates and reuses persistent prompt cache files for faster subsequent executions

### Basic Caching

To use basic internal caching, specify a top-level `cache` parameter in your PDL file:

```yaml
cache: "*"
description: ...
text:
  - model: mlx-community/Llama-3.2-3B-Instruct-4bit
    # ... rest of your program
```

### Advanced Cache Preparation with content_model

For persistent caching, you can create cache files using the `content_model` directive:

```yaml
cache:
  - text:
    - role: system
      content: "You are a helpful assistant"
    - content_model: mlx-community/Llama-3.2-3B-Instruct-4bit
      name: my_cache
      file: my_prompt_cache.safetensors
```

The `content_model` directive creates prompt caches that can be reused across multiple executions. This is particularly useful for prompts that don't change frequently but are expensive to process.

```yaml
cache:
  - text:
    - role: system
      read_from_wordloom: prompt_library.toml
      items: helpful_chatbot_system_prompt
    - content_model: mlx-community/Llama-3.2-3B-Instruct-4bit
      name: chat_cache
      file: chat_prompt_cache.safetensors
      kv_group_size: 64
      kv_bits: 4
      max_kv_size: 10000
```

Parameters for `content_model`:
- `name`: Identifier for referencing the cache in model blocks
- `file`: Path to save the prompt cache file
- `kv_group_size`: Group size for KV cache quantization (default: 64)
- `kv_bits`: Number of bits for KV cache quantization (default: 4)
- `quantized_kv_start`: When kv_bits is set, start quantizing the KV cache from this step onwards (default: 5000)
- `max_kv_size`: Maximum key-value cache size (default: 10000)
- `prefix_marker`: Wordloom marker indicating the end of the common prefix for caching

These caches can be referenced in model blocks:

```yaml
text:
  - model: chat_cache
    parameters:
      temperature: 0.7
      max_tokens: 500
```

## Service Deployment

PDL programs can be deployed as web services using FastAPI. Add a `server` section to your PDL file:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  request_body_marker: document_text
  path: /generate
  content_type: text/plain
  log_level: info
  log_path: /tmp/service.log

description: ...
text:
  - model: mlx-community/Llama-3.2-3B-Instruct-4bit
    # ... rest of your program
```

When you run the PDL file with generative_redfoot, it will automatically start a FastAPI service instead of executing locally.

Server configuration parameters:
- `host`: Host address to bind the service to
- `port`: Port number for the service
- `request_body_marker`: Context variable name for the incoming request body (acts as a wordloom marker for request binding)
- `path`: API endpoint path
- `content_type`: Expected content type of incoming requests (optional - if omitted, any content type is accepted)
- `log_level`: Logging level (debug, info, warning, error)
- `log_path`: Path to log file

The service accepts POST requests with the specified content type and processes the request body using the PDL program. The result is returned as plain text.

The `request_body_marker` is used to bind incoming request content to a context variable that can then be referenced throughout the PDL program using variable syntax like `{ $document_text }`.

### Variable References and Protocol Binding

Generative Redfoot supports dynamic variable references using the syntax `{ $variable_name }` to bind values from the execution context. This is particularly useful for:

- **Service Protocol Parameters**: Binding request body content to variables (e.g., `{ $file }` for file uploads)
- **Contextual Content**: Referencing values from the execution context in PDL blocks
- **Parameter Injection**: Dynamically injecting values based on runtime conditions

Example of variable reference in PDF processing:
```yaml
text:
  - text:
    - PDF_raw_read_ocr: { $file }
      contribute: [context]
```

### Caching
A PDL program can use mlx-lm's [Prompt caching](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations)
by specifying a top-level `cache` parameter at the top of the PDL document.  If the value of this parameter is _'*'_, 
then an internal, rotating K/V cache is used for the duration of the PDL evaluation for efficiency.

Otherwise, the value is expected to be the path to a previously saved cache, created using `mlx_lm.cache_prompt` for 
example (or the cache preparation capabilities described above), which is treated as a cached prompt that is a prefix to 
any prompts specified in the program. 

## Extensions

Generative Redfoot defines a number of extensions from PDL

### PDF Reading

Generative Redfoot now supports four distinct PDF reading modes using PyMuPDF, each designed for different input scenarios:

- `PDF_raw_read_ocr`: Extract text using OCR from raw PDF content
- `PDF_raw_read_txt`: Extract text directly from raw PDF content
- `PDF_filename_ocr`: Extract text from a PDF file using OCR
- `PDF_filename_txt`: Extract text directly from a PDF file

The choice of which extension to use depends on your input format and processing requirements:

**File-based reading** (`PDF_filename_ocr` and `PDF_filename_txt`):
Use these when you have a file path available. The `_ocr` variant is for scanned documents, while the `_txt` variant is for documents with selectable text.

**Raw content reading** (`PDF_raw_read_ocr` and `PDF_raw_read_txt`):
Use these when working with PDF content as raw bytes or when the PDF is provided as part of a web request (e.g., file uploads in the service deployment context).

Here's an example that reads PDF content from a run-time specified filename using direct text extraction and uses it 
as context for a model evaluation: 

```yaml
description: autocoding_from_pdf
text:
  - text:
    - PDF_filename_txt: { $context_file }
      contribute: [context]
    - |

      Provide a list of the 5 ICD-10 codes mentioned in the document above
    contribute: [context]
  - model: mlx-community/medgemma-4b-it-4bit
```

For OCR-based extraction from a file:
```yaml
description: autocoding_from_scanned_pdf
text:
  - text:
    - PDF_filename_ocr: { $context_file }
      contribute: [context]
    - |

      Provide a list of the 5 ICD-10 codes mentioned in the document above
    contribute: [context]
  - model: mlx-community/medgemma-4b-it-4bit
```

For processing PDF content provided as raw bytes (useful in service deployment contexts):
```yaml
description: pdf_processing_from_upload
text:
  - text:
    - PDF_raw_read_txt: { $uploaded_pdf_content }
      contribute: [context]
    - |

      Summarize the key points from the document
    contribute: [context]
  - model: mlx-community/medgemma-4b-it-4bit
```

#### Advanced PDF Processing Features

The enhanced PDF processing includes robust error handling and content validation:

- **File Path Validation**: Checks for existence and validity of PDF file paths
- **Raw Content Processing**: Handles both file paths and raw PDF byte content
- **UploadFile Support**: Direct integration with FastAPI's UploadFile objects for web services
- **Content Type Detection**: Proper handling of different PDF content types
- **OCR and Text Extraction**: Supports both direct text extraction and OCR processing for scanned documents
- **Temporary File Handling**: Fallback mechanism for problematic PDF files that PyMuPDF can't process directly

### Toolio

[Toolio](https://github.com/OoriData/Toolio) can be used for structured output by specifying a `structured_output`
block like so (from [Toolio algebra tutor demo](https://github.com/OoriData/Toolio/blob/main/demo/algebra_tutor.py)):

```yaml
description: structured_output
text:
  - structured_output: mlx-community/Llama-3.2-3B-Instruct-4bit
    insert_schema: true
    schema_file: ToolioGit/demo/algebra_tutor.schema.json
    parameters:
      temperature: 0.6
      max_tokens: 512
    input: |
      solve 8x + 31 = 2. Your answer should be only JSON, according to this schema: #!JSON_SCHEMA!#"
```

Beyond the approach above, its input can be specified in all the ways a PDL model can.

### Draft Model Support

Generative Redfoot supports speculative decoding using draft models for faster inference. Add the `draft_model` parameter to your model block:

```yaml
description: draft_model_example
text:
  - model: mlx-community/Llama-3.2-3B-Instruct-4bit
    draft_model: mlx-community/Llama-3.2-1B-Instruct-4bit
    parameters:
      temperature: 0.6
      max_tokens: 500
```

Draft models must have the same vocabulary size as the main model and can significantly improve generation speed.

### Alpha One Reasoning

[AlphaOne (&alpha;)](https://alphaone-project.github.io/) reasoning modulation can be used with 
a supported, reasoning model using [Alpha One MLX](https://github.com/chimezie/alpha-one-mlx) via providing
a `alpha_one` block within a model:

```yaml
description: alpha_one_reasoning
text:
    - model: mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit-AWQ
      parameters:
        temperature: 0.6
        max_tokens: 1200
        repetition_penalty: 1.4
        top_k: 20
        top_p: 0.95
      alpha_one:
        thinking_token_length: 250
        alpha: 1.4
        wait_words: ["Wait"]
      input: |
        What question has an answer of "42?"
```

This algorithm scales the average thinking phase token length (specified by the `thinking_token_length` parameter and 
with a default of 2,650) by the `alpha` parameter (which defaults to 1.4 per the paper).  After
exiting this phase, it suppresses attempts to transition to slow thinking by replacing references to 'Wait' words
with the _"&lt;/think>"_ token.  The specific list of these words can be provided via the `wait_words` parameter or the
defaults specified by alpha-one-mlx for each model type will be used.

### Prompt management via Wordloom

Prompt snippets can be loaded for use in a PDL file via a [word loom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom%3A-Format-%26-tools-for-managing-natural-language-for-AI-LLMs) 
library, providing a clean separation of concerns between prompt language management, prompt construction, and 
LLM workflow management and orchestration.  This requires [OgbujiPT](https://github.com/OoriData/OgbujiPT) and below is
an example that constructs the system and user prompt from a word loom ([TOML](https://toml.io/en/)) file: 

```yaml
description: wordloom_prompt_example
text:
  - text:
    - role: system
      read_from_wordloom: prompt_library.toml
      items: helpful_chatbot_system_prompt
      contribute: [context]
    - text:
      - read_from_wordloom: prompt_library.toml
        items: hello_prompt
        contribute: [context]
    - model: mlx-community/Llama-3.2-3B-Instruct-4bit
```

The `read_from_wordloom` block indicates the use of this extension mechanism and its value is a path to a Word loom.

The `items` parameter on the block is a space-separated list of language item names in the word loom.  Their values are 
joined together with `\n` and returned as a value for use (as an example from above) to construct the message for 
conversations used by downstream PDL directives.  

In the example above, the `helpful_chatbot_system_prompt` language item from the `prompt_library.toml` word loom is used
as the system prompt and the `hello_prompt` language item from the same file is used as the user prompt:

```JSON
[
  {"role": "system", "content": ".. helpful_chatbot_system_prompt language item .. "},
  {"role": "user", "content": ".. hello_prompt language item .. "}
]
```

### Chain of Thought (CoT) prefix
If specified within a model block, the `cot_prefix` parameter takes a path to a file that captures COT few shot content 
as a file in the [LLM chat conversation](https://model-spec.openai.com/2025-02-12.html#definitions) JSON format.  This 
will be incorporated into the conversation structure to ensure this files content is used as few shot / COT for the 
model generation

## Examples

Below is an example showing a PDL file constructing message contexts for prompts to chained LLM calls from fragments
in a [Wordloom](https://github.com/OoriData/WordLoom) 
library, providing a clean separation of concerns between prompt language management, prompt construction, and 
LLM workflow management and orchestration.  The keys in the YAML file in black use the PDL language.  Those in
red are generative_redfoot extensions shown in order of appearance: (mlx) prefix caching, COT few-shot loading, 
reading from a wordloom file, using Google's [__google/gemma-7b-aps-it__](https://huggingface.co/google/gemma-7b-aps-it) 
model to perform ["abstractive proposition segmentation"](https://arxiv.org/abs/2406.19803) from LLM output, 
etc.:

<img src="complex_pdl.png" alt="Animated GIF of PDL chatbot."/>