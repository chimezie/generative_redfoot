# Generative Redfoot
A generative, conversational workflow and multi-agent system using PDL and [mlx](https://github.com/ml-explore/mlx-examples/tree/main/llms)

Takes a minimal [Prompt Declaration Language (PDL)](https://github.com/IBM/prompt-declaration-language) file and generates a finite state generative machine
as Python objects for a subset of the PDL language.  These objects (the "programs" in particular) can be executed. 

It was mainly motivated by supporting this use case from the PDL documentation/paper:

<img src="animated_chatbot.gif" alt="Animated GIF of PDL chatbot."/>

The Model class can be extended and incorporated into how a dispatcher creates the PDL Python objects from a PDL file to incorporate the functionality for evaluating 
the prompts against the models specified in PDL.  This is done using any accumulated conversational context, prompts, and generation parameters (sampling parameters, for example), 
(optionally) updating the context as the program execution continues and how mlx is used to implement the model loading and inference.

However, the language of the PDL file can be extended with additional custom functionality, and 
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

## Extensions

Generative Redfoot defines a number of extensions from PDL

### PDF Reading

The `PDF_read` block can be used (within a `text` block) to read content from a PDF file (uses and requires [PyPDF2](https://pypdf2.readthedocs.io/en/3.x/)).
You can specify where the text is constibuted to (the 'context' is the most common scenario).  Below is an example
that reads PDF content from a run-time specified filename and uses it as context for a model evaluation: 

```yaml
description: autocoding_from_pdf
text:
  - text:
    - PDF_read: { $context_file }
      contribute: [context]
    - |
  
      Provide a list of the 5 ICD-10 codes mentioned in the document above
    contribute: [context]
  - model: mlx-community/medgemma-4b-it-4bit
```

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

### Caching
A PDL program can use mlx-lm's [Prompt caching](https://github.com/ml-explore/mlx-lm/blob/main/README.md#long-prompts-and-generations)
by specifying a top-level `cache` parameter at the top of the PDL document.  If the value of this parameter is _'*'_, 
then an internal, rotating K/V cache is used for the duration of the PDL evaluation for efficiency.

Otherwise, the value is expected to be the path to a previously saved cache, created using `mlx_lm.cache_prompt` for 
example, which is treated as a cached prompt that is a prefix to any prompts specified in the program. 

### Chain of Thought (CoT) prefix
If specified within a model block, the `cot_prefix` parameter takes a path to a file that captures COT few shot content 
as a file in the [LLM chat conversation](https://model-spec.openai.com/2025-02-12.html#definitions) JSON format.  This 
will be incorporated into the conversation structure to ensure this files content is used as few shot / COT for the 
model generation

## Complex example

Below is an example showing a PDL file constructing message contexts for prompts to chained LLM calls from fragments
in a [Wordloom](https://github.com/OoriData/OgbujiPT/wiki/Word-Loom%3A-Format-%26-tools-for-managing-natural-language-for-AI-LLMs) 
library, providing a clean separation of concerns between prompt language management, prompt construction, and 
LLM workflow management and orchestration.  The keys in the YAML file in black use the PDL language.  Those in
red are generative_redfoot extensions shown in order of appearance: (mlx) prefix caching, COT few-shot loading, 
reading from a wordloom file, using Google's [__google/gemma-7b-aps-it__](https://huggingface.co/google/gemma-7b-aps-it) 
model to perform ["abstractive proposition segmentation"](https://arxiv.org/abs/2406.19803) from LLM output, 
etc.:

<img src="complex_pdl.png" alt="Animated GIF of PDL chatbot."/>
