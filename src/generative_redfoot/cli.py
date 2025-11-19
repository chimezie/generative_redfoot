
"""
Command-line interface for Generative Redfoot, a generative, conversational workflow and multi-agent system
using PDL (Prompt Declaration Language) and MLX. Generative Redfoot is a PDL-based declarative approach to 
creating orchestrated, generative AI workflows and services, conceived as prompt programming composition.

This module provides:
1. Cache Preparation: Handles 'content_model' directives to create prompt caches for faster inference
2. Model Evaluation: Executes LLM inference using cached prompts and various sampling strategies
3. Service Orchestration: Launches FastAPI services based on PDL server configuration for RESTful deployment

The tool processes PDL files that define complex LLM workflows, implementing the design patterns for
composable AI workflows as described in the project documentation, including contextual state management,
extension-based architecture, and service orchestration.
"""
import click
import yaml
import json
import re
import time
import logging
import uvicorn

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response

from .utils import truncate_long_text
from .object_pdl_model import PDLModel, PDLProgram, ParseDispatcher, PDFRead, PDLRepeat, PDLText, PDLRead
from .extensions.loom import WorldLoomRead
from .extensions.toolio import ToolioCompletion
from transformers import PreTrainedTokenizer
from typing import Tuple, Dict, List, Mapping

@click.command()
@click.option('-t', '--temperature', default=1, type=float)
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--top-k', default=-1, type=int, help='Sampling top_k')
@click.option('--top-p', default=0.95, type=float, help='Sampling top_p')
@click.option('--max-tokens', default=800, type=int, help='Max tokens')
@click.option('--min-p', default=0, type=float, help='Sampling min-p')
@click.option('--verbose/--no-verbose', default=False)
@click.option("--variables", "-v", "variables", type=(str, str),  multiple=True)
@click.argument('pdl_file')
def main(temperature, repetition_penalty, top_k, top_p, max_tokens, min_p, verbose, variables, pdl_file):
    """
    Main entry point for the Generative Redfoot CLI tool that processes PDL (Prompt Declaration Language) 
    files to execute complex, declarative AI workflows.

    This tool implements the design patterns for flexible, composable AI workflows that include:

    1. Cache Preparation (Advanced Caching and Optimization):
       - Handles 'content_model' directives in the PDL cache section for persistent caching
       - Creates prompt caches using mlx_lm for faster subsequent inference
       - Supports KV cache quantization parameters for memory optimization
       - Implements both internal and external caching mechanisms

    2. Model Evaluation (Model Enhancement Features):
       - Evaluates models referenced by name (e.g., 'user_cache') from cached files
       - Loads from cached prompt files when available for efficient execution
       - Supports advanced features like Alpha-One reasoning and draft model speculative decoding

    3. Service Orchestration:
       - Launches a FastAPI web service when 'server' directive is present in PDL
       - Configures host, port, and request handling based on PDL parameters
       - Processes incoming requests by executing the PDL program with request content
       - Supports multi-format content processing with variable binding via request_body_marker

    The tool embodies the core concepts of contextual state management and declarative composition,
    enabling complex multi-step workflows where each step builds on previous results.

    Args:
        temperature (float): Sampling temperature for text generation
        repetition_penalty (float): Penalty for repeating tokens
        top_k (int): Top-K sampling parameter
        top_p (float): Top-P (nucleus) sampling parameter
        max_tokens (int): Maximum number of tokens to generate
        min_p (float): Minimum probability threshold for sampling
        verbose (bool): Enable verbose output
        variables (tuple): Key-value pairs to inject into the PDL context for dynamic configuration
        pdl_file (str): Path to the PDL program file to execute
    """
    import mlx.nn as nn
    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_lm.generate import generate, generate_step
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    start_marker = '<s>'
    end_marker = '</s>'
    separator = '\n'

    def create_propositions_input(text: str) -> str:
        import nltk
        nltk.download('punkt')
        input_sents = nltk.tokenize.sent_tokenize(text)
        propositions_input = ''
        for sent in input_sents:
            propositions_input += f'{start_marker} ' + sent + f' {end_marker}{separator}'
        propositions_input = propositions_input.strip(f'{separator}')
        return propositions_input

    def process_propositions_output(text):
        pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
        output_grouped_strs = re.findall(pattern, text)
        predicted_grouped_propositions = []
        for grouped_str in output_grouped_strs:
            grouped_str = grouped_str.strip(separator)
            props = [x[2:] for x in grouped_str.split(separator)]
            predicted_grouped_propositions.append(props)
        return predicted_grouped_propositions

    class CachePrep(PDLModel):
        """
        Cache Preparation model that handles 'content_model' instances in the PDL cache
        'cache' section, implementing the Advanced Caching and Optimization design pattern. 
        Extracts caching parameters and creates an mlx_lm prompt cache file using the content 
        generated in the preceding PDL steps, storing the cache in the specified `file` using a `name`
        that can be referenced by a model later in the execution that wishes to use the cache.

        A 'prefix_marker' property can specify the name of a wordloom marker that indicates the end of the common
        prefix to use for the cache, supporting the Contextual State Management pattern by maintaining 
        accumulated context through PDL blocks.

        The `kv_group_size`, `quantized_kv_start`, `kv_bits`, and `max_kv_size` properties can be used to control
        the caching behavior:
        - kv_group_size: Group size for KV cache quantization (default: 64)
        - quantized_kv_start: When kv_bits is set, start quantizing the KV cache from this step onwards (default: 5000)
        - kv_bits: Number of bits for KV cache quantization. Defaults to no quantization (default: 4)
        - max_kv_size: Maximum key-value cache size (default: 10000)

        Implements the caching mechanisms described in the project documentation's Caching section
        for efficient inference in generative AI workflows.
        """
        MODEL_KEY = "content_model"
        def __init__(self, content: Mapping, program):
            self.program = program
            self.model = content[self.MODEL_KEY]
            self.content = content
            self.name = self.content["name"]
            self.prefix_marker = self.content.get("prefix_marker")
            self.file = content["file"]
            self.kv_group_size = self.content.get("kv_group_size", 64)
            self.quantized_kv_start = self.content.get("quantized_kv_start", 5000)
            self.kv_bits = self.content.get("kv_bits", 4)
            self.max_kv_size = self.content.get("max_kv_size", 10000)
            self._get_common_attributes(content)

        @staticmethod
        def dispatch_check(item: Dict, program: PDLProgram):
            if "content_model" in item:
                return CachePrep(item, program)

        def _get_model_cache_and_tokenizer(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
            tokenizer_config = {}
            model, tokenizer = load(self.model, tokenizer_config=tokenizer_config)
            return model, tokenizer

        def execute(self, context: Dict, verbose: bool = False):
            model, tokenizer = load(self.model, tokenizer_config={})
            cache = make_prompt_cache(model, self.max_kv_size)
            messages = []
            if "_" in context and context["_"]:
                messages.extend(context["_"])
            if verbose:
                from pprint import pprint
                print("Generating response using ..")
                pprint([{k: v if k == "role" else truncate_long_text(v)} for i in messages for k,v in i.items()])
            else:
                print("Generating response ... ")

            m = [_ for _ in messages if _['role'] == 'user']
            assert len(m) == 1
            m = m[0]
            if self.prefix_marker:
                #Only use the prefix up to the first occurrence of the marker
                m['content'] = m['content'].split(f"{{{self.prefix_marker}}}")[0]
                if verbose:
                    print(f"Separating at prefix marker: {self.prefix_marker}: '.. {m['content'][-20:]}'")
            self.generate(messages, tokenizer, model, cache=cache)
            if verbose:
                print(f"Executed cache model: {self.model} using context {context} - (via mlx)")

            #Save the cache and its metadata
            metadata = {}
            metadata["model"] = self.model
            metadata["messages"] = messages
            metadata["chat_template"] = json.dumps(tokenizer.chat_template)
            tokenizer_config = {}
            metadata["tokenizer_config"] = json.dumps(tokenizer_config)
            save_prompt_cache(self.file, cache, metadata)
            if verbose:
                print(f"Saved prompt cache to {self.file}")

        def generate(self, messages, tokenizer, model, cache=None):
            """
            Store the cache
            """
            prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

            y = mx.array(prompt)
            # Process the prompt
            start = time.time()
            max_msg_len = 0

            def callback(processed, total_tokens):
                current = time.time()
                speed = processed / (current - start)
                msg = f"\rProcessed {processed:6d} tokens ({speed:6.2f} tok/s)"
                nonlocal max_msg_len
                max_msg_len = max(max_msg_len, len(msg))
                if verbose:
                    print(msg + " " * (max_msg_len - len(msg)), end="", flush=True)

            for _ in generate_step(
                    y,
                    model,
                    max_tokens=0,
                    prompt_cache=cache,
                    kv_bits=self.kv_bits,
                    kv_group_size=self.kv_group_size,
                    quantized_kv_start=self.quantized_kv_start,
                    prompt_progress_callback=callback
            ):
                pass

    class MLXModelEvaluationBase(PDLModel):
        """
        Base class for MLX model evaluation that implements the Advanced Caching and Optimization design pattern.
        This class handles loading models and managing both internal and external caching mechanisms to support 
        efficient inference in generative AI workflows.

        When a model is referenced by name (e.g., 'autocode_cache'), this class:
        1. Checks if the model name exists in the program's cache_lookup
        2. If found, loads the model from the cached file and sets up the prompt cache
        3. If not found, loads the model directly from its path
        4. Handles both internal and external cache mechanisms as described in the project documentation

        This implementation supports the Model Enhancement Features pattern with support for draft models
        and various sampling strategies, enabling efficient execution of complex LLM workflows.
        """
        def _get_model_cache_and_tokenizer(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
            eos_token = self.parameters.get("eos_token")
            if eos_token:
                tokenizer_config = {"eos_token": eos_token}
            else:
                tokenizer_config = {}
            if self.model in self.program.cache_lookup:
                #load model from file, set it for use in generation later, and extract the model path and tokenizer config
                prompt_cache, metadata = load_prompt_cache(self.program.cache_lookup[self.model], return_metadata = True)
                self.program.cache = prompt_cache
                model_path = metadata["model"]
                if verbose:
                    print(f"Set cache to content from {self.program.cache_lookup[self.model]} for {model_path}")
                tokenizer_config = json.loads(metadata["tokenizer_config"])
            else:
                model_path = self.model
            model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)
            if isinstance(self.program.cache, str):
                if self.program.cache == PDLProgram.INTERNAL_CACHE_NAME:
                    self.program.cache = (make_prompt_cache(model))
                    if verbose:
                        print("Using internal cache for prompts")
                else:
                    program_cache = self.program.cache
                    self.program.cache = load_prompt_cache(program_cache)
                    if verbose:
                        print(f"Using external cache for prompts: {program_cache}")
            return model, tokenizer

        def generate(self, messages, tokenizer, model, verbose):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if verbose:
                cache_info = " and cache" if self.program.cache else ""
                print(f"Using parameters: {self.parameters}{cache_info} for {self.model}")
            logits_processor = make_logits_processors(repetition_penalty=self.parameters.get("repetition_penalty",
                                                                                             repetition_penalty))
            draft_model = self.draft_model
            if draft_model:
                from mlx_lm.utils import load
                draft_model, draft_tokenizer = load(self.draft_model)
                if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                    raise ValueError("Draft model tokenizer does not match model tokenizer.")
                elif verbose:
                    print(f"Using draft model: {self.draft_model}")
            return generate(model, tokenizer, prompt,
                            max_tokens=self.parameters.get("max_tokens", max_tokens),
                            sampler=make_sampler(temp=float(self.parameters.get("temperature", temperature)),
                                                 min_p=float(self.parameters.get("min_p", min_p)),
                                                 top_k=int(self.parameters.get("top_k", top_k))),
                            logits_processors=logits_processor,
                            verbose=verbose,
                            prompt_cache=self.program.cache,
                            draft_model=draft_model), prompt

    class MLXModelEvaluation(MLXModelEvaluationBase):
        """
        MLX model evaluation implementation that executes model inference using the MLX framework
        and implements multiple design patterns for flexible AI workflows.

        This class embodies the following design patterns from the project documentation:
        1. Loading models and caches through the base class (Advanced Caching and Optimization)
        2. Processing messages and context for model input (Contextual State Management)
        3. Supporting chain-of-thought (CoT) prefixes for few-shot learning (Model Enhancement Features)
        4. Handling both regular generation and Alpha-One reasoning (Model Enhancement Features)
        5. Managing draft models for speculative decoding (Model Enhancement Features)
        6. Contributing results back to the execution context (Contextual State Management)

        The class supports the declarative composition pattern by processing PDL blocks that can be
        chained together to create complex multi-step workflows. It supports loading from cached 
        prompts when the model name exists in program.cache_lookup, using the prefix cache file 
        stored with the model as described in the project's caching section.
        """
        def _insert_cot_messages(self, messages: List[Dict], cot_prefix: List[Dict]):
            """
            Modifies LLM messaging with a chain-of-thought (COT) prefix after any system message.

            :param messages: List of message dictionaries where each dictionary contains keys like
                'role' or 'content' and other relevant items for message processing.
            :param cot_prefix: Chain-of-thought (COT) prefix, provided as a list of dictionaries
                that serve as the preparatory context/instructions to be inserted into the message
                list,  when the first item in `messages` is associated with the 'system' role.
            :return: Updated `messages` list with the COT prefix properly inserted when applicable.
            """
            idx = 1 if messages[0]['role'] == 'system' else 0
            messages[idx:idx] = cot_prefix
            return messages

        def execute(self, context: Dict, verbose: bool = False):
            model, tokenizer = self._get_model_cache_and_tokenizer()
            messages = []
            if self.input:
                self.input.execute(context, verbose=verbose)
            if "_" in context and context["_"]:
                messages.extend(context["_"])
            elif self.program.cache_lookup and self.model in self.program.cache_lookup:
                #If loading from cache, extract the messages from the cache metadata
                _, metadata = load_prompt_cache(self.program.cache_lookup[self.model], return_metadata = True)
                messages.extend(metadata["messages"])
            else:
                raise ValueError("No context found for model evaluation.")
            if self.cot_prefix:
                print(f"### Adding Chain-of Thought Few Shot examples specified in {self.cot_prefix} ###")
                with open(self.cot_prefix, 'r') as cot_content:
                    self._insert_cot_messages(messages, json.load(cot_content))
            if verbose:
                from pprint import pprint
                print("Generating response using ..")
                pprint([{k: v if k == "role" else truncate_long_text(v)} for i in messages for k,v in i.items()])
            else:
                print("Generating response ... ")
            if self.alpha_one:
                from alpha_one_mlx.reasoner import alpha_one
                from alpha_one_mlx.models import get_configuration

                configuration = get_configuration(model.model_type)
                alpha = self.alpha_one.get("alpha", 1.4)
                threshold = int(max_tokens - alpha * self.alpha_one["thinking_token_length"])
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                wait_words = self.alpha_one.get("wait_words", configuration.slow_thinking_stop_words)
                if verbose:
                    print(f"Using parameters: {self.parameters} and {self.alpha_one} for {self.model}")
                draft_model = self.draft_model
                if draft_model:
                    from mlx_lm.utils import load
                    draft_model, draft_tokenizer = load(self.draft_model)
                    if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                        raise ValueError("Draft model tokenizer does not match model tokenizer.")
                    elif verbose:
                        print(f"Using draft model: {self.draft_model}")
                response = alpha_one(model, tokenizer, prompt,
                                     configuration=configuration,
                                     max_tokens_per_call=self.parameters.get("max_tokens", max_tokens),
                                     threshold=threshold,
                                     temperature=self.parameters.get("temperature", temperature),
                                     top_p=self.parameters.get("top_p", top_p),
                                     min_p=self.parameters.get("min_p", min_p),
                                     top_k=self.parameters.get("top_k", top_k),
                                     apply_chat_template=False,
                                     verbose=verbose,
                                     wait_words=wait_words,
                                     prompt_cache=self.program.cache,
                                     draft_model=draft_model)

            else:
                response, prompt = self.generate(messages, tokenizer, model, verbose=verbose)
            if verbose:
                print(f"Executing model: {self.model} using context {context} - (via mlx)-> >\n{response}")
            self._handle_execution_contribution(response, context)
            if "context" not in self.contribute:
                if verbose:
                    print("Clearing context ...")
                context["_"] = []

        @staticmethod
        def dispatch_check(item: Dict, program: PDLProgram):
            if "model" in item:
                return MLXModelEvaluation(item, program)


    class MLXAPSModel(MLXModelEvaluationBase):
        MODEL_KEY = "APSModel"
        def execute(self, context, return_content=False, verbose=False):
            model, tokenizer = self._get_model_cache_and_tokenizer()
            msg = context["_"][-1].copy()
            if verbose:
                print("Extracting individual facts, statements, and ideas from using ",
                      truncate_long_text(msg["content"]))
            else:
                print("Generating response ... ")
            msg["content"] = create_propositions_input(msg["content"])
            msg["role"] = "user"
            response, prompt = self.generate([msg], tokenizer, model, verbose=verbose)
            response = process_propositions_output(response)
            self._handle_execution_contribution(response, context)
            if "context" not in self.contribute:
                if verbose:
                    print("Clearing context ...")
                context["_"] = []

        @staticmethod
        def dispatch_check(item: Dict, program: PDLProgram):
            if "APSModel" in item:
                return MLXAPSModel(item, program)

    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, WorldLoomRead, ToolioCompletion, PDLRepeat, PDLText,
                                            CachePrep, MLXModelEvaluation, MLXAPSModel, PDFRead]
    with open(pdl_file, "r") as file:
        program_yaml = yaml.safe_load(file)
        ctx = dict(variables) if variables else {}
        program = PDLProgram(program_yaml, dispatcher=dispatcher, initial_context=ctx)
        model_cache = {}
        cache_lookup = {}
        if program.cache:
            if not isinstance(program.cache, str):
                #Indicate any wordloom marker for use in cache boundaries, etc.
                final_text_item = program.cache[0]['text'][-1]
                prefix_marker = final_text_item['prefix_marker']

                name = final_text_item['name']
                file_name = final_text_item['file']
                model_cache[name] = file_name

                # Tautological substitution (since generative_redfoot's wordloom extension attempts a substitution
                ctx.update({prefix_marker: f'{{{prefix_marker}}}'} if prefix_marker else {})

                #Map the cache name to the cache file
                cache_lookup[name] = file_name

                #instanciates CachePrep to create the cache via the PDL program
                PDLText(program.cache[0], program).execute(ctx, verbose=verbose)

        program.evaluation_environment = ctx
        program.cache_lookup = cache_lookup

        server_config = program_yaml.get("server")
        if server_config:
            #Setup the FastAPI service using configuration from the PDL program
            host = server_config["host"]
            port = int(server_config["port"])
            request_body_marker = server_config["request_body_marker"]
            path = server_config["path"]
            expected_content_type = None
            if "content_type" in server_config:
                expected_content_type = server_config["content_type"]
            log_level = server_config.get("log_level", logging.DEBUG).upper()
            log_path = server_config.get("log_path", "/tmp/server.log")

            app = FastAPI()

            logger = logging.getLogger(f"{__name__}.generative_redfoot_service")
            logger.setLevel(log_level)

            if logger.handlers:
                logger.handlers.clear()

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Add file handler if log_file is provided
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if variables:
                logger.debug(f"Using variables: {variables} as initial_context")

            @app.post(path)
            async def process_request(request: Request) -> Response:
                """
                Handle requests to the FastAPI generative redfoot service.

                This endpoint:
                1. Validates the content type matches the expected type from PDL configuration
                2. Extracts the request body content
                3. Creates a new PDL program instance with the request content mapped to a marker
                4. Executes the PDL program in a separate thread to avoid blocking
                5. Returns the final assistant response as plain text

                The request body is mapped to the specified request_body_marker defined in the
                PDL server configuration, allowing the PDL program to reference the incoming
                request content using template substitution.

                Args:
                    request (Request): The incoming FastAPI request object with:
                        - Headers containing content-type matching the PDL configuration
                        - Body containing the text to process

                Returns:
                    Response: Plain text response containing the final assistant message
                        from the executed PDL program

                Raises:
                    HTTPException:
                        - 415: If content type doesn't match expected type
                        - 500: If there's an error executing the PDL program
                """

                start_time = time.time()
                request_id = id(request)  # Unique identifier for this request

                content_type = request.headers.get("content-type", "")
                form_data = {}
                logger.info(f"Request {request_id}: Received request with content type {content_type}")
                if "multipart/form-data" in content_type.lower():
                    # Handle multipart form data
                    form = await request.form()
                    # Process form data, handling UploadFile objects properly
                    form_data = {}
                    for key, value in form.items():
                        if hasattr(value, 'file') and hasattr(value, 'filename'):
                            # This is an UploadFile object, read its content
                            try:
                                # Read the file content as bytes
                                file_content = await value.read()
                                # Reset file pointer to beginning for potential re-reading
                                if hasattr(value, '_file'):
                                    value._file.seek(0)
                                # For file uploads that will be processed by PDFRead, keep as bytes
                                # Check if this might be a PDF file based on extension or content-type
                                if hasattr(value, 'content_type') and 'pdf' in (value.content_type or '').lower():
                                    # Keep PDF content as bytes for proper handling by PDFRead
                                    form_data[key] = file_content
                                elif isinstance(file_content, bytes):
                                    try:
                                        # For non-PDF binary files, try to decode as text if possible
                                        form_data[key] = file_content.decode('utf-8')
                                    except UnicodeDecodeError:
                                        # For binary files, we can store as bytes which can be handled by the PDL program
                                        form_data[key] = file_content
                                else:
                                    form_data[key] = file_content
                            except Exception as e:
                                logger.warning(f"Request {request_id}: Error reading file upload {key}: {str(e)}")
                                form_data[key] = str(value)  # fallback to string representation
                        else:
                            # Regular form field, convert to string
                            form_data[key] = str(value)
                    body_content = form_data.get(request_body_marker, "")
                elif "application/x-www-form-urlencoded" in content_type.lower():
                    # Handle URL-encoded form data
                    form = await request.form()
                    # For URL-encoded data, all values should be strings
                    form_data = {key: str(value) for key, value in form.items()}
                    body_content = form_data.get(request_body_marker, "")
                else:
                    if expected_content_type is not None and expected_content_type.lower() not in content_type.lower():
                        raise HTTPException(status_code=415,
                                            detail=f"Unsupported Media Type. Expected {expected_content_type}.")
                    body_content = await request.body()
                    # Ensure body_content is a string for PDL template substitution
                    if isinstance(body_content, bytes):
                        try:
                            body_content = body_content.decode('utf-8')
                        except UnicodeDecodeError:
                            # Try with error handling - replace invalid characters
                            body_content = body_content
                            # body_content = body_content.decode('utf-8', errors='replace')
                            logger.warning(f"Request {request_id}: Invalid UTF-8 characters found in body, replaced "
                                           f"with replacement character")
                    else:
                        body_content = str(body_content)

                # Add form data to context if available
                if form_data:
                    initial_context = {**form_data, request_body_marker: body_content}
                    form_data_short_hand = {k:v for k,v in form_data.items() if isinstance(v, str)}
                    logger.debug(f"Request {request_id}: Form data: {form_data_short_hand}")
                else:
                    initial_context = {request_body_marker: body_content}

                logger.info(
                    f"Request {request_id}: Started processing document of length {len(body_content):,} characters")

                # Load and execute the PDL program with the document content
                try:
                    request_program = PDLProgram(program_yaml, dispatcher=dispatcher, initial_context=initial_context)
                    request_program.cache_lookup = cache_lookup
                    logger.debug(
                        f"Request {request_id}: Initial context variable names: {list(initial_context.keys())}")
                    logger.debug(f"Request {request_id}: Named K/V caches (& corresponding cache file): {cache_lookup}")
                    logger.debug(
                        f"Request {request_id}: Document length: {len(initial_context.get(request_body_marker, ''))}")

                    # Run the PDL program execution in a separate thread to avoid blocking
                    # and to properly handle any async operations within the program
                    import asyncio
                    from concurrent.futures import ThreadPoolExecutor

                    def run_program():
                        request_program.execute(verbose=verbose)
                        return request_program
                    # Run in a separate thread to isolate the execution context
                    with ThreadPoolExecutor() as executor:
                        request_program = await asyncio.get_event_loop().run_in_executor(executor, run_program)
                except Exception as e:
                    logger.error(f"Request {request_id}: Error loading or executing PDL program: {str(e)}")
                    logger.error(f"Request {request_id}: Body content: {body_content[:200]}...")  # Log first 200 chars
                    logger.exception(
                        f"Request {request_id}: Full traceback for PDL program error:")  # This will log the stack trace
                    raise HTTPException(status_code=500, detail=f"Error loading or executing PDL program: {str(e)}")
                elapsed_time = time.time() - start_time
                elapsed_minutes = int(elapsed_time // 60)
                logger.info(f"Request {request_id}: Completed in {elapsed_minutes} minutes")

                final_msg = initial_context['_'][-1]
                assert final_msg['role'] == 'assistant'
                return Response(final_msg['content'], media_type="text/plain")

            logger.info(f"Starting generative redfoot service with config: {pdl_file}")
            # logger.info(f"Logging level set to: {args.log_level}")
            logger.info(f"Service will be available at: http://{host}:{port}{path}")
            logger.info("Endpoint: POST /metacoder (accepts text/plain content)")

            # Start the FastAPI server
            uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
            return 0
        else:
            program.execute(verbose=verbose)
            if verbose:
                print(program.evaluation_environment)
if __name__ == '__main__':
    main()
