'''
Object PDL Model

Generative Redfoot: A generative, conversational workflow and multi-agent system using PDL and MLX.
This module implements the core object model for the PDL (Prompt Declaration Language) system that
enables declarative, composable AI workflows following the design patterns documented in the project.

The system takes a Prompt Declaration Language (PDL) file and generates a finite state generative machine
as Python objects for a subset of the PDL language. These objects (the "programs" in particular) can
be executed as web services or locally, implementing the core concepts of contextual state management,
declarative composition, and extension-based architecture. The objects maintain conversational context
through an accumulated context that can be updated and shared across PDL blocks, enabling complex
multi-step workflows where each step builds on previous results.

The model evaluation is implemented using mlx for efficient inference, supporting the Advanced Caching
and Optimization design pattern with both internal and external caching mechanisms.

This system recasts the original Redfoot concept with firm roots in declarative and RESTful principles
in how generative AI actions and capabilities can be composed and orchestrated.

[1] https://jtauber.com/redfoot/
[2] https://github.com/IBM/prompt-declaration-language
[3] https://arxiv.org/pdf/2410.19135

'''

import logging
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from generative_redfoot.expressions import evaluate, resolve_variable_reference
from generative_redfoot.pdf_text import extract_pdf_text


def pretty_print_list(my_list, sep=', ', and_char=', & '):
    return (
        and_char.join([sep.join(my_list[:-1]), my_list[-1]])
        if len(my_list) > 2
        else f'{my_list[0]} and {my_list[1]}'
        if len(my_list) == 2
        else my_list[0]
    )


UNTIL_PATTERN = re.compile(r"^\$\{\s*(?P<variable>[^\s]+)\s+==\s*'(?P<value>[^']+)'\s*\}")
DEFAULT_READ_MESSAGE = 'How can I help you?: '

PDL = '''
description: chatbot
text:
- read:
  message: "What is your query?\n"
  contribute: [context]
- repeat:
    text:
    - model: google/gemma-2-9b-it
    - def: question
      read:
      contribute: [context]
      message: "Enter a query or say 'quit' to exit"
  until: "${ question == 'quit' }"
'''

PDL2 = '''
description: program
text:
- model: google/gemma-2-9b-it
  parameters:
    temperature: 0.6
    min_p: .03
    max_tokens: 600
  input:
    contribute: [context]
    text:
    - read: some_file.txt
      contribute: [context]
    - read: other_file.txt
      contribute: [context]
    - |

      Long text.'''


PDL3 = '''
description: program
cache: prompt_cache.safetensors
text:
- read_from_wordloom: file.loom
  items: "question answer"
  contribute: [context]
'''

PDL4 = '''
description: program
text:
- role: "system"
  text: "Do stuff well"
'''

PDL5 = '''
description: program
defs:
  file: /tmp/file.txt
text:
  - role: system
    text: foo
    contribute: [context]
  - text: bar
'''


class PDLObject(ABC):
    '''
    Abstract base class that implements the foundation for the Contextual State Management design pattern.
    All PDL objects inherit from this class and participate in maintaining and updating the conversational
    context that enables complex multi-step workflows where each step builds on previous results.

    The context is accumulated and shared across PDL blocks, supporting the core concept of generative AI
    workflows as described in the project documentation.
    '''

    @abstractmethod
    def execute(self, context: dict, verbose: bool = False) -> Any:
        '''Execute the PDL block as part of the declarative composition pattern.

        Args:
            context: The execution context dictionary that maintains conversational state
                     across PDL blocks, implementing the Contextual State Management pattern
            verbose: Whether to print verbose execution information

        Returns:
            Any: The result of executing this PDL block
        '''
        pass


class TextCollator(PDLObject):
    def __init__(self, content: Any, program):
        self.program = program
        self.content = (
            content
            if isinstance(content, str)
            else list(map(lambda i: self.program.dispatcher.handle(i, self.program), content))
        )


class PDLStructuredBlock:
    '''
    Implements the structured block pattern that supports the declarative composition design pattern.
    These blocks enable flexible, composable AI workflows by allowing different types of content and
    behavior to be specified declaratively in PDL files.

    Optional keywords for block: "description", "def", "role", and "contribute" currently

    - description: is a special comment for documentation
    - def: assigns the result of the block to a variable, supporting variable references and protocol binding
    - defs: creates multiple variable definitions, each with its own name 𝑥 and a value given
      by a nested PDL program, supporting the Contextual State Management design pattern:
    - role: causes the data resulting from a block to be decorated with a role, such as ‘user’,
      ‘assistant’, or ‘system’. If a block does not have an explicit role: , it defaults to
      ‘assistant’ for model blocks and to ‘user’ for all other blocks. Inner nested blocks have
      the same role as their outer enclosing block, maintaining conversational context.
    - contribute: [specifies] a (possibly empty) subset of the two destinations ‘result’ or
      ‘context’. By default, every block contributes to both its own result and the background
      context for later LLM calls. [..] [can limit] the contribution of a block to just the
      context to declutter the output, supporting the Contextual State Management design pattern.

    This structure enables the core concept of maintaining conversational context through accumulated
    context that can be updated and shared across PDL blocks, as described in the project documentation.
    '''

    default_contribution = ['context', 'result']
    default_role = 'user'

    def _get_common_attributes(self, item: Mapping):
        self.contribute = item.get('contribute', self.default_contribution)
        self.role = item.get('role', self.default_role)
        self.var_def = item.get('def')
        self.description = item.get('description')
        self.defs = {k: v for k, v in item.get('defs', {}).items()}

    def _handle_execution_contribution(self, content: list | str, context: dict):
        if content:
            msg = {'role': self.role, 'content': content}
            if 'result' in self.contribute:
                print(content)
            if 'context' in self.contribute:
                context.setdefault('_', []).append(msg)
        if self.var_def:  # `def` binds the result even when it is empty
            logging.debug(f'Defining variable {self.var_def} with value {content}')
            context[self.var_def] = content

    def _execute_defs(self, context: dict, verbose: bool = False):
        '''
        Evaluate this block's `defs` (in order) before its own content runs. A value that is a block is executed
        with `contribute: []` against a copy of the context, so only its result is bound here.
        '''
        for name, value in self.defs.items():
            block = None
            if isinstance(value, Mapping):
                try:
                    block = self.program.dispatcher.handle({**value, 'contribute': []}, self.program)
                except RuntimeError:  # not a block; treat it as a plain (evaluated) value below
                    block = None
            if block is None:
                context[name] = evaluate(value, context)
                continue
            sandbox = dict(context)
            if '_' in context:
                sandbox['_'] = [dict(message) for message in context['_']]  # no message leaks back out
            context[name] = block.execute(sandbox, verbose=verbose)

    def descriptive_text(self):
        var_suffix = f' -> ${self.var_def}' if self.var_def else ''
        if self.defs:
            var_suffix = var_suffix + f' (defining: {self.defs})'
        output_destination = ', '.join(self.contribute) if self.contribute else 'N/A'
        return f'outputs to {output_destination} as {self.role}{var_suffix}'


class PDLText(TextCollator, PDLStructuredBlock):
    '''
    PDL blcck for creating data

    Example:
        >>> import yaml
        >>> program = PDLProgram(yaml.safe_load(PDL))
        >>> program.text.role
        'user'

        >>> p = PDLProgram(yaml.safe_load(PDL4))
        >>> p.text.role
        'user'
        >>> p.text[0].role
        'system'


        >>> p = PDLProgram(yaml.safe_load(PDL5))
        >>> p.descriptive_text()
        "outputs to context, result as user (defining: {'file': '/tmp/file.txt'})"
        >>> len(p.text)
        2

        >>> p.text[0]
        PDLText('foo' [outputs to context as system])

        >>> p.text[1]
        PDLText('bar' [outputs to context, result as user])

        >>> p.text[0].role
        'system'

        >>> p.text[0].contribute
        ['context']

        >>> p.text[1].contribute
        ['context', 'result']

        >>> p.execute(verbose=True)
        Executing: program
        bar
        'bar'
        >>> p.evaluation_environment
        {'_': [{'role': 'system', 'content': 'foo'}, {'role': 'user', 'content': 'bar'}]}

    '''

    def __init__(self, content: Any, program: PDLObject):
        super().__init__(content['text'], program)
        self._get_common_attributes(content)
        self.program = program

    def __repr__(self):
        content_str = f"'{self.content}'" if isinstance(self.content, str) else f'{self.content}'
        return f'PDLText({content_str} [{self.descriptive_text()}])'

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index: int):
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def execute(self, context: dict, verbose: bool = False):
        '''Execute the text block and return the joined result of its items'''
        self._execute_defs(context, verbose)
        parts = []
        if isinstance(self.content, str):
            text = evaluate(self.content, context)
            self._contribute_string(text, context)
            parts.append(text)
        else:
            for item in self.content:
                if isinstance(item, str):
                    text = evaluate(item, context)
                    self._contribute_string(text, context)
                    parts.append(text)
                else:
                    result = item.execute(context, verbose=verbose)  # the child contributes itself to the context
                    if 'result' in getattr(item, 'contribute', []) and result is not None:
                        parts.append(result)  # the return value is only for this block's result
            self._merge_adjacent_same_role(context)
        result = ''.join(part if isinstance(part, str) else str(part) for part in parts)
        if self.var_def:
            context[self.var_def] = result
        return result

    def _contribute_string(self, text: str, context: dict):
        '''A string item prints / joins the context according to the `contribute` of its containing text block'''
        if 'result' in self.contribute:
            print(text)
        if 'context' in self.contribute:
            self.merge_content(context, text)

    @staticmethod
    def _merge_adjacent_same_role(context: dict):
        merged_context = []
        for message in context.get('_', []):
            if merged_context and merged_context[-1]['role'] == message['role']:
                merged_context[-1]['content'] += message['content']
            else:
                merged_context.append(message)
        context['_'] = merged_context

    def merge_content(self, context, item):
        messages = context.setdefault('_', [])
        if messages and [m for m in messages if m['role'] == 'user']:
            messages[-1]['content'] += item
        else:
            messages.append({'role': self.role, 'content': item})

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if 'text' in item:
            return PDLText(item, program)


class PDLRead(PDLObject, PDLStructuredBlock):
    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.read_from = pdl_block.get('read')
        self.message = pdl_block.get('message')
        self.multiline = pdl_block.get('multiline', False)
        self._get_common_attributes(pdl_block)

    def __repr__(self):
        if self.message:
            return f"PDLRead('{self.message}' [{self.descriptive_text()}])"
        else:
            return f"PDLRead( from '{self.read_from}' [{self.descriptive_text()}])"

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        resolved = resolve_variable_reference(self.read_from, context)
        if resolved is not self.read_from:  # legacy `{ $name }`: the path is used exactly as it is given
            file_name = resolved
            if verbose:
                logging.debug(f'Reading {file_name} from context')
            with open(file_name) as file:
                content = file.read()
        else:
            read_from = evaluate(self.read_from, context)
            if read_from:
                content = self._read_file(read_from)
            else:
                message = evaluate(self.message, context) if self.message is not None else DEFAULT_READ_MESSAGE
                if self.multiline:
                    print(message)
                    content = sys.stdin.read()
                else:
                    content = input(message)
        self._handle_execution_contribution(content, context)
        return content

    def _read_file(self, read_from) -> str:
        '''Read a file named in the PDL file (relative paths are relative to the PDL file's directory)'''
        try:
            path = Path(read_from)
        except TypeError as exc:
            raise ValueError(f'read expects a file path or no value, got {read_from!r}') from exc
        if not path.is_absolute():
            path = self.program.base_dir / path
        try:
            return path.read_text(encoding='utf-8')
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: {path}') from None

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if 'read' in item:
            return PDLRead(item, program)


class PDLRepeat(PDLObject, PDLStructuredBlock):
    def __init__(self, content: Mapping, program):
        self.program = program
        self.body = self.program.dispatcher.handle(content['repeat'], self.program)
        self.until = content['until']
        self._get_common_attributes(content)

    def __repr__(self):
        return f'PDLRepeat({self.body} UNTIL {self.until} [{self.descriptive_text()}])'

    def __getitem__(self, index: int):
        return self.body[index]

    def execute(self, context: Mapping, verbose: bool = False):
        '''
        Executes the provided action repeatedly until a condition is met based on the values in the provided context.

        Continues to execute its body until the specified variable in the context matches the target value as defined
        by the `until` condition.  The evaluation of the until expression determines the stopping criteria
        '''
        self._execute_defs(context, verbose)
        results = [self.body.execute(context, verbose=verbose)]
        until_pattern_groups = UNTIL_PATTERN.match(self.until)
        variable_name = until_pattern_groups.group('variable')
        value = until_pattern_groups.group('value')
        while value != context[variable_name]:
            results.append(self.body.execute(context, verbose=verbose))
        joined = ''.join(result if isinstance(result, str) else str(result) for result in results if result is not None)
        if self.var_def:
            context[self.var_def] = joined
        return joined

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if 'repeat' in item:
            return PDLRepeat(item, program)


class PDLModel(PDLObject, PDLStructuredBlock):
    '''
    Core model class that implements the Model Enhancement Features design pattern.
    This class is designed to be extended to incorporate functionality for evaluating
    prompts against models specified in PDL using accumulated conversational context,
    supporting the Contextual State Management pattern.

    The PDLModel class supports:
    - Chain-of-thought (CoT) prefixes for few-shot learning through the cot_prefix parameter
    - Alpha One reasoning through the alpha_one parameter
    - Draft model support for speculative decoding through the draft_model parameter
    - Model parameter management for sampling strategies (temperature, top-k, top-p, min-p, max_tokens)
    - Context contribution controls to manage conversational state

    This implementation enables flexible, composable AI workflows as described in the project
    documentation and supports the extension-based architecture pattern.
    '''

    default_role = 'assistant'
    MODEL_KEY = 'model'

    def __init__(self, content: Mapping, program):
        self.program = program
        self.model = content[self.MODEL_KEY]
        self.content = content
        self.input = self.program.dispatcher.handle(content['input'], self.program) if 'input' in content else None
        self.parameters = content['parameters'] if 'parameters' in content else {}
        self.alpha_one = content['alpha_one'] if 'alpha_one' in content else {}
        if 'draft_model' in content:
            self.draft_model = content['draft_model']
        else:
            self.draft_model = None
        self.cot_prefix = content['cot_prefix'] if 'cot_prefix' in content else None
        self._get_common_attributes(content)

    def __repr__(self):
        return f'PDLModel({self.model} [{self.descriptive_text()}])'

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        source_phrase = ''
        if self.input:
            source_phrase = f' from {self.input}'
        if verbose:
            logging.debug(f'Executing model: {self.model} using {context}{source_phrase}')
        response = '.. model response ..'
        self._handle_execution_contribution(response, context)
        return response

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject) -> PDLObject | None:
        if 'model' in item:
            return PDLModel(item, program)


PDF_READ_MODES = ['PDF_raw_read_ocr', 'PDF_raw_read_txt', 'PDF_filename_ocr', 'PDF_filename_txt']


class PDFRead(PDLObject, PDLStructuredBlock):
    '''
    Class that implements the Multi-Modal Input Processing design pattern by handling
    PDF reading as part of the execution of a PDL program. This class supports the
    four distinct PDF reading modes that enable flexible document processing workflows:
    PDF_raw_read_ocr, PDF_raw_read_txt, PDF_filename_ocr, and PDF_filename_txt.

    The class provides robust PDF processing capabilities including OCR support for scanned
    documents, raw content processing, file path validation, and UploadFile integration
    for web service contexts. It parses and extracts text from PDF documents and contributes
    to the larger operational workflow of the execution of a PDL program, supporting both
    direct text extraction and OCR processing for various document types.

    This implementation exemplifies the Extension-Based Architecture design pattern,
    demonstrating how the ParseDispatcher system allows for extensions to be registered
    and resolved based on content, enabling the capabilities described in the project
    documentation's Extensions section.
    '''

    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.read_mode = next((mode for mode in PDF_READ_MODES if mode in pdl_block), None)
        self.read_from = pdl_block[self.read_mode]
        self._get_common_attributes(pdl_block)
        self.message = None

    def __repr__(self):
        return f"PDFRead('{self.read_mode}' from '{self.read_from}' [{self.descriptive_text()}])"

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        via_ocr = self.read_mode in ['PDF_filename_ocr', 'PDF_raw_read_ocr']
        source = self.resolve_references(context)
        if self.read_mode in ['PDF_filename_ocr', 'PDF_filename_txt']:
            if verbose:
                logging.debug(f'Reading PDF content ({self.read_mode}) from filename (in context or given)')
            if not source:
                raise ValueError('PDF file path is empty or None')
            try:
                file_bytes = Path(source).read_bytes()
            except FileNotFoundError:
                raise FileNotFoundError(f'PDF file does not exist: {source}') from None
        else:
            if verbose:
                logging.debug(f'Reading PDF content ({self.read_mode}) from bytes ({self.read_from}) in context')
            file_bytes = self._pdf_bytes(source)
        content = extract_pdf_text(file_bytes, via_ocr)
        self._handle_execution_contribution(content, context)
        return content

    @staticmethod
    def _pdf_bytes(raw: Any) -> bytes:
        '''Coerce whatever the context holds (upload, bytes, text, a path) into PDF bytes'''
        if isinstance(raw, UploadFile):
            return raw.file.read()
        if isinstance(raw, str):
            # A short string naming an existing file is a path (not PDF content that happens to be a str)
            if len(raw) <= 1000 and not raw.startswith('%PDF-') and Path(raw).is_file():
                return Path(raw).read_bytes()
            return raw.encode('utf-8')
        if isinstance(raw, bytes | bytearray):
            return bytes(raw)
        return str(raw).encode('utf-8')

    def resolve_references(self, context: dict) -> Any:
        content = resolve_variable_reference(self.read_from, context)
        if content is self.read_from:  # an ordinary value written in the PDL file, so expressions apply
            content = evaluate(content, context)
        return content

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if any(read_type in item for read_type in PDF_READ_MODES):
            return PDFRead(item, program)


class ParseDispatcher:
    '''
    Implements the Extension-Based Architecture design pattern by providing a dispatcher system
    that allows for extensions to be registered and resolved based on content. This class enables
    the core concept that the language of the PDL file can be extended with additional custom
    functionality as described in the project documentation.

    The ParseDispatcher resolves PDL language constructs by checking them against registered
    extension classes in a defined order, enabling the modular extension system that supports
    various PDL capabilities like file reading, PDF processing, prompt templates, and custom
    model types through the PDLModel base class.
    '''

    DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDLModel]

    def handle(self, item: Mapping, program: PDLObject) -> PDLObject:
        '''
        Handle PDL item resolution implementing the Extension-Based Architecture design pattern.

        This method resolves PDL language constructs by checking them against registered
        extension classes in the DISPATCH_RESOLUTION_ORDER, enabling the system's extensibility
        as described in the project documentation's Extensions section.
        '''
        if isinstance(item, str):
            return item
        else:
            for klass in self.DISPATCH_RESOLUTION_ORDER:
                obj = klass.dispatch_check(item, program)
                if obj is not None:
                    return obj
            raise RuntimeError(f'Unknown block type: {item}')


class PDLProgram(PDLObject, PDLStructuredBlock):
    '''
    A PDL "program" that embodies the core design patterns of Generative Redfoot for creating
    flexible, composable AI workflows. This class implements the Declarative Composition pattern
    by processing blocks or a list of blocks where blocks are expressions or structured blocks.

    The PDLProgram class enables:
    - Contextual State Management: Maintains conversational context through accumulated context
      that can be updated and shared across PDL blocks
    - Declarative Composition: Allows complex workflows to be defined in YAML without complex programming logic
    - Caching and Optimization: Supports both internal and external caching mechanisms through the cache property
    - Extension Resolution: Works with the ParseDispatcher to handle various PDL extension blocks
    - Service Orchestration: Can be executed as part of web services when server configuration is present

    The program maintains the evaluation environment (context) that accumulates conversational state
    as the program executes, supporting the core concept that complex multi-step workflows where each
    step builds on previous results.

    Example:
        >>> import yaml
        >>> program = PDLProgram(yaml.safe_load(PDL))
        >>> print(program.text.content[0])
        PDLRead('What is your query? ' [outputs to context as user])
        >>> print(len(program.text.content))
        2
        >>> print(type(program.text[1]))
        <class 'object_pdl_model.PDLRepeat'>
        >>> print(type(program.text[1].body))
        <class 'object_pdl_model.PDLText'>

        >>> print(len(program.text[1].body))
        2
        >>> print(program.text[1][0])
        PDLModel(google/gemma-2-9b-it [outputs to context, result as assistant])
        >>> print(program.text[1][1])
        PDLRead('Enter a query or say 'quit' to exit' [outputs to context as user -> $question])
        >>> print(program.text[1].until)
        ${ question == 'quit' }


        >>> program = PDLProgram(yaml.safe_load(PDL2))
        >>> len(program.text.content)
        1

        >>> type(program.text.content[0])
        <class 'object_pdl_model.PDLModel'>

        >>> program.text[0].input[0]
        PDLRead( from 'some_file.txt' [outputs to context as user])

        >>> print(program.text[0].input[-1])
        <BLANKLINE>
        Long text.

        >>> program.text[0].parameters
        {'temperature': 0.6, 'min_p': 0.03, 'max_tokens': 600}

        >>> program.execute()
        '.. model response ..'

        >>> program.evaluation_environment
        {'_': [{'role': 'assistant', 'content': '.. model response ..'}]}
    '''

    INTERNAL_CACHE_NAME = '*'

    def __init__(self, pdl: dict, dispatcher: ParseDispatcher = None, initial_context: dict = None, base_dir=None):
        if dispatcher is None:
            self.dispatcher = ParseDispatcher()
        else:
            self.dispatcher = dispatcher
        # Relative paths written in the PDL file resolve from here; the CLI passes the PDL file's directory
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.text = PDLText(pdl, self)
        self.cache = pdl.get('cache')
        self._get_common_attributes(pdl)
        self.evaluation_environment = initial_context if initial_context else {}
        # Runtime wiring, normally set by the CLI: generation backend (generation.py), sampling defaults,
        # cache name -> cache file map, and the internal KV cache state shared by model blocks (cache: '*')
        self.backend = None
        self.defaults = None
        self.cache_lookup = {}
        self.kv_state = None

    def __repr__(self):
        return "PDLProgram('[..]')"

    def execute(self, verbose: bool = False):
        if verbose:
            print(f'Executing: {self.description}')
        # The inner PDLText owns the top-level `defs` / `def`, so they take effect exactly once here
        return self.text.execute(self.evaluation_environment, verbose=verbose)
