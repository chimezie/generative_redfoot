"""
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

"""
import os
import re
import io
from abc import ABC, abstractmethod
from typing import Mapping, Dict, Any, Optional, Union, List
from pprint import pprint

from fastapi import UploadFile


def pretty_print_list(my_list, sep=", ", and_char=", & "):
    return and_char.join([sep.join(my_list[:-1]), my_list[-1]]) if len(my_list) > 2 else '{} and {}'.format(
        my_list[0], my_list[1]
    ) if len(my_list) == 2 else my_list[0]


UNTIL_PATTERN = re.compile(r"^\$\{\s*(?P<variable>[^\s]+)\s+==\s*'(?P<value>[^']+)'\s*\}")
VAR_REFERENCE_PATTERN  = re.compile(r"^\$(?P<variable>[^\s]+)\s*$")

PDL = """
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
"""

PDL2 = """
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

      Long text."""


PDL3 = """
description: program
cache: prompt_cache.safetensors
text:
- read_from_wordloom: file.loom
  items: "question answer"
  contribute: [context]
"""

PDL4 = """
description: program
text:
- role: "system"
  text: "Do stuff well"
"""

PDL5 = """
description: program
defs:
  file: /tmp/file.txt
text:
  - role: system
    text: foo
    contribute: [context]
  - text: bar
"""

class PDLObject(ABC):
    """
    Abstract base class that implements the foundation for the Contextual State Management design pattern.
    All PDL objects inherit from this class and participate in maintaining and updating the conversational 
    context that enables complex multi-step workflows where each step builds on previous results.
    
    The context is accumulated and shared across PDL blocks, supporting the core concept of generative AI 
    workflows as described in the project documentation.
    """
    @abstractmethod
    def execute(self, context: Dict, verbose: bool = False) -> Any:
        """Execute the PDL block as part of the declarative composition pattern.
    
        Args:
            context: The execution context dictionary that maintains conversational state
                     across PDL blocks, implementing the Contextual State Management pattern
            verbose: Whether to print verbose execution information
        
        Returns:
            Any: The result of executing this PDL block
        """
        pass


class TextCollator(PDLObject):
    def __init__(self, content: Any, program):
        self.program = program
        self.content = content if isinstance(content,
                                             str) else list(map(lambda i: self.program.dispatcher.handle(i,
                                                                                                         self.program),
                                                                content))

class PDLStructuredBlock:
    """
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
    """
    default_contribution = ["context", "result"]
    default_role = "user"
    def _get_common_attributes(self, item: Mapping):
        self.contribute = item.get("contribute", self.default_contribution)
        self.role = item.get("role", self.default_role)
        self.var_def = item.get("def")
        self.description = item.get("description")
        self.defs = {k:v for k,v in item.get("defs", {}).items()}

    def _handle_execution_contribution(self, content: Union[List, str], context: Dict):
        if content:
            msg = {"role": self.role, "content": content}
            if "result" in self.contribute:
                print(content)
            if "context" in self.contribute:
                context.setdefault('_', []).append(msg)

    def descriptive_text(self):
        var_suffix = f" -> ${self.var_def}" if self.var_def else ""
        if self.defs:
            var_suffix = var_suffix + f" (defining: {self.defs})"
        output_destination = (", ".join(self.contribute) if self.contribute else "N/A")
        return f"outputs to {output_destination} as {self.role}{var_suffix}"

class PDLText(TextCollator, PDLStructuredBlock):
    """
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
        'bar'
        >>> p.evaluation_environment
        {'_': [{'role': 'system', 'content': 'foo'}, {'role': 'user', 'content': 'bar'}]}

    """

    def __init__(self,
                 content: Any,
                 program: PDLObject):
        super().__init__(content["text"], program)
        self._get_common_attributes(content)
        self.program = program

    def __repr__(self):
        content_str = f"'{self.content}'" if isinstance(self.content, str) else f"{self.content}"
        return f"PDLText({content_str} [{self.descriptive_text()}])"

    def __iter__(self):
        return iter(self.content)

    def __getitem__(self, index: int):
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def execute(self, context: Dict, verbose: bool = False):
        """"""
        content = ''
        if isinstance(self.content, str):
            self.merge_content(context, self.content)
            if "result" in self.contribute:
                pprint(self.content)
        else:
            for item in self.content:
                if isinstance(item, str):
                    self.merge_content(context, item)
                    if "result" in self.contribute:
                        pprint(content)
                else:
                    result = item.execute(context, verbose=verbose)
                    if result is not None:
                        self.merge_content(context, result)
            merged_context = []
            previous_item = None
            for idx, item in enumerate(context.get("_", [])):
                if idx > 0 and item["role"] == previous_item["role"]:
                    previous_item["content"] += item["content"]
                else:
                    merged_context.append(item)
                    previous_item = item
            context["_"] = merged_context
        self._handle_execution_contribution(content, context)

    def merge_content(self, context, item):
        messages = context.setdefault('_', [])
        if messages and [m for m in messages if m['role'] == "user"]:
            messages[-1]["content"] += item
        else:
            messages.append({"role": self.role, "content": item})

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "text" in item:
            return PDLText(item, program)

class PDLRead(PDLObject, PDLStructuredBlock):
    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.read_from = pdl_block["read"] if pdl_block["read"] else None
        self._get_common_attributes(pdl_block)
        if not self.read_from:
            self.message = pdl_block["message"]
            self.read_from = pdl_block["read"]
        else:
            self.message = None

    def __repr__(self):
        if self.message:
            return f"PDLRead('{self.message}' [{self.descriptive_text()}])"
        else:
            return f"PDLRead( from '{self.read_from}' [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        if self.read_from and isinstance(self.read_from, dict):
            var_reference_group = VAR_REFERENCE_PATTERN.match(list(self.read_from)[0])
            if var_reference_group:
                variable_name = var_reference_group.group('variable')
                file_name = context[variable_name]
            else:
                file_name = self.read_from
            if verbose:
                print(f"Reading {file_name} from context")
            with open(file_name, "r") as file:
                content = file.read()
        else:
            content = input(self.message + " ")
        self._handle_execution_contribution(content, context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "read" in item:
            return PDLRead(item, program)

class PDLRepeat(PDLObject, PDLStructuredBlock):
    def __init__(self, content: Mapping, program):
        self.program = program
        self.body = self.program.dispatcher.handle(content["repeat"], self.program)
        self.until = content["until"]
        self._get_common_attributes(content)

    def __repr__(self):
        return f"PDLRepeat({self.body} UNTIL {self.until} [{self.descriptive_text()}])"

    def __getitem__(self, index: int):
        return self.body[index]

    def execute(self, context: Mapping, verbose: bool = False):
        """
        Executes the provided action repeatedly until a condition is met based on the values in the provided context.

        Continues to execute its body until the specified variable in the context matches the target value as defined
        by the `until` condition.  The evaluation of the until expression determines the stopping criteria
        """
        self.body.execute(context)
        until_pattern_groups = UNTIL_PATTERN.match(self.until)
        variable_name = until_pattern_groups.group('variable')
        value = until_pattern_groups.group('value')
        while value != context[variable_name]:
            self.body.execute(context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "repeat" in item:
            return PDLRepeat(item, program)

class PDLModel(PDLObject, PDLStructuredBlock):
    """
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
    """
    default_role = "assistant"
    MODEL_KEY = "model"
    def __init__(self, content: Mapping, program):
        self.program = program
        self.model = content[self.MODEL_KEY]
        self.content = content
        self.input = self.program.dispatcher.handle(content["input"], self.program) if "input" in content else None
        self.parameters = content["parameters"] if "parameters" in content else {}
        self.alpha_one = content["alpha_one"] if "alpha_one" in content else {}
        if "draft_model" in content:
            self.draft_model = content["draft_model"]
        else:
            self.draft_model = None
        self.cot_prefix = content["cot_prefix"] if "cot_prefix" in content else None
        self._get_common_attributes(content)

    def __repr__(self):
        return f"PDLModel({self.model} [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        source_phrase = ""
        if self.input:
            source_phrase = f" from {self.input}"
        if verbose:
            print(f"Executing model: {self.model} using {context}{source_phrase}")
        self._handle_execution_contribution(".. model response ..", context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject) -> Optional[PDLObject]:
        if "model" in item:
            return PDLModel(item, program)

PDF_READ_MODES = ["PDF_raw_read_ocr", "PDF_raw_read_txt", "PDF_filename_ocr", "PDF_filename_txt"]

class PDFRead(PDLObject, PDLStructuredBlock):
    """
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
    """
    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.read_mode = next((mode for mode in PDF_READ_MODES if mode in pdl_block), None)
        self.read_from = pdl_block[self.read_mode]
        self._get_common_attributes(pdl_block)
        self.message = None

    def __repr__(self):
        return f"PDFRead('{self.read_mode}' from '{self.read_from}' [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        try:
            import pymupdf
        except ImportError:
            raise ImportError("PDF reading requires the pymupdf package to be installed")
        
        via_ocr = self.read_mode in ["PDF_filename_ocr", "PDF_raw_read_ocr"]
        
        if self.read_mode in ["PDF_filename_ocr", "PDF_filename_txt"]:
            # Handle file path reading
            if verbose:
                print(f"Reading PDF content ({self.read_mode}) from filename (in context or given)")
            file_name = self.resolve_references(context)
            
            # Validate the file path
            if not file_name:
                raise ValueError("PDF file path is empty or None")
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"PDF file does not exist: {file_name}")
            
            # Open file with explicit filetype to avoid MuPDF error
            with pymupdf.open(file_name) as doc:
                content = self._extract_content(doc, via_ocr)
        else:
            # Handle raw content reading
            raw_content = self.resolve_references(context)
            if verbose:
                print(f"Reading PDF content ({self.read_mode}) from bytes ({self.read_from}) provided in context)")
            
            if isinstance(raw_content, UploadFile):
                raw_content = raw_content.file.read()
            elif isinstance(raw_content, str):
                # Check if it's actually a file path
                # Only treat as file path if it's not already the raw PDF content as a string  
                # This prevents confusion between file paths and actual raw PDF byte strings
                if os.path.exists(raw_content) and len(raw_content) <= 1000 and not raw_content.startswith('%PDF-'):
                    # It's a valid file path, read the file content as bytes for consistent processing
                    with open(raw_content, 'rb') as f:
                        raw_content = f.read()
                else:
                    # It's raw content or a non-existent path, convert to bytes
                    raw_content = raw_content.encode('utf-8')
            elif not isinstance(raw_content, (bytes, bytearray)):
                # Convert other types to bytes
                raw_content = str(raw_content).encode('utf-8')

            # Open from bytes stream with explicit filetype to ensure PDF format is recognized
            # The "cannot find document handler for file type: ''" error occurs when PyMuPDF can't detect the format
            try:
                # First try direct stream approach
                with pymupdf.open(stream=raw_content, filetype="pdf") as doc:
                    content = self._extract_content(doc, via_ocr)
            except Exception as e:
                error_str = str(e)
                if "no objects found" in error_str or ("FzError" in str(type(e)) and "code=7" in error_str):
                    raise ValueError("The provided PDF content is invalid or corrupted and cannot be processed") from e
                elif "cannot find document handler for file type" in error_str or ("FzError" in str(type(e)) and "code=6" in error_str):
                    # When direct bytes stream fails, try using a temporary file approach
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(raw_content)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        with pymupdf.open(tmp_file_path) as doc:
                            content = self._extract_content(doc, via_ocr)
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_file_path)
                else:
                    raise
        
        self._handle_execution_contribution(content, context)

    def _extract_content(self, doc, via_ocr: bool):
        """Helper method to extract content from a PDF document."""
        import re
        out = io.StringIO()
        for page in doc:  # iterate the document pages
            if via_ocr:
                # For OCR, we need to get the text from the textpage
                textpage = page.get_textpage_ocr()
                out.write(re.sub(r'\s+', ' ', page.get_text(textpage=textpage)))
            else:
                out.write(page.get_text())
        return out.getvalue()

    def resolve_references(self, context: dict) -> Any:
        if self.read_from and isinstance(self.read_from, dict):
            var_reference_group = VAR_REFERENCE_PATTERN.match(list(self.read_from)[0])
            if var_reference_group:
                variable_name = var_reference_group.group('variable')
                content = context[variable_name]
            else:
                content = self.read_from
        else:
            content = self.read_from
        return content

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if any(read_type in item for read_type in PDF_READ_MODES):
            return PDFRead(item, program)

class ParseDispatcher:
    """
    Implements the Extension-Based Architecture design pattern by providing a dispatcher system
    that allows for extensions to be registered and resolved based on content. This class enables
    the core concept that the language of the PDL file can be extended with additional custom 
    functionality as described in the project documentation.

    The ParseDispatcher resolves PDL language constructs by checking them against registered
    extension classes in a defined order, enabling the modular extension system that supports
    various PDL capabilities like file reading, PDF processing, prompt templates, and custom
    model types through the PDLModel base class.
    """
    DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDLModel]

    def handle(self, item: Mapping, program: PDLObject) -> PDLObject:
        """
        Handle PDL item resolution implementing the Extension-Based Architecture design pattern.

        This method resolves PDL language constructs by checking them against registered 
        extension classes in the DISPATCH_RESOLUTION_ORDER, enabling the system's extensibility
        as described in the project documentation's Extensions section.
        """
        if isinstance(item, str):
            return item
        else:
            for klass in self.DISPATCH_RESOLUTION_ORDER:
                obj = klass.dispatch_check(item, program)
                if obj is not None:
                    return obj
            raise RuntimeError(f"Unknown block type: {item}")

class PDLProgram(PDLObject, PDLStructuredBlock):
    """
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
    """
    INTERNAL_CACHE_NAME = "*"
    def __init__(self, pdl: dict, dispatcher: ParseDispatcher = None, initial_context: Dict = None):
        if dispatcher is None:
            self.dispatcher = ParseDispatcher()
        else:
            self.dispatcher = dispatcher
        self.text = PDLText(pdl, self)
        self.cache = pdl.get("cache")
        self._get_common_attributes(pdl)
        self.evaluation_environment = initial_context if initial_context else {}

    def __repr__(self):
        program_state = self.evaluation_environment if self.evaluation_environment else 'unexecuted'
        caching_info = f" [caching to {self.cache}]" if self.cache else ""
        return f"PDLProgram('{self.description}'\n\t{self.text}\n\t{program_state}{caching_info})"

    def execute(self, verbose: bool = False):
        if verbose:
            print(f"Executing: {self.description}")
        self.text.execute(self.evaluation_environment, verbose=verbose)

