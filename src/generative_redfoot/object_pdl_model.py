"""
Object PDL Model

Takes a minimal Prompt Declaration Language (PDL) file and generates a finite state generative machine
as Python objects for a subset of the PDL language [1], [2].  These objects (the "programs" in particular)
can be executed, and their Model class can be extended to incorporate the functionality for
evaluating the prompts against the models specified in PDL using any accumulated conversational
context, prompts, and generation parameters (sampling parameters, for example), (optionally) updating
the context as the programs execution continues.

The model evaluation is implemented using mlx

A callback to Redfoot [1] by James Tauber and Dan Krech, way back in the dawn of the Semantic Web agent framework days.
This current AI dev ops wave could lean alot from that previous one, circa dawn of this millennium

[1] https://jtauber.com/redfoot/
[2] https://github.com/IBM/prompt-declaration-language
[3] https://arxiv.org/pdf/2410.19135

"""

import yaml
import re
from abc import ABC
from typing import Mapping, Dict, Any, Optional, Union, List
from pprint import pprint

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
    Optional keywords for block: "description", "def", "role", and "contribute" currently

    - description: is a special comment
    - def: assigns the result of the block to a variable
    - defs: creates multiple variable definitions, each with its own name 𝑥 and a value given
      by a nested PDL program:
    - role: causes the data resulting from a block to be decorated with a role, such as ‘user’,
      ‘assistant’, or ‘system’. If a block does not have an explicit role: , it defaults to
      ‘assistant’ for model blocks and to ‘user’ for all other blocks. Inner nested blocks have
      the same role as their outer enclosing block.
    - contribute: [specifies] a (possibly empty) subset of the two destinations ‘result’ or
      ‘context’. By default, every block contributes to both its own result and the background
      context for later LLM calls. [..] [can limit] the contribution of a block to just the
      context to declutter the output.
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
                pprint(content)
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
        for item in self.content:
            if isinstance(item, str):
                content +=  item
            else:
                result = item.execute(context, verbose=verbose)
                if result is not None:
                    content += result
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

class WorldLoomRead(PDLObject, PDLStructuredBlock):
    """
    PDL block for reading sections for a prompt from a Worldloom (TOML / YAML) file using ogbujipt.word_loom

    Example:
        >>> p = PDLProgram(yaml.safe_load(PDL3))
        >>> p.cache
        'prompt_cache.safetensors'
        >>> p.text[0]
        Wordloom('question answer' from file.loom [outputs to context as user])

    """

    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.loom_file = pdl_block["read_from_wordloom"]
        self.language_items = pdl_block["items"]
        self._get_common_attributes(pdl_block)

    def __repr__(self):
        return f"Wordloom('{self.language_items}' from {self.loom_file} [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        from ogbujipt import word_loom
        with open(self.loom_file, mode='rb') as fp:
            loom = word_loom.load(fp)
        items = self.language_items.split(' ')
        if verbose:
            print(f"Expanding {items} from {self.loom_file}")
        content = '\n'.join([WorldLoomRead.get_loom_entry(loom[name], context) for name in items])
        self._handle_execution_contribution(content, context)

    @staticmethod
    def get_loom_entry(loom_entry:word_loom.language_item, context: Mapping) -> Union[str, word_loom]:
        """
        Processes a language_item by formatting it with context-specific marker substitutions
        if markers are present in the language_item. If no markers are available, the original
        language_item is returned as is.

        :param loom_entry: A wordloom `language_item` object that contains potential markers to be
                           substituted and formatted with values from the context.
        :param context: A dictionary-like object (`Mapping`) that holds marker-to-value
                        mappings to be used for substitutions in the given loom_entry.
        :return: Returns a formatted string if markers are found and substitutions can be
                 applied; otherwise, returns the unprocessed `language_item` as is.

        """
        if loom_entry.markers:
            marker_kwargs = {}
            for marker in loom_entry.markers:
                marker_kwargs[marker] = context[marker]
            return loom_entry.format(**marker_kwargs)
        else:
            return loom_entry

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "read_from_wordloom" in item:
            return WorldLoomRead(item, program)

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
    Meant to be extended and for its execute method to be overridden for LLM evaluation
    """
    default_role = "assistant"
    MODEL_KEY = "model"
    def __init__(self, content: Mapping, program):
        self.program = program
        self.model = content[self.MODEL_KEY]
        self.content = content
        self.input = self.program.dispatcher.handle(content["input"], self.program) if "input" in content else None
        self.parameters = content["parameters"] if "parameters" in content else {}
        self.cot_prefix = content["cot_prefix"] if "cot_prefix" in content else {}
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

class PDFRead(PDLObject, PDLStructuredBlock):
    """
    Class that handles PDF reading as part of the execution of a PDL program

    Parses and extracts from PDF and contributes to the larger operational workflow
     of the execution of a PDL program, leveraging PyPDF2 for the extraction.

    """
    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.read_from = pdl_block["PDF_read"]
        self._get_common_attributes(pdl_block)
        if not self.read_from:
            self.message = pdl_block["message"]
            self.read_from = pdl_block["read"]
        else:
            self.message = None

    def __repr__(self):
        return f"PDFRead( from '{self.read_from}' [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        from PyPDF2 import PdfReader
        if self.read_from and isinstance(self.read_from, dict):
            var_reference_group = VAR_REFERENCE_PATTERN.match(list(self.read_from)[0])
            if var_reference_group:
                variable_name = var_reference_group.group('variable')
                file_name = context[variable_name]
            else:
                file_name = self.read_from
        else:
            file_name = self.read_from
        if verbose:
            print(f"Reading {file_name} from context")
        content = ''.join((page.extract_text() for page in PdfReader(file_name).pages))
        self._handle_execution_contribution(content, context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "PDF_read" in item:
            return PDFRead(item, program)

class ParseDispatcher:
    DISPATCH_RESOLUTION_ORDER = [PDLRead, WorldLoomRead, PDLRepeat, PDLText, PDLModel]

    def handle(self, item: Mapping, program: PDLObject) -> PDLObject:
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
    A PDL "program"

    A block or a list of blocks where blocks are expressions or structured blocks

    Example:
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

        >>> program.execute(verbose=True)
        Executing: program
        .. model response ..

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

