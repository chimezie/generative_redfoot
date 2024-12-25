"""
Declarative Prompts, Generative program generators, oh my...

Takes a minimal Prompt Declaration Language (PDL) file and generates a finite state generative machine
as Python objects for a subset of the PDL language [1], [2].  These objects (the "programs" in particular)
can be executed, and their Model class can be extended to incorporate the functionality for
evaluating the prompts against the models specified in PDL using any accumulated conversational
context, prompts, and generation parameters (sampling parameters, for example), (optionally) updating
the context as the programs execution continues.  A generative, conversational workflow system

The model evaluation is implemented using mlx

A callback to Redfoot [1] by James Tauber and Dan Krech, way back in the dawn of the Semantic Web agent framework days.
This current AI dev ops wave could lean alot from that previous one, circa dawn of this millennium

[1] https://jtauber.com/redfoot/
[2] https://github.com/IBM/prompt-declaration-language
[3] https://arxiv.org/pdf/2410.19135

"""

import click
import yaml
import json
import re
from abc import ABC

UNTIL_PATTERN = re.compile(r"^\$\{\s*(?P<variable>[^\s]+)\s+==\s*'(?P<value>[^']+)'\s*\}")

PDL ="""
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
  discard: true
  input:
    text:
    - read: some_file.txt
    - read: other_file.txt
    - |
    
      Long text."""

class TextCollator(ABC):
    def __init__(self, content, program):
        self.program = program
        self.content = list(map(lambda i: self.program.dispatcher.handle(i, self.program), content))

class PDLText(TextCollator):
    """
    PDL blcck for creating data, content can be any kind of block

    Example:
        >>> program = PDLProgram({"description": "chatbot", "text": [""]}, dispatcher=ParseDispatcher())
        >>> text_block = [{"read": None, "message": "What is your query?", "contribute": ["context"]}]
        >>> pdl_text = PDLText(text_block, program)
        >>> print(pdl_text)
        PDLText([PDLRead('What is your query?' -> ['context'])])
    """

    def __init__(self, content, program):
        super().__init__(content, program)
        self.program = program

    def __repr__(self):
        return f"PDLText({self.content})"

    def __getitem__(self, index):
        return self.content[index]

    def __len__(self):
        return len(self.content)

    def execute(self, context, return_content=False):
        return "\n".join(map(lambda i: "" if i is None else i,
                             map(lambda i: i if isinstance(i,
                                                           str) else i.execute(context, return_content),
                                 self.content)))

class PDLProgram:
    """
    A PDL "program"

    Example:
        >>> program = PDLProgram(yaml.safe_load(PDL))
        >>> print(program.text.content[0])
        PDLRead('What is your query? ' -> ['context'])
        >>> print(len(program.text.content))
        2
        >>> print(type(program.text[1]))
        <class 'generative_redfoot.PDLRepeat'>
        >>> print(type(program.text[1].body))
        <class 'generative_redfoot.PDLText'>

        >>> print(len(program.text[1].body))
        2
        >>> print(program.text[1][0])
        PDLModel(google/gemma-2-9b-it)
        >>> print(program.text[1][1])
        PDLVariableAssign(PDLRead('Enter a query or say 'quit' to exit' -> ['context']) -> $question)
        >>> print(program.text[1].until)
        ${ question == 'quit' }


        >>> program = PDLProgram(yaml.safe_load(PDL2))
        >>> len(program.text.content)
        1

        >>> type(program.text.content[0])
        <class 'generative_redfoot.PDLModel'>

        >>> program.text.content[0].input[0]
        PDLRead( from 'some_file.txt')

        >>> program.text.content[0].input[1]
        PDLRead( from 'other_file.txt')

        >>> print(program.text.content[0].input[-1])
        <BLANKLINE>
        Long text.

        >>> program.text[0].parameters
        {'temperature': 0.6, 'min_p': 0.03, 'max_tokens': 600}

        >>> program.execute()
        Executing: program
        Executing model: google/gemma-2-9b-it using {} and discarding the result
    """
    def __init__(self, pdl, dispatcher = None):
        if dispatcher is None:
            self.dispatcher = ParseDispatcher()
        else:
            self.dispatcher = dispatcher
        self.description = pdl["description"]
        self.text = PDLText(pdl["text"], self)
        self.evaluation_environment = {}

    def __repr__(self):
        program_state = self.evaluation_environment if self.evaluation_environment else 'unexecuted'
        return f"PDLProgram('{self.description}'\n\t{self.text}\n\t{program_state})"

    def execute(self):
        print(f"Executing: {self.description}")
        self.text.execute(self.evaluation_environment)

class PDLRead:
    def __init__(self, pdl_block, program):
        self.program = program
        self.read_from = pdl_block["read"].strip() if pdl_block["read"] else None
        if not self.read_from:
            self.message = pdl_block["message"]
            self.contribute = pdl_block["contribute"]
            self.read_from = pdl_block["read"]
            assert self.contribute == ['context']
        else:
            self.message = None
            self.contribute = None

    def __repr__(self):
        return f"PDLRead('{self.message}' -> {self.contribute})" if self.message else f"PDLRead( from '{self.read_from}')"

    def execute(self, context, return_content=False):
        if self.read_from:
            with open(self.read_from, "r") as file:
                content = file.read()
        else:
            content = input(self.message + " ")
        context.setdefault('_', []).append({"role": "user", "content": content})
        if return_content:
            return content

class PDLRepeat:
    def __init__(self, content, program):
        self.program = program
        self.body = self.program.dispatcher.handle(content["repeat"], self.program)
        self.until = content["until"]

    def __repr__(self):
        return f"PDLRepeat({self.body} UNTIL {self.until})"

    def __getitem__(self, index):
        return self.body[index]

    def execute(self, context):
        self.body.execute(context)
        until_pattern_groups = UNTIL_PATTERN.match(self.until)
        variable_name = until_pattern_groups.group('variable')
        value = until_pattern_groups.group('value')
        while value != context[variable_name]:
            self.body.execute(context)

class PDLModel:
    """
    Meant to be extended and for its execute method to be overridden for LLM evaluation
    """
    def __init__(self, content, program):
        self.program = program
        self.model = content["model"]
        self.discard = content["discard"] if "discard" in content else False
        self.content = content
        self.input = self.program.dispatcher.handle(content["input"], self.program) if "input" in content else None
        self.parameters = content["parameters"] if "parameters" in content else {}
        self.cot_prefix = content["cot_prefix"] if "cot_prefix" in content else {}

    def __repr__(self):
        return f"PDLModel({self.model})"

    def execute(self, context, return_content=False):
        print(f"Executing model: {self.model} using {context}{' and discarding the result' if self.discard else ''}")
        if not self.discard:
            context.setdefault('_', []).append({"role": "assistant", "content": ".. model response .."})
        if return_content:
            return ".. model response .."

class PDLVariableAssign:
    def __init__(self, content, program):
        self.program = program
        self.variable = content["def"]
        self.content = PDLRead(content, self.program)

    def __repr__(self):
        return f"PDLVariableAssign({self.content} -> ${self.variable})"

    def execute(self, context):
        context[self.variable] = self.content.execute(context, return_content=True)

ModelKlass = PDLModel

def key_exists(key):
    def fn(item):
        return key in item
    return fn

class ParseDispatcher:
    REGISTRY = {
        "read": PDLRead,
        "repeat": PDLRepeat,
        "text": PDLText,
        "model": ModelKlass,
        "def": PDLVariableAssign,
    }
    def handle(self, item, program):
        if isinstance(item, str):
            return item
        elif "def" in item:
            return self.REGISTRY["def"](item, program)
        elif {"read", "message", "contribute"}.intersection(item):
            return self.REGISTRY["read"](item, program)
        elif "repeat" in item:
            return self.REGISTRY["repeat"](item, program)
        elif "text" in item:
            return self.REGISTRY["text"](item["text"], program)
        elif "model" in item:
            return self.REGISTRY["model"](item, program)
        else:
            raise RuntimeError(f"Unknown block type: {item}")

@click.command()
@click.option('-t', '--temperature', default=1, type=float)
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--top_k', default=-1, type=int, help='Sampling top_k')
@click.option('--min-p', default=-1, type=float, help='Sampling min-p')
@click.option('--verbose/--no-verbose', default=False)
@click.argument('pdl_file')
def main(temperature, repetition_penalty, top_k, min_p, verbose, pdl_file):
    from mlx_lm.utils import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    class MLXModelEvaluation(PDLModel):
        def execute(self, context, return_content=False):
            # @TODO: incorporate tokenizer configuration into PDL
            model, tokenizer = load(self.model)
            messages = []
            if self.cot_prefix:
                print("### Adding Chain-of Thought Few Shot examples specified in PDL ###")
                with open(self.cot_prefix, 'r') as cot_content:
                    messages = json.load(cot_content)

            if self.input:
                messages.append({"role": "user", "content": self.input.execute(context, return_content=True)})
            else:
                messages.extend(context["_"])
            if verbose:
                print(f"Generating response using ", messages, self.program)
            else:
                print(f"Generating response ... ")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            response = generate(model, tokenizer, prompt, max_tokens=self.parameters.get("max_tokens", 600),
                                sampler=make_sampler(temp=self.parameters.get("temperature", 0.6),
                                                     min_p=self.parameters.get("min_p", 0.03),
                                                     top_k=self.parameters.get("top_k", 14)),
                                logits_processors=make_logits_processors(repetition_penalty=1.25),
                                verbose=verbose)
            if not verbose:
                print(response)
            if verbose:
                print(f"Executing model: {self.model} using context {context} - (via mlx)-> >\n{response}")
            if not self.discard:
                context.setdefault('_', []).extend([{"role": "user", "content": prompt},
                                                    {"role": "assistant", "content": response}])
            elif verbose:
                print("Discarding model response from subsequent executions")
    dispatcher = ParseDispatcher()
    dispatcher.REGISTRY["model"] = MLXModelEvaluation
    with open(pdl_file, "r") as file:
        program = PDLProgram(yaml.safe_load(file), dispatcher=dispatcher)
        program.execute()
        if verbose:
            print(program.evaluation_environment)
if __name__ == '__main__':
    main()
