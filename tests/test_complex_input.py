from typing import List

from generative_redfoot.object_pdl_model import (ParseDispatcher, PDLRead, PDLRepeat, PDLText, PDFRead, PDLProgram,
                                                 PDLModel)

import yaml
import logging

logger = logging.getLogger("generative_redfoot.testing")
logger.setLevel(logging.DEBUG)
if logger.handlers:
    logger.handlers.clear()
# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

PDL = """
description: stuff
text:
  - text:
    - PDF_filename_ocr: Test.pdf
      contribute: [context]
    - |
  
      Blazay skidaddle
    contribute: [context]
"""

PDL2 = """
description: program
text:
  - role: system
    text: foo
    contribute: [context]
  - text: bar
"""

PDL3 = """
description: chatbot
text:
- read:
  message: "What is your query?\n"
  contribute: [context]
"""

PDL4 = """
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

PDL5 = """
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


def test_mixed_role_to_result():
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDFRead]
    p = PDLProgram(yaml.safe_load(PDL2), dispatcher=dispatcher)
    p.execute()

def test_pdf_and_long_string():
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLText, PDFRead]
    p = PDLProgram(yaml.safe_load(PDL), dispatcher=dispatcher)
    p.execute()
    assert p.evaluation_environment == {'_': [{'content': 'The quick brown fox jumped over the lazy moon \nBlazay skidaddle\n',
                                               'role': 'user'}]}

def test_pdl_text_doctests_simple(monkeypatch):
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDFRead, PDLModel]
    p = PDLProgram(yaml.safe_load(PDL3), dispatcher=dispatcher)
    # Mock the input function to simulate user input
    monkeypatch.setattr('builtins.input', lambda _: "Test user input")

    # Execute the program
    p.execute()

    # Verify the structure
    assert p.text.role == 'user'
    assert isinstance(p.text[0], PDLRead)
    assert p.text[0].message.strip() == "What is your query?"

    # Verify that the evaluation environment contains the user input
    # Using '_' as the key according to the actual implementation
    assert isinstance(p.evaluation_environment["_"], list)
    assert len(p.evaluation_environment["_"]) == 1
    assert p.evaluation_environment["_"][0] == {'content': 'Test user input', 'role': 'user'}

def test_pdl_text_doctests_complex(monkeypatch):
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDFRead, PDLModel]
    p = PDLProgram(yaml.safe_load(PDL4), dispatcher=dispatcher)
    # Mock the input function to simulate user input - first the initial query then 'quit'
    input_calls = iter(["Test user query", "quit"])
    monkeypatch.setattr('builtins.input', lambda _: next(input_calls))

    # Verify the structure
    assert p.text.role == 'user'
    assert isinstance(p.text[0], PDLRead)
    assert isinstance(p.text[1], PDLRepeat)
    assert p.text[0].message.strip() == "What is your query?"

    # Check the repeat block structure
    repeat_block = p.text[1]
    assert isinstance(repeat_block.body, PDLText)
    assert len(repeat_block.body) == 2  # model and read blocks
    assert isinstance(repeat_block.body[0], PDLModel)
    assert isinstance(repeat_block.body[1], PDLRead)
    assert repeat_block.body[1].message.strip() == "Enter a query or say 'quit' to exit"
    assert repeat_block.body[1].var_def == "question"

    # Execute the program
    p.execute()

    # Verify that the evaluation environment contains the inputs and model responses
    assert isinstance(p.evaluation_environment["_"], list)
    # Should have 3 entries: initial query, model response, and 'quit' input
    assert len(p.evaluation_environment["_"]) >= 3
    assert p.evaluation_environment["_"][0] == {'content': 'Test user query', 'role': 'user'}
    # The 'quit' response should be stored in the context with the variable name 'question'
    assert p.evaluation_environment["question"] == "quit"

def test_pdl_program_doctests():
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDLModel, PDFRead]

    program = PDLProgram(yaml.safe_load(PDL5), dispatcher=dispatcher)
    assert len(program.text.content) == 1
    assert isinstance(program.text.content[0], PDLModel)

    # Check model input structure
    model_block = program.text[0]
    assert model_block.input[0].read_from == 'some_file.txt'
    assert model_block.input[-1].strip() == "Long text."

    # Check model parameters
    expected_params = {'temperature': 0.6, 'min_p': 0.03, 'max_tokens': 600}
    assert model_block.parameters == expected_params

    # Execute and check environment
    program.execute()
    assert program.evaluation_environment["_"][0] == {'role': 'assistant', 'content': '.. model response ..'}
