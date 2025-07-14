from generative_redfoot.object_pdl_model import ParseDispatcher, PDLRead, PDLRepeat, PDLText, PDFRead, PDLProgram
import yaml

PDL = """
description: stuff
text:
  - text:
    - PDF_read: Test.pdf
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

def test_mixed_role_to_result():
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDFRead]
    p = PDLProgram(yaml.safe_load(PDL2), dispatcher=dispatcher)
    p.execute()

def test_pdf_and_long_string():
    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, PDLRepeat, PDLText, PDFRead]
    p = PDLProgram(yaml.safe_load(PDL), dispatcher=dispatcher)
    p.execute()
    assert p.evaluation_environment == {'_': [{'content': 'The\n'
                   'quick\n'
                   'brown\n'
                   'fox\n'
                   'jumped\n'
                   'over\n'
                   'the\n'
                   'lazy\n'
                   'moon\n'
                   'Blazay skidaddle\n',
        'role': 'user'}]}
