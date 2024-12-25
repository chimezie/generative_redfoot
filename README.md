# generative_redfoot
A generative, conversational workflow and multi-agent system using PDL, Wordloom, and [mlx](https://github.com/ml-explore/mlx-examples/tree/main/llms)

Takes a minimal [Prompt Declaration Language (PDL)](https://github.com/IBM/prompt-declaration-language) file and generates a finite state generative machine
as Python objects for a subset of the PDL language.  These objects (the "programs" in particular) can be executed. Their Model class can be extended 
and incorporated into how its dispatcher creates the objects from a PDF file to incorporate the functionality for evaluating the prompts against the models 
specified in PDL using any accumulated conversational context, prompts, and generation parameters (sampling parameters, for example), (optionally) updating
the context as the program execution continues.  In this way, the language of the PDL file can be extended with additional custom functionality, and 
the evaluation of the LM can be handled by other LLM systems than just mlx, which is how the model evaluation is implemented in this module.
