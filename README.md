# generative_redfoot
A generative, conversational workflow and multi-agent system using PDL, Wordloom, and [mlx](https://github.com/ml-explore/mlx-examples/tree/main/llms)

Takes a minimal [Prompt Declaration Language (PDL)](https://github.com/IBM/prompt-declaration-language) file and generates a finite state generative machine
as Python objects for a subset of the PDL language.  These objects (the "programs" in particular) can be executed. Their Model class can be extended 
and incorporated into how its dispatcher creates the objects from a PDL file to incorporate the functionality for evaluating the prompts against the models 
specified in PDL using any accumulated conversational context, prompts, and generation parameters (sampling parameters, for example), (optionally) updating
the context as the program execution continues.  In this way, the language of the PDL file can be extended with additional custom functionality, and 
other LLM systems can handle the evaluation of the LM.  Currently, mlx is used to implement the model loading and inference.

It depends on the PyYaml and click third-party Python libraries as well as mlx.

It can be run this way, where `document.pdl` is a PDL file.
```commandline
python generative_redfoot.py document.pdl
```
