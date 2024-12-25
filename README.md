# Generative Redfoot
A generative, conversational workflow and multi-agent system using PDL and [mlx](https://github.com/ml-explore/mlx-examples/tree/main/llms)

Takes a minimal [Prompt Declaration Language (PDL)](https://github.com/IBM/prompt-declaration-language) file and generates a finite state generative machine
as Python objects for a subset of the PDL language.  These objects (the "programs" in particular) can be executed. 

It was mainly motivated by supporting this use case from the PDL documentation/paper:

<img src="animated_chatbot.gif" alt="Animated GIF of PDL chatbot."/>

The Model class can be extended and incorporated into how a dispatcher creates the PDL Python objects from a PDL file to incorporate the functionality for evaluating 
the prompts against the models specified in PDL using any accumulated conversational context, prompts, and generation parameters (sampling parameters, for example), 
(optionally) updating the context as the program execution continues.  This is how mlx is used to implement the model loading and inference.

However, the language of the PDL file can be extended with additional custom functionality, and 
other LLM systems can handle the evaluation.

It depends on the PyYaml and click third-party Python libraries as well as mlx and can be run this way, where `document.pdl` is a PDL file.
```commandline
python generative_redfoot.py document.pdl
```

It can also be used programmatically, re-using the examples from its Docstring tests:

Consider the earlier PDL document. It can be parsed into a PDL Program and "executed"

```python
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

        >>> context = {}
        >>> program.execute({})


```

