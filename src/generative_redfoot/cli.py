
import click
import yaml
import json
import re

from .utils import truncate_long_text
from .object_pdl_model import PDLModel, PDLProgram, ParseDispatcher, PDFRead, PDLRepeat, PDLText, PDLRead
from .extensions.wordloom import WorldLoomRead
from .extensions.toolio import ToolioCompletion
from pyarrow.lib import Mapping
from transformers import PreTrainedTokenizer
from typing import Tuple, Dict, List

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
    import mlx.nn as nn
    from mlx_lm.utils import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache

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

    class MLXModelEvaluationBase(PDLModel):
        def _get_model_cache_and_tokenizer(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
            eos_token = self.parameters.get("eos_token")
            if eos_token:
                tokenizer_config = {"eos_token": eos_token}
            else:
                tokenizer_config = {}
            model, tokenizer = load(self.model, tokenizer_config=tokenizer_config)
            if self.program.cache:
                if isinstance(self.program.cache, str):
                    if self.program.cache == PDLProgram.INTERNAL_CACHE_NAME:
                        self.program.cache = (make_prompt_cache(model))
                        if verbose:
                            print("Using internal cache for prompts")
                    else:
                        self.program.cache = load_prompt_cache(self.program.cache)
                        if verbose:
                            print(f"Using external cache for prompts: {self.program.cache}")
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
                            sampler=make_sampler(temp=self.parameters.get("temperature", temperature),
                                                 min_p=self.parameters.get("min_p", min_p),
                                                 top_k=self.parameters.get("top_k", top_k)),
                            logits_processors=logits_processor,
                            verbose=verbose,
                            prompt_cache=self.program.cache,
                            draft_model=draft_model), prompt

    class MLXModelEvaluation(MLXModelEvaluationBase):
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
            messages.extend(context["_"])
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
        def dispatch_check(item: Mapping, program: PDLProgram):
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
        def dispatch_check(item: Mapping, program: PDLProgram):
            if "APSModel" in item:
                return MLXAPSModel(item, program)

    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER = [PDLRead, WorldLoomRead, ToolioCompletion, PDLRepeat, PDLText,
                                            MLXModelEvaluation, MLXAPSModel, PDFRead]
    with open(pdl_file, "r") as file:
        program = PDLProgram(yaml.safe_load(file), dispatcher=dispatcher, initial_context=dict(variables))
        program.execute(verbose=verbose)
        if verbose:
            print(program.evaluation_environment)
if __name__ == '__main__':
    main()
