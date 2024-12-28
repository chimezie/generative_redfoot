
import click
import yaml
import json
import re

from .object_pdl_model import PDLModel, PDLProgram, ParseDispatcher
from pyarrow.lib import Mapping
from transformers import PreTrainedTokenizer
from typing import Tuple, Dict, List

def truncate_long_text(text, max_length=75):
    return (text[:max_length] + '..') if len(text) > max_length else text

@click.command()
@click.option('-t', '--temperature', default=1, type=float)
@click.option('-rp', '--repetition-penalty', default=0, type=float,
              help='The penalty factor for repeating tokens (none if not used)')
@click.option('--top_k', default=-1, type=int, help='Sampling top_k')
@click.option('--max_tokens', default=800, type=int, help='Max tokens')
@click.option('--min-p', default=0, type=float, help='Sampling min-p')
@click.option('--verbose/--no-verbose', default=False)
@click.argument('pdl_file')
def main(temperature, repetition_penalty, top_k, max_tokens, min_p, verbose, pdl_file):
    from mlx_lm.utils import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
    import mlx.nn as nn

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
        def _get_model_and_tokenizer(self) -> Tuple[nn.Module, PreTrainedTokenizer]:
            eos_token = self.parameters.get("eos_token")
            if eos_token:
                tokenizer_config = {"eos_token": eos_token}
            else:
                tokenizer_config = {}
            return load(self.model, tokenizer_config=tokenizer_config)

        def generate(self, messages, tokenizer, model, verbose):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if verbose:
                print(f"Using parameters: {self.parameters}")
            logits_processor = make_logits_processors(repetition_penalty=self.parameters.get("repetition_penalty",
                                                                                             repetition_penalty))
            return generate(model, tokenizer, prompt,
                            max_tokens=self.parameters.get("max_tokens", max_tokens),
                            sampler=make_sampler(temp=self.parameters.get("temperature", temperature),
                                                 min_p=self.parameters.get("min_p", min_p),
                                                 top_k=self.parameters.get("top_k", top_k)),
                            logits_processors=logits_processor,
                            verbose=verbose), prompt

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
            model, tokenizer = self._get_model_and_tokenizer()
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
                print(f"Generating response using ..")
                pprint([{k: v if k == "role" else truncate_long_text(v)} for i in messages for k,v in i.items()])
            else:
                print(f"Generating response ... ")
            response, prompt = self.generate(messages, tokenizer, model, verbose=verbose)
            # if not verbose:
            #     print(response)
            if verbose:
                print(f"Executing model: {self.model} using context {context} - (via mlx)-> >\n{response}")
            self._handle_execution_contribution(response, context)

        @staticmethod
        def dispatch_check(item: Mapping, program: PDLProgram):
            if "model" in item:
                return MLXModelEvaluation(item, program)


    class MLXAPSModel(MLXModelEvaluationBase):
        MODEL_KEY = "APSModel"
        def execute(self, context, return_content=False):
            model, tokenizer = self._get_model_and_tokenizer()
            msg = context["_"][-1].copy()
            assert msg["role"] == "assistant", "Last message must be from assistant to use APSModel"
            if verbose:
                print(f"Extracting individual facts, statements, and ideas from using ",
                      truncate_long_text(msg["content"]))
            else:
                print(f"Generating response ... ")
            msg["content"] = create_propositions_input(msg["content"])
            msg["role"] = "user"
            response, prompt = self.generate([msg], tokenizer, model, verbose=verbose)
            response = process_propositions_output(response)
            self._handle_execution_contribution(response, context)

        @staticmethod
        def dispatch_check(item: Mapping, program: PDLProgram):
            if "APSModel" in item:
                return MLXAPSModel(item, program)

    dispatcher = ParseDispatcher()
    dispatcher.DISPATCH_RESOLUTION_ORDER[-1] = MLXModelEvaluation
    dispatcher.DISPATCH_RESOLUTION_ORDER.append(MLXAPSModel)
    with open(pdl_file, "r") as file:
        program = PDLProgram(yaml.safe_load(file), dispatcher=dispatcher)
        program.execute()
        if verbose:
            print(program.evaluation_environment)
if __name__ == '__main__':
    main()
