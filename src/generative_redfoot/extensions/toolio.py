import warnings

from ..object_pdl_model import PDLObject, PDLStructuredBlock, PDLProgram, PDLModel
from ..utils import truncate_messages
import json
from typing import Mapping, Dict, List
import asyncio

class ToolioCompletion(PDLModel):
    """
    PDL block for structured LLM response generation via Toolio + MLX

    """
    MODEL_KEY = "structured_output"
    def __init__(self, content: Mapping, program: PDLProgram):
        super().__init__(content, program)
        self.insert_schema = content["insert_schema"]
        self.schema_file = content["schema_file"]
        self.max_tokens = self.parameters.get("max_tokens", 256)

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

    def __repr__(self):
        return f"ToolioCompletion(according to '{self.schema_file}' and up to {self.max_tokens:,} tokens)"

    def execute(self, context: Dict, verbose: bool = False):
        source_phrase = ""
        if self.input:
            source_phrase = f" from {self.input}"
        if verbose:
            messages = truncate_messages(context['_'])
            print(f"Running Toolio completion according to '{self.schema_file}', using {messages}"
                  f"{source_phrase} (max of {self.max_tokens:,} tokens)")
        if self.input:
            self.input.execute(context, verbose=verbose)
        asyncio.run(self.toolio_completion(context, verbose))

    async def toolio_completion(self, context: Dict, verbose: bool = False):
        from toolio.llm_helper import local_model_runner
        toolio_mm = local_model_runner(self.model)
        messages = context["_"]

        if self.cot_prefix:
            if verbose:
                print(f"### Adding Chain-of Thought Few Shot examples specified in {self.cot_prefix} ###")
            with open(self.cot_prefix, 'r') as cot_content:
                self._insert_cot_messages(messages, json.load(cot_content))
        if self.program.cache:
            warnings.warn(f"Prompt cache ({self.program.cache}) not supported with Toolio")

        with open(self.schema_file, mode='r') as schema_file:
            response = await toolio_mm.iter_complete(messages,
                                                     json_schema=schema_file.read(),
                                                     max_tokens=self.max_tokens,
                                                     temperature=self.parameters.get("temperature", 0),
                                                     insert_schema=self.insert_schema,
                                                     full_response=True)
            self._handle_execution_contribution(response.first_choice_text, context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLProgram):
        if "structured_output" in item:
            return ToolioCompletion(item, program)
