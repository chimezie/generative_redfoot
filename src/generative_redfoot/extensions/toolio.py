from ..object_pdl_model import PDLObject, PDLStructuredBlock, PDLProgram
from ..utils import truncate_messages
import json
from typing import Mapping, Dict
import asyncio

"""
from toolio.llm_helper import local_model_runner
toolio_mm = local_model_runner('..')

async def [..]([..]):
    prompt = [..]
    done = False
    msgs = [{'role': 'user', 'content': prompt}]
    while not done:
        rt = await tmm.complete(msgs, json_schema=[..], max_tokens=512)
        obj = json.loads(rt)
        # print('DEBUG return object:', obj)


"""

class ToolioCompletion(PDLObject, PDLStructuredBlock):
    """
    PDL block for structured LLM response generation via Toolio + MLX

    from toolio.llm_helper import local_model_runner
    toolio_mm = local_model_runner('..')

    async def [..]([..]):
        prompt = [..]
        done = False
        msgs = [{'role': 'user', 'content': prompt}]
        while not done:
            rt = await tmm.complete(msgs, json_schema=[..], max_tokens=512)
            obj = json.loads(rt)
            # print('DEBUG return object:', obj)
    """
    def __init__(self, pdl_block: Mapping, program: PDLProgram):
        self.program = program
        self.model = pdl_block["structured_output"]
        self.insert_schema = pdl_block["insert_schema"]
        self.schema_file = pdl_block["schema_file"]
        self.max_tokens = pdl_block.get("max_tokens", 512)
        self.temperature = pdl_block.get("temperature", .1)
        self.input = self.program.dispatcher.handle(pdl_block["input"], self.program) if "input" in pdl_block else None
        self._get_common_attributes(pdl_block)

    def __repr__(self):
        return f"ToolioCompletion(according to '{self.schema_file}' and up to {self.max_tokens:,} tokens)"

    def execute(self, context: Dict, verbose: bool = False):
        source_phrase = ""
        if self.input:
            source_phrase = f" from {self.input}"
        if verbose:
            print(f"Running Toolio completion according to '{self.schema_file}', using {truncate_messages(context)}"
                  f"{source_phrase} (max of {self.max_tokens:,} tokens)")
        if self.input:
            self.input.execute(context, verbose=verbose)
        asyncio.run(self.toolio_completion(context, verbose))

    async def toolio_completion(self, context: Dict, verbose: bool = False) -> str:
        from toolio.llm_helper import local_model_runner
        toolio_mm = local_model_runner(self.model)
        msgs = context["_"]

        with open(self.schema_file, mode='r') as schema_file:
            self._handle_execution_contribution(await toolio_mm.complete(msgs,
                                                                         json_schema=schema_file.read(),
                                                                         max_tokens=self.max_tokens,
                                                                         temperature=self.temperature,
                                                                         insert_schema=self.insert_schema),
                                                context)

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLProgram):
        if "structured_output" in item:
            return ToolioCompletion(item, program)
