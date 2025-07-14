from ..object_pdl_model import PDLObject, PDLStructuredBlock
from typing import Mapping, Dict, Any

class WorldLoomRead(PDLObject, PDLStructuredBlock):
    """
    PDL block for reading sections for a prompt from a Worldloom (TOML / YAML) file using ogbujipt.word_loom

    Example:
        >>> import yaml
        >>> from ..object_pdl_model import PDL3, PDLProgram
        >>> p = PDLProgram(yaml.safe_load(PDL3))
        >>> p.cache
        'prompt_cache.safetensors'
        >>> p.text[0]
        Wordloom('question answer' from file.loom [outputs to context as user])

    """

    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.loom_file = pdl_block["read_from_wordloom"]
        self.language_items = pdl_block["items"]
        self._get_common_attributes(pdl_block)

    def __repr__(self):
        return f"Wordloom('{self.language_items}' from {self.loom_file} [{self.descriptive_text()}])"

    def execute(self, context: Dict, verbose: bool = False):
        from ogbujipt import word_loom
        with open(self.loom_file, mode='rb') as fp:
            loom = word_loom.load(fp)
        items = self.language_items.split(' ')
        if verbose:
            print(f"Expanding {items} from {self.loom_file}")
        content = '\n'.join([WorldLoomRead.get_loom_entry(loom[name], context) for name in items])
        self._handle_execution_contribution(content, context)

    @staticmethod
    def get_loom_entry(loom_entry:Any, context: Mapping) -> str:
        """
        Processes a language_item by formatting it with context-specific marker substitutions
        if markers are present in the language_item. If no markers are available, the original
        language_item is returned as is.

        :param loom_entry: A wordloom `language_item` object that contains potential markers to be
                           substituted and formatted with values from the context.
        :param context: A dictionary-like object (`Mapping`) that holds marker-to-value
                        mappings to be used for substitutions in the given loom_entry.
        :return: Returns a formatted string if markers are found and substitutions can be
                 applied; otherwise, returns the unprocessed `language_item` as is.

        """
        if loom_entry.markers:
            marker_kwargs = {}
            for marker in loom_entry.markers:
                marker_kwargs[marker] = context[marker]
            return loom_entry.format(**marker_kwargs)
        else:
            return loom_entry

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if "read_from_wordloom" in item:
            return WorldLoomRead(item, program)
