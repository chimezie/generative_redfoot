from collections.abc import Mapping

from generative_redfoot import prompts
from generative_redfoot.object_pdl_model import PDLObject, PDLStructuredBlock


class WorldLoomRead(PDLObject, PDLStructuredBlock):
    '''
    PDL block for reading sections for a prompt from a WordLoom (TOML) file

    Example:
        >>> import yaml
        >>> from generative_redfoot.object_pdl_model import PDL3, PDLProgram
        >>> p = PDLProgram(yaml.safe_load(PDL3))
        >>> p.cache
        'prompt_cache.safetensors'
        >>> p.text[0]
        Wordloom('question answer' from file.loom [outputs to context as user])
    '''

    def __init__(self, pdl_block: Mapping, program: PDLObject):
        self.program = program
        self.loom_file = pdl_block['read_from_wordloom']
        self.language_items = pdl_block['items']
        self._get_common_attributes(pdl_block)

    def __repr__(self):
        return f"Wordloom('{self.language_items}' from {self.loom_file} [{self.descriptive_text()}])"

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        names = self.language_items.split()
        entries = [(name, prompts.load(self.loom_file, name)) for name in names]  # path rules live in prompts.py
        if verbose:
            print(f'Expanding {names} from {self.loom_file}')
            print('Loom markers: ', ', '.join(str(entry.markers) for _, entry in entries))
        content = '\n'.join(WorldLoomRead.get_loom_entry(entry, context, name) for name, entry in entries)
        self._handle_execution_contribution(content, context)
        return content

    @staticmethod
    def get_loom_entry(loom_entry, context: Mapping, key: str = '') -> str:
        '''
        Render a language_item using marker values from the context. Items with neither markers nor file bindings
        are returned as plain text (unformatted), so literal braces in such prompts are left alone.
        '''
        if not (loom_entry.markers or loom_entry.file_bindings):
            return str(loom_entry)
        missing = [m for m in loom_entry.markers if m not in context]
        if missing:
            raise KeyError(f'Wordloom item {key!r} needs context variable(s) {missing}')
        return loom_entry.render(**{m: context[m] for m in loom_entry.markers})

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLObject):
        if 'read_from_wordloom' in item:
            return WorldLoomRead(item, program)


WordLoomRead = WorldLoomRead  # correctly-spelled alias; the original name is kept for existing importers
