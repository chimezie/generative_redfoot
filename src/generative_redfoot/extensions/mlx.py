'''
MLX-backed PDL blocks: prefix-cache preparation (`content_model`), model evaluation (`model`), and the Gemma APS
proposition extractor (`APSModel`). All model work goes through `program.backend` (see generation.py), so nothing
here imports mlx directly; sampling defaults come from `program.defaults`.

Prefix caching follows the mlx-lm convention: the cached prefix is rendered with `continue_final_message=True`,
and at evaluation time only the text after that prefix is sent to the model.
'''

import functools
import json
import logging
import re
from collections.abc import Mapping
from types import SimpleNamespace

from generative_redfoot.expressions import evaluate, stringify
from generative_redfoot.generation import (
    common_prefix_len,
    insert_cot_messages,
    merge_cached_messages,
    resolve_sampling,
    timed_progress,
)
from generative_redfoot.object_pdl_model import PDLModel, PDLProgram
from generative_redfoot.utils import truncate_long_text

logger = logging.getLogger(__name__)

# Sentence delimiters used by the Gemma APS (propositions) model
START_MARKER = '<s>'
END_MARKER = '</s>'
SEPARATOR = '\n'


@functools.cache  # each download checks the nltk index over the network, so do it once per process
def _ensure_nltk_data():
    import nltk  # noqa: PLC0415

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # newer nltk releases look for this resource instead
    return nltk


def create_propositions_input(text: str) -> str:
    nltk = _ensure_nltk_data()
    sents = nltk.tokenize.sent_tokenize(text)
    return ''.join(f'{START_MARKER} {sent} {END_MARKER}{SEPARATOR}' for sent in sents).strip(SEPARATOR)


def process_propositions_output(text: str) -> list[list[str]]:
    pattern = re.compile(f'{re.escape(START_MARKER)}(.*?){re.escape(END_MARKER)}', re.DOTALL)
    grouped = (g.strip(SEPARATOR) for g in re.findall(pattern, text))
    return [[line[2:] for line in g.split(SEPARATOR)] for g in grouped]  # drop the leading '- ' of each line


def encode_prompt(tokenizer, text: str) -> list[int]:
    '''Tokenize a rendered chat prompt; skip special tokens if the template already supplied the BOS token'''
    bos = getattr(tokenizer, 'bos_token', None)
    return tokenizer.encode(text, add_special_tokens=bos is None or not text.startswith(bos))


class CachePrep(PDLModel):
    '''
    Handles `content_model` blocks in the PDL `cache` section: renders the messages accumulated so far, processes
    them through the model, and saves the resulting KV cache (plus metadata) to `file` under `name`, so that later
    `model: <name>` blocks can reuse it.

    `prefix_marker` names a Wordloom marker; the (single) user message is cut at the first `{marker}`, so the cache
    holds only the part of the prompt common to all requests. Quantization / size options: `kv_bits` (default none),
    `kv_group_size` (64), `quantized_kv_start` (5000), `max_kv_size` (default none: the cache is not capped). mlx-lm
    cannot quantize a size-capped (rotating) cache, so `kv_bits` and `max_kv_size` cannot be combined.
    '''

    MODEL_KEY = 'content_model'
    KEYS = frozenset({
        'content_model', 'name', 'file', 'prefix_marker', 'kv_bits', 'kv-bits', 'kv_group_size',
        'quantized_kv_start', 'max_kv_size', 'input', 'parameters', 'alpha_one', 'draft_model', 'cot_prefix',
    })

    def __init__(self, content: Mapping, program: PDLProgram):
        super().__init__(content, program)
        if 'name' not in content:
            raise ValueError("content_model is missing the required key 'name'")
        if 'file' not in content:
            raise ValueError("content_model is missing the required key 'file'")
        if self.draft_model:
            logger.warning('draft_model is ignored for content_model: prefix caches are built without a draft model')
        self.name = content['name']
        self.prefix_marker = content.get('prefix_marker')
        self.file = content['file']
        self.kv_group_size = content.get('kv_group_size', 64)
        self.quantized_kv_start = content.get('quantized_kv_start', 5000)
        self.max_kv_size = content.get('max_kv_size')  # None: an uncapped cache, which can also be quantized
        if 'kv-bits' in content:  # hyphenated spelling appeared in older example PDL files and was silently ignored
            logger.warning("'kv-bits' is deprecated, use 'kv_bits'")
        self.kv_bits = content.get('kv_bits', content.get('kv-bits'))
        if self.kv_bits is not None and self.max_kv_size is not None:
            # A capped cache is a RotatingKVCache, which mlx-lm cannot quantize: prefill would fail part-way through
            raise ValueError(
                f'content_model {self.name!r}: kv_bits and max_kv_size cannot be combined (mlx-lm cannot quantize a '
                'size-capped cache); drop one of them'
            )

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLProgram):
        if 'content_model' in item:
            return CachePrep(item, program)

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        backend = self.program.backend
        parameters = evaluate(self.parameters, context)
        messages = context.get('_') or []
        users = [m for m in messages if m['role'] == 'user']
        if len(users) != 1:
            raise ValueError(f'Cache preparation needs exactly one user message, found {len(users)}')
        if self.prefix_marker:
            # Cut in place on purpose: the rest of the program continues from this truncated message
            users[0]['content'] = users[0]['content'].split(f'{{{self.prefix_marker}}}')[0]
            if verbose:
                print(f"Separating at prefix marker: {self.prefix_marker}: '.. {users[0]['content'][-20:]}'")
        template_kwargs = {}
        if parameters.get('enable_thinking') is not None:
            template_kwargs['enable_thinking'] = parameters['enable_thinking']
        model, tokenizer = backend.load(self.model, {})
        # The prefix is left open (no end-of-turn / generation prompt) so the request text continues it
        prefix_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=True, **template_kwargs
        )
        cache = backend.make_cache(model, self.max_kv_size)
        backend.prefill(
            model,
            encode_prompt(tokenizer, prefix_prompt),
            cache,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            quantized_kv_start=self.quantized_kv_start,
            progress=timed_progress(verbose),
        )
        backend.save_cache(
            self.file,
            cache,
            {
                'model': self.model,
                'tokenizer_config': json.dumps({}),
                'messages': json.dumps([{'role': m['role'], 'content': m['content']} for m in messages]),
                'prefix_prompt': prefix_prompt,
                'template_kwargs': json.dumps(template_kwargs),
            },
        )
        if verbose:
            print(f'\nSaved prompt cache to {self.file}')


class MLXModelEvaluationBase(PDLModel):
    '''
    Loads models / caches and generates. Which cache applies depends on the model reference and program `cache`:

    - the model names a cache created by `content_model` (in program.cache_lookup): the cached prefix is reused and
      only the text after it is sent;
    - `cache: '*'`: an internal KV cache is kept across blocks; only tokens beyond what it already holds are sent;
    - `cache: <file>`: the saved cache is loaded for each evaluation and the program's prompt is sent in full;
    - otherwise no cache.
    '''

    KEYS = PDLModel.KEYS

    def _load(self, verbose: bool = False, parameters: dict | None = None):
        '''Returns (model, tokenizer, ctx) where ctx describes the cache in play (ctx.kind, ctx.cache, ...)'''
        program, backend = self.program, self.program.backend
        ctx = SimpleNamespace(kind=None, cache=None, metadata=None, state=None, draft=None, file=None)
        parameters = self.parameters if parameters is None else parameters
        eos_token = parameters.get('eos_token')
        tokenizer_config = {'eos_token': eos_token} if eos_token else {}
        model_path = self.model
        if self.model in program.cache_lookup:  # cache created by a content_model block: it names its own model
            ctx.file = program.cache_lookup[self.model]
            ctx.cache, ctx.metadata = backend.load_cache(ctx.file)
            ctx.kind = 'named'
            model_path = ctx.metadata['model']
            tokenizer_config = json.loads(ctx.metadata['tokenizer_config'])
            if verbose:
                print(f'Set cache to content from {ctx.file} for {model_path}')
        model, tokenizer = backend.load(model_path, tokenizer_config)
        if self.draft_model:
            draft, draft_tokenizer = backend.load(self.draft_model)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                raise ValueError('Draft model tokenizer does not match model tokenizer.')
            ctx.draft = draft
            if verbose:
                print(f'Using draft model: {self.draft_model}')

        spec = program.cache
        if ctx.kind != 'named' and spec == PDLProgram.INTERNAL_CACHE_NAME:
            state = program.kv_state
            if state is None or state['model'] != model_path or state['draft'] != (ctx.draft is not None):
                state = program.kv_state = {
                    'model': model_path,
                    'draft': ctx.draft is not None,
                    'tokens': [],
                    'cache': backend.make_cache(model, draft_model=ctx.draft),
                }
                if verbose:
                    print('Using internal cache for prompts')
            ctx.kind, ctx.cache, ctx.state = 'internal', state['cache'], state
        elif ctx.kind != 'named' and isinstance(spec, str):
            ctx.kind = 'external'
            ctx.file = spec
            ctx.cache, _ = backend.load_cache(spec)
            if verbose:
                print(f'Using external cache for prompts: {spec}')
        if (
            ctx.kind in ('named', 'external')
            and ctx.draft is not None
            and not backend.cache_covers_draft(ctx.cache, model)
        ):
            raise ValueError(
                f'Prompt cache {ctx.file} was created without the draft model, so it cannot be used '
                "with draft_model (speculative decoding); prefix caches cannot be built with a draft model, use "
                "`cache: '*'` (internal cache) with draft_model instead"
            )
        return model, tokenizer, ctx

    def generate(self, messages: list[dict], verbose: bool = False, parameters: dict | None = None) -> str:
        backend = self.program.backend
        parameters = self.parameters if parameters is None else parameters
        model, tokenizer, ctx = self._load(verbose, parameters)
        sampling = resolve_sampling(parameters, self.program.defaults)
        template_kwargs = json.loads(ctx.metadata.get('template_kwargs', '{}')) if ctx.kind == 'named' else {}
        if sampling['enable_thinking'] is not None:
            template_kwargs['enable_thinking'] = sampling['enable_thinking']

        def render(msgs):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)

        cache, cached_tokens = None, 0
        if ctx.kind == 'named':
            prefix = ctx.metadata.get('prefix_prompt')
            if prefix is None:
                raise ValueError(
                    f"Prompt cache {ctx.file} has no 'prefix_prompt' metadata (made by an older "
                    'version of generative_redfoot); regenerate it'
                )
            prompt = render(messages) if messages else ''
            if not prompt.startswith(prefix):  # program context lacks the cached messages: put them in front
                prompt = render(merge_cached_messages(json.loads(ctx.metadata['messages']), messages))
            if prompt.startswith(prefix):
                tokens = tokenizer.encode(prompt[len(prefix) :], add_special_tokens=False)  # as mlx_lm.generate does
                cache, cached_tokens = ctx.cache, backend.cache_length(ctx.cache) or 0
            else:
                logger.warning(f'Prompt does not start with the prefix cached in {ctx.file}; generating uncached')
                tokens = encode_prompt(tokenizer, prompt)
        else:
            tokens = full = encode_prompt(tokenizer, render(messages))
            if ctx.kind == 'external':
                cache = ctx.cache
            elif ctx.kind == 'internal':
                state, held = ctx.state, ctx.state['tokens']  # `held`: the tokens the KV cache currently contains
                shared = 0 if held is None else min(common_prefix_len(held, full), len(full) - 1)  # resend >= 1 token
                if held is None or (shared < len(held) and not backend.can_trim(ctx.cache)):
                    state['cache'] = ctx.cache = backend.make_cache(model, draft_model=ctx.draft)  # cannot reuse
                    state['tokens'], shared = [], 0
                elif shared < len(held):
                    backend.trim(ctx.cache, len(held) - shared)  # conversation diverged: drop the stale tail
                tokens, cache, cached_tokens = full[shared:], ctx.cache, shared
        if verbose:
            print(
                f'Sending {len(tokens):,} tokens ({cached_tokens:,} already cached) to {self.model}; '
                f'parameters: {parameters}'
            )
        try:
            text, generated = backend.generate(
                model, tokenizer, tokens, sampling, cache=cache, draft_model=ctx.draft, verbose=verbose
            )
        except BaseException:
            if ctx.kind == 'internal':
                ctx.state['tokens'] = None  # the cache may hold a partial generation; force a rebuild next time
            raise
        if ctx.kind == 'internal':
            # The cache holds a prefix of prompt + generated tokens; trust the cache's own length over assumptions
            seq, length = full + generated, backend.cache_length(cache)
            ctx.state['tokens'] = seq[:length] if length is not None and length <= len(seq) else None
        return text


class MLXModelEvaluation(MLXModelEvaluationBase):
    '''
    Evaluates `model:` blocks against the accumulated context, optionally with a chain-of-thought few-shot prefix
    (`cot_prefix`), a draft model (`draft_model`), and one of the cache modes described in the base class.
    The response is contributed back to the context per the block's `contribute` setting.
    '''

    KEYS = PDLModel.KEYS

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        parameters = evaluate(self.parameters, context)
        if isinstance(self.input, str):  # a plain string input is a user message added to the context
            self.add_message(context, stringify(evaluate(self.input, context)), role='user')  # its own message
        elif self.input:
            self.input.execute(context, verbose=verbose)
        messages = list(context.get('_') or [])
        if not messages and self.model not in self.program.cache_lookup:
            raise ValueError('No context found for model evaluation.')
        if self.cot_prefix:
            print(f'### Adding Chain-of Thought Few Shot examples specified in {self.cot_prefix} ###')
            with open(self.cot_prefix) as cot_content:
                insert_cot_messages(messages, json.load(cot_content))
        if verbose:
            from pprint import pprint  # noqa: PLC0415

            print('Generating response using ..')
            pprint([{k: v if k == 'role' else truncate_long_text(v)} for i in messages for k, v in i.items()])
        else:
            print('Generating response ... ')
        if self.alpha_one:
            response = self._alpha_one_response(messages, verbose, parameters)
        else:
            response = self.generate(messages, verbose=verbose, parameters=parameters)
        if verbose:
            print(f'Executing model: {self.model} using context {context} - (via mlx)-> >\n{response}')
        self._handle_execution_contribution(response, context)
        if 'context' not in self.contribute:
            if verbose:
                print('Clearing context ...')
            context['_'] = []
        return response

    def _alpha_one_response(self, messages: list[dict], verbose: bool, parameters: dict | None = None) -> str:
        '''
        Alpha-One reasoning via alpha_one_mlx. Carried over from the previous implementation; revisiting it against
        current alpha_one_mlx / mlx-lm is deferred (it is a separate effort), so it is unverified.
        '''
        from alpha_one_mlx.models import (  # noqa: PLC0415
            average_thinking_tokens,
            get_configuration,
            get_thinking_template_parser,
        )
        from alpha_one_mlx.reasoner import alpha_one  # noqa: PLC0415

        parameters = self.parameters if parameters is None else parameters
        model, tokenizer, ctx = self._load(verbose, parameters)
        if ctx.kind == 'internal':
            ctx.state['tokens'] = None  # alpha-one feeds the cache outside our token ledger
        sampling = resolve_sampling(parameters, self.program.defaults)
        max_tokens = sampling['max_tokens']
        configuration = get_configuration(model.model_type)
        thinking_parser = get_thinking_template_parser(model.model_type, tokenizer)
        alpha = self.alpha_one.get('alpha', 1.4)
        if 'thinking_token_length' in self.alpha_one:
            threshold = int(max_tokens - alpha * self.alpha_one['thinking_token_length'])
        else:
            num_avg_thinking_tokens = average_thinking_tokens(thinking_parser, messages)
            logger.info(f'Average thinking tokens length so far: {num_avg_thinking_tokens}')
            threshold = int(max_tokens - alpha * int(num_avg_thinking_tokens))
        logger.debug(f'Using threshold: {threshold}')
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        wait_words = self.alpha_one.get('wait_words', configuration.slow_thinking_stop_words)
        if verbose:
            print(f'Using parameters: {parameters} and {self.alpha_one} for {self.model}')
        response = alpha_one(
            model,
            tokenizer,
            prompt,
            configuration=configuration,
            max_tokens_per_call=max_tokens,
            threshold=threshold,
            temperature=sampling['temperature'],
            top_p=sampling['top_p'],
            min_p=sampling['min_p'],
            top_k=sampling['top_k'],
            apply_chat_template=False,
            verbose=verbose,
            wait_words=wait_words,
            prompt_cache=ctx.cache,
            draft_model=ctx.draft,
        )
        if parsed := thinking_parser.break_llm_response_parts(response):
            thoughts, answer = parsed
            logger.debug(
                f'Final thinking tokens length: {len(tokenizer.encode(thoughts, add_special_tokens=False))}.  '
                f'Non-thinking tokens length: {len(tokenizer.encode(answer, add_special_tokens=False))}'
            )
        else:
            logger.warning(f'Unable to discern thinking tokens from {len(response):,} characters')
        return response

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLProgram):
        if 'model' in item:
            return MLXModelEvaluation(item, program)


class MLXAPSModel(MLXModelEvaluationBase):
    '''Extracts propositions from the last message with Google's gemma-7b-aps-it (`APSModel:` block)'''

    MODEL_KEY = 'APSModel'
    KEYS = frozenset({'APSModel', 'input', 'parameters', 'alpha_one', 'draft_model', 'cot_prefix'})

    def execute(self, context: dict, verbose: bool = False):
        self._execute_defs(context, verbose)
        msg = context['_'][-1].copy()
        if verbose:
            print('Extracting individual facts, statements, and ideas from using ', truncate_long_text(msg['content']))
        else:
            print('Generating response ... ')
        if '<think>' in msg['content']:
            print('Removing <think> tags from input ...')
            msg['content'] = msg['content'].split('</think>')[-1].strip()
        msg['content'] = create_propositions_input(msg['content'])
        msg['role'] = 'user'
        parameters = evaluate(self.parameters, context)
        response = process_propositions_output(self.generate([msg], verbose=verbose, parameters=parameters))
        self._handle_execution_contribution(response, context)
        if 'context' not in self.contribute:
            if verbose:
                print('Clearing context ...')
            context['_'] = []
        return response

    @staticmethod
    def dispatch_check(item: Mapping, program: PDLProgram):
        if 'APSModel' in item:
            return MLXAPSModel(item, program)
