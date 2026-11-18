'''
Generation backends for Generative Redfoot.

`MLXBackend` is the only code that imports `mlx` / `mlx_lm` (always lazily, so this module imports anywhere).
`InMemoryBackend` is a real, deterministic implementation of the same interface used by unit tests, so the block
logic in `extensions/mlx.py` can be exercised without Apple hardware.

Backend interface (both classes implement it):

    load(path, tokenizer_config=None) -> (model, tokenizer)
    make_cache(model, max_kv_size=None, draft_model=None) -> cache
    load_cache(file) -> (cache, metadata)
    save_cache(file, cache, metadata)
    prefill(model, tokens, cache, kv_bits=None, kv_group_size=64, quantized_kv_start=5000, progress=None)
    generate(model, tokenizer, tokens, sampling, cache=None, draft_model=None, verbose=False)
        -> (text, generated_token_ids)
    can_trim(cache) / trim(cache, n) / cache_length(cache) / cache_covers_draft(cache, model)
'''

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass
class GenerationDefaults:
    '''CLI-level sampling defaults; PDL `parameters:` on a block override them'''

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 0.0
    max_tokens: int = 800
    trust_remote_code: bool = False


def resolve_sampling(parameters: dict, defaults: GenerationDefaults) -> dict:
    '''Merge a block's PDL `parameters` over the CLI defaults, coercing types (YAML may give strings)'''
    get = parameters.get
    return {
        'temperature': float(get('temperature', defaults.temperature)),
        'top_p': float(get('top_p', defaults.top_p)),
        'top_k': max(int(get('top_k', defaults.top_k)), 0),  # mlx-lm uses 0 (not -1) for 'disabled'
        'min_p': float(get('min_p', defaults.min_p)),
        'repetition_penalty': float(get('repetition_penalty', defaults.repetition_penalty)),
        'max_tokens': int(get('max_tokens', defaults.max_tokens)),
        'enable_thinking': get('enable_thinking'),  # None -> let the tokenizer wrapper decide
    }


def common_prefix_len(a, b) -> int:
    '''Length of the longest shared prefix of two token sequences'''
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def merge_cached_messages(cached: list[dict], context: list[dict]) -> list[dict]:
    '''
    Join the messages stored with a prefix cache to the messages from the running program. If the last cached message
    and the first program message have the same role, their contents are concatenated into one message (the cached
    user message ends at the prefix marker; the program supplies the rest of it).
    '''
    if not cached or not context:
        return [dict(m) for m in (cached or context)]
    merged = [dict(m) for m in cached]
    first, *rest = context
    if merged[-1]['role'] == first['role']:
        merged[-1]['content'] = merged[-1]['content'] + first['content']
    else:
        merged.append(dict(first))
    return merged + [dict(m) for m in rest]


def insert_cot_messages(messages: list[dict], cot_prefix: list[dict]) -> list[dict]:
    '''Insert chain-of-thought few-shot messages after a leading system message (in place)'''
    idx = 1 if messages and messages[0]['role'] == 'system' else 0
    messages[idx:idx] = cot_prefix
    return messages


class MLXBackend:
    '''Production backend on top of mlx-lm. Untestable off Apple Silicon; keep it thin.'''

    def __init__(self, trust_remote_code: bool = False):
        self.trust_remote_code = trust_remote_code
        self._loaded = {}  # (path, tokenizer config) -> (model, tokenizer); avoids reloading per block / request

    def load(self, path: str, tokenizer_config: dict | None = None):
        from mlx_lm.utils import load  # noqa: PLC0415

        cfg = {**(tokenizer_config or {}), 'trust_remote_code': self.trust_remote_code}
        key = (path, json.dumps(cfg, sort_keys=True))
        if key not in self._loaded:
            self._loaded[key] = load(path, tokenizer_config=cfg, trust_remote_code=self.trust_remote_code)
        return self._loaded[key]

    def make_cache(self, model, max_kv_size: int | None = None, draft_model=None):
        from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415

        cache = make_prompt_cache(model, max_kv_size)
        if draft_model is not None:  # speculative decoding slices the list: model layers first, then draft layers
            cache = cache + make_prompt_cache(draft_model)
        return cache

    def load_cache(self, file: str):
        from mlx_lm.models.cache import load_prompt_cache  # noqa: PLC0415

        return load_prompt_cache(file, return_metadata=True)

    def save_cache(self, file: str, cache, metadata: dict):
        from mlx_lm.models.cache import save_prompt_cache  # noqa: PLC0415

        save_prompt_cache(file, cache, metadata)

    def prefill(self, model, tokens, cache, kv_bits=None, kv_group_size=64, quantized_kv_start=5000, progress=None):
        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm.generate import generate_step  # noqa: PLC0415

        # max_tokens=0 processes the whole prompt into the cache without sampling (as mlx_lm.cache_prompt does)
        for _ in generate_step(
            mx.array(tokens),
            model,
            max_tokens=0,
            prompt_cache=cache,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            prompt_progress_callback=progress,
        ):
            pass

    def generate(self, model, tokenizer, tokens, sampling, cache=None, draft_model=None, verbose=False):
        from mlx_lm.generate import stream_generate  # noqa: PLC0415
        from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: PLC0415

        sampler = make_sampler(
            temp=sampling['temperature'], top_p=sampling['top_p'], min_p=sampling['min_p'], top_k=sampling['top_k']
        )
        processors = make_logits_processors(repetition_penalty=sampling['repetition_penalty'])
        text, ids, last = '', [], None
        for response in stream_generate(
            model,
            tokenizer,
            list(tokens),
            max_tokens=sampling['max_tokens'],
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=cache,
            draft_model=draft_model,
        ):
            text += response.text
            ids.append(int(response.token))
            last = response
            if verbose:
                print(response.text, end='', flush=True)
        if verbose and last is not None:
            print(
                f'\nPrompt: {last.prompt_tokens} tokens, {last.prompt_tps:.3f} tok/s; '
                f'generation: {last.generation_tokens} tokens, {last.generation_tps:.3f} tok/s; '
                f'peak memory {last.peak_memory:.3f} GB'
            )
        return text, ids

    def can_trim(self, cache) -> bool:
        from mlx_lm.models.cache import can_trim_prompt_cache  # noqa: PLC0415

        return can_trim_prompt_cache(cache)

    def trim(self, cache, n: int):
        from mlx_lm.models.cache import trim_prompt_cache  # noqa: PLC0415

        trim_prompt_cache(cache, n)

    def cache_length(self, cache) -> int | None:
        return getattr(cache[0], 'offset', None) if cache else None  # tokens held by the (first) KV layer

    def cache_covers_draft(self, cache, model) -> bool:
        return len(cache) > len(model.layers)


class _FakeTokenizer:
    '''Byte-level tokenizer with a trivial chat template, enough to exercise prefix/suffix logic'''

    bos_token = None
    vocab_size = 256

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, continue_final_message=False, **kwargs
    ):
        text = ''.join(f'<|{m["role"]}|>{m["content"]}<|end|>' for m in messages)
        if continue_final_message:
            text = text.removesuffix('<|end|>')  # leave the last message open, as real templates do
        elif add_generation_prompt:
            text += '<|assistant|>'
        return text

    def encode(self, text, add_special_tokens=True):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='replace')


class FakeCache(list):
    '''A KV cache that just remembers the tokens it has been fed'''

    def __init__(self, layers=2, has_draft=False, trimmable=True):
        super().__init__(range(layers))
        self.tokens = []
        self.has_draft = has_draft
        self.trimmable = trimmable


class InMemoryBackend:
    '''
    Deterministic stand-in for MLXBackend. `responses` is a list of reply strings consumed in order (the last one
    repeats); every call is recorded in `calls`, `prefills`, and `loads` for assertions.
    '''

    def __init__(self, responses=('ok',)):
        self.responses = list(responses)
        self.calls, self.prefills, self.loads = [], [], []
        self.model = SimpleNamespace(layers=[0, 0], model_type='fake')
        self.tokenizer = _FakeTokenizer()

    def load(self, path, tokenizer_config=None):
        self.loads.append((path, tokenizer_config))
        return self.model, self.tokenizer

    def make_cache(self, model, max_kv_size=None, draft_model=None):
        return FakeCache(has_draft=draft_model is not None)

    def load_cache(self, file):
        data = json.loads(Path(file).read_text())
        cache = FakeCache(has_draft=data['has_draft'])
        cache.tokens = data['tokens']
        return cache, data['metadata']

    def save_cache(self, file, cache, metadata):
        Path(file).write_text(json.dumps({'tokens': cache.tokens, 'metadata': metadata, 'has_draft': cache.has_draft}))

    def prefill(self, model, tokens, cache, kv_bits=None, kv_group_size=64, quantized_kv_start=5000, progress=None):
        self.prefills.append(
            {
                'tokens': list(tokens),
                'kv_bits': kv_bits,
                'kv_group_size': kv_group_size,
                'quantized_kv_start': quantized_kv_start,
            }
        )
        cache.tokens.extend(tokens)

    def generate(self, model, tokenizer, tokens, sampling, cache=None, draft_model=None, verbose=False):
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        ids = tokenizer.encode(text)
        self.calls.append(
            {
                'tokens': list(tokens),
                'sampling': sampling,
                'draft': draft_model is not None,
                'cache_len_before': None if cache is None else len(cache.tokens),
            }
        )
        if cache is not None:
            cache.tokens.extend(list(tokens) + ids)
        return text, ids

    def can_trim(self, cache):
        return cache.trimmable

    def trim(self, cache, n):
        del cache.tokens[len(cache.tokens) - n :]

    def cache_length(self, cache):
        return len(cache.tokens)

    def cache_covers_draft(self, cache, model):
        return cache.has_draft


def timed_progress(verbose: bool):
    '''Progress callback for prefill that prints tokens/sec (returns None when not verbose)'''
    if not verbose:
        return None
    start = time.time()

    def callback(processed, total):
        speed = processed / max(time.time() - start, 1e-9)
        print(f'\rProcessed {processed:6d}/{total} tokens ({speed:6.2f} tok/s)', end='', flush=True)

    return callback
