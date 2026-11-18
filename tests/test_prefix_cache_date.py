'''A prefix cache must keep matching when the chat template's date changes (Stage 5).'''

import json
import logging

import yaml

from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import GenerationDefaults, InMemoryBackend

CLOCK = {'date': '01 Jan 2024'}  # the pretend clock the test template falls back to


class DateAwareTokenizer:
    '''Like the byte-level fake, but the template puts a `date_string` into a leading system message'''

    bos_token = None
    vocab_size = 256

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, continue_final_message=False, **kwargs
    ):
        date_string = kwargs.get('date_string', CLOCK['date'])
        text = f'<|system|>Today is {date_string}<|end|>'
        text += ''.join(f'<|{m["role"]}|>{m["content"]}<|end|>' for m in messages)
        if continue_final_message:
            text = text.removesuffix('<|end|>')  # leave the last message open, as real templates do
        elif add_generation_prompt:
            text += '<|assistant|>'
        return text

    def encode(self, text, add_special_tokens=True):
        return list(text.encode('utf-8'))

    def decode(self, ids):
        return bytes(ids).decode('utf-8', errors='replace')


class DateAwareBackend(InMemoryBackend):
    '''The in-memory backend, with a tokenizer whose template reads the pretend clock'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = DateAwareTokenizer()


CACHE_PDL = '''
cache:
- text:
  - role: user
    text: "Instructions. {{document_text}}"
    contribute: [context]
  - content_model: some/model
    name: pc
    file: pc.json
    prefix_marker: document_text
description: t
text:
- model: pc
'''

REQUEST_PDL = '''
description: t
text:
- text: "${ document_text }"
  contribute: [context]
- model: pc
'''


def build_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('generative_redfoot.extensions.mlx.today_string', lambda: CLOCK['date'])
    CLOCK['date'] = '01 Jan 2024'  # day 1: the cache is built today

    backend = DateAwareBackend(['ok'])
    dispatcher = make_dispatcher()
    ctx = {}
    program = make_program(yaml.safe_load(CACHE_PDL), dispatcher, backend, GenerationDefaults(), {}, ctx)
    cache_lookup = prepare_caches(program, ctx, verbose=False)
    metadata = json.loads((tmp_path / 'pc.json').read_text())['metadata']
    return backend, dispatcher, cache_lookup, metadata


def test_prefix_cache_still_matches_after_the_date_changes(tmp_path, monkeypatch, caplog):
    backend, dispatcher, cache_lookup, metadata = build_cache(tmp_path, monkeypatch)
    assert json.loads(metadata['template_kwargs'])['date_string'] == '01 Jan 2024'
    prefix_prompt = metadata['prefix_prompt']
    assert '01 Jan 2024' in prefix_prompt

    CLOCK['date'] = '02 Jan 2024'  # midnight happened; the template would now use a different date
    request_program = make_program(
        yaml.safe_load(REQUEST_PDL), dispatcher, backend, GenerationDefaults(), cache_lookup,
        {'document_text': 'suffix'},
    )
    with caplog.at_level(logging.WARNING, logger='generative_redfoot.extensions.mlx'):
        request_program.execute()

    call = backend.calls[0]
    tokenizer = backend.tokenizer
    assert call['tokens'] == tokenizer.encode('suffix<|end|><|assistant|>', add_special_tokens=False)
    assert call['cache_len_before'] == len(tokenizer.encode(prefix_prompt))
    assert 'generating uncached' not in caplog.text


def test_without_the_saved_date_string_the_rendered_prompt_would_not_match(tmp_path, monkeypatch):
    backend, _dispatcher, _cache_lookup, metadata = build_cache(tmp_path, monkeypatch)
    prefix_prompt = metadata['prefix_prompt']
    saved_date = json.loads(metadata['template_kwargs'])['date_string']
    cached_messages = json.loads(metadata['messages'])

    CLOCK['date'] = '02 Jan 2024'
    tokenizer = backend.tokenizer
    with_saved = tokenizer.apply_chat_template(
        cached_messages, tokenize=False, add_generation_prompt=True, date_string=saved_date
    )
    without_saved = tokenizer.apply_chat_template(cached_messages, tokenize=False, add_generation_prompt=True)

    assert with_saved.startswith(prefix_prompt)
    assert not without_saved.startswith(prefix_prompt)
