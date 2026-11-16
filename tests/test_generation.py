import json
import logging

import pytest
import yaml

from generative_redfoot.cli import make_dispatcher, make_program, prepare_caches
from generative_redfoot.generation import (
    GenerationDefaults,
    InMemoryBackend,
    common_prefix_len,
    insert_cot_messages,
    merge_cached_messages,
    resolve_sampling,
)


def b(text):
    # The fake tokenizer is byte-level, so expected token ids are just the UTF-8 bytes
    return list(text.encode('utf-8'))


def program_from(pdl, backend, cache_lookup=None, ctx=None, **defaults):
    return make_program(yaml.safe_load(pdl), make_dispatcher(), backend, GenerationDefaults(**defaults),
                        cache_lookup or {}, ctx if ctx is not None else {})


# --- pure helpers ---------------------------------------------------------------------------------------------------

def test_resolve_sampling_overrides_and_coerces():
    defaults = GenerationDefaults(temperature=0.5, top_p=0.9, top_k=-1, max_tokens=100)
    got = resolve_sampling({'temperature': '0.2', 'top_k': '20', 'max_tokens': 7}, defaults)
    assert got['temperature'] == 0.2 and got['top_k'] == 20 and got['max_tokens'] == 7
    assert got['top_p'] == 0.9  # default flows through
    assert resolve_sampling({}, defaults)['top_k'] == 0  # -1 means disabled -> mlx-lm's 0
    assert got['enable_thinking'] is None


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1, 2], [1, 2]) == 2


def test_merge_cached_messages_same_role_concatenates():
    cached = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'Instr '}]
    context = [{'role': 'user', 'content': 'DOC'}, {'role': 'assistant', 'content': 'A'}]
    assert merge_cached_messages(cached, context) == [{'role': 'system', 'content': 'S'},
                                                      {'role': 'user', 'content': 'Instr DOC'},
                                                      {'role': 'assistant', 'content': 'A'}]
    assert cached[1]['content'] == 'Instr '  # inputs untouched


def test_merge_cached_messages_different_role_appends_and_empties():
    cached = [{'role': 'user', 'content': 'U'}]
    assert merge_cached_messages(cached, [{'role': 'assistant', 'content': 'A'}])[-1]['content'] == 'A'
    assert merge_cached_messages(cached, []) == cached
    assert merge_cached_messages([], cached) == cached


def test_insert_cot_messages_after_system():
    msgs = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]
    insert_cot_messages(msgs, [{'role': 'user', 'content': 'shot'}])
    assert [m['content'] for m in msgs] == ['s', 'shot', 'u']
    msgs = [{'role': 'user', 'content': 'u'}]
    insert_cot_messages(msgs, [{'role': 'user', 'content': 'shot'}])
    assert [m['content'] for m in msgs] == ['shot', 'u']


# --- model evaluation, no cache -------------------------------------------------------------------------------------

PLAIN = '''
description: t
text:
- role: system
  text: Be brief.
  contribute: [context]
- role: user
  text: Hello
  contribute: [context]
- model: some/model
  parameters:
    top_p: 0.5
    max_tokens: 9
'''


def test_plain_evaluation_renders_prompt_and_contributes_reply():
    backend = InMemoryBackend(['hi there'])
    p = program_from(PLAIN, backend)
    p.execute()
    call = backend.calls[0]
    assert call['tokens'] == b('<|system|>Be brief.<|end|><|user|>Hello<|end|><|assistant|>')
    assert call['cache_len_before'] is None
    assert p.evaluation_environment['_'][-1] == {'role': 'assistant', 'content': 'hi there'}


def test_block_parameters_reach_backend_and_defaults_fill_the_rest():
    backend = InMemoryBackend()
    program_from(PLAIN, backend, top_p=0.95, temperature=0.3).execute()
    sampling = backend.calls[0]['sampling']
    assert sampling['top_p'] == 0.5 and sampling['max_tokens'] == 9  # from the block
    assert sampling['temperature'] == 0.3  # from CLI defaults


def test_plain_string_input_becomes_a_user_message():
    backend = InMemoryBackend()
    p = program_from('description: t\ntext:\n- model: some/model\n  input: Hello there\n', backend)
    p.execute()
    assert backend.calls[0]['tokens'] == b('<|user|>Hello there<|end|><|assistant|>')


def test_model_without_context_is_an_error():
    p = program_from('description: t\ntext:\n- model: some/model\n', InMemoryBackend())
    with pytest.raises(ValueError, match='No context'):
        p.execute()


# --- internal cache -------------------------------------------------------------------------------------------------

INTERNAL = '''
cache: "*"
description: t
text:
- role: user
  text: Hello
  contribute: [context]
- model: some/model
'''


def second_turn(program, text):
    ctx = program.evaluation_environment
    ctx['_'].append({'role': 'user', 'content': text})
    program.text[1].execute(ctx)


def test_internal_cache_sends_only_new_tokens():
    backend = InMemoryBackend(['ok'])
    p = program_from(INTERNAL, backend)
    p.execute()
    first = b('<|user|>Hello<|end|><|assistant|>')
    assert backend.calls[0]['tokens'] == first and backend.calls[0]['cache_len_before'] == 0
    second_turn(p, 'Again')
    assert backend.calls[1]['cache_len_before'] == len(first) + len(b('ok'))  # prompt + reply are in the cache
    assert backend.calls[1]['tokens'] == b('<|end|><|user|>Again<|end|><|assistant|>')  # only what is new


def test_internal_cache_trims_when_conversation_diverges():
    backend = InMemoryBackend(['ok'])
    p = program_from(INTERNAL, backend)
    p.execute()
    ctx = p.evaluation_environment
    ctx['_'] = [{'role': 'user', 'content': 'Howdy'}]  # earlier history rewritten
    p.text[1].execute(ctx)
    shared = len(b('<|user|>H'))
    assert backend.calls[1]['cache_len_before'] == shared  # trimmed back to the common prefix
    assert backend.calls[1]['tokens'] == b('<|user|>Howdy<|end|><|assistant|>')[shared:]


def test_internal_cache_rebuilds_when_not_trimmable():
    backend = InMemoryBackend(['ok'])
    p = program_from(INTERNAL, backend)
    p.execute()
    p.kv_state['cache'].trimmable = False
    ctx = p.evaluation_environment
    ctx['_'] = [{'role': 'user', 'content': 'Howdy'}]
    p.text[1].execute(ctx)
    assert backend.calls[1]['cache_len_before'] == 0
    assert backend.calls[1]['tokens'] == b('<|user|>Howdy<|end|><|assistant|>')  # everything resent


def test_identical_prompt_still_sends_a_token():
    backend = InMemoryBackend(['ok'])
    p = program_from(INTERNAL, backend)
    p.execute()
    ctx = p.evaluation_environment
    ctx['_'] = [{'role': 'user', 'content': 'Hello'}]  # same prompt again, cache already holds it (and a reply)
    p.text[1].execute(ctx)
    assert len(backend.calls[1]['tokens']) >= 1


def test_internal_cache_with_draft_model_is_built_with_draft_layers():
    backend = InMemoryBackend()
    pdl = INTERNAL.replace('- model: some/model', '- model: some/model\n  draft_model: some/draft')
    p = program_from(pdl, backend)
    p.execute()
    assert backend.calls[0]['draft'] is True
    assert p.kv_state['cache'].has_draft


# --- named (content_model) prefix cache -----------------------------------------------------------------------------

def cache_pdl(file, extra=''):
    return f'''
cache:
- text:
  - role: system
    text: You code.
    contribute: [context]
  - role: user
    text: "Instructions. {{document_text}} Trailing."
    contribute: [context]
  - content_model: some/model
    name: pc
    file: {file}
    prefix_marker: document_text{extra}
description: t
text:
- role: user
  text: DOC BODY
  contribute: [context]
- model: pc
'''


PREFIX = '<|system|>You code.<|end|><|user|>Instructions. '
SUFFIX = 'DOC BODY<|end|><|assistant|>'


def prepared(tmp_path, backend, extra=''):
    file = str(tmp_path / 'pc.json')
    ctx = {}
    p = program_from(cache_pdl(file, extra), backend, ctx=ctx)
    prepare_caches(p, ctx, verbose=False)
    p.evaluation_environment = ctx
    return p, file


def test_cache_prep_saves_open_prefix(tmp_path):
    backend = InMemoryBackend()
    _, file = prepared(tmp_path, backend)
    assert backend.prefills[0]['tokens'] == b(PREFIX)  # cut at marker; left open for the request to continue
    saved = json.loads(open(file).read())
    assert saved['metadata']['prefix_prompt'] == PREFIX
    assert json.loads(saved['metadata']['messages'])[-1]['content'] == 'Instructions. '
    assert saved['metadata']['model'] == 'some/model'


def test_kv_bits_default_and_hyphen_alias(tmp_path, caplog):
    backend = InMemoryBackend()
    prepared(tmp_path, backend)
    assert backend.prefills[0]['kv_bits'] is None and backend.prefills[0]['kv_group_size'] == 64
    backend = InMemoryBackend()
    with caplog.at_level(logging.WARNING):
        prepared(tmp_path, backend, extra='\n    kv-bits: 4\n    kv_group_size: 32')
    assert backend.prefills[0]['kv_bits'] == 4 and backend.prefills[0]['kv_group_size'] == 32
    assert 'kv-bits' in caplog.text


def test_named_cache_local_flow_sends_only_suffix(tmp_path):
    backend = InMemoryBackend(['done'])
    p, _ = prepared(tmp_path, backend)
    p.execute()  # context already holds the cached messages + document, appended to the cut user message
    call = backend.calls[0]
    assert call['tokens'] == b(SUFFIX)
    assert call['cache_len_before'] == len(b(PREFIX))
    assert p.evaluation_environment['_'][-1]['content'] == 'done'


def test_named_cache_service_flow_merges_cached_messages(tmp_path):
    backend = InMemoryBackend(['done'])
    _, file = prepared(tmp_path, backend)
    # A per-request program starts with an empty context; the cached messages must supply the prefix
    request = program_from(cache_pdl(file), backend, cache_lookup={'pc': file})
    request.execute()
    assert backend.calls[0]['tokens'] == b(SUFFIX)
    assert backend.calls[0]['cache_len_before'] == len(b(PREFIX))


def test_named_cache_prefix_mismatch_falls_back_uncached(tmp_path, caplog):
    backend = InMemoryBackend()
    p, file = prepared(tmp_path, backend)
    saved = json.loads(open(file).read())
    saved['metadata']['prefix_prompt'] = 'SOMETHING ELSE'
    open(file, 'w').write(json.dumps(saved))
    with caplog.at_level(logging.WARNING):
        p.execute()
    assert backend.calls[0]['cache_len_before'] is None  # no cache used
    assert 'generating uncached' in caplog.text


def test_named_cache_from_older_version_asks_for_regeneration(tmp_path):
    backend = InMemoryBackend()
    p, file = prepared(tmp_path, backend)
    saved = json.loads(open(file).read())
    del saved['metadata']['prefix_prompt']
    open(file, 'w').write(json.dumps(saved))
    with pytest.raises(ValueError, match='regenerate'):
        p.execute()


def test_named_cache_without_draft_layers_refuses_draft_model(tmp_path):
    backend = InMemoryBackend()
    p, _ = prepared(tmp_path, backend)
    p.text.content[-1].draft_model = 'some/draft'
    with pytest.raises(ValueError, match='draft'):
        p.execute()
