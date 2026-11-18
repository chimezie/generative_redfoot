'''
Evaluation of the PDL `${ ... }` expression syntax.

A PDL string may contain `${ ... }` expressions. Anything else is literal text: prompts that contain `{`, `{#`
or `{%` must pass through untouched, so only `${` triggers Jinja and block/comment delimiters are made unlikely
to ever match.

- A string with no `${` is returned unchanged (and never parsed by Jinja).
- A string that is exactly one `${ ... }` evaluates to the typed value of the expression (int, list, bytes, ...).
- Any other string with `${ ... }` is rendered to text; trailing newlines are kept.
- Lists and dicts are evaluated recursively; every other type passes through untouched.
'''

import json
import re
from collections.abc import Mapping

from jinja2 import StrictUndefined, Undefined, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

EXPR_START = '${'
EXPR_END = '}'
LEGACY_REFERENCE_PATTERN = re.compile(r'^\$(?P<variable>[^\s]+)\s*$')

# Only `${` and `}` delimit expressions. Block and comment delimiters are given absurd spellings so that `{%` and
# `{#` in ordinary prompt text are treated as plain text instead of being parsed as Jinja constructs.
_ENV = SandboxedEnvironment(
    variable_start_string=EXPR_START,
    variable_end_string=EXPR_END,
    block_start_string='{%%%%%PDL%%%%%%%%%%',
    block_end_string='%%%%%PDL%%%%%%%%%%}',
    comment_start_string='{#####PDL#####',
    comment_end_string='#####PDL#####}',
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


def _scope_for_jinja(scope: Mapping) -> dict:
    '''The context variables plus the accumulated messages, exposed as pdl_context (overriding any context key)'''
    return {**scope, 'pdl_context': scope.get('_') or []}


def _is_single_expression(parsed) -> bool:
    '''True when the whole string is one `${ ... }` (one Jinja output node, no surrounding text)'''
    from jinja2 import nodes  # noqa: PLC0415

    body = parsed.body
    return (
        len(body) == 1
        and isinstance(body[0], nodes.Output)
        and len(body[0].nodes) == 1
        and not isinstance(body[0].nodes[0], nodes.TemplateData)
    )


def _evaluate_string(value: str, scope: Mapping):
    if EXPR_START not in value:  # fast path: no expression syntax at all, so never hand it to Jinja
        return value
    jinja_scope = _scope_for_jinja(scope)
    try:
        parsed = _ENV.parse(value)
        if _is_single_expression(parsed):
            result = _ENV.compile_expression(value[len(EXPR_START) : -len(EXPR_END)], undefined_to_none=False)(
                jinja_scope
            )
            if isinstance(result, Undefined):
                raise UndefinedError(str(result))
            return result
        return _ENV.from_string(value).render(jinja_scope)
    except Exception as exc:  # any failure here is the author's expression; Exception excludes KeyboardInterrupt
        raise ValueError(f'Error evaluating expression {value!r}: {type(exc).__name__}: {exc}') from exc


def evaluate(value, scope: Mapping):
    '''
    Evaluate every `${ ... }` expression in a value written in a PDL file.

    Strings, lists and dicts are traversed; other values (bytes, numbers, files, variables) pass through unchanged.
    `scope` is the execution context; its `'_'` message list is exposed to expressions as `pdl_context`.
    '''
    if isinstance(value, str):
        return _evaluate_string(value, scope)
    if isinstance(value, list):
        return [evaluate(item, scope) for item in value]
    if isinstance(value, dict):
        return {key: evaluate(item, scope) for key, item in value.items()}
    return value


def stringify(value) -> str:
    '''Text for a value headed into a message or a joined result: lists/dicts as JSON, bytes decoded, None empty'''
    if isinstance(value, str):
        return value
    if value is None:
        return ''
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode('utf-8', errors='replace')
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def resolve_variable_reference(value, scope: Mapping):
    '''
    Resolve the legacy variable reference `{ $name }` (a one-entry mapping whose key is `$name`).

    Anything that is not such a mapping is returned unchanged, so this is safe to call on every read value.
    '''
    if isinstance(value, Mapping) and len(value) == 1:
        key = next(iter(value))
        if isinstance(key, str) and (match := LEGACY_REFERENCE_PATTERN.match(key)):
            name = match.group('variable')
            if name not in scope:
                raise ValueError(f'Unknown variable {name!r} referenced by {{ ${name} }}')
            return scope[name]
    return value
