"""跨越自由字符串结尾的合并 token 仍须遵守完整 JSON 结构。"""

import json
import string

import pytest

from foren_searcher.local_models import _create_output_enforcer, _decision_output_schema


def enforcer_at_prefix(schema, prefix, candidates):
    from lmformatenforcer import TokenEnforcerTokenizerData

    characters = sorted(set(string.printable + prefix + ''.join(candidates)))
    tokens = characters + [text for text in candidates if text not in characters]
    eos = len(tokens)
    data = TokenEnforcerTokenizerData([(i, text, False) for i, text in enumerate(tokens)],
        lambda ids: ''.join(tokens[i] for i in ids), eos, False, eos + 1)
    enforcer = _create_output_enforcer(data, schema)
    generated = []
    for character in prefix:
        token = tokens.index(character)
        assert token in enforcer.get_allowed_tokens(generated).allowed_tokens
        generated.append(token)
    return enforcer, tokens, eos, generated


@pytest.mark.parametrize('prefix,token,accepted', [
    ('{"a":"', 'xy","b":1}', True),
    ('{"a":"', 'x","b":1}', False),
    ('{"a":"', 'xxxxxx","b":1}', False),
    ('{"a":"ok', '.","b":1}', True),
    ('{"a":"ok', 'x","z":1}', False),
    ('{"a":"ok', 'x","b":"bad"}', False),
    ('{"a":"ok', 'x","b":1,"extra":0}', False),
    ('{"a":"ok', 'x\n","b":1}', False),
    ('{"a":"ok', 'x","b":\n1}', True),
    ('{"a":"ok', '\\"x","b":1}', True),
    ('{"a":"', '港口","b":1}', True),
])
def test_string_transition_checks_length_escapes_and_following_structure(prefix, token, accepted):
    schema = {'type': 'object', 'properties': {
        'a': {'type': 'string', 'minLength': 2, 'maxLength': 5}, 'b': {'type': 'integer'}},
        'required': ['a', 'b'], 'additionalProperties': False}
    enforcer, tokens, eos, generated = enforcer_at_prefix(schema, prefix, [token])
    assert (tokens.index(token) in enforcer.get_allowed_tokens(generated).allowed_tokens) is accepted
    if accepted:
        import jsonschema
        jsonschema.validate(json.loads(prefix + token), schema)
        assert eos in enforcer.get_allowed_tokens(generated + [tokens.index(token)]).allowed_tokens


def test_gia_task_string_can_end_inside_a_merged_token():
    schema = _decision_output_schema({'global_view': {}, 'available_actions': []}, full_tool_schema=True)
    prefix = '{"decision":"INVESTIGATE","tasks":[{"task":"Check the page'
    fused = '.","'
    enforcer, tokens, eos, generated = enforcer_at_prefix(schema, prefix, [fused])
    assert tokens.index(fused) in enforcer.get_allowed_tokens(generated).allowed_tokens
    generated.append(tokens.index(fused))
    for char in 'materials":["E5"]}]}':
        token = tokens.index(char)
        assert token in enforcer.get_allowed_tokens(generated).allowed_tokens
        generated.append(token)
    assert eos in enforcer.get_allowed_tokens(generated).allowed_tokens


def test_report_transition_does_not_relax_required_evidence():
    schema = _decision_output_schema({'workspace_id': 'W1-1', 'model_view': {}, 'available_actions': []})
    prefix = '{"task_id":"W1-1","status":"COMPLETED","findings":[{"statement":"Observed'
    valid = '.","evidence_ids":["E1"],"limitations":[]}'
    invalid = '.","evidence_ids":[],"limitations":[]}'
    enforcer, tokens, _, generated = enforcer_at_prefix(schema, prefix, [valid, invalid])
    allowed = enforcer.get_allowed_tokens(generated).allowed_tokens
    assert tokens.index(valid) in allowed
    assert tokens.index(invalid) not in allowed
