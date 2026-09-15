import json

import pytest

from foren_searcher.local_models import _create_output_enforcer, _decision_output_schema


def _report_enforcer():
    from lmformatenforcer import TokenEnforcerTokenizerData

    tokens = [chr(i) for i in range(32, 127)]
    eos = len(tokens)
    data = TokenEnforcerTokenizerData([(i, text, False) for i, text in enumerate(tokens)],
        lambda ids: ''.join(tokens[i] for i in ids), eos, False, eos + 1)
    schema = _decision_output_schema({'workspace_id': 'W1-1', 'available_actions': [], 'model_view': {}})
    return _create_output_enforcer(data, schema), tokens, eos


@pytest.mark.parametrize('status', ['COMPLETED', 'INCOMPLETE'])
def test_report_decoder_requires_evidence_for_each_finding(status):
    enforcer, tokens, eos = _report_enforcer()
    prefix = [0]
    text = ('{"task_id":"W1-1","status":' + json.dumps(status)
            + ',"findings":[{"statement":"A visible fact","evidence_ids":[')
    for char in text:
        token = tokens.index(char)
        assert token in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(token)
    allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert tokens.index(']') not in allowed
    assert tokens.index('"') in allowed
    assert eos not in allowed


@pytest.mark.parametrize('status,allowed_empty', [('COMPLETED', False), ('INCOMPLETE', True)])
def test_only_incomplete_reports_can_have_no_findings(status, allowed_empty):
    enforcer, tokens, _ = _report_enforcer()
    prefix = [0]
    text = '{"task_id":"W1-1","status":' + json.dumps(status) + ',"findings":['
    for char in text:
        token = tokens.index(char)
        assert token in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(token)
    assert (tokens.index(']') in enforcer.get_allowed_tokens(prefix).allowed_tokens) is allowed_empty


@pytest.mark.parametrize('report', [
    {'task_id': 'W1-1', 'status': 'COMPLETED', 'findings': [
        {'statement': 'A visible fact', 'evidence_ids': ['E1'], 'limitations': []}], 'selected_evidence_ids': ['E1']},
    {'task_id': 'W1-1', 'status': 'INCOMPLETE', 'findings': [], 'selected_evidence_ids': [],
     'status_reason': 'The required recording was not supplied.'},
])
def test_report_decoder_can_finish_supported_and_unavailable_evidence_reports(report):
    from foren_searcher.contracts import parse_workspace_report

    enforcer, tokens, eos = _report_enforcer()
    prefix = [0]
    for char in json.dumps(report, separators=(',', ':')):
        token = tokens.index(char)
        assert token in enforcer.get_allowed_tokens(prefix).allowed_tokens
        prefix.append(token)
    assert eos in enforcer.get_allowed_tokens(prefix).allowed_tokens
    assert parse_workspace_report(report).status == report['status']


@pytest.mark.parametrize('status', ['COMPLETED', 'INCOMPLETE'])
def test_report_constraint_transitions_are_independent_of_ignored_prompt_prefix(status):
    compact, tokens, eos = _report_enforcer()
    with_context, _, _ = _report_enforcer()
    context = [eos] * 8000 + [tokens.index('x')] * 1024
    generated = []
    report = {'task_id': 'W1-1', 'status': status, 'findings': [
        {'statement': 'The sign reads "Harbor". Path C:\\port.',
         'evidence_ids': ['E1'], 'limitations': []}], 'selected_evidence_ids': ['E1']}
    for char in json.dumps(report, separators=(',', ':')):
        allowed = set(compact.get_allowed_tokens(generated).allowed_tokens)
        assert allowed == set(with_context.get_allowed_tokens(context + generated).allowed_tokens)
        token = tokens.index(char)
        assert token in allowed
        generated.append(token)
    assert set(compact.get_allowed_tokens(generated).allowed_tokens) == set(
        with_context.get_allowed_tokens(context + generated).allowed_tokens)
    assert eos in compact.get_allowed_tokens(generated).allowed_tokens
