import json

import pytest

from foren_searcher.local_models import LocalModelOutputError, extract_json_object


def report():
    return {"task_id": "W1-1", "status": "COMPLETED", "findings": [
        {"statement": 'Quoted "text" and literal [brackets] } with \\ escapes; E1 = 12.5%.',
         "evidence_ids": ["E1"], "limitations": ["Only one frame"]}], "selected_evidence_ids": ["E1"]}


@pytest.mark.parametrize("field", ["limitations", "selected_evidence_ids"])
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "<think>ignore [ this</think>\n{}"])
def test_one_missing_report_array_closure_preserves_every_value(field, wrapper):
    expected = report()
    raw = json.dumps(expected)
    end = raw.index(']', raw.index('"' + field + '"'))
    raw = wrapper.format(raw[:end] + raw[end + 1:])
    with pytest.raises(LocalModelOutputError):
        extract_json_object(raw)
    assert extract_json_object(raw, recover_report_array=True) == expected


@pytest.mark.parametrize("raw", [
    '{"action":"INSPECT","arguments":{"targets":["E1"},"focus_evidence_ids":[]}',
    '{"decision":"INVESTIGATE","tasks":[{"task":"do something","materials":["E1"}}]}',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1"',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1}',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1",}',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1"},"extra":1}',
    '{"task_id":"W1-1","task_id":"W2-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1"}',
    '{"status":"COMPLETED","findings":[],"selected_evidence_ids":["E1"}',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[],"selected_evidence_ids":["E1"} trailing',
])
def test_report_recovery_never_invents_content_or_repairs_actions(raw):
    with pytest.raises(LocalModelOutputError):
        extract_json_object(raw, recover_report_array=True)


def test_report_recovery_does_not_fill_missing_finding_fields():
    raw = json.dumps(report()).replace(', "limitations": ["Only one frame"]', '').replace('["E1"]}', '["E1"}')
    with pytest.raises(LocalModelOutputError):
        extract_json_object(raw, recover_report_array=True)
