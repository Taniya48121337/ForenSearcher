"""重复引用不改变证据集合；未知、未读、错误类型的引用仍拒绝。"""

import copy

import pytest

from foren_searcher.local_models import _normalize_hi_output, _validate_hi_output


def report():
    return {"task_id": "W1-1", "status": "COMPLETED", "findings": [{
        "statement": "A fact from E1", "evidence_ids": ["E1", "E1"], "limitations": ["Only this source"]}],
        "selected_evidence_ids": ["E1", "E1"]}


def view():
    return {"evidence_view": [{"evidence_id": "E1"}, {"evidence_id": "E2"}],
            "content_observed_evidence_ids": ["E1"]}


def validate(value):
    _validate_hi_output(value, workspace_id="W1-1", model_view=view(), actions=({"name": "READ_RECORD"},))


def test_repeated_report_references_preserve_facts_and_source_order():
    original = report()
    saved = copy.deepcopy(original)
    value = _normalize_hi_output(original, "W1-1", view())
    validate(value)
    assert value["selected_evidence_ids"] == ["E1"]
    assert value["findings"][0]["evidence_ids"] == ["E1"]
    assert value["findings"][0]["statement"] == saved["findings"][0]["statement"]
    assert value["findings"][0]["limitations"] == saved["findings"][0]["limitations"]
    assert original == saved


@pytest.mark.parametrize("identifier", ["E2", "E99"])
def test_deduplication_does_not_authorize_unread_or_unknown_evidence(identifier):
    value = report()
    value["findings"][0]["evidence_ids"] = [identifier, identifier]
    value["findings"][0]["statement"] = "A claim"
    value["selected_evidence_ids"] = [identifier, identifier]
    with pytest.raises(ValueError):
        validate(_normalize_hi_output(value, "W1-1", view()))


def test_malformed_reference_types_still_fail_validation():
    value = report()
    value["selected_evidence_ids"] = ["E1", {}, {}]
    with pytest.raises(ValueError):
        validate(_normalize_hi_output(value, "W1-1", view()))


def test_duplicate_focus_is_accepted_without_changing_the_requested_action():
    original = {"action": "READ_RECORD", "arguments": {"record_id": "E1"}, "focus_evidence_ids": ["E1", "E1"]}
    value = _normalize_hi_output(original, "W1-1", view())
    validate(value)
    assert value["arguments"] == original["arguments"]
    assert value["focus_evidence_ids"] == ["E1"]
    original["focus_evidence_ids"] = ["E99", "E99"]
    with pytest.raises(ValueError):
        validate(_normalize_hi_output(original, "W1-1", view()))
