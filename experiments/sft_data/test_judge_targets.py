import pytest

from judge_run import validate_targets


def test_unparsed_request_can_be_reviewed_but_not_accepted():
    packet = {"steps": [{"request": "request_0001"}], "execution_summary": {
        "incomplete_requests": [{"request_id": "request_0002"}]}}
    verdict = {"target_findings": [
        {"request_id": "request_0001", "disposition": "accept_candidate"},
        {"request_id": "request_0002", "disposition": "context_only"}]}
    validate_targets(packet, verdict)
    verdict["target_findings"][1]["disposition"] = "accept_candidate"
    with pytest.raises(RuntimeError, match="incomplete request"):
        validate_targets(packet, verdict)


def test_optional_failed_request_does_not_mask_missing_real_decision():
    packet = {"steps": [{"request": "request_0001"}], "execution_summary": {
        "incomplete_requests": [{"request_id": "request_0002"}]}}
    with pytest.raises(RuntimeError, match="omitted"):
        validate_targets(packet, {"target_findings": [
            {"request_id": "request_0002", "disposition": "exclude"}]})
def test_terminal_context_is_not_a_new_sft_target_and_ids_must_be_distinct():
    from judge_run import validate_targets
    import pytest
    packet = {"steps": [{"request": "terminal_1"}], "evaluation_mode": "terminal_replay",
              "recorded_prefix": [{"request": "old_1"}]}
    validate_targets(packet, {"target_findings": [{"request_id": "terminal_1"}, {"request_id": "old_1"}]})
    with pytest.raises(RuntimeError, match="omitted or duplicated"):
        validate_targets(packet, {"target_findings": [{"request_id": "old_1"}]})
    packet["recorded_prefix"][0]["request"] = "terminal_1"
    with pytest.raises(RuntimeError, match="ambiguous"):
        validate_targets(packet, {"target_findings": [{"request_id": "terminal_1"}]})
