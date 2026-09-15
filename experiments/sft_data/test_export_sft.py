from export_sft import eligible


def test_an_executable_action_is_not_automatically_a_good_target():
    link = {"status": "executed", "raw_equals_accepted": True}
    transport = {"returncode": 0, "external_tool_items": []}
    assert not eligible({"disposition": "context_only", "target_rating": 4, "critical_unknown": False}, link, transport)
    assert not eligible({"disposition": "accept_candidate", "target_rating": 4, "critical_unknown": True}, link, transport)
    assert eligible({"disposition": "accept_candidate", "target_rating": 3, "critical_unknown": False}, link, transport)


def test_rejected_or_host_rewritten_proposals_are_not_original_positive_targets():
    target = {"disposition": "accept_candidate", "target_rating": 4, "critical_unknown": False}
    transport = {"returncode": 0, "external_tool_items": []}
    assert not eligible(target, {"status": "proposal_not_accepted", "raw_equals_accepted": True}, transport)
    assert not eligible(target, {"status": "report_committed", "raw_equals_accepted": False}, transport)
