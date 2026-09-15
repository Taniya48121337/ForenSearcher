from assessment_packet import pack_repeated, unpack_repeated, workspace_outcome_summary


def test_repeated_state_is_lossless_and_keeps_different_visibility():
    evidence = {"evidence_id": "E4", "body": "Source text. " * 100}
    value = {"steps": [
        {"view": {"evidence": [evidence], "visible": ["E4"]}},
        {"view": {"evidence": [evidence], "visible": ["E4", "E5"]}},
    ]}
    packed = pack_repeated(value)
    assert packed["objects"]
    assert unpack_repeated(packed) == value


def test_nested_references_preserve_empty_null_and_numeric_values():
    common = {"text": "x" * 500, "list": [None, 0, False, {}, []]}
    value = [common, common, {"text": "x" * 500, "list": []}]
    assert unpack_repeated(pack_repeated(value)) == value


def test_recovered_workspace_preserves_error_but_uses_latest_outcome():
    events = [
        {"event_type": "round_started", "actor_id": "HARNESS", "payload": {"round_index": 1, "workspace_count": 2}},
        {"event_type": "workspace_outcome", "actor_id": "W1-1", "payload": {"run_status": "reported"}},
        {"event_type": "workspace_outcome", "actor_id": "W1-2", "payload": {"run_status": "runtime_error"}},
        {"event_type": "round_started", "actor_id": "HARNESS", "payload": {"round_index": 1, "workspace_count": 2}},
        {"event_type": "workspace_outcome", "actor_id": "W1-2", "payload": {"run_status": "reported"}},
    ]
    history, latest, complete = workspace_outcome_summary(events)
    assert len(history) == 3 and history[1]["run_status"] == "runtime_error"
    assert len(latest) == 2 and complete
    assert not workspace_outcome_summary(events[:-1])[2]


def test_pending_workspace_does_not_disappear_from_completion_summary():
    events = [
        {"event_type": "round_started", "actor_id": "HARNESS", "payload": {"round_index": 1, "workspace_count": 2}},
        {"event_type": "workspace_outcome", "actor_id": "W1-1", "payload": {"run_status": "reported"}},
    ]
    assert not workspace_outcome_summary(events)[2]
