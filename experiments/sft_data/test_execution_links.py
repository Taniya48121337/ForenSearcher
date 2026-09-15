from audit_run import link_events


def event(i, kind, payload, actor="W1-1", role="HI"):
    return {"id": i, "event_type": kind, "actor_id": actor, "actor_role": role, "payload": payload}


def test_rejected_proposal_is_not_exported_as_executed():
    rows = [event(1, "raw_model_output", {"raw_output": '{"action":"bad"}'}),
            event(2, "raw_model_output", {"raw_output": '{"action":"good"}'}),
            event(3, "model_output", {"output": {"action": "good"}}),
            event(4, "observation", {"status": "ok"})]
    linked = link_events(rows)
    assert linked[0]["status"] == "proposal_not_accepted"
    assert linked[0]["observation_event_id"] is None
    assert linked[1]["status"] == "executed"


def test_cross_workspace_events_do_not_share_observations():
    rows = [event(1, "raw_model_output", {"raw_output": '{"action":"inspect"}'}),
            event(2, "model_output", {"output": {"action": "inspect"}}),
            event(3, "observation", {}, actor="W1-2")]
    assert link_events(rows)[0]["observation_event_id"] is None


def test_host_normalization_is_preserved_as_a_difference():
    rows = [event(1, "raw_model_output", {"raw_output": '{"status":"done"}'}),
            event(2, "model_output", {"output": {"status": "COMPLETED"}}),
            event(3, "workspace_report", {})]
    row = link_events(rows)[0]
    assert row["raw_equals_accepted"] is False
    assert row["status"] == "report_committed"
