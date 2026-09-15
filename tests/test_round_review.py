import asyncio

import pytest
from jsonschema import Draft202012Validator

from foren_searcher.local_models import _decision_output_schema, _validate_gia_output
from foren_searcher.orchestration import InvestigationRuntime
from foren_searcher.store import GlobalStore
from test_framework_runtime import add_text_evidence


def report(actor, source, value):
    return {"task_id": actor, "status": "COMPLETED", "findings": [
        {"statement": f"Recorded value is {value}.", "evidence_ids": [source], "limitations": []}],
        "selected_evidence_ids": [source]}


@pytest.mark.asyncio
@pytest.mark.parametrize("review", ["finish", "finish_with_error", "wait", "arriving_report", "stalled_review", "failure"])
async def test_running_round_review_preserves_required_and_arriving_evidence(review):
    store = GlobalStore()
    source = add_text_evidence(store, "Recorded value is 12.")
    release, reported = asyncio.Event(), asyncio.Event()
    cancelled = []
    cancelled_reviews = []
    class Hi:
        async def decide(self, *, workspace_id, **kwargs):
            if workspace_id == "W1-3":
                raise RuntimeError("Evidence backend unavailable")
            if workspace_id == "W1-2":
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.append(workspace_id)
                    raise
                reported.set()
                return report(workspace_id, source.evidence_id, 13)
            return report(workspace_id, source.evidence_id, 12)

    class Gia:
        initial = True
        reviews = 0
        final_calls = 0
        async def decide(self, *, model_view, **kwargs):
            if self.initial:
                self.initial = False
                tasks = [
                    {"task": "Read the value", "materials": [source.evidence_id]},
                    {"task": "Check a potential discrepancy", "materials": [source.evidence_id]}]
                if review == "finish_with_error":
                    tasks.append({"task": "Check another source", "materials": [source.evidence_id]})
                return {"decision": "INVESTIGATE", "tasks": tasks}
            if model_view["runtime_constraints"].get("round_in_progress"):
                self.reviews += 1
                assert any(r["run_status"] == "running" for r in model_view["latest_reports"])
                if review not in {"finish", "finish_with_error"}:
                    release.set()
                if review == "wait":
                    return {"decision": "WAIT"}
                if review == "failure":
                    raise RuntimeError("Review unavailable")
                if review == "stalled_review":
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled_reviews.append(True)
                        raise
                if review == "arriving_report":
                    await reported.wait()
                    # Wait for the report transaction, not just the policy return.
                    for _ in range(100):
                        if store.workspace_task_record("W1-2")["run_status"] == "reported":
                            break
                        await asyncio.sleep(0)
                    assert store.workspace_task_record("W1-2")["run_status"] == "reported"
                return {"decision": "FINISH", "answer": f"Recorded value is 12 ({source.evidence_id})."}
            self.final_calls += 1
            assert len(model_view["latest_reports"]) == 2
            assert all(r["run_status"] == "reported" for r in model_view["latest_reports"])
            return {"decision": "FINISH", "answer": f"Reports disagree: 12 and 13 ({source.evidence_id})."}

    gia = Gia()
    async with InvestigationRuntime(store=store, gia_policy=gia, hi_policy=Hi(), max_rounds=1,
                                    report_review_delay_s=0, workspace_timeout_s=2) as runtime:
        result = await asyncio.wait_for(runtime.run(task="Verify the value", original_material_ids=(source.evidence_id,)), 2)
    assert gia.reviews == 1
    if review in {"finish", "finish_with_error"}:
        assert cancelled == ["W1-2"]
        assert gia.final_calls == 0
        assert "Recorded value is 12" in result.decision.answer
        assert store.workspace_task_record("W1-2")["run_status"] == "cancelled"
        assert store.round_records()[0]["status"] == ("completed_with_errors" if review == "finish_with_error" else "completed")
    else:
        assert not cancelled
        assert gia.final_calls == 1
        assert "Reports disagree" in result.decision.answer
    assert bool(cancelled_reviews) is (review == "stalled_review")
    assert store.round_records()[0]["status"] != "running"
    assert store.has_final_result()


def test_wait_exists_only_for_an_active_round_review():
    view = {"runtime_constraints": {"round_in_progress": True}}
    schema = _decision_output_schema({"global_view": view, "available_actions": []})
    check = Draft202012Validator(schema)
    check.validate({"decision": "WAIT"})
    check.validate({"decision": "FINISH", "answer": "Evidence is sufficient (E1)."})
    assert not check.is_valid({"decision": "INVESTIGATE", "tasks": []})
    _validate_gia_output({"decision": "WAIT"}, view, (), ())
    with pytest.raises(ValueError):
        _validate_gia_output({"decision": "WAIT"}, {}, (), ())
    with pytest.raises(ValueError):
        _validate_gia_output({"decision": "WAIT", "answer": "Skip the pending check"}, view, (), ())

def test_active_round_reminder_matches_the_available_review_decisions():
    from foren_searcher.local_models import _generation_messages, _gia_turn_prompt

    view = {'task': 'Read the chart and compare two images',
            'runtime_constraints': {'round_in_progress': True}}
    messages = _generation_messages(_gia_turn_prompt(view), {'global_view': view, 'available_actions': []})
    reminder = messages[-1]['content']
    assert '{"decision":"WAIT"}' in reminder
    assert 'INVESTIGATE' not in reminder
    assert 'FINISH' in reminder
    assert 'latest_reports.image_inputs' in reminder
    assert 'global_view.image_inputs' in reminder
    assert 'media_duration_s' in reminder
    assert '局部取样' in reminder
    assert 'Read the chart and compare two images' in messages[1]['content']


def test_normal_round_reminder_keeps_investigation_and_memory_guidance():
    from foren_searcher.local_models import _generation_messages, _gia_turn_prompt

    view = {'task': 'Read the chart and compare two images', 'runtime_constraints': {}}
    messages = _generation_messages(_gia_turn_prompt(view), {'global_view': view, 'available_actions': []})
    assert 'INVESTIGATE' in messages[-1]['content']
    assert 'READ_RECORD' in messages[-1]['content']
