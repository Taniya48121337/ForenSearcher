"""HI 可见的剩余时间与实际外层超时共用同一时钟。"""

import asyncio

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.local_models import _hi_turn_prompt
from foren_searcher.memory import MemoryPolicy
from foren_searcher.orchestration import RoundCoordinator
from foren_searcher.store import GlobalStore


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [5.0, None])
async def test_hi_time_budget_decreases_and_accounts_for_model_waiting(timeout):
    store = GlobalStore()
    evidence = store.record_evidence(content="A recorded fact", label="Source", source_ref=None,
        locator="paragraph:1", acquired_by="TEST/source", discovered_by="W0")
    task = WorkspaceTask("W1-1", "Read the source", (evidence.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    limits = []
    waits = []

    class Policy:
        async def decide(self, *, workspace_id, model_view, actions):
            limits.append(model_view["runtime_constraints"])
            if len(limits) == 1:
                started = asyncio.get_running_loop().time()
                await asyncio.sleep(0.02)
                waits.append(asyncio.get_running_loop().time() - started)
                return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id},
                        "focus_evidence_ids": []}
            return {"task_id": workspace_id, "status": "COMPLETED", "findings": [{
                "statement": "A recorded fact", "evidence_ids": [evidence.evidence_id], "limitations": []}],
                "selected_evidence_ids": [evidence.evidence_id]}

    outcomes = await RoundCoordinator(store=store, hi_policy=Policy(), search_service=None,
        inspect_service=None, memory_policy=MemoryPolicy(), workspace_timeout_s=timeout).run(round_index=1, tasks=(task,))
    assert outcomes[0].run_status == "reported"
    if timeout is None:
        assert all("remaining_seconds" not in limit and "last_model_call_seconds" not in limit for limit in limits)
    else:
        assert 0 < limits[1]["remaining_seconds"] < limits[0]["remaining_seconds"] <= timeout
        assert "last_model_call_seconds" not in limits[0]
        assert limits[1]["last_model_call_seconds"] >= waits[0] > 0
        assert limits[0]["remaining_seconds"] - limits[1]["remaining_seconds"] >= limits[1]["last_model_call_seconds"]
    saved = [e["payload"]["model_view"]["runtime_constraints"] for e in store.trajectory_events(
        actor_role="HI", actor_id=task.task_id, event_type="model_input")]
    assert saved == limits


@pytest.mark.asyncio
async def test_time_budget_information_does_not_extend_the_existing_timeout():
    store = GlobalStore()
    task = WorkspaceTask("W1-1", "Investigate", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    cancelled = asyncio.Event()

    class Policy:
        async def decide(self, *, workspace_id, model_view, actions):
            assert 0 < model_view["runtime_constraints"]["remaining_seconds"] <= 0.05
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    outcomes = await RoundCoordinator(store=store, hi_policy=Policy(), search_service=None,
        inspect_service=None, memory_policy=MemoryPolicy(), workspace_timeout_s=0.05).run(round_index=1, tasks=(task,))
    assert outcomes[0].run_status == "timed_out"
    assert cancelled.is_set()
    assert not store.workspace_report_records()


@pytest.mark.parametrize("vision", [False, True])
def test_hi_time_guidance_is_only_present_for_a_configured_deadline(vision):
    view = {"direct_visual_input": vision, "runtime_constraints": {"remaining_turns": 10}}
    assert "last_model_call_seconds" not in _hi_turn_prompt(view, ())
    view["runtime_constraints"].update(remaining_seconds=120, last_model_call_seconds=47)
    prompt = _hi_turn_prompt(view, ())
    assert "last_model_call_seconds" in prompt and "含模型排队和工具执行" in prompt
