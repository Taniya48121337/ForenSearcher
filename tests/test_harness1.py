"""在已配置的 Harness-1 源码上验证工作区执行与终止语义。"""

import asyncio
import os

import pytest

from foren_searcher.contracts import HiActionDecision, WorkspaceTask
from foren_searcher.harness1 import harness1_workspace_type
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceRunner
from test_framework_runtime import SequenceHi, add_text_evidence


@pytest.mark.parametrize("status", ["COMPLETED", "INCOMPLETE"])
@pytest.mark.asyncio
async def test_real_episode_recovers_invalid_tool_then_commits_direct_report(status):
    root = os.environ.get("FOREN_HARNESS_ROOT")
    if not root:
        pytest.skip("FOREN_HARNESS_ROOT is required for the original runtime integration")
    store = GlobalStore()
    evidence = add_text_evidence(store, "Documented fact")
    workspace_type = harness1_workspace_type(root)
    harness = workspace_type(task=WorkspaceTask("W1-1", "Check fact", ()), store=store)
    other = workspace_type(task=WorkspaceTask("W1-2", "Independent check", ()), store=store)
    other.start_episode(4)
    report = {"task_id":"W1-1", "status":status,
        "findings":[{"statement":"Documented fact", "evidence_ids":[evidence.evidence_id], "limitations":[]}],
        "selected_evidence_ids":[evidence.evidence_id]}
    if status == "INCOMPLETE":
        report["status_reason"] = "Independent confirmation unavailable"
    policy = SequenceHi({"W1-1":[
        {"action":"READ_RECORD", "arguments":{"record_id":evidence.evidence_id, "extra":True}, "focus_evidence_ids":[]},
        {"action":"READ_RECORD", "arguments":{"record_id":evidence.evidence_id}, "focus_evidence_ids":[]}, report]})
    result = await WorkspaceRunner(harness=harness, policy=policy, max_turns=3).run()
    assert result.status == status
    assert harness.episode.state.status == ("completed" if status == "COMPLETED" else "partial")
    assert harness.episode.state.output == store.workspace_report("W1-1")
    assert harness.episode.state.turn == 3
    assert other.episode.state.turn == 0
    assert harness.episode.state.task_state is not other.episode.state.task_state
    assert "workspace_report" not in [item["name"] for item in harness.eligible_actions()]
    assert not harness.episode.state.records[0].results[0].ok
    assert harness.episode.state.records[1].results[0].ok


@pytest.mark.asyncio
@pytest.mark.parametrize('deadline', [False, True])
async def test_original_tool_cancellation_preserves_asyncio_deadline_semantics(deadline):
    root = os.environ.get('FOREN_HARNESS_ROOT')
    if not root:
        pytest.skip('FOREN_HARNESS_ROOT required')
    workspace_type = harness1_workspace_type(root)
    harness = workspace_type(task=WorkspaceTask('W1-1', 'Check cancellation', ()), store=GlobalStore())
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def pending(arguments, workspace_id):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    harness.register_action('PENDING', schema={'type': 'object'}, handler=pending)
    decision = HiActionDecision('PENDING', {}, ())
    if deadline:
        with pytest.raises(TimeoutError) as caught:
            async with asyncio.timeout(.1):
                await harness.execute_action(decision)
        cancellation = caught.value.__cause__
    else:
        running = asyncio.create_task(harness.execute_action(decision))
        await asyncio.wait_for(entered.wait(), 5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await running
        cancellation = caught.value
    assert entered.is_set() and cleaned.is_set()
    assert type(cancellation) is asyncio.CancelledError
    assert type(cancellation.__cause__).__name__ == 'ToolBatchCancelled'
    assert not harness.memory.history
