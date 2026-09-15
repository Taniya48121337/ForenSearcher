"""主框架编排与 Workspace 运行时测试。

用可编程假策略覆盖多轮分派、并发、超时、取消、恢复、记忆视图和动作审计；断言关注
控制流与 Store 最终状态，而非模型文字质量。
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from foren_searcher.contracts import (
    Finding,
    FindingDraft,
    HiActionDecision,
    WorkspaceReport,
    WorkspaceReportDraft,
    WorkspaceTask,
)
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.memory import EvidenceViewBuilder, GlobalViewBuilder, MemoryPolicy, RecordAccess
from foren_searcher.orchestration import (
    GiaPolicyTimeout,
    GiaRunner,
    GiaTurnLimit,
    InitialMedia,
    InvestigationRuntime,
    RoundCoordinator,
    RuntimeClosedError,
)
from foren_searcher.search.contracts import MaterialResult, SearchResponse, SearchResult
from foren_searcher.store import ArtifactIntegrityError, GlobalStore
from foren_searcher.workspace import ToolObservation, WorkspaceHarness, WorkspaceRunner, WorkspaceTurnLimit
from foren_searcher.policies import PolicyOutputError

from inspect_helpers import image_bytes, ingest


def add_text_evidence(store: GlobalStore, text: str, *, workspace: str = "W0"):
    """测试辅助函数 add_text_evidence；为用例提供可重复的输入或可控行为。"""

    return store.record_evidence(
        content=text,
        label=text,
        source_ref="C0",
        locator="paragraph:1",
        acquired_by="TEST/source",
        discovered_by=workspace,
    )


@pytest.mark.asyncio
async def test_gia_can_finish_on_its_last_turn_after_a_rejected_decision():
    store = GlobalStore()
    evidence = add_text_evidence(store, "A saved source sentence")
    policy = MemoryPolicy()
    constraints = []

    class FinishWithinBudget:
        async def decide(self, *, model_view, history, actions):
            limits = model_view["runtime_constraints"]
            constraints.append(limits)
            if limits["turn_index"] == 1:
                return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id}}
            if limits["turn_index"] == 2:
                raise PolicyOutputError("invalid response")
            assert history[-1]["observation"]["action"] == "HARNESS_REJECTION"
            assert limits["remaining_turns"] == 0
            return {"decision": "FINISH", "answer": "A saved source sentence E1"}

    view = GlobalViewBuilder(store, policy).build(task="Read the source", original_material_ids=(), latest_round_index=None)
    decision = await GiaRunner(store=store, policy=FinishWithinBudget(), memory_policy=policy, max_turns=3).run(
        view, round_index=2, can_create_round=False)
    assert decision.decision == "FINISH"
    assert [c["remaining_turns"] for c in constraints] == [2, 1, 0]
    assert all(c["max_turns"] == 3 and c["can_create_round"] is False for c in constraints)
    recorded = [e["payload"]["model_view"]["runtime_constraints"] for e in store.trajectory_events()
                if e["event_type"] == "model_input"]
    assert recorded == constraints


@pytest.mark.asyncio
async def test_hi_can_report_on_its_last_turn_after_a_rejected_decision():
    store = GlobalStore()
    evidence = add_text_evidence(store, "A saved source sentence")
    task = WorkspaceTask("W1-1", "Read the source", (evidence.evidence_id,))
    harness = WorkspaceHarness(store=store, task=task)
    constraints = []

    class FinishWithinBudget:
        async def decide(self, *, workspace_id, model_view, actions):
            limits = model_view["runtime_constraints"]
            constraints.append(limits)
            if limits["turn_index"] == 1:
                return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id}, "focus_evidence_ids": []}
            if limits["turn_index"] == 2:
                raise PolicyOutputError("invalid response")
            assert model_view["latest_observation"]["action"] == "HARNESS_REJECTION"
            assert limits["remaining_turns"] == 0
            return {"task_id": workspace_id, "status": "COMPLETED", "findings": [{
                "statement": "A saved source sentence", "evidence_ids": [evidence.evidence_id], "limitations": []}],
                "selected_evidence_ids": [evidence.evidence_id]}

    report = await WorkspaceRunner(harness=harness, policy=FinishWithinBudget(), max_turns=3).run()
    assert report.status == "COMPLETED"
    assert [c["remaining_turns"] for c in constraints] == [2, 1, 0]
    recorded = [e["payload"]["model_view"]["runtime_constraints"] for e in store.trajectory_events()
                if e["event_type"] == "model_input"]
    assert recorded == constraints


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["GIA", "HI"])
async def test_turn_feedback_does_not_extend_the_existing_limit(role):
    store = GlobalStore()
    evidence = add_text_evidence(store, "A saved source sentence")

    class AlwaysRead:
        calls = 0

        async def decide(self, **kwargs):
            self.calls += 1
            assert kwargs["model_view"]["runtime_constraints"]["remaining_turns"] == 2 - self.calls
            output = {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id}}
            if role == "HI":
                output["focus_evidence_ids"] = []
            return output

    policy = AlwaysRead()
    if role == "GIA":
        view = GlobalViewBuilder(store, MemoryPolicy()).build(task="Read", original_material_ids=(), latest_round_index=None)
        with pytest.raises(GiaTurnLimit):
            await GiaRunner(store=store, policy=policy, memory_policy=MemoryPolicy(), max_turns=2).run(view, round_index=2)
    else:
        harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Read", (evidence.evidence_id,)))
        with pytest.raises(WorkspaceTurnLimit):
            await WorkspaceRunner(harness=harness, policy=policy, max_turns=2).run()
    assert policy.calls == 2


@pytest.mark.asyncio
async def test_gia_can_compare_three_read_pages_within_the_total_context_budget():
    import json
    store = GlobalStore()
    policy = MemoryPolicy(max_model_view_chars=40000, max_read_chars=6000)
    evidence = [add_text_evidence(store, f"Source {i}: " + "x" * 6500) for i in range(3)]

    class CompareThree:
        calls = 0

        async def decide(self, *, model_view, history, actions):
            assert len(json.dumps(model_view, ensure_ascii=False)) + len(json.dumps(history, ensure_ascii=False)) <= 40000
            self.calls += 1
            if self.calls <= 3:
                return {"action": "READ_RECORD", "arguments": {"record_id": evidence[self.calls - 1].evidence_id}}
            pages = {h["observation"]["record"]["evidence_id"]: h["observation"]["record"]["content"] for h in history}
            assert set(pages) == {e.evidence_id for e in evidence}
            assert all(len(text) == 6000 for text in pages.values())
            return {"decision": "FINISH", "answer": "Three original pages can be compared together: E1 E2 E3."}

    view = GlobalViewBuilder(store, policy).build(task="Compare three source pages", original_material_ids=(), latest_round_index=None)
    decision = await GiaRunner(store=store, policy=CompareThree(), memory_policy=policy).run(view, round_index=2)
    assert decision.decision == "FINISH"


@pytest.mark.asyncio
async def test_repeated_gia_page_read_keeps_latest_content_without_duplicate_context():
    store = GlobalStore()
    evidence = add_text_evidence(store, "Original source sentence")
    policy = MemoryPolicy()

    class Reread:
        calls = 0

        async def decide(self, *, model_view, history, actions):
            self.calls += 1
            if self.calls <= 2:
                return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id, "offset": 0}}
            assert len(history) == 1
            assert history[0]["turn_index"] == 2
            assert "Original source sentence" in history[0]["observation"]["record"]["content"]
            return {"decision": "FINISH", "answer": "Original source sentence E1"}

    view = GlobalViewBuilder(store, policy).build(task="Read source", original_material_ids=(), latest_round_index=None)
    await GiaRunner(store=store, policy=Reread(), memory_policy=policy).run(view, round_index=2)
    reads = [event for event in store.trajectory_events() if event["event_type"] == "model_output"]
    assert len(reads) == 3


@pytest.mark.asyncio
async def test_model_history_retains_actual_focus_at_each_executed_action():
    import json
    from foren_searcher.contracts import parse_hi_decision
    from foren_searcher.local_models import _generation_messages
    store = GlobalStore()
    first = add_text_evidence(store, "First source")
    second = add_text_evidence(store, "Second source")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Compare", (first.evidence_id, second.evidence_id)), store=store)
    for evidence in (first, second):
        await harness.execute_action(HiActionDecision(action="READ_RECORD",
            arguments={"record_id": evidence.evidence_id}, focus_evidence_ids=(evidence.evidence_id,)))
    view = harness.model_view()
    messages = _generation_messages("system", {"workspace_id": "W1-1", "model_view": view,
        "available_actions": list(harness.eligible_actions())})
    decisions = [parse_hi_decision(json.loads(m["content"])) for m in messages if m["role"] == "assistant"]
    assert [d.focus_evidence_ids for d in decisions] == [(first.evidence_id,), (second.evidence_id,)]
    assert view["focus_evidence_ids"] == [second.evidence_id]


@pytest.mark.asyncio
async def test_read_pages_reach_hi_intact_with_demo_observation_budget(tmp_path):
    import json
    from foren_searcher.local_models import _generation_messages

    store = GlobalStore(tmp_path / "records.sqlite3", tmp_path / "artifacts")
    text = ('Source says "quoted words". 中文。 Path C:\\new\\notes.txt\n' * 250) + 'END OF SOURCE'
    evidence = store.record_evidence(content=text, label="Long source", source_ref="C0",
        locator="full text", acquired_by="TEST/source", discovered_by="W1-1")
    policy = MemoryPolicy(max_model_view_chars=40_000, max_history_entries=16,
        max_observation_chars=10_000)
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Read source", (evidence.evidence_id,)),
        store=store, memory_policy=policy)
    pages = []
    offset = 0
    while True:
        observation = await harness.execute_action(HiActionDecision(action="READ_RECORD",
            arguments={"record_id": evidence.evidence_id, "offset": offset}, focus_evidence_ids=()))
        view = harness.model_view()
        messages = _generation_messages("system", {"workspace_id": "W1-1", "model_view": view,
            "available_actions": list(harness.eligible_actions())})
        shown = json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
        assert not shown.get("truncated")
        assert shown["details"] == observation.details
        page = shown["details"]
        assert page["content"] == text[offset:offset + policy.max_read_chars]
        assert page["content_offset"] == offset
        pages.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert ''.join(pages) == text
    assert all(not item["observation"].get("truncated")
        for item in view["history"] if item["observation"] is not None)
    assert store.evidence(evidence.evidence_id).content == text


@pytest.mark.asyncio
@pytest.mark.parametrize('repeated_reads', [0, 8, 24])
async def test_large_context_demo_keeps_all_pages_for_a_whole_source_comparison(repeated_reads):
    import json
    from foren_searcher.demo import _demo_memory_policy
    from foren_searcher.local_models import _generation_messages

    store = GlobalStore()
    text = ''.join((f'Chapter {index}: "original words" 中文路径 C:\\new\\notes.txt\n' * 120)[:6000].ljust(6000, 'x') for index in range(7)) + 'END OF SOURCE'
    evidence = store.record_evidence(content=text, label='Long source', source_ref='C0',
        locator='full text', acquired_by='TEST/source', discovered_by='W1-1')
    policy = _demo_memory_policy(131072)
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Compare the beginning and end of the entire source',
        (evidence.evidence_id,)), store=store, memory_policy=policy)
    offset = 0
    read_count = 0
    while True:
        observation = await harness.execute_action(HiActionDecision(action='READ_RECORD',
            arguments={'record_id': evidence.evidence_id, 'offset': offset}, focus_evidence_ids=()))
        read_count += 1
        if observation.details['next_offset'] is None:
            break
        offset = observation.details['next_offset']
    for _ in range(repeated_reads):
        await harness.execute_action(HiActionDecision(action='READ_RECORD',
            arguments={'record_id': evidence.evidence_id, 'offset': 0}, focus_evidence_ids=()))
    view = harness.model_view()
    messages = _generation_messages('system', {'workspace_id': 'W1-1', 'model_view': view,
        'available_actions': list(harness.eligible_actions())})
    pages = {}
    for message in messages:
        if message['role'] == 'tool':
            details = json.loads(message['content'])['details']
            pages[details['content_offset']] = details['content']
    assert ''.join(pages[key] for key in sorted(pages)) == text
    assert len(json.dumps(view, ensure_ascii=False)) <= policy.max_model_view_chars
    assert store.evidence(evidence.evidence_id).content == text
    assert len(harness.memory.history) == read_count + repeated_reads
    assert view.get('context_truncated', False) == bool(repeated_reads)


@pytest.mark.parametrize('variant', ['changed_content', 'changed_focus', 'candidate', 'query', 'failure'])
def test_history_preserves_distinct_observations_and_non_evidence_reads(variant):
    import copy
    from foren_searcher.workspace import WorkspaceHistoryEntry, WorkspaceMemory

    action = 'QUERY_MEMORY' if variant == 'query' else 'READ_RECORD'
    request = {'record_id': 'R1' if variant == 'candidate' else 'E1'}
    observation = {'action': action, 'items': ['record'], 'details': {
        'record_type': 'candidate' if variant == 'candidate' else 'evidence',
        'evidence_id': 'E1', 'content': 'original text'}}
    if variant == 'failure':
        observation = {'action': 'HARNESS_REJECTION', 'items': ['unavailable'], 'details': {}}
    first = WorkspaceHistoryEntry(action, request, {}, observation, ('E1',))
    second_observation = copy.deepcopy(observation)
    if variant == 'changed_content':
        second_observation['details']['content'] = 'different text'
    second = WorkspaceHistoryEntry(action, copy.deepcopy(request), {}, second_observation,
        ('E2',) if variant == 'changed_focus' else ('E1',))
    memory = WorkspaceMemory('W1-1', 'Compare evidence', {}, history=[first, second],
        latest_observation=second_observation)
    view = memory.to_model_view()
    assert len(view['history']) == 2
    assert not view.get('context_truncated', False)
    assert memory.history == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize('entry_limit,character_limit', [(3, 80000), (16, 20000)])
@pytest.mark.parametrize('latest_action', ['QUERY_MEMORY', 'HARNESS_REJECTION'])
async def test_focused_source_pages_survive_other_source_reads_and_latest_feedback(entry_limit, character_limit, latest_action):
    import copy
    import json
    from foren_searcher.local_models import _generation_messages

    store = GlobalStore()
    body = 'Beginning of the source. '.ljust(6000, 'a') + 'End of the source. '.ljust(6000, 'b')
    primary = store.record_evidence(content=body, label='Selected source', source_ref='C0',
        locator='full text', acquired_by='TEST/source', discovered_by='W1-1')
    others = [store.record_evidence(content=text.ljust(6000, 'x'), label=text, source_ref='C0',
        locator='full text', acquired_by='TEST/source', discovered_by='W1-1')
        for text in ('Another source.', 'Third source.')]
    policy = MemoryPolicy(max_history_entries=entry_limit, max_model_view_chars=character_limit,
                          max_observation_chars=10000)
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Compare the beginning and end of the selected source', ()),
                               store=store, memory_policy=policy)
    for evidence_id, offset in [(primary.evidence_id, 0), (primary.evidence_id, 6000),
                                *[(item.evidence_id, 0) for item in others]]:
        await harness.execute_action(HiActionDecision(action='READ_RECORD',
            arguments={'record_id': evidence_id, 'offset': offset}, focus_evidence_ids=(primary.evidence_id,)))
    if latest_action == 'QUERY_MEMORY':
        await harness.execute_action(HiActionDecision(action='QUERY_MEMORY', arguments={'query': 'source'},
                                                     focus_evidence_ids=(primary.evidence_id,)))
    else:
        with pytest.raises(ValueError, match='query must be non-empty') as error:
            await harness.execute_action(HiActionDecision(action='QUERY_MEMORY', arguments={'query': ''},
                                                         focus_evidence_ids=(primary.evidence_id,)))
        harness.reject(str(error.value))
    raw_history = copy.deepcopy(harness.memory.history)
    view = harness.model_view()
    messages = _generation_messages('system', {'workspace_id': 'W1-1', 'model_view': view,
        'available_actions': list(harness.eligible_actions())})
    pages = {}
    for message in messages:
        if message['role'] == 'tool':
            details = json.loads(message['content']).get('details', {})
            if details.get('evidence_id') == primary.evidence_id:
                pages[details['content_offset']] = details['content']
    assert ''.join(pages[key] for key in sorted(pages)) == body
    assert view['history'][-1]['action'] == latest_action
    feedback = json.loads(messages[-2]['content'])
    if latest_action == 'HARNESS_REJECTION':
        feedback = feedback['harness_event']['observation']
    assert feedback['action'] == latest_action
    assert view['latest_observation'] == harness.memory.latest_observation
    assert len(view['history']) <= entry_limit
    assert len(json.dumps(view, ensure_ascii=False)) <= character_limit
    assert view['context_truncated']
    assert harness.memory.history == raw_history
    assert store.evidence(primary.evidence_id).content == body


@pytest.mark.asyncio
async def test_memory_query_preserves_structured_pages_and_empty_results():
    store = GlobalStore()
    first = add_text_evidence(store, "source one")
    second = add_text_evidence(store, "source two")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Find sources", ()), store=store,
        memory_policy=MemoryPolicy(query_page_size=1))
    pages = []
    for cursor in (0, 1, 2):
        observation = await harness.execute_action(HiActionDecision(action="QUERY_MEMORY",
            arguments={"query": "source", "record_types": ["evidence"], "cursor": cursor},
            focus_evidence_ids=()))
        assert harness.model_view()["latest_observation"]["details"] == observation.details
        pages.append(observation.details)
    assert {page["results"][0]["evidence_id"] for page in pages[:2]} == {first.evidence_id, second.evidence_id}
    assert [page["next_cursor"] for page in pages] == [1, None, None]
    assert pages[-1]["results"] == []


@pytest.mark.parametrize("action", ["READ_RECORD", "QUERY_MEMORY"])
def test_restoring_legacy_memory_observations_keeps_records_without_escaped_duplicate(action, tmp_path):
    import json

    store = GlobalStore(tmp_path / "records.sqlite3", tmp_path / "artifacts")
    evidence = store.record_evidence(content="A long original sentence. " * 400, label="Original source",
        source_ref="C0", locator="full text", acquired_by="TEST/source", discovered_by="W1-1")
    task = WorkspaceTask("W1-1", "Read source", (evidence.evidence_id,))
    policy = MemoryPolicy(max_model_view_chars=40_000, max_observation_chars=10_000)
    harness = WorkspaceHarness(task=task, store=store, memory_policy=policy)
    if action == "READ_RECORD":
        request = {"record_id": evidence.evidence_id, "offset": 0}
        details = harness.record_access.read(evidence.evidence_id)
    else:
        request = {"query": "source"}
        details = harness.record_access.query("source").to_dict()
    legacy = {"action": action, "items": [json.dumps(details, ensure_ascii=False)], "details": details}
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="model_output",
        payload={"output": {"action": action, "arguments": request, "focus_evidence_ids": []}})
    store.record_event(actor_role="HI", actor_id=task.task_id, event_type="observation",
        payload={"observation": legacy})
    original_events = store.trajectory_events()
    reopened = GlobalStore(tmp_path / "records.sqlite3", tmp_path / "artifacts")
    restored = WorkspaceHarness(task=task, store=reopened, memory_policy=policy)
    restored.restore_working_memory()
    shown = restored.model_view()["latest_observation"]
    assert not shown.get("truncated")
    assert shown["details"] == details
    assert json.dumps(details, ensure_ascii=False) not in shown["items"]
    assert reopened.trajectory_events() == original_events


@pytest.mark.asyncio
@pytest.mark.parametrize('history_limit', [2, 3])
async def test_history_entry_limit_marks_omitted_observations(history_limit):
    store = GlobalStore()
    evidence = [add_text_evidence(store, text) for text in ('first', 'second', 'third')]
    harness = WorkspaceHarness(task=WorkspaceTask('W1-1', 'Read sources', ()), store=store,
        memory_policy=MemoryPolicy(max_history_entries=history_limit, max_model_view_chars=180000))
    for item in evidence:
        await harness.execute_action(HiActionDecision(action='READ_RECORD',
            arguments={'record_id': item.evidence_id}, focus_evidence_ids=(item.evidence_id,)))
    view = harness.model_view()
    assert len(view['history']) == history_limit
    assert view.get('context_truncated', False) == (history_limit < len(evidence))
    assert len(harness.memory.history) == 3
    assert harness.record_access.read(evidence[0].evidence_id)['content'] == 'first'


@pytest.mark.parametrize('history_limit', [2, 3])
def test_global_history_limit_marks_omitted_records_and_preserves_access(history_limit):
    store = GlobalStore()
    evidence = [add_text_evidence(store, text, workspace='W1-1') for text in ('oldest source', 'middle source', 'latest source')]
    policy = MemoryPolicy(max_history_entries=history_limit, max_model_view_chars=180000)
    view = GlobalViewBuilder(store, policy).build(task='Compare sources', original_material_ids=(), latest_round_index=2).to_dict()
    assert len(view['history_index']) == history_limit
    assert view.get('context_truncated', False) == (history_limit < len(evidence))
    access = RecordAccess(store=store, role='GIA', actor_id='GIA', round_index=3, policy=policy)
    matches = access.query('oldest', record_types=('evidence',))
    assert matches.results[0]['evidence_id'] == evidence[0].evidence_id
    assert access.read(evidence[0].evidence_id)['content'] == 'oldest source'
    assert len(store.evidence_records()) == 3


@pytest.mark.parametrize("run_status", ["runtime_error", "timed_out", "cancelled"])
def test_old_unreported_workspace_remains_searchable_after_restart_and_history_truncation(tmp_path, run_status):
    import json

    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    error = {"type": "SourceUnavailable", "message": "Archive unavailable. " * 250 + "zirconium_failure"}
    for n in (1, 2, 3):
        store.create_round(n, gia_input={}, gia_decision={})
        text = "Source context. " * 200 + "Original scope." if n == 1 else f"Archive source {n}"
        store.add_workspace_tasks(n, (WorkspaceTask(f"W{n}-1", text, ()),))
        store.set_workspace_run_status(f"W{n}-1", run_status, error=error if n == 1 else None)
    store = GlobalStore(store.db_path, store.artifact_root)
    complete_view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="Verify sources", original_material_ids=(), latest_round_index=3)
    assert [item["task_id"] for item in complete_view.history_index] == ["W1-1", "W2-1"]
    policy = MemoryPolicy(max_history_entries=1, max_read_chars=512)
    view = GlobalViewBuilder(store, policy).build(task="Verify sources", original_material_ids=(), latest_round_index=3)
    assert view.context_truncated
    assert [item["task_id"] for item in view.history_index] == ["W2-1"]
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=4, policy=policy)
    match = access.query("zirconium_failure", record_types=("workspace_task",)).results[0]
    assert match["task_id"] == "W1-1"
    assert match["run_status"] == run_status
    assert match["task_context_truncated"] and match["error_truncated"]
    assert "zirconium_failure" not in match["error_excerpt"]
    assert access.query("zirconium_failure").results[0]["task_id"] == "W1-1"
    pages, offset = [], 0
    while True:
        page = access.read(match["task_id"], offset=offset)
        pages.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    recovered = json.loads("".join(pages))
    assert recovered["error"] == error
    assert recovered["round_index"] == 1
    assert recovered["task"].endswith("Original scope.")
    sibling = RecordAccess(store=store, role="HI", actor_id="W3-1", round_index=3, policy=policy,
                           local_reference_ids={"W1-1"})
    with pytest.raises(PermissionError):
        sibling.query("zirconium_failure", record_types=("workspace_task",))
    assert sibling.query("zirconium_failure").results == ()
    with pytest.raises(PermissionError):
        sibling.read("W1-1", offset=512)
    assert not store.workspace_report_records()
    assert not store.evidence_records()


def test_workspace_task_queries_paginate_and_reported_tasks_have_one_search_result():
    store = GlobalStore()
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (WorkspaceTask("W1-1", "Archive source one", ()),
                                 WorkspaceTask("W1-2", "Archive source two", ())))
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=2,
                          policy=MemoryPolicy(query_page_size=1))
    first = access.query("Archive source", record_types=("workspace_task",))
    second = access.query("Archive source", record_types=("workspace_task",), cursor=first.next_cursor)
    assert {first.results[0]["task_id"], second.results[0]["task_id"]} == {"W1-1", "W1-2"}
    assert second.next_cursor is None
    store.add_workspace_report(WorkspaceReport("W1-1", "INCOMPLETE", (), (), status_reason="Source unavailable"))
    tasks = access.query("Archive source", record_types=("workspace_task",))
    assert [item["task_id"] for item in tasks.results] == ["W1-2"]
    assert tasks.next_cursor is None
    all_records = [*access.query("Archive source").results, *access.query("Archive source", cursor=1).results]
    assert {item["record_type"] for item in all_records} == {"report", "workspace_task"}
    assert {item["task_id"] for item in all_records} == {"W1-1", "W1-2"}
    assert len(all_records) == 2


@pytest.mark.asyncio
async def test_gia_recovers_a_failed_task_through_memory_actions_before_reassigning():
    class RecoverFailedTask:
        async def decide(self, *, model_view, history, actions):
            if not history:
                query_schema = next(item["schema"] for item in actions if item["name"] == "QUERY_MEMORY")
                assert "workspace_task" in query_schema["properties"]["record_types"]["items"]["enum"]
                return {"action": "QUERY_MEMORY", "arguments": {"query": "SourceUnavailable", "record_types": ["workspace_task"]}}
            if len(history) == 1:
                record = history[-1]["observation"]["results"][0]
                assert record["task_id"] == "W1-1"
                return {"action": "READ_RECORD", "arguments": {"record_id": record["task_id"]}}
            record = history[-1]["observation"]["record"]
            assert record["error"]["message"] == "Archive unavailable"
            assert record["task"] == "Verify the primary archive"
            return {"decision": "INVESTIGATE", "tasks": [{"task": "Find another original source for the same claim", "materials": []}]}

    store = GlobalStore()
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (WorkspaceTask("W1-1", "Verify the primary archive", ()),))
    store.set_workspace_run_status("W1-1", "runtime_error", error={"type": "SourceUnavailable", "message": "Archive unavailable"})
    policy = MemoryPolicy()
    view = GlobalViewBuilder(store, policy).build(task="Verify the claim", original_material_ids=(), latest_round_index=2)
    decision = await GiaRunner(store=store, policy=RecoverFailedTask(), memory_policy=policy).run(view, round_index=3)
    assert decision.decision == "INVESTIGATE"
    assert len(store.workspace_task_records()) == 1
    assert not store.workspace_report_records()
    assert {row["reference_id"] for row in store.record_reads(actor_role="GIA")} == {"W1-1"}


class SequenceHi:
    """测试辅助类 SequenceHi；用可控状态替代真实外部依赖。"""

    def __init__(self, outputs: dict[str, list[dict[str, Any]]]) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.outputs = outputs

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        return self.outputs[workspace_id].pop(0)


@pytest.mark.asyncio
async def test_hi_policy_output_error_can_recover_by_reading_missing_evidence():
    store = GlobalStore()
    evidence = add_text_evidence(store, "Only this saved source supports the finding.")
    task = WorkspaceTask("W1-1", "Read the source before reporting", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    class RecoveringHi:
        calls = 0
        async def decide(self, *, workspace_id, model_view, actions):
            self.calls += 1
            if self.calls == 1:
                raise PolicyOutputError("READ_RECORD the unread evidence first")
            if self.calls == 2:
                assert model_view["latest_observation"]["action"] == "HARNESS_REJECTION"
                assert not store.workspace_report_records()
                return {"action":"READ_RECORD", "arguments":{"record_id":evidence.evidence_id}, "focus_evidence_ids":[]}
            assert evidence.evidence_id in model_view["content_observed_evidence_ids"]
            return {"task_id":workspace_id, "status":"COMPLETED", "findings":[{
                "statement":evidence.content, "evidence_ids":[evidence.evidence_id], "limitations":[]}],
                "selected_evidence_ids":[evidence.evidence_id]}
    harness = WorkspaceHarness(task=task, store=store, round_index=1)
    report = await WorkspaceRunner(harness=harness, policy=RecoveringHi(), max_turns=3).run()
    assert report.findings[0].statement == evidence.content


@pytest.mark.asyncio
async def test_policy_output_errors_still_obey_workspace_turn_limit():
    from foren_searcher.workspace import WorkspaceTurnLimit
    store = GlobalStore()
    task = WorkspaceTask("W1-1", "Investigate", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    class InvalidHi:
        async def decide(self, **kwargs):
            raise PolicyOutputError("Invalid output")
    with pytest.raises(WorkspaceTurnLimit):
        await WorkspaceRunner(harness=WorkspaceHarness(task=task, store=store, round_index=1),
                              policy=InvalidHi(), max_turns=2).run()
    assert not store.workspace_report_records()


@pytest.mark.asyncio
async def test_inspect_adapter_preserves_public_status_result_feedback_and_limitations():
    """验证 inspect adapter preserves public status result feedback and limitations 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    image = ingest(
        store,
        image_bytes(),
        source="I0",
        filename="input.png",
        media_type="image/png",
        modality="image",
    )
    service = create_inspect_service(store=store)
    harness = WorkspaceHarness(
        task=WorkspaceTask("W1-1", "检查图像", (image.evidence_id,)),
        store=store,
        inspect_service=service,
    )
    try:
        observation = await harness.execute_action(
            HiActionDecision(
                action="INSPECT",
                arguments={
                    "targets": [image.evidence_id],
                    "request": {
                        "operation": "visual_qa",
                        "params": {"question": "图像尺寸是什么？"},
                    },
                },
                focus_evidence_ids=(image.evidence_id,),
            )
        )
    finally:
        await service.aclose()
    assert observation.details["operation"] == "visual_qa"
    assert observation.details["status"] == "ready"
    assert "limitations" in observation.details["result"]
    assert "feedback" in observation.details


@pytest.mark.asyncio
async def test_workspace_runner_rejects_unknown_action_then_accepts_direct_report():
    """验证 workspace runner rejects unknown action then accepts direct report 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "待核查说法"})
    store.create_round(1, gia_input={}, gia_decision={})
    task = WorkspaceTask("W1-1", "核查说法", ("C0",))
    store.add_workspace_tasks(1, (task,))
    policy = SequenceHi(
        {
            "W1-1": [
                {"action": "SPAWN", "arguments": {}, "focus_evidence_ids": []},
                {
                    "task_id": "W1-1",
                    "status": "INCOMPLETE",
                    "findings": [],
                    "selected_evidence_ids": [],
                    "status_reason": "没有取得可追溯材料。",
                },
            ]
        }
    )
    harness = WorkspaceHarness(task=task, store=store, round_index=1)
    report = await WorkspaceRunner(harness=harness, policy=policy).run()
    assert report.status == "INCOMPLETE"
    assert harness.memory.history[0].action == "HARNESS_REJECTION"
    assert store.workspace_task_record("W1-1")["run_status"] == "reported"


@pytest.mark.parametrize("source,statement", [
    ("The speech occurred on January 20, 1961.", "演讲日期为1961年1月20日。"),
    ("The two segments last 2 seconds and 3 seconds.", "两段时长相加为5秒（2+3）。"),
    ("The value was 10,000 metres.", "距离为10000米。"),
])
def test_workspace_report_accepts_translated_or_derived_numbers(source, statement):
    store = GlobalStore()
    evidence = add_text_evidence(store, source)
    task = WorkspaceTask("W1-1", "核验日期或数量", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    harness = WorkspaceHarness(task=task, store=store, round_index=1)
    report = harness.finish(WorkspaceReportDraft(task_id="W1-1", status="COMPLETED",
        findings=(FindingDraft(statement=statement, evidence_ids=(evidence.evidence_id,), limitations=()),),
        selected_evidence_ids=(evidence.evidence_id,)))
    assert report.findings[0].statement == statement


def test_role_access_keeps_other_workspace_findings_away_from_hi():
    """验证 role access keeps other workspace findings away from hi 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "待核查说法"})
    evidence = add_text_evidence(store, "页面显示日期为2022年。", workspace="W1-1")
    finding = Finding(
        finding_id="F1",
        workspace_id="W1-1",
        statement="材料至少在2022年已经出现。",
        evidence_ids=(evidence.evidence_id,),
        limitations=(),
    )
    store.add_workspace_report(
        WorkspaceReport(
            task_id="W1-1",
            status="COMPLETED",
            findings=(finding,),
            selected_evidence_ids=(evidence.evidence_id,),
        )
    )
    hi = RecordAccess(
        store=store,
        role="HI",
        actor_id="W1-2",
        round_index=1,
        policy=MemoryPolicy(),
    )
    gia = RecordAccess(
        store=store,
        role="GIA",
        actor_id="GIA",
        round_index=2,
        policy=MemoryPolicy(),
    )
    with pytest.raises(PermissionError):
        hi.read("F1")
    assert gia.read("F1")["workspace_id"] == "W1-1"


@pytest.mark.parametrize("record_id", ["F1", "W1-1"])
def test_long_judgment_records_preserve_tail_conditions_through_pagination(record_id):
    import json

    store = GlobalStore()
    evidence = add_text_evidence(store, "The source supports a limited observation.")
    finding = Finding("F1", "W1-1", "Observation. " * 500 + "Critical tail qualification.",
        (evidence.evidence_id,), ("Scope. " * 200 + "Final scope condition.",)
        + tuple(f"Condition {index}" for index in range(16)) + ("Final list condition.",))
    report = WorkspaceReport("W1-1", "INCOMPLETE", (finding,), (evidence.evidence_id,),
        status_reason="Unresolved requirement. " * 120 + "Final missing requirement.")
    store.add_workspace_report(report)
    policy = MemoryPolicy(max_read_chars=512)
    for role, actor_id in (("GIA", "GIA"), ("HI", "W1-1")):
        access = RecordAccess(store=store, role=role, actor_id=actor_id,
            round_index=2, policy=policy)
        pages, offset = [], 0
        while True:
            page = access.read(record_id, offset=offset)
            assert page["content_offset"] == offset
            assert len(page["content"]) <= policy.max_read_chars
            pages.append(page["content"])
            following = page["next_offset"]
            if following is None:
                break
            assert following > offset
            offset = following
        decoded = json.loads("".join(pages))
        detail = decoded if record_id == "F1" else decoded["findings"][0]
        assert detail["statement"].endswith("Critical tail qualification.")
        assert detail["limitations"][0].endswith("Final scope condition.")
        assert detail["limitations"][-1] == "Final list condition."
        assert detail["evidence_ids"] == [evidence.evidence_id]
        if record_id != "F1":
            assert decoded["status_reason"].endswith("Final missing requirement.")
        assert access.read(record_id, offset=sum(map(len, pages)))["content"] == ""
    sibling = RecordAccess(store=store, role="HI", actor_id="W1-2", round_index=2, policy=policy)
    for offset in (0, 4096):
        with pytest.raises(PermissionError):
            sibling.read(record_id, offset=offset)


def test_judgment_within_read_budget_keeps_complete_structured_fields():
    store = GlobalStore()
    evidence = add_text_evidence(store, "source")
    finding = Finding("F1", "W1-1", "S" * 4000 + " statement tail", (evidence.evidence_id,),
        ("L" * 1000 + " limitation tail",))
    report = WorkspaceReport("W1-1", "COMPLETED", (finding,), (evidence.evidence_id,))
    store.add_workspace_report(report)
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=2, policy=MemoryPolicy())
    card = access.query("statement", record_types=("finding",)).results[0]
    assert card["statement"].endswith("…")
    assert card["limitations"][0].endswith("…")
    assert card["limitations_truncated"] is True
    for record_id in ("F1", "W1-1"):
        result = access.read(record_id)
        detail = result if record_id == "F1" else result["findings"][0]
        assert detail["statement"].endswith(" statement tail")
        assert detail["limitations"][0].endswith(" limitation tail")
        assert result["next_offset"] is None


def test_global_reports_retain_assigned_tasks_across_rounds_and_failures():
    store = GlobalStore()
    evidence = add_text_evidence(store, "A limited source observation.")
    tasks = (WorkspaceTask("W1-1", "Verify the alleged year, not assume it.", (evidence.evidence_id,)),
             WorkspaceTask("W1-2", "Identify the recording independently.", (evidence.evidence_id,)))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, tasks)
    report = WorkspaceReport("W1-1", "COMPLETED", (), (evidence.evidence_id,))
    store.add_workspace_report(report)
    store.set_workspace_run_status("W1-2", "runtime_error", error={"message": "Unavailable"})
    builder = GlobalViewBuilder(store, MemoryPolicy())
    view = builder.build(task="Identify and verify the recording.", original_material_ids=(), latest_round_index=1)
    for item, assigned in zip(view.latest_reports, tasks, strict=True):
        assert item["task"] == assigned.task
        assert item["materials"] == list(assigned.materials)
    assert view.latest_reports[0]["report"]["status"] == "COMPLETED"
    assert view.latest_reports[1]["run_status"] == "runtime_error"
    history = builder.build(task=view.task, original_material_ids=(), latest_round_index=2).history_index
    historical_report = next(item for item in history if item["record_type"] == "report")
    assert historical_report["task"] == tasks[0].task
    assert "task" not in store.workspace_report("W1-1").to_dict()


def test_report_task_context_is_searchable_and_recoverable_after_view_truncation():
    import json

    store = GlobalStore()
    evidence = add_text_evidence(store, "A source observation.")
    task = WorkspaceTask("W1-1", "Verify source. " * 500 + "Distinctive requirement zirconium.",
                         (evidence.evidence_id,))
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    store.add_workspace_report(WorkspaceReport(task.task_id, "COMPLETED", (), task.materials))
    policy = MemoryPolicy(max_model_view_chars=1000, max_read_chars=512)
    view = GlobalViewBuilder(store, policy).build(task="Check the source.", original_material_ids=(), latest_round_index=1)
    assert len(json.dumps(view.to_dict(), ensure_ascii=False)) <= policy.max_model_view_chars
    assert view.latest_reports[0]["task_context_truncated"]
    assert view.latest_reports[0]["report"]["task_id"] == task.task_id
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=2, policy=policy)
    match = access.query("zirconium", record_types=("report",)).results[0]
    assert match["task_id"] == task.task_id
    assert match["task_context_truncated"]
    assert match["retrieval"]["score"] > 0
    for role, actor_id in (("GIA", "GIA"), ("HI", task.task_id)):
        reader = RecordAccess(store=store, role=role, actor_id=actor_id, round_index=2, policy=policy)
        pages, offset = [], 0
        while True:
            page = reader.read(task.task_id, offset=offset)
            pages.append(page["content"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        recovered = json.loads("".join(pages))
        assert recovered["task"] == task.task
        assert recovered["materials"] == list(task.materials)
    sibling = RecordAccess(store=store, role="HI", actor_id="W1-2", round_index=2, policy=policy)
    with pytest.raises(PermissionError):
        sibling.read(task.task_id, offset=512)
    with pytest.raises(PermissionError):
        sibling.query("zirconium", record_types=("report",))


@pytest.mark.parametrize("run_status", ["pending", "running", "runtime_error", "timed_out", "cancelled"])
def test_long_unreported_tasks_remain_readable_without_exposing_them_to_siblings(run_status):
    import json

    store = GlobalStore()
    task = WorkspaceTask("W1-1", "A local investigation requirement. " * 300 + "Final scope condition.", ())
    store.create_round(1, gia_input={}, gia_decision={})
    store.add_workspace_tasks(1, (task,))
    store.set_workspace_run_status(task.task_id, run_status)
    policy = MemoryPolicy(max_model_view_chars=800, max_read_chars=512)
    view = GlobalViewBuilder(store, policy).build(task="Investigate.", original_material_ids=(), latest_round_index=1)
    assert len(json.dumps(view.to_dict(), ensure_ascii=False)) <= policy.max_model_view_chars
    assert view.latest_reports[0]["task_context_truncated"]
    assert view.latest_reports[0]["run_status"] == run_status
    assert "report" not in view.latest_reports[0]
    assert store.has_reference(task.task_id)
    for role, actor_id in (("GIA", "GIA"), ("HI", task.task_id)):
        reader = RecordAccess(store=store, role=role, actor_id=actor_id, round_index=1, policy=policy)
        pages, offset = [], 0
        while True:
            page = reader.read(task.task_id, offset=offset)
            assert page["record_type"] == "workspace_task"
            pages.append(page["content"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        recovered = json.loads("".join(pages))
        assert recovered["task"] == task.task
        assert recovered["run_status"] == run_status
    sibling = RecordAccess(store=store, role="HI", actor_id="W1-2", round_index=1, policy=policy,
                           local_reference_ids={task.task_id})
    with pytest.raises(PermissionError):
        sibling.read(task.task_id)
    with pytest.raises(KeyError):
        store.workspace_report(task.task_id)
    report = WorkspaceReport(task.task_id, "INCOMPLETE", (), (), status_reason="Unresolved source.")
    store.add_workspace_report(report)
    assert store.resolve(task.task_id) == report
    with pytest.raises(ValueError, match="duplicate global reference"):
        store.add_workspace_report(report)


@pytest.mark.parametrize("record_type,expected_id,id_field", [
    ("finding", "F1", "finding_id"), ("report", "W1-1", "task_id"),
])
def test_memory_query_finds_old_judgment_text_and_scope_outside_summary(record_type, expected_id, id_field):
    store = GlobalStore()
    evidence = add_text_evidence(store, "原始材料")
    old = Finding("F1", "W1-1", "常规观察。" * 900 + "边界校正。", (evidence.evidence_id,),
        ("适用条件。" * 300 + "户外尚未验证。",))
    recent = Finding("F2", "W2-1", "另一份数量统计。", (evidence.evidence_id,))
    for finding in (old, recent):
        store.add_workspace_report(WorkspaceReport(finding.workspace_id, "COMPLETED",
            (finding,), (evidence.evidence_id,)))
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=3,
        policy=MemoryPolicy(query_page_size=1))
    for query in ("边界校正", "户外"):
        page = access.query(query, record_types=(record_type,))
        assert page.results[0][id_field] == expected_id
        assert page.results[0]["retrieval"]["match"] == "lexical"


def test_memory_candidate_query_uses_complete_snippet_and_final_url():
    store = GlobalStore()
    old, _ = store.get_or_create_candidate(identity_key="url:https://example.org/old",
        source_url="https://example.org/old", search_snippet="常规说明。" * 500 + "蝶泳。",
        candidate_type="page")
    final = "https://www.iana.org/help/example-domains"
    store.update_candidate_final_url(old, final, "url:" + final)
    store.get_or_create_candidate(identity_key="url:https://example.org/recent",
        source_url="https://example.org/recent", search_snippet="另一份数量统计。", candidate_type="page")
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=3,
        policy=MemoryPolicy(query_page_size=1))
    for query in ("蝶泳", "iana"):
        result = access.query(query, record_types=("candidate",)).results[0]
        assert result["candidate_id"] == old
        assert result["retrieval"]["match"] == "lexical"


def test_candidate_record_pages_keep_original_and_redirected_source_details():
    import json

    store = GlobalStore()
    source = "https://example.org/source-link"
    final = "https://www.iana.org/help/example-domains"
    candidate_id, _ = store.get_or_create_candidate(identity_key="url:" + source,
        source_url=source, search_title="Document", search_snippet="Context. " * 400 + "Final source note.",
        candidate_type="page", extra={"published": "2024-01-01"})
    store.update_candidate_final_url(candidate_id, final, "url:" + final)
    access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=2,
        policy=MemoryPolicy(max_read_chars=256))
    chunks, offset = [], 0
    while True:
        page = access.read(candidate_id, offset=offset)
        assert page["final_url"] == final
        assert page["candidate_id"] == candidate_id
        chunks.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    complete = json.loads("".join(chunks))
    assert complete["source_url"] == source
    assert complete["snippet"].endswith("Final source note.")
    assert complete["extra"]["published"] == "2024-01-01"
    short_access = RecordAccess(store=store, role="GIA", actor_id="GIA", round_index=2, policy=MemoryPolicy())
    assert short_access.read(candidate_id)["final_url"] == final
    sibling = RecordAccess(store=store, role="HI", actor_id="W1-2", round_index=2,
        policy=MemoryPolicy(), local_reference_ids=set())
    with pytest.raises(PermissionError):
        sibling.read(candidate_id, offset=256)


def test_unopened_candidate_source_stays_local_but_gia_can_query_it():
    """验证 unopened candidate source stays local but gia can query it 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    candidate_id, _ = store.get_or_create_candidate(
        identity_key="url:https://example.test/secret",
        source_url="https://example.test/secret",
        search_title="兄弟工作区线索",
        candidate_type="page",
    )
    hi = RecordAccess(
        store=store,
        role="HI",
        actor_id="W1-2",
        round_index=1,
        policy=MemoryPolicy(),
        local_reference_ids=set(),
    )
    gia = RecordAccess(
        store=store,
        role="GIA",
        actor_id="GIA",
        round_index=2,
        policy=MemoryPolicy(),
    )
    assert hi.query("兄弟工作区线索", record_types=("candidate",)).results == ()
    assert gia.query("兄弟工作区线索", record_types=("candidate",)).results[0][
        "candidate_id"
    ] == candidate_id


def test_memory_query_paginates_recent_evidence_on_zero_lexical_match():
    """同义改写无词面重合时返回有限候选，而不是错误宣称没有历史记录。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    add_text_evidence(store, "汽车出现在道路左侧。")
    latest = add_text_evidence(store, "公共交通工具停在站台旁。")
    access = RecordAccess(
        store=store,
        role="GIA",
        actor_id="GIA",
        round_index=2,
        policy=MemoryPolicy(query_page_size=1),
    )

    page = access.query("机动车", record_types=("evidence",))

    assert page.next_cursor == 1
    assert page.results[0]["evidence_id"] == latest.evidence_id
    assert page.results[0]["retrieval"] == {
        "match": "recent_fallback",
        "score": 0.0,
    }
    assert access.query("机动车", record_types=("evidence",), cursor=1).results[0]["evidence_id"] == "E1"
    assert access.query("机动车", record_types=("evidence",), cursor=1).next_cursor is None


@pytest.mark.parametrize("query,source,distractor", [
    ('"Ask not what America will do for you"',
     'Ask not what America\nwill do for you, but what together we can do.',
     'America will prosper. You may ask what to do. It is not for us to decide.'),
    ('世界公民共同努力', '因此世界公民共同努力解决问题。', '世界很大。公民参与。共同探索。努力研究。'),
    ('NASA火星任务', '这项NASA火星任务有公开档案。', 'NASA项目有火星轨道图；另一任务待定。'),
])
@pytest.mark.parametrize('role', ['GIA', 'HI'])
def test_memory_phrase_query_ranks_contiguous_source_before_newer_scattered_words(query, source, distractor, role):
    store = GlobalStore()
    matching = store.record_evidence(content='Background. ' * 1000 + source,
        label='Earlier source', source_ref='C0', locator='body', acquired_by='TEST/source', discovered_by='W1-1')
    distractor_record = store.record_evidence(content=distractor, label='Later source', source_ref='C0',
        locator='body', acquired_by='TEST/source', discovered_by='W1-2')
    access = RecordAccess(store=store, role=role, actor_id='GIA' if role == 'GIA' else 'W2-1',
        round_index=2, policy=MemoryPolicy(query_page_size=1), local_reference_ids=set())
    first = access.query(query, record_types=('evidence',))
    assert first.results[0]['evidence_id'] == matching.evidence_id
    assert first.results[0]['retrieval']['score'] == 1.0
    second = access.query(query, record_types=('evidence',), cursor=first.next_cursor)
    assert second.results[0]['evidence_id'] == distractor_record.evidence_id
    assert 0 < second.results[0]['retrieval']['score'] < first.results[0]['retrieval']['score']
    assert 'content' not in first.results[0] and 'content_excerpt' not in first.results[0]
    assert source in access.read(matching.evidence_id, offset=12_000)['content']


@pytest.mark.parametrize('query,text', [('cat', 'scarcity catalog'), ('ore', 'before more'),
    ('résumé', 'résumés'), ('мир', 'мировой')])
def test_memory_query_does_not_treat_word_fragments_as_lexical_matches(query, text):
    store = GlobalStore()
    record = add_text_evidence(store, text)
    access = RecordAccess(store=store, role='GIA', actor_id='GIA', round_index=1, policy=MemoryPolicy())
    result = access.query(query, record_types=('evidence',)).results[0]
    assert result['evidence_id'] == record.evidence_id
    assert result['retrieval'] == {'match': 'recent_fallback', 'score': 0.0}


def test_memory_keyword_query_keeps_noncontiguous_and_reordered_matches():
    store = GlobalStore()
    first = add_text_evidence(store, 'Lunar archives describe the research. NASA published the Apollo records.')
    second = add_text_evidence(store, 'Apollo records from NASA are available in the archives.')
    access = RecordAccess(store=store, role='GIA', actor_id='GIA', round_index=1, policy=MemoryPolicy())
    page = access.query('NASA Apollo archives', record_types=('evidence',))
    assert {item['evidence_id'] for item in page.results} == {first.evidence_id, second.evidence_id}
    assert all(item['retrieval']['match'] == 'lexical' and 0 < item['retrieval']['score'] < 1
        for item in page.results)


def test_memory_phrase_match_cannot_be_assembled_across_label_and_source_body():
    store = GlobalStore()
    exact = store.record_evidence(label='Source A', content='The launch date changed.', source_ref='C0',
        locator='body', acquired_by='TEST/source', discovered_by='W1-1')
    split = store.record_evidence(label='launch', content='date is unknown.', source_ref='C0',
        locator='body', acquired_by='TEST/source', discovered_by='W1-2')
    access = RecordAccess(store=store, role='GIA', actor_id='GIA', round_index=2, policy=MemoryPolicy())
    cards = access.query('launch date', record_types=('evidence',)).results
    assert [c['evidence_id'] for c in cards] == [exact.evidence_id, split.evidence_id]
    assert cards[0]['retrieval']['score'] == 1.0
    assert 0 < cards[1]['retrieval']['score'] < 1.0


def test_candidate_source_becomes_readable_when_its_evidence_is_globally_shared():
    """验证 candidate source becomes readable when its evidence is globally shared 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    candidate_id, _ = store.get_or_create_candidate(
        identity_key="url:https://example.test/opened",
        source_url="https://example.test/opened",
        search_title="已打开来源",
        candidate_type="page",
    )
    store.record_evidence(
        content="来源正文",
        label="来源正文",
        source_ref=candidate_id,
        locator="paragraph:1",
        acquired_by="SEARCH/open",
        discovered_by="W1-1",
    )
    harness = WorkspaceHarness(
        task=WorkspaceTask("W1-2", "复核", ("C0",)), store=store
    )
    assert harness.record_access.read(candidate_id)["candidate_id"] == candidate_id


@pytest.mark.parametrize('role,allowed', [('GIA', True), ('HI', True), ('HI', False)])
def test_evidence_pages_bind_content_to_permitted_source_urls(role, allowed):
    store = GlobalStore()
    original = 'https://mirror.example.test/document'
    final = 'https://archive.example.test/transcript'
    candidate, _ = store.get_or_create_candidate(identity_key='url:' + original,
        source_url=original, search_title='Official archive',
        search_snippet='Private candidate search context', candidate_type='page')
    store.update_candidate_final_url(candidate, final, 'url:' + final)
    evidence = store.record_evidence(content='First page. Second page.', label='Official archive',
        source_ref=candidate, locator='paragraph:1', acquired_by='SEARCH/open', discovered_by='W1-1')
    access = RecordAccess(store=store, role=role, actor_id='W1-2', round_index=1,
        policy=MemoryPolicy(max_read_chars=12), local_reference_ids={candidate} if allowed else set())
    for offset in (0, 12):
        page = access.read(evidence.evidence_id, offset=offset)
        assert page['content'] == evidence.content[offset:offset + 12]
        if allowed:
            assert page['source'] == {'candidate_id': candidate, 'source_url': original, 'final_url': final}
            assert 'Private candidate search context' not in str(page)
        else:
            assert 'source' not in page
            with pytest.raises(PermissionError):
                access.read(candidate)
    assert set(evidence.card().to_dict()) == {
        'evidence_id', 'modalities', 'label', 'source_ref', 'derived_from', 'acquired_by'}


class SharingSearch:
    """测试辅助类 SharingSearch；用可控状态替代真实外部依赖。"""

    def __init__(self, store: GlobalStore, ready: asyncio.Event) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.store = store
        self.ready = ready
        self.evidence_id: str | None = None

    async def execute(self, call, *, workspace_id):
        """模拟 execute 接口，返回由当前用例预设的可控结果。"""

        candidate_id, _ = self.store.get_or_create_candidate(
            identity_key="url:https://example.test/source",
            source_url="https://example.test/source",
            candidate_type="page",
        )
        evidence = self.store.record_evidence(
            content="共享证据：页面日期是2022年。",
            label="共享证据页面",
            source_ref=candidate_id,
            locator="paragraph:1",
            acquired_by="SEARCH/open",
            discovered_by=workspace_id,
        )
        self.evidence_id = evidence.evidence_id
        self.ready.set()
        return SearchResponse(
            "text",
            "target_reached",
            (
                SearchResult(
                    candidate_id,
                    "https://example.test/source",
                    "共享证据页面",
                    None,
                    (
                        MaterialResult("page", "ready", (evidence.card(),)),
                        MaterialResult("document", "absent"),
                    ),
                ),
            ),
        )


class SharingHi:
    """测试辅助类 SharingHi；用可控状态替代真实外部依赖。"""

    def __init__(self, ready: asyncio.Event, search: SharingSearch) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.ready = ready
        self.search = search
        self.turns: dict[str, int] = {}

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        turn = self.turns.get(workspace_id, 0) + 1
        self.turns[workspace_id] = turn
        if workspace_id == "W1-1" and turn == 1:
            return {
                "action": "SEARCH",
                "arguments": {"input_type": "text", "query": "共享证据", "output_type": "text"},
                "focus_evidence_ids": [],
            }
        if workspace_id == "W1-2" and turn == 1:
            await self.ready.wait()
            return {
                "action": "QUERY_MEMORY",
                "arguments": {"query": "共享证据", "record_types": ["evidence"]},
                "focus_evidence_ids": [],
            }
        evidence_id = self.search.evidence_id
        assert evidence_id is not None
        return {
            "task_id": workspace_id,
            "status": "COMPLETED",
            "findings": [
                {
                    "statement": "已读取同轮共享的日期材料。",
                    "evidence_ids": [evidence_id],
                    "limitations": [],
                }
            ],
            "selected_evidence_ids": [evidence_id],
        }


@pytest.mark.asyncio
async def test_same_round_workspaces_run_concurrently_and_share_committed_evidence():
    """验证 same round workspaces run concurrently and share committed evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "待核查说法"})
    store.create_round(1, gia_input={}, gia_decision={})
    tasks = (
        WorkspaceTask("W1-1", "寻找日期来源", ("C0",)),
        WorkspaceTask("W1-2", "复核已取得日期", ("C0",)),
    )
    store.add_workspace_tasks(1, tasks)
    ready = asyncio.Event()
    search = SharingSearch(store, ready)
    policy = SharingHi(ready, search)
    coordinator = RoundCoordinator(
        store=store,
        hi_policy=policy,
        search_service=search,  # type: ignore[arg-type]
        inspect_service=None,
        memory_policy=MemoryPolicy(),
        max_concurrency=2,
        workspace_timeout_s=2,
    )
    outcomes = await coordinator.run(round_index=1, tasks=tasks)
    assert [item.run_status for item in outcomes] == ["reported", "reported"]
    assert store.workspace_report("W1-2").findings[0].evidence_ids == (search.evidence_id,)
    query_observation = next(
        event for event in store.trajectory_events(actor_id="W1-2")
        if event["event_type"] == "observation"
    )
    assert search.evidence_id in str(query_observation["payload"])


class TwoRoundGia:
    """测试辅助类 TwoRoundGia；用可控状态替代真实外部依赖。"""

    def __init__(self, evidence_id: str) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0
        self.evidence_id = evidence_id

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            assert model_view["latest_reports"] == []
            return {
                "decision": "INVESTIGATE",
                "tasks": [
                    {"task": "调查时间", "materials": ["C0", self.evidence_id]},
                    {"task": "调查来源", "materials": ["C0", self.evidence_id]},
                ],
            }
        if self.calls == 2:
            assert len(model_view["latest_reports"]) == 2
            return {
                "decision": "INVESTIGATE",
                "tasks": [{"task": "综合复核", "materials": [self.evidence_id]}],
            }
        assert len(model_view["latest_reports"]) == 1
        assert model_view["history_index"]
        return {"decision": "FINISH", "answer": f"材料能够确认公开时间[{self.evidence_id}]。"}


class ImmediateReportHi:
    """测试辅助类 ImmediateReportHi；用可控状态替代真实外部依赖。"""

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        evidence_id = next(
            item["evidence_id"]
            for item in model_view["evidence_view"]
            if item["record_type"] == "evidence"
        )
        return {
            "task_id": workspace_id,
            "status": "COMPLETED",
            "findings": [
                {
                    "statement": "完成局部核查。",
                    "evidence_ids": [evidence_id],
                    "limitations": ["只依据当前材料。"],
                }
            ],
            "selected_evidence_ids": [evidence_id],
        }


@pytest.mark.asyncio
async def test_two_round_runtime_builds_global_view_and_finishes_with_evidence_chain():
    """验证 two round runtime builds global view and finishes with evidence chain 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "核查材料公开时间"})
    evidence = add_text_evidence(store, "页面写明2022年4月18日发布。")
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=TwoRoundGia(evidence.evidence_id),
        hi_policy=ImmediateReportHi(),
        max_rounds=3,
        max_concurrency=2,
    )
    result = await runtime.run(
        task="核查材料是否早已公开",
        original_material_ids=("C0", evidence.evidence_id),
    )
    assert result.cited_evidence == (evidence,)
    assert store.round_record(1)["status"] == "completed"
    assert store.round_record(2)["status"] == "completed"
    assert [item["task_id"] for item in store.workspace_task_records(1)] == ["W1-1", "W1-2"]
    assert store.workspace_report("W2-1").findings[0].workspace_id == "W2-1"
    assert store.final_result()["decision"] == "FINISH"


class ExplodingHi:
    """测试辅助类 ExplodingHi；用可控状态替代真实外部依赖。"""

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        raise RuntimeError("model transport failed")


@pytest.mark.asyncio
async def test_runtime_error_is_not_fabricated_as_incomplete_hi_report():
    """验证 runtime error is not fabricated as incomplete hi report 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    store.create_round(1, gia_input={}, gia_decision={})
    task = WorkspaceTask("W1-1", "调查", ("C0",))
    store.add_workspace_tasks(1, (task,))
    coordinator = RoundCoordinator(
        store=store,
        hi_policy=ExplodingHi(),
        search_service=None,
        inspect_service=None,
        memory_policy=MemoryPolicy(),
    )
    outcome = (await coordinator.run(round_index=1, tasks=(task,)))[0]
    assert outcome.run_status == "runtime_error"
    assert store.resolve("W1-1") == task
    with pytest.raises(KeyError):
        store.workspace_report("W1-1")
    assert store.workspace_task_record("W1-1")["error"]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_custom_detect_slot_is_hidden_until_a_real_handler_is_registered():
    """验证 custom detect slot is hidden until a real handler is registered 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    harness = WorkspaceHarness(
        task=WorkspaceTask("W1-1", "检测", ("C0",)), store=store
    )
    assert "DETECT" not in {item["name"] for item in harness.eligible_actions()}

    async def detect(arguments, workspace_id):
        """测试辅助函数 detect；为用例提供可重复的输入或可控行为。"""

        return ToolObservation(
            "DETECT",
            ("detector reported its bounded result",),
            {"workspace_id": workspace_id, "request": arguments},
        )

    harness.register_action(
        "DETECT",
        schema={"type": "object"},
        handler=detect,
    )
    assert "DETECT" in {item["name"] for item in harness.eligible_actions()}


class FinishAfterResumeGia:
    """测试辅助类 FinishAfterResumeGia；用可控状态替代真实外部依赖。"""

    def __init__(self, evidence_id: str) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.evidence_id = evidence_id

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        assert len(model_view["latest_reports"]) == 1
        return {"decision": "FINISH", "answer": f"恢复后完成[{self.evidence_id}]。"}


@pytest.mark.asyncio
async def test_runtime_resumes_an_open_round_from_its_last_complete_boundary():
    """验证 runtime resumes an open round from its last complete boundary 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "恢复测试"})
    evidence = add_text_evidence(store, "可复用证据")
    initial_view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="恢复调查",
        original_material_ids=("C0", evidence.evidence_id),
        latest_round_index=None,
    )
    store.create_round(
        1,
        gia_input=initial_view.to_dict(),
        gia_decision={
            "decision": "INVESTIGATE",
            "tasks": [{"task": "恢复局部任务", "materials": [evidence.evidence_id]}],
        },
    )
    store.add_workspace_tasks(
        1, (WorkspaceTask("W1-1", "恢复局部任务", (evidence.evidence_id,)),)
    )
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=FinishAfterResumeGia(evidence.evidence_id),
        hi_policy=ImmediateReportHi(),
        max_rounds=1,
    )
    result = await runtime.run(
        task="恢复调查", original_material_ids=("C0", evidence.evidence_id)
    )
    assert result.cited_evidence == (evidence,)
    assert store.workspace_task_record("W1-1")["run_status"] == "reported"
    assert len(store.round_records()) == 1


def test_trajectory_export_is_jsonl_and_contains_no_hidden_reasoning(tmp_path):
    """验证 trajectory export is jsonl and contains no hidden reasoning 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.record_event(
        actor_role="HI",
        actor_id="W1-1",
        event_type="observation",
        payload={"visible": "tool result"},
    )
    target = store.export_trajectory(tmp_path / "trajectory.jsonl")
    line = target.read_text(encoding="utf-8").strip()
    assert '"event_type":"observation"' in line
    assert "chain_of_thought" not in line


class ImmediateFinishGia:
    """测试辅助类 ImmediateFinishGia；用可控状态替代真实外部依赖。"""

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        assert model_view["original_materials"]
        return {"decision": "FINISH", "answer": "现有材料不足，无法判断。"}


class UnusedHi:
    """测试辅助类 UnusedHi；用可控状态替代真实外部依赖。"""

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        raise AssertionError("HI must not run when GIA finishes immediately")


@pytest.mark.asyncio
async def test_run_new_registers_claim_and_initial_media_before_gia():
    """验证 run new registers claim and initial media before gia 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=ImmediateFinishGia(),
        hi_policy=UnusedHi(),
    )
    result = await runtime.run_new(
        task="判断材料是否足够",
        claim_materials={"C0": {"claim": "示例说法"}},
        media=(
            InitialMedia(
                source_ref="I0",
                media_type="image/png",
                modalities=("image",),
                label="初始图片",
                original_filename="input.png",
                data=image_bytes(),
            ),
        ),
    )
    assert result.cited_evidence == ()
    assert store.resolve("C0")["claim"] == "示例说法"
    assert store.evidence_records()[0].acquired_by == "HARNESS/ingest"


def test_focus_and_explicit_materials_stay_visible_under_a_small_view_budget():
    """验证 focus and explicit materials stay visible under a small view budget 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    explicit = add_text_evidence(store, "明确附带材料")
    focused = add_text_evidence(store, "当前重点材料")
    for index in range(6):
        add_text_evidence(store, f"普通历史材料 {index}")
    view = EvidenceViewBuilder(
        store, MemoryPolicy(max_evidence_cards=2)
    ).build(
        task="完全不同的查询",
        workspace_id="W1-1",
        explicit_material_ids=(explicit.evidence_id,),
        focus_evidence_ids=(focused.evidence_id,),
    )
    assert [item["evidence_id"] for item in view] == [explicit.evidence_id, focused.evidence_id]


class UnknownCitationGia:
    """测试辅助类 UnknownCitationGia；用可控状态替代真实外部依赖。"""

    def __init__(self):
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            return {"decision": "FINISH", "answer": "这个结论引用了不存在的证据[E999]。"}
        assert history[-1]["observation"]["action"] == "HARNESS_REJECTION"
        return {"decision": "FINISH", "answer": "没有足够证据，无法判断。"}


@pytest.mark.asyncio
async def test_gia_can_repair_an_unknown_final_evidence_reference():
    """验证 gia can repair an unknown final evidence reference 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    policy = UnknownCitationGia()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
    )
    result = await runtime.run(task="核查", original_material_ids=("C0",))
    assert result.cited_evidence == ()
    assert policy.calls == 2
    assert store.has_final_result()


@pytest.mark.asyncio
async def test_gia_policy_output_error_is_recoverable_in_the_same_session():
    class RecoveringGia:
        calls = 0
        async def decide(self, *, model_view, history, actions):
            self.calls += 1
            if self.calls == 1:
                raise PolicyOutputError("Malformed decision")
            assert history[-1]["observation"]["action"] == "HARNESS_REJECTION"
            return {"decision":"FINISH", "answer":"现有证据不足。"}
    store = GlobalStore()
    policy = RecoveringGia()
    async with InvestigationRuntime(store=store, gia_policy=policy, hi_policy=UnusedHi()) as runtime:
        await runtime.run(task="核查", original_material_ids=())
    assert policy.calls == 2
    assert not store.workspace_report_records()


class SlowHi:
    """测试辅助类 SlowHi；用可控状态替代真实外部依赖。"""

    async def decide(self, *, workspace_id, model_view, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_workspace_timeout_is_an_operational_outcome_not_a_report():
    """验证 workspace timeout is an operational outcome not a report 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    store.create_round(1, gia_input={}, gia_decision={})
    task = WorkspaceTask("W1-1", "调查", ("C0",))
    store.add_workspace_tasks(1, (task,))
    coordinator = RoundCoordinator(
        store=store,
        hi_policy=SlowHi(),
        search_service=None,
        inspect_service=None,
        memory_policy=MemoryPolicy(),
        workspace_timeout_s=0.01,
    )
    outcome = (await coordinator.run(round_index=1, tasks=(task,)))[0]
    assert outcome.run_status == "timed_out"
    assert store.resolve("W1-1") == task
    with pytest.raises(KeyError):
        store.workspace_report("W1-1")
    assert store.round_record(1)["status"] == "completed_with_errors"


class BrowsingGia:
    """测试辅助类 BrowsingGia；用可控状态替代真实外部依赖。"""

    def __init__(self):
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            assert {item["name"] for item in actions} == {"QUERY_MEMORY", "READ_RECORD"}
            return {
                "action": "QUERY_MEMORY",
                "arguments": {"query": "2022", "record_types": ["finding"]},
            }
        assert history[-1]["observation"]["results"][0]["finding_id"] == "F1"
        return {"decision": "FINISH", "answer": "找到历史判断[E1]。"}


@pytest.mark.asyncio
async def test_gia_can_query_findings_before_returning_a_decision():
    """验证 gia can query findings before returning a decision 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    evidence = add_text_evidence(store, "2022年页面")
    store.add_workspace_report(
        WorkspaceReport(
            task_id="W1-1",
            status="COMPLETED",
            findings=(
                Finding(
                    finding_id="F1",
                    workspace_id="W1-1",
                    statement="材料在2022年已经出现。",
                    evidence_ids=(evidence.evidence_id,),
                ),
            ),
            selected_evidence_ids=(evidence.evidence_id,),
        )
    )
    policy = BrowsingGia()
    runner = GiaRunner(
        store=store,
        policy=policy,
        memory_policy=MemoryPolicy(),
    )
    view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="核查时间", original_material_ids=("C0",), latest_round_index=None
    )
    decision = await runner.run(view, round_index=2)
    assert decision.decision == "FINISH"
    assert policy.calls == 2


def test_start_round_rolls_back_the_round_when_a_task_insert_fails():
    """验证 start round rolls back the round when a task insert fails 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.create_round(1, gia_input={}, gia_decision={})
    existing = WorkspaceTask("W1-1", "existing", ())
    store.add_workspace_tasks(1, (existing,))

    with pytest.raises(sqlite3.IntegrityError):
        store.start_round(
            2,
            gia_input={"task": "x"},
            gia_decision={"decision": "INVESTIGATE", "tasks": []},
            tasks=(WorkspaceTask("W1-1", "duplicate", ()),),
        )

    assert [item["round_index"] for item in store.round_records()] == [1]


class BlockingFinishGia:
    """测试辅助类 BlockingFinishGia；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return {"decision": "FINISH", "answer": "材料不足。"}


@pytest.mark.asyncio
async def test_concurrent_runtime_calls_advance_one_investigation_once():
    """验证 concurrent runtime calls advance one investigation once 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    policy = BlockingFinishGia()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
        gia_turn_timeout_s=2,
    )

    first = asyncio.create_task(
        runtime.run(task="核查", original_material_ids=("C0",))
    )
    await policy.entered.wait()
    second = asyncio.create_task(
        runtime.run(task="核查", original_material_ids=("C0",))
    )
    await asyncio.sleep(0)
    policy.release.set()

    first_result, second_result = await asyncio.gather(first, second)
    assert first_result.decision == second_result.decision
    assert policy.calls == 1


class CloseProbe:
    """测试辅助类 CloseProbe；用可控状态替代真实外部依赖。"""

    def __init__(self, error: BaseException | None = None) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.error = error
        self.closed = False

    async def aclose(self) -> None:
        """模拟资源关闭，并记录生命周期状态供断言。"""

        self.closed = True
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_close_waits_for_the_active_run_and_rejects_new_work():
    """验证 close waits for the active run and rejects new work 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    policy = BlockingFinishGia()
    search = CloseProbe()
    inspect = CloseProbe()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
        search_service=search,  # type: ignore[arg-type]
        inspect_service=inspect,  # type: ignore[arg-type]
        gia_turn_timeout_s=2,
    )
    active = asyncio.create_task(
        runtime.run(task="核查", original_material_ids=("C0",))
    )
    await policy.entered.wait()
    closing = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeClosedError):
        await runtime.run(task="核查", original_material_ids=("C0",))
    assert not search.closed
    assert not inspect.closed

    policy.release.set()
    await active
    await closing
    assert search.closed
    assert inspect.closed


@pytest.mark.asyncio
async def test_close_attempts_every_owned_service_when_one_close_fails():
    """验证 close attempts every owned service when one close fails 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    search = CloseProbe()
    inspect = CloseProbe(RuntimeError("inspect close failed"))
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=ImmediateFinishGia(),
        hi_policy=UnusedHi(),
        search_service=search,  # type: ignore[arg-type]
        inspect_service=inspect,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="inspect close failed"):
        await runtime.aclose()
    assert inspect.closed
    assert search.closed


class HangingGia:
    """测试辅助类 HangingGia；用可控状态替代真实外部依赖。"""

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_gia_policy_turn_has_a_deadline_and_records_the_failure():
    """验证 gia policy turn has a deadline and records the failure 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="核查", original_material_ids=("C0",), latest_round_index=None
    )
    runner = GiaRunner(
        store=store,
        policy=HangingGia(),
        memory_policy=MemoryPolicy(),
        turn_timeout_s=0.01,
    )

    with pytest.raises(GiaPolicyTimeout):
        await runner.run(view, round_index=1)
    errors = [
        event
        for event in store.trajectory_events(actor_role="GIA")
        if event["event_type"] == "runtime_error"
    ]
    assert errors[-1]["payload"]["error"]["type"] == "GiaPolicyTimeout"


@pytest.mark.asyncio
async def test_run_new_is_idempotent_for_identical_initial_materials():
    """验证 run new is idempotent for identical initial materials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    policy = ImmediateFinishGia()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
    )
    initial = InitialMedia(
        source_ref="I0",
        media_type="image/png",
        modalities=("image",),
        label="初始图片",
        original_filename="input.png",
        data=image_bytes(),
    )

    first = await runtime.run_new(
        task="核查", claim_materials={"C0": {"claim": "x"}}, media=(initial,)
    )
    second = await runtime.run_new(
        task="核查", claim_materials={"C0": {"claim": "x"}}, media=(initial,)
    )

    assert first.decision == second.decision
    assert len(store.evidence_records()) == 1

    changed = InitialMedia(
        source_ref="I0",
        media_type="image/png",
        modalities=("image",),
        label="初始图片",
        original_filename="input.png",
        data=image_bytes() + b"changed",
    )
    with pytest.raises(ValueError, match="stored bytes"):
        await runtime.run_new(
            task="核查",
            claim_materials={"C0": {"claim": "x"}},
            media=(changed,),
        )


@pytest.mark.asyncio
async def test_pure_research_question_can_start_without_original_materials():
    """只有任务文本的开放网络调查不需要伪造 C 材料。"""

    class EmptyInputFinishGia:
        async def decide(self, *, model_view, history, actions):
            assert model_view["original_materials"] == []
            return {"decision": "FINISH", "answer": "没有原始材料。"}

    store = GlobalStore()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=EmptyInputFinishGia(),
        hi_policy=UnusedHi(),
    )

    result = await runtime.run_new(task="调查一个公开问题", claim_materials={})

    assert result.decision.decision == "FINISH"
    assert store.investigation_meta()["original_material_ids"] == []
    await runtime.aclose()


def test_store_healthcheck_detects_artifact_size_corruption():
    """验证 store healthcheck detects artifact size corruption 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    evidence = store.commit_ingested_evidence(
        source_ref="I0",
        media_type="image/png",
        modalities=("image",),
        label="initial",
        original_filename="input.png",
        data=image_bytes(),
    )
    assert store.healthcheck().status == "ok"
    assert evidence.artifact_ref is not None
    artifact_path = store.artifact_path(evidence.artifact_ref)
    artifact_path.write_bytes(b"x" * artifact_path.stat().st_size)

    with pytest.raises(ArtifactIntegrityError):
        store.artifact_path(evidence.artifact_ref)
    same_size_health = store.healthcheck()
    assert same_size_health.status == "degraded"
    assert same_size_health.content_mismatches == (evidence.artifact_ref,)

    artifact_path.write_bytes(b"damaged")

    health = store.healthcheck()
    assert health.status == "degraded"
    assert health.size_mismatches == (evidence.artifact_ref,)


def test_store_migration_backfills_artifact_digests(tmp_path):
    """验证 store migration backfills artifact digests 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    payload = b"legacy artifact"
    (artifact_root / "legacy.bin").write_bytes(payload)
    db_path = tmp_path / "store.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relative_path TEXT NOT NULL UNIQUE,
            media_type TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""
        )
        connection.execute(
            """INSERT INTO artifacts(relative_path,media_type,byte_size,role)
            VALUES (?,?,?,?)""",
            ("legacy.bin", "application/octet-stream", len(payload), "legacy"),
        )

    store = GlobalStore(db_path, artifact_root)

    assert len(store.artifact("A1").sha256) == 64
    assert store.healthcheck().status == "ok"


def test_model_view_bounds_large_observations():
    """验证 model view bounds large observations 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    harness = WorkspaceHarness(
        task=WorkspaceTask("W1-1", "核查", ("C0",)),
        store=store,
        memory_policy=MemoryPolicy(max_observation_chars=64),
    )
    harness.reject("x" * 1000)

    observation = harness.model_view()["latest_observation"]
    assert observation["truncated"] is True
    assert len(observation["preview"]) == 64


def test_record_read_rejects_non_text_ids_at_the_model_boundary():
    """验证 record read rejects non text ids at the model boundary 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    access = RecordAccess(
        store=store,
        role="GIA",
        actor_id="GIA",
        round_index=1,
        policy=MemoryPolicy(),
    )
    with pytest.raises(ValueError, match="record_id"):
        access.read(7)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resume_reconstructs_tasks_from_an_interrupted_legacy_round_start():
    """验证 resume reconstructs tasks from an interrupted legacy round start 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    evidence = add_text_evidence(store, "可复用证据")
    initial_view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="恢复调查",
        original_material_ids=("C0", evidence.evidence_id),
        latest_round_index=None,
    )
    store.create_round(
        1,
        gia_input=initial_view.to_dict(),
        gia_decision={
            "decision": "INVESTIGATE",
            "tasks": [
                {"task": "恢复局部任务", "materials": [evidence.evidence_id]}
            ],
        },
    )
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=FinishAfterResumeGia(evidence.evidence_id),
        hi_policy=ImmediateReportHi(),
        max_rounds=1,
    )

    result = await runtime.run(
        task="恢复调查", original_material_ids=("C0", evidence.evidence_id)
    )

    assert result.cited_evidence == (evidence,)
    assert store.workspace_task_record("W1-1")["run_status"] == "reported"


@pytest.mark.asyncio
async def test_resume_restarts_a_workspace_cancelled_with_an_open_round():
    """验证 resume restarts a workspace cancelled with an open round 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    evidence = add_text_evidence(store, "可复用证据")
    initial_view = GlobalViewBuilder(store, MemoryPolicy()).build(
        task="恢复调查",
        original_material_ids=("C0", evidence.evidence_id),
        latest_round_index=None,
    )
    task = WorkspaceTask("W1-1", "恢复局部任务", (evidence.evidence_id,))
    store.start_round(
        1,
        gia_input=initial_view.to_dict(),
        gia_decision={
            "decision": "INVESTIGATE",
            "tasks": [
                {"task": task.task, "materials": list(task.materials)}
            ],
        },
        tasks=(task,),
    )
    store.set_workspace_run_status(
        task.task_id,
        "cancelled",
        error={"type": "cancelled", "message": "shutdown"},
    )
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=FinishAfterResumeGia(evidence.evidence_id),
        hi_policy=ImmediateReportHi(),
        max_rounds=1,
    )

    await runtime.run(
        task="恢复调查", original_material_ids=("C0", evidence.evidence_id)
    )

    assert store.workspace_task_record(task.task_id)["run_status"] == "reported"


@pytest.mark.asyncio
async def test_committed_report_remains_successful_when_terminal_event_logging_fails():
    """验证 committed report remains successful when terminal event logging fails 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    store.create_round(1, gia_input={}, gia_decision={})
    task = WorkspaceTask("W1-1", "核查", ("C0",))
    store.add_workspace_tasks(1, (task,))
    original_record_event = store.record_event

    def record_event_with_terminal_failure(**kwargs):
        """测试辅助函数 record_event_with_terminal_failure；为用例提供可重复的输入或可控行为。"""

        if kwargs["event_type"] == "workspace_report":
            raise OSError("trajectory disk failure")
        return original_record_event(**kwargs)

    store.record_event = record_event_with_terminal_failure  # type: ignore[method-assign]
    policy = SequenceHi(
        {
            "W1-1": [
                {
                    "task_id": "W1-1",
                    "status": "INCOMPLETE",
                    "findings": [],
                    "selected_evidence_ids": [],
                    "status_reason": "材料不足。",
                }
            ]
        }
    )

    report = await WorkspaceRunner(
        harness=WorkspaceHarness(task=task, store=store, round_index=1),
        policy=policy,
    ).run()

    assert report.status == "INCOMPLETE"
    assert store.workspace_task_record("W1-1")["run_status"] == "reported"


class RepairNonCanonicalCitationGia:
    """测试辅助类 RepairNonCanonicalCitationGia；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            return {"decision": "FINISH", "answer": "错误编号E01不能作为证据。"}
        assert history[-1]["observation"]["action"] == "HARNESS_REJECTION"
        return {"decision": "FINISH", "answer": "没有足够证据。"}


@pytest.mark.asyncio
async def test_non_canonical_final_evidence_ids_are_rejected_for_repair():
    """验证 non canonical final evidence ids are rejected for repair 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    policy = RepairNonCanonicalCitationGia()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
    )

    result = await runtime.run(task="核查", original_material_ids=("C0",))

    assert result.cited_evidence == ()
    assert policy.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,count", [("证据E1支持此结论。", 1), ("参考(E1)。", 1), ("型号ACE1没有证据含义。", 0)])
async def test_final_citations_work_next_to_chinese_text(answer, count):
    store = GlobalStore()
    store.add_material("C0", {"claim":"check"})
    evidence = add_text_evidence(store, "可核查内容")

    class Policy:
        async def decide(self, **kwargs):
            return {"decision":"FINISH", "answer":answer}

    async with InvestigationRuntime(store=store, gia_policy=Policy(), hi_policy=UnusedHi()) as runtime:
        result = await runtime.run(task="核查引用", original_material_ids=(evidence.evidence_id,))
    assert len(result.cited_evidence) == count
    assert tuple(item.evidence_id for item in result.cited_evidence) == ((evidence.evidence_id,) if count else ())


@pytest.mark.asyncio
async def test_original_material_limit_is_checked_before_any_material_is_written():
    """验证 original material limit is checked before any material is written 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=ImmediateFinishGia(),
        hi_policy=UnusedHi(),
        max_original_materials=1,
    )

    with pytest.raises(ValueError, match="original materials exceed"):
        await runtime.run_new(
            task="核查",
            claim_materials={"C0": {"claim": "x"}, "C1": {"claim": "y"}},
        )

    assert not store.has_reference("C0")
    assert not store.has_reference("C1")


class RepairOversizedRoundGia:
    """测试辅助类 RepairOversizedRoundGia；用可控状态替代真实外部依赖。"""

    def __init__(self) -> None:
        """保存测试替身需要的预设输入、状态和调用记录。"""

        self.calls = 0

    async def decide(self, *, model_view, history, actions):
        """模拟 decide 接口，返回由当前用例预设的可控结果。"""

        self.calls += 1
        if self.calls == 1:
            return {
                "decision": "INVESTIGATE",
                "tasks": [
                    {"task": "a", "materials": ["C0"]},
                    {"task": "b", "materials": ["C0"]},
                ],
            }
        assert history[-1]["observation"]["action"] == "HARNESS_REJECTION"
        return {"decision": "FINISH", "answer": "没有足够证据。"}


@pytest.mark.asyncio
async def test_oversized_round_is_rejected_before_it_is_persisted():
    """验证 oversized round is rejected before it is persisted 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    store.add_material("C0", {"claim": "x"})
    policy = RepairOversizedRoundGia()
    runtime = InvestigationRuntime(
        store=store,
        gia_policy=policy,
        hi_policy=UnusedHi(),
        max_workspaces_per_round=1,
    )

    await runtime.run(task="核查", original_material_ids=("C0",))

    assert store.round_records() == []
    assert policy.calls == 2


def test_evidence_index_uses_a_bounded_excerpt_instead_of_full_content():
    """验证 evidence index uses a bounded excerpt instead of full content 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    content = "a" * 3000 + "UNIQUE_TAIL"
    evidence = add_text_evidence(store, content)

    record = store.evidence_index_records(content_chars=2000)[0]
    view = EvidenceViewBuilder(store, MemoryPolicy()).build(
        task="a",
        workspace_id="W1-1",
        explicit_material_ids=(evidence.evidence_id,),
        focus_evidence_ids=(),
    )

    assert len(record["content_excerpt"]) == 2000
    assert record["content_chars"] == len(content)
    assert "UNIQUE_TAIL" not in str(view)
