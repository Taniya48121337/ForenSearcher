"""真实审查中发现的跨模块失败场景。"""

import jsonschema
import pytest

from foren_searcher.contracts import HiActionDecision, WorkspaceTask
from foren_searcher.inspect.contracts import inspect_action_schema, parse_inspect_call
from foren_searcher.local_models import TransformersGiaPolicy, TransformersHiPolicy
from foren_searcher.memory import MemoryPolicy
from foren_searcher.memory import bound_global_view
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from test_local_models import CaptureGenerator


def add(store, content, workspace="W1-1"):
    return store.record_evidence(content=content, label="Source", source_ref="C0",
        locator="paragraph:1", acquired_by="TEST/source", discovered_by=workspace)


def test_query_memory_finds_content_beyond_card_excerpt():
    store = GlobalStore()
    target = add(store, "unrelated " * 500 + "quantum-butterfly-secret")
    for _ in range(15):
        add(store, "new unrelated material")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Locate quantum-butterfly-secret", ()), store=store)
    page = harness.record_access.query("quantum-butterfly-secret", record_types=("evidence",))
    assert page.results[0]["evidence_id"] == target.evidence_id
    assert "content" not in page.results[0]


@pytest.mark.asyncio
async def test_shared_evidence_can_be_read_then_cited_after_card_eviction():
    store = GlobalStore()
    target = add(store, "critical contradiction", "W1-2")
    for _ in range(12):
        add(store, "irrelevant memo")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "critical contradiction", ()), store=store,
        memory_policy=MemoryPolicy(max_evidence_cards=1))
    assert harness.model_view()["evidence_view"][0]["evidence_id"] == target.evidence_id
    await harness.execute_action(HiActionDecision("READ_RECORD", {"record_id": target.evidence_id}, ()))
    newest = add(store, "critical contradiction critical", "W1-1")
    harness.set_focus((newest.evidence_id,))
    view = harness.model_view()
    assert target.evidence_id not in [card["evidence_id"] for card in view["evidence_view"]]
    report = {"task_id": "W1-1", "status": "COMPLETED", "findings": [{
        "statement": "critical contradiction", "evidence_ids": [target.evidence_id], "limitations": []}],
        "selected_evidence_ids": [target.evidence_id]}
    result = await TransformersHiPolicy(CaptureGenerator(report)).decide(
        workspace_id="W1-1", model_view=view, actions=harness.eligible_actions())
    assert result == report


@pytest.mark.parametrize("operation,params", [
    ("get_frames", {"times_s": [0, 1.5]}), ("get_frames", {"range_s": [0, 4], "count": 8}),
    ("get_pages", {"pages": [1, 3]}), ("get_pages", {"page_range": [1, 3], "count": 3}),
    ("crop", {"box": [0.1, 0.2, 0.8, 0.9]}), ("locate_regions", {"query": "cat", "top_k": 2}),
    ("ocr", {"range_s": [2, 4]}), ("asr", {"range_s": [2, 4], "language": "en"}),
    ("audio_qa", {"question": "Which sounds?", "audio_track": 1}),
    ("visual_qa", {"question": "What is visible?"}),
    ("locate_frames", {"query": {"type": "text", "text": "cat"}, "range_s": [0, 4], "top_k": 3}),
    ("locate_frames", {"query": {"type": "text_image", "text": "cat", "evidence_id": "E3"}}),
    ("align_media", {"mode": "visual", "range_a_s": [0, 4], "range_b_s": [2, 6], "top_k": 2}),
])
def test_model_facing_inspect_schema_accepts_real_operations(operation, params):
    call = {"targets": ["E1", "E2"] if operation == "align_media" else ["E1"],
            "request": {"operation": operation, "params": params}}
    parse_inspect_call(call)
    jsonschema.validate(call, inspect_action_schema())


def test_irreducible_workspace_view_reports_budget_error():
    store = GlobalStore()
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "核验" * 1000, ()), store=store,
                               memory_policy=MemoryPolicy(max_model_view_chars=1000))
    with pytest.raises(ValueError, match="max_model_view_chars"):
        harness.model_view()


@pytest.mark.parametrize("operation,params,targets", [
    ("ocr", {}, ["E1", "E2"]),
    ("get_pages", {}, ["E1", "E2"]),
    ("asr", {}, ["E1", "E2"]),
    ("align_media", {"mode":"visual"}, ["E1"]),
    ("visual_qa", {"question":"Describe the images"}, ["E1", "E1"]),
])
def test_tool_schema_rejects_target_counts_rejected_by_parser(operation, params, targets):
    call = {"targets":targets, "request":{"operation":operation, "params":params}}
    with pytest.raises(ValueError):
        parse_inspect_call(call)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(call, inspect_action_schema())


@pytest.mark.asyncio
async def test_failed_investigation_cannot_publish_unsupported_factual_answer():
    generator = CaptureGenerator({"decision": "FINISH", "answer": "example.com can be registered; see W2-1."})
    result = await TransformersGiaPolicy(generator).decide(model_view={
        "task": "Check example.com registration rules", "latest_reports": [
            {"task_id": "W2-1", "run_status": "runtime_error"}],
        "runtime_constraints": {"can_create_round": False}}, history=(), actions=())
    assert "证据不足" in result["answer"]
    assert "can be registered" not in result["answer"]
    assert not generator.calls


def test_large_global_reports_remain_retrievable_within_total_budget():
    import json
    reports = [{"run_status":"reported", "report":{"task_id":f"W1-{i}", "status":"COMPLETED",
        "finding_ids":[f"F{i}"], "selected_evidence_ids":[f"E{i}"],
        "findings":[{"statement":"长结论" * 2000, "evidence_ids":[f"E{i}"]}]}} for i in range(1, 6)]
    view = bound_global_view({"task":"Compare all five reports", "original_materials":[],
        "current_materials":[], "history_index":[], "latest_reports":reports}, 2000)
    assert len(json.dumps(view, ensure_ascii=False)) <= 2000
    assert view["context_truncated"]
    assert [r["report"]["task_id"] for r in view["latest_reports"]] == [f"W1-{i}" for i in range(1, 6)]
    assert all(r["report"]["status"] == "COMPLETED" for r in view["latest_reports"])


@pytest.mark.asyncio
async def test_retained_original_material_is_not_mistaken_for_an_unread_finding_citation():
    report = {"task_id":"W1-1", "status":"COMPLETED", "findings":[{
        "statement":"The visual analysis supports this description", "evidence_ids":["E2"], "limitations":[]}],
        "selected_evidence_ids":["E1","E2"]}
    view = {"materials":{"E1":{"evidence_id":"E1"}}, "evidence_view":[{"evidence_id":"E2"}],
            "content_observed_evidence_ids":["E2"]}
    output = await TransformersHiPolicy(CaptureGenerator(report)).decide(
        workspace_id="W1-1", model_view=view, actions=())
    assert output == report


@pytest.mark.asyncio
async def test_gia_can_recover_existing_evidence_after_workspace_failure_at_round_limit():
    output = {"action":"READ_RECORD", "arguments":{"record_id":"E1"}}
    generator = CaptureGenerator(output)
    result = await TransformersGiaPolicy(generator).decide(model_view={
        "task":"Recover the evidence", "current_materials":[{"evidence_id":"E1"}],
        "latest_reports":[{"task_id":"W1-1", "run_status":"runtime_error"}],
        "runtime_constraints":{"can_create_round":False}}, history=(), actions=({"name":"READ_RECORD"},))
    assert result == output
    assert generator.calls


def test_constrained_nested_tool_arguments_keep_chinese_strings_open_until_complete():
    import json
    import string
    package = pytest.importorskip("lmformatenforcer")
    from foren_searcher.local_models import _create_output_enforcer, _decision_output_schema
    value = {"action":"INSPECT", "arguments":{"targets":["E1"],
        "request":{"operation":"visual_qa", "params":{"question":"画面中是否有猫？请说明依据。"}}},
        "focus_evidence_ids":["E1"]}
    rendered = json.dumps(value, ensure_ascii=False)
    characters = sorted(set(string.printable + rendered))
    regular = [(i, char, True) for i, char in enumerate(characters)]
    eos = len(characters)
    data = package.TokenEnforcerTokenizerData(regular,
        lambda ids: "".join(characters[i] for i in ids if 0 <= i < eos), eos, False, eos + 1)
    schema = _decision_output_schema({"workspace_id":"W1-1", "available_actions":[
        {"name":"INSPECT", "schema":inspect_action_schema()}]})
    enforcer = _create_output_enforcer(data, schema)
    prefix = [eos]
    for char in rendered:
        allowed = enforcer.get_allowed_tokens(prefix).allowed_tokens
        assert eos not in allowed
        token = characters.index(char)
        assert token in allowed, char
        prefix.append(token)
    assert eos in enforcer.get_allowed_tokens(prefix).allowed_tokens
