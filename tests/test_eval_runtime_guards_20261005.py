"""评测兜底（policies.RuntimeGuards，run_cases --runtime-guards eval）：claude/EVID基线架构修改方案_20261005.md 第 1 步。

默认关闭时行为与以前完全相同；打开后：重复动作短路、工作区撞上限自动 INCOMPLETE、FINISH 不引 E 第 3 次接受并打标、
GIA 耗尽时用草稿 / 报告 / 截断任务收尾、GIA 材料帧均匀抽、教师 JSON 模式。另有一条不受开关控制的语义等价整理：
get_frames / get_pages 的 count 恰好等于列表长度时删掉 count。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy
from foren_searcher.gia.runner import GiaRunner, GiaTurnLimit
from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceRunner, WorkspaceTurnLimit
from foren_searcher.policies import (EVAL_GUARDS, EVAL_PLUS_GUARDS, OutputRepairLimit, RuntimeGuards, active_guards, guard_state,
                                     runtime_guard_scope)
from foren_searcher.store import GlobalStore


def _evidence(store: GlobalStore, text: str = "A saved source sentence"):
    return store.record_evidence(content=text, label=text, source_ref="C0", locator="paragraph:1",
                                 acquired_by="TEST/source", discovered_by="W0")


class _AlwaysRead:
    def __init__(self, evidence_id: str) -> None:
        self.evidence_id, self.calls = evidence_id, 0

    async def decide(self, **kwargs):
        self.calls += 1
        return {"action": "READ_RECORD", "arguments": {"record_id": self.evidence_id}, "focus_evidence_ids": []}


def _run_hi(store, policy, evidence_id, max_turns=3):
    harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Read", (evidence_id,)))
    return asyncio.run(WorkspaceRunner(harness=harness, policy=policy, max_turns=max_turns).run())


def test_guards_are_off_by_default():
    assert active_guards() == RuntimeGuards() and guard_state() is None
    store = GlobalStore()
    evidence = _evidence(store)
    policy = _AlwaysRead(evidence.evidence_id)
    with pytest.raises(WorkspaceTurnLimit):            # 原行为：每次都执行，撞上限抛错
        _run_hi(store, policy, evidence.evidence_id)
    assert policy.calls == 3


def test_repeats_short_circuit_without_spending_turns_then_auto_incomplete():
    store = GlobalStore()
    evidence = _evidence(store)
    policy = _AlwaysRead(evidence.evidence_id)
    with runtime_guard_scope(EVAL_GUARDS) as state:
        report = _run_hi(store, policy, evidence.evidence_id)
    # 第 1 次执行；之后 3 次重复不耗轮次被短路；第 5 次重复预算已尽 → 自动 INCOMPLETE
    assert policy.calls == 5
    assert report.status == "INCOMPLETE" and report.status_reason.startswith("auto_report:")
    assert state.counters == {"repeated_action_short_circuit": 3, "auto_incomplete_report": 1}
    reads = [e for e in store.trajectory_events() if e["event_type"] == "observation"]
    assert sum("REPEATED_ACTION" in json.dumps(e["payload"]) for e in reads) == 3
    assert any(e["event_type"] == "auto_report" for e in store.trajectory_events())


def test_turn_limit_becomes_auto_incomplete_report_with_observed_evidence():
    store = GlobalStore()
    evidence = _evidence(store)

    class Wander:                                     # 每次换参数，不算重复
        def __init__(self):
            self.n = 0

        async def decide(self, **kwargs):
            self.n += 1
            return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id, "offset": self.n},
                    "focus_evidence_ids": []}

    with runtime_guard_scope(EVAL_GUARDS) as state:
        report = _run_hi(store, Wander(), evidence.evidence_id, max_turns=2)
    assert report.status == "INCOMPLETE" and "turn limit reached" in report.status_reason
    assert state.counters["auto_incomplete_report"] == 1


def test_redundant_count_is_dropped_only_when_it_equals_the_list_length():
    from foren_searcher.local_models import _drop_redundant_count

    def action(params, op="get_frames"):
        return {"action": "INSPECT", "arguments": {"targets": ["E1"], "request": {"operation": op, "params": params}},
                "focus_evidence_ids": []}

    same = action({"times_s": [1, 2, 3], "count": 3})
    _drop_redundant_count(same)
    assert same["arguments"]["request"]["params"] == {"times_s": [1, 2, 3]}
    pages = action({"pages": [1, 2], "count": 2}, "get_pages")
    _drop_redundant_count(pages)
    assert pages["arguments"]["request"]["params"] == {"pages": [1, 2]}
    different = action({"times_s": [0, 60], "count": 30})     # evidb4 里真实出现过：照旧交给契约拒绝
    _drop_redundant_count(different)
    assert different["arguments"]["request"]["params"] == {"times_s": [0, 60], "count": 30}


def test_spread_evenly_keeps_first_and_last_frames():
    from foren_searcher.model_images import _spread_evenly

    frames = [f"E{i}" for i in range(3, 33)]                    # 30 帧
    picked = _spread_evenly(frames, 8)
    assert len(picked) == 8 and picked[0] == "E3" and picked[-1] == "E32"
    assert _spread_evenly(frames[:5], 8) == frames[:5]


def test_uncited_finish_is_accepted_on_the_third_rejection(monkeypatch):
    from foren_searcher import local_models

    calls = {"n": 0}

    def fake_validate(output, model_view, history, actions, *, require_citation=True):
        if require_citation and output.get("decision") == "FINISH":
            raise ValueError(local_models._UNCITED_FINISH)

    class Gen:
        async def generate_json(self, *, system_prompt, payload):
            calls["n"] += 1
            return {"decision": "FINISH", "answer": "1. The video was spliced."}

    monkeypatch.setattr(local_models, "_validate_gia_output", fake_validate)
    monkeypatch.setattr(local_models, "_gia_turn_prompt", lambda view, history=(): "prompt")
    policy = local_models.TransformersGiaPolicy(Gen())
    view = {"direct_visual_input": True, "runtime_constraints": {"can_create_round": True}}
    with pytest.raises(local_models.LocalModelOutputError):     # 没开兜底：原样失败
        asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    with runtime_guard_scope(EVAL_GUARDS) as state:
        with pytest.raises(local_models.LocalModelOutputError):  # 第 1、2 次照旧拒绝
            asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
        output = asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    assert output["decision"] == "FINISH" and state.contract_flags == ["finish_without_citation"]
    assert state.last_finish_draft == "1. The video was spliced."


def _gia_view(store):
    return GlobalViewBuilder(store, MemoryPolicy()).build(task="Check", original_material_ids=(), latest_round_index=None)


def test_gia_forced_finish_uses_last_rejected_draft():
    store = GlobalStore()

    class DraftThenInvestigate:
        def __init__(self):
            self.n = 0

        async def decide(self, **kwargs):
            self.n += 1
            if self.n == 1:   # 多了一个字段 → 契约拒绝，但答案留作草稿
                return {"decision": "FINISH", "answer": "1. Draft point.", "extra": True}
            return {"decision": "INVESTIGATE", "tasks": [{"task": "more", "materials": []}]}

    runner = GiaRunner(store=store, policy=DraftThenInvestigate(), memory_policy=MemoryPolicy(), max_turns=8)
    with pytest.raises((OutputRepairLimit, GiaTurnLimit)):
        asyncio.run(runner.run(_gia_view(store), round_index=2, can_create_round=False))
    with runtime_guard_scope(EVAL_GUARDS) as state:
        runner = GiaRunner(store=store, policy=DraftThenInvestigate(), memory_policy=MemoryPolicy(), max_turns=8)
        decision = asyncio.run(runner.run(_gia_view(store), round_index=2, can_create_round=False))
    assert decision.to_dict() == {"decision": "FINISH", "answer": "1. Draft point."}
    assert state.contract_flags == ["forced_finish_from_draft"]
    assert any(e["event_type"] == "forced_decision" for e in store.trajectory_events())


def test_gia_over_limit_tasks_are_truncated():
    store = GlobalStore()

    class TooMany:
        async def decide(self, **kwargs):
            return {"decision": "INVESTIGATE", "tasks": [{"task": f"t{i}", "materials": []} for i in range(5)]}

    def validator(decision):
        if len(decision.tasks) > 3:
            raise ValueError("INVESTIGATE tasks exceed the configured limit of 3")

    with runtime_guard_scope(EVAL_GUARDS) as state:
        runner = GiaRunner(store=store, policy=TooMany(), memory_policy=MemoryPolicy(), max_turns=8)
        decision = asyncio.run(runner.run(_gia_view(store), round_index=1, can_create_round=True, validator=validator))
    assert [task.task for task in decision.tasks] == ["t0", "t1", "t2"]
    assert state.contract_flags == ["forced_task_truncation"]


def test_answer_from_reports_numbers_findings():
    from foren_searcher.gia.runner import _answer_from_reports

    view = {"latest_reports": [{"report": {"findings": [{"statement": "A"}, {"statement": "B"}]}},
                               {"report": {"findings": [{"statement": "A"}, {"statement": "C"}]}}]}
    assert _answer_from_reports(view) == "1. A\n2. B\n3. C"
    assert _answer_from_reports({"latest_reports": []}) is None


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": '{"decision": "FINISH", "answer": "1. x"}'},
                                                  "finish_reason": "stop"}], "usage": {}})


def test_teacher_json_mode_and_fallback(tmp_path: Path):
    from foren_searcher.datagen.teacher import EndpointTeacher

    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, text='{"error":"unknown field response_format"}')
        return _ok()

    teacher = EndpointTeacher(store=None, output_dir=tmp_path / "r", endpoint="http://127.0.0.1:11434/v1", model="q",
                              transport=httpx.MockTransport(handler), sleep=lambda s: None)
    asyncio.run(teacher.generate_json(system_prompt="S", payload={"global_view": {}}))
    assert seen == [False]                                      # 默认不发
    with runtime_guard_scope(EVAL_GUARDS):
        asyncio.run(teacher.generate_json(system_prompt="S", payload={"global_view": {}}))
        asyncio.run(teacher.generate_json(system_prompt="S", payload={"global_view": {}}))
    assert seen == [False, True, False, False]                  # 被拒后去掉重发，之后不再发
    transport = json.loads((tmp_path / "r" / "request_0002" / "transport.json").read_text(encoding="utf-8"))
    assert transport["returncode"] == 0 and transport["attempts"][0]["json_mode_rejected"] is True


def test_run_cases_passes_runtime_guards_flag():
    from foren_searcher.datagen import run_cases

    assert run_cases.RUNTIME_GUARD_MODES == ("off", "eval", "eval_plus")
    item = run_cases.Planned("c.json", "evid:video_007", "test", 1, "r", "todo")
    base = dict(round="t0", runs_dir="runs", teacher="endpoint", specs="specs", hi_spec="hi@0.5.0", gia_spec="gia@0.6.0",
                no_think=False, blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s",
                tool_cache="off", tool_cache_root="t", harness_params="apply", inspect_ops="auto", harness_override=None,
                instruction_mode="teacher", query_guard="on", video_search="ytdlp")
    assert "--runtime-guards" not in run_cases._worker_command(argparse.Namespace(**base, runtime_guards="off"), item, "e", "m")
    assert run_cases._worker_command(argparse.Namespace(**base, runtime_guards="eval"), item, "e", "m")[-2:] == \
        ["--runtime-guards", "eval"]


# ---------------------------------------------------------------- 第 2 步（eval_plus，改模型输入）


def test_eval_does_not_change_model_inputs_but_eval_plus_does():
    for name in ("state_dependent_actions", "gia_limits_visible", "open_as_video_hint"):
        assert getattr(EVAL_GUARDS, name) is False and getattr(EVAL_PLUS_GUARDS, name) is True


def test_gia_sees_limits_and_finish_only_under_eval_plus():
    store = GlobalStore()
    seen = []

    class Finish:
        async def decide(self, *, model_view, history, actions):
            seen.append(model_view["runtime_constraints"])
            return {"decision": "FINISH", "answer": "1. x"}

    for guards in (EVAL_GUARDS, EVAL_PLUS_GUARDS):
        runner = GiaRunner(store=store, policy=Finish(), memory_policy=MemoryPolicy(), max_turns=2)
        runner.task_limits = {"max_tasks_per_round": 3, "max_materials_per_task": 8}
        with runtime_guard_scope(guards):
            asyncio.run(runner.run(_gia_view(store), round_index=2, can_create_round=False))
    assert "next_output" not in seen[0] and "max_tasks_per_round" not in seen[0]
    assert seen[1]["next_output"].startswith("FINISH only") and seen[1]["max_tasks_per_round"] == 3


def test_hi_last_turn_offers_no_tools_under_eval_plus():
    store = GlobalStore()
    evidence = _evidence(store)
    catalogs = []

    class Record:
        async def decide(self, *, workspace_id, model_view, actions):
            catalogs.append((len(actions), model_view["runtime_constraints"].get("next_output")))
            if not actions:
                return {"task_id": workspace_id, "status": "INCOMPLETE", "findings": [],
                        "selected_evidence_ids": [], "status_reason": "out of turns"}
            return {"action": "READ_RECORD", "arguments": {"record_id": evidence.evidence_id, "offset": len(catalogs)},
                    "focus_evidence_ids": []}

    with runtime_guard_scope(EVAL_PLUS_GUARDS):
        report = _run_hi(store, Record(), evidence.evidence_id, max_turns=2)
    assert report.status == "INCOMPLETE" and report.status_reason == "out of turns"
    assert catalogs[0][0] > 0 and catalogs[0][1] is None
    assert catalogs[1] == (0, "WORKSPACE REPORT only: this is the last turn; tool actions will be rejected")


def test_search_cards_offer_open_and_compare_calls():
    """10-05 修正优化计划 P1.4：下载由 INSPECT 对候选页面 E 触发（SEARCH 不下载），提示给的是 INSPECT 调用。"""
    from types import SimpleNamespace

    store = GlobalStore()
    video = store.record_evidence(content="material video", modalities=("video",), label="material video", source_ref="C0",
                                  locator="file", acquired_by="TEST/source", discovered_by="W0")
    harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Compare", (video.evidence_id,)))
    cid, _ = store.get_or_create_candidate(identity_key="url:https://www.youtube.com/watch?v=abc",
                                           source_url="https://www.youtube.com/watch?v=abc", candidate_type="video")
    page = SimpleNamespace(evidence_id="E7")
    pending = SimpleNamespace(candidate_id=cid, source_url="https://www.youtube.com/watch?v=abc",
                              materials=(SimpleNamespace(kind="page", status="ready", evidence=(page,)),))
    open_hint, compare = harness._comparison_hints(pending)
    assert open_hint.startswith(f"open_as_video[{cid}]: INSPECT get_frames/ocr/asr/align_media on E7 downloads")
    assert '"operation": "get_frames"' in open_hint and '"targets": ["E7"]' in open_hint and '"SEARCH"' not in open_hint
    assert compare.startswith(f"compare_with_material[{cid}]:") and f'"targets": ["{video.evidence_id}", "E7"]' in compare
    opened = SimpleNamespace(candidate_id=cid, source_url="https://www.youtube.com/watch?v=abc",
                             materials=(SimpleNamespace(kind="video", status="ready", evidence=(SimpleNamespace(evidence_id="E99"),)),))
    [compare] = harness._comparison_hints(opened)
    assert compare.startswith(f"compare_with_material[{cid}]:") and f'"targets": ["{video.evidence_id}", "E99"]' in compare
    other, _ = store.get_or_create_candidate(identity_key="url:https://example.org/a", source_url="https://example.org/a",
                                             candidate_type="page")
    page_only = SimpleNamespace(candidate_id=other, source_url="https://example.org/a",
                                materials=(SimpleNamespace(kind="page", status="ready", evidence=(page,)),))
    assert harness._comparison_hints(page_only) == []          # 非视频候选：对页面 E 取帧不会下载，不给提示


# ---------------------------------------------------------------- 重复点检查（10-05 晚）


def test_answer_points_dedupe_keeps_first_and_renumbers():
    from foren_searcher.answer_points import dedupe_answer, duplicate_indices, split_points

    same = "The original shows an iPhone 14 [E4]. Material A relabels it as iPhone 18."
    answer = f"Summary:\n1. {same}\n2. {same.replace('[E4]', '(E6)')}\n3. A different point about the audio.\n4. {same}"
    intro, points = split_points(answer)
    assert intro == "Summary:" and len(points) == 4
    assert duplicate_indices(points) == [1, 3]                       # 证据编号不同也算同一点
    new, removed = dedupe_answer(answer)
    assert removed == 2 and new == f"Summary:\n1. {same}\n2. A different point about the audio."
    assert dedupe_answer("1. a cat\n2. a dog") == ("1. a cat\n2. a dog", 0)
    assert dedupe_answer("no numbering here") == ("no numbering here", 0)


def test_duplicate_points_are_rejected_once_then_deduplicated(monkeypatch):
    from foren_searcher import local_models

    point = "The original shows a phone; Material A relabels it, creating a false impression."
    answers = iter([f"1. {point}\n2. {point}\n3. {point}"] * 4)
    prompts = []

    class Gen:
        async def generate_json(self, *, system_prompt, payload):
            prompts.append(payload.get("validation_error"))
            return {"decision": "FINISH", "answer": next(answers)}

    monkeypatch.setattr(local_models, "_validate_gia_output", lambda *a, **k: None)
    monkeypatch.setattr(local_models, "_gia_turn_prompt", lambda view, history=(): "prompt")
    policy = local_models.TransformersGiaPolicy(Gen())
    view = {"direct_visual_input": True, "runtime_constraints": {"can_create_round": True}}
    output = asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    assert output["answer"].count("The original shows") == 3                                   # 没开兜底：原样
    with runtime_guard_scope(EVAL_GUARDS) as state:
        output = asyncio.run(policy.decide(model_view=dict(view), history=(), actions=()))
    assert output["answer"] == f"1. {point}"
    assert "repeats the same point" in prompts[-1]                    # 第一次被退回，提示写进了重试输入
    assert state.contract_flags == ["finish_points_deduplicated"]
    assert state.counters["duplicate_points_rejected"] == 1 and state.counters["duplicate_points_removed"] == 2
