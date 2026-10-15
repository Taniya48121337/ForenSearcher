"""EVID 修正优化计划（claude/EVID修正优化计划_20261005.md）P0 / P1 的回归测试。

P0（不改模型输入）：--state-renderer 强制档位并让端点按档位渲染、schema 约束解码（退回 json_object）、行为面板、答案语言。
P1（改模型输入，规格 hi@0.6.0 / gia@0.7.0 的 harness_params 打开）：入库关键帧、GIA 原件帧优先、证据卡 origin 与
candidate_materials、候选可供性、跨轮交接（R 作材料 + 工作区预留原件帧）、约束可见（含 GIA 最后一个回合只给决定）。
旧规格 / 开关全关时与以前逐字相同。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.datagen import run_cases
from foren_searcher.datagen.harness_params import InputFeatures, build_harness_settings, resolve_harness_params
from foren_searcher.datagen.specs import Spec, validate_bump
from foren_searcher.datagen.teacher import EndpointTeacher
from foren_searcher.hi.workspace import WorkspaceHarness
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy
from foren_searcher.model_images import ModelImageInputs
from foren_searcher.policies import EVAL_GUARDS, RuntimeGuards, merge_input_features, runtime_guard_scope
from foren_searcher.state_render import TIERS, card_line
from foren_searcher.store import GlobalStore
from foren_searcher.view_annotations import OriginResolver, annotate_view, original_frames, reserved_original_frames

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
P1 = RuntimeGuards(ingest_keyframes=8, gia_original_first=True, evidence_origin=True, candidate_affordance=True,
                   cross_round_handoff=True, workspace_original_slots=4, state_dependent_actions=True,
                   gia_limits_visible=True, open_as_video_hint=True)


def _png(color: tuple[int, int, int] = (200, 10, 10)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 18), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _video_store(times: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0, 120.0, 300.0, 400.0, 470.0)):
    """原件 E1（视频）+ 媒体属性 + 若干原件帧（真图片），调查已登记。"""

    store = GlobalStore()
    store.add_material("C0", {"task": "t"})
    video = store.record_evidence(content="{}", modalities=("video",), label="材料 A", source_ref="I0",
                                  locator='{"scope":"complete_media"}', acquired_by="HARNESS/ingest", discovered_by="HARNESS")
    store.ensure_investigation(task="t", original_material_ids=["C0", video.evidence_id])
    store.commit_derived_evidence(derived_from=(video.evidence_id,), modalities=("text",), label="媒体属性：video，477.233秒",
                                  content=json.dumps({"duration_s": 477.233}), locator='{"probe":1}',
                                  acquired_by="INSPECT/media_profile", discovered_by="HARNESS")
    frames = []
    for t in times:
        locator = json.dumps({"time_s": t, "frame_duration_s": 0.0333})
        frames.append(store.commit_derived_evidence(
            derived_from=(video.evidence_id,), modalities=("image",), label=f"视频{video.evidence_id}在{t:.3f}秒的帧",
            content=locator, locator=locator, acquired_by="INSPECT/get_frames", discovered_by="W1-1",
            artifact_bytes=_png(), media_type="image/png", suffix=".png", role="video_frame").evidence_id)
    return store, video.evidence_id, frames


def _candidate(store: GlobalStore, *, kind: str = "video"):
    cid, _ = store.get_or_create_candidate(identity_key=f"url:https://www.youtube.com/watch?v={kind}",
                                           source_url=f"https://www.youtube.com/watch?v={kind}", candidate_type=kind)
    page = store.commit_material_evidence(candidate_id=cid, material_kind="page", material_status="ready",
                                          artifact_bytes=b'{"title": "x"}', media_type="application/json", suffix=".json",
                                          modalities=("text",), label="Some other video", content="page text",
                                          locator='{"kind":"page"}', acquired_by="SEARCH/open/yt-dlp", discovered_by="W1-1")
    cover = store.commit_material_evidence(candidate_id=cid, material_kind="cover", material_status="ready",
                                           artifact_bytes=_png((0, 0, 200)), media_type="image/png", suffix=".png",
                                           modalities=("image",), label="cover: Some other video", content="{}",
                                           locator='{"kind":"cover"}', acquired_by="SEARCH/open/yt-dlp+http", discovered_by="W1-1")
    return cid, page.evidence_id, cover.evidence_id


# ---------------------------------------------------------------- 规格与参数


def test_new_specs_turn_on_input_features_old_specs_stay_off():
    from foren_searcher.config import load_config

    config = load_config(ROOT / "configs" / "evid_local.toml")
    new = build_harness_settings(config, Spec.load(SPECS / "hi@0.6.0"), Spec.load(SPECS / "gia@0.7.0"))
    assert new.memory_policy.state_renderer == "compact_16k"
    assert new.input_features == InputFeatures(8, True, True, True, True, 4, True)
    old = build_harness_settings(config, Spec.load(SPECS / "hi@0.5.0"), Spec.load(SPECS / "gia@0.6.0"))
    assert old.input_features.enabled() == {} and old.memory_policy.state_renderer == "legacy"
    assert merge_input_features(None, {}) is None and merge_input_features(EVAL_GUARDS, {}) is EVAL_GUARDS
    merged = merge_input_features(EVAL_GUARDS, new.input_features.enabled())
    assert merged.json_schema and merged.ingest_keyframes == 8 and merged.workspace_original_slots == 4
    assert merged.state_dependent_actions and merged.gia_limits_visible and merged.open_as_video_hint
    applied, _ = resolve_harness_params(Spec.load(SPECS / "hi@0.6.0"), Spec.load(SPECS / "gia@0.7.0"))
    assert applied["workspace_original_slots"]["source"] == "hi@0.6.0"
    assert applied["cross_round_handoff"]["source"] == "gia@0.7.0"


def test_new_specs_keep_prompts_and_pass_bump_rules():
    for parent, child in (("hi@0.5.0", "hi@0.6.0"), ("gia@0.6.0", "gia@0.7.0")):
        p, c = Spec.load(SPECS / parent), Spec.load(SPECS / child)
        assert c.student_prompt == p.student_prompt and c.teacher_instruction == p.teacher_instruction
        assert validate_bump(p, c) == []
        targets = set(c.changelog[-1]["targets"])
        assert {"evidence_dropped", "wrong_tool", "task_not_gap_driven", "premature_finish"} <= targets
    # 只改教师指令仍应升 patch
    hi = Spec.load(SPECS / "hi@0.5.0")
    teacher_only = replace(hi, version="0.6.0", parent="0.5.0", teacher_instruction=hi.teacher_instruction + "x",
                           changelog=({"targets": ["wrong_tool"], "rationale": "t"},))
    assert any("应升 patch" in problem for problem in validate_bump(hi, teacher_only))


def test_worker_command_forwards_state_renderer_and_max_rounds():
    args = argparse.Namespace(round="evidb6", runs_dir="runs", teacher="endpoint", specs="specs", hi_spec="hi@0.5.0",
                              gia_spec="gia@0.6.0", no_think=False, blocklist="exclude", date_cutoff=None,
                              search_mode="live", snapshot_root="s", tool_cache="off", tool_cache_root="t",
                              harness_params="apply", inspect_ops="auto", instruction_mode="student",
                              runtime_guards="eval", state_renderer="compact_16k", max_rounds=2, harness_override=None)
    item = run_cases.Planned("case.json", "evid:video_002", "test", 1, "evid_video_002_evidb6_1", "todo")
    command = run_cases._worker_command(args, item, "http://x/v1", "m")
    assert command[command.index("--state-renderer") + 1] == "compact_16k"
    assert command[command.index("--max-rounds") + 1] == "2"


# ---------------------------------------------------------------- P0.1 / P0.2 端点教师


def _hi_payload(renderer: str) -> dict:
    history = [{"action": "SEARCH", "request": {"query": "q"}, "focus_evidence_ids": [],
                "observation": {"action": "SEARCH", "items": ["[R1] u"], "details": {"results": [{"blob": "x" * 9000}]}}}
               for _ in range(8)]
    return {"available_actions": [{"name": "READ_RECORD", "kind": "harness",
                                   "schema": {"type": "object", "required": ["record_id"], "additionalProperties": False,
                                              "properties": {"record_id": {"type": "string"}}}}],
            "workspace_id": "W1-1",
            "model_view": {"task_id": "W1-1", "task": "t", "materials": {}, "candidate_sources": [], "evidence_view": [],
                           "history": history, "latest_observation": None,
                           "runtime_constraints": {"state_renderer": renderer, "turn_index": 3, "max_turns": 8}}}


def _teacher(tmp_path: Path, seen: list, *, render: bool, reject_schema: bool = False) -> EndpointTeacher:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if reject_schema and (body.get("response_format") or {}).get("type") == "json_schema":
            return httpx.Response(400, text='{"error":"invalid JSON schema in format"}')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"action":"READ_RECORD","arguments":{"record_id":"E1"},"focus_evidence_ids":[]}'},
                                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return EndpointTeacher(store=None, output_dir=tmp_path / "requests", endpoint="http://127.0.0.1:11434/v1", model="m",
                           thinking=False, transport=httpx.MockTransport(handler), sleep=lambda s: None, render_inputs=render)


def test_endpoint_teacher_renders_state_when_asked(tmp_path: Path):
    seen: list = []
    asyncio.run(_teacher(tmp_path / "a", seen, render=True).generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
    request = json.loads((tmp_path / "a" / "requests" / "request_0001" / "request.json").read_text(encoding="utf-8"))
    assert request["state_render"]["renderer"] == "compact_16k"
    assert request["state_render"]["rendered_chars"] < request["state_render"]["raw_chars"] <= 10 ** 6
    assert request["payload"]["model_view"]["history"][0]["observation"]["details"]["results"]      # 原视图照存
    sent = seen[0]["messages"][1]["content"]
    assert "x" * 9000 not in sent and len(sent) < 30_000                                             # 发出去的是渲染后的
    # 教师模式（render_inputs=False）或 legacy 档位：与以前一样发原视图
    seen.clear()
    asyncio.run(_teacher(tmp_path / "b", seen, render=False).generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
    assert "x" * 9000 in seen[0]["messages"][1]["content"]
    seen.clear()
    asyncio.run(_teacher(tmp_path / "c", seen, render=True).generate_json(system_prompt="S", payload=_hi_payload("legacy")))
    assert "x" * 9000 in seen[0]["messages"][1]["content"]
    assert "state_render" not in json.loads((tmp_path / "c" / "requests" / "request_0001" / "request.json").read_text(encoding="utf-8"))


def test_json_schema_decoding_and_fallback(tmp_path: Path):
    seen: list = []
    with runtime_guard_scope(EVAL_GUARDS):
        asyncio.run(_teacher(tmp_path / "a", seen, render=True).generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema" and "anyOf" in fmt["json_schema"]["schema"]
    branches = fmt["json_schema"]["schema"]["anyOf"]
    assert any(b["properties"].get("action", {}).get("enum") == ["READ_RECORD"] for b in branches)
    assert any(b["properties"].get("status", {}).get("const") == "COMPLETED" for b in branches)     # HI 报告也在 schema 里
    transport = json.loads((tmp_path / "a" / "requests" / "request_0001" / "transport.json").read_text(encoding="utf-8"))
    assert transport["response_format"] == "json_schema"
    # 服务拒收 schema → 本次立即改发 json_object，之后都不再发 schema
    seen.clear()
    teacher = _teacher(tmp_path / "b", seen, render=True, reject_schema=True)
    with runtime_guard_scope(EVAL_GUARDS):
        asyncio.run(teacher.generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
        asyncio.run(teacher.generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
    assert [b.get("response_format", {}).get("type") for b in seen] == ["json_schema", "json_object", "json_object"]
    # 档位关闭时不带 response_format（旧行为）
    seen.clear()
    asyncio.run(_teacher(tmp_path / "c", seen, render=True).generate_json(system_prompt="S", payload=_hi_payload("compact_16k")))
    assert "response_format" not in seen[0]


# ---------------------------------------------------------------- P1.3 / P1.4 展示层


def test_origin_classification_and_card_annotations():
    store, video, frames = _video_store()
    cid, page, cover = _candidate(store)
    crop = store.commit_derived_evidence(derived_from=(frames[0],), modalities=("image",), label="crop", content="{}",
                                         locator="{}", acquired_by="INSPECT/crop", discovered_by="W1-1",
                                         artifact_bytes=_png(), media_type="image/png", suffix=".png").evidence_id
    mixed = store.record_evidence(content="{}", label="INSPECT align_media 结果", locator="{}", acquired_by="INSPECT/align_media",
                                  discovered_by="W1-1", derived_from=(video, page)).evidence_id
    origin = OriginResolver(store)
    assert [origin(e) for e in (video, frames[0], crop, page, cover, mixed)] == [
        "original", "derived_from_original", "derived_from_original", "candidate_source", "candidate_source", "tool_result"]
    view = {"materials": {video: {"evidence_id": video, "modalities": ["video"], "label": "材料 A", "acquired_by": "HARNESS/ingest"}},
            "evidence_view": [{"evidence_id": page, "modalities": ["text"], "label": "Some other video", "source_ref": cid,
                               "derived_from": [], "acquired_by": "SEARCH/open/yt-dlp"}],
            "candidate_sources": [{"record_type": "candidate", "candidate_id": cid, "source_url": "https://x", "candidate_type": "video"}]}
    before = json.dumps(view, sort_keys=True)
    annotate_view(view, store)                       # 开关全关：不动
    assert json.dumps(view, sort_keys=True) == before
    shared = view["materials"]
    with runtime_guard_scope(P1):
        annotate_view(view, store)
    assert shared[video] == {"evidence_id": video, "modalities": ["video"], "label": "材料 A", "acquired_by": "HARNESS/ingest"}
    card = view["materials"][video]
    assert card["origin"] == "original" and card["duration_s"] == 477.233
    assert view["evidence_view"][0]["origin"] == "candidate_source" and "downloads the source video" in view["evidence_view"][0]["video_access"]
    r_card = view["candidate_sources"][0]
    assert r_card["page_evidence_id"] == page and r_card["cover_evidence_id"] == cover and page in r_card["video_access"]
    tier = TIERS["compact_16k"]
    assert card_line(card, tier).endswith("origin=original duration_s=477.233")
    assert "video_access:" in card_line(view["evidence_view"][0], tier)
    assert f"page={page} cover={cover}" in card_line(r_card, tier) and "video_access:" in card_line(r_card, tier)
    plain = {"evidence_id": "E5", "modalities": ["image"], "label": "x", "derived_from": ["E1"], "acquired_by": "INSPECT/get_frames"}
    assert "origin" not in card_line(plain, tier)        # 旧卡片输出逐字不变


def test_gia_view_lists_candidate_materials_separately():
    store, video, frames = _video_store(times=(0.0, 60.0))
    cid, page, cover = _candidate(store)
    store.start_round(1, gia_input={}, gia_decision={"decision": "INVESTIGATE", "tasks": [{"task": "t", "materials": [video]}]},
                      tasks=(WorkspaceTask("W1-1", "t", (video,)),))
    for reference in (frames[0], page, cover):
        store.record_read(actor_role="HI", actor_id="W1-1", reference_id=reference, round_index=1)
    builder = GlobalViewBuilder(store, MemoryPolicy())
    old = builder.build(task="t", original_material_ids=("C0", video), latest_round_index=1).to_dict()
    assert "candidate_materials" not in old and {page, cover} <= {c.get("evidence_id") for c in old["current_materials"]}
    with runtime_guard_scope(P1):
        new = builder.build(task="t", original_material_ids=("C0", video), latest_round_index=1).to_dict()
    assert [c["evidence_id"] for c in new["candidate_materials"]] == [page, cover]
    assert {page, cover}.isdisjoint(c.get("evidence_id") for c in new["current_materials"])
    assert frames[0] in [c.get("evidence_id") for c in new["current_materials"]]


# ---------------------------------------------------------------- P1.2 GIA 附图优先级


def test_gia_images_put_original_frames_first():
    times = tuple(float(t) for t in range(0, 480, 16))          # 30 帧，模拟 evidb4 的 30 帧
    store, video, frames = _video_store(times=times)
    cid, page, cover = _candidate(store)
    covers = [cover] + [store.commit_material_evidence(
        candidate_id=cid, material_kind="cover", material_status="ready", artifact_bytes=_png((0, 0, i)),
        media_type="image/png", suffix=".png", modalities=("image",), label=f"cover {i}", content="{}",
        locator=json.dumps({"kind": "cover", "i": i}), acquired_by="SEARCH/open/yt-dlp+http", discovered_by="W1-1").evidence_id
        for i in range(1, 6)]
    view = {"task": "t", "original_materials": [{"record_type": "evidence", "evidence_id": video, "derived_from": []}],
            "current_materials": [{"record_type": "evidence", "evidence_id": f, "derived_from": [video]} for f in frames[:3]],
            "latest_reports": [{"run_status": "reported", "report": {"status": "COMPLETED", "findings": [
                {"statement": "s", "evidence_ids": covers, "limitations": []}], "selected_evidence_ids": covers}}]}
    images = ModelImageInputs(store=store)
    old = dict(view)
    images.prepare_global_model_view(old)
    assert len([i for i in old["image_inputs"] if i["evidence_id"] in frames]) == 3      # 旧：只有本轮 HI 读过的 3 帧
    with runtime_guard_scope(P1):
        new = dict(view)
        images.prepare_global_model_view(new)
    chosen = [i["evidence_id"] for i in new["image_inputs"]]
    originals = [e for e in chosen if e in frames]
    assert len(chosen) == 8 and len(originals) >= 4 and chosen[: len(originals)] == originals    # 原件帧在前、至少一半
    times_chosen = [times[frames.index(e)] for e in originals]
    assert times_chosen[0] == 0.0 and times_chosen[-1] == times[-1]                              # 均匀覆盖全片
    assert [e for e in chosen if e in covers]                                                    # 报告引用填余量
    assert new["original_materials"][0]["origin"] == "original"                                  # annotate_view 也跑了


# ---------------------------------------------------------------- P1.5 跨轮交接


def test_workspace_reserves_original_frames_and_attaches_them_on_first_turn():
    times = tuple(float(t) for t in range(0, 480, 16))
    store, video, frames = _video_store(times=times)
    cid, page, cover = _candidate(store)
    for i in range(14):                                            # 14 张同名卡把 12 个名额占满
        store.record_evidence(content=f"Material A note {i}", label=f"Material A note {i}", locator="x",
                              acquired_by="SEARCH/open", discovered_by="W1-1")
    task = WorkspaceTask("W2-1", "Compare Material A", (page, cover))
    assert not set(frames) & {c["evidence_id"] for c in WorkspaceHarness(store=store, task=task).memory.evidence_view}
    with runtime_guard_scope(P1):
        harness = WorkspaceHarness(store=store, task=task)
        cards = [c["evidence_id"] for c in harness.memory.evidence_view]
        reserved = reserved_original_frames(store, 4)
        assert len(reserved) == 4 and set(reserved) <= set(cards) and len(cards) <= 12
        assert [times[frames.index(e)] for e in reserved][0] == 0.0
        view = harness.model_view(runtime_constraints={"turn_index": 1, "max_turns": 8})
        ModelImageInputs(store=store).prepare_model_view(view)
        attached = [i["evidence_id"] for i in view["image_inputs"]]
        assert set(reserved) <= set(attached)                       # 首轮附图里有材料 A 的帧
        assert view["materials"][page]["origin"] == "candidate_source"


def test_gia_may_hand_a_candidate_to_the_next_round():
    from foren_searcher.contracts import parse_gia_decision
    from foren_searcher.gia.runtime import InvestigationRuntime
    from foren_searcher.local_models import _gia_visible_reference_ids

    store, video, frames = _video_store(times=(0.0,))
    cid, page, cover = _candidate(store)
    runtime = InvestigationRuntime(store=store, gia_policy=None, hi_policy=None)
    decision = parse_gia_decision({"decision": "INVESTIGATE", "tasks": [{"task": "Compare with the candidate", "materials": [video, cid]}]})
    with pytest.raises(Exception):
        runtime._validate_tasks(decision)                           # 旧行为：R 不能当材料
    view = {"candidate_materials": [{"evidence_id": page, "source_ref": cid}]}
    assert cid not in _gia_visible_reference_ids(view, ())
    with runtime_guard_scope(P1):
        runtime._validate_tasks(decision)
        assert cid in _gia_visible_reference_ids(view, ())
        bogus = parse_gia_decision({"decision": "INVESTIGATE", "tasks": [{"task": "x", "materials": ["R99"]}]})
        with pytest.raises(KeyError):
            runtime._validate_tasks(bogus)
        harness = WorkspaceHarness(store=store, task=WorkspaceTask("W2-1", "Compare", (video, cid)))
        assert harness.memory.candidate_source_ids == [cid]
        assert {page, cover} <= set(harness.memory.materials)
        harness.restore_working_memory()
        assert harness.memory.candidate_source_ids == [cid]


# ---------------------------------------------------------------- P1.1 入库关键帧


class _FakeInspect:
    def __init__(self, store: GlobalStore, video: str) -> None:
        self.store, self.video, self.calls = store, video, []

    async def execute(self, call, *, workspace_id):
        from types import SimpleNamespace

        self.calls.append((call["request"]["params"]["times_s"], workspace_id))
        if len(self.calls) == 1:
            self.store.commit_derived_evidence(derived_from=(self.video,), modalities=("text",), label="媒体属性",
                                               content=json.dumps({"duration_s": 80.0}), locator='{"p":1}',
                                               acquired_by="INSPECT/media_profile", discovered_by=workspace_id)
        frames = [SimpleNamespace(evidence_id=f"E{100 + i}") for i, _ in enumerate(call["request"]["params"]["times_s"])]
        return SimpleNamespace(status="ready", result=SimpleNamespace(frames=frames), feedback=())

    async def aclose(self):
        pass


def test_ingest_keyframes_spread_over_the_whole_video():
    from foren_searcher.gia.runtime import InvestigationRuntime

    store = GlobalStore()
    video = store.record_evidence(content="{}", modalities=("video",), label="材料 A", source_ref="I0", locator="{}",
                                  acquired_by="HARNESS/ingest", discovered_by="HARNESS").evidence_id
    store.ensure_investigation(task="t", original_material_ids=[video])
    inspect = _FakeInspect(store, video)
    runtime = InvestigationRuntime(store=store, gia_policy=None, hi_policy=None, inspect_service=inspect)
    asyncio.run(runtime._ingest_keyframes((video,), 8))
    assert inspect.calls[0] == ([0.0], "HARNESS")
    assert inspect.calls[1][0] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    [event] = store.trajectory_events(actor_role="HARNESS", actor_id="HARNESS", event_type="ingest_frames")
    assert event["payload"]["duration_s"] == 80.0 and len(event["payload"]["frame_evidence_ids"]) == 8


# ---------------------------------------------------------------- P1.6 GIA 最后一个回合


def test_gia_last_turn_offers_only_decisions():
    from foren_searcher.gia.runner import GiaRunner

    store = GlobalStore()
    store.add_material("C0", {"task": "t"})
    seen: list = []

    class Policy:
        async def decide(self, *, model_view, history, actions):
            seen.append((model_view["runtime_constraints"].get("next_output"), [a["name"] for a in actions]))
            if actions:
                return {"action": "QUERY_MEMORY", "arguments": {"query": "x"}}
            return {"decision": "FINISH", "answer": "done"}

    view = GlobalViewBuilder(store, MemoryPolicy()).build(task="t", original_material_ids=("C0",), latest_round_index=None)
    runner = GiaRunner(store=store, policy=Policy(), memory_policy=MemoryPolicy(), max_turns=3, turn_timeout_s=None)
    with runtime_guard_scope(P1):
        decision = asyncio.run(runner.run(view, round_index=1, can_create_round=True))
    assert decision.decision == "FINISH"
    assert seen[-1][1] == [] and seen[-1][0].startswith("INVESTIGATE or FINISH only")
    assert all(names for _, names in seen[:-1])


# ---------------------------------------------------------------- P0.3 / P0.4 行为面板


def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "evid_video_009_evidbx_1"
    (run / "requests" / "request_0001").mkdir(parents=True)
    (run / "requests" / "request_0001" / "transport.json").write_text(json.dumps(
        {"usage": {"input_tokens": 30000}, "context_retries": [{"images_before": 8}], "json_mode": True}), encoding="utf-8")
    frame = lambda e, t: {"evidence_id": e, "modalities": ["image"], "label": f"视频E1在{t:.3f}秒的帧", "derived_from": ["E1"],  # noqa: E731
                          "acquired_by": "INSPECT/get_frames"}
    events = [
        {"actor_role": "HARNESS", "actor_id": "HARNESS", "event_type": "investigation_run", "payload": {"original_material_ids": ["C0", "E1"]}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "raw_model_output", "payload": {"image_evidence_ids": []}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "model_output", "payload": {"output": {
            "action": "INSPECT", "arguments": {"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [0, 60]}}}}}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "observation", "payload": {"observation": {"action": "INSPECT", "details": {
            "operation": "get_frames", "result": {"frames": [{"evidence_id": "E3", "time_s": 0.0}, {"evidence_id": "E4", "time_s": 60.0}]},
            "evidence": [{"evidence_id": "E2", "modalities": ["text"], "label": "媒体属性：video，240.000秒", "derived_from": ["E1"],
                          "acquired_by": "INSPECT/media_profile"}, frame("E3", 0), frame("E4", 60)]}}}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "raw_model_output", "payload": {"image_evidence_ids": ["E3", "E4"]}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "model_output", "payload": {"output": {
            "action": "INSPECT", "arguments": {"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [0, 60]}}}}}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "observation", "payload": {"observation": {"action": "INSPECT", "details": {
            "operation": "get_frames", "status": "ready", "acquisition": [{"candidate_id": "R1", "reused": False}]}}}},
        {"actor_role": "HI", "actor_id": "W1-1", "event_type": "raw_model_output", "payload": {"image_evidence_ids": []}},
        {"actor_role": "GIA", "actor_id": "GIA", "event_type": "raw_model_output", "payload": {"image_evidence_ids": ["E4", "E9"]}},
    ]
    (run / "trajectory.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    return run


def test_behavior_panel_reads_coverage_repeats_downloads_and_crashes(tmp_path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _evid_behavior import answer_language, run_behavior, summarize_behavior

    logs = tmp_path / "logs"
    logs.mkdir()
    run = _write_run(tmp_path)
    (logs / f"{run.name}.log").write_text("Traceback\nfoo\nforen_searcher.gia.runner.GiaTurnLimit: GIA exceeded its configured turn limit\n",
                                          encoding="utf-8")
    row = run_behavior(run, logs)
    assert row["duration_s"] == 240.0 and row["coverage"] == 0.25 and row["coverage_extracted"] == 0.25
    assert row["hi_actions"] == 2 and row["hi_repeats"] == 1 and row["repeat_rate"] == 0.5
    assert row["video_downloads"] == 1
    assert row["gia_last_images"] == 2 and row["gia_last_original_frames"] == 1
    assert row["context_retry_requests"] == 1 and row["input_tokens_max"] == 30000
    assert row["hi_rejected"] == 1                                   # 3 次原始输出，2 次通过
    assert row["crash_reason"].startswith("GiaTurnLimit: GIA exceeded")
    summary = summarize_behavior([row])
    assert summary["crashes"] == {"GiaTurnLimit": 1} and summary["runs_with_download"] == 1
    assert answer_language("1. 原始画面显示的是一处工业设施。2. 材料 A 加了字幕。") == "zh"
    assert answer_language("1. The original shows a factory [E3]. A caption 突发 was added.") == "en"
    assert answer_language("") == "none"


def test_eval_guards_add_schema_decoding_only():
    assert EVAL_GUARDS.json_schema and EVAL_GUARDS.json_mode
    assert EVAL_GUARDS.ingest_keyframes == 0 and not EVAL_GUARDS.evidence_origin and not EVAL_GUARDS.cross_round_handoff
    assert RuntimeGuards().json_schema is False


def test_gia_turn_exhaustion_without_draft_falls_back_to_default_investigation():
    """evidq27a（27B）：GIA 6 个回合全在读记忆 → GiaTurnLimit 崩溃。eval 档：按首轮默认派工收尾并打 forced_investigate_task。"""
    from foren_searcher.gia.runner import GiaRunner, GiaTurnLimit

    store, video, _ = _video_store(times=(0.0,))

    class Reader:
        async def decide(self, *, model_view, history, actions):
            return {"action": "QUERY_MEMORY", "arguments": {"query": "x"}}

    view = GlobalViewBuilder(store, MemoryPolicy()).build(task="t", original_material_ids=("C0", video), latest_round_index=None)
    runner = GiaRunner(store=store, policy=Reader(), memory_policy=MemoryPolicy(), max_turns=2, turn_timeout_s=None)
    with pytest.raises(GiaTurnLimit):
        asyncio.run(runner.run(view, round_index=1, can_create_round=True))
    with runtime_guard_scope(EVAL_GUARDS) as state:
        decision = asyncio.run(runner.run(view, round_index=1, can_create_round=True))
    assert decision.decision == "INVESTIGATE" and decision.tasks[0].task == "t" and decision.tasks[0].materials == (video,)
    assert state.contract_flags == ["forced_investigate_task"]


def test_export_holds_runs_rewritten_by_runtime_guards(tmp_path: Path):
    """P3：forced_* / finish_without_citation / auto_report 的 run 不当教师样本。"""
    from types import SimpleNamespace

    from foren_searcher.datagen.export import runtime_guard_markers

    (tmp_path / "result.json").write_text(json.dumps({"contract_flags": ["finish_without_citation", "finish_points_deduplicated",
                                                                         "forced_finish_from_draft"]}), encoding="utf-8")
    log = SimpleNamespace(run_dir=tmp_path, events=[{"event_type": "auto_report"}])
    assert runtime_guard_markers(log) == ["finish_without_citation", "forced_finish_from_draft", "auto_report"]
    clean = tmp_path / "clean"
    clean.mkdir()
    assert runtime_guard_markers(SimpleNamespace(run_dir=clean, events=[])) == []


def test_schema_is_bounded_and_drops_branches_the_runtime_would_reject():
    """evidb6 复盘：schema 解码下 times_s 从 0 数到 36570、GIA 无限追加任务（吃满 8192 token）；
    超过每轮任务上限 / 预算耗尽仍 INVESTIGATE 是 GIA 被拒的两大来源。"""
    from foren_searcher.datagen.teacher import SCHEMA_ARRAY_CAP, bounded_schema
    from foren_searcher.local_models import _decision_output_schema

    hi = _hi_payload("compact_16k")
    hi["available_actions"].append({"name": "INSPECT", "kind": "tool", "schema": {"oneOf": [{"type": "object", "properties": {
        "targets": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "request": {"type": "object", "properties": {"operation": {"const": "get_frames", "description": "d"},
                    "params": {"type": "object", "properties": {"times_s": {"type": "array", "items": {"type": "number"}}},
                               "not": {"required": ["x"]}}}}}}]}})
    bounded = json.dumps(bounded_schema(_decision_output_schema(hi, full_tool_schema=True), hi, {}))
    assert '"not"' not in bounded and "uniqueItems" not in bounded and f'"maxItems": {SCHEMA_ARRAY_CAP}' in bounded
    gia = {"available_actions": [], "global_view": {"runtime_constraints": {"can_create_round": True}}}
    tasks = [b["properties"]["tasks"] for b in bounded_schema(_decision_output_schema(gia), gia, {"max_tasks_per_round": 3})["anyOf"]
             if "tasks" in b["properties"]]
    assert tasks[0]["maxItems"] == 3
    done = {"available_actions": [], "global_view": {"runtime_constraints": {"can_create_round": False}}}
    branches = bounded_schema(_decision_output_schema(done), done, {})["anyOf"]
    assert [b["properties"]["decision"] for b in branches] == [{"type": "string", "enum": ["FINISH"]}]
    review = {"available_actions": [], "global_view": {"runtime_constraints": {"can_create_round": False, "round_in_progress": True}}}
    assert len(bounded_schema(_decision_output_schema(review), review, {})["anyOf"]) == 2      # WAIT / FINISH 都留着
