"""状态渲染器：档位预算、编号保留、纯函数、与运行时/消息构造的接线。"""

import copy
import json

import pytest

from foren_searcher import state_render as sr
from foren_searcher.local_models import _generation_messages
from foren_searcher.store.records import MemoryPolicy

SEARCH_SCHEMA = {"oneOf": [
    {"type": "object", "required": ["input_type", "query", "output_type"], "additionalProperties": False,
     "properties": {"input_type": {"const": "text"}, "query": {"type": "string"},
                    "output_type": {"enum": ["text", "image"]}}},
    {"type": "object", "required": ["source", "output_type"], "additionalProperties": False,
     "properties": {"source": {"type": "string"}, "output_type": {"enum": ["text", "image"]},
                    "include": {"type": "array", "items": {"enum": ["subtitles"]}}}},
]}


def search_observation(n: int, snippet: str = "很长的摘要 " * 40) -> dict:
    return {"action": "SEARCH", "items": [f"[R{i}] https://example.org/{i}; page=ready(E{i+10})" for i in range(n)],
            "details": {"output_type": "text", "stop_reason": "target_reached", "results": [
                {"candidate_id": f"R{i}", "source_url": f"https://example.org/{i}", "search_title": f"标题 {i}",
                 "search_snippet": snippet,
                 "materials": [{"kind": "page", "status": "ready", "error": None,
                                "evidence": [{"record_type": "evidence", "evidence_id": f"E{i+10}", "modalities": ["text"],
                                              "label": f"标题 {i}", "source_ref": f"R{i}", "derived_from": [],
                                              "acquired_by": "SEARCH/open/http+trafilatura"}]},
                               {"kind": "document", "status": "failed", "evidence": [],
                                "error": {"stage": "open", "code": "ACCESS_RESTRICTED", "message": "HTTP 403"}}]}
                for i in range(n)]}}


def hi_payload(history_steps: int = 6, renderer: str | None = "compact_8k") -> dict:
    history = []
    for step in range(history_steps):
        history.append({"action": "SEARCH", "request": {"input_type": "text", "query": f"查询 {step}", "output_type": "text"},
                        "focus_evidence_ids": ["E1"], "observation": search_observation(8)})
    history.append({"action": "READ_RECORD", "request": {"record_id": "E11"}, "focus_evidence_ids": ["E1", "E11"],
                    "observation": None})
    view = {
        "task_id": "W1-1", "task": "核查声明 X", "materials": {
            "C0": {"record_type": "material", "reference_id": "C0", "content": "{\"task\": \"核查声明 X\"}", "truncated": False},
            "E1": {"record_type": "evidence", "evidence_id": "E1", "modalities": ["video"], "label": "材料 A",
                   "source_ref": "I0", "derived_from": [], "acquired_by": "HARNESS/ingest"}},
        "candidate_sources": [{"record_type": "candidate", "candidate_id": f"R{i}", "source_url": f"https://example.org/{i}",
                               "title": f"标题 {i}", "snippet": "摘要 " * 60, "candidate_type": "page"} for i in range(8)],
        "evidence_view": [{"record_type": "evidence", "evidence_id": f"E{i}", "modalities": ["image"],
                           "label": f"视频E1在{i * 1.23456789:.3f}秒的帧", "source_ref": None, "derived_from": ["E1"],
                           "acquired_by": "INSPECT/get_frames"} for i in range(2, 10)],
        "focus_evidence_ids": ["E1", "E11"], "content_observed_evidence_ids": ["E11"],
        "history": history,
        "latest_observation": {"action": "READ_RECORD", "items": ["READ_RECORD E11; offset=0; next_offset=6000"],
                               "details": {"record_type": "evidence", "evidence_id": "E11", "modalities": ["text"],
                                           "label": "标题 1", "content": "正文" * 3000, "content_offset": 0, "next_offset": 6000}},
        "runtime_constraints": {"turn_index": 8, "max_turns": 12, "remaining_turns": 4, "max_model_view_chars": 180000,
                                **({"state_renderer": renderer} if renderer else {})},
        "direct_visual_input": True, "image_input_limit": 8,
        "image_inputs": [{"evidence_id": "E2", "locator": json.dumps({"time_s": 1.2345678901}), "source_ref": None,
                          "derived_from": ["E1"], "width": 1280, "height": 720}],
        "image_inputs_omitted": [], "image_input_errors": [],
        "visual_observation_summary": {"image_evidence_count": 1, "videos": [{"evidence_id": "E1", "observed_frame_count": 1,
                                       "media_duration_s": 25.1, "observed_ranges_s": [[1.2345678901, 1.2679012345]],
                                       "unconfirmed_ranges_s": [[0.0, 1.2345678901], [1.2679012345, 25.1]]}]},
    }
    actions = [{"name": "SEARCH", "kind": "tool", "description": "Search external sources.", "schema": SEARCH_SCHEMA},
               {"name": "READ_RECORD", "kind": "harness", "description": "Read a saved record by ID.",
                "schema": {"type": "object", "required": ["record_id"], "properties": {"record_id": {"type": "string"}, "offset": {"type": "integer"}}}}]
    return {"available_actions": actions, "workspace_id": "W1-1", "model_view": view}


def ids(value) -> set[str]:
    import re
    return set(re.findall(r"\b([ERCF]\d+)\b", json.dumps(value, ensure_ascii=False)))


# ------------------------------------------------------------------ 基本性质

def test_legacy_is_identity_copy():
    payload = hi_payload(renderer="legacy")
    out = sr.render_payload(payload, "legacy")
    assert out == payload and out is not payload


def test_unknown_tier_rejected():
    with pytest.raises(sr.RenderError, match="state_renderer"):
        sr.render_payload(hi_payload(), "compact_4k")


def test_render_is_pure_and_deterministic():
    payload = hi_payload()
    frozen = copy.deepcopy(payload)
    first = sr.render_payload(payload, "compact_8k")
    second = sr.render_payload(payload, "compact_8k")
    assert payload == frozen and first == second


@pytest.mark.parametrize("tier", ["compact_8k", "compact_16k"])
def test_hi_fits_budget_and_keeps_visible_ids(tier):
    payload = hi_payload(history_steps=10)
    assert sr.payload_chars(payload) > 60_000
    out = sr.render_payload(payload, tier)
    assert sr.payload_chars(out) <= sr.TIERS[tier].budget_chars
    view = out["model_view"]
    original = payload["model_view"]
    must = ids({k: original[k] for k in ("materials", "evidence_view", "focus_evidence_ids", "content_observed_evidence_ids", "image_inputs")})
    assert must <= ids(out)
    assert view["runtime_constraints"]["state_renderer"] == tier
    assert view["task"] == original["task"] and view["task_id"] == "W1-1"
    # 最新观察保留正文（截断），历史只留摘要行，不再带完整 results
    assert "content" in view["latest_observation"]["details"]
    assert all("results" not in (h["observation"] or {}).get("details", {}) for h in view["history"] if h["observation"])


def test_history_dropped_marks_context_truncated_and_keeps_last_step():
    payload = hi_payload(history_steps=14)
    out = sr.render_payload(payload, "compact_8k")
    view = out["model_view"]
    assert view["context_truncated"] is True
    assert view["history"][-1]["action"] == "READ_RECORD" and view["history"][-1]["observation"] is None
    assert len(view["history"]) < len(payload["model_view"]["history"])
    roomy = sr.render_payload(hi_payload(history_steps=1), "compact_16k")
    assert "context_truncated" not in roomy["model_view"]


def test_actions_become_signatures_without_schema():
    out = sr.compact_actions(hi_payload()["available_actions"])
    search = out[0]
    assert "schema" not in search
    assert search["signature"] == ("{input_type:text, query:string, output_type:text|image} | "
                                   "{source:string, output_type:text|image, include?:[subtitles]}")
    assert out[1]["signature"] == "{record_id:string, offset?:integer}"


def test_cards_become_lines_and_floats_round():
    out = sr.render_payload(hi_payload(), "compact_8k")
    view = out["model_view"]
    assert view["evidence_view"][0] == "E2 [image] 视频E1在2.469秒的帧 ← E1"
    assert view["materials"]["C0"]["content"].startswith("{")           # 任务正文原样
    assert view["materials"]["E1"] == "E1 [video] 材料 A ← I0"
    assert '"time_s":1.235' in view["image_inputs"][0]["locator"]
    assert view["visual_observation_summary"]["videos"][0]["observed_ranges_s"] == [[1.235, 1.268]]
    rich = sr.render_payload(hi_payload(), "compact_16k")["model_view"]
    assert rich["evidence_view"][0].endswith("(INSPECT/get_frames)")   # 16K 档保留获取方式


def test_latest_search_results_keep_evidence_and_failures():
    payload = hi_payload(history_steps=0)
    payload["model_view"]["latest_observation"] = search_observation(3)
    out = sr.render_payload(payload, "compact_16k")["model_view"]["latest_observation"]
    result = out["details"]["results"][0]
    assert result["evidence"] == ["E10:page"] and result["failed"] == ["document:ACCESS_RESTRICTED"]
    assert "materials" not in result and len(result["search_snippet"]) < 300


def test_over_budget_base_state_raises():
    payload = hi_payload(history_steps=0)
    payload["model_view"]["task"] = "任务" * 8000
    with pytest.raises(sr.RenderError, match="超过 compact_8k 的预算"):
        sr.render_payload(payload, "compact_8k")


# ------------------------------------------------------------------ GIA

def gia_payload() -> dict:
    report = {"task": "子任务", "materials": ["C0", "E1"], "run_status": "reported",
              "report": {"record_type": "report", "task_id": "W1-1", "status": "COMPLETED", "finding_ids": ["F1"],
                         "selected_evidence_ids": ["E21"], "findings": [{"record_type": "finding", "finding_id": "F1",
                         "workspace_id": "W1-1", "statement": "结论", "evidence_ids": ["E21"], "limitations": ["限制"]}]},
              "supporting_records": [{"record_type": "evidence", "evidence_id": "E21", "modalities": ["text"], "label": "页面",
                                      "content": "正文" * 2000, "source_ref": "R3", "derived_from": [], "artifact_ref": "A2",
                                      "locator": None, "acquired_by": "SEARCH/open", "content_offset": 0, "next_offset": 6000,
                                      "content_notice": None},
                                     {"record_type": "evidence", "evidence_id": "E22", "modalities": ["image"], "label": "帧",
                                      "content": "{\"time_s\": 1.0}", "derived_from": ["E1"], "artifact_ref": "A3",
                                      "acquired_by": "INSPECT/get_frames", "content_offset": 0, "next_offset": None,
                                      "content_notice": "本记录正文是图片元数据，不是像素。…"}],
              "image_inputs": [{"evidence_id": "E22", "locator": "t=1", "source_ref": None, "derived_from": ["E1"], "width": 1, "height": 1}],
              "visual_observation_summary": {"image_evidence_count": 1, "videos": []}}
    view = {"task": "总任务", "original_materials": [{"record_type": "material", "reference_id": "C0", "content": "总任务"}],
            "current_materials": [{"record_type": "evidence", "evidence_id": f"E{i}", "modalities": ["image"], "label": f"帧 {i}",
                                   "source_ref": None, "derived_from": ["E1"], "acquired_by": "INSPECT/get_frames"} for i in range(2, 40)],
            "latest_reports": [report, copy.deepcopy(report)],
            "history_index": [{"record_type": "evidence", "evidence_id": f"E{i}", "label": f"旧帧 {i}", "source_workspace": "W0-1"} for i in range(100, 140)],
            "runtime_constraints": {"round_index": 2, "can_create_round": True, "turn_index": 1, "max_turns": 8,
                                    "remaining_turns": 7, "state_renderer": "compact_8k"},
            "direct_visual_input": True, "image_input_limit": 8, "image_inputs": [], "image_inputs_omitted": [], "image_input_errors": []}
    history = [{"turn_index": 1, "output": {"action": "READ_RECORD", "arguments": {"record_id": "E21", "offset": 0}},
                "observation": {"action": "READ_RECORD", "items": ["READ_RECORD E21; offset=0; next_offset=6000"],
                                "details": {"evidence_id": "E21", "content": "正文" * 2000, "content_offset": 0, "next_offset": 6000}}}]
    actions = [{"name": "READ_RECORD", "kind": "harness", "description": "Read", "schema": {"type": "object", "required": ["record_id"],
                "properties": {"record_id": {"type": "string"}}}}]
    return {"available_actions": actions, "memory_history": history, "global_view": view}


@pytest.mark.parametrize("tier", ["compact_8k", "compact_16k"])
def test_gia_fits_budget_and_keeps_findings(tier):
    payload = gia_payload()
    out = sr.render_payload(payload, tier)
    assert sr.payload_chars(out) <= sr.TIERS[tier].budget_chars
    view = out["global_view"]
    report = view["latest_reports"][0]
    assert report["report"]["findings"][0]["statement"] == "结论" and "F1" in ids(report)
    assert {"E21", "E22"} <= ids(report["supporting_records"])
    assert view["runtime_constraints"]["state_renderer"] == tier
    assert all(isinstance(line, str) and line.startswith("E") for line in view["current_materials"])


def test_gia_tiers_differ_in_support_text():
    small = sr.render_payload(gia_payload(), "compact_8k")["global_view"]["latest_reports"][0]["supporting_records"]
    big = sr.render_payload(gia_payload(), "compact_16k")["global_view"]["latest_reports"][0]["supporting_records"]
    assert isinstance(small[0], str) and "正文" not in small[0]          # 8K：只留编号和标签，正文靠 READ_RECORD
    assert big[0]["content"].startswith("正文") and "已截断" in big[0]["content"] and big[0]["next_offset"] == 6000
    assert big[1]["content"] == "media_metadata_not_pixels"


# ------------------------------------------------------------------ 接线

def test_memory_policy_validates_renderer():
    assert MemoryPolicy().state_renderer == "legacy"
    assert MemoryPolicy(state_renderer="compact_16k").state_renderer == "compact_16k"
    with pytest.raises(ValueError, match="state_renderer"):
        MemoryPolicy(state_renderer="tiny")


def test_generation_messages_apply_renderer_from_runtime_constraints():
    compact = _generation_messages("系统提示", hi_payload(history_steps=6, renderer="compact_8k"))
    legacy = _generation_messages("系统提示", hi_payload(history_steps=6, renderer="legacy"))
    absent = _generation_messages("系统提示", hi_payload(history_steps=6, renderer=None))
    total = lambda msgs: sum(len(m["content"]) for m in msgs)
    assert total(compact) < total(legacy) / 3
    assert abs(total(absent) - total(legacy)) == len(',"state_renderer":"legacy"')   # 两者都不压缩
    assert compact[0]["role"] == "system" and compact[1]["role"] == "user"
    assert '"signature"' in compact[1]["content"] and '"schema"' not in compact[1]["content"]
    tool_turns = [m for m in compact if m["role"] == "tool"]
    assert "E11" in tool_turns[-1]["content"] and len(tool_turns[-1]["content"]) < 4000   # 最新观察正文按 8K 档截断
