"""10-15 export 质量门：B2 证据放行、B3 引用可见、B4 截断依赖、B6 INCOMPLETE 比对、C5 兼容、复核行剔除、分支名、
token 估计；scripts/export_gate.py 与 scripts/walker_coverage.py --export 在小导出目录上的判定。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from foren_searcher.datagen.export import (
    RunEvidence, WorkspaceFacts, check_incomplete_reason, cited_ids_outside_view, compat_problems, decision_branch, hidden_context,
    is_review_row, release_by_evidence, row_tokens_est, truncated_dependency,
)
from foren_searcher.datagen.export import _token_in as X_token_in
from foren_searcher.datagen.runlog import Decision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_gate  # noqa: E402
import walker_coverage  # noqa: E402


def _decision(role="HI", kind="step", output=None, view=None, raw_event_id=50, actor="W1-1", payload=None) -> Decision:
    key = "model_view" if role == "HI" else "global_view"
    payload = payload if payload is not None else ({"workspace_id": actor, key: view or {}} if role == "HI" else {key: view or {}})
    return Decision(request_id="request_0005", role=role, actor_id=actor if role == "HI" else "GIA", turn=3, round_index=None, kind=kind,
                    output=output or {}, status="executed", system_prompt="sys", payload=payload, image_evidence_ids=[], image_notes=[],
                    image_paths=[], raw_event_id=raw_event_id)


def _messages(student: str, output: dict) -> list[dict]:
    return [{"role": "system", "content": "You are HI."}, {"role": "user", "content": student},
            {"role": "assistant", "content": json.dumps(output)}]


# --------------------------------------------------------------------------- B2

def test_release_word_from_visible_evidence_and_earlier_observation() -> None:
    corpus = RunEvidence(records={"E28": "OCR: A conversation about CRPS", "E40": "Nanjing station"},
                         observations=[(13, "W1-1", "bullet trains in Nanjing"), (60, "W1-1", "later Guatemalan"), (14, "W1-2", "Guatemalan")])
    out = {"status": "COMPLETED", "findings": [{"statement": "B reads CRPS", "evidence_ids": ["E28"]}]}
    msgs = _messages("evidence E28 [image] OCR result", out)
    d = _decision(kind="report", output=out)
    assert release_by_evidence(["CRPS"], d, msgs, corpus) == {"CRPS": "E28"}
    # E40 不在学生输入里：读不到它，只能靠本工作区更早的观察放行
    assert release_by_evidence(["Nanjing"], d, msgs, corpus) == {"Nanjing": "observation@13"}
    # 本步之后的观察、别的工作区的观察都不算
    assert release_by_evidence(["Guatemalan"], d, msgs, corpus) == {}


def test_numbers_are_matched_by_value_not_substring() -> None:
    corpus = RunEvidence(records={"E28": '{"compared_pixels":129600}', "E34": "frame of video E2 at 3.500 s"})
    out = {"status": "INCOMPLETE", "findings": [{"statement": "frames 99-129 at 3.5 s", "evidence_ids": ["E34", "E28"]}]}
    msgs = _messages("E28 E34", out)
    d = _decision(kind="report", output=out)
    assert release_by_evidence(["129"], d, msgs, corpus) == {}
    assert release_by_evidence(["3.5"], d, msgs, corpus) == {"3.5": "E34"}
    assert X_token_in("112.6", '"range_s": [112.61, 118.42]') and not X_token_in("112.6", "112.7 113")
    # 数字只认本步输出自己引用的 E
    d2 = _decision(kind="report", output={"status": "INCOMPLETE", "findings": [{"statement": "3.5", "evidence_ids": ["E28"]}]})
    assert release_by_evidence(["3.5"], d2, _messages("E28 E34", d2.output), corpus) == {}


def test_hidden_context_ignores_numbers_that_only_appear_inside_longer_numbers() -> None:
    payload = {"model_view": {"x": '{"compared_pixels": 129600, "label": "Kisumu"}'}}
    out = {"rationale": "frame 129 shows Kisumu"}
    msgs = _messages("nothing here", out)
    assert hidden_context(msgs, payload, out) == ["Kisumu"]


# --------------------------------------------------------------------------- B3 / B4

def test_cited_ids_outside_view() -> None:
    out = {"action": "READ_RECORD", "arguments": {"record_id": "E12"}, "focus_evidence_ids": ["E3"], "rationale": "R4 page"}
    assert cited_ids_outside_view(_messages("E3 R4 visible", out), out) == ["E12"]
    assert cited_ids_outside_view(_messages("E3 E12 R4", out), out) == []


def test_truncated_dependency_kinds() -> None:
    view = {"latest_observation": {"items": ["E44 frame at 112.6 s shows Lebanon flag"]}}
    out = {"action": "INSPECT", "arguments": {"targets": ["E44"]}, "rationale": "Look at 112.6 s"}
    d = _decision(view=view, output=out)
    truncated = truncated_dependency(d, _messages("E44 …[truncated 900 chars; READ_RECORD reads the full text]", out), out)
    assert truncated == {"kind": "truncated", "items": ["112.6"]}
    omitted = truncated_dependency(d, _messages("E44 frame", out), out)
    assert omitted and omitted["kind"] == "omitted"
    assert truncated_dependency(d, _messages("E44 frame at 112.6 s", out), out) is None


# --------------------------------------------------------------------------- B6

def test_incomplete_reason_checks() -> None:
    facts = WorkspaceFacts(searches=2, compare_runs=1, compare_best_pct=62, compare_spans=True, audio_tracks={"E1": 0, "E2": 1},
                           turns_used=5, max_turns=12)
    def claims(reason: str) -> dict[str, str]:
        return {c["claim"]: c["verdict"] for c in check_incomplete_reason({"status_reason": reason}, facts)}
    assert claims("Material A has no audio track") == {"no_audio": "ok"}
    assert claims("Original source video not found after three searches")["searches"] == "mismatch"      # 只搜了 2 次
    assert claims("Original source video not found after three searches")["original_not_found"] == "mismatch"   # 62% 对上
    assert claims("correspondence A 9.9 s: start and end boundaries not established") == {"boundaries_missing": "mismatch"}
    assert claims("the page could not be downloaded (403)") == {"tool_failed": "mismatch"}
    assert claims("ran out of turns") == {"budget": "mismatch"}
    assert claims("not every part of the task is answered by the evidence: part 3") == {}


# --------------------------------------------------------------------------- C5 / 复核 / 分支 / token

def _spec(language="en", merged=True):
    from types import SimpleNamespace

    return SimpleNamespace(harness_params={"language": language, "inspect_merged_catalog": merged})


def test_compat_problems_language_and_catalog() -> None:
    zh = {"action": "INSPECT", "arguments": {"targets": ["E1"], "request": {"operation": "align_media"}}, "rationale": "对齐两段视频，核对剪辑位置。"}
    msgs = _messages("English state with a quoted 你爱我 caption", zh)
    problems = compat_problems(_decision(output=zh), _spec(), msgs)
    assert "language_mismatch:output_zh" in problems and "op_not_in_current_catalog:align_media" in problems
    en = {"decision": "FINISH", "answer": "Material B shows '你爱我' where A shows '你爱我呀我爱你呀' [E26]; the rest matches."}
    assert compat_problems(_decision(role="GIA", kind="answer", output=en), _spec(), _messages("state", en)) == []
    assert compat_problems(_decision(output=zh), _spec(language="zh", merged=False), msgs) == []


def test_review_rows() -> None:
    wait = _decision(role="GIA", kind="memory", output={"decision": "WAIT"})
    assert is_review_row(wait, None)
    running = _decision(role="GIA", kind="answer", output={"decision": "FINISH"}, view={"runtime_constraints": {"round_in_progress": True}})
    assert is_review_row(running, "findings")
    assert not is_review_row(_decision(role="GIA", kind="answer", output={"decision": "FINISH"}), "findings")
    assert is_review_row(_decision(role="GIA", kind="answer", output={"decision": "FINISH"}), "wait")


def test_decision_branch_names_match_coverage_table() -> None:
    hi = {"workspace_id": "W1-1", "model_view": {}}
    assert decision_branch({"action": "SEARCH", "arguments": {"source": "R1", "include": ["dates"]}}, hi) == "SEARCH open dates"
    assert decision_branch({"action": "SEARCH", "arguments": {"input_type": "image", "output_type": "text"}}, hi) == "SEARCH image→text"
    assert decision_branch({"action": "INSPECT", "arguments": {"request": {"operation": "compare_media", "params": {}}}}, hi) == "INSPECT compare_media"
    assert decision_branch({"action": "READ_RECORD", "arguments": {"record_id": "E1"}, "focus_evidence_ids": ["E1"]}, hi) == "READ_RECORD focus"
    assert decision_branch({"status": "INCOMPLETE", "findings": []}, hi) == "report_incomplete"
    gia = {"global_view": {"runtime_constraints": {"round_index": 2}}}
    assert decision_branch({"decision": "INVESTIGATE", "tasks": []}, gia) == "GIA INVESTIGATE round2"
    names = {m[0] for m in walker_coverage.MATRIX}
    assert "SEARCH open dates" in names and "READ_RECORD focus" in names


def test_row_tokens_est_language_aware() -> None:
    en = [{"role": "user", "content": "a" * 3000}, {"role": "assistant", "content": "{}"}]
    zh = [{"role": "user", "content": "中" * 1000}, {"role": "assistant", "content": "{}"}]
    assert 990 <= row_tokens_est(en, 0) <= 1010
    assert row_tokens_est(zh, 1) == 1000 + 880


# --------------------------------------------------------------------------- 门本身

def _row(bucket="accepted", unit="step", role="hi", case="videosham:4126", output=None, student="E1 E2", **meta) -> dict:
    output = output if output is not None else {"action": "INSPECT", "arguments": {"targets": ["E1"]}, "rationale": f"Check E1 for case {case}."}
    m = {"role": role, "unit": unit, "origin": "A", "run": f"{case.replace(':', '_')}_wk99_1", "workspace": "W1-1", "turn": 1,
         "request": "request_0002", "case_id": case, "dataset": case.split(":")[0], "case_family": case, "split": "dev_train",
         "teacher_model": "t", "spec": "hi@0.10.0", "spec_hash": "x", "state_renderer": "compact_16k", "verifier": "verifier@0.3.6",
         "judge_model": "j", "verdict": "positive", "error_types": [], "sample_weight": 1.0, "input_tokens_est": 10,
         "supervised_tokens_est": 5, "loss_scope": "last_assistant_only", "exported_at": "2026-10-15", "bucket": bucket,
         "bucket_reason": None, "action": output.get("action") or output.get("decision") or output.get("status"),
         "hidden_context": [], "hidden_context_released": {}, "tool_result_truncation": None, "incomplete_reason_checks": [],
         "rationale_source": "s0_teacher", "branch": "INSPECT compare_media", **meta}
    return {"messages": [{"role": "system", "content": "You are HI."}, {"role": "user", "content": student},
                         {"role": "assistant", "content": json.dumps(output)}], "images": [], "metadata": m}


def _write(tmp: Path, rows: list[dict]) -> Path:
    files: dict[str, list[dict]] = {}
    for r in rows:
        files.setdefault(f"{r['metadata']['role']}_{r['metadata']['bucket']}", []).append(r)
    for name in export_gate.FILES:
        (tmp / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in files.get(name, [])), encoding="utf-8")
    (tmp / "manifest.json").write_text(json.dumps({"export_options": {"plan_copy": "hold"}, "excluded_rows": []}), encoding="utf-8")
    return tmp


def _gate(tmp: Path) -> dict[str, dict]:
    args = export_gate.main.__globals__["argparse"].Namespace(
        export=str(tmp), cases=str(tmp / "no_cases"), evid_bench=None, evid_cases=None, evid_range=[131, 222], evid_channels=None,
        phash=False, evid_videos=None, phash_max_distance=6, model=None, unanswerable=[], strict=False)
    report = export_gate.run_gate(args)
    return {i["id"]: i for i in report["items"]}


def test_gate_flags_citation_outside_view_and_bad_json(tmp_path: Path) -> None:
    good = _row()
    outside = _row(output={"action": "READ_RECORD", "arguments": {"record_id": "E9"}, "rationale": "read it"}, student="E1")
    broken = _row()
    broken["messages"][-1]["content"] = "{not json"
    items = _gate(_write(tmp_path, [good, outside, broken]))
    assert items["B3"]["status"] == "不通过" and items["B3"]["value"]["violations"] == 1
    assert items["B8"]["status"] == "不通过" and items["B8"]["value"]["invalid_json"] == 1
    assert items["C1"]["status"] == "不通过"
    assert (tmp_path / "gate_report.md").is_file() and (tmp_path / "gate_b2_review.md").is_file()


def test_gate_framework_language_and_template_sentences(tmp_path: Path) -> None:
    rows = [_row(output={"action": "SEARCH", "arguments": {"input_type": "text", "output_type": "text", "query": f"q{i}"},
                         "rationale": "Search for reports of the event the claim describes."}) for i in range(6)]
    zh = _row()
    zh["messages"].insert(2, {"role": "user", "content": "当前局部任务依据已执行动作实际附图和文本证据选择下一步工具行动或提交报告逐项保留任务要求"})
    items = _gate(_write(tmp_path, rows + [zh]))
    assert items["B7"]["value"]["max_repeat"] == 6 and items["B7"]["status"] == "不通过"
    assert items["B8"]["value"]["framework_text_zh_rows"] == 1


def test_gate_incomplete_rate_and_mismatch(tmp_path: Path) -> None:
    rep = {"task_id": "W1-1", "status": "INCOMPLETE", "findings": [{"statement": "s", "evidence_ids": ["E1"]}], "status_reason": "x"}
    rows = [_row(unit="report", output=rep, incomplete_reason_checks=[{"claim": "tool_failed", "verdict": "mismatch", "why": ""}]),
            _row(unit="report", output={**rep, "status": "COMPLETED"})]
    b6 = _gate(_write(tmp_path, rows))["B6"]
    assert b6["value"]["accepted_reason_mismatch"] == 1 and b6["value"]["incomplete_rate_accepted"] == 0.5 and b6["status"] == "不通过"


def test_walker_coverage_family_and_chains(tmp_path: Path) -> None:
    assert walker_coverage.family_of("fmnv:a_x") == "A" and walker_coverage.family_of("fmnv:c_x") == "C"
    assert walker_coverage.family_of("averimatec:e12") == "E" and walker_coverage.family_of("averimatec:l208") == "L"
    assert walker_coverage.family_of("groundlie360:v1") == "C" and walker_coverage.family_of("true:1") == "other"
    rows = [_row(turn=1, branch="INSPECT compare_media"), _row(turn=2, branch="READ_RECORD focus"),
            _row(turn=3, branch="INSPECT get_frames times_s")]
    rows += [_row(case="averimatec:e2", turn=1, branch="SEARCH open dates"), _row(case="averimatec:e2", turn=2, branch="READ_RECORD text")]
    _write(tmp_path, rows)
    result = walker_coverage.export_branch_table(tmp_path)
    chains = {c["chain"]: c["rows"] for c in result["chains"]}
    assert chains["B compare→读差分→取帧"] == 3 and chains["E 日期查询+读最早页"] == 2


# --------------------------------------------------------------------------- 本机真实 run（有数据才跑）

RUN_1028 = ROOT / "experiments" / "sft_data" / "runs" / "videosham_1028_wk8_1"


@pytest.mark.skipif(not (RUN_1028 / "store.sqlite3").is_file(), reason="需要本机 wk8 run 目录")
def test_crps_is_released_by_ocr_evidence() -> None:
    from foren_searcher.datagen.export import build_run_evidence
    from foren_searcher.datagen.runlog import load_run

    log = load_run(RUN_1028)
    try:
        corpus = build_run_evidence(log)
        assert any("CRPS" in text for text in corpus.records.values())
    finally:
        log.close()


def test_c6_phash_and_id_dedup(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import argparse

    import numpy as np

    def video(path: Path, seed: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        rng = np.random.default_rng(seed)
        for _ in range(30):
            writer.write((rng.random((48, 64, 3)) * 255).astype("uint8"))
        writer.release()

    for d in ("evid", "cases", "media"):
        (tmp_path / d).mkdir()
    video(tmp_path / "evid" / "video_131.mp4", 1)
    video(tmp_path / "media" / "same.mp4", 1)
    video(tmp_path / "media" / "other.mp4", 2)
    base = json.loads((ROOT / "experiments" / "sft_data" / "cases" / "videosham_0069.json").read_text(encoding="utf-8")) \
        if (ROOT / "experiments" / "sft_data" / "cases" / "videosham_0069.json").is_file() else None
    if base is None:
        pytest.skip("需要案卷模板")
    for name, media in (("9", "same"), ("8", "other")):
        case = {**base, "case_id": f"videosham:{name}", "source_id": name, "family_key": f"videosham:{name}",
                "media": [{"ref": "I0", "path": f"../media/{media}.mp4", "media_type": "video/mp4"}],
                "notes": "源视频 dApuKGrRVG8" if name == "8" else ""}
        (tmp_path / "cases" / f"videosham_{name}.json").write_text(json.dumps(case), encoding="utf-8")
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps([{"id": "video_131", "source": ["dApuKGrRVG8"]}]), encoding="utf-8")
    rows = [{"messages": [], "images": [], "metadata": {"bucket": "accepted", "case_id": f"videosham:{n}", "case_family": f"videosham:{n}",
                                                         "split": "dev_train"}} for n in ("9", "8")]
    args = argparse.Namespace(cases=str(tmp_path / "cases"), evid_bench=str(bench), evid_cases=None, evid_range=[131, 222],
                              evid_channels=None, phash=True, evid_videos=str(tmp_path / "evid"), phash_max_distance=6)
    result = export_gate.check_c6(rows, args)
    assert result["status"] == "不通过"
    assert [h["case"] for h in result["value"]["phash_hits"]] == ["videosham:9"]
    assert [h["case"] for h in result["value"]["id_hits"]] == ["videosham:8"]


def test_gate_b10_counts_rereads_of_compare_records(tmp_path: Path) -> None:
    card = "E21 [text] INSPECT compare_media result ← E1,E2 (INSPECT/compare_media)"
    rr = {"action": "READ_RECORD", "arguments": {"record_id": "E21"}, "rationale": "Read the comparison E21 again."}
    rows = [_row(output=rr, student=card, turn=t) for t in range(1, 4)]
    rows.append(_row(output={"action": "READ_RECORD", "arguments": {"record_id": "E7"}, "rationale": "Read page E7."}, student="E7 [text] page"))
    b10 = _gate(_write(tmp_path, rows))["B10"]
    assert b10["status"] == "不通过" and b10["value"]["max_reads"] == 3 and b10["value"]["records_read"] == 1
