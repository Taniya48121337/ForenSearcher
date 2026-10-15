"""10-14 修复 5：合并目录的工具记录 = 结论（与 DETECT 的结果记录同形），HI 不再读到测量表。

根因（evidM27h）：compare_media / video_structure 的结果 E 存整份测量表（约 10 KB、键按字母排、结论在第 8400 字后），
HI 反复 READ_RECORD 第一页，答案照着 1.5 s 的 reversed / sped_up 段（video_154）和静止帧（video_209）写。
"""

from __future__ import annotations

import json
import cv2
import numpy as np

from foren_searcher.inspect.contracts import CompareMediaResult, ReportImage, VideoStructureResult
from foren_searcher.inspect.service import CONCLUSION_RECORD_OPERATIONS, conclusion_record

_RAW = {"correspondences": [{"a_range_s": [78.2, 79.7], "b_range_s": [64.7, 65.5], "kind": "reversed"}],
        "segments_not_in_b": [{"a_range_s": [15.7, 25.7]}], "b_cuts": [], "source_match": True,
        "limitations": ["Half-second sampling."], "timings_s": {"total_s": 40.0}}


def test_compare_record_is_conclusions_first_and_short():
    lines = ("B matches A: 94% of A's half-second samples are found in B.",
             "Not in B: A 15.7-25.7 s (10.0 s) ... most likely an AI-generated clip made to fit this scene; say what it shows. "
             "Visual check of comparison image 1: a tablet on a desk.")
    result = CompareMediaResult("video", lines, dict(_RAW), (ReportImage("E28", "unmatched", 0, "A 15.7-25.7 s has no counterpart in B"),))
    text = conclusion_record("compare_media", result, {"backend": "compare_media"})
    record = json.loads(text)
    assert list(record)[:3] == ["operation", "summary_lines", "images"] and text.index("summary_lines") < 40
    assert "Visual check of E28" in record["summary_lines"][1]
    assert record["images"] == [{"evidence_id": "E28", "caption": "A 15.7-25.7 s has no counterpart in B"}]
    assert "correspondences" not in record and "timings_s" not in record and record["limitations"] == ["Half-second sampling."]
    assert len(text) < 2000 and CONCLUSION_RECORD_OPERATIONS == {"compare_media", "video_structure"}


def test_structure_record_keeps_range_but_not_the_shot_table():
    report = {"shots": [{"shot": k} for k in range(37)], "frozen_spans": [{"range_s": [2.0, 2.9]}],
              "coverage": {"range_s": [0.0, 468.2], "frames": 900, "fps": 2.0}, "generator_length_shots": [{"shot": 4}]}
    result = VideoStructureResult(("37 shot(s) between hard cuts in 0-468 s.",), report, ())
    record = json.loads(conclusion_record("video_structure", result, {}))
    assert record["coverage"] == {"range_s": [0.0, 468.2]}
    assert "shots" not in record and "frozen_spans" not in record and "generator_length_shots" not in record


def test_latest_observation_for_compare_keeps_only_conclusions():
    from foren_searcher.state_render import _compact_report

    observation = {"action": "INSPECT", "items": ["x"], "details": {
        "operation": "compare_media", "status": "ready", "result_evidence_id": "E31", "evidence": [], "call_timings_ms": {},
        "result": {"summary_lines": ["B matches A"], "images": [{"evidence_id": "E28"}], **_RAW}}}
    out = _compact_report(observation)
    assert set(out["details"]["result"]) == {"summary_lines", "images", "limitations"}
    assert out["details"]["result_evidence_id"] == "E31" and "evidence" not in out["details"]
    assert observation["details"]["result"]["correspondences"]          # the original observation is untouched


def test_corresponding_spans_of_three_seconds_or_more_stay_visible():
    """10-15（A9 回放）：白名单只留 summary_lines 时，正常速度的对应段不在摘要行里，VCSL 类任务（给出两侧对应区间）
    的学生视图没有答案要的区间，walker 回放 vcsl_pair_5/6 也因此改了取帧时刻。对应段以结论行保留：≥ 3 s、最长 6 段、按 A 排。"""

    from foren_searcher.state_render import _compact_report, corresponding_spans

    corr = [{"a_range_s": [28.2, 34.8], "b_range_s": [16.5, 23.7], "kind": "sped_up"},
            {"a_range_s": [0.0, 6.8], "b_range_s": [5.0, 11.6], "kind": "normal"},
            {"a_range_s": [78.2, 79.7], "b_range_s": [64.7, 65.5], "kind": "reversed"},
            {"a_range_s": [10.2, 13.8], "b_range_s": [15.6, 18.5], "kind": "normal", "picture_differs": True, "mirrored": True}]
    spans = corresponding_spans(corr)
    assert spans == ["A 0.0-6.8 s = B 5.0-11.6 s", "A 10.2-13.8 s = B 15.6-18.5 s (mirrored, picture differs)",
                     "A 28.2-34.8 s = B 16.5-23.7 s (sped_up)"]
    many = [{"a_range_s": [10.0 * k, 10.0 * k + 3 + k], "b_range_s": [0.0, 1.0]} for k in range(8)]
    assert [s.split(" s =")[0] for s in corresponding_spans(many)] == [f"A {10.0 * k}-{10.0 * k + 3 + k}" for k in range(2, 8)]

    observation = {"action": "INSPECT", "items": ["x"], "details": {
        "operation": "compare_media", "status": "ready", "result_evidence_id": "E31",
        "result": {"summary_lines": ["B matches A"], "images": [], **_RAW, "correspondences": corr}}}
    result = _compact_report(observation)["details"]["result"]
    assert list(result) == ["summary_lines", "corresponding_spans", "images", "limitations"] and result["corresponding_spans"] == spans

    record = json.loads(conclusion_record("compare_media", CompareMediaResult("video", ("B matches A",), {**_RAW, "correspondences": corr}, ()), {}))
    assert list(record)[:3] == ["operation", "summary_lines", "corresponding_spans"] and record["corresponding_spans"] == spans
    assert "correspondences" not in record


def test_content_match_line_lists_only_runs_of_three_seconds_or_more():
    from foren_searcher.search.source_match import content_match_line

    match = {"material": "E1", "status": "source", "fraction": 0.94, "runs": [
        {"a_range_s": [0.0, 15.7], "b_range_s": [0.0, 15.5], "kind": "normal"},
        {"a_range_s": [79.7, 81.7], "b_range_s": [67.0, 68.2], "kind": "reversed"},
        {"a_range_s": [100.0, 110.0], "b_range_s": [90.0, 95.0], "kind": "sped_up"}]}
    line = content_match_line("R1", match)
    assert "E1 0.0-15.7 s = this video 0.0-15.5 s" in line and "79.7" not in line
    assert "(sped_up)" in line and "(reversed)" not in line


def test_structure_folds_frozen_repeated_and_black_into_one_line(tmp_path):
    from foren_searcher.inspect.video_structure import analyse

    fps, w, h = 10, 160, 90
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(1)
    for colour, seconds, moving in (((200, 40, 40), 3, True), ((40, 200, 40), 4, False), ((40, 40, 200), 3, True),
                                    ((0, 0, 0), 2, False)):
        base = np.zeros((h, w, 3), np.uint8)
        base[:] = colour
        for t in range(seconds * fps):
            frame = base.copy()
            if moving:
                cv2.circle(frame, (10 + (t * 3) % (w - 20), h // 2), 8, (int(rng.integers(0, 255)), 128, 128), -1)
            out.write(frame)
    out.release()
    report, _images = analyse(path)
    lines = report["summary_lines"]
    assert not any(x.startswith(("Frozen:", "Repeated picture:", "Black / blank:")) for x in lines)
    if report["frozen_spans"] or report["black_spans"]:
        (extra,) = [x for x in lines if x.startswith("Also measured: ")]
        assert "not a sign of manipulation by themselves" in extra
    assert report["frozen_spans"] is not None                                # the measurements stay in the result


def test_auto_tools_compare_mode_turns_off_ingest_structure():
    from foren_searcher.datagen import run_cases as R

    text = open(R.__file__, encoding="utf-8").read()
    assert "auto_video_structure=auto_tools == \"on\"" in text and "compare" in R.AUTO_TOOLS_MODES
