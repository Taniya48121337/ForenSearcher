"""10-16 r0+：status_rewrite.build_reason 只去掉与案卷 GT 字段实际匹配的条目；循环内充分性检查与 GIA 第 2 轮派工。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen import status_rewrite as sr  # noqa: E402

CASE = {"notes": "标注来源 https://web.archive.org/web/20191217225513/https://www.kshs.org/p/kansas-state-capitol/16586。",
        "required_evidence": [{"key": "earliest_source", "description": "标注的最早来源：tribuneindia.com/2011/main/x.htm（2011-11-21）",
                               "url_pattern": "tribuneindia.com/2011/main/x.htm*"}],
        "checks": [{"kind": "date_in_range", "params": {}, "note": "标注：最早发布 2011-11-21（When…）"},
                   {"kind": "entity_in_evidence", "params": {"text": "Kansas"}}],
        "requirements": [{"id": "REQ1", "text": "图片是否拍于声明所说的地点（与该地点的参考照片当面比对：建筑、地标、地形、标牌）",
                          "text_en": "Whether the picture shows the place the claim names (compared with reference photos)"},
                         {"id": "REQ2", "text": "载有这张图片的来源说它实际拍于哪里",
                          "text_en": "Where the sources that carry the picture say it was actually taken"}]}


def test_gt_fields_from_case():
    gt = sr.gt_fields(CASE)
    assert "kshs.org" in gt["hosts"] and "tribuneindia" in gt["sites"] and "kshs" in gt["sites"]
    assert gt["dates"] == ["2011-11-21"] and gt["entities"] == ["Kansas"]


def test_capitalised_words_are_not_gt():
    suff = {"missing": ["Confirmation that the Andhrafriends page actually contains this photograph.",
                        "Whether the people filming were local residents",
                        "The first publication source and date, which the labelled earliest source (Tribune India, 2011-11-21) would supply",
                        "Where it was really taken: the Kansas State Capitol was never checked."]}
    got = sr.build_reason(suff, [], "{}", CASE["requirements"], CASE)
    assert got["kept"] == ["Confirmation that the Andhrafriends page actually contains this photograph",
                           "Whether the people filming were local residents"]
    assert [d["tokens"] for d in got["dropped"]] == [["Tribune India", "2011-11-21"], ["Kansas"]]


def test_visible_gt_token_is_kept():
    # 学生自己读到了 Tribune India 的页面：条目里提到它不是泄漏
    visible = json.dumps({"page": "https://www.tribuneindia.com/2011/main/x.htm dated 2011-11-21"})
    got = sr.build_reason({"missing": ["The Tribune India page (2011-11-21) is not cited."]}, [], visible, [], CASE)
    assert got["dropped"] == [] and got["kept"] == ["The Tribune India page (2011-11-21) is not cited"]


def test_times_is_not_hindustantimes():
    case = {"required_evidence": [{"key": "earliest_source", "description": "标注的最早来源：hindustantimes.com/india-news/x（2021-08-12）"}]}
    got = sr.build_reason({"missing": ["The report did not confirm that the photo matches the Times of India report."]}, [], "{}", [], case)
    assert got["dropped"] == []


def test_chinese_items_map_to_requirement_text_en():
    suff = {"missing": ["图片是否确实拍摄于美国国会大厦，未得到核验（没有与国会大厦参考照片做建筑特征比对）", "照片实际拍摄地点未确定",
                        "没有找到载有这张图片或该地点照片的页面，只有一条文字帖子"]}
    got = sr.build_reason(suff, [], "{}", CASE["requirements"], CASE)
    assert got["kept"] == ["not established: whether the picture shows the place the claim names (compared with reference photos)",
                           "not established: where the sources that carry the picture say it was actually taken"]
    assert len(got["mapped_cjk"]) == 3 and got["source"] == "missing"


# ---------------------------------------------------------------------------- 循环内检查 → GIA 第 2 轮

from foren_searcher.datagen import walker as w  # noqa: E402

E_TASK = ('The given image circulated with the following claim: "Photo shows a protest in Delhi" (claim date: 2023-01-16). '
          "Find where and when this image was first published, as far as the evidence allows, and whether its original "
          "context matches the claim. Cite the sources and dates you rely on, and state what could not be determined.")


def _gia_payload(reason: str, task: str = E_TASK) -> dict:
    report = {"task_id": "W1-1", "status": "INCOMPLETE", "status_reason": reason,
              "findings": [{"finding_id": "F1", "statement": "x", "evidence_ids": ["E4"]}]}
    return {"global_view": {"task": task, "latest_reports": [{"task": task, "report": report}],
                            "original_materials": [{"evidence_id": "E1", "reference_id": "E1", "modalities": ["image"]}],
                            "image_inputs": [{"evidence_id": "E1"}],
                            "runtime_constraints": {"round_index": 2, "can_create_round": True, "remaining_turns": 6,
                                                    "max_tasks_per_round": 2}}}


def test_inloop_reason_names_unread_pages_and_gia_dispatches_round2():
    hints = {"unread": [{"evidence_id": "E19", "url": "https://geologyscience.com/landslides", "reverse": True}], "refs": [],
             "image": "E1"}
    items = ["The report does not identify where and when the image was first published",
             "8 of 19 candidate pages failed to load and were not examined"]
    reason = w.inloop_status_reason(items, hints, "E")
    assert reason.endswith("candidate pages that were found but not read: https://geologyscience.com/landslides")
    got = w.decide_gia(_gia_payload(reason), {})
    assert got.rule == "G2_round2_search_gap"
    assert "open and read https://geologyscience.com/landslides" in got.slot["scope"]
    # 没答上的点作为要补的点交给 S2；"N 个页面打不开"不算
    assert items[0] in got.slot["gaps"] and items[1] not in got.slot["gaps"]


def test_no_unread_pages_means_search_more():
    reason = w.inloop_status_reason(["Whether the original context matches the claim is not established"], {"unread": []}, "E")
    assert reason.endswith("a further search for sources can address the points above")
    got = w.decide_gia(_gia_payload(reason), {})
    assert got.rule == "G2_round2_search_gap" and "search for sources on the open points" in got.slot["scope"]
    assert got.slot["gaps"] == ["Whether the original context matches the claim is not established"]
    assert w._SEARCH_SCOPE.search("Check the date. " + got.slot["scope"])


def test_old_unfixable_reason_still_finishes():
    got = w.decide_gia(_gia_payload("no date of any copy of the image was established"), {})
    assert got.kind == "finish"


def test_l_references_clause_only_for_location_points():
    hints = {"unread": [], "refs": ["E5", "E6"], "image": "E1"}
    reason = w.inloop_status_reason(["Whether the picture shows the claimed place (no comparison with reference photos)"], hints, "L")
    assert "reference photos E5, E6 of the claimed place were not compared feature by feature with E1" in reason
    assert w._search_gap("the reference photos E5, E6 of the claimed place were not compared feature by feature with E1")["kind"] == "references"
    assert w.inloop_actions(["The report does not cite a date"], hints, "L") == ["a further search for sources can address the points above"]
    assert w.inloop_actions(["x"], hints, "B") == []


def _replay(case: str, upto: str) -> tuple:
    run = ROOT / "experiments" / "sft_data" / "runs" / f"averimatec_{case}_r0g_1" / "requests"
    if not run.is_dir():
        import pytest
        pytest.skip("r0g run not present")
    memory: dict = {}
    got = None
    for rd in sorted(run.glob("request_*"), key=lambda p: int(p.name.split("_")[-1])):
        req = json.loads((rd / "request.json").read_text(encoding="utf-8"))
        if req["payload"].get("workspace_id") != "W1-1":
            continue
        got = w.decide_hi(req["payload"], memory, req.get("rendered_payload"))
        if rd.name == upto:
            break
    return got, memory


def test_l_reads_reverse_image_pages_first_and_keeps_report_closed():
    # r0g l208：以图搜之后（request_0006），以图搜结果页 E19（geologyscience，写着 Peru）排在未读清单最前，不许交报告
    got, _ = _replay("l208", "request_0006")
    assert got.kind == "next_slot" and "report" not in got.slot["choices"]
    assert got.slot["unread"][0]["evidence_id"] == "E19"
    assert got.slot["report_decision"]["rule"] == "L9_report"


def test_l3b_is_the_step_right_before_report():
    from types import SimpleNamespace

    state, memory = SimpleNamespace(remaining=5), {}
    got = w._l3b_before_report(state, memory, "E1", ["E7", "E8"], compared=True, refs_shown=False, reviewed=False)
    # 读给定照片本身（L3 读的是参考图 side[0]），参考图作焦点；理由交给 S0
    assert got.rule == "L3b_view_references_before_report" and got.output["arguments"]["record_id"] == "E1"
    assert got.output["focus_evidence_ids"] == ["E1", "E7", "E8"] and got.slot.get("substituted") and memory["l3b_offered"]
    # 已经补看过（同记录同焦点再发会被运行时合并、插中文 harness_event）、参考图正在附图里、没做过比对 → 不发
    for kw in ({"reviewed": True, "refs_shown": False}, {"reviewed": False, "refs_shown": True}):
        assert w._l3b_before_report(state, {}, "E1", ["E7"], compared=True, **kw) is None
    assert w._l3b_before_report(state, {}, "E1", ["E7"], compared=False, refs_shown=False, reviewed=False) is None
