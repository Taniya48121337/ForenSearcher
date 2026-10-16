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
