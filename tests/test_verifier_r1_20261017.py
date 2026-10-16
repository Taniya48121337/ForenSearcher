"""verifier@0.4.0（r1 预演之一，2026-10-17）：缓存分键 + 五处误报修正，每处一组测试。

金标条目编号见 claude/r1核验器修正_20261017.md（r0g2 的 F001–F015、r0a 的 F001/F002）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from foren_searcher.datagen.verify import judges
from foren_searcher.datagen.verify.judges import CachedJudge, CallableJudge
from foren_searcher.datagen.verify.verdict import VERIFIER_VERSION


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------- 0. 缓存键带核验器版本

def test_cache_key_namespaced_by_verifier_version(tmp_path: Path, monkeypatch: Any) -> None:
    assert VERIFIER_VERSION == "verifier@0.4.0"
    new = CachedJudge.key_for("S", {"a": 1}, [])
    monkeypatch.setattr(judges, "cache_namespace", lambda: "")
    old = CachedJudge.key_for("S", {"a": 1}, [])
    assert new != old                                    # 同一道题，0.3.x 的旧回答不命中
    # 旧缓存里有这道题的回答，0.4.0 的 CachedJudge 照样重新调用
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"key": old, "answer": {"x": "old"}}) + "\n", encoding="utf-8")
    monkeypatch.undo()
    calls: list[int] = []

    async def gen(system_prompt: str, payload: dict[str, Any], images: list[Path]) -> dict[str, Any]:
        calls.append(1)
        return {"x": "new"}

    judge = CachedJudge(CallableJudge(gen), path)
    assert run(judge.judge(system_prompt="S", payload={"a": 1}, images=[])) == {"x": "new"} and calls == [1]


# ---------------------------------------------------------------- 1. 充分性层不给 GT

from foren_searcher.datagen import cases as cases_mod                                   # noqa: E402
from foren_searcher.datagen.errors import ErrorType, Severity                            # noqa: E402
from foren_searcher.datagen.verify.context import DictLookup, EvidenceText              # noqa: E402
from foren_searcher.datagen.verify.sufficiency import (GT_FREE_EDIT_REQUIREMENT, sufficiency_payload,  # noqa: E402
                                                       verify_sufficiency)


class RecordingJudge:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer, self.payloads = answer, []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path],
                    schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.answer


def e17_case() -> cases_mod.Case:
    """averimatec:e17 的结构：dates（按取得方式）+ external + 标注的最早来源（GT，按网址），REQ1 必需、REQ3 可选。"""

    return cases_mod.Case(
        case_id="averimatec:e17", dataset="averimatec", source_id="e17", split="dev_train", task_kind="provenance",
        task_text="t", family_key="averimatec:17",
        requirements=(cases_mod.Requirement("REQ1", "最早一份", text_en="The earliest copy of the image that can be found and its date"),
                      cases_mod.Requirement("REQ3", "未能确定", must_cite=False, text_en="What could not be determined")),
        required_evidence=(
            cases_mod.RequiredEvidence("dates", "SEARCH 日期查询的结果", acquired_by="SEARCH/dates", supports=("REQ1",)),
            cases_mod.RequiredEvidence("external", "外部网页", acquired_by="SEARCH/open", supports=("REQ1",)),
            cases_mod.RequiredEvidence("earliest_source", "标注的最早来源：gettyimages.co.uk/detail/news-photo/461948087（2013-12-12）",
                                       url_pattern="gettyimages.co.uk/detail/news-photo/461948087*", must_cite=False, supports=("REQ1",))),
        required_evidence_rule="all:dates,external; any_one_of:earliest_source,dates")


def e17_report(status: str = "COMPLETED") -> dict[str, Any]:
    return {"task_id": "W1-1", "status": status, "findings": [
        {"finding_id": "F1", "statement": "The earliest dated copy found is on example.org, declared 2022-07-09.",
         "evidence_ids": ["E4", "E5"], "limitations": []}]}


def test_sufficiency_payload_has_no_gt() -> None:
    lookup = DictLookup({"E4": EvidenceText("E4", "page", label="Slum on rail tracks", source_url="https://example.org/a",
                                            acquired_by="SEARCH/open"),
                         "E5": EvidenceText("E5", "published date declared by the page: 2022-07-09", label="dates of https://example.org/a",
                                            acquired_by="SEARCH/dates")})
    payload = sufficiency_payload(e17_report(), "Find where and when this image was first published.", case=e17_case(), lookup=lookup)
    blob = json.dumps(payload, ensure_ascii=False)
    assert "gettyimages" not in blob and "2013-12-12" not in blob and "标注" not in blob
    assert [r["key"] for r in payload["required_evidence"]] == ["dates", "external"]     # any_one_of 组里只剩 dates，组合并成单项
    assert [(r["key"], r["must_cite"]) for r in payload["requirements"]] == [("REQ1", True), ("REQ3", False)]
    assert payload["evidence_index"] == [
        {"evidence_id": "E4", "label": "Slum on rail tracks", "acquired_by": "SEARCH/open", "source_url": "https://example.org/a",
         "modalities": ["text"]},
        {"evidence_id": "E5", "label": "dates of https://example.org/a", "acquired_by": "SEARCH/dates", "source_url": None,
         "modalities": ["text"]}]
    # 没有 lookup（walker 循环内）：只列编号
    assert sufficiency_payload(e17_report(), "t", case=e17_case())["evidence_index"] == [{"evidence_id": "E4"}, {"evidence_id": "E5"}]


def test_videosham_requirement_with_official_description_is_redacted() -> None:
    case = cases_mod.Case(
        case_id="videosham:4112", dataset="videosham", source_id="4112", split="dev_train", task_kind="media_compare", task_text="t",
        family_key="videosham:4112",
        requirements=(cases_mod.Requirement("REQ1", "A 是否有官方描述的编辑（\"Replace the sound with an accordion sound 00: 00-00: 10\"）",
                                            text_en='Whether A has the edit the official description gives ("Replace the sound with an '
                                                    'accordion sound 00: 00-00: 10"; to be confirmed from the footage)'),
                      cases_mod.Requirement("REQ2", "音轨", text_en="Whether the audio tracks could be obtained")))
    payload = sufficiency_payload({"findings": []}, "Compare A with B.", case=case)
    assert payload["requirements"][0]["requirement"] == GT_FREE_EDIT_REQUIREMENT
    assert "accordion" not in json.dumps(payload) and payload["requirements"][1]["requirement"] == "Whether the audio tracks could be obtained"


def test_requirement_gap_sets_status_without_extra_omitted_error() -> None:
    judge = RecordingJudge({"requirements": [{"key": "REQ1", "covered": False, "by": None, "note": "no date kind"},
                                             {"key": "dates", "covered": True, "by": "F1", "note": ""},
                                             {"key": "external", "covered": True, "by": "F1", "note": ""}],
                            "answerable_from_report": False, "status_should_be": "INCOMPLETE",
                            "missing": ["What kind of date the earliest copy carries"]})
    result = run(verify_sufficiency(e17_report(), "t", judge, case=e17_case()))
    assert [e.type for e in result.errors] == [ErrorType.COMPLETED_WITHOUT_FINISHING]
    assert "REQ1" in result.errors[0].detail and result.raw["required_keys"] == ["dates", "external", "REQ1", "REQ3"]
    # 必需证据没取到照旧记 omitted（major），可选的 REQ3 不记
    judge = RecordingJudge({"requirements": [{"key": "dates", "covered": False, "by": None, "note": ""},
                                             {"key": "REQ3", "covered": False, "by": None, "note": ""}],
                            "answerable_from_report": True, "status_should_be": "COMPLETED", "missing": []})
    result = run(verify_sufficiency(e17_report("INCOMPLETE"), "t", judge, case=e17_case()))
    assert [(e.type, e.severity, e.extra.get("requirement")) for e in result.errors] == [
        (ErrorType.OMITTED_REQUIRED_FACT, Severity.MAJOR, "dates")]


# ---------------------------------------------------------------- 2. failed 页 / B 族精度不单独算缺口

from foren_searcher.datagen.verify.sufficiency import SUFFICIENCY_SYSTEM_PROMPT, failed_pages_only   # noqa: E402

FAILED_ONLY = ["2 of 6 candidate pages failed to load and were not examined",
               "Examination of the 8 candidate pages that failed to load",
               "3 of 8 candidate pages were not examined, so an earlier copy may exist.",
               "The 10 of 14 candidate pages that failed to load were not retried",
               "8 of 12 candidate external pages were not examined, so an earlier or other source reporting a booster loss may exist among them.",
               "6 of 9 candidate pages were not examined.",
               "3 个候选页打不开，未检查"]
REAL_GAPS = ["Only one external source (Business Insider) was used. 3 of 5 candidate pages failed to load and were not examined.",
             "The image's original publisher and first publication date were not determined, and 9 of 15 candidate pages were not examined.",
             "There is no independent corroboration or SpaceX primary source; only CBS was used, and 2 of 5 candidate pages did not load.",
             "No visual comparison confirming that the Diplomat image is the same as the supplied image",
             "No findings from video frames or a transcript describing what the video shows or says"]


def test_failed_pages_only_classifier_on_r0g_items() -> None:
    assert all(failed_pages_only(x) for x in FAILED_ONLY)
    assert not any(failed_pages_only(x) for x in REAL_GAPS)


def test_failed_pages_alone_do_not_flag_completed() -> None:
    covered = [{"key": "REQ1", "covered": True, "by": "F1", "note": ""}, {"key": "dates", "covered": True, "by": "F1", "note": ""},
               {"key": "external", "covered": True, "by": "F1", "note": ""}]
    judge = RecordingJudge({"requirements": covered, "answerable_from_report": True, "status_should_be": "INCOMPLETE",
                            "missing": [FAILED_ONLY[0]], "limitations_not_gaps": []})
    result = run(verify_sufficiency(e17_report(), "t", judge, case=e17_case()))
    assert result.errors == [] and result.raw["missing"] == [] and result.raw["missing_failed_pages"] == [FAILED_ONLY[0]]
    # 同一份报告，真缺口（E 族没确认是同一张图）照样判
    judge = RecordingJudge({"requirements": covered, "answerable_from_report": False, "status_should_be": "INCOMPLETE",
                            "missing": [FAILED_ONLY[0], REAL_GAPS[3]], "limitations_not_gaps": []})
    result = run(verify_sufficiency(e17_report(), "t", judge, case=e17_case()))
    assert [e.type for e in result.errors] == [ErrorType.COMPLETED_WITHOUT_FINISHING] and result.raw["missing"] == [REAL_GAPS[3]]


def test_prompt_rules_for_failed_pages_and_unrequested_precision() -> None:
    assert "limitations_not_gaps" in SUFFICIENCY_SYSTEM_PROMPT and "候选页打不开" in SUFFICIENCY_SYSTEM_PROMPT
    assert "任务没有明确要求的精度" in SUFFICIENCY_SYSTEM_PROMPT and "网页上的图是否就是给定的" in SUFFICIENCY_SYSTEM_PROMPT


# ---------------------------------------------------------------- 3. 像素检查只对句子所指的那张图

from foren_searcher.datagen.verify import anchored as A                                # noqa: E402
from foren_searcher.datagen.verify.anchored import Judgment, Part                      # noqa: E402
from foren_searcher.datagen.verify.pixel_rules import image_type, is_tool_measurement   # noqa: E402


def J(eid: str, *, image: bool, label: str, provenance: str, sup: list[str] = (), con: list[tuple[str, str]] = (),
      media_note: str = "") -> Judgment:
    return Judgment("F1", eid, "contradicted" if con else "supported", supported=[Part(p, "q", None if image else True) for p in sup],
                    contradicted=[Part(p, q, None if image else True) for p, q in con], is_image=image, source_label=label,
                    provenance=provenance, media_note=media_note)


def firm(errors):
    return [(e.type.value, e.severity.value, e.extra.get("program_rule")) for e in errors]


def test_tool_measurement_not_refuted_by_pixels_F006() -> None:
    statement = ("The comparison matches 100% of Material A's half-second samples to Material B, with matched footage in A 0–4 s and "
                 "B 0–4 s. The comparison image at A 3.2 s and B 3.2 s shows corresponding river scenery.")
    part = "The comparison matches 100% of Material A's half-second samples to Material B, with matched footage in A 0–4 s and B 0–4 s."
    text = J("E25", image=False, label="INSPECT compare_media result", provenance="来源材料：A、B（E1、E2 → INSPECT compare_media result）",
             sup=["The comparison matches 100% of Material A's half-second samples", "with matched footage in A 0–4 s and B 0–4 s"])
    panel = J("E21", image=True, label="A 0.2-4.2 s: region in red differs from B at the same moment", provenance="来源材料：A、B（E1、E2 → …）",
              sup=["corresponding river scenery"], con=[(part, "A 3.2s shows a wooden boat inside the red box")])
    cov = {"finding_id": "F1", "grounded": True, "model_grounded": True, "unsupported_parts": [], "notes": ""}
    assert firm(A._errors_for_finding("F1", statement, [text, panel], cov)) == [("wrong_citation", "minor", "tool_measurement")]
    assert is_tool_measurement("100% of A's half-second samples match footage in B",
                               "The comparison reports that 100% of A's half-second samples match footage in B.")
    assert not is_tool_measurement("a wooden boat appears in both frames", "The comparison image shows a wooden boat in both frames.")


def test_other_image_referent_F012_and_real_contradiction_kept() -> None:
    statement = ("The specific visual evidence consists of whole-frame A/B comparison panels identifying the participants and paired crops "
                 "of the lower-center participant. At both 0.0 s and 2.8 s, A's crop contains the decorated cyan hat where B's crop shows "
                 "hair and background.")
    panel = J("E21", image=True, label="A 0.3-5.3 s: region in red differs from B at the same moment", provenance="来源材料：A、B（…）",
              sup=["whole-frame A/B comparison panels"],
              con=[("paired crops of the lower-center participant", "the detail is on the top-left tile"),
                   ("the decorated cyan hat", "A detail shows an orange hat")])
    crop_a = J("E37", image=True, label="frame of video E1 at 0.000 s (cropped region)", provenance="来源材料：A（E1 → …）",
               sup=["A's crop contains the decorated cyan hat"], media_note="这张图是视频 E1 的第 0.000 秒的帧")
    crop_b = J("E34", image=True, label="frame of video E2 at 0.000 s (cropped region)", provenance="来源材料：B（E2 → …）",
               sup=["B's crop shows hair and background"], media_note="这张图是视频 E2 的第 0.000 秒的帧")
    assert [image_type(x) for x in (panel, crop_a)] == ["diff_panel", "crop"]
    errors = A._errors_for_finding("F1", statement, [panel, crop_a, crop_b], None)
    assert firm(errors) == [("wrong_citation", "minor", "other_image_referent")] * 2
    # 只引了这一张对照面板：没有"另一张图"可指，画面反驳照记
    errors = A._errors_for_finding("F1", statement, [panel], None)
    assert [e.severity.value for e in errors if e.type.value == "wrong_citation"] == ["critical", "critical"]


def test_unrelated_reference_images_collapse_to_one_minor_F007() -> None:
    statement = "The photograph shows a rural road with large sections of its paved surface displaced, alongside motorcycles, people and vegetation."
    given = J("E1", image=True, label="Material A", provenance="来源材料：A（E1，原始材料）",
              sup=["The photograph shows a rural road with large sections of its paved surface displaced", "alongside motorcycles",
                   "people", "and vegetation"])
    refs = [J(e, image=True, label=f"Highways page {e}", provenance=f"来源：外部网页 site{e}.com",
              con=[("The photograph shows a rural road with large sections of its paved surface displaced", "an intact highway"),
                   ("alongside motorcycles, people and vegetation", "only cars")]) for e in ("E6", "E9", "E12")]
    errors = A._errors_for_finding("F1", statement, [given, *refs], None)
    assert firm(errors) == [("wrong_citation", "minor", "cited_unrelated_image")]
    assert list(errors[0].evidence_ids) == ["E6", "E9", "E12"] and errors[0].extra["merged"] == 6 and errors[0].extra["described"] == ["E1"]
    # 外部参考图反驳、给定图并不支持同一部分：照记
    lone = J("E1", image=True, label="Material A", provenance="来源材料：A（E1，原始材料）", sup=["a city street"])
    errors = A._errors_for_finding("F1", statement, [lone, refs[0]], None)
    assert {e.severity.value for e in errors if e.type.value == "wrong_citation"} == {"critical"}


def test_image_pair_prompt_rules_and_other_cited_images() -> None:
    assert A._IMAGE_RULE in A.pair_system_prompt(provenance=True, image=True)
    assert A._OTHERS_RULE in A.pair_system_prompt(provenance=True, image=True, others=True)
    assert A.pair_system_prompt(provenance=False) == A._PAIR_SYSTEM_PROMPT_LEGACY       # 文本对不带图片规则
    seen: list[dict[str, Any]] = []

    class Judge:
        async def judge(self, *, system_prompt, payload, images, schema=None):
            seen.append({"prompt": system_prompt, **payload})
            return {"supported_parts": [], "contradicted_parts": []}

    img = tmp_png()
    texts = [EvidenceText("E21", "", modalities=("image",), label="A 0.3-5.3 s: region in red differs", image_path=img,
                          provenance="来源材料：A、B（…）"),
             EvidenceText("E37", "", modalities=("image",), label="frame of video E1 at 0.000 s (cropped region)", image_path=img,
                          provenance="来源材料：A（…）", media_note="这张图是视频 E1 的第 0.000 秒的帧"),
             EvidenceText("E27", "compare_media text", label="INSPECT compare_media result")]
    run(A._verify_finding_once("F1", "s", texts, Judge()))
    pairs = {p["source"]["evidence_id"]: p for p in seen if "source" in p}
    assert pairs["E21"]["other_cited_images"] == [{"evidence_id": "E37", "label": "frame of video E1 at 0.000 s (cropped region)",
                                                   "media_note": "这张图是视频 E1 的第 0.000 秒的帧"}]
    assert "other_cited_images" not in pairs["E27"] and A._IMAGE_RULE not in pairs["E27"]["prompt"]


def tmp_png() -> Path:
    import tempfile
    path = Path(tempfile.mkdtemp()) / "x.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


# ---------------------------------------------------------------- 4. 引语回查（短词 / 省略号）、超长来源、汇总带任务文字

from foren_searcher.datagen.verify.anchored import focused_content, quote_in_source   # noqa: E402

OCR = json.dumps({"summary": {"texts": [{"text": t} for t in ("Guardian", "ISSF", "PARiS", "2024", "May 2024", "TEAMSTERSINTERNALPOLL",
                                                              "34%", "59.6%", "KAMALA", "HARRIS", "TRUMP")]}})


def test_quote_check_accepts_short_tokens_and_ellipsis_F008_F011() -> None:
    assert all(quote_in_source(q, OCR) for q in ("ISSF", "PARiS", "2024", "“ISSF”", "KAMALA ... HARRIS ... TRUMP", "KAMALA … HARRIS",
                                                 '"text": "34%" ... "text": "59.6%"', "TEAMSTERSINTERNALPOLL"))
    assert not any(quote_in_source(q, OCR) for q in ("OBAMA ... HARRIS", "ISS", "SS", "2025", "KAMALA ... BIDEN"))
    assert quote_in_source("visited the cemetery … photographed", "AFP reporters visited the cemetery on May 2 and photographed graves")


def test_focused_content_keeps_head_and_statement_hits() -> None:
    body = "HEAD " + "x" * 200 + "".join(f" frame{i:05d}" for i in range(20000)) + ' "KAMALA" at 105.5 s ' + "y" * 5000
    shown, cut = focused_content(body, 'OCR identifies "KAMALA" in the video', limit=4000)
    assert cut and shown.startswith("HEAD ") and "KAMALA" in shown and len(shown) < 4000 + 200
    assert focused_content("short", "s") == ("short", False)


def test_coverage_gets_task_text_F002_F003() -> None:
    assert A._COVERAGE_TASK_RULE in A.coverage_system_prompt(provenance=False, task=True)
    seen: list[dict[str, Any]] = []

    class Judge:
        async def judge(self, *, system_prompt, payload, images, schema=None):
            seen.append({"prompt": system_prompt, **payload})
            if "source" in payload:
                return {"supported_parts": [{"part": "HISTORICTRUMPVICTORY", "quote": "TRUMP"}], "contradicted_parts": []}
            return {"unsupported_parts": [], "grounded": True, "notes": ""}

    task = "Fact-check the following claim, which circulated as the title of the given video: Donald Trump wins 2024 presidential election"
    texts = [EvidenceText("E12", OCR, label="OCR of E1", acquired_by="INSPECT/ocr")]
    run(A._verify_finding_once("F2", "OCR shows HISTORICTRUMPVICTORY, matching the title's assertion of a Trump victory.", texts, Judge(),
                               task=task))
    coverage = [p for p in seen if "supported_parts" in p and "source" not in p]
    assert coverage and coverage[0]["task"] == task and A._COVERAGE_TASK_RULE in coverage[0]["prompt"]
