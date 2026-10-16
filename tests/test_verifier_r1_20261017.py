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
