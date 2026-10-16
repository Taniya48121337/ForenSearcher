"""verifier@0.3.3（09-30，r0c 用 Sonnet 核验后发现）：帧的 media_note、图片对的任务上下文、多图汇总附图、答案标签语义判定。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from foren_searcher.datagen.errors import ErrorType, Severity, error
from foren_searcher.datagen.verify.anchored import (
    _COVERAGE_SYSTEM_PROMPT_LEGACY, _PAIR_SYSTEM_PROMPT_LEGACY, PAIR_SYSTEM_PROMPT, PAIR_SYSTEM_PROMPTS, coverage_system_prompt,
    pair_system_prompt, verify_anchored,
)
from foren_searcher.datagen.verify.context import DictLookup, EvidenceText, StoreLookup, frame_media_note
from foren_searcher.datagen.verify.label_judged import LABEL_SYSTEM_PROMPT, apply_label_judgment, check_labels
from foren_searcher.datagen.verify.units import label_hit
from foren_searcher.datagen.verify.verdict import VERIFIER_VERSION, Verdict


def test_version_bumped():
    # 0.3.3 的行为之后的版本都保留（0.3.4 起 anchored.FEATURES 关掉即 0.3.3）；这里只要求不低于 0.3.3
    assert tuple(int(x) for x in VERIFIER_VERSION.split("@")[1].split(".")) >= (0, 3, 3)


def test_frame_media_note_marks_first_and_last_frames():
    step = 1 / 30
    last = frame_media_note({"time_s": 6 - step, "frame_duration_s": step}, "E1", 6.0)
    assert "第 5.967 秒" in last and "总长 6.000 秒" in last and "最后一帧" in last
    first = frame_media_note(json.dumps({"time_s": 0.0, "frame_duration_s": step}), "E2", 6.0)
    assert "第一帧" in first and "视频 E2" in first
    middle = frame_media_note({"time_s": 3.0, "frame_duration_s": step}, "E1", None)
    assert "总长" not in middle and "帧" in middle
    assert frame_media_note({"page": 3}, "E9", 6.0) == "" and frame_media_note("not json", "E1", 6.0) == ""


def _record(eid: str, **kw: Any) -> SimpleNamespace:
    base = dict(evidence_id=eid, modalities=("text",), label="", content="", source_ref=None, derived_from=(), artifact_ref=None,
                locator="{}", acquired_by="", discovered_by="", production_method="")
    base.update(kw)
    return SimpleNamespace(**base)


class _FakeStore:
    def __init__(self, tmp: Path) -> None:
        (tmp / "f.png").write_bytes(b"png")
        step = 1 / 30
        self.records = {
            "E1": _record("E1", modalities=("video",), label="材料 A", content="{}", artifact_ref="1"),
            "E3": _record("E3", label="媒体属性", content=json.dumps({"duration_s": 6.0}), derived_from=("E1",),
                          acquired_by="INSPECT/media_profile"),
            "E5": _record("E5", modalities=("image",), label="视频E1在5.967秒的帧", derived_from=("E1",), artifact_ref="3",
                          locator=json.dumps({"time_s": 6 - step, "frame_duration_s": step})),
        }
        self.tmp = tmp

    def evidence(self, eid: str) -> SimpleNamespace:
        return self.records[eid]

    def evidence_records(self) -> tuple[SimpleNamespace, ...]:
        return tuple(self.records.values())

    def artifact_path(self, ref: str) -> Path:
        return self.tmp / "f.png"

    def candidate(self, ref: str) -> dict[str, Any]:
        raise KeyError(ref)


def test_store_lookup_attaches_media_note_to_frames(tmp_path: Path):
    lookup = StoreLookup(_FakeStore(tmp_path))
    frame = lookup.get("E5")
    assert "最后一帧" in frame.media_note and "总长 6.000 秒" in frame.media_note
    assert lookup.get("E3").media_note == "" and lookup.get("E1").media_note == ""


class _Judge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], list[Path]]] = []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path],
                    schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((system_prompt, payload, list(images)))
        if system_prompt in PAIR_SYSTEM_PROMPTS:
            quote = payload["source"]["content"][:10] if not images else "画面"
            return {"supported_parts": [{"part": payload["statement"][:4], "quote": quote}], "contradicted_parts": [],
                    "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": ""}
        if system_prompt == LABEL_SYSTEM_PROMPT:
            return self.label
        return {"unsupported_parts": [], "grounded": True, "notes": ""}

    label: dict[str, Any] = {}


def _frames_lookup(tmp: Path) -> DictLookup:
    (tmp / "a.png").write_bytes(b"a")
    (tmp / "b.png").write_bytes(b"b")
    return DictLookup({
        "E1": EvidenceText("E1", "{}", modalities=("video",), label="材料 A"),
        "E5": EvidenceText("E5", "", modalities=("image",), label="视频E1在0.000秒的帧", derived_from=("E1",),
                           image_path=tmp / "a.png", media_note="这张图是视频 E1 的第 0.000 秒的帧；该视频总长 6.000 秒，这是第一帧（程序记录）"),
        "E6": EvidenceText("E6", "", modalities=("image",), label="视频E1在5.967秒的帧", derived_from=("E1",),
                           image_path=tmp / "b.png", media_note="…这是最后一帧（程序记录）"),
        "E9": EvidenceText("E9", "Ventura refused to return the flag", source_url="https://example.test/x"),
    })


def test_image_pairs_carry_media_note_and_task_but_web_pairs_do_not(tmp_path: Path):
    judge = _Judge()
    report = {"findings": [
        {"finding_id": "F1", "statement": "材料A首尾两帧画面相同", "evidence_ids": ["E5", "E6"]},
        {"finding_id": "F2", "statement": "Ventura refused", "evidence_ids": ["E9"]},
        {"finding_id": "F3", "statement": "材料A开头是第一帧", "evidence_ids": ["E5"]},
    ]}
    asyncio.run(verify_anchored(report, _frames_lookup(tmp_path), judge, task="比较材料A的开头与结尾"))
    pairs = {(p["statement"][:4], p["source"]["evidence_id"]): (prompt, p) for prompt, p, _ in judge.calls if "source" in p}
    prompt, payload = pairs[("材料A首", "E5")]
    assert payload["source"]["media_note"].endswith("这是第一帧（程序记录）") and payload["task"] == "比较材料A的开头与结尾"
    assert prompt == pair_system_prompt(provenance=True, media_note=True, task=True, image=True, others="other_cited_images" in payload) and "media_note" in prompt and "指代" in prompt
    assert prompt.startswith(PAIR_SYSTEM_PROMPT.split("- 两个列表都可以为空。")[0])
    web_prompt, web_payload = pairs[("Vent", "E9")]
    assert web_prompt == _PAIR_SYSTEM_PROMPT_LEGACY and "task" not in web_payload and "media_note" not in web_payload["source"]
    coverage = {p["statement"][:4]: (prompt, p, imgs) for prompt, p, imgs in judge.calls if "supported_parts" in p}
    c_prompt, c_payload, c_images = coverage["材料A首"]
    assert c_prompt == coverage_system_prompt(provenance=True, images=True, task=True) and c_payload["task"] == "比较材料A的开头与结尾"   # 0.4.0：汇总带任务文字
    assert [i["evidence_id"] for i in c_payload["attached_images"]] == ["E5", "E6"] and len(c_images) == 2
    assert "media_note" in c_payload["attached_images"][1]
    single_prompt, single_payload, single_images = coverage["材料A开"]          # 只引一张图：不附图，提示词不变
    assert "attached_images" not in single_payload and single_images == []
    assert single_prompt == coverage_system_prompt(provenance=True, task=True)
    web_c = coverage["Vent"]
    assert web_c[0] == coverage_system_prompt(provenance=False, task=True) and "attached_images" not in web_c[1]


def test_without_task_or_note_prompts_are_unchanged():
    assert pair_system_prompt(provenance=True) == PAIR_SYSTEM_PROMPT
    assert pair_system_prompt(provenance=False) == _PAIR_SYSTEM_PROMPT_LEGACY
    assert PAIR_SYSTEM_PROMPTS[:2] == (PAIR_SYSTEM_PROMPT, _PAIR_SYSTEM_PROMPT_LEGACY)


def test_label_hit_ignores_negated_and_qualified_forms():
    assert label_hit("属实", "结论：该说法属实。")
    assert not label_hit("属实", "结论：该说法不属实。")
    assert not label_hit("属实", "结论：部分属实。")
    assert label_hit("属实", "部分细节存疑，但整体属实。")          # 另有一处未被否定
    assert label_hit("不实", "该说法不实。") and label_hit("基本属实", "结论：基本属实")
    assert label_hit("False", "Verdict: false")


def _answer_verdict(label_missing: bool) -> Verdict:
    errors = []
    if label_missing:
        errors.append(error(ErrorType.OMITTED_REQUIRED_FACT, "答案没有出现应有的标签", method="exec",
                            extra={"check": "label_expected", "forms": ["属实", "True"]}))
    return Verdict(subject={"run": "r"}, unit="answer", errors=errors, disposition="negative" if label_missing else "positive",
                   extra={"exec": {"label_match": not label_missing}})


def test_label_judgment_replaces_substring_result():
    answer = "现有证据支持该声明：视频拍摄于 2019 年。"
    v = apply_label_judgment(_answer_verdict(True), "属实", ["True"],
                             {"status": "matches", "quote": "现有证据支持该声明", "quote_verified": True, "notes": ""})
    assert v.errors == [] and v.disposition == "positive" and v.extra["label_checks"][0]["method"] == "judge"
    assert v.extra["exec"]["label_match_judged"] is True
    v = apply_label_judgment(_answer_verdict(False), "属实", ["True"],
                             {"status": "contradicts", "quote": "该说法不属实", "quote_verified": True, "notes": ""})
    assert [(e.type, e.severity) for e in v.errors] == [(ErrorType.UNSUPPORTED_ASSERTION, Severity.CRITICAL)]
    assert v.disposition == "negative"
    v = apply_label_judgment(_answer_verdict(True), "属实", ["True"],
                             {"status": "absent", "quote": "", "quote_verified": False, "notes": ""})
    assert [e.method for e in v.errors] == ["model"] and v.errors[0].type is ErrorType.OMITTED_REQUIRED_FACT
    # 摘录回查不上 / 调用失败：保留程序层结果
    v = apply_label_judgment(_answer_verdict(True), "属实", ["True"],
                             {"status": "matches", "quote": "答案里没有这句", "quote_verified": False, "notes": ""})
    assert [e.method for e in v.errors] == ["exec"] and "judge_unverified" in v.extra["label_checks"][0]
    v = apply_label_judgment(_answer_verdict(True), "属实", ["True"], None, failure="timeout")
    assert [e.method for e in v.errors] == ["exec"] and v.extra["label_checks"][0]["judge_failed"] == "timeout"
    del answer


def test_check_labels_calls_judge_with_task_and_verifies_quote():
    from foren_searcher.datagen.cases import Check

    judge = _Judge()
    judge.label = {"status": "matches", "quote": "与声明相符", "notes": ""}
    case = SimpleNamespace(task_text="核查：该视频是否拍摄于 2019 年？",
                           checks=[Check(kind="label_expected", params={"label": "属实", "aliases": ["True"]})])
    v = asyncio.run(check_labels(_answer_verdict(True), "经核对，视频内容与声明相符。", case, judge))  # type: ignore[arg-type]
    assert v.errors == [] and v.disposition == "positive"
    prompt, payload, _ = judge.calls[-1]
    assert prompt == LABEL_SYSTEM_PROMPT and payload["label"] == "属实" and payload["task"].startswith("核查")
