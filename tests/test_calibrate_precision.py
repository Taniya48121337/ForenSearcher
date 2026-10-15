"""calibrate_precision：分句定位、C1 修复、C2 替换、hard_items 去重（假评委，不调模型）。"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

from foren_searcher.datagen.verify.anchored import PAIR_SYSTEM_PROMPTS
from foren_searcher.datagen.verify.calibrate_precision import (
    PairRef, Unit, all_pairs, char_coverage, hard_items, layer_repair, layer_swap, locate, pick_donor, split_sentences, window_at)
from foren_searcher.datagen.verify.context import EvidenceText
from foren_searcher.datagen.verify.judges import CallableJudge

LONG = ("Intro sentence about nothing in particular. " * 60
        + "The truck carrying fireworks caught fire on the Pont d'Iena at 23:15 on 14 July 2016. "
        + "Police said the Eiffel Tower itself was not on fire. " + "Closing remarks that are unrelated. " * 60)


def unit(key: str, statement: str, texts: list[EvidenceText], verdicts: list[str], *, case: str = "c:1",
         errors: list[dict] | None = None, contradicted: list[list[dict]] | None = None) -> Unit:
    js = [{"finding_id": "F1", "evidence_id": t.evidence_id, "verdict": v, "supported": [], "rejected": [],
           "contradicted": (contradicted or [[]] * len(texts))[i]} for i, (t, v) in enumerate(zip(texts, verdicts))]
    return Unit(key=key, kind="report", run=key.split("|")[0], dataset="true", case_id=case, scope="W1-1", finding_id="F1",
                statement=statement, task=None, texts=texts, orig_judgments=js, orig_errors=errors or [])


def test_split_and_locate_verbatim() -> None:
    spans = split_sentences("第一句。第二句！Third one. Fourth\n第五句")
    assert len(spans) == 5
    loc = locate("The truck carrying fireworks caught fire on the Pont d'Iena at 23:15 on 14 July 2016.", LONG)
    assert loc is not None and loc.coverage >= 0.95
    assert "Pont d'Iena" in loc.window and len(loc.window) <= 1500


def test_locate_cross_lingual_by_anchors() -> None:
    loc = locate("警方称 Eiffel Tower 本身没有起火，火情发生在 23:15 的 Iena 桥上", LONG)
    assert loc is not None and loc.tokens > 0 and "23:15" in loc.window


def test_window_at_quote() -> None:
    loc = window_at(LONG, "Police said the Eiffel Tower itself was not on fire.")
    assert loc is not None and "Police said" in loc.window
    assert window_at(LONG, "this sentence is not in the source at all") is None


def _judge(verdict_for: dict[str, str]):
    async def gen(system: str, payload: dict, images: list[Path]) -> dict:
        assert system in PAIR_SYSTEM_PROMPTS
        content = payload["source"]["content"]
        for needle, verdict in verdict_for.items():
            if needle in content:
                if verdict == "supported":
                    quote = needle
                    return {"supported_parts": [{"part": payload["statement"][:10], "quote": quote}], "contradicted_parts": [],
                            "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": ""}
        return {"supported_parts": [], "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                "scope_overreach": False, "notes": ""}
    return CallableJudge(gen)


def test_repair_outcomes() -> None:
    verbatim = EvidenceText("E1", LONG)
    fixable = EvidenceText("E2", LONG)
    hopeless = EvidenceText("E3", "完全无关的一段中文。" * 400)
    units = [unit("r1|report:W1-1|F1", "Police said the Eiffel Tower itself was not on fire.", [verbatim], ["not_found"]),
             unit("r2|report:W1-1|F1", "警方说 Eiffel Tower 没有起火，起火的是 Iena 上的 truck carrying fireworks", [fixable], ["not_found"]),
             unit("r3|report:W1-1|F1", "某个从未出现的说法", [hopeless], ["not_found"])]
    judge = _judge({"Police said the Eiffel Tower itself was not on fire.": "supported"})
    summary, rows = asyncio.run(layer_repair(all_pairs(units), judge, lambda _: None))
    by = {r["unit"]: r["outcome"] for r in rows}
    assert by["r1|report:W1-1|F1"] == "verbatim_but_unsupported"
    assert by["r2|report:W1-1|F1"] == "repair_fixed"
    assert by["r3|report:W1-1|F1"] == "unlocatable"
    assert summary["n"] == 3 and summary["repair_fixed"] == 1


def test_swap_replaces_content_and_flags_stamp(tmp_path: Path) -> None:
    img_a, img_b = tmp_path / "a.png", tmp_path / "b.png"
    img_a.write_bytes(b"a")
    img_b.write_bytes(b"b")
    t1 = EvidenceText("E5", "Original source mentions Paris fireworks. " * 20)
    t2 = EvidenceText("E9", "Another case talks about breathalyser readings. " * 20)
    i1 = EvidenceText("E7", "", modalities=("image",), image_path=img_a, media_note="这张图是第 1 秒的帧（程序记录）")
    i2 = EvidenceText("E8", "", modalities=("image",), image_path=img_b)
    units = [unit("r1|report:W1-1|F1", "巴黎烟花", [t1, i1], ["supported", "supported"], case="c:1"),
             unit("r2|report:W1-1|F1", "酒精读数", [t2, i2], ["supported", "supported"], case="c:2")]
    p = PairRef(units[0], 1)
    donor = pick_donor(p, [("c:1", t1), ("c:2", t2)], [("c:1", i1), ("c:2", i2)], random.Random(0))
    assert donor is not None and donor.image_path == img_b and donor.media_note == "" and donor.evidence_id == "E7"

    async def gen(system: str, payload: dict, images: list[Path]) -> dict:
        sup = [{"part": "x", "quote": "看到烟花"}] if images else []
        return {"supported_parts": sup, "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                "scope_overreach": False, "notes": ""}
    summary, rows = asyncio.run(layer_swap(all_pairs(units), units, CallableJudge(gen), n=10, seed=1, say=lambda _: None))
    assert summary["by_modality"]["image"]["rubber_stamp"] == 2
    assert summary["by_modality"]["text"]["rubber_stamp"] == 0     # 文本引语回查不上 → 不采信


def test_hard_items_dedupe() -> None:
    t = EvidenceText("E1", LONG)
    u = unit("r1|report:W1-1|F1", "s", [t], ["not_found"], errors=[{"type": "unsupported_assertion", "severity": "critical"}])
    rows = [{"unit": u.key, "outcome": "repair_failed", "evidence_id": "E1"}]
    agree = [{"unit": u.key, "disagree": True, "second_types": [], "sonnet_types": ["unsupported_assertion"]}]
    stable = [{"unit": u.key, "unstable": True, "perturbed_types": []}]
    hard = hard_items([u], rows, [], agree, stable, {"items": []})
    assert len(hard) == 1
    assert hard[0]["reasons"] == ["firm_error", "repair_failed", "disagree", "unstable"]
    assert char_coverage("abcdefgh", "xxabcdefghxx") == 1.0


# ---------------------------------------------------------------- verifier@0.3.4：限制句规则 + 实锤复核

from foren_searcher.datagen.verify import anchored as A  # noqa: E402


def _cov_judge(first_unsupported: list[str], second_unsupported: list[str], seen: list[dict]):
    async def gen(system: str, payload: dict, images: list[Path]) -> dict:
        seen.append({"system": system, "payload": payload})
        if "supported_parts" in payload:              # 覆盖汇总
            parts = second_unsupported if payload.get("resample") else first_unsupported
            return {"unsupported_parts": parts, "grounded": not parts, "notes": ""}
        return {"supported_parts": [{"part": "火情", "quote": "fireworks caught fire"}], "contradicted_parts": [],
                "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": ""}
    return CallableJudge(gen)


def test_confirm_downgrades_unreproduced_firm_error() -> None:
    text = EvidenceText("E1", LONG)
    seen: list[dict] = []
    out = asyncio.run(A._verify_finding("F1", "火情发生在桥上，现场 37 人受伤", [text], _cov_judge(["现场 37 人受伤"], [], seen)))
    ua = [e for e in out.errors if e.type.value == "unsupported_assertion"]
    assert ua and ua[0].severity.value == "minor" and ua[0].extra.get("unconfirmed")
    assert any(x["payload"].get("resample") == 1 for x in seen)


def test_confirm_keeps_reproduced_firm_error() -> None:
    text = EvidenceText("E1", LONG)
    out = asyncio.run(A._verify_finding("F1", "火情发生在桥上，现场 37 人受伤", [text], _cov_judge(["37 人受伤"], ["现场 37 人受伤"], [])))
    ua = [e for e in out.errors if e.type.value == "unsupported_assertion"]
    assert ua and ua[0].severity.value == "critical" and ua[0].extra.get("confirmed")


def test_features_off_is_033(monkeypatch) -> None:
    monkeypatch.setitem(A.FEATURES, "confirm_firm", False)
    monkeypatch.setitem(A.FEATURES, "limitation_rule", False)
    seen: list[dict] = []
    out = asyncio.run(A._verify_finding("F1", "本次未能核验其余时刻，火情发生在桥上", [EvidenceText("E1", LONG)], _cov_judge(["x"], [], seen)))
    assert all("resample" not in x["payload"] for x in seen)
    assert all(A._COVERAGE_LIMITATION_RULE not in x["system"] for x in seen)
    assert any(e.severity.value == "critical" for e in out.errors)


def test_limitation_rule_only_for_limitation_sentences() -> None:
    seen: list[dict] = []
    asyncio.run(A._verify_finding("F1", "本次未能核验其余时刻", [EvidenceText("E1", LONG)], _cov_judge([], [], seen)))
    asyncio.run(A._verify_finding("F2", "火情发生在桥上", [EvidenceText("E1", LONG)], _cov_judge([], [], seen)))
    cov = [x for x in seen if "supported_parts" in x["payload"]]
    assert A._COVERAGE_LIMITATION_RULE in cov[0]["system"] and A._COVERAGE_LIMITATION_RULE not in cov[1]["system"]
    assert A.is_limitation_statement("画面直接核查仅覆盖上述四个时刻") and not A.is_limitation_statement("2.333秒右缘有细窄片段")


def test_limitation_part_guard() -> None:
    assert A.is_limitation_part("画面直接核查仅覆盖上述四个时刻") is False      # "四" 是具体数量 → 不降（保守）
    assert A.is_limitation_part("完整连续对应尚未建立")
    assert A.is_limitation_part("现有媒体属性记录也未核验作者、拍摄日期或地点")
    assert not A.is_limitation_part("尚不能确定，但警方已逮捕嫌疑人")
    assert not A.is_limitation_part("未能确认 37 人受伤")
    assert not A.is_limitation_part("警方随后逮捕了涉事司机")


def test_limitation_part_downgraded_only_in_limitation_sentence() -> None:
    out = asyncio.run(A._verify_finding("F1", "火情发生在桥上；完整连续对应尚未建立", [EvidenceText("E1", LONG)],
                                        _cov_judge(["完整连续对应尚未建立"], ["完整连续对应尚未建立"], [])))
    ua = [e for e in out.errors if e.type.value == "unsupported_assertion"]
    assert ua and ua[0].severity.value == "minor" and ua[0].extra.get("limitation_part")
    out = asyncio.run(A._verify_finding("F1", "火情发生在桥上；尚不能确定，但警方已逮捕嫌疑人", [EvidenceText("E1", LONG)],
                                        _cov_judge(["尚不能确定，但警方已逮捕嫌疑人"], ["警方已逮捕嫌疑人"], [])))
    assert any(e.severity.value == "critical" for e in out.errors)
