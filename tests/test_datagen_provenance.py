"""来源归属（provenance）：媒体对照案例里 A/B 帧交叉引用时，核验模型要知道每条 E 属于哪份材料。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.verify.anchored import (
    _COVERAGE_SYSTEM_PROMPT_LEGACY, _PAIR_SYSTEM_PROMPT_LEGACY, COVERAGE_SYSTEM_PROMPT, PAIR_SYSTEM_PROMPT, PAIR_SYSTEM_PROMPTS,
    _source_payload, verify_anchored,
)
from foren_searcher.datagen.verify.context import DictLookup, EvidenceText

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "videosham_1028_r2"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file(), reason="videosham_1028_r2 run 未随仓库入库")


def _lookup(tmp_path: Path) -> DictLookup:
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    frame_a.write_bytes(b"A")
    frame_b.write_bytes(b"B")
    return DictLookup({
        "E1": EvidenceText("E1", "", modalities=("video",), label="材料 A", source_ref="I0"),
        "E2": EvidenceText("E2", "", modalities=("video",), label="材料 B", source_ref="I1"),
        "E3": EvidenceText("E3", "时长 4.84 秒", label="媒体属性", derived_from=("E1",)),
        "E5": EvidenceText("E5", "", modalities=("image",), label="视频E1在0.000秒的帧", derived_from=("E1",), image_path=frame_a),
        "E7": EvidenceText("E7", "", modalities=("image",), label="视频E2在0.000秒的帧", derived_from=("E2",), image_path=frame_b),
        "E8": EvidenceText("E8", "", modalities=("image",), label="图片E7裁剪区域", derived_from=("E7",), image_path=frame_b),
        "E9": EvidenceText("E9", "对齐结果", label="INSPECT align_media 结果", derived_from=("E1", "E2")),
        "E11": EvidenceText("E11", "Eiffel Tower fire was an accident", label="Snopes", source_ref="R3",
                            source_url="https://www.snopes.com/fact-check/x/"),
        "E12": "裸字符串证据",
        "E13": EvidenceText("E13", "", modalities=("image",), label="x", provenance="来源材料：Z（手工给定）"),
    })


def test_dict_lookup_provenance(tmp_path: Path):
    lookup = _lookup(tmp_path)
    assert lookup.get("E1").provenance == "来源材料：A（E1，原始材料）"
    assert lookup.get("E2").provenance == "来源材料：B（E2，原始材料）"
    assert lookup.get("E5").provenance == "来源材料：A（E1 → 视频E1在0.000秒的帧）"
    assert lookup.get("E7").provenance == "来源材料：B（E2 → 视频E2在0.000秒的帧）"
    assert lookup.get("E8").provenance == "来源材料：B（E2 → 视频E2在0.000秒的帧 → 图片E7裁剪区域）"
    assert lookup.get("E9").provenance == "来源材料：A、B（E1、E2 → INSPECT align_media 结果）"
    assert lookup.get("E11").provenance == "来源：外部网页 snopes.com"
    assert lookup.get("E12").provenance == ""
    assert lookup.get("E13").provenance == "来源材料：Z（手工给定）"       # 直接给的不覆盖


@needs_run
def test_store_lookup_provenance_on_real_run():
    from foren_searcher.datagen.runlog import load_run

    lookup = load_run(RUN).lookup
    e5, e7, e10, e12 = (lookup.get(e) for e in ("E5", "E7", "E10", "E12"))
    assert e5.provenance.startswith("来源材料：A（E1 → ") and "视频E1" in e5.provenance
    assert e10.provenance.startswith("来源材料：A（E1 → ")
    assert e7.provenance.startswith("来源材料：B（E2 → ") and "视频E2" in e7.provenance
    assert e12.provenance.startswith("来源材料：B（E2 → ")
    assert lookup.get("E1").provenance == "来源材料：A（E1，原始材料）"


def test_source_payload_provenance_policy(tmp_path: Path):
    lookup = _lookup(tmp_path)
    web = _source_payload(lookup.get("E11"))
    assert "provenance" not in web                                      # 外部网页：payload 与改动前一致，缓存继续命中
    assert web["content"] == "Eiffel Tower fire was an accident"
    frame = _source_payload(lookup.get("E7"))
    assert frame["provenance"] == "来源材料：B（E2 → 视频E2在0.000秒的帧）"
    assert frame["content"] == "（图片来源：请看附图；来源材料：B（E2 → 视频E2在0.000秒的帧））"
    derived_text = _source_payload(lookup.get("E3"))
    assert derived_text["provenance"] == "来源材料：A（E1 → 媒体属性）"
    assert derived_text["content"] == "时长 4.84 秒"


def test_prompts_carry_material_rules():
    assert "另一份材料" in PAIR_SYSTEM_PROMPT and "另一份材料" not in _PAIR_SYSTEM_PROMPT_LEGACY
    assert PAIR_SYSTEM_PROMPT.startswith(_PAIR_SYSTEM_PROMPT_LEGACY.split("- 只判断")[0])
    assert "B 的部分由 B 的来源支持" in COVERAGE_SYSTEM_PROMPT and COVERAGE_SYSTEM_PROMPT.startswith(_COVERAGE_SYSTEM_PROMPT_LEGACY)


class _FakeJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path],
                    schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((system_prompt, payload))
        if system_prompt in PAIR_SYSTEM_PROMPTS:
            source = payload["source"]
            quote = source["content"][:20] if not images else "画面文字"
            return {"supported_parts": [{"part": payload["statement"][:6], "quote": quote}], "contradicted_parts": [],
                    "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": ""}
        return {"unsupported_parts": [], "grounded": True, "notes": ""}


def test_pair_and_coverage_payloads_show_provenance(tmp_path: Path):
    lookup = _lookup(tmp_path)
    judge = _FakeJudge()
    report = {"findings": [
        {"finding_id": "F1", "statement": "A显示 lorem ipsum；B显示 A conversation", "evidence_ids": ["E5", "E7"]},
        {"finding_id": "F2", "statement": "Eiffel Tower fire was an accident", "evidence_ids": ["E11"]},
    ]}
    result = asyncio.run(verify_anchored(report, lookup, judge))
    pairs = {(p["statement"][:2], p["source"]["evidence_id"]): (prompt, p) for prompt, p in judge.calls if "source" in p}
    prompt_e7, payload_e7 = pairs[("A显", "E7")]
    assert prompt_e7 == PAIR_SYSTEM_PROMPT
    assert payload_e7["source"]["provenance"].startswith("来源材料：B")
    assert "来源材料：B" in payload_e7["source"]["content"]
    assert pairs[("A显", "E5")][1]["source"]["provenance"].startswith("来源材料：A")
    prompt_web, payload_web = pairs[("Ei", "E11")]
    assert prompt_web == _PAIR_SYSTEM_PROMPT_LEGACY and "provenance" not in payload_web["source"]

    coverage = {p["statement"][:2]: (prompt, p) for prompt, p in judge.calls if "supported_parts" in p}
    prompt_f1, payload_f1 = coverage["A显"]
    assert prompt_f1 == COVERAGE_SYSTEM_PROMPT
    assert [item["provenance"][:6] for item in payload_f1["supported_parts"]] == ["来源材料：A", "来源材料：B"]
    prompt_f2, payload_f2 = coverage["Ei"]
    assert prompt_f2 == _COVERAGE_SYSTEM_PROMPT_LEGACY and all("provenance" not in i for i in payload_f2["supported_parts"])
    assert [j.provenance[:6] for j in result.judgments] == ["来源材料：A", "来源材料：B", ""]
