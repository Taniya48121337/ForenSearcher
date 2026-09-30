"""规格声明的提示变体必须与运行时逐字一致（漂移守卫），导出按运行时实际用的变体取 system。"""

from __future__ import annotations

from pathlib import Path

import pytest

from foren_searcher.datagen.prompt_variants import (GIA_VARIANTS, join_variants, match_runtime, repair_suffix,
                                                    runtime_gia_variants, split_variants)
from foren_searcher.datagen.specs import Spec, latest

SPECS = Path(__file__).resolve().parents[1] / "specs"
GIA_LATEST = latest(SPECS, "gia").name      # 运行时提示改了就得发新版规格；漂移守卫对最新版比
HI_LATEST = latest(SPECS, "hi").name


def test_runtime_has_eight_distinct_gia_variants() -> None:
    variants = runtime_gia_variants()
    assert tuple(variants) == GIA_VARIANTS
    assert len(set(variants.values())) == 8


def test_latest_gia_declares_every_runtime_variant_verbatim() -> None:
    spec = Spec.load(SPECS / GIA_LATEST)
    declared = split_variants(spec.student_prompt)
    runtime = {k: v.strip() for k, v in runtime_gia_variants().items()}
    assert declared == runtime       # 运行时改了提示而规格没跟上（或反过来），这里立刻失败


def test_legacy_gia_020_reads_as_findings_and_dispatch() -> None:
    # 0.2.0 的旧分隔符按 findings + dispatch 读；文本与同期收编运行时的 0.3.0 逐字相同（运行时此后已改，不再与最新版比）
    declared = split_variants(Spec.load(SPECS / "gia@0.2.0").student_prompt)
    frozen = split_variants(Spec.load(SPECS / "gia@0.3.0").student_prompt)
    assert set(declared) == {"findings", "dispatch"}
    assert declared["findings"] == frozen["findings"] and declared["dispatch"] == frozen["dispatch"]


def test_join_split_round_trip() -> None:
    variants = {"findings": "主", "wait": "等", "dispatch": "派"}
    assert split_variants(join_variants(variants)) == variants


@pytest.mark.parametrize("name", GIA_VARIANTS)
def test_match_runtime_finds_each_variant_and_repair_turns(name: str) -> None:
    prompt = Spec.load(SPECS / GIA_LATEST).student_prompt
    text = runtime_gia_variants()[name]
    system, ok, matched, repair = match_runtime(prompt, text, "GIA")
    assert (ok, matched, repair) == (True, name, False) and system == text.strip()
    repaired = text + repair_suffix("GIA")
    system, ok, matched, repair = match_runtime(prompt, repaired, "GIA")
    assert (ok, matched, repair) == (True, name, True) and system == repaired.strip()


def test_match_runtime_unknown_falls_back_to_main() -> None:
    prompt = Spec.load(SPECS / GIA_LATEST).student_prompt
    system, ok, matched, repair = match_runtime(prompt, "完全不同的提示", "GIA")
    assert not ok and matched is None and system == split_variants(prompt)["findings"]
    hi = Spec.load(SPECS / HI_LATEST).student_prompt
    assert match_runtime(hi, hi, "HI")[1:3] == (True, "main")


def test_latest_hi_matches_runtime_multimodal_prompt_verbatim() -> None:
    from foren_searcher.local_models import _hi_turn_prompt

    spec = Spec.load(SPECS / HI_LATEST)
    runtime = _hi_turn_prompt({"direct_visual_input": True, "runtime_constraints": {"remaining_seconds": 1}}, ())
    assert spec.student_prompt.strip() == runtime.strip()
    assert "rationale" in spec.student_prompt
