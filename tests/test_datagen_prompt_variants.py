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
# 10-05 晚：规格 harness_params.language 决定运行时提示的语言（hi@0.7.0 / gia@0.8.0 起 en）；漂移守卫按规格语言生成
GIA_LANGUAGE = Spec.load(SPECS / GIA_LATEST).harness_params.get("language", "zh")
HI_LANGUAGE = Spec.load(SPECS / HI_LATEST).harness_params.get("language", "zh")
# 10-12：规格 inspect_merged_catalog（hi@0.8.0 / gia@0.9.0 起）也改运行时提示，守卫按规格的开关生成
GIA_MERGED = bool(Spec.load(SPECS / GIA_LATEST).harness_params.get("inspect_merged_catalog", False))
HI_MERGED = bool(Spec.load(SPECS / HI_LATEST).harness_params.get("inspect_merged_catalog", False))


def test_runtime_has_eight_distinct_gia_variants() -> None:
    variants = runtime_gia_variants()
    assert tuple(variants) == GIA_VARIANTS
    assert len(set(variants.values())) == 8


def test_latest_gia_declares_every_runtime_variant_verbatim() -> None:
    spec = Spec.load(SPECS / GIA_LATEST)
    declared = split_variants(spec.student_prompt)
    runtime = {k: v.strip() for k, v in runtime_gia_variants(GIA_LANGUAGE, GIA_MERGED).items()}
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
    text = runtime_gia_variants(GIA_LANGUAGE, GIA_MERGED)[name]
    system, ok, matched, repair = match_runtime(prompt, text, "GIA")
    assert (ok, matched, repair) == (True, name, False) and system == text.strip()
    repaired = text + repair_suffix("GIA", GIA_LANGUAGE)
    system, ok, matched, repair = match_runtime(prompt, repaired, "GIA")
    assert (ok, matched, repair) == (True, name, True) and system == repaired.strip()


def test_match_runtime_unknown_falls_back_to_main() -> None:
    prompt = Spec.load(SPECS / GIA_LATEST).student_prompt
    system, ok, matched, repair = match_runtime(prompt, "完全不同的提示", "GIA")
    assert not ok and matched is None and system == split_variants(prompt)["findings"]
    hi = Spec.load(SPECS / HI_LATEST).student_prompt
    assert match_runtime(hi, hi, "HI")[1:3] == (True, "main")


def test_latest_hi_matches_runtime_multimodal_prompt_verbatim() -> None:
    from foren_searcher.datagen.prompt_variants import _language_scope
    from foren_searcher.local_models import _hi_turn_prompt

    spec = Spec.load(SPECS / HI_LATEST)
    with _language_scope(HI_LANGUAGE, HI_MERGED):
        runtime = _hi_turn_prompt({"direct_visual_input": True, "runtime_constraints": {"remaining_seconds": 1}}, ())
    assert spec.student_prompt.strip() == runtime.strip()
    assert "rationale" in spec.student_prompt


@pytest.mark.parametrize("name,language", [("gia@0.7.0", "zh"), ("gia@0.8.0", "en")])   # 合并目录关（默认）
def test_frozen_gia_specs_still_match_runtime_in_their_language(name: str, language: str) -> None:
    # 英文开关只在 language=en 的规格下打开：中文的 gia@0.7.0 仍与运行时中文提示逐字一致（旧 run 可复现）
    spec = Spec.load(SPECS / name)
    assert spec.harness_params.get("language", "zh") == language
    assert split_variants(spec.student_prompt) == {k: v.strip() for k, v in runtime_gia_variants(language).items()}


@pytest.mark.parametrize("name,language", [("hi@0.6.0", "zh"), ("hi@0.7.0", "en")])
def test_frozen_hi_specs_still_match_runtime_in_their_language(name: str, language: str) -> None:
    from foren_searcher.datagen.prompt_variants import _language_scope
    from foren_searcher.local_models import _hi_turn_prompt

    spec = Spec.load(SPECS / name)
    with _language_scope(language):
        runtime = _hi_turn_prompt({"direct_visual_input": True, "runtime_constraints": {"remaining_seconds": 1}}, ())
    assert spec.student_prompt.strip() == runtime.strip()


def test_english_specs_have_no_cjk_in_model_facing_text() -> None:
    import re

    cjk = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
    for name in ("hi@0.7.0", "gia@0.8.0", "hi@0.8.0", "gia@0.9.0"):
        spec = Spec.load(SPECS / name)
        assert not cjk.search(spec.student_prompt), name
        assert not cjk.search(spec.teacher_instruction), name
    assert not cjk.search(repair_suffix("GIA", "en")) and not cjk.search(repair_suffix("HI", "en"))
