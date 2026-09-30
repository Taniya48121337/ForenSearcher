"""规格 harness_params → 运行时配置：改一个参数，驱动构造出的运行时配置真的变；未知键、类型、范围、冲突都报错。"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.config import Config, RuntimeConfig
from foren_searcher.datagen.harness_params import (
    HARNESS_PARAM_MAP, HarnessParamError, build_harness_settings, mapping_table, resolve_harness_params, unwired_keys,
)
from foren_searcher.datagen.specs import DEFAULT_HARNESS_PARAMS, Spec

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"


def _specs() -> tuple[Spec, Spec]:
    return Spec.load(SPECS / "hi@0.3.0"), Spec.load(SPECS / "gia@0.4.0")


def _with(spec: Spec, **params) -> Spec:
    return replace(spec, harness_params={**spec.harness_params, **params})


def _fake(spec_id: str, **params):
    """绕过 Spec 自身校验的假规格，用来测映射层对未知键的处理。"""

    return SimpleNamespace(spec_id=spec_id, name=f"{spec_id}@9.9.9", harness_params={**DEFAULT_HARNESS_PARAMS, **params})


def test_every_spec_key_has_an_explicit_mapping() -> None:
    assert set(HARNESS_PARAM_MAP) == set(DEFAULT_HARNESS_PARAMS)
    table = {row["key"]: row for row in mapping_table()}
    assert table["image_input_limit"]["target"] == "inspect_policy.maximum_visual_qa_images"
    assert table["max_workspace_turns"]["target"] == "runtime.max_workspace_turns" and table["max_workspace_turns"]["owner"] == "hi"
    assert table["read_page_chars"]["target"] == "memory_policy.max_read_chars"
    assert table["state_renderer"]["target"] == "memory_policy.state_renderer"
    assert unwired_keys() == []


def test_changing_a_spec_param_changes_the_built_runtime_config() -> None:
    hi, gia = _specs()
    config = Config(runtime=RuntimeConfig(max_workspace_turns=8))
    base = build_harness_settings(config, hi, gia)
    assert base.runtime.max_workspace_turns == 12                       # 规格 12 覆盖了配置文件的 8
    assert base.differences == {"max_workspace_turns": {"target": "runtime.max_workspace_turns", "default": 8, "spec": 12,
                                                        "source": "hi@0.3.0", "effective": 12}}
    changed = build_harness_settings(config, _with(hi, max_workspace_turns=5, read_page_chars=9000, image_input_limit=4,
                                                   state_renderer="compact_16k"),
                                     _with(gia, read_page_chars=9000, image_input_limit=4, state_renderer="compact_16k"))
    assert changed.runtime.max_workspace_turns == 5
    assert changed.memory_policy.max_read_chars == 9000 and changed.memory_policy.state_renderer == "compact_16k"
    assert changed.inspect_policy.maximum_visual_qa_images == 4
    assert changed.applied["read_page_chars"] == {"value": 9000, "source": "hi@0.3.0 + gia@0.4.0",
                                                  "target": "memory_policy.max_read_chars"}
    assert changed.runtime.max_rounds == config.runtime.max_rounds     # 规格没管的字段保持配置文件的值
    record = changed.record()
    assert record["mode"] == "apply" and set(record["differences_from_default"]) == {
        "max_workspace_turns", "read_page_chars", "image_input_limit", "state_renderer"}
    assert record["not_applicable"]["max_workspace_turns"]["source"] == "gia@0.4.0"


def test_record_mode_keeps_old_behaviour_but_reports_what_would_change() -> None:
    hi, gia = _specs()
    config = Config(runtime=RuntimeConfig(max_workspace_turns=8))
    settings = build_harness_settings(config, _with(hi, max_workspace_turns=5), gia, mode="record")
    assert settings.runtime.max_workspace_turns == 8 and settings.applied == {}
    assert settings.differences["max_workspace_turns"]["effective"] == 8
    assert settings.record()["specified_not_applied"]["max_workspace_turns"]["value"] == 5
    with pytest.raises(HarnessParamError):
        build_harness_settings(config, hi, gia, mode="bogus")  # type: ignore[arg-type]


def test_unknown_key_type_range_and_conflicts_are_errors() -> None:
    with pytest.raises(HarnessParamError, match="没有运行时映射"):
        resolve_harness_params(_fake("hi", max_tool_calls=3), _fake("gia"))  # type: ignore[arg-type]
    hi, gia = _specs()
    with pytest.raises(HarnessParamError, match="应为整数"):
        resolve_harness_params(_with(hi, max_evidence_cards="12"), gia)
    with pytest.raises(HarnessParamError, match="应为整数"):
        resolve_harness_params(_with(hi, max_workspace_turns=True), gia)
    with pytest.raises(HarnessParamError, match="下限"):
        resolve_harness_params(_with(hi, image_input_limit=0), _with(gia, image_input_limit=0))
    with pytest.raises(HarnessParamError, match="上限"):
        resolve_harness_params(_with(hi, max_workspace_turns=10_000), gia)
    with pytest.raises(HarnessParamError, match="只能是"):
        resolve_harness_params(_with(hi, state_renderer="compact_2k"), _with(gia, state_renderer="compact_2k"))
    with pytest.raises(HarnessParamError, match="两份规格里不同"):
        resolve_harness_params(_with(hi, max_model_view_chars=90_000), gia)
    # 只属于 HI 的键：GIA 规格里的不同值不报错，但记成 not_applicable
    applied, not_applicable = resolve_harness_params(_with(hi, max_workspace_turns=20), gia)
    assert applied["max_workspace_turns"]["value"] == 20 and not_applicable["max_workspace_turns"]["value"] == 12


def test_run_cases_builds_runtime_from_selected_specs(tmp_path: Path) -> None:
    """驱动入口：从 --specs 目录读 --hi-spec / --gia-spec，改了规格文件里的参数，构造出的配置随之变化。"""

    from foren_searcher.datagen import run_cases

    specs = tmp_path / "specs"
    shutil.copytree(SPECS / "hi@0.3.0", specs / "hi@0.3.0")
    shutil.copytree(SPECS / "gia@0.4.0", specs / "gia@0.4.0")
    config = Config(runtime=RuntimeConfig(max_workspace_turns=8))
    before = run_cases.harness_settings_for(config, specs, "hi@0.3.0", "gia@0.4.0")
    assert before.runtime.max_workspace_turns == 12 and before.memory_policy.max_evidence_cards == 12

    hi = Spec.load(specs / "hi@0.3.0")
    gia = Spec.load(specs / "gia@0.4.0")
    changelog = ({"targets": ["premature_finish"], "rationale": "测试"},)
    replace(hi, version="0.3.1", parent="0.3.0", harness_params={**hi.harness_params, "max_workspace_turns": 16,
                                                                  "max_evidence_cards": 20}, changelog=changelog).save(specs)
    replace(gia, version="0.4.1", parent="0.4.0", harness_params={**gia.harness_params, "max_evidence_cards": 20},
            changelog=({"targets": ["premature_finish"], "rationale": "测试"},)).save(specs)
    after = run_cases.harness_settings_for(config, specs, "hi@0.3.1", "gia@0.4.1")
    assert after.runtime.max_workspace_turns == 16 and after.memory_policy.max_evidence_cards == 20
    assert after.applied["max_evidence_cards"]["source"] == "hi@0.3.1 + gia@0.4.1"

    # 规格里写了运行时没有的键 → 报错（Spec 自己就拒绝；映射层对绕过 Spec 的输入同样拒绝）
    meta = json.loads((specs / "hi@0.3.1" / "spec.json").read_text(encoding="utf-8"))
    meta["harness_params"]["max_tool_calls"] = 3
    (specs / "hi@0.3.1" / "spec.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        run_cases.harness_settings_for(config, specs, "hi@0.3.1", "gia@0.4.1")


def test_run_cases_main_stops_before_launching_on_bad_spec(tmp_path: Path, capsys) -> None:
    from foren_searcher.datagen import run_cases

    cases = tmp_path / "cases"
    (cases / "media").mkdir(parents=True)
    (cases / "media" / "a.mp4").write_bytes(b"x")
    (cases / "videosham_1.json").write_text(json.dumps({
        "case_id": "videosham:1", "dataset": "videosham", "source_id": "1", "split": "dev_train", "task_kind": "video_diff",
        "task_text": "t", "family_key": "videosham:1", "media": [{"ref": "I0", "path": "media/a.mp4", "media_type": "video/mp4"}],
        "source_policy": {"mode": "provided_media"}}), encoding="utf-8")
    specs = tmp_path / "specs"
    shutil.copytree(SPECS / "hi@0.3.0", specs / "hi@0.3.0")
    shutil.copytree(SPECS / "gia@0.4.0", specs / "gia@0.4.0")
    assert run_cases.main(["--cases", str(cases), "--runs-dir", str(tmp_path / "runs"), "--specs", str(specs),
                           "--hi-spec", "hi@0.3.0", "--gia-spec", "gia@0.4.0", "--dry-run"]) == 0
    assert "harness_params（覆盖）" in capsys.readouterr().out
    hi = Spec.load(specs / "hi@0.3.0")
    replace(hi, version="0.3.1", parent="0.3.0", harness_params={**hi.harness_params, "max_model_view_chars": 90_000},
            changelog=({"targets": ["premature_finish"], "rationale": "测试"},)).save(specs)
    with pytest.raises(SystemExit, match="两份规格里不同"):
        run_cases.main(["--cases", str(cases), "--runs-dir", str(tmp_path / "runs"), "--specs", str(specs),
                        "--hi-spec", "hi@0.3.1", "--gia-spec", "gia@0.4.0", "--dry-run"])


def test_run_one_hands_spec_params_to_runtime_and_records_them(tmp_path: Path, monkeypatch) -> None:
    """run_one 端到端（假教师、假 Runtime）：规格值进了 InvestigationRuntime / InspectPolicy / 教师附图上限，run_config 有记录。"""

    import asyncio
    from dataclasses import dataclass

    import foren_searcher.gia as gia_package
    from foren_searcher.datagen import run_cases

    captured: dict = {}

    @dataclass
    class FakeResult:
        decision: dict

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            await captured["inspect_service"].aclose()

        async def run_new(self, **kwargs):
            return FakeResult({"verdict": "x"})

    class FakeTeacher:
        generation_observer = None

        def __init__(self) -> None:
            from foren_searcher.inspect.policy import InspectPolicy

            self.image_policy = InspectPolicy()

    teacher = FakeTeacher()
    monkeypatch.setattr(gia_package, "InvestigationRuntime", FakeRuntime)
    monkeypatch.setattr(run_cases, "make_teacher", lambda *a, **k: teacher)
    cases = tmp_path / "cases"
    (cases / "media").mkdir(parents=True)
    (cases / "media" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    case_file = cases / "videosham_1.json"
    case_file.write_text(json.dumps({
        "case_id": "videosham:1", "dataset": "videosham", "source_id": "1", "split": "dev_train", "task_kind": "video_diff",
        "task_text": "t", "family_key": "videosham:1", "media": [{"ref": "I0", "path": "media/a.png", "media_type": "image/png"}],
        "source_policy": {"mode": "provided_media"}}), encoding="utf-8")
    specs = tmp_path / "specs"
    for name in ("hi@0.3.0", "gia@0.4.0"):
        shutil.copytree(SPECS / name, specs / name)
    hi, gia = Spec.load(specs / "hi@0.3.0"), Spec.load(specs / "gia@0.4.0")
    log = ({"targets": ["premature_finish"], "rationale": "测试"},)
    replace(hi, version="0.3.1", parent="0.3.0", changelog=log,
            harness_params={**hi.harness_params, "max_workspace_turns": 7, "image_input_limit": 3, "max_candidate_cards": 5}).save(specs)
    replace(gia, version="0.4.1", parent="0.4.0", changelog=log,
            harness_params={**gia.harness_params, "image_input_limit": 3, "max_candidate_cards": 5}).save(specs)
    monkeypatch.chdir(tmp_path)          # 不读仓库的 configs/local.toml：用内置默认配置
    monkeypatch.delenv("FOREN_CONFIG", raising=False)
    asyncio.run(run_cases.run_one(case_file, "videosham_1_r0_1", runs_dir=tmp_path / "runs", round_tag="r0", sample=1,
                                  teacher="endpoint", endpoint="http://t/v1", model="m", thinking=True, specs_root=specs,
                                  hi_spec="hi@0.3.1", gia_spec="gia@0.4.1", search_mode="record",
                                  snapshot_root=tmp_path / "snap", tool_cache="on", tool_cache_root=tmp_path / "tc"))
    assert captured["max_workspace_turns"] == 7 and captured["memory_policy"].max_candidate_cards == 5
    assert captured["inspect_service"].policy.maximum_visual_qa_images == 3 and teacher.image_policy.maximum_visual_qa_images == 3
    run_config = json.loads((tmp_path / "runs" / "videosham_1_r0_1" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["harness_params_applied"]["max_workspace_turns"] == {"value": 7, "source": "hi@0.3.1",
                                                                           "target": "runtime.max_workspace_turns"}
    assert run_config["harness_params"]["differences_from_default"]["max_workspace_turns"]["default"] == 12
    assert run_config["runtime_limits"]["max_workspace_turns"] == 7
    assert run_config["search_mode"] == "record" and run_config["search_snapshot"]["snapshot_dir"].endswith("videosham_1")
    assert run_config["search_snapshot"]["counts"]["hit"] == 0 and run_config["tool_cache"]["mode"] == "on"
