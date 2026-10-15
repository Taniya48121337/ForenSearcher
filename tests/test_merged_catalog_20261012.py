"""10-12 合并目录：规格 hi@0.8.0 / gia@0.9.0（inspect_merged_catalog）、运行时提示、run_cases 的操作集与 --detect 别名。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

from foren_searcher.config import load_config
from foren_searcher.datagen import run_cases as R
from foren_searcher.datagen.harness_params import InspectOperationSwitches, build_harness_settings
from foren_searcher.datagen.prompt_variants import runtime_gia_variants, spec_scope, split_variants
from foren_searcher.datagen.specs import Spec
from foren_searcher.inspect.contracts import MERGED_OPERATIONS

SPECS = Path(__file__).resolve().parents[1] / "specs"
OLD_OPS = ("align_media", "diff_media", "get_pages", "locate_frames", "locate_regions", "crop", "visual_qa", "audio_qa")


def test_new_specs_match_the_runtime_under_their_own_switches():
    from foren_searcher.local_models import _hi_turn_prompt

    hi, gia = Spec.load(SPECS / "hi@0.8.0"), Spec.load(SPECS / "gia@0.9.0")
    assert hi.harness_params["inspect_merged_catalog"] is gia.harness_params["inspect_merged_catalog"] is True
    assert "inspect_diff_media" not in hi.harness_params
    with spec_scope(hi):
        runtime = _hi_turn_prompt({"direct_visual_input": True, "runtime_constraints": {"remaining_seconds": 1}}, ())
    assert hi.student_prompt.strip() == runtime.strip()
    assert split_variants(gia.student_prompt) == {k: v.strip() for k, v in runtime_gia_variants("en", True).items()}


def test_merged_hi_prompt_names_only_merged_operations():
    hi = Spec.load(SPECS / "hi@0.8.0").student_prompt
    for name in OLD_OPS:
        assert not re.search(rf"\b{name}\b", hi), name
    for name in ("compare_media", "video_structure", "content_match", "get_frames box", "locate"):
        assert name in hi
    # 归因规则在提示里，工具结果只陈述事实
    assert "likely generated clip" in hi and "never evidence that the video is authentic" in hi
    old = Spec.load(SPECS / "hi@0.7.0").student_prompt
    assert "align_media" in old and "compare_media" not in old


def test_gia_compare_note_only_when_merged():
    merged = runtime_gia_variants("en", True)
    plain = runtime_gia_variants("en", False)
    assert "compare_media" in merged["dispatch"] and "compare_media" in merged["findings_budget_out"]
    assert all("compare_media" not in text and "DETECT" not in text for text in plain.values())
    assert merged["budget_out"] == plain["budget_out"]


def test_text_mode_hi_prompt_is_also_rewritten():
    from foren_searcher.datagen.prompt_variants import _language_scope
    from foren_searcher.local_models import _hi_turn_prompt

    with _language_scope("en", True):
        text = _hi_turn_prompt({"runtime_constraints": {}}, ())
    assert "compare_media" in text and "align_media" not in text and "get_pages" not in text


def test_shared_switch_and_mismatched_pairs():
    config = load_config()
    both = build_harness_settings(config, Spec.load(SPECS / "hi@0.8.0"), Spec.load(SPECS / "gia@0.9.0"))
    assert both.inspect_operations.merged_catalog is True
    old = build_harness_settings(config, Spec.load(SPECS / "hi@0.7.0"), Spec.load(SPECS / "gia@0.8.0"))
    assert old.inspect_operations.merged_catalog is False and old.inspect_operations.diff_media is True
    assert InspectOperationSwitches(diff_media=True, merged_catalog=True).operations() == ()


def test_merged_operation_set_follows_available_models():
    ops, detail = R.merged_catalog_operations(("get_frames", "get_pages", "crop", "ocr", "align_media", "asr", "locate_frames"), {})
    assert ops == ("get_frames", "ocr", "asr", "locate", "video_structure", "compare_media")
    assert set(ops) == MERGED_OPERATIONS and detail["merged_catalog"]["locate_media"] == ["video"]
    ops, detail = R.merged_catalog_operations(("get_frames", "get_pages", "crop", "ocr", "align_media"), {})
    assert ops == ("get_frames", "ocr", "video_structure", "compare_media")
    assert detail["merged_catalog"]["not_enabled"] == ["asr", "locate"]
    harness = SimpleNamespace(mode="apply", inspect_operations=InspectOperationSwitches(merged_catalog=True))
    assert R.with_spec_operations(("get_frames", "ocr", "asr"), {}, harness)[0] == ("get_frames", "ocr", "asr", "video_structure", "compare_media")
    record = SimpleNamespace(mode="record", inspect_operations=InspectOperationSwitches(merged_catalog=True))
    assert R.with_spec_operations(("get_frames",), {}, record)[0] == ("get_frames",)


def test_workspace_catalog_has_the_six_operations(tmp_path):
    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask
    from foren_searcher.inspect.factory import create_inspect_service
    from foren_searcher.store import GlobalStore

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    service = create_inspect_service(store=store, enabled_operations=MERGED_OPERATIONS)
    try:
        harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=store, inspect_service=service)
        inspect = next(a for a in harness.eligible_actions() if a["name"] == "INSPECT")
        ops = [b["properties"]["request"]["properties"]["operation"]["const"] for b in inspect["schema"]["oneOf"]]
        assert sorted(ops) == sorted(MERGED_OPERATIONS)
    finally:
        asyncio.run(service.aclose())


def test_detect_on_is_a_deprecated_alias_and_image_search_is_passed_to_workers():
    args = SimpleNamespace(teacher="endpoint", runs_dir="runs", round="r", specs="specs", hi_spec="hi@0.7.0", gia_spec="gia@0.8.0",
                           blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s", tool_cache="off",
                           tool_cache_root="t", harness_params="apply", inspect_ops="auto", detect="off",
                           detect_config="x", image_search="off", no_think=True)
    item = SimpleNamespace(case_file="c.json", run_name="n", sample=1, case_id="evid:1")
    command = R._worker_command(args, item, None, None)
    assert command[command.index("--image-search") + 1] == "off" and "--detect" not in command


def test_run_one_wires_merged_catalog_matcher_describer_and_image_search(tmp_path, monkeypatch):
    import json
    import shutil
    from dataclasses import dataclass

    import foren_searcher.gia as gia_package
    from foren_searcher.policies import active_guards

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
            await captured["search_service"].aclose()

        async def run_new(self, **kwargs):
            captured["merged_guard"] = active_guards().merged_catalog
            return FakeResult({"verdict": "x"})

    class FakeTeacher:
        generation_observer = None
        model_config = {"model_provider": "openai_compatible", "endpoint": "http://t/v1", "model": "m"}

        def __init__(self) -> None:
            from foren_searcher.inspect.policy import InspectPolicy

            self.image_policy = InspectPolicy()

    monkeypatch.setattr(gia_package, "InvestigationRuntime", FakeRuntime)
    monkeypatch.setattr(R, "make_teacher", lambda *a, **k: FakeTeacher())
    for name in ("GOOGLE_VISION_API_KEY", "SERPAPI_API_KEY", "SERPER_API_KEY", "TINEYE_API_KEY", "FOREN_DESCRIBE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    cases = tmp_path / "cases"
    (cases / "media").mkdir(parents=True)
    (cases / "media" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    case = cases / "evid_1.json"
    case.write_text(json.dumps({"dataset": "evid", "split": "test", "task_kind": "provenance", "task_text": "t",
                                "task_text_en": "Find the original", "case_id": "evid:1", "source_id": "1", "family_key": "evid:1",
                                "media": [{"ref": "I0", "path": "media/a.png", "media_type": "image/png"}],
                                "source_policy": {"mode": "unrestricted"}}), encoding="utf-8")
    specs = tmp_path / "specs"
    for name in ("hi@0.7.0", "gia@0.8.0", "hi@0.8.0", "gia@0.9.0"):
        shutil.copytree(SPECS / name, specs / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREN_CONFIG", raising=False)
    common = dict(runs_dir=tmp_path / "runs", round_tag="m", sample=1, teacher="endpoint", endpoint="http://t/v1", model="m",
                  thinking=False, specs_root=specs, search_mode="live", snapshot_root=tmp_path / "snap", tool_cache="off",
                  blocklist="off", video_search="ddgs", inspect_ops="base")
    asyncio.run(R.run_one(case, "evid_1_m_1", hi_spec="hi@0.8.0", gia_spec="gia@0.9.0", **common))
    config = json.loads((tmp_path / "runs" / "evid_1_m_1" / "run_config.json").read_text(encoding="utf-8"))
    assert config["inspect_operations"] == ["get_frames", "ocr", "video_structure", "compare_media"]
    assert config["inspect"]["merged_catalog"] is True and config["inspect"]["describer"]["source"] == "run_model"
    assert {k: config["image_search"][k] for k in ("mode", "engine", "uploads_images")} == {
        "mode": "auto", "engine": "yandex", "uploads_images": True}
    assert config["search_match"]["enabled"] is True and captured["merged_guard"] is True
    assert captured["search_service"].matcher is not None
    # 别名：旧规格 + --detect on = 合并目录
    asyncio.run(R.run_one(case, "evid_1_m_2", hi_spec="hi@0.7.0", gia_spec="gia@0.8.0", detect="on", **common))
    alias = json.loads((tmp_path / "runs" / "evid_1_m_2" / "run_config.json").read_text(encoding="utf-8"))
    assert alias["inspect"]["merged_catalog"] is True and alias["detect"]["alias_of"] == "inspect_merged_catalog"
    assert alias["harness_params_applied"]["inspect_merged_catalog"]["source"] == "override"
