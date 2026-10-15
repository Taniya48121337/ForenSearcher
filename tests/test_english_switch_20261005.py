"""10-05 晚：模型读到的框架文字全英文（规格 harness_params.language="en"，hi@0.7.0 / gia@0.8.0 起）。

开关是 RuntimeGuards.english（merge_input_features 把 language="en" 映射过去）；没打开时一切保持中文、逐字不变，
旧规格与 r0e / r0f 可复现。外部取得的内容（网页、标题、字幕、转写）不翻译，这里不测。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from foren_searcher.datagen.harness_params import InputFeatures, build_harness_settings
from foren_searcher.datagen.specs import Spec, validate_bump
from foren_searcher.i18n import english, tr
from foren_searcher.inspect.backends.base import MediaProfileData
from foren_searcher.inspect.contracts import inspect_action_schema
from foren_searcher.inspect.service import InspectService
from foren_searcher.policies import EVAL_GUARDS, RuntimeGuards, merge_input_features, runtime_guard_scope
from foren_searcher.state_render import TIERS, card_line, cut_list, cut_text

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
CJK = re.compile(r"[　-鿿＀-￯]")
EN = merge_input_features(EVAL_GUARDS, {"language": "en"})


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_language_feature_maps_to_english_guard_and_zh_is_not_a_feature():
    assert EN.english is True
    assert merge_input_features(EVAL_GUARDS, {"language": "zh"}).english is False
    assert RuntimeGuards().english is False
    assert InputFeatures().enabled() == {} and "language" not in InputFeatures(language="zh").enabled()
    assert InputFeatures(language="en").enabled()["language"] == "en"
    assert english() is False and tr("中", "en") == "中"
    with runtime_guard_scope(EN):
        assert english() is True and tr("中", "en") == "en"


def test_new_specs_turn_english_on_and_only_add_language():
    config_specs = {}
    for old, new in (("hi@0.6.0", "hi@0.7.0"), ("gia@0.7.0", "gia@0.8.0")):
        parent, child = Spec.load(SPECS / old), Spec.load(SPECS / new)
        assert validate_bump(parent, child) == []
        assert child.harness_params == {**parent.harness_params, "language": "en"}
        assert child.rubric == parent.rubric
        config_specs[new] = child
    from foren_searcher.config import load_config

    settings = build_harness_settings(load_config(), config_specs["hi@0.7.0"], config_specs["gia@0.8.0"])
    assert settings.input_features.language == "en" and settings.input_features.enabled()["language"] == "en"


def test_tool_schema_descriptions_follow_the_switch():
    zh = list(_strings(inspect_action_schema()))
    assert any(CJK.search(text) for text in zh)
    with runtime_guard_scope(EN):
        en = list(_strings(inspect_action_schema()))
    assert not [text for text in en if CJK.search(text)]
    assert inspect_action_schema() == json.loads(json.dumps(inspect_action_schema()))   # 中文那份不受影响


def test_labels_marks_and_notices_in_english():
    profile = MediaProfileData(media_kind="video", width=1280, height=720, duration_s=477.233, fps=30.0, audio_tracks=1)
    zh_label = InspectService._profile_label(profile)
    assert zh_label.startswith("媒体属性：video，477.233秒")
    with runtime_guard_scope(EN):
        label = InspectService._profile_label(profile)
        assert label.startswith("media profile: video, 477.233 s, 1280×720, 30.000 fps, 1 audio track;")
        assert not CJK.search(label)
        tier = TIERS["compact_16k"]
        line = card_line({"evidence_id": "E2", "modalities": ["text"], "label": label, "derived_from": ["E1"]}, tier)
        assert line.startswith("E2 [text] media profile: video, 477.233 s") and "verify" not in line   # 固定尾巴照样剥掉
        assert cut_text("x" * 50, 10).endswith("[truncated 40 chars; READ_RECORD reads the full text]")
        assert cut_list(list(range(5)), 2)[-1] == "…(3 more not shown)"
        from foren_searcher.store.records import _media_notice

        assert "not pixels" in _media_notice(True) and not CJK.search(_media_notice(False))
    assert "已截断" in cut_text("x" * 50, 10)
    assert "像素" in __import__("foren_searcher.store.records", fromlist=["_media_notice"])._media_notice(True)


def test_material_labels_follow_spec_language(tmp_path: Path):
    from foren_searcher.datagen.cases import Case
    from foren_searcher.datagen.run_cases import media_inputs

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "a.mp4").write_bytes(b"x")
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"case_id": "evid:video_001", "dataset": "evid", "source_id": "video_001", "split": "test",
                                "task_kind": "video_diff", "task_text": "Explain", "family_key": "evid:video_001",
                                "media": [{"ref": "I0", "path": "media/a.mp4", "media_type": "video/mp4"}],
                                "source_policy": {"mode": "provided_media"}}), encoding="utf-8")
    assert media_inputs(Case.load(path), path)[0].label == "材料 A"
    assert media_inputs(Case.load(path), path, "en")[0].label == "Material A"
    from foren_searcher.datagen.verify.context import _MATERIAL_LABEL

    assert _MATERIAL_LABEL.match("Material A").group(1) == "A" and _MATERIAL_LABEL.match("材料 B").group(1) == "B"


def test_behavior_panel_reads_english_labels():
    sys.path.insert(0, str(ROOT / "scripts"))
    from _evid_behavior import _FRAME_LABEL, _PROFILE_LABEL

    match = _FRAME_LABEL.search("frame of video E1 at 14.133 s")
    assert (match.group(1) or match.group(3), match.group(2) or match.group(4)) == ("E1", "14.133")
    match = _PROFILE_LABEL.search("media profile: video, 477.233 s, 1280×720")
    assert (match.group(1) or match.group(2)) == "477.233"
    match = _FRAME_LABEL.search("视频E1在14.133秒的帧")
    assert (match.group(1), match.group(2)) == ("E1", "14.133")


@pytest.mark.parametrize("role", ["GIA", "HI"])
def test_repair_and_harness_notes_are_english(role: str):
    from foren_searcher.datagen.prompt_variants import repair_suffix, runtime_gia_variants

    assert not CJK.search(repair_suffix(role, "en"))
    assert CJK.search(repair_suffix(role, "zh"))
    if role == "GIA":
        assert not [name for name, text in runtime_gia_variants("en").items() if CJK.search(text)]
