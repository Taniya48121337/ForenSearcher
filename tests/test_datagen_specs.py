"""datagen 版本化规格：RubricItem、Spec 落盘/指纹、升号规则、初始版本生成、与运行时提示不漂移。"""

import json
from pathlib import Path

import pytest

from foren_searcher.datagen import specs as specs_mod
from foren_searcher.datagen.errors import ErrorType, Role, Severity
from foren_searcher.datagen.specs import RubricItem, Spec, SpecError, diff, latest, list_versions, validate_bump

SFT_DATA = Path(__file__).resolve().parents[1] / "experiments" / "sft_data"


def item(**overrides) -> RubricItem:
    base = dict(id="hi.no_wrong_citation", error_type=ErrorType.WRONG_CITATION, axis="absence",
                text="不得把讲别的事的 E 当依据。", role=Role.HI)
    base.update(overrides)
    return RubricItem(**base)


def make_spec(**overrides) -> Spec:
    base = dict(spec_id="hi", version="0.1.0", parent=None, student_prompt="你是工作区调查员。",
                teacher_instruction="老师傅注意：先看材料再搜索。", rubric=(item(),),
                changelog=({"targets": ["wrong_citation"], "rationale": "初始"},))
    base.update(overrides)
    return Spec(**base)


# ------------------------------------------------------------------ RubricItem

def test_rubric_item_defaults_and_round_trip():
    it = item()
    assert it.effective_severity is Severity.CRITICAL and it.exec_checkable is False
    assert RubricItem.from_dict(it.to_dict()) == it
    assert item(severity=Severity.MINOR).effective_severity is Severity.MINOR


@pytest.mark.parametrize("overrides, message", [
    ({"id": "Bad ID"}, "小写"),
    ({"axis": "maybe"}, "axis"),
    ({"text": " "}, "不能为空"),
    ({"error_type": ErrorType.PREMATURE_FINISH, "role": Role.HI}, "不适用于角色"),
])
def test_rubric_item_validation(overrides, message):
    with pytest.raises(SpecError, match=message):
        item(**overrides)


# ------------------------------------------------------------------ Spec

def test_spec_validation():
    with pytest.raises(SpecError, match="spec_id"):
        make_spec(spec_id="teacher")
    with pytest.raises(SpecError, match="x.y.z"):
        make_spec(version="1.0")
    with pytest.raises(SpecError, match="早于"):
        make_spec(version="0.1.0", parent="0.2.0")
    with pytest.raises(SpecError, match="未知参数"):
        make_spec(harness_params={"nope": 1})
    with pytest.raises(SpecError, match="重复"):
        make_spec(rubric=(item(), item()))
    with pytest.raises(SpecError, match="与规格"):
        make_spec(spec_id="gia", rubric=(item(),))
    with pytest.raises(ValueError):
        make_spec(changelog=({"targets": ["not_an_error"]},))


def test_spec_save_load_and_hash_guard(tmp_path: Path):
    spec = make_spec()
    directory = spec.save(tmp_path)
    assert directory == tmp_path / "hi@0.1.0"
    assert {p.name for p in directory.iterdir()} == {"spec.json", "student_prompt.md", "teacher_instruction.md", "rubric.json"}
    loaded = Spec.load(directory)
    assert loaded == spec and loaded.hash() == spec.hash()
    assert json.loads((directory / "spec.json").read_text(encoding="utf-8"))["hash"] == spec.hash()
    # 原地改已发布版本的文件 → load 报错；关掉校验才能读
    (directory / "student_prompt.md").write_text("偷偷改了", encoding="utf-8")
    with pytest.raises(SpecError, match="hash 不一致"):
        Spec.load(directory)
    assert Spec.load(directory, verify_hash=False).student_prompt == "偷偷改了"


def test_hash_changes_with_content_but_not_notes():
    base = make_spec()
    assert make_spec(notes="随便写").hash() == base.hash()
    assert make_spec(teacher_instruction="改了").hash() != base.hash()
    assert make_spec(harness_params={**specs_mod.DEFAULT_HARNESS_PARAMS, "image_input_limit": 4}).hash() != base.hash()


# ------------------------------------------------------------------ 升号规则

def test_validate_bump_rules():
    parent = make_spec()
    ok_patch = make_spec(version="0.1.1", parent="0.1.0", teacher_instruction="补充：先核对日期。")
    assert validate_bump(parent, ok_patch) == []
    ok_minor = make_spec(version="0.2.0", parent="0.1.0", student_prompt="你是工作区调查员，引用必须带 E 编号。")
    assert validate_bump(parent, ok_minor) == []

    same = make_spec(version="0.1.1", parent="0.1.0")
    assert any("完全相同" in p for p in validate_bump(parent, same))
    student_as_patch = make_spec(version="0.1.1", parent="0.1.0", student_prompt="变了")
    assert any("minor 或 major" in p for p in validate_bump(parent, student_as_patch))
    teacher_as_minor = make_spec(version="0.2.0", parent="0.1.0", teacher_instruction="变了")
    assert any("应升 patch" in p for p in validate_bump(parent, teacher_as_minor))
    no_targets = make_spec(version="0.1.1", parent="0.1.0", teacher_instruction="变了", changelog=({"rationale": "x"},))
    assert any("targets" in p for p in validate_bump(parent, no_targets))
    wrong_parent = make_spec(version="0.1.2", parent="0.1.1", teacher_instruction="变了")
    assert any("child.parent" in p for p in validate_bump(parent, wrong_parent))


def test_diff_lists_rubric_and_param_changes():
    parent = make_spec()
    child = make_spec(version="0.1.1", parent="0.1.0",
                      harness_params={**specs_mod.DEFAULT_HARNESS_PARAMS, "state_renderer": "compact_8k"},
                      rubric=(item(text="措辞改了"), item(id="hi.no_empty_report", error_type=ErrorType.EMPTY_REPORT)))
    d = diff(parent, child)
    assert d["harness_params"] == {"state_renderer": ("legacy", "compact_8k")}
    assert d["rubric_added"] == ["hi.no_empty_report"] and d["rubric_changed"] == ["hi.no_wrong_citation"]
    assert d["student_prompt"] == [] and d["rubric_removed"] == []


def test_list_versions_and_latest(tmp_path: Path):
    assert list_versions(tmp_path, "hi") == [] and latest(tmp_path, "hi") is None
    make_spec().save(tmp_path)
    make_spec(version="0.10.0", parent="0.1.0", student_prompt="v10").save(tmp_path)
    make_spec(version="0.2.0", parent="0.1.0", student_prompt="v2").save(tmp_path)
    assert list_versions(tmp_path, "hi") == ["0.1.0", "0.2.0", "0.10.0"]   # 数字排序，不是字符串排序
    assert latest(tmp_path, "hi").student_prompt == "v10"
    assert latest(tmp_path, "gia") is None


# ------------------------------------------------------------------ bootstrap（导入 local_models，较慢）

@pytest.fixture(scope="module")
def initial_specs():
    from foren_searcher.datagen import bootstrap
    return bootstrap.build_initial_specs(SFT_DATA)


def test_bootstrap_builds_both_roles_with_full_rubric(initial_specs):
    hi, gia = initial_specs
    assert (hi.name, gia.name) == ("hi@0.1.0", "gia@0.1.0")
    for spec in (hi, gia):
        assert spec.rubric_by_axis()["absence"] >= 5 and spec.rubric_by_axis()["presence"] >= 2
        assert all(it.role is spec.role for it in spec.rubric)
        assert spec.changelog and spec.changelog[0]["targets"]
    hi_types = {it.error_type for it in hi.rubric if it.axis == "absence"}
    gia_types = {it.error_type for it in gia.rubric if it.axis == "absence"}
    assert ErrorType.PREMATURE_FINISH in gia_types and ErrorType.PREMATURE_FINISH not in hi_types
    assert ErrorType.EVIDENCE_DROPPED in hi_types and ErrorType.EVIDENCE_DROPPED not in gia_types
    if (SFT_DATA / "actor_generation_r8.txt").is_file():
        assert "待补" not in hi.teacher_instruction and "---- 报告阶段 ----" in hi.teacher_instruction
        assert "待补" not in gia.teacher_instruction
    # 教师指令绝不能出现在学生提示里（以前是拼进去的）
    assert "focus_evidence_ids 使用当前可见" not in hi.student_prompt


def test_bootstrap_student_prompt_matches_runtime_constants(initial_specs):
    """运行时仍从 local_models 读提示；在切换为从 specs 读之前，两边必须一字不差。"""

    from foren_searcher import local_models
    from foren_searcher.datagen.bootstrap import GIA_PROMPT_SEPARATOR
    hi, gia = initial_specs
    assert hi.student_prompt == local_models._HI_MULTIMODAL_SYSTEM_PROMPT
    main, dispatch = gia.student_prompt.split(GIA_PROMPT_SEPARATOR)
    assert main == local_models._GIA_SYSTEM_PROMPT and dispatch == local_models._GIA_DISPATCH_PROMPT


def test_bootstrap_writes_and_refuses_overwrite(tmp_path: Path, initial_specs):
    from foren_searcher.datagen import bootstrap
    written = bootstrap.bootstrap(tmp_path, SFT_DATA)
    assert [p.name for p in written] == ["hi@0.1.0", "gia@0.1.0"]
    hi, _ = initial_specs
    assert Spec.load(tmp_path / "hi@0.1.0") == hi
    with pytest.raises(FileExistsError):
        bootstrap.bootstrap(tmp_path, SFT_DATA)
    bootstrap.bootstrap(tmp_path, SFT_DATA, overwrite=True)
