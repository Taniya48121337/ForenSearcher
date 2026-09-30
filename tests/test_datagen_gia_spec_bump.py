"""gia@0.2.0 规格升版：主提示收编运行时附注，与 run 里 GIA 决策的 system 逐字一致。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foren_searcher.datagen.export import GIA_VARIANT_MARKER, spec_system_prompt
from foren_searcher.datagen.specs import Spec, validate_bump

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r1"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file(), reason="试点 run 未随仓库入库")


def test_gia_020_loads_and_hash_matches():
    spec = Spec.load(SPECS / "gia@0.2.0")
    meta = json.loads((SPECS / "gia@0.2.0" / "spec.json").read_text(encoding="utf-8"))
    assert spec.name == "gia@0.2.0" and spec.parent == "0.1.0"
    assert spec.hash() == meta["hash"]
    assert spec.verifier == "verifier@0.3.0"
    assert GIA_VARIANT_MARKER in spec.student_prompt


def test_gia_020_bump_is_valid_and_only_student_prompt_changed():
    parent, child = Spec.load(SPECS / "gia@0.1.0"), Spec.load(SPECS / "gia@0.2.0")
    assert validate_bump(parent, child) == []
    assert child.teacher_instruction == parent.teacher_instruction
    assert child.harness_params == parent.harness_params
    assert [i.to_dict() for i in child.rubric] == [i.to_dict() for i in parent.rubric]
    p_main, p_var = parent.student_prompt.split(GIA_VARIANT_MARKER, 1)
    c_main, c_var = child.student_prompt.split(GIA_VARIANT_MARKER, 1)
    assert c_var == p_var                                   # 首轮变体原样保留
    assert c_main.strip().startswith(p_main.strip())        # 只在主提示后追加运行时附注
    assert "当前或历史轮次存在可汇总的 Finding" in c_main


@needs_run
def test_gia_020_matches_runtime_system_verbatim():
    from foren_searcher.datagen.runlog import load_run

    spec = Spec.load(SPECS / "gia@0.2.0")
    main, variant = (part.strip() for part in spec.student_prompt.split(GIA_VARIANT_MARKER, 1))
    decisions = [d for d in load_run(RUN).decisions if d.role == "GIA"]
    assert decisions
    runtime = {d.system_prompt.strip() for d in decisions}
    assert main in runtime                                  # 主提示与运行时逐字相等
    assert runtime <= {main, variant}
    assert all(spec_system_prompt(spec, d)[1] for d in decisions)
