"""补全核验脚本的产物命名：不同 run 集合、不同核验模型互不覆盖；默认 12 个 run 保持原名。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("complete_verify_local", ROOT / "scripts" / "complete_verify_local.py")
cvl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvl)


def test_default_set_keeps_short_names() -> None:
    paths = cvl.output_paths("local-m", [str(cvl.RUNS_DIR / n) for n in reversed(cvl.DEFAULT_RUNS)])
    assert Path(paths["verdicts"]).name == "verdicts_r0_judged.local-m.jsonl"
    assert Path(paths["out"]).name.startswith("r0_bridge12_")


def test_other_sets_do_not_collide() -> None:
    one = cvl.output_paths("local-m", ["runs/a_1"])
    other = cvl.output_paths("local-m", ["runs/b_1"])
    pair = cvl.output_paths("local-m", ["runs/a_1", "runs/b_1"])
    assert Path(one["verdicts"]).name == "verdicts_r0_judged.local-m.a_1.jsonl"
    assert len({one["out"], other["out"], pair["out"]}) == 3
    assert len({one["verdicts"], other["verdicts"], pair["verdicts"]}) == 3
    assert cvl.run_set_id(["x/b_1", "y/a_1"]) == cvl.run_set_id(["a_1", "b_1"])      # 与顺序、目录无关
