"""案卷五种划分：旧值兼容、降级规则、family 泄漏检查、split_summary、export manifest 里的 splits。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.cases import SPLITS, Case, CaseError, check_split_leakage, load_cases, split_summary

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r1"
VERDICTS = ROOT / "experiments" / "sft_data" / "verdicts_r0_judged.jsonl"
CASES = ROOT / "experiments" / "sft_data" / "cases"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not VERDICTS.is_file(), reason="试点 run / 判定文件未随仓库入库")


def make(source_id: str = "1", split: str = "dev_train", family: str | None = None, **extra: Any) -> Case:
    return Case(case_id=f"true:{source_id}", dataset="true", source_id=source_id, split=split,  # type: ignore[arg-type]
                task_kind="claim_check", task_text="t", family_key=family or f"true:{source_id}", **extra)


# ------------------------------------------------------------------ 旧值兼容

@pytest.mark.parametrize("old, new", [("dev", "dev_train"), ("heldout", "dev_eval")])
def test_legacy_split_values_are_mapped_on_load_and_written_as_new(tmp_path: Path, old: str, new: str):
    raw = make().to_dict()
    raw["split"] = old
    raw.pop("effective_split")
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    case = Case.load(path)
    assert case.split == new and case.to_dict()["split"] == new
    case.save(tmp_path / "new.json")
    written = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    assert written["split"] == new and written["effective_split"] == new
    assert make(split=old) == make(split=new)          # 构造时同样映射


def test_all_new_splits_valid_and_unknown_rejected():
    for split in SPLITS:
        assert make(split=split).split == split
    with pytest.raises(CaseError, match="split"):
        make(split="train")


def test_repo_case_file_uses_new_split():
    src = CASES / "true_4124903.json"
    if not src.is_file():
        pytest.skip("案卷未入库")
    assert json.loads(src.read_text(encoding="utf-8"))["split"] == "dev_train"


# ------------------------------------------------------------------ 降级

@pytest.mark.parametrize("split, expected", [
    ("dev_eval", "dev_train"), ("test", "dev_train"),
    ("dev_train", "dev_train"), ("dev_select", "dev_select"), ("dev_confirm", "dev_confirm"), ("contrast_only", "contrast_only"),
])
def test_only_eval_splits_downgrade_after_revision(split: str, expected: str):
    case = make(split=split)
    assert case.effective_split == split
    used = case.mark_used_for_revision("hi@0.2.1")
    assert used.effective_split == expected and used.split == split
    assert used.to_dict()["split"] == split and used.to_dict()["effective_split"] == expected


# ------------------------------------------------------------------ 泄漏

def test_leakage_cross_group_and_within_group():
    assert check_split_leakage([make("1", "dev_train", "f"), make("2", "dev_eval", "f")])[0].endswith("（跨训练/评测组）")
    assert check_split_leakage([make("1", "dev_select", "f"), make("2", "test", "f")])
    # 更严规则：组内也不许共享 family
    assert check_split_leakage([make("1", "dev_train", "f"), make("2", "dev_select", "f")])[0].endswith("（组内重复）")
    assert check_split_leakage([make("1", "dev_eval", "f"), make("2", "test", "f")])
    # 同一划分内共享 family 没问题；不同 family 没问题
    assert not check_split_leakage([make("1", "dev_eval", "f"), make("2", "dev_eval", "f"), make("3", "test", "g")])


def test_leakage_uses_effective_split():
    # 被用来改过规格的 dev_eval 降为 dev_train，与同 family 的 dev_train 不再冲突，与 test 冲突
    used = make("1", "dev_eval", "f").mark_used_for_revision("hi@0.2.1")
    assert not check_split_leakage([used, make("2", "dev_train", "f")])
    assert check_split_leakage([used, make("3", "test", "f")])


# ------------------------------------------------------------------ split_summary

def test_split_summary_counts_cases_and_families():
    cases = [make("1", "dev_train", "a"), make("2", "dev_train", "a"), make("3", "dev_select", "b"),
             make("4", "test", "c").mark_used_for_revision("x"), make("5", "test", "d")]
    summary = split_summary(cases)
    assert set(summary) == set(SPLITS)
    assert summary["dev_train"] == {"cases": 3, "families": 2}      # 含降级的 test
    assert summary["dev_select"] == {"cases": 1, "families": 1}
    assert summary["test"] == {"cases": 1, "families": 1}
    assert summary["dev_eval"] == {"cases": 0, "families": 0}


# ------------------------------------------------------------------ manifest

@needs_run
def test_manifest_splits_on_true_4124903(tmp_path: Path):
    from foren_searcher.datagen.export import apply_family_caps, export_run, write_outputs
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.specs import Spec
    from foren_searcher.datagen.verify.verdict import read_verdicts

    cases = load_cases(CASES)
    log = load_run(RUN)
    try:
        case = cases.get(log.case_id or "")
        assert case is not None
        reports = [v for v in read_verdicts(VERDICTS) if v.subject.get("run") == log.name]
        specs = {"hi": Spec.load(ROOT / "specs" / "hi@0.1.0"), "gia": Spec.load(ROOT / "specs" / "gia@0.1.0")}
        out = tmp_path / "export"
        rows, verdicts = export_run(log, case, reports, specs, renderer="compact_8k", out_dir=out, image_root=out / "images")
        no_case_rows, _ = export_run(log, None, reports, specs, renderer="compact_8k", out_dir=out, image_root=out / "images")
    finally:
        log.close()
    assert rows and all(r["metadata"]["split"] == "dev_train" for r in rows)
    assert all(r["metadata"]["split"] is None for r in no_case_rows)
    manifest = write_outputs(apply_family_caps(rows), verdicts, out_dir=out, specs=specs, renderer="compact_8k",
                             verdicts_source=str(VERDICTS), cases_dir=str(CASES), cases=cases)
    splits = manifest["splits"]
    assert splits["by_split"]["dev_train"] == {"cases": 1, "families": 1}
    assert all(v == {"cases": 0, "families": 0} for k, v in splits["by_split"].items() if k != "dev_train")
    assert splits["rows_by_split"] == {"dev_train": len(rows)} and splits["no_case_rows"] == 0
    on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["splits"] == splits
    # 不传 cases 时从 cases_dir 读，结果一致
    again = write_outputs(rows, verdicts, out_dir=tmp_path / "again", specs=specs, renderer="compact_8k",
                          verdicts_source=str(VERDICTS), cases_dir=str(CASES))
    assert again["splits"] == splits
