from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import prepare_semantic_repair_increment as module


def _row(case_id: str, dataset: str = "VidDiffBench") -> dict:
    return {
        "metadata": {
            "case_id": case_id,
            "dataset": dataset,
            "target_tool": "bind_evidence",
            "row_id": f"{dataset}:{case_id}:1",
        }
    }


def test_build_keeps_train_validation_and_protected_cases_disjoint(monkeypatch) -> None:
    train_path = Path("train.jsonl")
    validation_path = Path("validation.jsonl")
    rows = {
        train_path: [_row("viddiff-train")],
        validation_path: [_row("viddiff-validation")],
    }
    monkeypatch.setattr(
        module, "_load_split", lambda paths: [row for path in paths for row in rows[path]]
    )
    splits, manifest = module.build(
        [train_path], [validation_path], ["viddiff-protected"]
    )
    assert len(splits["train"]) == 1
    assert len(splits["validation"]) == 1
    assert manifest["case_overlap"] == []
    assert manifest["protected_case_overlap"] == []
    assert manifest["protected_evaluation_cases"] == ["viddiff-protected"]


def test_build_rejects_protected_case_leak(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "_load_split",
        lambda paths: [_row("viddiff-surgery_26")]
        if paths[0].name == "train.jsonl"
        else [_row("viddiff-validation")],
    )
    with pytest.raises(ValueError, match="protected evaluation case leaked"):
        module.build(
            [Path("train.jsonl")],
            [Path("validation.jsonl")],
            ["viddiff-surgery_26"],
        )


def test_build_rejects_train_validation_case_overlap(monkeypatch) -> None:
    monkeypatch.setattr(module, "_load_split", lambda paths: [_row("shared-case")])
    with pytest.raises(ValueError, match="train/validation case overlap"):
        module.build(
            [Path("train.jsonl")],
            [Path("validation.jsonl")],
            ["viddiff-protected"],
        )
