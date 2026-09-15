from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_case_disjoint_sft_gain_experiment import ARM, prepare  # noqa: E402


def _write(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [],
                    "metadata": {
                        "case_id": case_id,
                        "dataset": "D",
                        "trajectory_mode": "clean",
                        "target_tool": "inspect",
                    },
                }
            )
            + "\n"
            for case_id in case_ids
        ),
        encoding="utf-8",
    )


def test_prepare_preserves_disjoint_evaluation(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    base = tmp_path / "base"
    out = tmp_path / "out"
    _write(train, ["train-1", "train-2"])
    _write(base / ARM / "validation" / "sft_all.jsonl", ["val-1"])
    _write(base / ARM / "test" / "sft_all.jsonl", ["test-1"])
    report = prepare(train, base, out, evaluation_splits=("validation", "test"))
    assert report["case_disjoint"] is True
    assert report["train"]["cases"] == 2
    assert report["evaluation"]["validation"]["overlap_with_train"] == []
    assert (out / ARM / "train" / "sft_all.jsonl").exists()


def test_prepare_rejects_case_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    base = tmp_path / "base"
    _write(train, ["shared"])
    _write(base / ARM / "validation" / "sft_all.jsonl", ["shared"])
    _write(base / ARM / "test" / "sft_all.jsonl", ["test-1"])
    with pytest.raises(ValueError, match="case leakage"):
        prepare(
            train,
            base,
            tmp_path / "out",
            evaluation_splits=("validation", "test"),
        )
