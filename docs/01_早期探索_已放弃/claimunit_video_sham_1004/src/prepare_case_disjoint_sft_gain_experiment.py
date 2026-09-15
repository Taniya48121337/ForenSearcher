"""Prepare an incremental SFT root with case-disjoint evaluation splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ARM = "decision_sufficient_state_v6_executable_affordances"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str((row.get("metadata") or {}).get("case_id") or "") for row in rows}


def prepare(
    train_path: Path,
    base_root: Path,
    output_root: Path,
    *,
    evaluation_splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_path)
    split_rows = {
        split: _read_jsonl(base_root / ARM / split / "sft_all.jsonl")
        for split in evaluation_splits
    }
    train_cases = _case_ids(train_rows)
    overlaps = {
        split: sorted(train_cases & _case_ids(rows))
        for split, rows in split_rows.items()
    }
    if any(overlaps.values()):
        raise ValueError(f"case leakage detected: {overlaps}")
    _write_jsonl(output_root / ARM / "train" / "sft_all.jsonl", train_rows)
    for split, rows in split_rows.items():
        _write_jsonl(output_root / ARM / split / "sft_all.jsonl", rows)

    metadata = [row.get("metadata") or {} for row in train_rows]
    report = {
        "artifact": "case_disjoint_incremental_sft_gain_experiment_v1",
        "arm": ARM,
        "train": {
            "rows": len(train_rows),
            "cases": len(train_cases),
            "datasets": dict(sorted(Counter(str(item.get("dataset")) for item in metadata).items())),
            "modes": dict(
                sorted(Counter(str(item.get("trajectory_mode")) for item in metadata).items())
            ),
            "tools": dict(
                sorted(Counter(str(item.get("target_tool")) for item in metadata).items())
            ),
        },
        "evaluation": {
            split: {
                "rows": len(rows),
                "cases": len(_case_ids(rows)),
                "case_ids": sorted(_case_ids(rows)),
                "overlap_with_train": overlaps[split],
            }
            for split, rows in split_rows.items()
        },
        "case_disjoint": not any(overlaps.values()),
    }
    report_path = output_root / "preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sft", type=Path, required=True)
    parser.add_argument("--base-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--eval-split",
        action="append",
        choices=("validation", "test"),
        default=None,
        help="Evaluation split to preserve; repeat for more than one. Defaults to validation.",
    )
    args = parser.parse_args()
    report = prepare(
        args.train_sft,
        args.base_data_root,
        args.output_root,
        evaluation_splits=tuple(args.eval_split or ["validation"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
