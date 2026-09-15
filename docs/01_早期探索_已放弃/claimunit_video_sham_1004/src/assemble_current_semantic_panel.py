"""Resolve the current accepted panel from baseline results and replacement audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


DATASET_DIR = {"VideoSham": "videosham", "VCSL": "vcsl", "VidDiffBench": "viddiffbench"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_path(row: dict[str, Any]) -> Path:
    report = Path(str(row["source_report"]))
    if not report.is_absolute():
        report = ROOT / report
    dataset = str(row["dataset"])
    return (
        report.parent
        / "cases"
        / DATASET_DIR[dataset]
        / str(row["case_id"])
        / f"{row['mode']}.json"
    )


def assemble(baseline: Path, replacements: list[Path]) -> dict[str, Any]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _load(baseline).get("episodes") or []:
        if row.get("accepted_for_high_quality_sft") is not True:
            continue
        identity = (str(row["dataset"]), str(row["case_id"]), str(row["mode"]))
        selected[identity] = row
    for replacement in replacements:
        for row in _load(replacement).get("episodes") or []:
            if row.get("accepted_for_high_quality_sft") is not True:
                continue
            identity = (str(row["dataset"]), str(row["case_id"]), str(row["mode"]))
            selected[identity] = row

    episodes = []
    missing = []
    for identity, row in sorted(selected.items()):
        path = _episode_path(row)
        item = {
            "dataset": identity[0],
            "case_id": identity[1],
            "mode": identity[2],
            "episode_path": str(path),
            "source_report": row["source_report"],
        }
        episodes.append(item)
        if not path.exists():
            missing.append(item)
    return {
        "artifact": "current_semantic_panel_manifest_v1",
        "episodes": len(episodes),
        "missing_episode_files": missing,
        "all_episode_files_resolved": not missing,
        "items": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assemble(args.baseline, args.replacement)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_episode_files_resolved"] else 1)


if __name__ == "__main__":
    main()
