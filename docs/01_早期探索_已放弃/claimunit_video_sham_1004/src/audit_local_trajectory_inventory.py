"""Inventory locally accepted trajectory cases without treating exposure as holdout.

The inventory is deliberately conservative: any case that already appears in an
accepted trajectory artifact is classified as development-exposed.  It can be
reused to enlarge a development SFT batch, but it cannot support a sealed-holdout
claim merely because it was omitted from the latest compiler run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from pipeline import ROOT, write_json


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/production_readiness_audit_v1"
    / "local_trajectory_inventory.json"
)
DEFAULT_RECOVERY_MANIFEST = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_manifest_v1/manifest.json"
)
DEFAULT_CURRENT6_AUDIT = (
    ROOT
    / "outputs/student_external_contract_v1/cross_dataset_blind_canary_6episodes_v3"
    / "semantic_audit.json"
)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dataset_for_case(case_id: str) -> str:
    lowered = case_id.lower()
    if lowered.startswith("videosham-"):
        return "VideoSham"
    if lowered.startswith("vcsl-"):
        return "VCSL"
    if lowered.startswith("viddiff-"):
        return "VidDiffBench"
    return "Other"


def entity_ids(dataset: str, case_id: str) -> tuple[str, ...]:
    if dataset == "VCSL":
        parts = case_id.removeprefix("vcsl-").split("-")
        if len(parts) == 2:
            return tuple(parts)
    return (case_id,)


def _maximum_entity_disjoint(cases: list[dict[str, Any]]) -> list[str]:
    """Return an exact maximum non-overlapping subset for the small local pool."""

    ordered = sorted(cases, key=lambda item: item["case_id"])
    best: list[str] = []

    def visit(index: int, used: set[str], selected: list[str]) -> None:
        nonlocal best
        if len(selected) + len(ordered) - index <= len(best):
            return
        if index == len(ordered):
            if len(selected) > len(best):
                best = selected.copy()
            return
        item = ordered[index]
        entities = set(item["entity_ids"])
        if entities.isdisjoint(used):
            selected.append(item["case_id"])
            visit(index + 1, used | entities, selected)
            selected.pop()
        visit(index + 1, used, selected)

    visit(0, set(), [])
    return best


def build_inventory(
    outputs_root: Path,
    recovery_manifest_path: Path,
    current6_audit_path: Path | None = None,
) -> dict[str, Any]:
    recovery_manifest = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    recovery_cases = {
        str(job["case_id"]).lower() for job in recovery_manifest.get("jobs", [])
    }
    current6_cases: set[str] = set()
    if current6_audit_path and current6_audit_path.exists():
        audit = json.loads(current6_audit_path.read_text(encoding="utf-8"))
        current6_cases = {
            str(item.get("case_id", "")).lower()
            for item in audit.get("episode_reports", [])
            if item.get("case_id")
        }

    by_case: dict[str, dict[str, Any]] = {}
    accepted_files = sorted(outputs_root.rglob("sft_clean_accepted.jsonl"))
    parsed_rows = 0
    for path in accepted_files:
        relative = path.relative_to(ROOT).as_posix()
        for row in _jsonl(path):
            metadata = row.get("metadata") or {}
            case_id = str(metadata.get("case_id") or "").lower()
            if not case_id:
                continue
            parsed_rows += 1
            dataset = dataset_for_case(case_id)
            item = by_case.setdefault(
                case_id,
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "entity_ids": list(entity_ids(dataset, case_id)),
                    "artifact_paths": set(),
                    "trajectory_ids": set(),
                    "accepted_turn_rows": 0,
                    "accepted_terminal_rows": 0,
                },
            )
            item["artifact_paths"].add(relative)
            if metadata.get("trajectory_id"):
                item["trajectory_ids"].add(str(metadata["trajectory_id"]))
            item["accepted_turn_rows"] += 1
            if metadata.get("target_tool") == "stop":
                item["accepted_terminal_rows"] += 1

    cases: list[dict[str, Any]] = []
    for case_id, raw in sorted(by_case.items()):
        tags = ["accepted_local_artifact", "development_exposed"]
        if case_id in recovery_cases:
            tags.append("recovery_v6_development")
        else:
            tags.append("available_for_development_expansion")
        if case_id in current6_cases:
            tags.append("current6_diagnostic_exposed")
        cases.append(
            {
                **raw,
                "artifact_paths": sorted(raw["artifact_paths"]),
                "trajectory_ids": sorted(raw["trajectory_ids"]),
                "exposure_tags": tags,
                "sealed_holdout_eligible": False,
                "sealed_holdout_reason": (
                    "The case already appears in a locally accepted generation or "
                    "evaluation artifact."
                ),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cases:
        grouped[item["dataset"]].append(item)

    required = recovery_manifest.get("datasets") or {}
    dataset_summary: dict[str, Any] = {}
    for dataset in ("VideoSham", "VCSL", "VidDiffBench"):
        pool = grouped.get(dataset, [])
        recovery_count = sum(
            "recovery_v6_development" in item["exposure_tags"] for item in pool
        )
        disjoint = _maximum_entity_disjoint(pool)
        dataset_summary[dataset] = {
            "accepted_exposed_cases": len(pool),
            "maximum_entity_disjoint_exposed_cases": len(disjoint),
            "maximum_entity_disjoint_case_ids": disjoint,
            "recovery_v6_cases": recovery_count,
            "additional_development_cases": len(pool) - recovery_count,
            "sealed_cases": 0,
            "manifest_reported_cases": int(required.get(dataset, 0)),
        }

    return {
        "artifact": "local_trajectory_inventory_v1",
        "scope": "accepted trajectory artifacts already present under outputs/",
        "accepted_artifact_files": len(accepted_files),
        "parsed_accepted_turn_rows": parsed_rows,
        "unique_cases": len(cases),
        "dataset_summary": dataset_summary,
        "classification_rule": (
            "Every accepted case is development-exposed. Omission from recovery v6 "
            "does not restore sealed-holdout status."
        ),
        "sealed_holdout_available": False,
        "sealed_holdout_caveat": (
            "This audit cannot certify unseen raw-dataset cases. A future sealed split "
            "must be declared before its trajectories, labels, or verifier outcomes are inspected."
        ),
        "current6_diagnostic_case_ids": sorted(current6_cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--recovery-manifest", type=Path, default=DEFAULT_RECOVERY_MANIFEST)
    parser.add_argument("--current6-audit", type=Path, default=DEFAULT_CURRENT6_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_inventory(args.outputs_root, args.recovery_manifest, args.current6_audit)
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "unique_cases": report["unique_cases"],
        "dataset_summary": report["dataset_summary"],
        "sealed_holdout_available": report["sealed_holdout_available"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
