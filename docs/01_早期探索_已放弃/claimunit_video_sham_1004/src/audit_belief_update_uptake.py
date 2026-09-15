"""Audit observation-to-language uptake in existing accepted trajectories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from causal_trajectory_generation import _latest_harness_state
from pipeline import ROOT, write_json
from production_sft_compiler import DEFAULT_DATASET_ROOTS, _action_group
from step_utility_auditor import _external_observation_uptake


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _preaction_state(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    return _latest_harness_state(messages[:-1])


def audit(dataset_roots: dict[str, Path] | None = None) -> dict[str, Any]:
    dataset_roots = dataset_roots or DEFAULT_DATASET_ROOTS
    records: list[dict[str, Any]] = []
    for dataset, root in dataset_roots.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for kind in ("clean", "repair"):
            for row in _rows(Path(root) / f"sft_{kind}_accepted.jsonl"):
                metadata = row.get("metadata") or {}
                grouped[(str(metadata.get("trajectory_id")), kind)].append(row)
        for (trajectory_id, kind), trajectory_rows in grouped.items():
            trajectory_rows.sort(key=lambda row: int((row.get("metadata") or {}).get("turn_index", 0)))
            for index, row in enumerate(trajectory_rows[:-1]):
                metadata = row.get("metadata") or {}
                tool_name = str(metadata.get("target_tool") or "")
                if _action_group(tool_name) != "external_measurement":
                    continue
                before = _preaction_state(row)
                after = _preaction_state(trajectory_rows[index + 1])
                before_ids = set((before.get("evidence_store") or {}).keys())
                after_store = after.get("evidence_store") or {}
                created_ids = sorted(set(after_store).difference(before_ids))
                atoms = [after_store[evidence_id] for evidence_id in created_ids]
                next_row = trajectory_rows[index + 1]
                next_metadata = next_row.get("metadata") or {}
                next_reasoning = str((next_row.get("messages") or [{}])[-1].get("content") or "")
                uptake = _external_observation_uptake(
                    {
                        "observation": {
                            "created_evidence_ids": created_ids,
                            "evidence_atoms": atoms,
                            "observation": {},
                        }
                    },
                    {
                        "tool_name": str(next_metadata.get("target_tool") or ""),
                        "reasoning": next_reasoning,
                    },
                    set(),
                )
                records.append(
                    {
                        "dataset": dataset,
                        "trajectory_id": trajectory_id,
                        "trajectory_kind": kind,
                        "measurement_turn": metadata.get("turn_index"),
                        "measurement_tool": tool_name,
                        "next_turn": next_metadata.get("turn_index"),
                        "next_tool": next_metadata.get("target_tool"),
                        **uptake,
                    }
                )
    by_dataset = {}
    for dataset in dataset_roots:
        items = [record for record in records if record["dataset"] == dataset]
        by_dataset[dataset] = {
            "expected_updates": sum(record["expected"] for record in items),
            "passed_updates": sum(record["expected"] and record["passed"] for record in items),
            "pass_rate": round(
                sum(record["expected"] and record["passed"] for record in items)
                / max(sum(record["expected"] for record in items), 1),
                6,
            ),
            "failure_counts": dict(
                Counter(
                    failure
                    for record in items
                    for failure in record.get("failures", [])
                )
            ),
        }
    expected = sum(record["expected"] for record in records)
    passed = sum(record["expected"] and record["passed"] for record in records)
    return {
        "audit_id": "OBSERVATION-BELIEF-UPTAKE-V1",
        "scope": "existing local accepted structural candidate trajectories",
        "training_qualification_changed": False,
        "overall": {
            "expected_updates": expected,
            "passed_updates": passed,
            "pass_rate": round(passed / max(expected, 1), 6),
        },
        "by_dataset": by_dataset,
        "records": records,
    }


def main() -> None:
    report = audit()
    destination = (
        ROOT
        / "outputs"
        / "production_sft_v10_learning_value_candidate"
        / "belief_update_uptake_audit.json"
    )
    write_json(destination, report)
    print(json.dumps({"overall": report["overall"], "by_dataset": report["by_dataset"]}, indent=2))


if __name__ == "__main__":
    main()
