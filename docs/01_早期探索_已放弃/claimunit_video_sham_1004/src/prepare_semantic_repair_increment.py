"""Compile leakage-safe accepted semantic repairs for a small adapter continuation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl
from validate_student_batch_sft import validate_rows


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/semantic_repair_increment_v1"
    / "decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_TRAIN = [
    ROOT
    / "outputs/student_external_contract_v1/student_vcsl_semantic_repair_testcase_v1"
    / "accepted_student_sft.jsonl",
    ROOT
    / "outputs/student_external_contract_v1/student_viddiff_semantic_repair_surgery27_v2"
    / "accepted_student_sft.jsonl",
    ROOT
    / "outputs/student_external_contract_v1/student_viddiff_semantic_repair_strict9_v1"
    / "accepted_student_sft.jsonl",
]
DEFAULT_VALIDATION = [
    ROOT
    / "outputs/student_external_contract_v1/student_vcsl_semantic_repair_spd_v1"
    / "accepted_student_sft.jsonl",
    ROOT
    / "outputs/student_external_contract_v1/student_viddiff_semantic_repair_strict3_sample20260823_v2"
    / "accepted_student_sft.jsonl",
]
DEFAULT_PROTECTED_CASES = ["viddiff-surgery_26"]


def _case_id(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("case_id") or "")


def _load_split(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [row for path in paths for row in _jsonl(path)]
    row_ids = [str((row.get("metadata") or {}).get("row_id") or "") for row in rows]
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate SFT row IDs in semantic increment: {duplicates[:5]}")
    validation = validate_rows(rows)
    if validation.get("all_passed") is not True:
        raise ValueError(
            f"semantic increment contains invalid SFT rows: {validation.get('failed_rows')}"
        )
    return rows


def build(
    train_paths: list[Path],
    validation_paths: list[Path],
    protected_cases: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    train = _load_split(train_paths)
    validation = _load_split(validation_paths)
    train_cases = {_case_id(row) for row in train}
    validation_cases = {_case_id(row) for row in validation}
    protected = set(protected_cases)
    overlap = sorted(train_cases & validation_cases)
    protected_overlap = sorted((train_cases | validation_cases) & protected)
    if overlap:
        raise ValueError(f"train/validation case overlap: {overlap}")
    if protected_overlap:
        raise ValueError(f"protected evaluation case leaked into increment: {protected_overlap}")

    splits = {"train": train, "validation": validation, "test": []}
    manifest = {
        "artifact": "semantic_repair_increment_v1",
        "assistant_target_only": True,
        "teacher_actions_used_during_generation": False,
        "oracle_or_verifier_used_during_generation": False,
        "train_sources": [str(path) for path in train_paths],
        "validation_sources": [str(path) for path in validation_paths],
        "protected_evaluation_cases": sorted(protected),
        "case_overlap": overlap,
        "protected_case_overlap": protected_overlap,
        "splits": {
            name: {
                "rows": len(rows),
                "cases": sorted({_case_id(row) for row in rows}),
                "datasets": dict(
                    Counter(str(row["metadata"]["dataset"]) for row in rows)
                ),
                "target_tools": dict(
                    Counter(str(row["metadata"]["target_tool"]) for row in rows)
                ),
            }
            for name, rows in splits.items()
        },
        "selection_intent": (
            "Train on deterministic accepted repairs; retain the official-SPD route and the "
            "three hardest sampled VidDiff repairs as disjoint generalization checks."
        ),
    }
    return splits, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, action="append")
    parser.add_argument("--validation", type=Path, action="append")
    parser.add_argument("--protected-case", action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train_paths = list(args.train or DEFAULT_TRAIN)
    validation_paths = list(args.validation or DEFAULT_VALIDATION)
    protected_cases = list(args.protected_case or DEFAULT_PROTECTED_CASES)
    splits, manifest = build(train_paths, validation_paths, protected_cases)
    for split, rows in splits.items():
        write_jsonl(args.output / split / "sft_all.jsonl", rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
