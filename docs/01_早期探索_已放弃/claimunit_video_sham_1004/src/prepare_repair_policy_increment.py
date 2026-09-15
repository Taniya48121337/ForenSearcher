"""Compile case-disjoint, executable repair-policy SFT for a small continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import ROOT, write_json, write_jsonl
from prepare_semantic_repair_increment import build


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/repair_policy_increment_v1"
    / "decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_TRAIN = [
    ROOT
    / "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot108_v3"
    / "accepted_student_sft.jsonl",
    ROOT
    / "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot117_v2"
    / "accepted_student_sft.jsonl",
    ROOT
    / "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pair16_84_v1"
    / "accepted_student_sft.jsonl",
]
DEFAULT_VALIDATION = [
    ROOT
    / "outputs/student_external_contract_v1/student_repair_passk_seed404_v1"
    / "accepted_student_sft.jsonl"
]
DEFAULT_PROTECTED = ["viddiff-surgery_16", "viddiff-surgery_26"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, action="append")
    parser.add_argument("--validation", type=Path, action="append")
    parser.add_argument("--protected-case", action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train_paths = list(args.train or DEFAULT_TRAIN)
    validation_paths = list(args.validation or DEFAULT_VALIDATION)
    protected = list(args.protected_case or DEFAULT_PROTECTED)
    splits, manifest = build(train_paths, validation_paths, protected)
    manifest.update(
        {
            "artifact": "repair_policy_increment_v1",
            "training_stratum": "true_executable_harness_repair",
            "counterfactual_evidence_dropout_train_cases": [
                "viddiff-surgery_108",
                "viddiff-surgery_117",
                "viddiff-surgery_84",
            ],
            "selection_intent": (
                "Train only on independently accepted executable repair trajectories; "
                "retain cross-dataset repair cases for validation and keep VidDiff "
                "surgery_16 and surgery_26 fully outside the adapter update."
            ),
        }
    )
    for split, rows in splits.items():
        write_jsonl(args.output / split / "sft_all.jsonl", rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
