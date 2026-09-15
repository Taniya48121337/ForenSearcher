"""Cross-dataset scale calibration of UTQR-5 with minimal-edit negatives."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from calibrate_unified_rubric import (
    _decision_overrepair,
    _fabricated_support,
    _generic_reasoning,
    _ineligible_target_action,
    _noncausal_step,
    _nonvisible_evidence_reasoning,
    _off_target_change,
    _oracle_leak,
    _private_locator_in_target,
    _premature_terminal,
    _rejected_training_target,
    _subgold_localization,
    _wrong_outcome,
)
from pipeline import ROOT, write_json
from production_sft_compiler import DEFAULT_DATASET_ROOTS, _gold
from unified_trajectory_rubric import evaluate_bundle, load_saved_bundle


DEFAULT_OUTPUT = ROOT / "outputs" / "rubric_v4_scale" / "hard_negative_scale_calibration.json"


def _set_arguments(function: dict[str, Any], arguments: dict[str, Any], was_string: bool) -> None:
    function["arguments"] = (
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":")) if was_string else arguments
    )


def _target_schema_extra_argument(bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    del validation
    datum = (bundle.get("clean_turn_sft") or bundle.get("repair_turn_sft"))[0]
    function = datum["messages"][-1]["tool_calls"][0]["function"]
    raw = function.get("arguments", {})
    arguments = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
    arguments["unexpected_internal_only"] = True
    _set_arguments(function, arguments, isinstance(raw, str))


def _missing_eligible_tool_schema(bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    del validation
    datum = (bundle.get("clean_turn_sft") or bundle.get("repair_turn_sft"))[0]
    target_name = datum["messages"][-1]["tool_calls"][0]["function"]["name"]
    initial = datum["messages"][1]
    payload = json.loads(initial["content"])
    payload["tool_registry"] = [
        item for item in payload.get("tool_registry", []) if item.get("name") != target_name
    ]
    initial["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _history_target_envelope_collision(bundle: dict[str, Any], validation: dict[str, Any]) -> None:
    del validation
    datums = bundle.get("clean_turn_sft", []) + bundle.get("repair_turn_sft", [])
    for datum in datums:
        prior = [message for message in datum["messages"][:-1] if message.get("role") == "assistant"]
        if not prior:
            continue
        function = prior[-1]["tool_calls"][0]["function"]
        raw = function.get("arguments", {})
        arguments = json.loads(raw) if isinstance(raw, str) else copy.deepcopy(raw)
        arguments["call_id"] = "history-envelope-collision"
        _set_arguments(function, arguments, isinstance(raw, str))
        return
    raise ValueError("no datum with prior assistant history")
COMMON_MUTATIONS = {
    "wrong_outcome": _wrong_outcome,
    "oracle_leak": _oracle_leak,
    "fabricated_support": _fabricated_support,
    "premature_terminal": _premature_terminal,
    "noncausal_redundant_step": _noncausal_step,
    "generic_copy_pasted_reasoning": _generic_reasoning,
    "off_target_repair_change": _off_target_change,
    "rejected_training_target": _rejected_training_target,
    "private_locator_in_target": _private_locator_in_target,
    "ineligible_target_action": _ineligible_target_action,
    "nonvisible_evidence_reasoning": _nonvisible_evidence_reasoning,
    "target_arguments_violate_visible_schema": _target_schema_extra_argument,
    "missing_eligible_tool_schema": _missing_eligible_tool_schema,
    "history_target_envelope_collision": _history_target_envelope_collision,
}


def _artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(dataset_roots: dict[str, Path], output: Path | None = DEFAULT_OUTPUT) -> dict[str, Any]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for dataset, run_root in dataset_roots.items():
        report = json.loads((Path(run_root) / "batch_report.json").read_text(encoding="utf-8"))
        for case in report.get("cases", []):
            if not _gold(case):
                continue
            case_id = str(case["case_id"])
            bundle, validation = load_saved_bundle(_artifact_path(str(case["source_artifact_dir"])))
            positive = evaluate_bundle(copy.deepcopy(bundle), copy.deepcopy(validation), dataset=dataset)
            positives.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "accepted": positive["accepted_for_sft"],
                    "score": positive["total_score"],
                }
            )
            mutations = dict(COMMON_MUTATIONS)
            if case.get("repair_operator") == "decision_reopen":
                mutations["decision_reopen_overrepair"] = _decision_overrepair
            if dataset == "VCSL":
                mutations["vcsl_subgold_localization"] = _subgold_localization
            for family, mutation in mutations.items():
                mutated_bundle = copy.deepcopy(bundle)
                mutated_validation = copy.deepcopy(validation)
                mutation(mutated_bundle, mutated_validation)
                result = evaluate_bundle(mutated_bundle, mutated_validation, dataset=dataset)
                negatives.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "family": family,
                        "detected": not result["accepted_for_sft"],
                        "score": result["total_score"],
                        "failed_gates": result["diagnosis"]["failed_gates"],
                        "failed_criteria": result["diagnosis"]["failed_criteria"],
                    }
                )

    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dataset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in negatives:
        family_rows[row["family"]].append(row)
        dataset_rows[row["dataset"]].append(row)
    per_family = {
        family: {
            "count": len(rows),
            "detected": sum(row["detected"] for row in rows),
            "detection_rate": round(sum(row["detected"] for row in rows) / len(rows), 4),
        }
        for family, rows in sorted(family_rows.items())
    }
    per_dataset = {
        dataset: {
            "count": len(rows),
            "detected": sum(row["detected"] for row in rows),
            "detection_rate": round(sum(row["detected"] for row in rows) / len(rows), 4),
        }
        for dataset, rows in sorted(dataset_rows.items())
    }
    detected = sum(row["detected"] for row in negatives)
    report = {
        "rubric_id": "UTQR-5",
        "method": "cross_dataset_minimal_edit_hard_negative_scale_calibration",
        "source_gold_cases": len(positives),
        "positive_controls_passed": all(row["accepted"] for row in positives),
        "summary": {
            "negative_count": len(negatives),
            "detected": detected,
            "detection_rate": round(detected / len(negatives), 4) if negatives else 0.0,
            "families": len(per_family),
            "datasets": len(per_dataset),
            "calibration_passed": (
                bool(negatives)
                and all(row["accepted"] for row in positives)
                and detected / len(negatives) >= 0.90
                and min(item["detection_rate"] for item in per_family.values()) >= 0.80
                and min(item["detection_rate"] for item in per_dataset.values()) >= 0.90
            ),
        },
        "per_family": per_family,
        "per_dataset": per_dataset,
        "positive_controls": positives,
        "hard_negatives": negatives,
        "failure_counts": dict(Counter(row["family"] for row in negatives if not row["detected"])),
    }
    if output is not None:
        write_json(Path(output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(DEFAULT_DATASET_ROOTS, args.output)
    print(json.dumps({"summary": report["summary"], "failure_counts": report["failure_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
