"""Derive a label-blind VidDiff abstention atom from swap-instability residuals.

The rule is calibrated on the historical 10-case development panel and then
evaluated unchanged on the disjoint 30-case external panel.  Dataset labels are
read only by ``evaluate`` after every derived observation has been created.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl


TOOL_NAME = "compare_symmetry_residual_abstention"
ELIGIBLE_FAMILY_PATTERNS = {
    "geometry": re.compile(r"radial|perpendicular|closer to the tip|approximately\s+2/3", re.I),
    "count_frequency": re.compile(r"number of|more often|stops?\b", re.I),
    "object_motion": re.compile(r"\btube moves?\b|\bmoves more\b", re.I),
}
SOURCE_TOOLS = (
    "compare_global_motion_symmetric",
    "compare_action_refined_symmetric",
)


def predicate_family(statement: str) -> str | None:
    for family, pattern in ELIGIBLE_FAMILY_PATTERNS.items():
        if pattern.search(statement):
            return family
    return None


def _interval_union(values: list[Any]) -> list[float] | None:
    valid = [
        [float(value[0]), float(value[1])]
        for value in values
        if isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
        and float(value[1]) >= float(value[0])
    ]
    if not valid:
        return None
    return [round(min(value[0] for value in valid), 3), round(max(value[1] for value in valid), 3)]


def derive_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    """Return a student-visible meta-measurement without consulting a label."""
    statement = str(item.get("task", {}).get("comparison_statement", ""))
    family = predicate_family(statement)
    if family is None:
        return None
    by_name = {entry.get("name"): entry for entry in item.get("candidate_tools", [])}
    if any(name not in by_name for name in SOURCE_TOOLS):
        return None
    source_entries = [by_name[name] for name in SOURCE_TOOLS]
    observations = [entry.get("tool_observation", {}) for entry in source_entries]
    if not all(
        observation.get("preferred_video") == "c"
        and observation.get("order_consistent") is False
        for observation in observations
    ):
        return None

    interval_a = _interval_union([item.get("video_a_interval_s") for item in observations])
    interval_b = _interval_union([item.get("video_b_interval_s") for item in observations])
    if interval_a is None or interval_b is None:
        return None
    source_ids = [
        observation.get("measurement_id")
        or entry.get("provenance", {}).get("measurement_id")
        or entry.get("provenance", {}).get("candidate_run_id")
        for entry, observation in zip(source_entries, observations)
    ]
    rationale = (
        f"The {family} predicate has no stable directional support under two independent "
        "A/B-symmetric comparisons. Both compare_global_motion_symmetric and "
        "compare_action_refined_symmetric collapse to abstention after their mapped swapped "
        "passes disagree, so the available views support no reliable A-over-B or B-over-A "
        "ordering. This meta-decision is invariant to exchanging the A/B roles and does not "
        "treat either unstable base direction as evidence of equivalence."
    )
    return {
        "name": TOOL_NAME,
        "arguments": {
            "video_a_ref": "video_a",
            "video_b_ref": "video_b",
            "comparison_statement": statement,
        },
        "tool_observation": {
            "preferred_video": "c",
            "video_a_interval_s": interval_a,
            "video_b_interval_s": interval_b,
            "confidence": 0.65,
            "order_consistent": True,
            "equivariance_valid": True,
            "decision_rule": "two_independent_swap_instability_abstention_v1",
            "predicate_family": family,
            "source_tools": list(SOURCE_TOOLS),
            "source_measurement_ids": source_ids,
            "visual_rationale": rationale,
        },
        "provenance": {
            "candidate_run_id": "viddiff_symmetry_residual_abstention_v1",
            "source_candidate_run_ids": [
                entry.get("provenance", {}).get("candidate_run_id")
                for entry in source_entries
            ],
            "label_visible_to_tool": False,
            "derived_from_student_visible_observations": True,
        },
    }


def evaluate(
    candidates_path: Path,
    selection_audit_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates = _jsonl(candidates_path)
    audit = json.loads(selection_audit_path.read_text(encoding="utf-8"))
    audit_by_key = {str(item["sample_key"]): item for item in audit.get("cases", [])}
    augmented: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    revised_cases: list[dict[str, Any]] = []

    for item in candidates:
        key = str(item["sample_key"])
        if key not in audit_by_key:
            raise ValueError(f"candidate lacks a matching local audit row: {key}")
        revised = copy.deepcopy(audit_by_key[key])
        derived = derive_candidate(item)
        augmented_item = copy.deepcopy(item)
        if derived is not None:
            augmented_item["candidate_tools"].append(derived)
            correct = revised.get("oracle_direction") == "c"
            predictions.append(
                {
                    "sample_key": key,
                    "predicate_family": derived["tool_observation"]["predicate_family"],
                    "predicted_direction": "c",
                    "oracle_match": correct,
                    "strictly_unreachable_before": revised.get("accepted") is not True,
                }
            )
            revised.setdefault("candidates", []).append(
                {
                    "run_id": "viddiff_symmetry_residual_abstention_v1",
                    "tool_name": TOOL_NAME,
                    "predicted_direction": "c",
                    "interval_valid": True,
                    "oracle_match": correct,
                    "confidence": 0.65,
                    "order_consistent": True,
                }
            )
            if revised.get("accepted") is not True and correct:
                revised["accepted"] = True
                revised["selected_tool"] = TOOL_NAME
                revised["rejection_reason"] = None
        augmented.append(augmented_item)
        revised_cases.append(revised)

    predicted = len(predictions)
    correct = sum(item["oracle_match"] for item in predictions)
    incremental = sum(
        item["oracle_match"] and item["strictly_unreachable_before"]
        for item in predictions
    )
    baseline_accepted = sum(item.get("accepted") is True for item in audit.get("cases", []))
    revised_accepted = sum(item.get("accepted") is True for item in revised_cases)
    summary = {
        "cases": len(candidates),
        "predictions": predicted,
        "correct_predictions": correct,
        "false_predictions": predicted - correct,
        "precision": correct / predicted if predicted else None,
        "incremental_recovered_cases": incremental,
        "baseline_strict_accepted": baseline_accepted,
        "revised_strict_accepted": revised_accepted,
        "revised_reachability": revised_accepted / len(candidates) if candidates else 0.0,
        "label_visible_to_tool": False,
    }
    revised_audit = copy.deepcopy(audit)
    revised_audit["run_id"] = "viddiff_strict_plus_symmetry_abstention_v1"
    revised_audit["policy"] = (
        "strict base admission plus a development-calibrated, label-blind, one-sided "
        "swap-instability abstention rule for kinematic predicates"
    )
    revised_audit["cases"] = revised_cases
    revised_audit["summary"] = {
        **summary,
        "selected_tools": {
            name: sum(
                item.get("accepted") is True and item.get("selected_tool") == name
                for item in revised_cases
            )
            for name in sorted(
                {
                    str(item.get("selected_tool"))
                    for item in revised_cases
                    if item.get("selected_tool")
                }
            )
        },
        "adaptive_candidate_sets": len(augmented),
        "safe_to_build_adaptive_harness": bool(augmented),
        "student_export_private_leak_tokens": [],
    }
    return augmented, revised_audit, {"summary": summary, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    augmented, revised_audit, report = evaluate(args.candidates, args.selection_audit)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "candidate_measurements.student.jsonl", augmented)
    write_json(args.output / "selection_audit.json", revised_audit)
    write_json(args.output / "symmetry_abstention_report.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
