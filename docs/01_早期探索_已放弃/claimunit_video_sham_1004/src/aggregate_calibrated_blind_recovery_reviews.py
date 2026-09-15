"""Aggregate primary blind quality scores only after reviewer sentinel calibration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aggregate_blind_recovery_reviews import (
    _boolean,
    _has_meaningful_note,
    _load,
    _packet_index,
    _read_adjudication,
    _read_csv,
    _score,
    _turn_reference,
    aggregate_reviews,
)
from export_blind_recovery_delta_review_packets import SUPPLEMENTAL_DIMENSIONS
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_PACKET = BASE / "recovery_lattice_blind_delta_packet_v2_calibrated"
DEFAULT_SCORECARDS = (
    BASE
    / "recovery_intervention_lattice_batch_v1/utqr12_episode_scorecards.json"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v12.json"


def evaluate_reviewer_calibration(
    reviewer: dict[str, dict[str, str]],
    sentinel_key: dict[str, dict[str, Any]],
    dimensions: list[str],
    assistant_turn_counts: dict[str, int],
    *,
    minimum_negative_detected: int = 15,
    minimum_clean_accepted: int = 7,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    invalid: list[str] = []
    mutation_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"items": 0, "detected": 0}
    )
    negative_detected = 0
    clean_accepted = 0
    negative_total = 0
    clean_total = 0
    for review_id, key in sentinel_key.items():
        row = reviewer.get(review_id) or {}
        scores = {dimension: _score(row.get(dimension, "")) for dimension in dimensions}
        hard = _boolean(row.get("visible_hard_failure", ""))
        accept = _boolean(row.get("accept_for_sft", ""))
        strongest = _turn_reference(
            row.get("strongest_turn", ""), assistant_turn_counts.get(review_id)
        )
        weakest = _turn_reference(
            row.get("weakest_turn", ""), assistant_turn_counts.get(review_id)
        )
        notes = row.get("reviewer_notes", "").strip()
        complete = bool(
            all(value is not None for value in scores.values())
            and hard is not None
            and accept is not None
            and strongest is not None
            and weakest is not None
            and (strongest != weakest or assistant_turn_counts.get(review_id, 0) <= 1)
            and _has_meaningful_note(notes)
        )
        if not complete:
            invalid.append(review_id)
        role = str(key["review_role"])
        passed = False
        if role == "calibration_clean":
            clean_total += 1
            passed = bool(
                complete
                and hard is False
                and accept is True
                and all(float(value) >= 4.0 for value in scores.values())
            )
            clean_accepted += int(passed)
        elif role == "calibration_negative":
            negative_total += 1
            expected = str(key["expected_dimension"])
            passed = bool(
                complete
                and (hard is True or accept is False)
                and scores.get(expected) is not None
                and float(scores[expected]) <= 3.0
            )
            negative_detected += int(passed)
            family = str(key["mutation"])
            mutation_summary[family]["items"] += 1
            mutation_summary[family]["detected"] += int(passed)
        else:
            raise ValueError(f"Unexpected sentinel role: {role}")
        items.append(
            {
                "review_id": review_id,
                "role": role,
                "mutation": key.get("mutation"),
                "expected_dimension": key.get("expected_dimension"),
                "complete": complete,
                "calibration_item_passed": passed,
            }
        )
    complete = not invalid
    passed = bool(
        complete
        and negative_total == 16
        and clean_total == 8
        and negative_detected >= minimum_negative_detected
        and clean_accepted >= minimum_clean_accepted
    )
    return {
        "complete": complete,
        "passed": passed,
        "negative_items": negative_total,
        "negative_detected": negative_detected,
        "minimum_negative_detected": minimum_negative_detected,
        "clean_items": clean_total,
        "clean_accepted": clean_accepted,
        "minimum_clean_accepted": minimum_clean_accepted,
        "invalid_or_missing_review_ids": sorted(invalid),
        "per_mutation": dict(sorted(mutation_summary.items())),
        "items": items,
    }


def aggregate_calibrated_packet(
    packet_root: Path,
    scorecards_path: Path,
    rubric_path: Path,
    reviewer_a_path: Path | None = None,
    reviewer_b_path: Path | None = None,
    adjudication_path: Path | None = None,
) -> dict[str, Any]:
    public = packet_root / "public_share_only"
    private = packet_root / "private_do_not_share_with_judges"
    expected_ids, turn_counts = _packet_index(public / "review_packets_delta.jsonl")
    reviewer_a, duplicates_a = _read_csv(reviewer_a_path or public / "reviewer_A.csv")
    reviewer_b, duplicates_b = _read_csv(reviewer_b_path or public / "reviewer_B.csv")
    adjudication, duplicate_adjudications = _read_adjudication(
        adjudication_path or private / "adjudication.csv"
    )
    key_rows = _load(private / "review_key.json")["items"]
    key = {str(item["review_id"]): item for item in key_rows}
    expected = set(expected_ids)
    structural_errors = {
        "reviewer_A_duplicate_ids": sorted(set(duplicates_a)),
        "reviewer_B_duplicate_ids": sorted(set(duplicates_b)),
        "reviewer_A_unknown_ids": sorted(set(reviewer_a) - expected),
        "reviewer_B_unknown_ids": sorted(set(reviewer_b) - expected),
        "reviewer_A_missing_rows": sorted(expected - set(reviewer_a)),
        "reviewer_B_missing_rows": sorted(expected - set(reviewer_b)),
        "adjudication_duplicate_keys": sorted(set(duplicate_adjudications)),
    }
    structural_ok = all(not value for value in structural_errors.values())

    rubric = _load(rubric_path)
    base_dimensions = list(rubric["blind_judge_contract"]["dimensions"])
    dimensions = [*base_dimensions, *SUPPLEMENTAL_DIMENSIONS]
    primary_key = {
        review_id: item
        for review_id, item in key.items()
        if item["review_role"] == "primary_quality"
    }
    sentinel_key = {
        review_id: item
        for review_id, item in key.items()
        if item["review_role"] != "primary_quality"
    }
    primary_ids = [review_id for review_id in expected_ids if review_id in primary_key]
    primary_adjudication = {
        name: item
        for name, item in adjudication.items()
        if name.split(":", 1)[0] in primary_key
    }
    scorecards = _load(scorecards_path)
    automated_by_episode = {
        str(item["episode_id"]): float(item["automated_core_score"])
        for item in scorecards["scorecards"]
    }
    hard_by_episode = {
        str(item["episode_id"]): bool(
            item.get("hard_gates", {}).get("H1_to_H7_automated", False)
        )
        for item in scorecards["scorecards"]
    }
    automated = {
        review_id: automated_by_episode[str(item["episode_id"])]
        for review_id, item in primary_key.items()
    }
    automated_hard = {
        review_id: hard_by_episode[str(item["episode_id"])]
        for review_id, item in primary_key.items()
    }
    readiness = rubric["batch_readiness"]
    primary_report = aggregate_reviews(
        primary_ids,
        {item: reviewer_a.get(item, {}) for item in primary_ids},
        {item: reviewer_b.get(item, {}) for item in primary_ids},
        primary_adjudication,
        primary_key,
        automated,
        base_dimensions,
        float(rubric["blind_judge_contract"]["maximum_unadjudicated_range"]),
        float(rubric["blind_judge_contract"]["minimum_dimension_mean"]),
        float(readiness["minimum_packet_acceptance_rate_per_dataset"]),
        float(readiness["minimum_score_p10_per_dataset"]),
        {item: turn_counts[item] for item in primary_ids},
        list(readiness["required_datasets"]),
        None,
        automated_hard,
        float(rubric["episode_acceptance"]["total_score_minimum"]),
        supplemental_dimensions=SUPPLEMENTAL_DIMENSIONS,
        minimum_supplemental_dimension_mean=4.0,
        minimum_acceptance_wilson_lower_bound=0.9,
    )
    calibration_a = evaluate_reviewer_calibration(
        reviewer_a, sentinel_key, dimensions, turn_counts
    )
    calibration_b = evaluate_reviewer_calibration(
        reviewer_b, sentinel_key, dimensions, turn_counts
    )
    all_forms_complete = bool(
        structural_ok
        and primary_report["review_completed"]
        and calibration_a["complete"]
        and calibration_b["complete"]
    )
    reviewers_reliable = bool(calibration_a["passed"] and calibration_b["passed"])
    blind_passed = bool(
        all_forms_complete
        and reviewers_reliable
        and primary_report["blind_review_passed"]
    )
    if not structural_ok or not calibration_a["complete"] or not calibration_b["complete"]:
        status = "awaiting_scores"
    elif primary_report["status"] == "awaiting_adjudication":
        status = "awaiting_primary_adjudication"
    elif not reviewers_reliable:
        status = "complete_but_reviewer_calibration_failed"
    else:
        status = "complete"
    blockers = [
        "no_sealed_entity_disjoint_holdout",
        "closed_loop_student_gain_not_evaluated",
    ]
    if not reviewers_reliable:
        blockers.insert(0, "reviewer_calibration_not_passed")
    if not blind_passed:
        blockers.insert(0, "independent_blind_review_not_passed")
    return {
        "artifact": "calibrated_blind_recovery_review_aggregation_v1",
        "status": status,
        "expected_public_items": len(expected_ids),
        "primary_quality_items": len(primary_ids),
        "calibration_items": len(sentinel_key),
        "all_forms_complete": all_forms_complete,
        "reviewers_reliable": reviewers_reliable,
        "blind_review_passed": blind_passed,
        "structural_errors": structural_errors,
        "reviewer_A_calibration": calibration_a,
        "reviewer_B_calibration": calibration_b,
        "primary_quality_report": primary_report,
        "training_qualified": False,
        "qualification_blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--scorecards", type=Path, default=DEFAULT_SCORECARDS)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = aggregate_calibrated_packet(
        args.packet_root,
        args.scorecards,
        args.rubric,
        args.reviewer_a,
        args.reviewer_b,
        args.adjudication,
    )
    output = args.output or (
        args.packet_root
        / "private_do_not_share_with_judges/aggregation_status.json"
    )
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "public_items": report["expected_public_items"],
                "primary_items": report["primary_quality_items"],
                "calibration_items": report["calibration_items"],
                "reviewers_reliable": report["reviewers_reliable"],
                "blind_review_passed": report["blind_review_passed"],
                "training_qualified": report["training_qualified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
