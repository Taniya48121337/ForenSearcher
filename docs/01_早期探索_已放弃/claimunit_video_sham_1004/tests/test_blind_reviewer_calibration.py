from __future__ import annotations

import csv
import json
from pathlib import Path

from aggregate_calibrated_blind_recovery_reviews import (
    DEFAULT_PACKET,
    DEFAULT_RUBRIC,
    DEFAULT_SCORECARDS,
    aggregate_calibrated_packet,
    evaluate_reviewer_calibration,
)
from export_blind_recovery_delta_review_packets import SUPPLEMENTAL_DIMENSIONS


BASE_DIMENSIONS = [
    "prospective_reasoning",
    "visible_evidence_grounding",
    "belief_update",
    "method_fit",
    "locality",
    "naturalness",
]
DIMENSIONS = [*BASE_DIMENSIONS, *SUPPLEMENTAL_DIMENSIONS]


def _row(score: str = "5") -> dict[str, str]:
    return {
        **{dimension: score for dimension in DIMENSIONS},
        "visible_hard_failure": "no",
        "accept_for_sft": "yes",
        "strongest_turn": "A2",
        "weakest_turn": "A1",
        "required_revision": "",
        "reviewer_notes": "A2 uses visible evidence; A1 is the least specific decision step.",
    }


def _keys() -> dict[str, dict]:
    rows = json.loads(
        (
            DEFAULT_PACKET / "private_do_not_share_with_judges/review_key.json"
        ).read_text(encoding="utf-8")
    )["items"]
    return {str(item["review_id"]): item for item in rows}


def _write_reviews(path: Path, rows: dict[str, dict[str, str]]) -> None:
    fields = ["review_id", *DIMENSIONS]
    fields.extend(
        [
            "visible_hard_failure",
            "accept_for_sft",
            "strongest_turn",
            "weakest_turn",
            "required_revision",
            "reviewer_notes",
        ]
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_id, row in rows.items():
            writer.writerow({"review_id": review_id, **row})


def test_calibrated_packet_has_a_one_to_one_public_private_mapping() -> None:
    packet_ids = [
        json.loads(line)["review_id"]
        for line in (
            DEFAULT_PACKET / "public_share_only/review_packets_delta.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = _keys()
    assert len(packet_ids) == len(set(packet_ids)) == 129
    assert set(packet_ids) == set(keys)
    roles = [item["review_role"] for item in keys.values()]
    assert roles.count("primary_quality") == 105
    assert roles.count("calibration_clean") == 8
    assert roles.count("calibration_negative") == 16


def test_sentinels_reject_both_rubber_stamping_and_blanket_rejection() -> None:
    keys = {
        review_id: item
        for review_id, item in _keys().items()
        if item["review_role"] != "primary_quality"
    }
    turn_counts = {review_id: 6 for review_id in keys}
    correct = {review_id: _row() for review_id in keys}
    for review_id, item in keys.items():
        if item["review_role"] != "calibration_negative":
            continue
        correct[review_id][item["expected_dimension"]] = "3"
        correct[review_id]["visible_hard_failure"] = "yes"
        correct[review_id]["accept_for_sft"] = "no"
    assert evaluate_reviewer_calibration(
        correct, keys, DIMENSIONS, turn_counts
    )["passed"] is True

    rubber_stamp = {review_id: _row() for review_id in keys}
    rubber = evaluate_reviewer_calibration(
        rubber_stamp, keys, DIMENSIONS, turn_counts
    )
    assert rubber["negative_detected"] == 0
    assert rubber["passed"] is False

    blanket_reject = {review_id: _row("1") for review_id in keys}
    for row in blanket_reject.values():
        row["visible_hard_failure"] = "yes"
        row["accept_for_sft"] = "no"
    blanket = evaluate_reviewer_calibration(
        blanket_reject, keys, DIMENSIONS, turn_counts
    )
    assert blanket["clean_accepted"] == 0
    assert blanket["passed"] is False


def test_full_aggregation_requires_reliable_reviewers(tmp_path: Path) -> None:
    keys = _keys()
    calibrated = {review_id: _row() for review_id in keys}
    for review_id, item in keys.items():
        if item["review_role"] != "calibration_negative":
            continue
        calibrated[review_id][item["expected_dimension"]] = "3"
        calibrated[review_id]["visible_hard_failure"] = "yes"
        calibrated[review_id]["accept_for_sft"] = "no"
    reviewer_a = tmp_path / "reviewer_A.csv"
    reviewer_b = tmp_path / "reviewer_B.csv"
    _write_reviews(reviewer_a, calibrated)
    _write_reviews(reviewer_b, calibrated)
    report = aggregate_calibrated_packet(
        DEFAULT_PACKET,
        DEFAULT_SCORECARDS,
        DEFAULT_RUBRIC,
        reviewer_a,
        reviewer_b,
    )
    assert report["status"] == "complete"
    assert report["reviewers_reliable"] is True
    assert report["primary_quality_report"]["blind_review_passed"] is True
    assert report["blind_review_passed"] is True
    assert report["training_qualified"] is False

    rubber = {review_id: _row() for review_id in keys}
    _write_reviews(reviewer_a, rubber)
    _write_reviews(reviewer_b, rubber)
    failed = aggregate_calibrated_packet(
        DEFAULT_PACKET,
        DEFAULT_SCORECARDS,
        DEFAULT_RUBRIC,
        reviewer_a,
        reviewer_b,
    )
    assert failed["primary_quality_report"]["blind_review_passed"] is True
    assert failed["reviewers_reliable"] is False
    assert failed["blind_review_passed"] is False
    assert failed["status"] == "complete_but_reviewer_calibration_failed"
