"""Validate and aggregate two independent blind-review sheets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_PACKET = BASE / "recovery_blind_review_packet_v3_expanded"
DEFAULT_SCORECARDS = (
    BASE
    / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio"
    / "utqr_episode_scorecards.json"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _packet_index(path: Path) -> tuple[list[str], dict[str, int]]:
    review_ids: list[str] = []
    assistant_turn_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packet = json.loads(line)
            review_id = str(packet["review_id"])
            review_ids.append(review_id)
            if "turns" in packet:
                assistant_turn_counts[review_id] = len(packet.get("turns", []))
            else:
                assistant_turn_counts[review_id] = sum(
                    message.get("role") == "assistant"
                    for message in packet.get("messages", [])
                )
    return review_ids, assistant_turn_counts


def _read_csv(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review_id = str(row.get("review_id") or "").strip()
            if not review_id:
                continue
            if review_id in rows:
                duplicates.append(review_id)
            rows[review_id] = {key: str(value or "").strip() for key, value in row.items()}
    return rows, duplicates


def _read_adjudication(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cleaned = {key: str(value or "").strip() for key, value in row.items()}
            review_id = cleaned.get("review_id", "")
            dimension = cleaned.get("dimension", "")
            if not review_id or not dimension:
                continue
            key = f"{review_id}:{dimension}"
            if key in rows:
                duplicates.append(key)
            rows[key] = cleaned
    return rows, duplicates


def _score(value: str) -> float | None:
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if 1.0 <= result <= 5.0 else None


def _boolean(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return True
    if lowered in {"no", "n", "false", "0"}:
        return False
    return None


def _turn_reference(value: str, maximum: int | None) -> str | None:
    match = re.fullmatch(r"A\s*0*([1-9]\d*)", value.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    number = int(match.group(1))
    if maximum is not None and number > maximum:
        return None
    return f"A{number}"


def _has_meaningful_note(value: str) -> bool:
    return len(re.sub(r"\s+", "", value)) >= 12


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + (z * z / trials)
    centre = proportion + (z * z / (2.0 * trials))
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / trials)
        + (z * z / (4.0 * trials * trials))
    )
    return (centre - margin) / denominator


def aggregate_reviews(
    expected_ids: list[str],
    reviewer_a: dict[str, dict[str, str]],
    reviewer_b: dict[str, dict[str, str]],
    adjudication: dict[str, dict[str, str]],
    review_key: dict[str, dict[str, Any]],
    automated_scores: dict[str, float],
    dimensions: list[str],
    maximum_unadjudicated_range: float,
    minimum_dimension_mean: float,
    minimum_acceptance_rate: float,
    minimum_score_p10: float,
    assistant_turn_counts: dict[str, int] | None = None,
    required_datasets: list[str] | None = None,
    minimum_entity_disjoint_cases_per_dataset: int | None = None,
    automated_hard_gate_passed: dict[str, bool] | None = None,
    minimum_total_score: float = 90.0,
    duplicate_ids_a: list[str] | None = None,
    duplicate_ids_b: list[str] | None = None,
    duplicate_adjudication_keys: list[str] | None = None,
    supplemental_dimensions: list[str] | None = None,
    minimum_supplemental_dimension_mean: float | None = None,
    minimum_acceptance_wilson_lower_bound: float | None = None,
) -> dict[str, Any]:
    expected = set(expected_ids)
    structural_errors = {
        "reviewer_A_duplicate_ids": sorted(set(duplicate_ids_a or [])),
        "reviewer_B_duplicate_ids": sorted(set(duplicate_ids_b or [])),
        "reviewer_A_unknown_ids": sorted(set(reviewer_a) - expected),
        "reviewer_B_unknown_ids": sorted(set(reviewer_b) - expected),
        "reviewer_A_missing_rows": sorted(expected - set(reviewer_a)),
        "reviewer_B_missing_rows": sorted(expected - set(reviewer_b)),
        "adjudication_duplicate_keys": sorted(
            set(duplicate_adjudication_keys or [])
        ),
        "adjudication_unknown_or_unneeded_keys": [],
    }
    items: list[dict[str, Any]] = []
    missing_scores: list[str] = []
    unresolved: list[dict[str, str]] = []
    required_adjudication_keys: set[str] = set()
    for review_id in expected_ids:
        left = reviewer_a.get(review_id) or {}
        right = reviewer_b.get(review_id) or {}
        dimension_scores: dict[str, Any] = {}
        complete = True
        input_missing = False
        for dimension in dimensions:
            score_a = _score(left.get(dimension, ""))
            score_b = _score(right.get(dimension, ""))
            if score_a is None or score_b is None:
                complete = False
                input_missing = True
                dimension_scores[dimension] = {
                    "reviewer_A": score_a,
                    "reviewer_B": score_b,
                    "final": None,
                    "status": "missing",
                }
                continue
            gap = abs(score_a - score_b)
            final: float | None = None
            status = "mean_within_range"
            if gap > maximum_unadjudicated_range:
                key = f"{review_id}:{dimension}"
                required_adjudication_keys.add(key)
                adjudicated = adjudication.get(key) or {}
                final = _score(adjudicated.get("adjudicated_score", ""))
                reason = adjudicated.get("adjudication_reason", "").strip()
                if final is None or not reason:
                    status = "unresolved_disagreement"
                    unresolved.append(
                        {"review_id": review_id, "dimension": dimension}
                    )
                    complete = False
                    final = None
                else:
                    status = "adjudicated"
            else:
                final = round((score_a + score_b) / 2, 3)
            dimension_scores[dimension] = {
                "reviewer_A": score_a,
                "reviewer_B": score_b,
                "range": gap,
                "final": final,
                "status": status,
            }
        supplemental_scores: dict[str, Any] = {}
        for dimension in supplemental_dimensions or []:
            score_a = _score(left.get(dimension, ""))
            score_b = _score(right.get(dimension, ""))
            if score_a is None or score_b is None:
                complete = False
                input_missing = True
                supplemental_scores[dimension] = {
                    "reviewer_A": score_a,
                    "reviewer_B": score_b,
                    "final": None,
                    "status": "missing",
                }
                continue
            gap = abs(score_a - score_b)
            final: float | None = None
            status = "mean_within_range"
            if gap > maximum_unadjudicated_range:
                key = f"{review_id}:{dimension}"
                required_adjudication_keys.add(key)
                adjudicated = adjudication.get(key) or {}
                final = _score(adjudicated.get("adjudicated_score", ""))
                reason = adjudicated.get("adjudication_reason", "").strip()
                if final is None or not reason:
                    status = "unresolved_disagreement"
                    unresolved.append(
                        {"review_id": review_id, "dimension": dimension}
                    )
                    complete = False
                    final = None
                else:
                    status = "adjudicated"
            else:
                final = round((score_a + score_b) / 2, 3)
            supplemental_scores[dimension] = {
                "reviewer_A": score_a,
                "reviewer_B": score_b,
                "range": gap,
                "final": final,
                "status": status,
            }
        hard_a = _boolean(left.get("visible_hard_failure", ""))
        hard_b = _boolean(right.get("visible_hard_failure", ""))
        accept_a = _boolean(left.get("accept_for_sft", ""))
        accept_b = _boolean(right.get("accept_for_sft", ""))
        if None in {hard_a, hard_b, accept_a, accept_b}:
            complete = False
            input_missing = True
        review_evidence: dict[str, Any] = {}
        maximum_turn = (assistant_turn_counts or {}).get(review_id)
        for judge, review in (("reviewer_A", left), ("reviewer_B", right)):
            strongest = _turn_reference(review.get("strongest_turn", ""), maximum_turn)
            weakest = _turn_reference(review.get("weakest_turn", ""), maximum_turn)
            notes = review.get("reviewer_notes", "").strip()
            evidence_errors: list[str] = []
            if strongest is None:
                evidence_errors.append("invalid_or_missing_strongest_turn")
            if weakest is None:
                evidence_errors.append("invalid_or_missing_weakest_turn")
            if strongest is not None and strongest == weakest and (maximum_turn or 0) > 1:
                evidence_errors.append("strongest_and_weakest_turn_must_differ")
            if not _has_meaningful_note(notes):
                evidence_errors.append("reviewer_notes_too_short")
            if evidence_errors:
                complete = False
                input_missing = True
            review_evidence[judge] = {
                "strongest_turn": strongest,
                "weakest_turn": weakest,
                "reviewer_notes": notes,
                "errors": evidence_errors,
            }
        revisions = [
            text
            for text in (
                left.get("required_revision", "").strip(),
                right.get("required_revision", "").strip(),
            )
            if text
        ]
        final_dimensions = [
            value["final"]
            for value in dimension_scores.values()
            if value["final"] is not None
        ]
        if input_missing:
            missing_scores.append(review_id)
        mean = (
            round(sum(final_dimensions) / len(final_dimensions), 3)
            if complete and len(final_dimensions) == len(dimensions)
            else None
        )
        human_c9 = round((mean / 5.0) * 8.0, 3) if mean is not None else None
        automated = automated_scores.get(review_id)
        final_utqr = (
            round(automated + human_c9, 3)
            if automated is not None and human_c9 is not None
            else None
        )
        h8_passed = bool(
            complete
            and mean is not None
            and all(
                value["final"] >= minimum_dimension_mean
                for value in dimension_scores.values()
            )
            and hard_a is False
            and hard_b is False
            and accept_a is True
            and accept_b is True
            and not revisions
        )
        supplemental_finals = [
            value["final"]
            for value in supplemental_scores.values()
            if value["final"] is not None
        ]
        supplemental_mean = (
            round(sum(supplemental_finals) / len(supplemental_finals), 3)
            if supplemental_finals
            and len(supplemental_finals) == len(supplemental_scores)
            else None
        )
        supplemental_passed = bool(
            not supplemental_scores
            or (
                supplemental_mean is not None
                and all(
                    value["final"]
                    >= float(
                        4.0
                        if minimum_supplemental_dimension_mean is None
                        else minimum_supplemental_dimension_mean
                    )
                    for value in supplemental_scores.values()
                )
            )
        )
        hard_gates_ok = (automated_hard_gate_passed or {}).get(review_id, True)
        accepted = bool(
            h8_passed
            and supplemental_passed
            and hard_gates_ok
            and final_utqr is not None
            and final_utqr >= minimum_total_score
        )
        key_item = review_key.get(review_id) or {}
        items.append(
            {
                "review_id": review_id,
                "episode_id": key_item.get("episode_id"),
                "dataset": key_item.get("dataset"),
                "dimension_scores": dimension_scores,
                "dimension_mean": mean,
                "development_addendum_scores": supplemental_scores,
                "development_addendum_mean": supplemental_mean,
                "development_addendum_passed": supplemental_passed,
                "human_C9_score_out_of_8": human_c9,
                "automated_core_score_out_of_92": automated,
                "automated_H1_to_H7_passed": hard_gates_ok,
                "final_UTQR_score_out_of_100": final_utqr,
                "visible_hard_failure": (
                    None if hard_a is None or hard_b is None else hard_a or hard_b
                ),
                "both_judges_accept_for_sft": (
                    None
                    if accept_a is None or accept_b is None
                    else accept_a and accept_b
                ),
                "required_revisions": revisions,
                "review_evidence": review_evidence,
                "review_complete": complete,
                "H8_episode_passed": h8_passed,
                "accepted_for_sft": accepted,
            }
        )

    structural_errors["adjudication_unknown_or_unneeded_keys"] = sorted(
        set(adjudication) - required_adjudication_keys
    )
    structural_ok = all(not value for value in structural_errors.values())
    completed = structural_ok and not missing_scores and not unresolved
    packet_counts: dict[str, int] = defaultdict(int)
    for item in items:
        packet_counts[str(item["dataset"])] += 1
    count_readiness = {
        dataset: {
            "episodes": packet_counts.get(dataset, 0),
            "minimum_required": minimum_entity_disjoint_cases_per_dataset,
            "passed": (
                minimum_entity_disjoint_cases_per_dataset is None
                or packet_counts.get(dataset, 0)
                >= minimum_entity_disjoint_cases_per_dataset
            ),
        }
        for dataset in (required_datasets or sorted(packet_counts))
    }
    dataset_summary: dict[str, Any] = {}
    if completed:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[str(item["dataset"])].append(item)
        for dataset, values in sorted(grouped.items()):
            accepted = sum(item["accepted_for_sft"] for item in values)
            scores = [float(item["final_UTQR_score_out_of_100"]) for item in values]
            rate = accepted / len(values)
            p10 = _percentile(scores, 0.10)
            dataset_summary[dataset] = {
                "episodes": len(values),
                "accepted": accepted,
                "acceptance_rate": round(rate, 4),
                "acceptance_wilson95_lower_bound": round(
                    _wilson_lower_bound(accepted, len(values)), 6
                ),
                "score_mean": round(sum(scores) / len(scores), 3),
                "score_p10": p10,
                "acceptance_rate_passed": rate >= minimum_acceptance_rate,
                "acceptance_wilson95_lower_bound_passed": (
                    minimum_acceptance_wilson_lower_bound is None
                    or _wilson_lower_bound(accepted, len(values))
                    >= minimum_acceptance_wilson_lower_bound
                ),
                "score_p10_passed": p10 is not None and p10 >= minimum_score_p10,
            }
    blind_passed = bool(
        completed
        and dataset_summary
        and all(
            item["acceptance_rate_passed"]
            and item["acceptance_wilson95_lower_bound_passed"]
            and item["score_p10_passed"]
            for item in dataset_summary.values()
        )
    )
    status = (
        "awaiting_scores"
        if not structural_ok or missing_scores
        else "awaiting_adjudication"
        if unresolved
        else "complete"
    )
    blockers = ["sealed_entity_disjoint_holdout_not_completed", "student_free_run_gain_not_evaluated"]
    if any(not item["passed"] for item in count_readiness.values()):
        blockers.insert(0, "minimum_entity_disjoint_case_count_not_met")
    if not blind_passed:
        blockers.insert(0, "independent_blind_review_not_passed")
    return {
        "artifact": "blind_recovery_review_aggregation_v1",
        "status": status,
        "expected_episodes": len(expected_ids),
        "review_completed": completed,
        "blind_review_passed": blind_passed,
        "minimum_acceptance_wilson95_lower_bound": (
            minimum_acceptance_wilson_lower_bound
        ),
        "development_addendum_dimensions": supplemental_dimensions or [],
        "structural_errors": structural_errors,
        "missing_or_invalid_review_ids": sorted(set(missing_scores)),
        "unresolved_disagreements": unresolved,
        "dataset_packet_count_readiness": count_readiness,
        "dataset_summary": dataset_summary,
        "training_qualified": False,
        "qualification_blockers": blockers,
        "items": items,
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
    parser.add_argument("--packet-file", default="review_packets.jsonl")
    parser.add_argument("--supplemental-dimension", action="append", default=[])
    parser.add_argument("--minimum-supplemental-mean", type=float, default=4.0)
    parser.add_argument("--minimum-wilson-lower-bound", type=float)
    parser.add_argument(
        "--development-variant-packet",
        action="store_true",
        help=(
            "Do not mislabel repeated intervention variants as entity-disjoint cases."
        ),
    )
    args = parser.parse_args()
    public = args.packet_root / "public_share_only"
    private = args.packet_root / "private_do_not_share_with_judges"
    reviewer_a_path = args.reviewer_a or public / "reviewer_A.csv"
    reviewer_b_path = args.reviewer_b or public / "reviewer_B.csv"
    adjudication_path = args.adjudication or private / "adjudication.csv"
    output_path = args.output or private / "aggregation_status.json"
    expected_ids, assistant_turn_counts = _packet_index(public / args.packet_file)
    reviewer_a, duplicates_a = _read_csv(reviewer_a_path)
    reviewer_b, duplicates_b = _read_csv(reviewer_b_path)
    adjudication, duplicate_adjudications = _read_adjudication(adjudication_path)
    key = {
        str(item["review_id"]): item
        for item in _load(private / "review_key.json")["items"]
    }
    scorecards = _load(args.scorecards)
    automated_by_episode = {
        str(item["episode_id"]): float(item["automated_core_score"])
        for item in scorecards["scorecards"]
    }
    automated = {
        review_id: automated_by_episode[str(item["episode_id"])]
        for review_id, item in key.items()
    }
    automated_hard_gate_by_episode = {
        str(item["episode_id"]): bool(
            item.get("hard_gates", {}).get("H1_to_H7_automated", False)
        )
        for item in scorecards["scorecards"]
    }
    automated_hard_gates = {
        review_id: automated_hard_gate_by_episode[str(item["episode_id"])]
        for review_id, item in key.items()
    }
    rubric = _load(args.rubric)
    blind = rubric["blind_judge_contract"]
    readiness = rubric["batch_readiness"]
    report = aggregate_reviews(
        expected_ids,
        reviewer_a,
        reviewer_b,
        adjudication,
        key,
        automated,
        list(blind["dimensions"]),
        float(blind["maximum_unadjudicated_range"]),
        float(blind["minimum_dimension_mean"]),
        float(readiness["minimum_packet_acceptance_rate_per_dataset"]),
        float(readiness["minimum_score_p10_per_dataset"]),
        assistant_turn_counts,
        list(readiness["required_datasets"]),
        (
            None
            if args.development_variant_packet
            else int(readiness["minimum_entity_disjoint_cases_per_dataset"])
        ),
        automated_hard_gates,
        float(rubric["episode_acceptance"]["total_score_minimum"]),
        duplicates_a,
        duplicates_b,
        duplicate_adjudications,
        list(args.supplemental_dimension),
        args.minimum_supplemental_mean,
        args.minimum_wilson_lower_bound,
    )
    write_json(output_path, report)
    print(json.dumps({
        "output": str(output_path),
        "status": report["status"],
        "review_completed": report["review_completed"],
        "missing": len(report["missing_or_invalid_review_ids"]),
        "unresolved": len(report["unresolved_disagreements"]),
        "training_qualified": report["training_qualified"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
