"""UTQR-8 packet evaluation with a complete unannounced-recovery contract.

UTQR-8 keeps UTQR-7's outcome, provenance, causal-neighborhood, autonomy,
economy, naturalness, and SFT-hygiene checks.  It strengthens the existing H5
boundary so merely calling ``challenge_evidence`` cannot qualify a packet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from utqr_v7_causal import evaluate_packet as evaluate_packet_v7


DEFAULT_RUBRIC = ROOT / "rubric" / "unified_trajectory_quality_rubric_v8.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recovery_checks(repair: dict[str, Any]) -> dict[str, bool]:
    correct_completion = (
        repair.get("rational_abstention") is True
        if repair.get("correct_behavior") == "abstain"
        else (
            repair.get("terminal_success_raw") is True
            and int(repair.get("rejection_count", 1)) == 0
        )
    )
    return {
        "probe_executed_label_blind": (
            repair.get("executed") is True and repair.get("label_blind") is True
        ),
        "detected_before_stop": repair.get("detected_before_stop") is True,
        "challenge_executed": repair.get("challenge_executed") is True,
        "challenge_target_valid": repair.get("challenged_evidence_ids_valid") is True,
        "defect_visibly_grounded": repair.get("defect_visibly_grounded") is True,
        "requested_check_method_fit": repair.get("requested_check_method_fit") is True,
        "fresh_check_executed": repair.get("fresh_check_executed") is True,
        "fresh_evidence_consumed": repair.get("fresh_evidence_consumed") is True,
        "targeted_action_executed": repair.get("targeted_action_executed") is True,
        "localized_reopen": repair.get("localized_reopen") is True,
        "no_overinvalidation": not list(repair.get("overinvalidated_fields") or []),
        "no_explicit_hint": repair.get("explicit_repair_hint_visible") is False,
        "no_premature_stop": repair.get("stopped_without_targeted_action") is False,
        "clean_control_suppresses_challenge": (
            repair.get("clean_control_unnecessary_challenge") is False
        ),
        "correct_completion": bool(correct_completion),
    }


def evaluate_packet(
    packet: dict[str, Any], *, rubric: dict[str, Any] | None = None
) -> dict[str, Any]:
    rubric = rubric or _json(DEFAULT_RUBRIC)
    report = evaluate_packet_v7(packet, rubric=rubric)
    repair = ((packet.get("probes") or {}).get("unannounced_repair") or {})
    checks = _recovery_checks(repair)
    score = sum(checks.values()) / len(checks)
    full_recovery = all(checks.values())

    criterion = next(item for item in report["criteria"] if item["id"] == "C7")
    criterion["normalized_score"] = round(score, 4)
    criterion["earned"] = round(float(criterion["max_points"]) * score, 3)
    criterion["passed"] = score >= float(criterion["minimum"])

    gate = next(item for item in report["hard_gates"] if item["id"] == "H5")
    gate["passed"] = full_recovery
    gate["failure"] = None if full_recovery else next(
        item["failure"] for item in rubric["hard_gates"] if item["id"] == "H5"
    )
    report["failed_gates"] = [
        item["id"] for item in report["hard_gates"] if not item["passed"]
    ]
    report["total_score"] = round(
        sum(float(item["earned"]) for item in report["criteria"]), 3
    )

    acceptance = rubric["episode_acceptance"]
    critical_ok = all(
        float(item["normalized_score"])
        >= float(acceptance["all_critical_criteria_minimum"])
        for item in report["criteria"]
        if next(
            spec.get("critical", False)
            for spec in rubric["criteria"]
            if spec["id"] == item["id"]
        )
    )
    any_ok = all(
        float(item["normalized_score"]) >= float(acceptance["any_criterion_minimum"])
        for item in report["criteria"]
    )
    report["accepted_for_sft"] = bool(
        all(item["passed"] for item in report["hard_gates"])
        and report["total_score"] >= float(acceptance["total_score_minimum"])
        and critical_ok
        and any_ok
        and report["blind_judging"]["passed"] is True
    )
    report["diagnosis"]["unannounced_causal_recovery"] = {
        "passed": full_recovery,
        "checks": checks,
        "missing_or_failed": [name for name, passed in checks.items() if not passed],
    }
    priorities = [
        item["id"]
        for item in sorted(report["criteria"], key=lambda row: row["normalized_score"])
        if not item["passed"]
    ]
    if not full_recovery and "C7" not in priorities:
        priorities.insert(0, "C7")
    report["diagnosis"]["priority_repairs"] = priorities
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    args = parser.parse_args()
    rubric = _json(args.rubric)
    reports = [evaluate_packet(packet, rubric=rubric) for packet in _jsonl(args.packets)]
    write_json(
        args.output,
        {
            "rubric_id": rubric["rubric_id"],
            "packets": len(reports),
            "accepted": sum(item["accepted_for_sft"] for item in reports),
            "reports": reports,
        },
    )


if __name__ == "__main__":
    main()
