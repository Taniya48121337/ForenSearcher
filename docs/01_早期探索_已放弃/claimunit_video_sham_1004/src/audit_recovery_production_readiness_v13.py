"""Attach hidden-sentinel reviewer calibration to recovery readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v12 import build_report as build_v12
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
PACKET = BASE / "recovery_lattice_blind_delta_packet_v2_calibrated"
DEFAULT_OUTPUT = (
    BASE
    / "production_readiness_audit_v13_reviewer_calibration129"
    / "readiness_report.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    report = build_v12()
    manifest = _load(PACKET / "manifest.json")
    aggregation = _load(
        PACKET / "private_do_not_share_with_judges/aggregation_status.json"
    )
    report["artifact"] = "recovery_production_readiness_audit_v13_reviewer_calibration129"
    report["calibrated_independent_blind_review"] = {
        "status": aggregation["status"],
        "total_public_items": manifest["total_public_items"],
        "primary_quality_items": manifest["primary_quality_items"],
        "calibration_clean_items": manifest["calibration_clean_items"],
        "calibration_negative_items": manifest["calibration_negative_items"],
        "calibration_mutation_families": manifest[
            "calibration_mutation_families"
        ],
        "each_mutation_family_count": manifest["each_mutation_family_count"],
        "reviewers_informed_that_hidden_calibration_exists": manifest[
            "reviewers_informed_that_hidden_calibration_exists"
        ],
        "sentinel_identity_visible_to_reviewers": manifest[
            "sentinel_identity_visible_to_reviewers"
        ],
        "sentinels_included_in_quality_estimate": manifest[
            "sentinels_included_in_quality_estimate"
        ],
        "negative_sentinels_included_in_sft": manifest[
            "negative_sentinels_included_in_sft"
        ],
        "reviewer_reliability_rule": manifest["reviewer_reliability_rule"],
        "public_privacy_hits": manifest["public_privacy_hits"],
        "sentinel_construction_audit_passed": manifest[
            "sentinel_construction_audit_passed"
        ],
        "all_forms_complete": aggregation["all_forms_complete"],
        "reviewers_reliable": aggregation["reviewers_reliable"],
        "blind_review_passed": aggregation["blind_review_passed"],
        "scope": (
            "The 24 sentinels calibrate human discrimination and are excluded from "
            "the 105-item development quality estimate and all SFT artifacts."
        ),
    }
    report["training_qualified"] = False
    report["qualification_blockers"] = [
        "two_independent_calibrated_blind_reviews_not_completed",
        "no_sealed_entity_disjoint_holdout",
        "minimum_30_entity_disjoint_cases_not_met_for_each_dataset",
        "closed_loop_student_gain_not_evaluated",
    ]
    report["permitted_claim"] = (
        "The 105-item development blind-review design now includes 24 hidden "
        "reviewer-calibration controls spanning eight semantic defect families. "
        "Automated tests show that uniform acceptance and blanket rejection both fail "
        "calibration. Human forms remain blank, so neither reviewer reliability nor "
        "trajectory H8 quality has been established."
    )
    report["prohibited_claims"] = sorted(
        set(report.get("prohibited_claims") or [])
        | {
            "Synthetic sentinel labels are additional SFT positives.",
            "A reviewer who agrees with another reviewer is reliable without discriminating clean controls from negative sentinels.",
            "Passing reviewer calibration means the 105 primary trajectories pass H8.",
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "artifact": report["artifact"],
                "calibrated_blind_review": report[
                    "calibrated_independent_blind_review"
                ],
                "training_qualified": report["training_qualified"],
                "qualification_blockers": report["qualification_blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
