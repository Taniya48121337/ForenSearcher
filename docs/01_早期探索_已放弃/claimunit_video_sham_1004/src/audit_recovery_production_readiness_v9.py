"""Readiness view for UTQR-11 n-ary contrast-identifiable evidence-first SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_recovery_production_readiness_v3 import (
    DEFAULT_INVENTORY,
    _load,
    build_readiness_report_v3,
)
from audit_recovery_production_readiness_v7 import build_report as build_v7_report
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v31_evidence_first55"
DEFAULT_CLAIM_FIRST_BATCH = (
    BASE / "recovery_batch_candidate_v20_abstention_semantic_transparency"
)
DEFAULT_BLIND = BASE / "recovery_blind_review_packet_v7_evidence_first55"
DEFAULT_CONTRAST = BASE / "recovery_sft_semantic_contrast_v6_neighborhood55"
DEFAULT_PORTABLE = (
    BASE / "recovery_sft_semantic_contrast_portable_v7_neighborhood55"
)
DEFAULT_QUALITY = (
    BASE
    / "recovery_batch_candidate_v16_edge_aware_humanlike"
    / "humanlike_quality_comparison.json"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v11.json"
DEFAULT_OUTPUT = (
    BASE
    / "production_readiness_audit_v9_utqr11_neighborhood55"
    / "readiness_report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rubric = _load(DEFAULT_RUBRIC)
    current_scores = _load(DEFAULT_BATCH / "utqr11_episode_scorecards.json")
    claim_first_scores = _load(
        DEFAULT_CLAIM_FIRST_BATCH / "utqr11_episode_scorecards.json"
    )
    falsification = _load(DEFAULT_BATCH / "utqr11_falsification_report.json")
    contrast = _load(DEFAULT_CONTRAST / "contrast_report.json")
    contrast_integrity = _load(DEFAULT_CONTRAST / "contrast_integrity_audit.json")
    portable_integrity = _load(DEFAULT_PORTABLE / "contrast_integrity_audit.json")

    base = build_readiness_report_v3(
        _load(DEFAULT_INVENTORY),
        _load(DEFAULT_BATCH / "compile_report.json"),
        current_scores,
        falsification,
        _load(DEFAULT_BLIND / "manifest.json"),
        _load(
            DEFAULT_BLIND
            / "private_do_not_share_with_judges"
            / "aggregation_status.json"
        ),
        _load(DEFAULT_PORTABLE / "portability_audit.json"),
        _load(DEFAULT_PORTABLE / "reasoning_quality_audit_v2.json"),
        rubric,
        _load(DEFAULT_QUALITY),
    )
    claim_first_failed = [
        item
        for item in claim_first_scores["scorecards"]
        if item["criteria"]["C11"] == 0.0
    ]
    calibration = {
        "rubric_id": current_scores["rubric_id"],
        "evidence_first_dataset_summary": current_scores["dataset_summary"],
        "claim_first_dataset_summary": claim_first_scores["dataset_summary"],
        "claim_first_adjudications_failed": len(claim_first_failed),
        "claim_first_wrong_winner_count": sum(
            any(
                not detail["checks"]["selected_matches_visible_recompute"]
                for detail in item["evidence_first_adjudication"]["details"]
            )
            for item in claim_first_failed
        ),
        "negative_mutations_detected": falsification["all_detected"],
        "scope_limit": (
            "Development calibration only; no sealed holdout or human blind scores "
            "were used."
        ),
    }
    report = build_v7_report(
        base,
        _load(DEFAULT_BATCH / "policy_information_audit.json"),
        contrast,
        _load(DEFAULT_BATCH / "semantic_transparency_audit.json"),
        _load(DEFAULT_BATCH / "evidence_first_adjudication_audit.json"),
        calibration,
    )
    contract = rubric["batch_contrastive_identifiability_contract"]
    covered_stages = set(contrast_integrity["covered_rows_by_stage"])
    required_stages = set(contract["required_covered_stages"])
    coverage_passed = (
        contrast_integrity["covered_row_fraction"]
        >= contract["minimum_covered_row_fraction"]
    )
    stage_coverage_passed = required_stages.issubset(covered_stages)
    report["artifact"] = "recovery_production_readiness_audit_v9_utqr11_neighborhood55"
    report["development_batch"]["status"] = (
        "evidence_first_contrast_identifiable_development_candidate"
    )
    report["contrastive_identifiability"] = {
        "groups": contrast["contrast_groups"],
        "new_structure_matched_groups": contrast[
            "structure_matched_target_collision_groups"
        ],
        "implied_cross_target_pairs": contrast[
            "structure_matched_collision_implied_cross_target_pairs"
        ],
        "group_shape": "n-ary semantic neighborhoods; sample different targets",
        "covered_rows": contrast_integrity["covered_rows"],
        "covered_row_fraction": contrast_integrity["covered_row_fraction"],
        "minimum_covered_row_fraction": contract[
            "minimum_covered_row_fraction"
        ],
        "covered_rows_by_stage": contrast_integrity["covered_rows_by_stage"],
        "required_stages": sorted(required_stages),
        "coverage_passed": coverage_passed,
        "required_stage_coverage_passed": stage_coverage_passed,
        "canonical_integrity_passed": contrast_integrity["passed"],
        "portable_integrity_passed": portable_integrity["passed"],
        "messages_tools_weights_unchanged": bool(
            contrast["messages_unchanged"]
            and contrast["tools_unchanged"]
            and contrast["sampling_weights_unchanged"]
        ),
        "target_grouping_visible_to_student": contrast[
            "target_grouping_visible_to_student"
        ],
        "passed": bool(
            coverage_passed
            and stage_coverage_passed
            and contrast_integrity["passed"]
            and portable_integrity["passed"]
            and not contrast["target_grouping_visible_to_student"]
        ),
        "scope_limit": contract["scope_limit"],
    }
    report["permitted_claim"] = (
        "Fifty-five local development trajectories are executable, evidence-first, "
        "and organized into policy-invisible structure-matched semantic neighborhoods "
        "covering 333/418 SFT turns with 1,429 implied cross-target pairs. This "
        "strengthens training identifiability but "
        "does not establish human quality, sealed generalization, or student gain."
    )
    report["training_qualified"] = False
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "episodes": report["development_batch"]["episodes"],
                "rubric_id": report["utqr10_calibration"]["rubric_id"],
                "contrast_groups": report["contrastive_identifiability"]["groups"],
                "contrast_covered_fraction": report["contrastive_identifiability"][
                    "covered_row_fraction"
                ],
                "contrast_integrity_passed": report[
                    "contrastive_identifiability"
                ]["passed"],
                "blind_status": report["blind_review"]["aggregation_status"],
                "training_qualified": report["training_qualified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
