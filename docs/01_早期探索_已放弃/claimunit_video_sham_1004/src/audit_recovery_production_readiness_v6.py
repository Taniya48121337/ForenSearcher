"""Readiness view for the evidence-first 55-episode development package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v3 import (
    DEFAULT_INVENTORY,
    DEFAULT_RUBRIC,
    _load,
    build_readiness_report_v3,
)
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v31_evidence_first55"
DEFAULT_BLIND = BASE / "recovery_blind_review_packet_v7_evidence_first55"
DEFAULT_CONTRAST = BASE / "recovery_sft_semantic_contrast_v3_evidence_first55"
DEFAULT_PORTABLE = BASE / "recovery_sft_semantic_contrast_portable_v4_evidence_first55"
DEFAULT_QUALITY = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike/humanlike_quality_comparison.json"
DEFAULT_OUTPUT = BASE / "production_readiness_audit_v6_evidence_first55/readiness_report.json"


def build_report(base: dict[str, Any], policy: dict[str, Any], contrast: dict[str, Any], semantic: dict[str, Any], evidence_first: dict[str, Any]) -> dict[str, Any]:
    report = dict(base)
    report["artifact"] = "recovery_production_readiness_audit_v6_evidence_first55"
    report["development_batch"]["status"] = "evidence_first_adjudication_development_candidate"
    report["policy_information"] = {
        "turns": policy["source_rows"],
        "whole_calls_unique_from_control_plane": policy["interpretation"]["control_plane_can_copy_whole_call"],
        "compiled_targets_visible": policy["interpretation"]["compiled_target_is_visible"],
        "choice_bearing_turns": policy["interpretation"]["choice_bearing_turns"],
        "weighted_mass_share_by_tier": policy["weighted_training_mass"]["share_by_tier"],
        "scope_limit": policy["training_qualification_note"],
    }
    report["semantic_contrast_view"] = {
        "rows": contrast["output_rows"],
        "contrast_groups": contrast["contrast_groups"],
        "contrast_covered_rows": contrast["contrast_covered_rows"],
        "same_menu_different_tool_groups": contrast["contrast_groups_by_family"]["same_menu_different_tool"],
        "same_menu_different_arguments_groups": contrast["contrast_groups_by_family"]["same_menu_different_arguments"],
        "construction_failures": contrast["construction_failures"],
        "messages_tools_weights_unchanged": bool(
            contrast["messages_unchanged"]
            and contrast["tools_unchanged"]
            and contrast["sampling_weights_unchanged"]
        ),
    }
    report["semantic_transparency"] = {
        "passed": semantic["passed"],
        "evidence_first_action_changes_verified": semantic["evidence_first_action_changes_verified"],
        "after": semantic["after"],
        "scope_limit": semantic["scope_limit"],
    }
    report["evidence_first_adjudication"] = {
        "passed": evidence_first["passed"],
        "adjudicated_episodes": evidence_first["adjudicated_episodes"],
        "changed_from_baseline": evidence_first["changed_from_baseline"],
        "checks": evidence_first["checks"],
    }
    report["permitted_claim"] = (
        "Fifty-five local development trajectories are executable and strict-H7 clean. "
        "All 12 multi-candidate episodes use a label-blind, one-vote-per-method "
        "adjudication rule; selected atoms alone determine later claim revision and binding."
    )
    report["training_qualified"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base = build_readiness_report_v3(
        _load(DEFAULT_INVENTORY),
        _load(DEFAULT_BATCH / "compile_report.json"),
        _load(DEFAULT_BATCH / "utqr_episode_scorecards.json"),
        _load(DEFAULT_BATCH / "utqr_falsification_report.json"),
        _load(DEFAULT_BLIND / "manifest.json"),
        _load(DEFAULT_BLIND / "private_do_not_share_with_judges/aggregation_status.json"),
        _load(DEFAULT_PORTABLE / "portability_audit.json"),
        _load(DEFAULT_PORTABLE / "reasoning_quality_audit_v2.json"),
        _load(DEFAULT_RUBRIC),
        _load(DEFAULT_QUALITY),
    )
    report = build_report(
        base,
        _load(DEFAULT_BATCH / "policy_information_audit.json"),
        _load(DEFAULT_CONTRAST / "contrast_report.json"),
        _load(DEFAULT_BATCH / "semantic_transparency_audit.json"),
        _load(DEFAULT_BATCH / "evidence_first_adjudication_audit.json"),
    )
    write_json(args.output, report)
    print(json.dumps({
        "episodes": report["development_batch"]["episodes"],
        "evidence_first_adjudication_passed": report["evidence_first_adjudication"]["passed"],
        "contrast_groups": report["semantic_contrast_view"]["contrast_groups"],
        "blind_status": report["blind_review"]["aggregation_status"],
        "training_qualified": report["training_qualified"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
