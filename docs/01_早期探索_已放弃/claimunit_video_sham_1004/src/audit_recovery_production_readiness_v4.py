"""Add policy-information and semantic-contrast evidence to readiness v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v3 import (
    DEFAULT_BATCH,
    DEFAULT_BLIND,
    DEFAULT_INVENTORY,
    DEFAULT_PORTABLE,
    DEFAULT_RUBRIC,
    _load,
    build_readiness_report_v3,
)
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_CONTRAST_PORTABLE = (
    BASE / "recovery_sft_semantic_contrast_portable_v1_edgeaware55"
)
DEFAULT_OUTPUT = (
    BASE / "production_readiness_audit_v4_semantic_contrast55/readiness_report.json"
)


def build_readiness_report_v4(
    base_report: dict[str, Any],
    policy_information: dict[str, Any],
    contrast: dict[str, Any],
    contrast_portability: dict[str, Any],
    contrast_reasoning: dict[str, Any],
) -> dict[str, Any]:
    report = dict(base_report)
    report["artifact"] = "recovery_production_readiness_audit_v4_semantic_contrast55"
    report["policy_information"] = {
        "turns": policy_information["source_rows"],
        "whole_calls_unique_from_control_plane": policy_information["interpretation"][
            "control_plane_can_copy_whole_call"
        ],
        "compiled_targets_visible": policy_information["interpretation"][
            "compiled_target_is_visible"
        ],
        "choice_bearing_turns": policy_information["interpretation"][
            "choice_bearing_turns"
        ],
        "weighted_mass_share_by_tier": policy_information[
            "weighted_training_mass"
        ]["share_by_tier"],
        "scope_limit": policy_information["training_qualification_note"],
    }
    report["semantic_contrast_view"] = {
        "rows": contrast["rows"],
        "contrast_groups": contrast["contrast_groups"],
        "contrast_covered_rows": contrast["contrast_covered_rows"],
        "same_menu_different_tool_groups": contrast[
            "same_menu_different_tool_groups"
        ],
        "same_menu_different_arguments_groups": contrast[
            "same_menu_different_arguments_groups"
        ],
        "construction_failures": contrast["construction_failures"],
        "messages_tools_weights_unchanged": bool(
            contrast["messages_unchanged"]
            and contrast["tools_unchanged"]
            and contrast["sampling_weights_unchanged"]
        ),
        "portable_semantic_equivalence": contrast_portability[
            "all_semantic_equivalence_checks_passed"
        ],
        "portable_forbidden_identifier_hits_zero": contrast_portability[
            "all_forbidden_identifier_hits_zero"
        ],
        "strict_reasoning_passed": contrast_reasoning["passed"],
        "training_qualified": False,
    }
    report["permitted_claim"] = (
        "Fifty-five local development trajectories are structurally executable, "
        "strict-H7 clean, and contain 67 verified structure-matched semantic contrast "
        "groups without changing their messages, targets, or sampling weights."
    )
    # These additions improve the candidate SFT organization but do not supply
    # the missing external evidence required by v3.
    report["training_qualified"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base_report = build_readiness_report_v3(
        _load(DEFAULT_INVENTORY),
        _load(DEFAULT_BATCH / "compile_report.json"),
        _load(DEFAULT_BATCH / "utqr_episode_scorecards.json"),
        _load(DEFAULT_BATCH / "utqr_falsification_report.json"),
        _load(DEFAULT_BLIND / "manifest.json"),
        _load(DEFAULT_BLIND / "private_do_not_share_with_judges/aggregation_status.json"),
        _load(DEFAULT_PORTABLE / "portability_audit.json"),
        _load(DEFAULT_PORTABLE / "reasoning_quality_audit_v2.json"),
        _load(DEFAULT_RUBRIC),
        _load(DEFAULT_BATCH / "humanlike_quality_comparison.json"),
    )
    report = build_readiness_report_v4(
        base_report,
        _load(DEFAULT_BATCH / "policy_information_audit.json"),
        _load(DEFAULT_CONTRAST_PORTABLE / "semantic_contrast_report.json"),
        _load(DEFAULT_CONTRAST_PORTABLE / "portability_audit.json"),
        _load(DEFAULT_CONTRAST_PORTABLE / "reasoning_quality_audit_v2.json"),
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "episodes": report["development_batch"]["episodes"],
                "contrast_groups": report["semantic_contrast_view"][
                    "contrast_groups"
                ],
                "whole_calls_unique_from_control_plane": report[
                    "policy_information"
                ]["whole_calls_unique_from_control_plane"],
                "blind_status": report["blind_review"]["aggregation_status"],
                "training_qualified": report["training_qualified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
