"""Readiness view for the v20 abstention-aware semantic-contrast package."""

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
from audit_recovery_production_readiness_v4 import build_readiness_report_v4
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v20_abstention_semantic_transparency"
DEFAULT_BLIND = BASE / "recovery_blind_review_packet_v6_abstention55"
DEFAULT_PORTABLE = BASE / "recovery_sft_semantic_contrast_portable_v2_abstention55"
DEFAULT_QUALITY = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike/humanlike_quality_comparison.json"
DEFAULT_OUTPUT = BASE / "production_readiness_audit_v5_abstention55/readiness_report.json"


def build_readiness_report_v5(
    report_v4: dict[str, Any], semantic: dict[str, Any]
) -> dict[str, Any]:
    report = dict(report_v4)
    report["artifact"] = "recovery_production_readiness_audit_v5_abstention55"
    report["development_batch"]["status"] = (
        "abstention_aware_semantic_contrast_development_candidate"
    )
    report["semantic_transparency"] = {
        "passed": semantic["passed"],
        "executed_action_signatures_unchanged": semantic[
            "executed_action_signatures_unchanged"
        ],
        "before": semantic["before"],
        "after": semantic["after"],
        "uses_oracle_or_verifier": semantic["uses_oracle_or_verifier"],
        "scope_limit": semantic["scope_limit"],
    }
    report["permitted_claim"] = (
        "Fifty-five local development trajectories are executable, strict-H7 clean, "
        "abstention-aware, and organized into 67 structure-matched semantic contrast "
        "groups without changing executed action targets or sampling weights."
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
    report_v4 = build_readiness_report_v4(
        base,
        _load(DEFAULT_BATCH / "policy_information_audit.json"),
        _load(DEFAULT_PORTABLE / "semantic_contrast_report.json"),
        _load(DEFAULT_PORTABLE / "portability_audit.json"),
        _load(DEFAULT_PORTABLE / "reasoning_quality_audit_v2.json"),
    )
    report = build_readiness_report_v5(
        report_v4,
        _load(DEFAULT_BATCH / "semantic_transparency_audit.json"),
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "episodes": report["development_batch"]["episodes"],
                "semantic_transparency_passed": report["semantic_transparency"]["passed"],
                "contrast_groups": report["semantic_contrast_view"]["contrast_groups"],
                "blind_status": report["blind_review"]["aggregation_status"],
                "training_qualified": report["training_qualified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
