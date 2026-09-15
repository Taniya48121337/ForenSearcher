"""Production-readiness view for the edge-aware human-like recovery batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v2 import build_readiness_report_v2
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_INVENTORY = BASE / "production_readiness_audit_v1/local_trajectory_inventory.json"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike"
DEFAULT_BLIND = BASE / "recovery_blind_review_packet_v5_edgeaware55"
DEFAULT_PORTABLE = BASE / "recovery_sft_portable_v6_edgeaware55"
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = BASE / "production_readiness_audit_v3_edgeaware55/readiness_report.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness_report_v3(
    inventory: dict[str, Any],
    compile_report: dict[str, Any],
    scorecards: dict[str, Any],
    falsification: dict[str, Any],
    blind_manifest: dict[str, Any],
    blind_aggregation: dict[str, Any],
    portability: dict[str, Any],
    portable_reasoning: dict[str, Any],
    rubric: dict[str, Any],
    quality_comparison: dict[str, Any],
) -> dict[str, Any]:
    report = build_readiness_report_v2(
        inventory,
        compile_report,
        scorecards,
        falsification,
        blind_manifest,
        blind_aggregation,
        portability,
        portable_reasoning,
        rubric,
    )
    report["artifact"] = "recovery_production_readiness_audit_v3_edgeaware55"
    report["development_batch"]["status"] = (
        "edge_aware_humanlike_development_candidate"
    )
    report["humanlike_quality_improvement"] = {
        "established_without_action_target_changes": bool(
            quality_comparison["improvement_established"]
            and quality_comparison["executed_action_signatures_unchanged"]
        ),
        "strict_reasoning_rows_before": quality_comparison["before"][
            "strict_additional_check_passed_rows"
        ],
        "strict_reasoning_rows_after": quality_comparison["after"][
            "strict_additional_check_passed_rows"
        ],
        "meta_process_hits_before_after": [
            quality_comparison["before"]["meta_process_narration_hits"],
            quality_comparison["after"]["meta_process_narration_hits"],
        ],
        "unbalanced_serialization_before_after": [
            quality_comparison["before"]["unbalanced_inline_serialization_hits"],
            quality_comparison["after"]["unbalanced_inline_serialization_hits"],
        ],
        "edge_aware_binding_diagnoses": quality_comparison[
            "edge_aware_binding_diagnoses"
        ],
        "scope_limit": quality_comparison["scope_limit"],
    }
    report["permitted_claim"] = (
        "Fifty-five local development trajectories retain the same executed action "
        "targets while passing the stricter H7 text audit, automated UTQR core, "
        "generic mutation falsification, causal recovery checks, and portable audit."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_readiness_report_v3(
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
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "episodes": report["development_batch"]["episodes"],
                "structurally_clean": report["development_batch"]["structurally_clean"],
                "humanlike_quality_improvement": report[
                    "humanlike_quality_improvement"
                ],
                "blind_status": report["blind_review"]["aggregation_status"],
                "training_qualified": report["training_qualified"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
