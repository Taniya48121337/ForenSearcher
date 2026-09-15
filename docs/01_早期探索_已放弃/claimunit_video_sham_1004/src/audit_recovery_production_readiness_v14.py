"""Attach the loss-preserving contrast-aware trainer schedule to readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v13 import build_report as build_v13
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
PACKAGE = BASE / "recovery_causal_frontier_trainer_ready_v1"
DEFAULT_OUTPUT = (
    BASE
    / "production_readiness_audit_v14_trainer_ready_schedule"
    / "readiness_report.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    report = build_v13()
    manifest = _load(PACKAGE / "manifest.json")
    audit = _load(PACKAGE / "schedule_audit.json")
    smoke = _load(PACKAGE / "materialization_smoke_audit.json")
    report["artifact"] = "recovery_production_readiness_audit_v14_trainer_ready_schedule"
    report["contrast_aware_trainer_schedule"] = {
        "source_rows": manifest["source_rows"],
        "epochs": manifest["epochs"],
        "batch_size": manifest["batch_size"],
        "schedule_shards": manifest["schedule_shards"],
        "contrast_groups": manifest["contrast_groups"],
        "batches_per_epoch": audit["batches_per_epoch"],
        "occurrences_per_epoch": audit["occurrences_per_epoch"],
        "all_source_rows_have_base_exposure_every_epoch": audit[
            "all_source_rows_have_base_exposure_every_epoch"
        ],
        "all_groups_co_located_every_epoch": audit[
            "all_groups_co_located_every_epoch"
        ],
        "all_group_anchors_have_different_targets": audit[
            "all_group_anchors_have_different_targets"
        ],
        "all_epochs_preserve_row_loss_mass": audit[
            "all_epochs_preserve_row_loss_mass"
        ],
        "case_loss_mass_identical_across_epochs": audit[
            "case_loss_mass_identical_across_epochs"
        ],
        "stage_loss_mass_identical_across_epochs": audit[
            "stage_loss_mass_identical_across_epochs"
        ],
        "tool_loss_mass_identical_across_epochs": audit[
            "tool_loss_mass_identical_across_epochs"
        ],
        "maximum_single_dataset_share_in_any_batch": audit[
            "maximum_single_dataset_share_in_any_batch"
        ],
        "anchor_member_coverage_over_8_epochs_minimum": audit[
            "anchor_member_coverage_minimum"
        ],
        "anchor_member_coverage_over_8_epochs_mean": audit[
            "anchor_member_coverage_mean"
        ],
        "materialization_smoke_rows": smoke["rows"],
        "materialized_messages_unchanged": smoke["messages_unchanged"],
        "materialized_tools_unchanged": smoke["tools_unchanged"],
        "materialized_metadata_only_schedule_change": smoke[
            "metadata_unchanged_except_schedule"
        ],
        "calibration_rows_included": smoke[
            "negative_or_calibration_rows_included"
        ],
        "training_executed": manifest["training_executed"],
        "scope": (
            "Offline trainer-input organization only. Co-location is not a contrastive "
            "objective and has no student-gain interpretation before B5."
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
        "The 1,098-row portable causal frontier now has an executable eight-epoch, "
        "48-shard trainer schedule. Every row receives base exposure, every semantic "
        "group receives a different-target co-located pair, and per-row loss mass is "
        "preserved. This establishes loader/schedule readiness only; no training or "
        "student benefit has been measured."
    )
    report["prohibited_claims"] = sorted(
        set(report.get("prohibited_claims") or [])
        | {
            "Contrast-group co-location is equivalent to a contrastive training objective.",
            "A loss-preserving schedule proves student evidence reasoning improves.",
            "Materializing one shard means an SFT run was completed.",
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
                "trainer_schedule": report["contrast_aware_trainer_schedule"],
                "training_qualified": report["training_qualified"],
                "qualification_blockers": report["qualification_blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

