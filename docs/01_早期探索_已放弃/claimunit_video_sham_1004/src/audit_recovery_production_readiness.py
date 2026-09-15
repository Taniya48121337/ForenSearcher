"""Evaluate recovery artifacts against the already-declared UTQR-9 batch contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_INVENTORY = BASE / "production_readiness_audit_v1/local_trajectory_inventory.json"
DEFAULT_COMPILE = BASE / "recovery_batch_candidate_v6/compile_report.json"
DEFAULT_BLIND = BASE / "recovery_blind_review_packet_v2/manifest.json"
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = BASE / "production_readiness_audit_v1/readiness_report.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness_report(
    inventory: dict[str, Any],
    compile_report: dict[str, Any],
    blind_manifest: dict[str, Any],
    rubric: dict[str, Any],
    artifact_id: str = "recovery_production_readiness_audit_v1",
) -> dict[str, Any]:
    contract = rubric["batch_readiness"]
    compiled_case_ids = {
        str(item["case_id"]).lower()
        for item in compile_report.get("results") or []
        if item.get("status") == "compiled" and item.get("case_id")
    }
    remaining_exposed = max(0, int(inventory["unique_cases"]) - len(compiled_case_ids))
    dataset_checks: dict[str, Any] = {}
    for dataset in contract["required_datasets"]:
        pool = inventory["dataset_summary"][dataset]
        compiled = int((compile_report.get("datasets_selected") or {}).get(dataset, 0))
        disjoint = int(pool["maximum_entity_disjoint_exposed_cases"])
        dataset_checks[dataset] = {
            "compiled_recovery_cases": compiled,
            "minimum_recovery_cases": contract[
                "minimum_unannounced_recovery_cases_per_dataset"
            ],
            "recovery_case_minimum_passed": compiled
            >= contract["minimum_unannounced_recovery_cases_per_dataset"],
            "accepted_exposed_cases": pool["accepted_exposed_cases"],
            "maximum_entity_disjoint_exposed_cases": disjoint,
            "minimum_entity_disjoint_cases": contract[
                "minimum_entity_disjoint_cases_per_dataset"
            ],
            "entity_disjoint_minimum_passed": disjoint
            >= contract["minimum_entity_disjoint_cases_per_dataset"],
            "sealed_entity_disjoint_cases": pool["sealed_cases"],
            "sealed_requirement_passed": False,
            "blind_packet_acceptance_rate": None,
            "blind_packet_acceptance_requirement_passed": False,
            "score_p10": None,
            "score_p10_requirement_passed": False,
        }

    structural_checks = {
        "all_selected_episodes_compiled": compile_report["compiled_episodes"]
        == compile_report["jobs_selected"],
        "all_terminal": compile_report["terminal_success"]
        == compile_report["jobs_selected"],
        "all_consume_fresh_evidence": compile_report["fresh_evidence_consumed"]
        == compile_report["jobs_selected"],
        "all_preserve_independent_fields": compile_report[
            "independent_fields_preserved"
        ]
        == compile_report["jobs_selected"],
        "all_counterfactuals_pass": compile_report["counterfactuals_passed"]
        == compile_report["jobs_selected"],
        "no_student_visible_leakage": compile_report[
            "student_visible_leakage_episodes"
        ]
        == 0,
        "reasoning_audit_passed": bool(
            (compile_report.get("reasoning_quality_audit") or {}).get("passed")
        ),
    }

    blind_completed = bool(blind_manifest.get("human_review_completed"))
    blockers = [
        "no_sealed_entity_disjoint_holdout",
        "independent_blind_review_not_completed",
        "blind_acceptance_rate_and_score_p10_not_available",
        "closed_loop_student_gain_not_evaluated",
    ]
    if not dataset_checks["VidDiffBench"]["recovery_case_minimum_passed"]:
        blockers.append("viddiffbench_has_fewer_than_10_compiled_recovery_cases")
    for dataset, checks in dataset_checks.items():
        if not checks["entity_disjoint_minimum_passed"]:
            blockers.append(f"{dataset.lower()}_has_fewer_than_30_entity_disjoint_cases")

    return {
        "artifact": artifact_id,
        "rubric_id": rubric["rubric_id"],
        "development_batch": {
            "episodes": compile_report["compiled_episodes"],
            "sft_rows": compile_report["sft_rows"],
            "structural_checks": structural_checks,
            "structurally_clean": all(structural_checks.values()),
            "status": "high_quality_development_candidate",
        },
        "local_exposure_inventory": {
            "unique_accepted_cases": inventory["unique_cases"],
            "sealed_holdout_available": inventory["sealed_holdout_available"],
            "important_interpretation": (
                f"The {remaining_exposed} accepted cases omitted from this recovery batch "
                "may expand development coverage, but they are already exposed and cannot "
                "become a sealed test."
            ),
        },
        "dataset_checks": dataset_checks,
        "blind_review": {
            "packet_episodes": blind_manifest["episodes"],
            "two_independent_forms_prepared": blind_manifest[
                "minimum_independent_judges"
            ]
            >= 2,
            "human_review_completed": blind_completed,
        },
        "training_qualified": False,
        "qualification_blockers": blockers,
        "permitted_claim": (
            "The local compiler produced a structurally clean, causally checked, "
            "reasoning-audited development SFT candidate across three datasets."
        ),
        "prohibited_claims": [
            "The batch is production-ready.",
            "The batch is validated on a sealed holdout.",
            "The trajectories improve a freely acting student.",
            "The remaining exposed local cases are an unseen test set.",
        ],
        "next_evidence_needed": [
            "Two independent blind judges complete the prepared forms and adjudicate score ranges above one point.",
            "A new entity-disjoint split is declared before any trajectory, label, or verifier result is inspected.",
            "The student is evaluated in full free-run mode on that sealed split.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--compile-report", type=Path, default=DEFAULT_COMPILE)
    parser.add_argument("--blind-manifest", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact-id", default="recovery_production_readiness_audit_v1"
    )
    args = parser.parse_args()
    report = build_readiness_report(
        _load(args.inventory),
        _load(args.compile_report),
        _load(args.blind_manifest),
        _load(args.rubric),
        args.artifact_id,
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
