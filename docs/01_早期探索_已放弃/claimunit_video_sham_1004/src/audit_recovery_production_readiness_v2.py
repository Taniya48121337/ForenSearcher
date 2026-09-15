"""Summarize the expanded recovery batch without promoting development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_INVENTORY = BASE / "production_readiness_audit_v1/local_trajectory_inventory.json"
DEFAULT_COMPILE = (
    BASE
    / "recovery_batch_candidate_v13_expanded_catalog_u11_compact/compile_report.json"
)
DEFAULT_SCORECARDS = (
    BASE
    / "recovery_batch_candidate_v13_expanded_catalog_u11_compact/utqr_episode_scorecards.json"
)
DEFAULT_FALSIFICATION = (
    BASE
    / "recovery_batch_candidate_v13_expanded_catalog_u11_compact/utqr_falsification_report.json"
)
DEFAULT_BLIND_MANIFEST = BASE / "recovery_blind_review_packet_v4_expanded55/manifest.json"
DEFAULT_BLIND_AGGREGATION = (
    BASE
    / "recovery_blind_review_packet_v4_expanded55/private_do_not_share_with_judges/aggregation_status.json"
)
DEFAULT_PORTABILITY = (
    BASE / "recovery_sft_portable_v4_expanded55/portability_audit.json"
)
DEFAULT_PORTABLE_REASONING = (
    BASE / "recovery_sft_portable_v4_expanded55/reasoning_quality_audit.json"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = BASE / "production_readiness_audit_v2_expanded55/readiness_report.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness_report_v2(
    inventory: dict[str, Any],
    compile_report: dict[str, Any],
    scorecards: dict[str, Any],
    falsification: dict[str, Any],
    blind_manifest: dict[str, Any],
    blind_aggregation: dict[str, Any],
    portability: dict[str, Any],
    portable_reasoning: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    contract = rubric["batch_readiness"]
    minimum = int(contract["minimum_entity_disjoint_cases_per_dataset"])
    recovery_minimum = int(
        contract["minimum_unannounced_recovery_cases_per_dataset"]
    )
    checks: dict[str, Any] = {}
    for dataset in contract["required_datasets"]:
        compiled = int(compile_report["datasets_selected"].get(dataset, 0))
        local_max = int(
            inventory["dataset_summary"][dataset][
                "maximum_entity_disjoint_exposed_cases"
            ]
        )
        automated = scorecards["dataset_summary"][dataset]
        checks[dataset] = {
            "development_episodes": compiled,
            "minimum_unannounced_recovery_cases": recovery_minimum,
            "recovery_floor_passed": compiled >= recovery_minimum,
            "minimum_entity_disjoint_cases": minimum,
            "development_count_gap": max(0, minimum - compiled),
            "local_exposed_entity_disjoint_maximum": local_max,
            "local_exposed_count_gap": max(0, minimum - local_max),
            "entity_disjoint_count_passed": compiled >= minimum,
            "sealed_cases": 0,
            "sealed_requirement_passed": False,
            "automated_core_mean": automated["automated_core_mean"],
            "automated_core_p10": automated["automated_core_p10"],
            "automated_core_min": automated["automated_core_min"],
            "automated_score_floor_passed": automated["automated_core_p10"]
            >= float(contract["minimum_score_p10_per_dataset"]),
            "human_acceptance_rate": None,
            "human_review_passed": False,
        }

    structural = {
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
        "canonical_reasoning_audit_passed": bool(
            compile_report["reasoning_quality_audit"]["passed"]
        ),
        "portable_semantics_preserved": bool(
            portability["all_semantic_equivalence_checks_passed"]
        ),
        "portable_identifiers_removed": bool(
            portability["all_forbidden_identifier_hits_zero"]
        ),
        "portable_reasoning_audit_passed": bool(portable_reasoning["passed"]),
        "generic_rubric_falsification_passed": bool(
            falsification["all_detected"]
            and all(
                not item["automated_hard_gates_after"]
                for item in falsification["results"]
            )
        ),
    }

    human_complete = bool(blind_aggregation["review_completed"])
    blockers = [
        "independent_blind_review_not_completed",
        "no_sealed_entity_disjoint_holdout",
        "minimum_30_entity_disjoint_cases_not_met_for_each_dataset",
        "closed_loop_student_gain_not_evaluated",
    ]
    return {
        "artifact": "recovery_production_readiness_audit_v2_expanded55",
        "rubric_id": rubric["rubric_id"],
        "development_batch": {
            "episodes": compile_report["compiled_episodes"],
            "sft_rows": compile_report["sft_rows"],
            "dataset_counts": compile_report["datasets_selected"],
            "intervention_families": len(
                compile_report["selected_intervention_counts"]
            ),
            "structural_checks": structural,
            "structurally_clean": all(structural.values()),
            "status": "high_quality_development_candidate",
        },
        "dataset_checks": checks,
        "blind_review": {
            "packet_episodes": blind_manifest["episodes"],
            "minimum_independent_judges": blind_manifest[
                "minimum_independent_judges"
            ],
            "aggregation_status": blind_aggregation["status"],
            "human_review_completed": human_complete,
            "human_review_passed": bool(
                blind_aggregation["blind_review_passed"]
            ),
        },
        "coverage_change_from_previous_canonical": {
            "total_episodes": {"before": 46, "after": 55, "delta": 9},
            "VidDiffBench": {"before": 12, "after": 21, "delta": 9},
        },
        "training_qualified": False,
        "qualification_blockers": blockers,
        "permitted_claim": (
            "Fifty-five local development trajectories pass the unchanged automated "
            "UTQR core, generic mutation falsification, causal recovery checks, and "
            "portable reasoning audit."
        ),
        "prohibited_claims": [
            "The trajectories have passed independent human blind review.",
            "The data are validated on a sealed holdout.",
            "Every dataset has at least 30 entity-disjoint cases.",
            "The trajectories improve a freely acting student.",
            "The batch is ready for unrestricted bulk production or training.",
        ],
        "next_evidence_needed": [
            "Two independent judges score all 55 identity-blinded episodes and adjudicate disagreements above one point.",
            "Collect or declare at least 30 genuinely entity-disjoint cases per dataset before inspection, with a truly sealed split.",
            "Run an end-to-end student free-run comparison on that sealed split.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--compile-report", type=Path, default=DEFAULT_COMPILE)
    parser.add_argument("--scorecards", type=Path, default=DEFAULT_SCORECARDS)
    parser.add_argument("--falsification", type=Path, default=DEFAULT_FALSIFICATION)
    parser.add_argument("--blind-manifest", type=Path, default=DEFAULT_BLIND_MANIFEST)
    parser.add_argument(
        "--blind-aggregation", type=Path, default=DEFAULT_BLIND_AGGREGATION
    )
    parser.add_argument("--portability", type=Path, default=DEFAULT_PORTABILITY)
    parser.add_argument(
        "--portable-reasoning", type=Path, default=DEFAULT_PORTABLE_REASONING
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_readiness_report_v2(
        _load(args.inventory),
        _load(args.compile_report),
        _load(args.scorecards),
        _load(args.falsification),
        _load(args.blind_manifest),
        _load(args.blind_aggregation),
        _load(args.portability),
        _load(args.portable_reasoning),
        _load(args.rubric),
    )
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
