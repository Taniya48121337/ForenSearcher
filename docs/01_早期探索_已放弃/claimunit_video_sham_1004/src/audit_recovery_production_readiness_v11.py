"""Extend readiness evidence with the executed causal intervention lattice."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_OUTPUT = (
    BASE
    / "production_readiness_audit_v11_causal_frontier1098"
    / "readiness_report.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    report = copy.deepcopy(
        _load(
            BASE
            / "production_readiness_audit_v10_utqr12_step_value55"
            / "readiness_report.json"
        )
    )
    lattice_root = BASE / "recovery_intervention_lattice_batch_v1"
    frontier_root = BASE / "recovery_causal_frontier_training_mix_v1"
    contrast_root = BASE / "recovery_causal_frontier_semantic_contrast_v1"
    portable_root = BASE / "recovery_causal_frontier_portable_v3_case_equivalence"
    lattice_compile = _load(lattice_root / "compile_report.json")
    lattice_scores = _load(lattice_root / "utqr12_episode_scorecards.json")
    lattice_steps = _load(lattice_root / "step_value_audit.json")
    selection = _load(frontier_root / "selection_report.json")
    sampling = _load(frontier_root / "sampling_report.json")
    policy = _load(frontier_root / "policy_information_audit.json")
    contrast = _load(contrast_root / "contrast_report.json")
    contrast_integrity = _load(contrast_root / "contrast_integrity_audit.json")
    portable = _load(portable_root / "portability_audit.json")
    portable_reasoning = _load(portable_root / "reasoning_quality_audit.json")
    portable_contrast = _load(portable_root / "contrast_integrity_audit.json")

    report["artifact"] = "recovery_production_readiness_audit_v11_causal_frontier1098"
    report["causal_intervention_lattice"] = {
        "candidate_jobs": lattice_compile["jobs_selected"],
        "executed_admitted_episodes": lattice_compile["compiled_episodes"],
        "failed_variants_preserved": lattice_compile["status_counts"].get(
            "compile_failed", 0
        ),
        "admission_rate": round(
            lattice_compile["compiled_episodes"] / lattice_compile["jobs_selected"],
            6,
        ),
        "datasets": lattice_scores["dataset_summary"],
        "terminal_success": lattice_compile["terminal_success"],
        "counterfactuals_passed": lattice_compile["counterfactuals_passed"],
        "step_value_episodes_passed": lattice_steps["episodes_passed"],
        "step_value_turns": lattice_steps["turns"],
        "student_visible_leakage_episodes": lattice_compile[
            "student_visible_leakage_episodes"
        ],
        "scope": (
            "Development SFT augmentation only; variants share the 55 clean case "
            "checkpoints and are not additional entity-disjoint evaluation cases."
        ),
    }
    report["causal_frontier_sft"] = {
        "complete_backbone_rows": selection["backbone_rows"],
        "executed_lattice_rows_considered": selection["lattice_candidate_rows"],
        "novel_lattice_rows_added": selection["lattice_rows_added"],
        "training_rows": selection["output_rows"],
        "exact_duplicate_fraction": selection["reasoning_quality"]["metrics"][
            "exact_duplicate_fraction"
        ],
        "normalized_template_duplicate_fraction": selection["reasoning_quality"][
            "metrics"
        ]["normalized_template_duplicate_fraction"],
        "reasoning_quality_passed": selection["reasoning_quality"]["passed"],
        "dataset_weighted_mass": sampling["weighted_mass_by_dataset"],
        "case_mass_range_by_dataset": sampling[
            "weighted_case_mass_range_by_dataset"
        ],
        "effective_sample_fraction": sampling["weight_distribution"][
            "effective_sample_fraction"
        ],
        "whole_calls_unique_from_control_plane": policy["overall"]["counts"][
            "whole_call_unique_from_control_plane"
        ],
        "compiled_targets_visible": policy["overall"]["counts"][
            "exact_compiled_target_visible"
        ],
        "choice_bearing_turns": policy["overall"]["counts"][
            "choice_bearing_supervision"
        ],
    }
    report["causal_frontier_contrast"] = {
        "groups": contrast["contrast_groups"],
        "same_case_cross_variant_stop_challenge_groups": contrast[
            "same_case_cross_variant_stop_vs_challenge_groups"
        ],
        "covered_rows": contrast["contrast_covered_rows"],
        "covered_row_fraction": contrast["contrast_covered_row_fraction"],
        "minimum_required_fraction": 0.75,
        "canonical_integrity_passed": contrast_integrity["passed"],
        "portable_integrity_passed": portable_contrast["passed"],
        "messages_tools_weights_unchanged": all(
            contrast[key]
            for key in (
                "messages_unchanged",
                "tools_unchanged",
                "sampling_weights_unchanged",
            )
        ),
    }
    report["causal_frontier_portable"] = {
        "rows": portable["rows"],
        "semantic_equivalence_passed": portable[
            "all_semantic_equivalence_checks_passed"
        ],
        "forbidden_identifier_hits": portable["forbidden_identifier_hits"],
        "case_equivalence_classes_preserved": portable[
            "case_equivalence_classes_preserved"
        ],
        "reasoning_quality_passed": portable_reasoning["passed"],
        "contrast_integrity_passed": portable_contrast["passed"],
    }
    report["training_qualified"] = False
    report["qualification_blockers"] = [
        "independent_blind_review_not_completed",
        "no_sealed_entity_disjoint_holdout",
        "minimum_30_entity_disjoint_cases_not_met_for_each_dataset",
        "closed_loop_student_gain_not_evaluated",
    ]
    report["permitted_claim"] = (
        "A 55-case evidence-first backbone was expanded through 207 local causal "
        "intervention attempts; 196 executed variants were admitted, and a compact "
        "1,098-row case-balanced causal-frontier SFT view passes local reasoning, "
        "control-plane, contrast, and portable audits. This is training-data evidence, "
        "not proof of student gain or sealed generalization."
    )
    report["prohibited_claims"] = sorted(
        set(report.get("prohibited_claims") or [])
        | {
            "The 196 variants are 196 entity-disjoint evaluation cases.",
            "Passing teacher trajectory audits proves a student will use evidence semantics.",
            "The causal-frontier package is production-qualified before blind review and sealed free-run evaluation.",
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
                "lattice": report["causal_intervention_lattice"],
                "frontier": report["causal_frontier_sft"],
                "contrast": report["causal_frontier_contrast"],
                "portable": report["causal_frontier_portable"],
                "training_qualified": report["training_qualified"],
                "qualification_blockers": report["qualification_blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
