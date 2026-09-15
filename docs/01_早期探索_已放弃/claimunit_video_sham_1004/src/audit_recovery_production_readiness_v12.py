"""Attach the strict blind-delta review status to causal-frontier readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_recovery_production_readiness_v11 import build_report as build_v11
from pipeline import ROOT, write_json


BASE = ROOT / "outputs/student_external_contract_v1"
PACKET = BASE / "recovery_lattice_blind_delta_packet_v1"
DEFAULT_OUTPUT = (
    BASE
    / "production_readiness_audit_v12_blind_delta105"
    / "readiness_report.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report() -> dict[str, Any]:
    report = build_v11()
    manifest = _load(PACKET / "manifest.json")
    sampling = _load(
        PACKET / "private_do_not_share_with_judges/sampling_report.json"
    )
    representation = _load(
        PACKET
        / "private_do_not_share_with_judges/representation_and_privacy_audit.json"
    )
    aggregation = _load(
        PACKET / "private_do_not_share_with_judges/aggregation_status.json"
    )
    report["artifact"] = "recovery_production_readiness_audit_v12_blind_delta105"
    report["independent_blind_delta_review"] = {
        "status": aggregation["status"],
        "sampled_development_variants": manifest["packets"],
        "per_dataset_private_stratum": manifest["per_dataset_private_stratum"],
        "covered_development_cases_by_dataset": {
            dataset: item["selected_cases"] for dataset, item in sampling.items()
        },
        "intervention_families_covered_by_dataset": {
            dataset: len(item["intervention_quotas"])
            for dataset, item in sampling.items()
        },
        "selection_predeclared_and_nonadaptive": manifest[
            "selection_predeclared_and_nonadaptive"
        ],
        "selection_uses_oracle_verifier_or_automated_score": manifest[
            "selection_uses_oracle_verifier_or_automated_score"
        ],
        "all_state_deltas_reconstructable": representation[
            "all_projected_state_deltas_reconstructable"
        ],
        "public_privacy_audit_passed": representation["privacy_passed"],
        "public_packet_byte_reduction_fraction": representation[
            "byte_reduction_fraction"
        ],
        "utqr_h8_dimensions": manifest["utqr_h8_dimensions"],
        "development_addendum_dimensions": manifest[
            "development_addendum_dimensions"
        ],
        "acceptance_wilson95_lower_bound_minimum": manifest[
            "acceptance_wilson95_lower_bound_minimum"
        ],
        "perfect_sample_wilson95_lower_bound": manifest[
            "perfect_sample_wilson95_lower_bound"
        ],
        "one_failure_wilson95_lower_bound": manifest[
            "one_failure_wilson95_lower_bound"
        ],
        "human_review_completed": aggregation["review_completed"],
        "missing_reviews": len(aggregation["missing_or_invalid_review_ids"]),
        "blind_review_passed": aggregation["blind_review_passed"],
        "scope": (
            "A fixed sample of development intervention variants, not sealed or "
            "entity-disjoint evaluation cases."
        ),
    }
    report["training_qualified"] = False
    report["qualification_blockers"] = [
        "independent_blind_review_not_completed",
        "no_sealed_entity_disjoint_holdout",
        "minimum_30_entity_disjoint_cases_not_met_for_each_dataset",
        "closed_loop_student_gain_not_evaluated",
    ]
    report["permitted_claim"] = (
        "The executable causal-frontier package has a predeclared, reconstructable, "
        "privacy-checked 105-variant development blind-review packet covering all 55 "
        "cases and 11 intervention families. Human scores remain absent, so this is "
        "review readiness rather than blind-review success or student evidence."
    )
    report["prohibited_claims"] = sorted(
        set(report.get("prohibited_claims") or [])
        | {
            "Preparing a two-judge packet means H8 has passed.",
            "The 105 sampled intervention variants are 105 entity-disjoint cases.",
            "A privacy and reconstructability audit substitutes for human semantic review.",
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
                "blind_delta_review": report["independent_blind_delta_review"],
                "training_qualified": report["training_qualified"],
                "qualification_blockers": report["qualification_blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
