"""Evaluate label-blind VCSL routing with a strictly local verifier.

This diagnostic never calls a teacher. Oracle boxes are used only after every
localizer has executed, and are written under an explicit verifier-only field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from vcsl_batch import DEFAULT_FEATURE_ROOT, LOCALIZER_SPECS, VCSLFeatureEnvironment
from vcsl_probe import auto_route_localize, segment_precision_recall, select_cases


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "utqr_v7_authorized_batch_20260824"
    / "diagnostics"
    / "vcsl_router_verifier_only.json"
)


def audit(
    *,
    seed: int,
    sample_size: int,
    case_indices: list[int],
    feature_root: Path,
) -> dict[str, Any]:
    sampled = select_cases(seed, sample_size)
    rows: list[dict[str, Any]] = []
    for case_index in case_indices:
        case = sampled[case_index]
        per_variant: dict[str, Any] = {}
        for variant, spec in LOCALIZER_SPECS.items():
            try:
                environment = VCSLFeatureEnvironment(
                    case=case,
                    feature_root=feature_root,
                    routing_variant=variant,
                )
                environment.execute(
                    "inspect_pair_features",
                    {"query_ref": "query", "reference_ref": "reference"},
                )
                result = environment.execute(
                    spec["tool_name"],
                    {
                        "query_ref": "query",
                        "reference_ref": "reference",
                        "method": spec["method"],
                    },
                )
                boxes = result["result"]["copied_segments_frames"]
                per_variant[variant] = {
                    "method": spec["method"],
                    "status": "executed",
                    "box_count": len(boxes),
                    "student_visible_observation": result["result"],
                    "verifier_only_metrics": {
                        key: round(value, 6)
                        for key, value in segment_precision_recall(
                            boxes, case["oracle_boxes"]
                        ).items()
                    },
                }
            except (ValueError, RuntimeError) as error:
                per_variant[variant] = {
                    "method": spec["method"],
                    "status": "execution_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "verifier_only_metrics": None,
                }
        auto_variant, auto_boxes, diagnostics = auto_route_localize(
            VCSLFeatureEnvironment(
                case=case, feature_root=feature_root, routing_variant="dp"
            ).similarity
        )
        executed_variants = [
            name
            for name, row in per_variant.items()
            if row["verifier_only_metrics"] is not None
        ]
        best_variant = max(
            executed_variants,
            key=lambda name: per_variant[name]["verifier_only_metrics"]["f1"],
        )
        rows.append(
            {
                "case_index": case_index,
                "pair_id": case["pair_id"],
                "auto_variant": auto_variant,
                "auto_diagnostics": diagnostics,
                "auto_verifier_only_metrics": {
                    key: round(value, 6)
                    for key, value in segment_precision_recall(
                        auto_boxes, case["oracle_boxes"]
                    ).items()
                },
                "best_verifier_only_variant": best_variant,
                "variants": per_variant,
            }
        )
    return {
        "artifact": "vcsl_router_holdout_audit_v1",
        "teacher_calls": 0,
        "oracle_visible_to_router": False,
        "oracle_used_only_after_localizer_execution": True,
        "not_training_data": True,
        "seed": seed,
        "sample_size": sample_size,
        "case_indices": case_indices,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--case-indices", nargs="+", type=int, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(
        seed=args.seed,
        sample_size=args.sample_size,
        case_indices=args.case_indices,
        feature_root=args.feature_root,
    )
    write_json(args.output, report)
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
