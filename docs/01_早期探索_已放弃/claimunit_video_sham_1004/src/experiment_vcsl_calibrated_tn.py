"""Verifier-isolated calibration experiment for a label-blind VCSL TN localizer."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import ROOT, write_json
from vcsl_probe import (
    DEFAULT_FEATURE_ROOT,
    segment_precision_recall,
    temporal_network_stability_localize,
)


DEFAULT_SELECTION = (
    ROOT
    / "outputs/selection/vcsl_best_of_tree_seed_20260818_n30/selection_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/vcsl_calibrated_tn_experiment_v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict(metrics: dict[str, float]) -> bool:
    return min(float(metrics[name]) for name in ("precision", "recall", "f1")) >= 0.85


def _similarity(feature_root: Path, pair_id: str) -> np.ndarray:
    query_id, reference_id = pair_id.split("-", 1)
    query = np.load(feature_root / f"{query_id}.npy")
    reference = np.load(feature_root / f"{reference_id}.npy")
    return query @ reference.T


def _evaluate(
    cases: list[dict[str, Any]],
    similarities: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    details = []
    for case in cases:
        pair_id = str(case["pair_id"])
        boxes = temporal_network_stability_localize(
            similarities[pair_id],
            **parameters,
        )
        metrics = segment_precision_recall(boxes, case["oracle_boxes"])
        details.append(
            {
                "pair_id": pair_id,
                "predicted_boxes": boxes,
                "metrics": {name: round(float(value), 6) for name, value in metrics.items()},
                "strict_reachable": _strict(metrics),
            }
        )
    minimum_scores = [min(item["metrics"].values()) for item in details]
    return {
        "parameters": parameters,
        "cases": len(details),
        "strict_reachable": sum(item["strict_reachable"] for item in details),
        "mean_minimum_metric": round(statistics.mean(minimum_scores), 6),
        "median_f1": round(statistics.median(item["metrics"]["f1"] for item in details), 6),
        "details": details,
    }


def run(selection_path: Path, feature_root: Path) -> dict[str, Any]:
    selection = _load(selection_path)
    missed = sorted(
        [case for case in selection["cases"] if not case.get("accepted")],
        key=lambda case: str(case["pair_id"]),
    )
    development = missed[:5]
    holdout = missed[5:]
    similarities = {
        str(case["pair_id"]): _similarity(feature_root, str(case["pair_id"]))
        for case in missed
    }
    variants = [
        {
            "top_k": top_k,
            "max_step": max_step,
            "min_similarity": minimum,
            "min_average_similarity": minimum + 0.1,
            "min_length": 4,
            "max_paths": 10,
        }
        for top_k, max_step, minimum in itertools.product(
            (3, 5),
            (6, 10, 16),
            (0.35, 0.45, 0.55),
        )
    ]
    started = time.perf_counter()
    development_results = [
        _evaluate(development, similarities, parameters) for parameters in variants
    ]
    ranked = sorted(
        development_results,
        key=lambda item: (
            item["strict_reachable"],
            item["mean_minimum_metric"],
            item["median_f1"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    frozen_parameters = dict(selected["parameters"])
    holdout_result = _evaluate(holdout, similarities, frozen_parameters)
    all_missed_result = _evaluate(missed, similarities, frozen_parameters)
    return {
        "experiment": "vcsl_calibrated_temporal_network_v1",
        "label_boundary": {
            "localizer_inputs": "released feature arrays only",
            "oracle_used_during_localizer_execution": False,
            "oracle_used_for_development_parameter_selection": True,
            "oracle_used_for_frozen_holdout_evaluation": True,
            "student_or_teacher_artifact_contains_oracle": False,
        },
        "split_policy": "sort the 9 previously unreachable pair IDs; first 5 development, last 4 frozen holdout",
        "development_case_ids": [case["pair_id"] for case in development],
        "holdout_case_ids": [case["pair_id"] for case in holdout],
        "variant_count": len(variants),
        "selected_parameters": frozen_parameters,
        "development_result": selected,
        "holdout_result": holdout_result,
        "all_previously_missed_result": all_missed_result,
        "top_development_variants": [
            {
                key: value
                for key, value in item.items()
                if key != "details"
            }
            for item in ranked[:5]
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "eligible_for_toolbox_integration": holdout_result["strict_reachable"] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.selection, args.feature_root)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "selected_parameters": report["selected_parameters"],
                "development": {
                    key: value
                    for key, value in report["development_result"].items()
                    if key != "details"
                },
                "holdout": {
                    key: value
                    for key, value in report["holdout_result"].items()
                    if key != "details"
                },
                "eligible_for_toolbox_integration": report[
                    "eligible_for_toolbox_integration"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
