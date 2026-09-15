"""Explore a label-blind connected-component consensus VCSL localizer."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from pipeline import ROOT, write_json
from vcsl_probe import DEFAULT_FEATURE_ROOT, _box_iou, segment_precision_recall


DEFAULT_SELECTION = (
    ROOT
    / "outputs/selection/vcsl_best_of_tree_seed_20260818_n30/selection_audit.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/vcsl_component_consensus_experiment_v1.json"
)


def component_consensus_localize(
    similarity: np.ndarray,
    *,
    percentile: float,
    minimum_similarity: float,
    dilation_iterations: int,
    minimum_points: int,
    minimum_correlation: float,
    max_iou: float = 0.4,
    max_boxes: int = 12,
) -> list[list[int]]:
    threshold = max(float(np.percentile(similarity, percentile)), minimum_similarity)
    strong = similarity >= threshold
    connected = ndimage.binary_dilation(
        strong,
        structure=np.ones((3, 3), dtype=bool),
        iterations=dilation_iterations,
    )
    labels, count = ndimage.label(connected, structure=np.ones((3, 3), dtype=bool))
    proposals: list[tuple[float, list[int]]] = []
    for component_id in range(1, count + 1):
        points = np.argwhere(strong & (labels == component_id))
        if len(points) < minimum_points:
            continue
        query = points[:, 0]
        reference = points[:, 1]
        query_span = int(query.max() - query.min())
        reference_span = int(reference.max() - reference.min())
        if min(query_span, reference_span) < 4:
            continue
        correlation = (
            float(np.corrcoef(query, reference)[0, 1])
            if len(np.unique(query)) > 1 and len(np.unique(reference)) > 1
            else 0.0
        )
        if not np.isfinite(correlation) or correlation < minimum_correlation:
            continue
        box = [
            int(query.min()),
            int(reference.min()),
            int(query.max()),
            int(reference.max()),
        ]
        area = max(query_span * reference_span, 1)
        density = len(points) / area
        mean_similarity = float(similarity[query, reference].mean())
        score = mean_similarity * len(points) * max(correlation, 0.0) * np.sqrt(density)
        proposals.append((float(score), box))
    proposals.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[int]] = []
    for _score, box in proposals:
        overlaps = _box_iou(
            np.asarray([box], dtype=np.float32),
            np.asarray(selected, dtype=np.float32),
        )
        if np.any(overlaps > max_iou):
            continue
        selected.append(box)
        if len(selected) >= max_boxes:
            break
    return selected


def _strict(metrics: dict[str, float]) -> bool:
    return min(float(metrics[name]) for name in ("precision", "recall", "f1")) >= 0.85


def _similarity(feature_root: Path, pair_id: str) -> np.ndarray:
    query_id, reference_id = pair_id.split("-", 1)
    return (
        np.load(feature_root / f"{query_id}.npy")
        @ np.load(feature_root / f"{reference_id}.npy").T
    )


def _evaluate(
    cases: list[dict[str, Any]],
    similarities: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    details = []
    for case in cases:
        pair_id = str(case["pair_id"])
        boxes = component_consensus_localize(similarities[pair_id], **parameters)
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
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    missed = sorted(
        [case for case in selection["cases"] if not case.get("accepted")],
        key=lambda case: str(case["pair_id"]),
    )
    development, holdout = missed[:5], missed[5:]
    similarities = {
        str(case["pair_id"]): _similarity(feature_root, str(case["pair_id"]))
        for case in missed
    }
    variants = [
        {
            "percentile": percentile,
            "minimum_similarity": minimum_similarity,
            "dilation_iterations": dilation,
            "minimum_points": 6,
            "minimum_correlation": correlation,
            "max_iou": 0.4,
            "max_boxes": 12,
        }
        for percentile, minimum_similarity, dilation, correlation in itertools.product(
            (97.5, 98.5, 99.0),
            (0.25, 0.35),
            (1, 2, 3),
            (0.3, 0.6),
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
    frozen = dict(selected["parameters"])
    holdout_result = _evaluate(holdout, similarities, frozen)
    all_result = _evaluate(missed, similarities, frozen)
    return {
        "experiment": "vcsl_component_consensus_v1",
        "label_boundary": {
            "localizer_inputs": "released feature arrays only",
            "oracle_used_during_localizer_execution": False,
            "oracle_used_for_development_parameter_selection": True,
            "oracle_used_for_frozen_holdout_evaluation": True,
            "student_or_teacher_artifact_contains_oracle": False,
        },
        "split_policy": "same fixed sorted 5 development / 4 holdout split as calibrated TN",
        "development_case_ids": [case["pair_id"] for case in development],
        "holdout_case_ids": [case["pair_id"] for case in holdout],
        "variant_count": len(variants),
        "selected_parameters": frozen,
        "development_result": selected,
        "holdout_result": holdout_result,
        "all_previously_missed_result": all_result,
        "top_development_variants": [
            {key: value for key, value in item.items() if key != "details"}
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
