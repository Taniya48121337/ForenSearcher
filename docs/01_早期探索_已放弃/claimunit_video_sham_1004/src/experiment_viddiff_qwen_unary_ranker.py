"""Evaluate a local unary VLM scorer over every pair of one VidDiff predicate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiment_viddiff_local_qwen3vl import DEFAULT_MODEL, LocalQwenPredicateScorer
from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, available_surgery_rows, build_mosaic


DEFAULT_STATEMENT = "The needle is grasped closer to the tip"
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_qwen_unary_tip_grasp_v1.json"
)
DEFAULT_ASSET_ROOT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_qwen_unary_tip_grasp_assets_v1"
)


def _rows(statement: str) -> list[dict[str, Any]]:
    selected = []
    for source in available_surgery_rows():
        for key, descriptor in source["differences_annotated"].items():
            if isinstance(descriptor, dict) and descriptor.get("description") == statement:
                selected.append({**source, "selected_difference_key": key})
    return selected


def _predict(a: float, b: float, margin: float) -> str:
    if abs(a - b) < margin:
        return "c"
    return "a" if a > b else "b"


def _metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    pairs = [
        (row["oracle_direction"], _predict(row["score_a"], row["score_b"], margin))
        for row in rows
    ]
    labels = ("a", "b", "c")
    recalls = {}
    f1s = {}
    for label in labels:
        tp = sum(truth == label and prediction == label for truth, prediction in pairs)
        fn = sum(truth == label and prediction != label for truth, prediction in pairs)
        fp = sum(truth != label and prediction == label for truth, prediction in pairs)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls[label] = recall
        f1s[label] = 2 * precision * recall / max(precision + recall, 1e-12)
    non_c = [(truth, prediction) for truth, prediction in pairs if truth != "c"]
    truth_counts = Counter(truth for truth, _prediction in pairs)
    return {
        "cases": len(rows),
        "accuracy": round(sum(a == b for a, b in pairs) / max(len(rows), 1), 4),
        "balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
        "macro_f1": round(float(np.mean(list(f1s.values()))), 4),
        "non_c_recall": round(
            sum(a == b for a, b in non_c) / max(len(non_c), 1),
            4,
        ),
        "majority_baseline_accuracy": round(
            max(truth_counts.values(), default=0) / max(len(rows), 1),
            4,
        ),
        "recall_by_label": {key: round(value, 4) for key, value in recalls.items()},
        "prediction_counts": dict(Counter(prediction for _truth, prediction in pairs)),
    }


def run(
    *,
    model_path: Path,
    statement: str,
    frame_count: int,
    margin: float,
    asset_root: Path,
) -> dict[str, Any]:
    rows = _rows(statement)
    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        for video in row["videos"]:
            videos.setdefault(str(video["path"]), video)
    scorer = LocalQwenPredicateScorer(model_path)
    video_scores: dict[str, dict[str, Any]] = {}
    for index, (key, video) in enumerate(sorted(videos.items()), start=1):
        safe_name = Path(key).stem
        mosaic_path = asset_root / f"{safe_name}_uniform.jpg"
        build_mosaic(
            video_path=VIDDIFF_ROOT / video["path"],
            frames_trim=video["frames_trim"],
            fps=float(video["fps_original"]),
            role="single_video",
            output_path=mosaic_path,
            count=frame_count,
        )
        video_scores[key] = scorer.score(
            image_path=mosaic_path,
            action_description=(
                "The subject picks up a curved surgical needle and passes it through four small metal hoops "
                "from right to left."
            ),
            statement=statement,
            view="uniform",
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(videos)}",
                    "video": safe_name,
                    "support": video_scores[key]["support_score"],
                    "observability": video_scores[key]["observability_score"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    measurements = []
    for row in rows:
        left_key, right_key = (str(video["path"]) for video in row["videos"])
        left, right = video_scores[left_key], video_scores[right_key]
        predicted = (
            "c"
            if min(float(left["observability_score"]), float(right["observability_score"])) < 45
            else _predict(float(left["support_score"]), float(right["support_score"]), margin)
        )
        oracle = str(row["differences_gt"][row["selected_difference_key"]])
        measurements.append(
            {
                "sample_key": row["sample_key"],
                "videos": [left_key, right_key],
                "score_a": float(left["support_score"]),
                "score_b": float(right["support_score"]),
                "observability_a": float(left["observability_score"]),
                "observability_b": float(right["observability_score"]),
                "predicted_direction": predicted,
                "oracle_direction": oracle,
                "direction_correct": predicted == oracle,
            }
        )
    # Metrics call uses the same fixed label-blind rule; replace predictions for
    # low-observability pairs by setting equal scores in the metric view.
    metric_rows = []
    for row in measurements:
        metric_row = dict(row)
        if min(row["observability_a"], row["observability_b"]) < 45:
            metric_row["score_a"] = metric_row["score_b"]
        metric_rows.append(metric_row)
    return {
        "experiment": "viddiff_local_qwen3vl_unary_video_ranker_v1",
        "statement": statement,
        "label_boundary": {
            "video_scores_computed_before_labels_are_read": True,
            "oracle_used_only_for_post_measurement_metrics": True,
            "external_service_calls": 0,
        },
        "decision_policy": {
            "fixed_absolute_support_margin": margin,
            "minimum_observability": 45,
            "thresholds_tuned_on_this_predicate": False,
        },
        "summary": _metrics(metric_rows, margin),
        "target_surgery_123": next(
            row for row in measurements if row["sample_key"] == "surgery_123"
        ),
        "video_scores": video_scores,
        "cases": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--statement", default=DEFAULT_STATEMENT)
    parser.add_argument("--frame-count", type=int, default=12)
    parser.add_argument("--margin", type=float, default=15.0)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(
        model_path=args.model,
        statement=args.statement,
        frame_count=args.frame_count,
        margin=args.margin,
        asset_root=args.asset_root,
    )
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

