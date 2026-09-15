"""Object-centric yellow-tube motion measurements for VidDiff Knot Tying."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows


DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_yellow_tube_motion_v1.json"
)


def yellow_tube_mask(image: Any) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    height, width = hue.shape
    # The task tube is the yellow-green vertical strip near the workspace
    # center. Spatial restriction excludes orange fixtures at the borders.
    mask = (
        (hue >= 18)
        & (hue <= 48)
        & (saturation >= 45)
        & (value >= 45)
    ).astype(np.uint8)
    mask[:, : int(width * 0.27)] = 0
    mask[:, int(width * 0.73) :] = 0
    mask[: int(height * 0.02)] = 0
    mask[int(height * 0.98) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    return mask.astype(bool)


def _centerline(mask: np.ndarray) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in range(mask.shape[0]):
        columns = np.flatnonzero(mask[row])
        if len(columns) >= 2:
            result[row] = float(np.median(columns))
    return result


def tube_motion_measurement(images: list[Any]) -> dict[str, Any]:
    masks = [yellow_tube_mask(image) for image in images]
    lines = [_centerline(mask) for mask in masks]
    width = float(masks[0].shape[1])
    centerline_steps: list[float] = []
    bending_steps: list[float] = []
    silhouette_changes: list[float] = []
    valid_pairs = 0
    for index in range(1, len(masks)):
        common = sorted(set(lines[index - 1]).intersection(lines[index]))
        if len(common) >= 20:
            previous = np.asarray([lines[index - 1][row] for row in common])
            current = np.asarray([lines[index][row] for row in common])
            shift = current - previous
            centerline_steps.append(float(np.median(np.abs(shift)) / width))
            bending_steps.append(
                float(np.median(np.abs(shift - np.median(shift))) / width)
            )
            valid_pairs += 1
        union = np.logical_or(masks[index - 1], masks[index]).sum()
        if union:
            silhouette_changes.append(
                float(np.logical_xor(masks[index - 1], masks[index]).sum() / union)
            )
    centroid_x = []
    for mask in masks:
        points = np.argwhere(mask)
        if len(points) >= 20:
            centroid_x.append(float(points[:, 1].mean() / width))
    centroid_range = max(centroid_x) - min(centroid_x) if centroid_x else 0.0
    q75_shift = float(np.quantile(centerline_steps, 0.75)) if centerline_steps else 0.0
    q75_bending = float(np.quantile(bending_steps, 0.75)) if bending_steps else 0.0
    q75_silhouette = float(np.quantile(silhouette_changes, 0.75)) if silhouette_changes else 0.0
    # Translation and deformation are the direct physical terms. Silhouette
    # change is downweighted because grasper occlusion also alters the mask.
    score = 0.50 * q75_shift + 0.30 * q75_bending + 0.15 * centroid_range + 0.05 * q75_silhouette
    return {
        "motion_score": round(score, 6),
        "q75_centerline_shift": round(q75_shift, 6),
        "q75_bending_change": round(q75_bending, 6),
        "centroid_range": round(centroid_range, 6),
        "q75_silhouette_change": round(q75_silhouette, 6),
        "valid_temporal_pairs": valid_pairs,
        "mask_coverage_median": round(float(np.median([mask.mean() for mask in masks])), 6),
        "sampled_frames": len(images),
    }


def _rows() -> list[dict[str, Any]]:
    result = []
    for source in available_surgery_rows():
        for key, descriptor in source["differences_annotated"].items():
            if isinstance(descriptor, dict) and descriptor.get("description") == "The tube moves more":
                result.append({**source, "selected_difference_key": key})
    return result


def _predict(a: float, b: float, margin: float) -> str:
    relative = abs(a - b) / max(abs(a), abs(b), 1e-6)
    if relative < margin:
        return "c"
    return "a" if a > b else "b"


def _metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    predictions = [
        (row["oracle_direction"], _predict(row["score_a"], row["score_b"], margin))
        for row in rows
    ]
    labels = ("a", "b", "c")
    recalls = {}
    for label in labels:
        selected = [(truth, prediction) for truth, prediction in predictions if truth == label]
        recalls[label] = sum(truth == prediction for truth, prediction in selected) / max(len(selected), 1)
    return {
        "cases": len(rows),
        "accuracy": round(sum(a == b for a, b in predictions) / max(len(rows), 1), 4),
        "balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
        "recall_by_label": {key: round(value, 4) for key, value in recalls.items()},
        "prediction_counts": dict(Counter(prediction for _truth, prediction in predictions)),
    }


def run(frame_count: int) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    measurements = []
    for row in _rows():
        per_video = []
        for video in row["videos"]:
            key = str(video["path"])
            if key not in cache:
                frames = _sample_frames(
                    VIDDIFF_ROOT / video["path"],
                    video["frames_trim"],
                    frame_count,
                )
                cache[key] = tube_motion_measurement([image for image, _index in frames])
            per_video.append(cache[key])
        measurements.append(
            {
                "sample_key": row["sample_key"],
                "videos": [str(video["path"]) for video in row["videos"]],
                "score_a": per_video[0]["motion_score"],
                "score_b": per_video[1]["motion_score"],
                "video_a": per_video[0],
                "video_b": per_video[1],
                "oracle_direction": row["differences_gt"][row["selected_difference_key"]],
            }
        )
    development = [row for row in measurements if int(row["sample_key"].rsplit("_", 1)[1]) % 2 == 0]
    holdout = [row for row in measurements if row not in development]
    candidates = [round(value, 2) for value in np.arange(0.05, 0.76, 0.05)]
    scored = [
        {
            "margin": margin,
            "metrics": _metrics(development, margin),
        }
        for margin in candidates
    ]
    selected = max(
        scored,
        key=lambda item: (
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["accuracy"],
            -item["margin"],
        ),
    )
    margin = float(selected["margin"])
    for row in measurements:
        row["predicted_direction"] = _predict(row["score_a"], row["score_b"], margin)
        row["direction_correct"] = row["predicted_direction"] == row["oracle_direction"]
    return {
        "experiment": "viddiff_object_centric_yellow_tube_motion_v1",
        "method": "yellow_tube_centerline_translation_and_bending",
        "label_boundary": {
            "oracle_used_during_pixel_measurement": False,
            "development_labels_used_only_to_choose_abstention_margin": True,
            "holdout_labels_used_only_after_margin_selection": True,
            "external_service_calls": 0,
        },
        "selected_abstention_margin": margin,
        "development_metrics": _metrics(development, margin),
        "holdout_metrics": _metrics(holdout, margin),
        "full_metrics": _metrics(measurements, margin),
        "target_cases": [
            row for row in measurements if row["sample_key"] in {"surgery_43", "surgery_55"}
        ],
        "cases": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.frame_count)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "margin": report["selected_abstention_margin"],
                "development": report["development_metrics"],
                "holdout": report["holdout_metrics"],
                "targets": report["target_cases"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

