"""Local-only geometric probes for under-covered VidDiff surgical predicates.

The measurements are deterministic and label blind. Oracle directions are read
only after both videos have been scored. A deterministic development/holdout
split is used to choose the abstention margin; registration decisions are made
from holdout metrics, not from the full dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows


FAMILIES = {
    "dot_accuracy": "The dot is more accurately hit",
    "dual_grasper_efficiency": "Both graspers are used efficiently.",
}
DEFAULT_OUTPUT = ROOT / "outputs" / "probes" / "viddiff_local_geometry" / "probe_report.json"


def _rgb(image: Any) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def red_dot_centroids(image: Any) -> tuple[list[tuple[float, float]], float]:
    rgb = _rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = (((hue <= 10) | (hue >= 165)) & (sat >= 70) & (val >= 70)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points: list[tuple[float, float]] = []
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (2 <= area <= 240):
            continue
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        cx, cy = centroids[index]
        points.append((float(cx), float(cy)))
    if len(points) < 2:
        return points, float(max(height, width))
    array = np.asarray(points, dtype=np.float32)
    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=-1)
    distances[distances == 0] = np.inf
    spacing = float(np.median(np.min(distances, axis=1)))
    return points, max(spacing, 1.0)


def instrument_tips(image: Any) -> list[tuple[float, float]]:
    rgb = _rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    # Dark shafts are stable in JIGSAWS. Exclude the outer fixture band and
    # retain only components connected to a side/bottom entry region.
    dark = (gray <= 72).astype(np.uint8)
    dark[: int(height * 0.08)] = 0
    dark[int(height * 0.94) :] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, 8)
    tips: list[tuple[float, float]] = []
    min_area = max(30, int(height * width * 0.0012))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        points_y, points_x = np.where(labels == index)
        touches_left = x <= int(width * 0.08)
        touches_right = x + w >= int(width * 0.92)
        touches_bottom = y + h >= int(height * 0.88)
        if not (touches_left or touches_right or touches_bottom):
            continue
        candidates: list[tuple[float, float, float]] = []
        if touches_left:
            position = int(np.argmax(points_x))
            candidates.append((float(points_x[position]), float(points_y[position]), float(points_x[position])))
        if touches_right:
            position = int(np.argmin(points_x))
            candidates.append((float(points_x[position]), float(points_y[position]), float(width - points_x[position])))
        if touches_bottom:
            position = int(np.argmin(points_y))
            candidates.append((float(points_x[position]), float(points_y[position]), float(height - points_y[position])))
        if candidates:
            tip_x, tip_y, _reach = max(candidates, key=lambda item: item[2])
            tips.append((tip_x, tip_y))
    # Merge duplicate detections from a component touching two borders.
    merged: list[tuple[float, float]] = []
    for point in tips:
        if all(np.hypot(point[0] - old[0], point[1] - old[1]) > 12 for old in merged):
            merged.append(point)
    return merged[:3]


def dot_accuracy_score(images: list[Any]) -> dict[str, Any]:
    normalized_distances: list[float] = []
    valid_frames = 0
    for image in images:
        dots, spacing = red_dot_centroids(image)
        tips = instrument_tips(image)
        if len(dots) < 3 or not tips:
            continue
        valid_frames += 1
        dot_array = np.asarray(dots, dtype=np.float32)
        tip_array = np.asarray(tips, dtype=np.float32)
        nearest = np.linalg.norm(tip_array[:, None, :] - dot_array[None, :, :], axis=-1).min()
        normalized_distances.append(float(nearest / spacing))
    if not normalized_distances:
        return {"score": 0.0, "valid_frames": 0, "median_normalized_distance": None}
    # More accurate means smaller distance, so expose a higher-is-better score.
    distance = float(np.quantile(normalized_distances, 0.25))
    return {
        "score": round(1.0 / (1.0 + distance), 6),
        "valid_frames": valid_frames,
        "median_normalized_distance": round(float(np.median(normalized_distances)), 6),
        "best_quartile_normalized_distance": round(distance, 6),
    }


def dual_grasper_score(images: list[Any]) -> dict[str, Any]:
    active = 0
    valid = 0
    previous: list[tuple[float, float]] | None = None
    motion_values: list[float] = []
    for image in images:
        rgb = _rgb(image)
        height, width = rgb.shape[:2]
        tips = instrument_tips(image)
        if len(tips) < 2:
            previous = tips
            continue
        valid += 1
        central = [
            point for point in tips
            if width * 0.18 <= point[0] <= width * 0.82 and height * 0.12 <= point[1] <= height * 0.90
        ]
        separated = len(central) >= 2 and np.hypot(
            central[0][0] - central[1][0], central[0][1] - central[1][1]
        ) >= width * 0.05
        if separated:
            active += 1
        if previous and len(previous) >= 2:
            current = np.asarray(sorted(tips[:2]), dtype=np.float32)
            old = np.asarray(sorted(previous[:2]), dtype=np.float32)
            motion_values.append(float(np.linalg.norm(current - old) / max(np.hypot(height, width), 1.0)))
        previous = tips
    simultaneous_fraction = active / max(valid, 1)
    motion = float(np.median(motion_values)) if motion_values else 0.0
    # Sustained simultaneous workspace presence dominates; small motion breaks
    # ties between passive parked tools and genuinely participating graspers.
    score = simultaneous_fraction + min(motion * 4.0, 0.20)
    return {
        "score": round(score, 6),
        "valid_frames": valid,
        "simultaneous_workspace_fraction": round(simultaneous_fraction, 6),
        "median_tip_motion_normalized": round(motion, 6),
    }


def _cases_for_statement(statement: str) -> list[dict[str, Any]]:
    selected = []
    for source in available_surgery_rows():
        for key, descriptor in source["differences_annotated"].items():
            if isinstance(descriptor, dict) and descriptor.get("description") == statement:
                selected.append({**source, "selected_difference_key": key})
    return selected


def _split(sample_key: str) -> str:
    index = int(sample_key.rsplit("_", 1)[1])
    return "development" if index % 2 == 0 else "holdout"


def _predict(score_a: float, score_b: float, margin: float) -> str:
    relative = abs(score_a - score_b) / max(abs(score_a), abs(score_b), 1e-6)
    if relative < margin:
        return "c"
    return "a" if score_a > score_b else "b"


def _metrics(rows: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    labels = ("a", "b", "c")
    pairs = [(item["oracle_direction"], _predict(item["score_a"], item["score_b"], margin)) for item in rows]
    recalls = {}
    f1s = {}
    for label in labels:
        tp = sum(y == label and p == label for y, p in pairs)
        fn = sum(y == label and p != label for y, p in pairs)
        fp = sum(y != label and p == label for y, p in pairs)
        recalls[label] = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1s[label] = 2 * precision * recalls[label] / max(precision + recalls[label], 1e-12)
    correct = sum(y == p for y, p in pairs)
    non_c = [(y, p) for y, p in pairs if y != "c"]
    counts = Counter(y for y, _p in pairs)
    majority = max(counts.values(), default=0) / max(len(pairs), 1)
    return {
        "cases": len(rows),
        "accuracy": round(correct / max(len(rows), 1), 4),
        "balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
        "macro_f1": round(float(np.mean(list(f1s.values()))), 4),
        "non_c_recall": round(sum(y == p for y, p in non_c) / max(len(non_c), 1), 4),
        "majority_baseline_accuracy": round(majority, 4),
        "recall_by_label": {key: round(value, 4) for key, value in recalls.items()},
        "prediction_counts": dict(Counter(p for _y, p in pairs)),
    }


def run_family(family: str, frame_count: int) -> dict[str, Any]:
    statement = FAMILIES[family]
    scorer: Callable[[list[Any]], dict[str, Any]] = (
        dot_accuracy_score if family == "dot_accuracy" else dual_grasper_score
    )
    measurements = []
    for row in _cases_for_statement(statement):
        video_results = []
        for video in row["videos"]:
            sampled = _sample_frames(VIDDIFF_ROOT / video["path"], video["frames_trim"], frame_count)
            video_results.append(scorer([image for image, _index in sampled]))
        measurements.append(
            {
                "sample_key": row["sample_key"],
                "split": _split(row["sample_key"]),
                "comparison_statement": statement,
                "score_a": video_results[0]["score"],
                "score_b": video_results[1]["score"],
                "video_a": video_results[0],
                "video_b": video_results[1],
                "oracle_direction": row["differences_gt"][row["selected_difference_key"]],
            }
        )
    development = [item for item in measurements if item["split"] == "development"]
    holdout = [item for item in measurements if item["split"] == "holdout"]
    candidates = [round(value, 3) for value in np.linspace(0.05, 0.50, 19)]
    scored = [(margin, _metrics(development, margin)) for margin in candidates]
    # Balanced accuracy is primary; macro F1 and accuracy break ties.
    margin, development_metrics = max(
        scored,
        key=lambda item: (
            item[1]["balanced_accuracy"], item[1]["macro_f1"], item[1]["accuracy"], -item[0]
        ),
    )
    holdout_metrics = _metrics(holdout, margin)
    safe = (
        holdout_metrics["cases"] >= 20
        and holdout_metrics["balanced_accuracy"] >= 0.55
        and holdout_metrics["macro_f1"] >= 0.50
        and holdout_metrics["non_c_recall"] >= 0.50
        and holdout_metrics["accuracy"] > holdout_metrics["majority_baseline_accuracy"]
    )
    for item in measurements:
        item["predicted_direction"] = _predict(item["score_a"], item["score_b"], margin)
        item["direction_correct"] = item["predicted_direction"] == item["oracle_direction"]
    return {
        "family": family,
        "comparison_statement": statement,
        "measurement_policy": "deterministic_local_geometry_then_posthoc_oracle_evaluation",
        "abstention_margin_selected_on": "development_split_even_sample_index",
        "abstention_margin": margin,
        "development_metrics": development_metrics,
        "holdout_metrics": holdout_metrics,
        "swap_consistent_by_construction": True,
        "safe_to_register_tool": safe,
        "cases": measurements,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    families = args.families or list(FAMILIES)
    reports = [run_family(family, args.frame_count) for family in families]
    report = {
        "run_id": "viddiff_local_geometry_probes_v1",
        "external_service_calls": 0,
        "oracle_visibility": "evaluation_only_after_measurement",
        "families": reports,
        "summary": {
            "families_tested": len(reports),
            "families_safe_to_register": [
                item["family"] for item in reports if item["safe_to_register_tool"]
            ],
        },
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES))
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args)
    compact = {
        item["family"]: {
            "margin": item["abstention_margin"],
            "development": item["development_metrics"],
            "holdout": item["holdout_metrics"],
            "safe_to_register_tool": item["safe_to_register_tool"],
        }
        for item in report["families"]
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
