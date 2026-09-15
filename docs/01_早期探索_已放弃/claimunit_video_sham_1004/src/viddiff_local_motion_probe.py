"""Local-only color-track motion proxy for VidDiff tube/thread movement claims."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows, select_cases


DEFAULT_OUTPUT = ROOT / "outputs" / "probes" / "viddiff_local_tube_motion" / "probe_report.json"


def green_thread_mask(image: Any) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # The JIGSAWS thread is a muted green. Relative channel tests are more
    # stable across exposure than a fixed RGB or HSV value.
    mask = (
        (green >= 38)
        & (green - red >= 5)
        & (green - blue >= 2)
        & (green >= red * 1.06)
        & (green >= blue * 1.025)
    )
    # One-pixel thread fragments are expanded locally so small translations
    # survive video compression without introducing an external CV model.
    expanded = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        expanded |= np.roll(mask, shift=(dy, dx), axis=(0, 1))
    return expanded


def motion_measurement(images: list[Any]) -> dict[str, Any]:
    masks = np.stack([green_thread_mask(image) for image in images])
    static = masks.mean(axis=0) >= 0.80
    dynamic = masks & ~static
    changes: list[float] = []
    centroid_steps: list[float] = []
    height, width = masks.shape[1:]
    diagonal = max(float(np.hypot(height, width)), 1.0)
    previous_centroid: np.ndarray | None = None
    for index in range(len(dynamic)):
        points = np.argwhere(dynamic[index])
        centroid = points.mean(axis=0) if len(points) >= 8 else None
        if centroid is not None and previous_centroid is not None:
            centroid_steps.append(float(np.linalg.norm(centroid - previous_centroid) / diagonal))
        if centroid is not None:
            previous_centroid = centroid
        if index:
            union = np.logical_or(dynamic[index - 1], dynamic[index]).sum()
            xor = np.logical_xor(dynamic[index - 1], dynamic[index]).sum()
            changes.append(float(xor / max(union, 1)))
    shape_change = float(np.median(changes)) if changes else 0.0
    centroid_motion = float(np.median(centroid_steps)) if centroid_steps else 0.0
    coverage = float(dynamic.mean())
    # Pixel XOR is dominated by temporary grasper occlusion. The centroid
    # trajectory of the non-static thread mask is the closer proxy to the
    # actual claim that the tube itself moves more.
    score = min(centroid_motion * 12.0, 1.0)
    return {
        "motion_score": round(score, 6),
        "median_shape_change": round(shape_change, 6),
        "median_centroid_step_normalized": round(centroid_motion, 6),
        "dynamic_mask_fraction": round(coverage, 6),
        "sampled_frames": len(images),
    }


def compare_case(row: dict[str, Any], frame_count: int = 24) -> dict[str, Any]:
    measurements = []
    for video in row["videos"]:
        frames = _sample_frames(
            VIDDIFF_ROOT / video["path"],
            video["frames_trim"],
            frame_count,
        )
        measurements.append(motion_measurement([image for image, _index in frames]))
    left = measurements[0]["motion_score"]
    right = measurements[1]["motion_score"]
    relative_margin = abs(left - right) / max(left, right, 1e-6)
    if relative_margin < 0.12:
        preferred = "c"
        rule = "local_thread_motion_insufficient_relative_margin"
    else:
        preferred = "a" if left > right else "b"
        rule = "local_thread_motion_relative_margin"
    return {
        "preferred_video": preferred,
        "decision_rule": rule,
        "confidence": round(min(0.9, 0.45 + relative_margin), 3),
        "relative_margin": round(relative_margin, 6),
        "video_a": measurements[0],
        "video_b": measurements[1],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.all_available_statements:
        selected = []
        for source in available_surgery_rows():
            for key, descriptor in source["differences_annotated"].items():
                if isinstance(descriptor, dict) and "tube moves more" in descriptor.get(
                    "description", ""
                ).lower():
                    selected.append({**source, "selected_difference_key": key})
    else:
        pool = select_cases(args.seed, args.sample_size)
        selected = [pool[index] for index in args.case_indices]
    results = []
    for row in selected:
        statement = row["differences_annotated"][row["selected_difference_key"]]["description"]
        if "tube moves more" not in statement.lower():
            continue
        measurement = compare_case(row, args.frame_count)
        oracle = row["differences_gt"][row["selected_difference_key"]]
        results.append(
            {
                "case_id": f"viddiff-{row['sample_key']}-{row['selected_difference_key']}",
                "sample_key": row["sample_key"],
                "comparison_statement": statement,
                "measurement": measurement,
                "oracle_direction": oracle,
                "direction_correct": measurement["preferred_video"] == oracle,
            }
        )
    correct = sum(item["direction_correct"] for item in results)
    oracle_counts = Counter(item["oracle_direction"] for item in results)
    majority_baseline = max(oracle_counts.values(), default=0)
    report = {
        "run_id": "viddiff_local_tube_motion_proxy",
        "label_access_policy": "oracle read only after deterministic local measurement",
        "external_service_calls": 0,
        "summary": {
            "cases": len(results),
            "passed": correct,
            "accuracy": round(correct / len(results), 4) if results else 0.0,
            "oracle_majority_baseline_accuracy": (
                round(majority_baseline / len(results), 4) if results else 0.0
            ),
            "lift_over_majority_baseline": (
                round((correct - majority_baseline) / len(results), 4) if results else 0.0
            ),
            "safe_to_register_tool": (
                bool(results)
                and correct / len(results) >= 0.8
                and correct > majority_baseline
            ),
        },
        "cases": results,
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--case-indices", nargs="+", type=int, default=[21, 22, 24])
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--all-available-statements", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
