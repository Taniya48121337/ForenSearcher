"""Full-predicate validation for grounded needle-tip geometry measurements."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiment_viddiff_grounded_tip_crop import (
    DEFAULT_DINO,
    DEFAULT_QWEN,
    NeedleProposalDetector,
    _contact_sheet,
    _expanded_crop,
)
from experiment_viddiff_grounded_tip_geometry import (
    _aggregate,
    _measure_geometry,
    _select_proposals,
)
from experiment_viddiff_local_qwen3vl import LocalQwenPredicateScorer
from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows


STATEMENT = "The needle is grasped closer to the tip"
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_full_v1.json"
)
DEFAULT_CACHE = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_full_cache_v1.json"
)
DEFAULT_ASSETS = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_full_assets_v1"
)
DEFAULT_SMOKE = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_surgery123_v1.json"
)


def _rows() -> list[dict[str, Any]]:
    selected = []
    for source in available_surgery_rows():
        for key, descriptor in source["differences_annotated"].items():
            if isinstance(descriptor, dict) and descriptor.get("description") == STATEMENT:
                selected.append({**source, "selected_difference_key": key})
    return selected


def _predict(left: dict[str, Any], right: dict[str, Any], margin: float) -> str:
    a, b = left.get("median_contact_from_tip_pct"), right.get("median_contact_from_tip_pct")
    if not left.get("observable") or not right.get("observable") or a is None or b is None:
        return "c"
    if abs(float(a) - float(b)) < margin:
        return "c"
    return "a" if float(a) < float(b) else "b"


def _metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("a", "b", "c")
    recalls, f1s = {}, {}
    for label in labels:
        tp = sum(x["oracle_direction"] == label and x["predicted_direction"] == label for x in cases)
        fn = sum(x["oracle_direction"] == label and x["predicted_direction"] != label for x in cases)
        fp = sum(x["oracle_direction"] != label and x["predicted_direction"] == label for x in cases)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls[label] = recall
        f1s[label] = 2 * precision * recall / max(precision + recall, 1e-12)
    truth_counts = Counter(x["oracle_direction"] for x in cases)
    directional = [x for x in cases if x["predicted_direction"] != "c"]
    non_c = [x for x in cases if x["oracle_direction"] != "c"]
    return {
        "cases": len(cases),
        "accuracy": round(sum(x["direction_correct"] for x in cases) / max(len(cases), 1), 4),
        "balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
        "macro_f1": round(float(np.mean(list(f1s.values()))), 4),
        "non_c_recall": round(sum(x["direction_correct"] for x in non_c) / max(len(non_c), 1), 4),
        "directional_precision": round(sum(x["direction_correct"] for x in directional) / max(len(directional), 1), 4),
        "majority_baseline_accuracy": round(max(truth_counts.values()) / max(len(cases), 1), 4),
        "recall_by_label": {key: round(value, 4) for key, value in recalls.items()},
        "prediction_counts": dict(Counter(x["predicted_direction"] for x in cases)),
        "truth_counts": dict(truth_counts),
    }


def _score_video(
    *,
    video: dict[str, Any],
    detector: NeedleProposalDetector,
    scorer: LocalQwenPredicateScorer,
    frame_count: int,
    asset_root: Path,
) -> dict[str, Any]:
    frames = _sample_frames(VIDDIFF_ROOT / video["path"], video["frames_trim"], frame_count)
    proposals = detector.proposals([image for image, _frame_index in frames])
    selected, suspects = _select_proposals(frames, proposals, limit=8)
    crops = [_expanded_crop(image, candidate["box"], scale=3.5) for _rank, image, _frame, candidate in selected]
    labels = [f"crop {index + 1}" for index in range(len(crops))]
    safe_name = Path(str(video["path"])).stem
    sheet_path = asset_root / f"{safe_name}_geometry_crops.jpg"
    _contact_sheet(crops, labels, sheet_path)
    raw = _measure_geometry(scorer, crops, labels)
    return {
        "aggregate": _aggregate(raw),
        "raw_measurement": raw,
        "selected_proposals": [
            {"frame_index": frame_index, **candidate}
            for _rank, _image, frame_index, candidate in selected
        ],
        "static_repeat_suspect_count": len(suspects),
        "contact_sheet": str(sheet_path),
    }


def run(
    *,
    qwen_path: Path,
    dino_path: Path,
    frame_count: int,
    margin: float,
    cache_path: Path,
    asset_root: Path,
) -> dict[str, Any]:
    rows = _rows()
    videos: dict[str, dict[str, Any]] = {}
    for row in rows:
        for video in row["videos"]:
            videos.setdefault(str(video["path"]), video)
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8")).get("video_scores", {})
    elif DEFAULT_SMOKE.exists():
        smoke = json.loads(DEFAULT_SMOKE.read_text(encoding="utf-8"))
        target = next(row for row in rows if row["sample_key"] == "surgery_123")
        for role, video in zip(("a", "b"), target["videos"]):
            cache[str(video["path"])] = smoke["role_results"][role]
        write_json(cache_path, {"video_scores": cache})
    missing = [(key, video) for key, video in sorted(videos.items()) if key not in cache]
    if missing:
        detector = NeedleProposalDetector(dino_path)
        scorer = LocalQwenPredicateScorer(qwen_path)
        for index, (key, video) in enumerate(missing, start=1):
            cache[key] = _score_video(
                video=video,
                detector=detector,
                scorer=scorer,
                frame_count=frame_count,
                asset_root=asset_root,
            )
            write_json(cache_path, {"video_scores": cache})
            agg = cache[key]["aggregate"]
            print(json.dumps({"progress": f"{index}/{len(missing)}", "video": Path(key).stem, "valid": agg["valid_crop_count"], "position": agg["median_contact_from_tip_pct"]}, ensure_ascii=False), flush=True)

    cases = []
    for row in rows:
        left_key, right_key = (str(video["path"]) for video in row["videos"])
        left, right = cache[left_key]["aggregate"], cache[right_key]["aggregate"]
        predicted = _predict(left, right, margin)
        oracle = str(row["differences_gt"][row["selected_difference_key"]])
        cases.append(
            {
                "sample_key": row["sample_key"],
                "videos": [left_key, right_key],
                "contact_from_tip_pct_a": left.get("median_contact_from_tip_pct"),
                "contact_from_tip_pct_b": right.get("median_contact_from_tip_pct"),
                "valid_crop_count_a": left.get("valid_crop_count"),
                "valid_crop_count_b": right.get("valid_crop_count"),
                "predicted_direction": predicted,
                "oracle_direction": oracle,
                "direction_correct": predicted == oracle,
            }
        )
    return {
        "experiment": "viddiff_grounded_per_crop_tip_geometry_full_v1",
        "statement": STATEMENT,
        "label_boundary": {
            "video_measurements_cached_before_labels_are_read": True,
            "oracle_used_only_for_post_measurement_metrics": True,
            "external_service_calls": 0,
        },
        "decision_policy": {
            "minimum_valid_crops_per_video": 2,
            "minimum_crop_confidence": 55,
            "fixed_absolute_contact_position_margin": margin,
        },
        "summary": _metrics(cases),
        "target_surgery_123": next(x for x in cases if x["sample_key"] == "surgery_123"),
        "cases": cases,
        "video_scores": cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--margin", type=float, default=12.0)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(qwen_path=args.qwen, dino_path=args.dino, frame_count=args.frame_count, margin=args.margin, cache_path=args.cache, asset_root=args.assets)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
