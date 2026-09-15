"""Measure needle grasp position from grounded original-frame crops.

This experiment deliberately separates three jobs:
1. Grounding-DINO proposes needle regions without seeing labels.
2. Repeated, spatially fixed proposals are treated as fixture/background suspects.
3. Qwen3-VL performs per-crop geometric measurements rather than a holistic
   predicate-support score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiment_viddiff_grounded_tip_crop import (
    DEFAULT_DINO,
    DEFAULT_QWEN,
    NeedleProposalDetector,
    _contact_sheet,
    _expanded_crop,
)
from experiment_viddiff_local_qwen3vl import LocalQwenPredicateScorer, _json_object
from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows


DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_surgery123_v1.json"
)
DEFAULT_ASSETS = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_surgery123_assets_v1"
)


def _near_same_region(a: dict[str, Any], b: dict[str, Any], width: int, height: int) -> bool:
    ax1, ay1, ax2, ay2 = a["box"]
    bx1, by1, bx2, by2 = b["box"]
    ac = np.array([(ax1 + ax2) / (2 * width), (ay1 + ay2) / (2 * height)])
    bc = np.array([(bx1 + bx2) / (2 * width), (by1 + by2) / (2 * height)])
    asize = np.array([(ax2 - ax1) / width, (ay2 - ay1) / height])
    bsize = np.array([(bx2 - bx1) / width, (by2 - by1) / height])
    center_close = float(np.linalg.norm(ac - bc)) <= 0.028
    size_close = float(np.linalg.norm(asize - bsize)) <= 0.035
    return center_close and size_close


def _select_proposals(
    frames: list[tuple[Image.Image, int]],
    proposals: list[list[dict[str, Any]]],
    *,
    limit: int,
) -> tuple[list[tuple[float, Image.Image, int, dict[str, Any]]], list[dict[str, Any]]]:
    """Suppress persistent fixture-like boxes, then retain one proposal per frame."""
    width, height = frames[0][0].size
    flat: list[tuple[int, dict[str, Any]]] = []
    for frame_slot, candidates in enumerate(proposals):
        for candidate in candidates[:4]:
            flat.append((frame_slot, candidate))

    repeat_counts: list[int] = []
    for frame_slot, candidate in flat:
        repeated_frames = {
            other_slot
            for other_slot, other in flat
            if other_slot != frame_slot
            and _near_same_region(candidate, other, width, height)
        }
        repeat_counts.append(1 + len(repeated_frames))

    suspects: list[dict[str, Any]] = []
    by_frame: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for (frame_slot, candidate), repeats in zip(flat, repeat_counts):
        enriched = {**candidate, "spatial_repeat_count": repeats}
        if repeats >= 3:
            suspects.append({"frame_slot": frame_slot, **enriched})
            continue
        # A small repeat penalty prevents a twice-seen fixture from dominating.
        adjusted = float(candidate["rank_score"]) - 0.025 * max(repeats - 1, 0)
        by_frame.setdefault(frame_slot, []).append((adjusted, enriched))

    ranked: list[tuple[float, Image.Image, int, dict[str, Any]]] = []
    for frame_slot, candidates in by_frame.items():
        if not candidates:
            continue
        adjusted, candidate = max(candidates, key=lambda item: item[0])
        image, frame_index = frames[frame_slot]
        ranked.append((adjusted, image, frame_index, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit], suspects


def _measure_geometry(
    scorer: LocalQwenPredicateScorer,
    crops: list[Image.Image],
    labels: list[str],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for crop, label in zip(crops, labels):
        content.extend(({"type": "text", "text": label}, {"type": "image", "image": crop}))
    content.append(
        {
            "type": "text",
            "text": (
                "These are enlarged proposals sampled across ONE Needle Passing video. Some are false detections. "
                "Act as a cautious surgical-video measurer, not a task classifier. For EACH crop independently, "
                "check whether (1) the curved needle body, (2) its SHARP free tip, (3) active grasper jaws, and "
                "(4) the actual jaw-to-needle contact point are all visible. If any item is absent or ambiguous, "
                "valid_measurement must be false and contact_from_sharp_tip_pct must be null. If all are visible, "
                "estimate the contact point along the needle arc: 0 means exactly at the sharp tip; 100 means the "
                "suture-attached tail. Do not confuse fixture pegs, hoop wires, reflections, or a merely nearby jaw "
                "with the needle/contact. Return ONLY JSON with key crop_measurements (array). Each item must contain "
                "label, needle_visible, sharp_tip_visible, active_jaws_visible, contact_visible, valid_measurement, "
                "contact_from_sharp_tip_pct (number or null), confidence (0-100), and one-sentence rationale. Also "
                "return video_summary and uncertainty_sources. Never fill a number just to complete the schema."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    prompt = scorer.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = scorer.processor(text=[prompt], images=crops, return_tensors="pt").to(scorer.model.device)
    with torch.inference_mode():
        generated = scorer.model.generate(**inputs, max_new_tokens=1500, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1] :]
    decoded = scorer.processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _json_object(decoded)


def _aggregate(measurement: dict[str, Any]) -> dict[str, Any]:
    valid = []
    for item in measurement.get("crop_measurements", []):
        value = item.get("contact_from_sharp_tip_pct")
        confidence = float(item.get("confidence") or 0)
        if item.get("valid_measurement") is True and value is not None and confidence >= 55:
            valid.append({**item, "contact_from_sharp_tip_pct": float(value), "confidence": confidence})
    values = [item["contact_from_sharp_tip_pct"] for item in valid]
    # A single VLM assertion is insufficient evidence for a video-level comparison.
    observable = len(valid) >= 2
    return {
        "valid_crop_count": len(valid),
        "observable": observable,
        "median_contact_from_tip_pct": round(float(median(values)), 2) if observable else None,
        "valid_crops": valid,
    }


def run(
    *,
    case_id: str,
    qwen_path: Path,
    dino_path: Path,
    asset_root: Path,
    frame_count: int,
) -> dict[str, Any]:
    row = next(item for item in available_surgery_rows() if item["sample_key"] == case_id)
    difference_key = next(
        key
        for key, descriptor in row["differences_annotated"].items()
        if isinstance(descriptor, dict)
        and descriptor.get("description") == "The needle is grasped closer to the tip"
    )
    detector = NeedleProposalDetector(dino_path)
    scorer = LocalQwenPredicateScorer(qwen_path)
    role_results: dict[str, Any] = {}
    for role, video in zip(("a", "b"), row["videos"]):
        frames = _sample_frames(
            VIDDIFF_ROOT / video["path"], video["frames_trim"], frame_count
        )
        proposals = detector.proposals([image for image, _frame_index in frames])
        selected, suspects = _select_proposals(frames, proposals, limit=8)
        crops = [_expanded_crop(image, candidate["box"], scale=3.5) for _rank, image, _frame, candidate in selected]
        trim_start = int(video["frames_trim"][0] or 0)
        fps = float(video["fps_original"])
        labels = [
            f"{role.upper()} crop {index + 1}: t={(frame_index - trim_start) / fps:.2f}s"
            for index, (_rank, _image, frame_index, _candidate) in enumerate(selected)
        ]
        sheet_path = asset_root / f"{case_id}_{role}_geometry_crops.jpg"
        _contact_sheet(crops, labels, sheet_path)
        raw = _measure_geometry(scorer, crops, labels)
        role_results[role] = {
            "aggregate": _aggregate(raw),
            "raw_measurement": raw,
            "selected_proposals": [
                {"frame_index": frame_index, **candidate}
                for _rank, _image, frame_index, candidate in selected
            ],
            "static_repeat_suspects": suspects,
            "contact_sheet": str(sheet_path),
        }

    agg_a, agg_b = role_results["a"]["aggregate"], role_results["b"]["aggregate"]
    value_a, value_b = agg_a["median_contact_from_tip_pct"], agg_b["median_contact_from_tip_pct"]
    if not agg_a["observable"] or not agg_b["observable"] or abs(value_a - value_b) < 12:
        preferred = "c"
    else:
        preferred = "a" if value_a < value_b else "b"
    oracle = str(row["differences_gt"][difference_key])
    return {
        "experiment": "viddiff_grounded_per_crop_tip_geometry_v1",
        "sample_key": case_id,
        "comparison_statement": "The needle is grasped closer to the tip",
        "label_boundary": {
            "detector_and_qwen_inputs": "original visible frames only",
            "oracle_used_during_measurement": False,
            "oracle_used_after_measurement_for_verifier": True,
            "external_service_calls": 0,
        },
        "decision_policy": {
            "minimum_valid_crops_per_video": 2,
            "minimum_crop_confidence": 55,
            "minimum_absolute_contact_position_margin": 12,
        },
        "measurement": {
            "preferred_video": preferred,
            "contact_from_tip_pct_a": value_a,
            "contact_from_tip_pct_b": value_b,
        },
        "role_results": role_results,
        "oracle_direction": oracle,
        "direction_correct": preferred == oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="surgery_123")
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(
        case_id=args.case_id,
        qwen_path=args.qwen,
        dino_path=args.dino,
        asset_root=args.assets,
        frame_count=args.frame_count,
    )
    write_json(args.output, report)
    print(json.dumps({"measurement": report["measurement"], "oracle": report["oracle_direction"], "correct": report["direction_correct"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
