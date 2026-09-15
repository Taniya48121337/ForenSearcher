"""Point-grounded needle grasp measurement with flip-equivariance checking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageOps

from experiment_viddiff_local_qwen3vl import (
    DEFAULT_MODEL,
    LocalQwenPredicateScorer,
)
from pipeline import ROOT, write_json


DEFAULT_INPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_surgery123_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_tip_point_overlay_surgery123_v1.json"
)
DEFAULT_ASSETS = (
    ROOT / "outputs/student_external_contract_v1/viddiff_tip_point_overlay_surgery123_assets_v1"
)


def _split_sheet(path: Path, count: int = 8) -> list[Image.Image]:
    sheet = Image.open(path).convert("RGB")
    tile, label_h, columns = 448, 28, 4
    return [
        sheet.crop(
            (
                index % columns * tile,
                index // columns * (tile + label_h),
                index % columns * tile + tile,
                index // columns * (tile + label_h) + tile,
            )
        )
        for index in range(count)
    ]


def _locate_one(
    scorer: LocalQwenPredicateScorer,
    crop: Image.Image,
    crop_id: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": crop},
        {
            "type": "text",
            "text": (
                f"This is {crop_id}. Perform point annotation only; do not compare videos and do not estimate a "
                "score. Use Qwen normalized image coordinates x,y from 0 to 1000. Locate "
                "T = the sharp free tip of the curved "
                "surgical needle, S = the opposite needle end where the suture thread attaches, and C = the exact "
                "point on the needle currently enclosed by active grasper jaws. A nearby jaw is not contact. Fixture "
                "pegs, hoop wires, loose suture, and reflections are not the needle. If T, S, and C are not all "
                "unambiguously visible, set observable false and all points null. Return ONLY one JSON object with "
                "crop_id, observable, T, S, C (each [x,y] or null), confidence 0-100, and a short "
                "visibility_rationale. Never invent a point to complete the schema and do not return an array."
            ),
        },
    ]
    messages = [{"role": "user", "content": content}]
    prompt = scorer.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = scorer.processor(text=[prompt], images=[crop], return_tensors="pt").to(scorer.model.device)
    with torch.inference_mode():
        generated = scorer.model.generate(**inputs, max_new_tokens=320, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1] :]
    decoded = scorer.processor.batch_decode(generated, skip_special_tokens=True)[0]
    start = decoded.find("{")
    if start < 0:
        raise ValueError(f"point localizer returned no JSON object: {decoded[:300]!r}")
    parsed, consumed = json.JSONDecoder().raw_decode(decoded[start:])
    parsed["_raw_suffix"] = decoded[start + consumed :].strip()
    return parsed


def _valid_point(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and 0 <= float(item) <= 1000 for item in value)
    )


def _mapped_flip_point(value: list[float]) -> list[float]:
    return [1000.0 - float(value[0]), float(value[1])]


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _equivariant_measurements(
    originals: list[dict[str, Any]],
    flipped: list[dict[str, Any]],
    labels: list[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    results = []
    for label, a, b in zip(labels, originals, flipped):
        reasons = []
        if a.get("observable") is not True or b.get("observable") is not True:
            reasons.append("one_or_both_views_unobservable")
        if min(float(a.get("confidence") or 0), float(b.get("confidence") or 0)) < 55:
            reasons.append("low_confidence")
        contradiction_terms = ("not visible", "not clearly", "ambiguous", "obscured", "set to null")
        for view_name, annotation in (("original", a), ("flipped", b)):
            rationale = str(annotation.get("visibility_rationale") or "").lower()
            if annotation.get("observable") is True and any(term in rationale for term in contradiction_terms):
                reasons.append(f"{view_name}_field_rationale_contradiction")
        point_errors: dict[str, float] = {}
        for name in ("T", "S", "C"):
            if not _valid_point(a.get(name)) or not _valid_point(b.get(name)):
                reasons.append(f"invalid_{name}_point")
                continue
            mapped = _mapped_flip_point(b[name])
            point_errors[name] = round(_distance(a[name], mapped), 2)
            if point_errors[name] > tolerance:
                reasons.append(f"{name}_flip_inconsistent")
        ratio = None
        if not reasons:
            needle_span = _distance(a["T"], a["S"])
            contact_distance = _distance(a["T"], a["C"])
            if needle_span < 120:
                reasons.append("tip_tail_span_too_small")
            elif contact_distance > needle_span * 1.35:
                reasons.append("contact_not_on_needle_extent")
            else:
                ratio = round(contact_distance / needle_span, 4)
        results.append(
            {
                "label": label,
                "accepted": not reasons,
                "contact_distance_ratio": ratio,
                "flip_point_errors": point_errors,
                "rejection_reasons": reasons,
                "original": a,
                "flipped": b,
            }
        )
    return results


def _draw_overlay(crop: Image.Image, measurement: dict[str, Any]) -> Image.Image:
    canvas = crop.copy()
    draw = ImageDraw.Draw(canvas)
    colors = {"T": "#00ff66", "S": "#00aaff", "C": "#ff3355"}
    for name, color in colors.items():
        point = measurement.get("original", {}).get(name)
        if not _valid_point(point):
            continue
        x, y = float(point[0]) * canvas.width / 1000, float(point[1]) * canvas.height / 1000
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=4)
        draw.text((x + 10, y - 12), name, fill=color, stroke_width=2, stroke_fill="black")
    status = "ACCEPT" if measurement["accepted"] else "REJECT"
    draw.rectangle((0, 0, 165, 28), fill="black")
    draw.text((6, 6), status, fill="#00ff66" if measurement["accepted"] else "#ff3355")
    return canvas


def _save_overlay_sheet(images: list[Image.Image], path: Path) -> None:
    tile, columns = 448, 4
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (tile * columns, tile * rows), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (index % columns * tile, index // columns * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def run(*, model_path: Path, input_path: Path, asset_root: Path, tolerance: float) -> dict[str, Any]:
    source = json.loads(input_path.read_text(encoding="utf-8"))
    scorer = LocalQwenPredicateScorer(model_path)
    role_results = {}
    for role in ("a", "b"):
        sheet_path = Path(source["role_results"][role]["contact_sheet"])
        crops = _split_sheet(sheet_path)
        labels = [f"crop {index + 1}" for index in range(len(crops))]
        original = [
            _locate_one(scorer, crop, label)
            for crop, label in zip(crops, labels)
        ]
        flipped_crops = [ImageOps.mirror(crop) for crop in crops]
        flipped = [
            _locate_one(scorer, crop, label)
            for crop, label in zip(flipped_crops, labels)
        ]
        measurements = _equivariant_measurements(original, flipped, labels, tolerance)
        accepted = [x["contact_distance_ratio"] for x in measurements if x["accepted"]]
        value = round(float(median(accepted)), 4) if len(accepted) >= 2 else None
        overlays = [_draw_overlay(crop, item) for crop, item in zip(crops, measurements)]
        overlay_path = asset_root / f"surgery_123_{role}_point_overlays.jpg"
        _save_overlay_sheet(overlays, overlay_path)
        role_results[role] = {
            "accepted_crop_count": len(accepted),
            "median_contact_distance_ratio": value,
            "measurements": measurements,
            "raw_original": original,
            "raw_flipped": flipped,
            "overlay_sheet": str(overlay_path),
        }
    value_a = role_results["a"]["median_contact_distance_ratio"]
    value_b = role_results["b"]["median_contact_distance_ratio"]
    if value_a is None or value_b is None or abs(value_a - value_b) < 0.15:
        preferred = "c"
    else:
        preferred = "a" if value_a < value_b else "b"
    oracle = str(source["oracle_direction"])
    return {
        "experiment": "viddiff_point_grounding_flip_equivariance_v1",
        "sample_key": "surgery_123",
        "label_boundary": {
            "point_localizer_inputs": "student-visible grounded crops only",
            "oracle_used_during_measurement": False,
            "oracle_used_after_measurement_for_verifier": True,
            "external_service_calls": 0,
        },
        "decision_policy": {
            "minimum_accepted_crops_per_video": 2,
            "minimum_localizer_confidence": 55,
            "flip_point_tolerance_normalized": tolerance,
            "minimum_absolute_ratio_margin": 0.15,
        },
        "measurement": {
            "preferred_video": preferred,
            "contact_distance_ratio_a": value_a,
            "contact_distance_ratio_b": value_b,
        },
        "role_results": role_results,
        "oracle_direction": oracle,
        "direction_correct": preferred == oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--flip-tolerance", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(model_path=args.model, input_path=args.input, asset_root=args.assets, tolerance=args.flip_tolerance)
    write_json(args.output, report)
    print(json.dumps({"measurement": report["measurement"], "accepted_a": report["role_results"]["a"]["accepted_crop_count"], "accepted_b": report["role_results"]["b"]["accepted_crop_count"], "oracle": report["oracle_direction"], "correct": report["direction_correct"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
