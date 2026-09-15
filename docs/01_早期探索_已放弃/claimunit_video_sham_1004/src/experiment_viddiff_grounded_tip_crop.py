"""Coarse-to-fine needle-grasp measurement using Grounding-DINO crops and Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from experiment_viddiff_local_qwen3vl import (
    DEFAULT_MODEL as DEFAULT_QWEN,
    LocalQwenPredicateScorer,
    _json_object,
)
from pipeline import ROOT, write_json
from viddiff_probe import VIDDIFF_ROOT, _sample_frames, available_surgery_rows


DEFAULT_DINO = Path(
    "C:/Users/Zya/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base/"
    "snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_crop_surgery123_v1.json"
)
DEFAULT_ASSETS = (
    ROOT / "outputs/student_external_contract_v1/viddiff_grounded_tip_crop_surgery123_assets_v1"
)


class NeedleProposalDetector:
    def __init__(self, model_path: Path) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
        ).cuda().eval()

    def proposals(self, images: list[Image.Image]) -> list[list[dict[str, Any]]]:
        prompt = "curved surgical needle. surgical needle. grasper jaws."
        all_results: list[list[dict[str, Any]]] = []
        for start in range(0, len(images), 4):
            batch = images[start : start + 4]
            inputs = self.processor(
                images=batch,
                text=[prompt] * len(batch),
                return_tensors="pt",
                padding=True,
            ).to("cuda")
            with torch.inference_mode():
                outputs = self.model(**inputs)
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.10,
                text_threshold=0.10,
                target_sizes=[image.size[::-1] for image in batch],
            )
            for image, result in zip(batch, processed):
                width, height = image.size
                image_area = float(width * height)
                candidates = []
                for box, score, label in zip(
                    result["boxes"].cpu().tolist(),
                    result["scores"].cpu().tolist(),
                    result["text_labels"],
                ):
                    if "needle" not in str(label).lower():
                        continue
                    x1, y1, x2, y2 = map(float, box)
                    area_fraction = max(x2 - x1, 0) * max(y2 - y1, 0) / image_area
                    if not 0.002 <= area_fraction <= 0.08:
                        continue
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    center_distance = np.hypot(cx / width - 0.5, cy / height - 0.5)
                    if center_distance > 0.48:
                        continue
                    rank_score = float(score) - 0.12 * float(center_distance)
                    candidates.append(
                        {
                            "box": [round(value, 2) for value in box],
                            "detector_score": round(float(score), 4),
                            "label": str(label),
                            "area_fraction": round(area_fraction, 6),
                            "center_distance": round(float(center_distance), 4),
                            "rank_score": round(rank_score, 4),
                        }
                    )
                all_results.append(
                    sorted(candidates, key=lambda item: item["rank_score"], reverse=True)
                )
        return all_results


def _expanded_crop(image: Image.Image, box: list[float], scale: float = 3.0) -> Image.Image:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    side = max(side, min(image.size) * 0.22)
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(image.width, int(round(cx + side / 2)))
    bottom = min(image.height, int(round(cy + side / 2)))
    crop = image.crop((left, top, right, bottom))
    return crop.resize((448, 448), Image.Resampling.LANCZOS)


def _contact_sheet(crops: list[Image.Image], labels: list[str], path: Path) -> None:
    tile, label_h, columns = 448, 28, 4
    rows = (len(crops) + columns - 1) // columns
    sheet = Image.new("RGB", (tile * columns, (tile + label_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (crop, label) in enumerate(zip(crops, labels)):
        x = index % columns * tile
        y = index // columns * (tile + label_h)
        sheet.paste(crop, (x, y))
        draw.text((x + 5, y + tile + 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def _qwen_score(
    scorer: LocalQwenPredicateScorer,
    crops: list[Image.Image],
    labels: list[str],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for crop, label in zip(crops, labels):
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": crop})
    content.append(
        {
            "type": "text",
            "text": (
                "Each image is an enlarged detector proposal from ONE Needle Passing video. Some proposals may be "
                "false or partially occluded. Identify only frames where a curved surgical needle, its sharp tip, and "
                "the active grasper jaws are jointly visible. Measure whether the jaw contact lies close to the SHARP "
                "tip rather than the suture-attached tail or mid-arc. Return only JSON with support_score (0-100), "
                "observability_score (0-100), best_crop_labels (array), positive_cues (array), counterevidence (array), "
                "uncertainty_sources (array), and summary. High support requires repeated crops showing a short "
                "tip-to-jaw distance; detector confidence alone is not evidence."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    text = scorer.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = scorer.processor(text=[text], images=crops, return_tensors="pt")
    inputs = inputs.to(scorer.model.device)
    with torch.inference_mode():
        generated = scorer.model.generate(
            **inputs,
            max_new_tokens=640,
            do_sample=False,
        )
    generated = generated[:, inputs.input_ids.shape[1] :]
    decoded = scorer.processor.batch_decode(generated, skip_special_tokens=True)[0]
    return _json_object(decoded)


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
    frames_by_role: dict[str, list[tuple[Image.Image, int]]] = {}
    for role, video in zip(("a", "b"), row["videos"]):
        frames_by_role[role] = _sample_frames(
            VIDDIFF_ROOT / video["path"],
            video["frames_trim"],
            frame_count,
        )
    detector = NeedleProposalDetector(dino_path)
    scorer = LocalQwenPredicateScorer(qwen_path)
    role_results = {}
    for role, video in zip(("a", "b"), row["videos"]):
        frames = frames_by_role[role]
        proposals = detector.proposals([image for image, _index in frames])
        ranked = []
        for (image, frame_index), candidates in zip(frames, proposals):
            if not candidates:
                continue
            candidate = candidates[0]
            local_time = (frame_index - int(video["frames_trim"][0] or 0)) / float(video["fps_original"])
            ranked.append((candidate["rank_score"], image, frame_index, local_time, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:8]
        crops = [_expanded_crop(image, candidate["box"]) for _rank, image, _frame, _time, candidate in selected]
        labels = [
            f"{role.upper()} crop {index + 1}: t={local_time:.2f}s detector={candidate['detector_score']:.3f}"
            for index, (_rank, _image, _frame, local_time, candidate) in enumerate(selected)
        ]
        sheet_path = asset_root / f"{case_id}_{role}_grounded_crops.jpg"
        _contact_sheet(crops, labels, sheet_path)
        role_results[role] = {
            "measurement": _qwen_score(scorer, crops, labels),
            "selected_proposals": [
                {
                    "frame_index": frame_index,
                    "local_time_s": round(local_time, 3),
                    **candidate,
                }
                for _rank, _image, frame_index, local_time, candidate in selected
            ],
            "contact_sheet": str(sheet_path),
        }
    score_a = float(role_results["a"]["measurement"]["support_score"])
    score_b = float(role_results["b"]["measurement"]["support_score"])
    obs_a = float(role_results["a"]["measurement"]["observability_score"])
    obs_b = float(role_results["b"]["measurement"]["observability_score"])
    if min(obs_a, obs_b) < 45 or abs(score_a - score_b) < 15:
        preferred = "c"
    else:
        preferred = "a" if score_a > score_b else "b"
    oracle = str(row["differences_gt"][difference_key])
    return {
        "experiment": "viddiff_grounding_dino_qwen_tip_crop_v1",
        "sample_key": case_id,
        "comparison_statement": "The needle is grasped closer to the tip",
        "label_boundary": {
            "detector_and_qwen_inputs": "original visible frames only",
            "oracle_used_during_measurement": False,
            "oracle_used_after_measurement_for_verifier": True,
            "external_service_calls": 0,
        },
        "measurement": {
            "preferred_video": preferred,
            "support_score_a": score_a,
            "support_score_b": score_b,
            "observability_a": obs_a,
            "observability_b": obs_b,
        },
        "independent_role_results": role_results,
        "oracle_direction": oracle,
        "direction_correct": preferred == oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="surgery_123")
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--dino", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--frame-count", type=int, default=12)
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

