"""Label-blind local Qwen3-VL multi-view predicate measurements for VidDiff."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from pipeline import ROOT, write_json


DEFAULT_MODEL = Path(
    "C:/Users/Zya/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_local_qwen3vl_multiview_v1.json"
)
SELECTION_GLOB = (
    "scale30_viddiff_selection*/viddiff_best_of_tree_seed_20260818_n5/"
    "candidate_measurements.student.jsonl"
)
TARGET_CASES = {
    "surgery_90",
    "surgery_72",
    "surgery_2",
    "surgery_115",
    "surgery_29",
    "surgery_43",
    "surgery_55",
    "surgery_56",
    "surgery_123",
}


def _items() -> dict[str, dict[str, Any]]:
    result = {}
    for path in (ROOT / "outputs").glob(SELECTION_GLOB):
        audit = json.loads((path.parent / "selection_audit.json").read_text(encoding="utf-8"))
        oracle = {str(row["sample_key"]): str(row["oracle_direction"]) for row in audit["cases"]}
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            key = str(item["sample_key"])
            if key in TARGET_CASES:
                item["_oracle_direction"] = oracle[key]
                result[key] = item
    return result


def _view_paths(item: dict[str, Any], role: str) -> dict[str, Path]:
    candidates = {entry["name"]: entry for entry in item["candidate_tools"]}
    uniform_report = ROOT / candidates["compare_uniform_timeline"]["provenance"]["candidate_report"]
    targeted_report = ROOT / candidates["compare_action_refined_symmetric"]["provenance"]["candidate_report"]
    suffix = role.lower()
    return {
        "uniform": uniform_report.parent / item["sample_key"] / f"video_{suffix}_mosaic.jpg",
        "targeted": targeted_report.parent / item["sample_key"] / f"video_{suffix}_targeted.jpg",
    }


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"local VLM returned no JSON object: {text[-500:]}")
    return json.loads(cleaned[start : end + 1])


class LocalQwenPredicateScorer:
    def __init__(self, model_path: Path) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()

    def score(
        self,
        *,
        image_path: Path,
        action_description: str,
        statement: str,
        view: str,
    ) -> dict[str, Any]:
        prompt = (
            "You are one label-blind measurement component. You see frames from ONE video only, so do not compare "
            "against another video and do not guess a benchmark answer. Measure how strongly the visible pixels support "
            f"this statement: {statement!r}. The shared action is {action_description!r}. This is the {view} view; tile "
            "labels are local times. Compile comparative wording into a unary degree: for example, 'the tube moves more' "
            "means score the absolute amount of tube displacement in THIS video, 'tension is lower' means score visible "
            "slack/low tension in THIS video, and 'more accurately' means score absolute accuracy in THIS video. Never "
            "interpret 'more' as more than other objects inside the same video. Distinguish the exact predicate from "
            "generic skill, speed, duration, or whole-frame motion. Return only JSON with keys support_score (0-100), "
            "observability_score (0-100), best_interval_s "
            "([start,end]), positive_cues (array), counterevidence (array), uncertainty_sources (array), and summary. "
            "A high support score requires direct repeated pixel evidence; low observability must not be converted into "
            "a directional guess."
        )
        mosaic = Image.open(image_path).convert("RGB")
        if view == "uniform":
            columns, rows, label_height = 4, 3, 24
        else:
            columns, rows, label_height = 12, 2, 24
        tile_width = mosaic.width // columns
        row_height = mosaic.height // rows
        tile_height = row_height - label_height
        positions = list(range(columns)) if columns <= 6 else list(range(0, columns, 2))
        images = [
            mosaic.crop(
                (
                    column * tile_width,
                    row * row_height,
                    (column + 1) * tile_width,
                    row * row_height + tile_height,
                )
            )
            for row in range(rows)
            for column in positions
        ]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        result = _json_object(decoded)
        result["support_score"] = float(result["support_score"])
        result["observability_score"] = float(result["observability_score"])
        return result

    def compare_pair(
        self,
        *,
        mosaic_a: Path,
        mosaic_b: Path,
        statement: str,
    ) -> dict[str, Any]:
        def tiles(path: Path) -> list[Image.Image]:
            mosaic = Image.open(path).convert("RGB")
            tile_width = mosaic.width // 4
            row_height = mosaic.height // 3
            return [
                mosaic.crop(
                    (
                        column * tile_width,
                        row * row_height,
                        (column + 1) * tile_width,
                        row * row_height + row_height - 24,
                    )
                )
                for row in range(3)
                for column in (0, 2)
            ]

        images_a, images_b = tiles(mosaic_a), tiles(mosaic_b)
        prompt = (
            f"Compare the exact statement {statement!r}. Use only the labeled frame groups above. Return c when the "
            "difference is equal, insufficiently visible, or not stable across the sampled sequence. Do not substitute "
            "generic skill, duration, or overall motion for the named predicate. Return only JSON with preferred_video "
            "('a','b','c'), support_score_a (0-100), support_score_b (0-100), confidence (0-1), direct_cues (array), "
            "counterevidence (array), and rationale."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": "VIDEO A FRAMES"}]
        content.extend({"type": "image", "image": image} for image in images_a)
        content.append({"type": "text", "text": "VIDEO B FRAMES"})
        content.extend({"type": "image", "image": image} for image in images_b)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images = [*images_a, *images_b]
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _json_object(decoded)


def _combine(role_views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observable_views = {
        name: value
        for name, value in role_views.items()
        if float(value["observability_score"]) >= 45
    }
    scores = [float(value["support_score"]) for value in observable_views.values()]
    observability = [float(value["observability_score"]) for value in observable_views.values()]
    if not scores:
        return {
            "support_score": 0.0,
            "minimum_observability": 0.0,
            "view_score_range": 100.0,
            "observable_views": [],
            "discarded_unobservable_views": sorted(role_views),
        }
    return {
        "support_score": round(sum(scores) / len(scores), 3),
        "minimum_observability": round(min(observability), 3),
        "view_score_range": round(max(scores) - min(scores), 3),
        "observable_views": sorted(observable_views),
        "discarded_unobservable_views": sorted(set(role_views) - set(observable_views)),
    }


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    delta = float(a["support_score"]) - float(b["support_score"])
    stable = a["view_score_range"] <= 25 and b["view_score_range"] <= 25
    observable = min(a["minimum_observability"], b["minimum_observability"]) >= 45
    separated = abs(delta) >= 15
    if stable and observable and separated:
        preferred = "a" if delta > 0 else "b"
        rule = "independent_multiview_stable_margin"
    else:
        preferred = "c"
        rule = "independent_multiview_abstain"
    return {
        "preferred_video": preferred,
        "decision_rule": rule,
        "score_delta_a_minus_b": round(delta, 3),
        "stable_across_views": stable,
        "sufficiently_observable": observable,
        "sufficient_margin": separated,
        "confidence": round(min(0.9, 0.35 + abs(delta) / 100), 3),
    }


def run(model_path: Path, case_ids: list[str]) -> dict[str, Any]:
    items = _items()
    scorer = LocalQwenPredicateScorer(model_path)
    cases = []
    for index, key in enumerate(case_ids, start=1):
        item = items[key]
        role_results: dict[str, Any] = {}
        for role in ("A", "B"):
            views = {}
            for view, image_path in _view_paths(item, role).items():
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                views[view] = scorer.score(
                    image_path=image_path,
                    action_description=item["task"]["action_description"],
                    statement=item["task"]["comparison_statement"],
                    view=view,
                )
            role_results[role.lower()] = {
                "views": views,
                "aggregate": _combine(views),
            }
        measurement = _compare(
            role_results["a"]["aggregate"],
            role_results["b"]["aggregate"],
        )
        oracle = str(item["_oracle_direction"])
        cases.append(
            {
                "sample_key": key,
                "comparison_statement": item["task"]["comparison_statement"],
                "measurement": measurement,
                "independent_video_measurements": role_results,
                "oracle_direction": oracle,
                "direction_correct": measurement["preferred_video"] == oracle,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(case_ids)}",
                    "sample_key": key,
                    "prediction": measurement["preferred_video"],
                    "oracle": oracle,
                    "correct": cases[-1]["direction_correct"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    passed = sum(row["direction_correct"] for row in cases)
    return {
        "experiment": "viddiff_local_qwen3vl_independent_multiview_v1",
        "model": str(model_path),
        "label_boundary": {
            "oracle_used_during_measurement": False,
            "each_video_measured_independently": True,
            "oracle_used_after_measurement_for_verifier_metrics": True,
            "external_service_calls": 0,
        },
        "decision_policy": {
            "view_score_range_max": 25,
            "minimum_observability": 45,
            "absolute_support_margin": 15,
            "thresholds_tuned_on_target_cases": False,
        },
        "summary": {
            "cases": len(cases),
            "passed": passed,
            "accuracy": round(passed / len(cases), 4) if cases else 0.0,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    case_ids = args.case_id or sorted(TARGET_CASES)
    report = run(args.model, case_ids)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
