"""VidDiffBench coarse-to-fine, order-symmetric CLI vision canary."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw

from cli_teacher_rollout import DEFAULT_CODEX_JS, DEFAULT_NODE, _usage_from_events
from pipeline import ROOT, write_json
from viddiff_probe import (
    DEFAULT_OUTPUT_ROOT,
    VIDDIFF_ROOT,
    _case_index_suffix,
    _interval_valid,
    _resume_case_map,
    _validate_teacher_payload,
    build_mosaic,
    select_cases_for_run,
)


def _motion_peaks(video_path: Path, frames_trim: list[Any], fps: float, count: int = 3) -> list[float]:
    start = int(frames_trim[0] or 0)
    stop = int(frames_trim[1])
    stride = max(1, round(float(fps) / 10.0))
    previous: np.ndarray | None = None
    times: list[float] = []
    scores: list[float] = []
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index < start or index >= stop:
                if index >= stop:
                    break
                continue
            if (index - start) % stride:
                continue
            gray = np.asarray(
                Image.fromarray(frame.to_ndarray(format="gray")).resize((96, 72), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            score = 0.0 if previous is None else float(np.mean(np.abs(gray - previous)) / 255.0)
            previous = gray
            times.append((index - start) / float(fps))
            scores.append(score)
    if not scores:
        return [0.0]
    values = np.asarray(scores, dtype=np.float64)
    smooth_width = max(3, round(10.0 * 0.8))
    kernel = np.ones(smooth_width, dtype=np.float64) / smooth_width
    smoothed = np.convolve(values, kernel, mode="same")
    candidates = list(np.argsort(smoothed)[::-1])
    selected: list[float] = []
    minimum_separation_s = 2.5
    for candidate in candidates:
        time_s = times[int(candidate)]
        if all(abs(time_s - existing) >= minimum_separation_s for existing in selected):
            selected.append(time_s)
        if len(selected) == count:
            break
    if len(selected) < count:
        duration = (stop - start) / float(fps)
        for time_s in np.linspace(0.1 * duration, 0.9 * duration, count):
            if all(abs(float(time_s) - existing) >= 0.5 for existing in selected):
                selected.append(float(time_s))
            if len(selected) == count:
                break
    return sorted(selected[:count])


def _decode_indices(video_path: Path, indices: list[int]) -> dict[int, Image.Image]:
    requested = set(indices)
    decoded: dict[int, Image.Image] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                decoded[index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if index >= max(indices):
                break
    if requested != set(decoded):
        raise ValueError(f"could not decode motion-strip indices from {video_path}")
    return decoded


def build_motion_strip(
    *,
    video_path: Path,
    frames_trim: list[Any],
    fps: float,
    role: str,
    output_path: Path,
    window_count: int = 3,
    frames_per_window: int = 8,
) -> dict[str, Any]:
    start = int(frames_trim[0] or 0)
    stop = int(frames_trim[1])
    duration = (stop - start) / float(fps)
    centers = _motion_peaks(video_path, frames_trim, fps, count=window_count)
    windows: list[dict[str, Any]] = []
    all_indices: list[int] = []
    for center in centers:
        local_start = max(0.0, center - 1.2)
        local_end = min(duration, center + 1.2)
        indices = [
            min(stop - 1, start + round((local_start + position * (local_end - local_start) / max(frames_per_window - 1, 1)) * fps))
            for position in range(frames_per_window)
        ]
        windows.append(
            {
                "center_s": round(center, 3),
                "interval_s": [round(local_start, 3), round(local_end, 3)],
                "indices": indices,
            }
        )
        all_indices.extend(indices)
    decoded = _decode_indices(video_path, sorted(set(all_indices)))
    tile_w, tile_h, label_h = 200, 135, 22
    canvas = Image.new(
        "RGB",
        (tile_w * frames_per_window, (tile_h + label_h) * len(windows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, window in enumerate(windows):
        for column, index in enumerate(window["indices"]):
            x = column * tile_w
            y = row * (tile_h + label_h)
            image = decoded[index].resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            canvas.paste(image, (x, y))
            time_s = (index - start) / float(fps)
            draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
            draw.text((x + 4, y + tile_h + 4), f"{role} W{row + 1} t={time_s:.2f}s", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)
    return {
        "role": role,
        "selection_method": "label_blind_smoothed_frame_difference_peaks",
        "window_count": len(windows),
        "frames_per_window": frames_per_window,
        "windows": [{key: value for key, value in item.items() if key != "indices"} for item in windows],
        "local_duration_s": round(duration, 3),
    }


def comparison_schema() -> dict[str, Any]:
    interval = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "statement_score_a": {"type": "number", "minimum": 0, "maximum": 100},
            "statement_score_b": {"type": "number", "minimum": 0, "maximum": 100},
            "preferred_video": {"type": "string", "enum": ["a", "b", "c"]},
            "video_a_interval_s": interval,
            "video_b_interval_s": interval,
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "independent_observations": {"type": "string", "minLength": 80},
            "comparative_rationale": {"type": "string", "minLength": 60},
        },
        "required": [
            "statement_score_a",
            "statement_score_b",
            "preferred_video",
            "video_a_interval_s",
            "video_b_interval_s",
            "confidence",
            "independent_observations",
            "comparative_rationale",
        ],
        "additionalProperties": False,
    }


class SymmetricCodexComparator:
    source = "codex_cli_coarse_to_fine_symmetric_vision_tool"

    def __init__(self, *, model: str = "gpt-5.4", timeout_s: int = 300) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def _one_pass(
        self,
        *,
        global_a: Path,
        motion_a: Path,
        global_b: Path,
        motion_b: Path,
        action_description: str,
        candidate_difference: str,
        metadata_a: dict[str, Any],
        metadata_b: dict[str, Any],
        pass_name: str,
    ) -> dict[str, Any]:
        _validate_teacher_payload(
            action_description=action_description,
            candidate_difference=candidate_difference,
            metadata={"video_a": metadata_a, "video_b": metadata_b},
        )
        prompt = (
            "You are a blinded measurement backend for fine-grained surgical action comparison. Use only the four "
            "attached images. Images 1-2 belong to video A: a global chronological mosaic and label-blind high-motion "
            "local strips. Images 3-4 are the same views for video B. First assess A and B independently; do not infer "
            "skill from duration, image order, subject identity, or general appearance. Then compare this exact statement: "
            f"{candidate_difference!r}. Shared action: {action_description!r}. Score how strongly visible evidence in "
            "each video supports the statement from 0 to 100. preferred_video='a' only if A visibly supports it more, "
            "'b' only if B supports it more, and 'c' when neither/equal/uncertain. For speed, force, movement, or radiality, "
            "use changes across adjacent frames rather than a single pose. For 'never out of view', inspect coverage across "
            "the global mosaic and treat missing evidence conservatively. Cite visible timestamps; never guess hidden frames.\n"
            f"PASS={pass_name}\nVIDEO_A_METADATA={json.dumps(metadata_a, ensure_ascii=False)}\n"
            f"VIDEO_B_METADATA={json.dumps(metadata_b, ensure_ascii=False)}"
        )
        with tempfile.TemporaryDirectory(prefix="viddiff_v2_cli_") as directory:
            temp_root = Path(directory)
            schema_path = temp_root / "comparison.schema.json"
            output_path = temp_root / "comparison.json"
            schema_path.write_text(json.dumps(comparison_schema()), encoding="utf-8")
            command = [
                str(DEFAULT_NODE),
                str(DEFAULT_CODEX_JS),
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--json",
            ]
            for image_path in (global_a, motion_a, global_b, motion_b):
                command.extend(["--image", str(image_path.resolve())])
            command.extend(
                [
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--model",
                    self.model,
                    "-",
                ]
            )
            import subprocess

            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout_s,
                cwd=directory,
                check=False,
            )
            if completed.returncode != 0:
                diagnostic = (completed.stderr or completed.stdout)[-4000:]
                raise RuntimeError(f"Codex CLI symmetric comparison failed: {diagnostic}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "measurement_id": f"VIDDIFF-V2-{uuid.uuid4().hex}",
            "pass_name": pass_name,
            "usage": _usage_from_events(completed.stdout),
            **payload,
        }

    @staticmethod
    def _map_swapped_preference(value: str) -> str:
        return {"a": "b", "b": "a", "c": "c"}[value]

    def compare(self, **kwargs: Any) -> dict[str, Any]:
        forward = self._one_pass(pass_name="forward_A_then_B", **kwargs)
        swapped = self._one_pass(
            global_a=kwargs["global_b"],
            motion_a=kwargs["motion_b"],
            global_b=kwargs["global_a"],
            motion_b=kwargs["motion_a"],
            action_description=kwargs["action_description"],
            candidate_difference=kwargs["candidate_difference"],
            metadata_a=kwargs["metadata_b"],
            metadata_b=kwargs["metadata_a"],
            pass_name="swapped_B_then_A",
        )
        mapped_swapped = self._map_swapped_preference(swapped["preferred_video"])
        forward_delta = float(forward["statement_score_a"]) - float(forward["statement_score_b"])
        swapped_delta_original_order = float(swapped["statement_score_b"]) - float(swapped["statement_score_a"])
        mean_delta = (forward_delta + swapped_delta_original_order) / 2.0
        order_consistent = forward["preferred_video"] == mapped_swapped
        if order_consistent:
            final_preference = forward["preferred_video"]
            decision_rule = "order_consistent_preference"
        elif abs(mean_delta) >= 18.0 and np.sign(forward_delta) == np.sign(swapped_delta_original_order):
            final_preference = "a" if mean_delta > 0 else "b"
            decision_rule = "consistent_absolute_score_margin"
        else:
            final_preference = "c"
            decision_rule = "order_disagreement_abstain"
        mapped_swapped_a_interval = swapped["video_b_interval_s"]
        mapped_swapped_b_interval = swapped["video_a_interval_s"]
        return {
            "preferred_video": final_preference,
            "decision_rule": decision_rule,
            "order_consistent": order_consistent,
            "mean_score_delta_a_minus_b": round(mean_delta, 3),
            "video_a_interval_s": forward["video_a_interval_s"],
            "video_b_interval_s": forward["video_b_interval_s"],
            "mapped_swapped_intervals": {
                "video_a_interval_s": mapped_swapped_a_interval,
                "video_b_interval_s": mapped_swapped_b_interval,
            },
            "confidence": round((float(forward["confidence"]) + float(swapped["confidence"])) / 2.0, 3),
            "forward": forward,
            "swapped": swapped,
            "usage": {
                key: int(forward["usage"].get(key, 0)) + int(swapped["usage"].get(key, 0))
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_cases_for_run(
        seed=args.seed,
        sample_size=args.sample_size,
        selection_report=getattr(args, "selection_report", None),
    )
    case_indices = getattr(args, "case_indices", None)
    if case_indices:
        invalid = [index for index in case_indices if index < 0 or index >= len(selected)]
        if invalid:
            raise ValueError(f"case indices outside sampled range: {invalid}")
        selected = [selected[index] for index in case_indices]
    suffix = _case_index_suffix(case_indices)
    run_id = f"viddiff_v2_surgery_seed_{args.seed}_n{len(selected)}{suffix}"
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    comparator = SymmetricCodexComparator(model=args.model) if args.run_cli else None
    resumed = _resume_case_map(
        output_root / "probe_report.json",
        enabled=getattr(args, "resume", False),
        require_measurement=comparator is not None,
    )
    results = []
    resumed_cases = 0
    for row in selected:
        key = row["selected_difference_key"]
        descriptor = row["differences_annotated"][key]
        previous = resumed.get(row["sample_key"])
        if (
            previous is not None
            and previous.get("selected_difference_key") == key
            and previous.get("candidate_difference") == descriptor["description"]
        ):
            results.append(previous)
            resumed_cases += 1
            continue
        case_root = output_root / row["sample_key"]
        visual_assets = []
        for role, video in zip(("A", "B"), row["videos"]):
            video_path = VIDDIFF_ROOT / video["path"]
            global_metadata = build_mosaic(
                video_path=video_path,
                frames_trim=video["frames_trim"],
                fps=float(video["fps_original"]),
                role=role,
                output_path=case_root / f"video_{role.lower()}_global.jpg",
                count=args.global_frame_count,
            )
            motion_metadata = build_motion_strip(
                video_path=video_path,
                frames_trim=video["frames_trim"],
                fps=float(video["fps_original"]),
                role=role,
                output_path=case_root / f"video_{role.lower()}_motion.jpg",
            )
            visual_assets.append({"global": global_metadata, "motion": motion_metadata})
        result: dict[str, Any] = {
            "case_id": f"viddiff-{row['sample_key']}",
            "sample_key": row["sample_key"],
            "action_name": row["action_name"],
            "action_description": row["action_description"],
            "selected_difference_key": key,
            "candidate_difference": descriptor["description"],
            "visual_assets": visual_assets,
            "status": "ready",
        }
        if comparator is not None:
            try:
                measurement = comparator.compare(
                    global_a=case_root / "video_a_global.jpg",
                    motion_a=case_root / "video_a_motion.jpg",
                    global_b=case_root / "video_b_global.jpg",
                    motion_b=case_root / "video_b_motion.jpg",
                    action_description=row["action_description"],
                    candidate_difference=descriptor["description"],
                    metadata_a=visual_assets[0],
                    metadata_b=visual_assets[1],
                )
                oracle_direction = row["differences_gt"][key]
                interval_valid = _interval_valid(
                    measurement["video_a_interval_s"], visual_assets[0]["global"]["local_duration_s"]
                ) and _interval_valid(
                    measurement["video_b_interval_s"], visual_assets[1]["global"]["local_duration_s"]
                )
                result.update(
                    {
                        "status": "passed"
                        if measurement["preferred_video"] == oracle_direction and interval_valid
                        else "failed",
                        "measurement": measurement,
                        "oracle_direction": oracle_direction,
                        "direction_correct": measurement["preferred_video"] == oracle_direction,
                        "interval_format_valid": interval_valid,
                    }
                )
            except Exception as error:
                result.update(
                    {"status": "execution_failed", "error_type": type(error).__name__, "error": str(error)}
                )
        results.append(result)
        write_json(output_root / "probe_report.json", {"run_id": run_id, "cases": results})
    correct = sum(item.get("direction_correct") is True for item in results)
    valid = sum(item.get("interval_format_valid") is True for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VidDiffBench",
        "subset": "JIGSAWS surgery hard",
        "seed": args.seed,
        "sample_size": len(selected),
        "source_sample_size": args.sample_size,
        "case_indices": case_indices,
        "method": "coarse_global_plus_motion_peaks_plus_AB_BA_symmetric_comparison",
        "label_access_policy": "retrieval_frames_and_differences_gt_never_used by tool; gt only scores final result",
        "summary": {
            "direction_correct": correct,
            "interval_format_valid": valid,
            "passed": sum(item["status"] == "passed" for item in results),
            "execution_failed": sum(item["status"] == "execution_failed" for item in results),
            "resumed_cases": resumed_cases,
            "executed_cases_current_run": len(results) - resumed_cases,
            "order_consistent": sum(item.get("measurement", {}).get("order_consistent") is True for item in results),
            "safe_to_register_tool": (
                bool(selected) and correct / len(selected) >= 0.8 and valid == len(selected)
            ),
        },
        "cases": results,
    }
    write_json(output_root / "probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--case-indices", nargs="+", type=int)
    parser.add_argument("--global-frame-count", type=int, default=24)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--run-cli", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
