"""VidDiffBench action-conditioned refinement over two label-blind v2 proposals."""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import av
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
    select_cases_for_run,
)
from viddiff_probe_v2 import comparison_schema


DEFAULT_V2_REPORT = (
    DEFAULT_OUTPUT_ROOT / "viddiff_v2_surgery_seed_20260818_n10" / "probe_report.json"
)


def _normalize_interval(value: Any, duration: float) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        return [0.0, duration]
    start = max(0.0, min(float(value[0]), duration))
    end = max(start + 0.1, min(float(value[1]), duration))
    return [start, min(end, duration)]


def build_targeted_strip(
    *,
    video_path: Path,
    frames_trim: list[Any],
    fps: float,
    role: str,
    candidate_intervals: list[list[float]],
    output_path: Path,
    frames_per_interval: int = 12,
) -> dict[str, Any]:
    trim_start = int(frames_trim[0] or 0)
    trim_stop = int(frames_trim[1])
    duration = (trim_stop - trim_start) / float(fps)
    intervals = [_normalize_interval(value, duration) for value in candidate_intervals]
    indices_by_row = []
    for start_s, end_s in intervals:
        indices_by_row.append(
            [
                min(
                    trim_stop - 1,
                    trim_start
                    + round((start_s + position * (end_s - start_s) / max(frames_per_interval - 1, 1)) * fps),
                )
                for position in range(frames_per_interval)
            ]
        )
    requested = {index for row in indices_by_row for index in row}
    decoded: dict[int, Image.Image] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                decoded[index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if index >= max(requested):
                break
    if requested != set(decoded):
        raise ValueError(f"could not decode all action-conditioned frames from {video_path}")

    tile_w, tile_h, label_h = 160, 112, 20
    canvas = Image.new(
        "RGB",
        (tile_w * frames_per_interval, (tile_h + label_h) * len(intervals)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, indices in enumerate(indices_by_row):
        for column, index in enumerate(indices):
            x = column * tile_w
            y = row * (tile_h + label_h)
            canvas.paste(decoded[index].resize((tile_w, tile_h), Image.Resampling.BILINEAR), (x, y))
            time_s = (index - trim_start) / float(fps)
            draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
            draw.text((x + 3, y + tile_h + 3), f"{role} P{row + 1} {time_s:.2f}s", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return {
        "role": role,
        "selection_method": "two_independent_label_blind_action_interval_proposals_from_v2",
        "candidate_intervals_s": [[round(a, 3), round(b, 3)] for a, b in intervals],
        "frames_per_interval": frames_per_interval,
        "local_duration_s": round(duration, 3),
    }


class TargetedSymmetricComparator:
    def __init__(self, *, model: str = "gpt-5.4", timeout_s: int = 300) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def _one_pass(
        self,
        *,
        global_a: Path,
        targeted_a: Path,
        global_b: Path,
        targeted_b: Path,
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
            "You are the final blinded verifier for a fine-grained surgical video difference. Images 1 and 2 are "
            "video A: a global timeline and dense chronological strips from two independently proposed relevant "
            "intervals. Images 3 and 4 are the corresponding views for video B. The proposal stage had no labels and "
            "its previous conclusions are hidden. Judge from pixels only. First describe A and B independently, then "
            f"compare the exact statement {candidate_difference!r}. Shared action: {action_description!r}. For speed, "
            "force, movement, contact, angle, or radiality, explicitly use change across adjacent frames. Score visible "
            "support from 0-100 per video. Select c when the evidence is equal, absent, or too ambiguous; do not force "
            "a directional answer. Ignore duration, image order, and subject identity. Return valid local intervals.\n"
            f"PASS={pass_name}\nA={json.dumps(metadata_a, ensure_ascii=False)}\n"
            f"B={json.dumps(metadata_b, ensure_ascii=False)}"
        )
        with tempfile.TemporaryDirectory(prefix="viddiff_v3_cli_") as directory:
            temp_root = Path(directory)
            schema_path = temp_root / "schema.json"
            output_path = temp_root / "output.json"
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
            for image_path in (global_a, targeted_a, global_b, targeted_b):
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
            if completed.returncode:
                raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "measurement_id": f"VIDDIFF-V3-{uuid.uuid4().hex}",
            "pass_name": pass_name,
            "usage": _usage_from_events(completed.stdout),
            **payload,
        }

    def compare(self, **kwargs: Any) -> dict[str, Any]:
        forward = self._one_pass(pass_name="forward_A_then_B", **kwargs)
        swapped = self._one_pass(
            global_a=kwargs["global_b"],
            targeted_a=kwargs["targeted_b"],
            global_b=kwargs["global_a"],
            targeted_b=kwargs["targeted_a"],
            action_description=kwargs["action_description"],
            candidate_difference=kwargs["candidate_difference"],
            metadata_a=kwargs["metadata_b"],
            metadata_b=kwargs["metadata_a"],
            pass_name="swapped_B_then_A",
        )
        mapped = {"a": "b", "b": "a", "c": "c"}[swapped["preferred_video"]]
        forward_delta = float(forward["statement_score_a"]) - float(forward["statement_score_b"])
        swapped_delta = float(swapped["statement_score_b"]) - float(swapped["statement_score_a"])
        mean_delta = (forward_delta + swapped_delta) / 2.0
        consistent = forward["preferred_video"] == mapped
        if consistent:
            final = forward["preferred_video"]
            rule = "targeted_order_consistent"
        elif abs(mean_delta) >= 20 and (forward_delta > 0) == (swapped_delta > 0):
            final = "a" if mean_delta > 0 else "b"
            rule = "targeted_consistent_score_margin"
        else:
            final = "c"
            rule = "targeted_disagreement_abstain"
        return {
            "preferred_video": final,
            "decision_rule": rule,
            "order_consistent": consistent,
            "mean_score_delta_a_minus_b": round(mean_delta, 3),
            "video_a_interval_s": forward["video_a_interval_s"],
            "video_b_interval_s": forward["video_b_interval_s"],
            "confidence": round((float(forward["confidence"]) + float(swapped["confidence"])) / 2.0, 3),
            "forward": forward,
            "swapped": swapped,
            "usage": {
                key: int(forward["usage"].get(key, 0)) + int(swapped["usage"].get(key, 0))
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    v2 = json.loads(args.v2_report.read_text(encoding="utf-8"))
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
    if v2.get("seed") != args.seed or v2.get("source_sample_size", v2.get("sample_size")) != args.sample_size:
        raise ValueError("v2 report must use the same seed and source sample size")
    v2_cases = {item["sample_key"]: item for item in v2["cases"]}
    missing = {row["sample_key"] for row in selected} - set(v2_cases)
    if missing:
        raise ValueError(f"v2 report is missing selected cases: {sorted(missing)}")
    suffix = _case_index_suffix(case_indices)
    run_id = f"viddiff_v3_surgery_seed_{args.seed}_n{len(selected)}{suffix}"
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    comparator = TargetedSymmetricComparator(model=args.model) if args.run_cli else None
    resumed = _resume_case_map(
        output_root / "probe_report.json",
        enabled=getattr(args, "resume", False),
        require_measurement=comparator is not None,
    )
    results = []
    resumed_cases = 0
    for row in selected:
        key = row["selected_difference_key"]
        descriptor = row["differences_annotated"][key]["description"]
        prior_result = resumed.get(row["sample_key"])
        if (
            prior_result is not None
            and prior_result.get("selected_difference_key") == key
            and prior_result.get("candidate_difference") == descriptor
        ):
            results.append(prior_result)
            resumed_cases += 1
            continue
        previous = v2_cases[row["sample_key"]]
        case_root = output_root / row["sample_key"]
        assets = []
        mapped = previous["measurement"]["mapped_swapped_intervals"]
        for index, (role, video) in enumerate(zip(("A", "B"), row["videos"])):
            interval_key = f"video_{role.lower()}_interval_s"
            proposals = [
                previous["measurement"]["forward"][interval_key],
                mapped[interval_key],
            ]
            assets.append(
                build_targeted_strip(
                    video_path=VIDDIFF_ROOT / video["path"],
                    frames_trim=video["frames_trim"],
                    fps=float(video["fps_original"]),
                    role=role,
                    candidate_intervals=proposals,
                    output_path=case_root / f"video_{role.lower()}_targeted.jpg",
                )
            )
        v2_root = args.v2_report.parent
        result: dict[str, Any] = {
            "case_id": f"viddiff-{row['sample_key']}",
            "sample_key": row["sample_key"],
            "selected_difference_key": key,
            "candidate_difference": descriptor,
            "targeted_assets": assets,
            "status": "ready",
        }
        if comparator is not None:
            try:
                measurement = comparator.compare(
                    global_a=v2_root / row["sample_key"] / "video_a_global.jpg",
                    targeted_a=case_root / "video_a_targeted.jpg",
                    global_b=v2_root / row["sample_key"] / "video_b_global.jpg",
                    targeted_b=case_root / "video_b_targeted.jpg",
                    action_description=row["action_description"],
                    candidate_difference=result["candidate_difference"],
                    metadata_a=assets[0],
                    metadata_b=assets[1],
                )
                oracle = row["differences_gt"][key]
                interval_valid = _interval_valid(
                    measurement["video_a_interval_s"], assets[0]["local_duration_s"]
                ) and _interval_valid(measurement["video_b_interval_s"], assets[1]["local_duration_s"])
                result.update(
                    {
                        "status": "passed" if measurement["preferred_video"] == oracle and interval_valid else "failed",
                        "measurement": measurement,
                        "oracle_direction": oracle,
                        "direction_correct": measurement["preferred_video"] == oracle,
                        "interval_format_valid": interval_valid,
                    }
                )
            except Exception as error:
                result.update({"status": "execution_failed", "error_type": type(error).__name__, "error": str(error)})
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
        "method": "two_label_blind_interval_proposals_then_dense_targeted_symmetric_verification",
        "label_access_policy": "v2 proposals and v3 tool never use retrieval_frames or differences_gt; gt only scores final result",
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
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--v2-report", type=Path, default=DEFAULT_V2_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--run-cli", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
