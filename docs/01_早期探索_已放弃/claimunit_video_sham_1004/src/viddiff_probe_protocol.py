"""Predicate-conditioned, order-symmetric VidDiff measurement canary."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from cli_teacher_rollout import DEFAULT_CODEX_JS, DEFAULT_NODE, _usage_from_events
from pipeline import write_json
from viddiff_predicate_observer import (
    adjudicate_predicate_observations,
    predicate_observer_schema,
    swap_adjudication,
)
from viddiff_predicate_protocol import compile_protocol
from viddiff_probe import (
    DEFAULT_OUTPUT_ROOT,
    VIDDIFF_ROOT,
    _case_index_suffix,
    _interval_valid,
    _resume_case_map,
    _sample_frames,
    _validate_teacher_payload,
    select_cases_for_run,
)


def build_protocol_timeline(
    *,
    video_path: Path,
    frames_trim: list[Any],
    fps: float,
    role: str,
    output_path: Path,
    frame_count: int = 48,
) -> dict[str, Any]:
    frames = _sample_frames(video_path, frames_trim, frame_count)
    columns = 8
    tile_w, tile_h, label_h = 240, 180, 20
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    local_times: list[float] = []
    trim_start = int(frames_trim[0] or 0)
    for position, (frame, frame_index) in enumerate(frames):
        x = (position % columns) * tile_w
        y = (position // columns) * (tile_h + label_h)
        canvas.paste(frame.resize((tile_w, tile_h), Image.Resampling.BILINEAR), (x, y))
        local_time = (frame_index - trim_start) / float(fps)
        local_times.append(round(local_time, 3))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
        draw.text((x + 3, y + tile_h + 3), f"{role} F{position + 1} {local_time:.2f}s", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)
    return {
        "role": role,
        "selection_method": "label_blind_exhaustive_uniform_protocol_timeline",
        "sample_count": len(frames),
        "local_times_s": local_times,
        "local_duration_s": round(
            (int(frames_trim[1]) - trim_start) / float(fps), 3
        ),
    }


class ProtocolSymmetricComparator:
    source = "codex_cli_predicate_protocol_symmetric_vision_tool"

    def __init__(self, *, model: str = "gpt-5.4", timeout_s: int = 300) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def _one_pass(
        self,
        *,
        timeline_a: Path,
        motion_a: Path,
        targeted_a: Path,
        timeline_b: Path,
        motion_b: Path,
        targeted_b: Path,
        action_description: str,
        candidate_difference: str,
        protocol: dict[str, Any],
        metadata_a: dict[str, Any],
        metadata_b: dict[str, Any],
        pass_name: str,
    ) -> dict[str, Any]:
        outbound_metadata = {
            "protocol": protocol,
            "video_a": metadata_a,
            "video_b": metadata_b,
        }
        _validate_teacher_payload(
            action_description=action_description,
            candidate_difference=candidate_difference,
            metadata=outbound_metadata,
        )
        prompt = (
            "You are a blinded predicate-observation backend. Use only the six attached derived "
            "images. Images 1-3 belong to video A: an exhaustive chronological timeline, "
            "label-blind motion windows, and dense targeted detail. Images 4-6 are the same "
            "views for video B. Never infer skill, identity, hidden force, unseen frames, or a "
            "dataset direction. Do not choose A, B, or C and do not emit support scores. Instead "
            "fill one typed observation record for each video independently. Follow every "
            "proof_obligation and disallowed_shortcut in the protocol. scalar_value must be the "
            "protocol target_variable in its stated units; use event_count only for count or "
            "persistent-event predicates and violation_count only for visibility violations. "
            "observable=false when required landmarks, event boundaries, or visible proxies are "
            "not resolved. cue_times_s and counterexample_times_s must cite concrete local times. "
            "The deterministic harness, not you, compares the two records.\n"
            f"STATEMENT={candidate_difference!r}\n"
            f"SHARED_ACTION={action_description!r}\n"
            f"PROTOCOL={json.dumps(protocol, ensure_ascii=False)}\n"
            f"PASS={pass_name}\n"
            f"VIDEO_A_METADATA={json.dumps(metadata_a, ensure_ascii=False)}\n"
            f"VIDEO_B_METADATA={json.dumps(metadata_b, ensure_ascii=False)}"
        )
        with tempfile.TemporaryDirectory(prefix="viddiff_protocol_cli_") as directory:
            temp_root = Path(directory)
            schema_path = temp_root / "schema.json"
            output_path = temp_root / "output.json"
            schema_path.write_text(json.dumps(predicate_observer_schema()), encoding="utf-8")
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
            for image_path in (
                timeline_a,
                motion_a,
                targeted_a,
                timeline_b,
                motion_b,
                targeted_b,
            ):
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
            "measurement_id": f"VIDDIFF-PROTOCOL-{uuid.uuid4().hex}",
            "pass_name": pass_name,
            "usage": _usage_from_events(completed.stdout),
            **payload,
        }

    def compare(self, **kwargs: Any) -> dict[str, Any]:
        forward = self._one_pass(pass_name="forward_A_then_B", **kwargs)
        swapped = self._one_pass(
            timeline_a=kwargs["timeline_b"],
            motion_a=kwargs["motion_b"],
            targeted_a=kwargs["targeted_b"],
            timeline_b=kwargs["timeline_a"],
            motion_b=kwargs["motion_a"],
            targeted_b=kwargs["targeted_a"],
            action_description=kwargs["action_description"],
            candidate_difference=kwargs["candidate_difference"],
            protocol=kwargs["protocol"],
            metadata_a=kwargs["metadata_b"],
            metadata_b=kwargs["metadata_a"],
            pass_name="swapped_B_then_A",
        )
        forward_decision = adjudicate_predicate_observations(
            kwargs["protocol"], forward["video_a"], forward["video_b"]
        )
        swapped_decision = swap_adjudication(
            adjudicate_predicate_observations(
                kwargs["protocol"], swapped["video_a"], swapped["video_b"]
            )
        )
        consistent = (
            forward_decision["preferred_video"]
            == swapped_decision["preferred_video"]
        )
        if consistent:
            preferred = forward_decision["preferred_video"]
            rule = "typed_predicate_order_consistent"
        else:
            preferred = "c"
            rule = "typed_predicate_order_disagreement_abstain"
        metric_pairs = [
            value
            for value in (
                forward_decision.get("metric_a"),
                forward_decision.get("metric_b"),
                swapped_decision.get("metric_a"),
                swapped_decision.get("metric_b"),
            )
            if value is not None
        ]
        mean_metric_delta = None
        if len(metric_pairs) == 4:
            mean_metric_delta = round(
                (
                    float(forward_decision["metric_a"])
                    - float(forward_decision["metric_b"])
                    + float(swapped_decision["metric_a"])
                    - float(swapped_decision["metric_b"])
                )
                / 2.0,
                6,
            )
        return {
            "preferred_video": preferred,
            "decision_rule": rule,
            "order_consistent": consistent,
            "protocol_family": kwargs["protocol"]["family"],
            "mean_typed_metric_delta_a_minus_b": mean_metric_delta,
            "video_a_interval_s": forward["video_a"]["interval_s"],
            "video_b_interval_s": forward["video_b"]["interval_s"],
            "confidence": (
                round(
                    (
                        float(forward_decision["confidence"])
                        + float(swapped_decision["confidence"])
                    )
                    / 2.0,
                    3,
                )
                if consistent
                else 0.0
            ),
            "forward": {"raw_observations": forward, "adjudication": forward_decision},
            "swapped": {"raw_observations": swapped, "adjudication": swapped_decision},
            "usage": {
                key: int(forward["usage"].get(key, 0))
                + int(swapped["usage"].get(key, 0))
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_cases_for_run(
        seed=args.seed,
        sample_size=args.sample_size,
        selection_report=args.selection_report,
    )
    if args.case_indices:
        selected = [selected[index] for index in args.case_indices]
    v2 = json.loads(args.v2_report.read_text(encoding="utf-8"))
    v3 = json.loads(args.v3_report.read_text(encoding="utf-8"))
    v2_cases = {item["sample_key"]: item for item in v2.get("cases") or []}
    v3_cases = {item["sample_key"]: item for item in v3.get("cases") or []}
    missing = {item["sample_key"] for item in selected} - (set(v2_cases) & set(v3_cases))
    if missing:
        raise ValueError(f"source reports miss frozen cases: {sorted(missing)}")
    suffix = _case_index_suffix(args.case_indices)
    run_id = f"viddiff_typed_protocol_v2_surgery_seed_{args.seed}_n{len(selected)}{suffix}"
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    comparator = ProtocolSymmetricComparator(model=args.model) if args.run_cli else None
    resumed = _resume_case_map(
        output_root / "probe_report.json",
        enabled=args.resume,
        require_measurement=comparator is not None,
    )
    results: list[dict[str, Any]] = []
    resumed_cases = 0
    for row in selected:
        key = str(row["selected_difference_key"])
        statement = str(row["differences_annotated"][key]["description"])
        previous = resumed.get(str(row["sample_key"]))
        if previous and previous.get("candidate_difference") == statement:
            results.append(previous)
            resumed_cases += 1
            continue
        case_root = output_root / row["sample_key"]
        protocol = compile_protocol(statement)
        timelines: list[dict[str, Any]] = []
        for role, video in zip(("A", "B"), row["videos"]):
            timelines.append(
                build_protocol_timeline(
                    video_path=VIDDIFF_ROOT / video["path"],
                    frames_trim=video["frames_trim"],
                    fps=float(video["fps_original"]),
                    role=role,
                    output_path=case_root / f"video_{role.lower()}_protocol_timeline.jpg",
                    frame_count=args.timeline_frame_count,
                )
            )
        v2_case = v2_cases[row["sample_key"]]
        v3_case = v3_cases[row["sample_key"]]
        metadata = [
            {
                "timeline": timelines[index],
                "motion": v2_case["visual_assets"][index]["motion"],
                "targeted": v3_case["targeted_assets"][index],
            }
            for index in range(2)
        ]
        result: dict[str, Any] = {
            "case_id": f"viddiff-{row['sample_key']}",
            "sample_key": row["sample_key"],
            "action_name": row["action_name"],
            "action_description": row["action_description"],
            "selected_difference_key": key,
            "candidate_difference": statement,
            "protocol": protocol,
            "protocol_assets": metadata,
            "status": "ready",
        }
        if comparator is not None:
            try:
                v2_root = args.v2_report.parent / row["sample_key"]
                v3_root = args.v3_report.parent / row["sample_key"]
                measurement = comparator.compare(
                    timeline_a=case_root / "video_a_protocol_timeline.jpg",
                    motion_a=v2_root / "video_a_motion.jpg",
                    targeted_a=v3_root / "video_a_targeted.jpg",
                    timeline_b=case_root / "video_b_protocol_timeline.jpg",
                    motion_b=v2_root / "video_b_motion.jpg",
                    targeted_b=v3_root / "video_b_targeted.jpg",
                    action_description=row["action_description"],
                    candidate_difference=statement,
                    protocol=protocol,
                    metadata_a=metadata[0],
                    metadata_b=metadata[1],
                )
                oracle = row["differences_gt"][key]
                interval_valid = _interval_valid(
                    measurement["video_a_interval_s"], timelines[0]["local_duration_s"]
                ) and _interval_valid(
                    measurement["video_b_interval_s"], timelines[1]["local_duration_s"]
                )
                result.update(
                    {
                        "status": "passed"
                        if measurement["preferred_video"] == oracle and interval_valid
                        else "failed",
                        "measurement": measurement,
                        "oracle_direction": oracle,
                        "direction_correct": measurement["preferred_video"] == oracle,
                        "interval_format_valid": interval_valid,
                    }
                )
            except Exception as error:
                result.update(
                    {
                        "status": "execution_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        results.append(result)
        write_json(output_root / "probe_report.json", {"run_id": run_id, "cases": results})
    correct = sum(item.get("direction_correct") is True for item in results)
    valid = sum(item.get("interval_format_valid") is True for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VidDiffBench",
        "seed": args.seed,
        "sample_size": len(selected),
        "source_sample_size": args.sample_size,
        "case_indices": args.case_indices,
        "method": "typed_predicate_observation_then_deterministic_symmetric_adjudication_v2",
        "label_access_policy": "vision records contain no direction; deterministic label-blind adjudication runs before gt evaluation",
        "summary": {
            "direction_correct": correct,
            "interval_format_valid": valid,
            "passed": sum(item["status"] == "passed" for item in results),
            "execution_failed": sum(item["status"] == "execution_failed" for item in results),
            "resumed_cases": resumed_cases,
            "executed_cases_current_run": len(results) - resumed_cases,
            "order_consistent": sum(
                item.get("measurement", {}).get("order_consistent") is True
                for item in results
            ),
            "safe_to_register_tool": bool(selected)
            and correct / len(selected) >= 0.8
            and valid == len(selected),
        },
        "cases": results,
    }
    write_json(output_root / "probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--case-indices", nargs="+", type=int, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--v3-report", type=Path, required=True)
    parser.add_argument("--timeline-frame-count", type=int, default=48)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-cli", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args)["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
