"""VidDiffBench surgery canary using a real CLI vision comparison tool."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from cli_teacher_rollout import DEFAULT_CODEX_JS, DEFAULT_NODE, _usage_from_events
from pipeline import ROOT, write_json

import av
import pandas as pd


VIDDIFF_ROOT = ROOT.parent / "external" / "VidDiffBench"
PARQUET_PATH = VIDDIFF_ROOT / "data" / "test-00000-of-00001.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "probes"
_FORBIDDEN_TEACHER_KEYS = {
    "candidate_report",
    "differences_gt",
    "ground_truth",
    "local_path",
    "metrics",
    "oracle_boxes",
    "oracle_direction",
    "url",
    "urls",
    "verifier",
    "verifier_result",
    "video_path",
}
_PATH_OR_URL = re.compile(r"https?://|[A-Za-z]:[\\/]|\\\\[^\\]", re.IGNORECASE)


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def available_surgery_rows() -> list[dict[str, Any]]:
    table = pd.read_parquet(PARQUET_PATH)
    rows: list[dict[str, Any]] = []
    for payload in table[table["source_dataset"] == "JIGSAWS"].to_dict("records"):
        videos = _json_value(payload["videos"])
        if not all((VIDDIFF_ROOT / item["path"]).exists() for item in videos):
            continue
        payload["videos"] = videos
        payload["differences_annotated"] = _json_value(payload["differences_annotated"])
        payload["differences_gt"] = _json_value(payload["differences_gt"])
        rows.append(payload)
    return rows


def select_cases(seed: int, sample_size: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = rng.sample(available_surgery_rows(), sample_size)
    selected = []
    for row in rows:
        keys = sorted(
            key
            for key, label in row["differences_gt"].items()
            if label in {"a", "b", "c"} and row["differences_annotated"].get(key)
        )
        difference_key = rng.choice(keys)
        selected.append({**row, "selected_difference_key": difference_key})
    return selected


def bind_frozen_selection(
    report: dict[str, Any],
    available_rows: list[dict[str, Any]],
    *,
    seed: int,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Rebind a dry-run case/statement manifest to local media rows without reselection."""

    if int(report.get("seed", -1)) != seed:
        raise ValueError("selection report seed does not match --seed")
    if int(report.get("sample_size", -1)) != sample_size:
        raise ValueError("selection report sample_size does not match --sample-size")
    frozen_cases = list(report.get("cases") or [])
    if len(frozen_cases) != sample_size:
        raise ValueError("selection report does not contain the declared sample_size")
    rows_by_key = {str(row.get("sample_key")): row for row in available_rows}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frozen in frozen_cases:
        sample_key = str(frozen.get("sample_key") or "")
        if not sample_key or sample_key in seen:
            raise ValueError(f"invalid or duplicate frozen sample_key: {sample_key!r}")
        seen.add(sample_key)
        row = rows_by_key.get(sample_key)
        if row is None:
            raise ValueError(f"frozen selection lacks a local media row: {sample_key}")
        difference_key = str(frozen.get("selected_difference_key") or "")
        descriptor = (row.get("differences_annotated") or {}).get(difference_key) or {}
        current_statement = str(descriptor.get("description") or "")
        frozen_statement = str(frozen.get("candidate_difference") or "")
        if not difference_key or current_statement != frozen_statement:
            raise ValueError(
                "frozen comparison statement does not match the local annotation: "
                f"{sample_key}:{difference_key}"
            )
        selected.append({**row, "selected_difference_key": difference_key})
    return selected


def select_cases_for_run(
    *,
    seed: int,
    sample_size: int,
    selection_report: Path | None,
) -> list[dict[str, Any]]:
    if selection_report is None:
        return select_cases(seed, sample_size)
    report = json.loads(selection_report.read_text(encoding="utf-8"))
    return bind_frozen_selection(
        report,
        available_surgery_rows(),
        seed=seed,
        sample_size=sample_size,
    )


def _case_index_suffix(case_indices: list[int] | None) -> str:
    if not case_indices:
        return ""
    expanded = "_idx" + "-".join(map(str, case_indices))
    if len(expanded) <= 80:
        return expanded
    return f"_subset{len(case_indices)}_range{min(case_indices)}-{max(case_indices)}"


def _validate_teacher_payload(
    *,
    action_description: str,
    candidate_difference: str,
    metadata: dict[str, Any],
) -> None:
    """Reject private or verifier-only values immediately before a CLI upload."""

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).strip().lower()
                if normalized in _FORBIDDEN_TEACHER_KEYS:
                    raise ValueError(f"forbidden outbound key at {location}.{key}")
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, str) and _PATH_OR_URL.search(value):
            raise ValueError(f"path or URL detected in outbound value at {location}")

    walk(action_description, "action_description")
    walk(candidate_difference, "candidate_difference")
    walk(metadata, "metadata")


def _resume_case_map(
    report_path: Path,
    *,
    enabled: bool,
    require_measurement: bool,
) -> dict[str, dict[str, Any]]:
    if not enabled or not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reusable: dict[str, dict[str, Any]] = {}
    for case in report.get("cases", []):
        status = case.get("status")
        complete = status in {"passed", "failed"}
        if not require_measurement:
            complete = complete or status == "ready"
        if complete and (not require_measurement or isinstance(case.get("measurement"), dict)):
            reusable[str(case["sample_key"])] = case
    return reusable


def _sample_frames(video_path: Path, frames_trim: list[Any], count: int) -> list[tuple[Image.Image, int]]:
    start = int(frames_trim[0] or 0)
    stop = int(frames_trim[1]) if frames_trim[1] is not None else None
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total = int(stream.frames or 0)
        if stop is None:
            stop = total
        if stop <= start:
            raise ValueError(f"invalid frame trim {frames_trim} for {video_path}")
        indices = [
            round(start + position * (stop - start - 1) / max(count - 1, 1))
            for position in range(count)
        ]
        requested = set(indices)
        frames: dict[int, Image.Image] = {}
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                frames[index] = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if index >= indices[-1]:
                break
    if set(indices) != set(frames):
        raise ValueError(f"could not decode requested frames from {video_path}")
    return [(frames[index], index) for index in indices]


def build_mosaic(
    *,
    video_path: Path,
    frames_trim: list[Any],
    fps: float,
    role: str,
    output_path: Path,
    count: int = 12,
) -> dict[str, Any]:
    frames = _sample_frames(video_path, frames_trim, count)
    tile_w, tile_h, label_h, columns = 320, 240, 24, 4
    rows = (len(frames) + columns - 1) // columns
    mosaic = Image.new("RGB", (tile_w * columns, (tile_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(mosaic)
    local_times = []
    for position, (image, frame_index) in enumerate(frames):
        x = (position % columns) * tile_w
        y = (position // columns) * (tile_h + label_h)
        image = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        mosaic.paste(image, (x, y))
        local_time = (frame_index - int(frames_trim[0] or 0)) / float(fps)
        local_times.append(round(local_time, 3))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill="white")
        draw.text((x + 5, y + tile_h + 5), f"{role} tile {position + 1}: t={local_time:.2f}s", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path, quality=92)
    return {
        "role": role,
        "sample_count": len(frames),
        "local_times_s": local_times,
        "local_duration_s": round((int(frames_trim[1]) - int(frames_trim[0])) / float(fps), 3),
        "sampling_method": "uniform_over_dataset_trim_without_retrieval_frame_annotations",
    }


def measurement_schema() -> dict[str, Any]:
    interval = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "preferred_video": {"type": "string", "enum": ["a", "b", "c"]},
            "video_a_interval_s": interval,
            "video_b_interval_s": interval,
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "visual_rationale": {"type": "string", "minLength": 48},
        },
        "required": [
            "preferred_video",
            "video_a_interval_s",
            "video_b_interval_s",
            "confidence",
            "visual_rationale",
        ],
        "additionalProperties": False,
    }


class CodexCLIVisionComparator:
    source = "codex_cli_vision_tool"

    def __init__(self, *, model: str = "gpt-5.4", timeout_s: int = 300) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def compare(
        self,
        *,
        mosaic_a: Path,
        mosaic_b: Path,
        action_description: str,
        candidate_difference: str,
        metadata_a: dict[str, Any],
        metadata_b: dict[str, Any],
    ) -> dict[str, Any]:
        _validate_teacher_payload(
            action_description=action_description,
            candidate_difference=candidate_difference,
            metadata={"video_a": metadata_a, "video_b": metadata_b},
        )
        prompt = (
            "You are the visual measurement backend of a video-comparison tool. Do not use shell, files, web, "
            "or any other tool. Inspect only the two attached chronological frame mosaics. The first image is "
            "video A and the second is video B. Each tile is labeled with local time. Evaluate whether the "
            f"comparative statement is more true in A, more true in B, or neither: {candidate_difference!r}. "
            f"The shared action is: {action_description!r}. Return preferred_video='a', 'b', or 'c' (c means "
            "neither/equal/visually indeterminate). Give the most relevant visible local interval in each video; "
            "interval endpoints must stay within the supplied sampled duration. Explain only visible comparative "
            "evidence and state uncertainty honestly.\n"
            f"VIDEO_A_SAMPLING={json.dumps(metadata_a, ensure_ascii=False)}\n"
            f"VIDEO_B_SAMPLING={json.dumps(metadata_b, ensure_ascii=False)}"
        )
        with tempfile.TemporaryDirectory(prefix="viddiff_cli_vision_") as directory:
            temp_root = Path(directory)
            schema_path = temp_root / "measurement.schema.json"
            output_path = temp_root / "measurement.json"
            schema_path.write_text(json.dumps(measurement_schema()), encoding="utf-8")
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
                "--image",
                str(mosaic_a.resolve()),
                "--image",
                str(mosaic_b.resolve()),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--model",
                self.model,
                "-",
            ]
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
                raise RuntimeError(f"Codex CLI vision comparison failed: {diagnostic}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "measurement_id": f"VIDDIFF-MEASURE-{uuid.uuid4().hex}",
            "model": self.model,
            "usage": _usage_from_events(completed.stdout),
            **payload,
        }


def _interval_valid(interval: Any, duration: float) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
        and 0.0 <= float(interval[0]) < float(interval[1]) <= duration + 0.05
    )


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
    run_id = f"viddiff_surgery_seed_{args.seed}_n{len(selected)}{suffix}"
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    comparator = CodexCLIVisionComparator(model=args.model) if args.run_cli else None
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
        mosaics = []
        for role, video in zip(("A", "B"), row["videos"]):
            mosaics.append(
                build_mosaic(
                    video_path=VIDDIFF_ROOT / video["path"],
                    frames_trim=video["frames_trim"],
                    fps=float(video["fps_original"]),
                    role=role,
                    output_path=case_root / f"video_{role.lower()}_mosaic.jpg",
                    count=args.frame_count,
                )
            )
        result = {
            "case_id": f"viddiff-{row['sample_key']}",
            "sample_key": row["sample_key"],
            "source_dataset": row["source_dataset"],
            "split": row["split"],
            "action_name": row["action_name"],
            "action_description": row["action_description"],
            "selected_difference_key": key,
            "candidate_difference": descriptor["description"],
            "sampling": mosaics,
            "status": "ready",
        }
        if comparator is not None:
            try:
                measurement = comparator.compare(
                    mosaic_a=case_root / "video_a_mosaic.jpg",
                    mosaic_b=case_root / "video_b_mosaic.jpg",
                    action_description=row["action_description"],
                    candidate_difference=descriptor["description"],
                    metadata_a=mosaics[0],
                    metadata_b=mosaics[1],
                )
                oracle_direction = row["differences_gt"][key]
                interval_format_valid = _interval_valid(
                    measurement["video_a_interval_s"], mosaics[0]["local_duration_s"]
                ) and _interval_valid(measurement["video_b_interval_s"], mosaics[1]["local_duration_s"])
                result.update({
                    "status": "passed" if measurement["preferred_video"] == oracle_direction and interval_format_valid else "failed",
                    "measurement": measurement,
                    "oracle_direction": oracle_direction,
                    "direction_correct": measurement["preferred_video"] == oracle_direction,
                    "interval_format_valid": interval_format_valid,
                })
            except Exception as error:
                result.update({"status": "execution_failed", "error_type": type(error).__name__, "error": str(error)})
        results.append(result)
        write_json(output_root / "probe_report.json", {"run_id": run_id, "cases": results})
    direction_correct = sum(item.get("direction_correct") is True for item in results)
    interval_valid = sum(item.get("interval_format_valid") is True for item in results)
    report = {
        "run_id": run_id,
        "dataset": "VidDiffBench",
        "subset": "JIGSAWS surgery hard",
        "seed": args.seed,
        "sample_size": len(selected),
        "source_sample_size": args.sample_size,
        "case_indices": case_indices,
        "cli_requested": args.run_cli,
        "summary": {
            "direction_correct": direction_correct,
            "interval_format_valid": interval_valid,
            "passed": sum(item["status"] == "passed" for item in results),
            "execution_failed": sum(item["status"] == "execution_failed" for item in results),
            "resumed_cases": resumed_cases,
            "executed_cases_current_run": len(results) - resumed_cases,
            "safe_to_compile_sft": (
                bool(selected)
                and direction_correct / len(selected) >= 0.8
                and interval_valid == len(selected)
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
    parser.add_argument("--frame-count", type=int, default=12)
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
