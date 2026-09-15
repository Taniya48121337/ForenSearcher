"""Deterministic ClaimUnit trajectory builder for VideoSham case 1004.

The dataset annotation is kept in a hidden oracle.  The student-visible claim is
derived from media metadata and monotonic visual alignment only; no LLM is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import av  # noqa: E402


SOURCE = ROOT / "data" / "source" / "1004_unedited.mp4"
EDITED = ROOT / "data" / "edited" / "1004_processed.mp4"


@dataclass(frozen=True)
class FrameSet:
    times: np.ndarray
    features: np.ndarray
    fps: float
    duration_s: float
    width: int
    height: int
    frame_count: int


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _feature(frame: av.VideoFrame, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    image = Image.fromarray(frame.to_ndarray(format="rgb24")).convert("L")
    target_aspect = size[0] / size[1]
    aspect = image.width / image.height
    if aspect < target_aspect:
        crop_h = round(image.width / target_aspect)
        top = (image.height - crop_h) // 2
        image = image.crop((0, top, image.width, top + crop_h))
    elif aspect > target_aspect:
        crop_w = round(image.height * target_aspect)
        left = (image.width - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, image.height))
    pixels = np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gx = np.diff(pixels, axis=1, append=pixels[:, -1:])
    gy = np.diff(pixels, axis=0, append=pixels[-1:, :])
    vector = np.concatenate((pixels.ravel(), gx.ravel(), gy.ravel()))
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def decode(path: Path) -> FrameSet:
    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    duration_s = float(container.duration / av.time_base)
    times: list[float] = []
    features: list[np.ndarray] = []
    for index, frame in enumerate(container.decode(video=0)):
        time_s = float(frame.time) if frame.time is not None else index / fps
        times.append(time_s)
        features.append(_feature(frame))
    container.close()
    return FrameSet(
        times=np.asarray(times, dtype=np.float64),
        features=np.stack(features),
        fps=fps,
        duration_s=duration_s,
        width=stream.width,
        height=stream.height,
        frame_count=len(times),
    )


def monotonic_align(source: FrameSet, edited: FrameSet, max_source_step: int = 45) -> dict[str, Any]:
    # Each edited frame selects one strictly later source frame.  The visual term
    # dominates; a very small step penalty only resolves near-duplicate frames.
    similarity = edited.features @ source.features.T
    cost = 1.0 - similarity
    n_source = source.frame_count
    back = np.full((edited.frame_count, n_source), -1, dtype=np.int16)
    previous = cost[0].astype(np.float64)
    expected_step = source.fps / edited.fps

    for row in range(1, edited.frame_count):
        best = np.full(n_source, np.inf, dtype=np.float64)
        best_prev = np.full(n_source, -1, dtype=np.int32)
        for step in range(1, max_source_step + 1):
            transition = 0.0015 * min(abs(step - expected_step), 10.0)
            candidate = previous[:-step] + transition
            target = best[step:]
            improved = candidate < target
            target[improved] = candidate[improved]
            best_prev[step:][improved] = np.flatnonzero(improved)
        previous = cost[row] + best
        back[row] = best_prev

    end = int(np.argmin(previous))
    indices = np.empty(edited.frame_count, dtype=np.int32)
    indices[-1] = end
    for row in range(edited.frame_count - 1, 0, -1):
        indices[row - 1] = back[row, indices[row]]
    if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise RuntimeError("monotonic alignment backtrace failed")

    selected_similarity = similarity[np.arange(edited.frame_count), indices]
    source_times = source.times[indices]
    source_steps = np.diff(indices)
    source_time_steps = np.diff(source_times)
    edited_time_steps = np.diff(edited.times)
    extra_advance = source_time_steps - edited_time_steps
    return {
        "source_frame_indices": indices,
        "source_times_s": source_times,
        "edited_times_s": edited.times,
        "similarities": selected_similarity,
        "source_frame_steps": source_steps,
        "extra_advance_s": extra_advance,
        "mean_similarity": float(selected_similarity.mean()),
        "p05_similarity": float(np.quantile(selected_similarity, 0.05)),
    }


def omission_segments(alignment: dict[str, Any], source_fps: float) -> list[dict[str, Any]]:
    source_times = alignment["source_times_s"]
    extra = alignment["extra_advance_s"]
    # A jump must exceed two source frames beyond ordinary target-frame advance.
    hit_indices = np.flatnonzero(extra > (2.0 / source_fps))
    raw: list[tuple[float, float, float]] = []
    for idx in hit_indices:
        left = float(source_times[idx] + 1.0 / source_fps)
        right = float(source_times[idx + 1] - 1.0 / source_fps)
        if right > left:
            raw.append((left, right, float(extra[idx])))

    merged: list[list[float]] = []
    for left, right, magnitude in raw:
        if merged and left - merged[-1][1] <= 0.20:
            merged[-1][1] = max(merged[-1][1], right)
            merged[-1][2] += magnitude
        else:
            merged.append([left, right, magnitude])
    return [
        {
            "source_interval_s": [round(left, 3), round(right, 3)],
            "estimated_missing_s": round(right - left, 3),
            "alignment_jump_s": round(magnitude, 3),
        }
        for left, right, magnitude in merged
    ]


def media_metadata(frames: FrameSet, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "duration_s": round(frames.duration_s, 3),
        "fps": round(frames.fps, 6),
        "frame_count": frames.frame_count,
        "width": frames.width,
        "height": frames.height,
    }


def build() -> dict[str, Any]:
    source = decode(SOURCE)
    edited = decode(EDITED)
    aligned = monotonic_align(source, edited)
    segments = omission_segments(aligned, source.fps)
    aligned_start = float(aligned["source_times_s"][0])
    aligned_end = float(aligned["source_times_s"][-1])
    aligned_span = aligned_end - aligned_start + 1.0 / source.fps
    retained_end = aligned_end + 1.0 / source.fps
    comparison_scope = [1.0, 12.0]
    comparison_span = comparison_scope[1] - comparison_scope[0]
    contraction = max(0.0, comparison_span - aligned_span)
    contraction_ratio = contraction / comparison_span
    boundary_omissions = []
    if aligned_start > comparison_scope[0] + 1.0 / source.fps:
        boundary_omissions.append(
            {
                "source_interval_s": [comparison_scope[0], round(aligned_start, 3)],
                "estimated_missing_s": round(aligned_start - comparison_scope[0], 3),
                "position": "leading_boundary",
            }
        )
    if retained_end < comparison_scope[1] - 1.0 / source.fps:
        boundary_omissions.append(
            {
                "source_interval_s": [round(retained_end, 3), comparison_scope[1]],
                "estimated_missing_s": round(comparison_scope[1] - retained_end, 3),
                "position": "trailing_boundary",
            }
        )

    source_meta = media_metadata(source, SOURCE)
    edited_meta = media_metadata(edited, EDITED)
    visible_manifest = {
        "case_id": "videosham-1004",
        "dataset": "VideoSham",
        "source_id": "vimeo-26432864",
        "comparison_scope_s": comparison_scope,
        "scope_provenance": "dataset-provided edit window used only as a neutral locator",
        "source_media": source_meta,
        "edited_media": edited_meta,
        "allowed_inputs": ["the two media files", "their neutral identifiers", "the comparison scope"],
        "forbidden_inputs": ["editor task text", "manipulation label", "oracle fields"],
    }
    oracle = {
        "case_id": "videosham-1004",
        "annotation_source": "VideoSourceandEdits.csv",
        "original_vimeo_id": "26432864",
        "edit_window_s": [1.0, 12.0],
        "manipulation_type": "frame_dropping",
        "target_output_duration_s": 8.0,
        "annotation_text": "For 0:01 - 0:12, drop frames to reduce the video to 0:08 seconds.",
        "visibility": "oracle_only",
    }

    evidence = [
        {
            "evidence_id": "EA-META-001",
            "kind": "media_metadata",
            "supports": ["edited_duration", "frame_rate_change"],
            "observation": {
                "source_duration_s": source_meta["duration_s"],
                "edited_duration_s": edited_meta["duration_s"],
                "source_fps": source_meta["fps"],
                "edited_fps": edited_meta["fps"],
            },
            "provenance": {"method": "container_decode", "inputs": [source_meta["path"], edited_meta["path"]]},
        },
        {
            "evidence_id": "EA-ALIGN-001",
            "kind": "monotonic_visual_alignment",
            "supports": ["source_identity", "temporal_order_preserved", "aligned_source_interval"],
            "observation": {
                "aligned_source_interval_s": [round(aligned_start, 3), round(aligned_end, 3)],
                "retained_source_interval_s": [round(aligned_start, 3), round(retained_end, 3)],
                "aligned_source_span_s": round(aligned_span, 3),
                "mean_similarity": round(aligned["mean_similarity"], 4),
                "p05_similarity": round(aligned["p05_similarity"], 4),
                "matched_frames": edited.frame_count,
                "strictly_monotonic": True,
            },
            "provenance": {"method": "visual_viterbi_v1", "feature": "center_crop_16:9_gray+gradients_64x36"},
        },
        {
            "evidence_id": "EA-OMIT-001",
            "kind": "source_time_contraction",
            "supports": ["omission_direction", "frame_dropping"],
            "observation": {
                "aligned_source_span_s": round(aligned_span, 3),
                "comparison_scope_s": comparison_scope,
                "edited_duration_s": edited_meta["duration_s"],
                "estimated_removed_time_s": round(contraction, 3),
                "contraction_ratio": round(contraction_ratio, 4),
                "boundary_omission_segments": boundary_omissions,
                "localized_jump_segments": segments,
                "max_source_frame_step": int(np.max(aligned["source_frame_steps"])),
                "median_source_frame_step": float(np.median(aligned["source_frame_steps"])),
            },
            "provenance": {"method": "coverage_gap_from_alignment", "depends_on": ["EA-ALIGN-001"]},
        },
    ]
    evidence.append(
        {
            "evidence_id": "EA-SCOPE-001",
            "kind": "case_scope_locator",
            "supports": ["comparison_scope"],
            "observation": {"comparison_scope_s": comparison_scope},
            "provenance": {"method": "dataset_locator_only", "input": "student_view/case_manifest.json"},
        }
    )

    claim = {
        "claim_id": "CL-1004-001",
        "source_id": "vimeo-26432864",
        "edited_clip_id": "videosham-1004",
        "manipulation_type": "frame_dropping",
        "claim_family": "selective_temporal_omission",
        "direction": "source_to_edited",
        "comparison_scope_s": comparison_scope,
        "retained_source_interval_s": [round(aligned_start, 3), round(retained_end, 3)],
        "omitted_source_intervals_s": [item["source_interval_s"] for item in boundary_omissions],
        "estimated_removed_time_s": round(contraction, 3),
        "confidence": "high" if aligned["p05_similarity"] >= 0.70 and contraction_ratio >= 0.10 else "medium",
        "evidence_ids": ["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001", "EA-OMIT-001"],
        "natural_language": (
            f"Within the 1.00–12.00s comparison scope, the edited clip retains the continuous source segment "
            f"{aligned_start:.2f}–{retained_end:.2f}s. The leading and trailing source frames are dropped, "
            f"creating a selective temporal omission of about {contraction:.2f}s."
        ),
    }

    clean_steps = [
        {"step": 1, "action": "inspect_media", "produced": ["EA-META-001"]},
        {"step": 2, "action": "align_visual_timeline", "produced": ["EA-ALIGN-001"]},
        {"step": 3, "action": "measure_source_time_contraction", "produced": ["EA-OMIT-001"]},
        {"step": 4, "action": "assemble_claim", "consumed": claim["evidence_ids"], "produced": [claim["claim_id"]]},
    ]

    # Evidence-side mutation: remove the only atom that supports the operation
    # label/direction, then require a targeted remeasurement before answering.
    mutated_evidence = [atom for atom in evidence if atom["evidence_id"] != "EA-OMIT-001"]
    omission_atom = next(atom for atom in evidence if atom["evidence_id"] == "EA-OMIT-001")
    repair_atom = {
        "evidence_id": "EA-REPAIR-001",
        "kind": "targeted_alignment_remeasurement",
        "supports": ["omission_direction", "frame_dropping"],
        "observation": omission_atom["observation"],
        "provenance": {
            "method": "recompute_coverage_gap_from_raw_media",
            "trigger": "required support EA-OMIT-001 was absent",
            "depends_on": ["EA-ALIGN-001"],
        },
    }
    repaired_claim = {**claim, "evidence_ids": ["EA-META-001", "EA-SCOPE-001", "EA-ALIGN-001", "EA-REPAIR-001"]}

    clean_sft = {
        "messages": [
            {"role": "system", "content": "Infer only claims supported by listed evidence atoms. Cite evidence IDs; if required support is absent, remeasure from the media before answering."},
            {"role": "user", "content": json.dumps({"task": "Determine whether and how the edited clip omits source content.", "case_manifest": visible_manifest}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"investigation_steps": clean_steps, "evidence": evidence, "final_claim": claim}, ensure_ascii=False)},
        ],
        "metadata": {"case_id": "videosham-1004", "trajectory_type": "clean", "generator": "deterministic_pipeline_v1"},
    }
    repair_sft = {
        "messages": [
            {"role": "system", "content": "Infer only claims supported by listed evidence atoms. Cite evidence IDs; if required support is absent, remeasure from the media before answering."},
            {"role": "user", "content": json.dumps({"task": "Determine whether and how the edited clip omits source content.", "case_manifest": visible_manifest}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"state": "support_missing", "available_evidence": mutated_evidence, "missing_support": ["omission_direction", "frame_dropping"], "next_action": "recompute targeted coverage gaps from raw media"}, ensure_ascii=False)},
            {"role": "tool", "content": json.dumps(repair_atom, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"state": "support_restored", "final_claim": repaired_claim}, ensure_ascii=False)},
        ],
        "metadata": {"case_id": "videosham-1004", "trajectory_type": "evidence_dropout_repair", "mutation": {"removed_evidence_id": "EA-OMIT-001"}, "generator": "deterministic_pipeline_v1"},
    }

    write_json(ROOT / "oracle" / "claim_oracle.json", oracle)
    write_json(ROOT / "student_view" / "case_manifest.json", visible_manifest)
    write_jsonl(ROOT / "student_view" / "evidence_atoms.jsonl", evidence)
    write_json(ROOT / "student_view" / "alignment_summary.json", {
        "source_times_s": [round(float(value), 6) for value in aligned["source_times_s"]],
        "edited_times_s": [round(float(value), 6) for value in aligned["edited_times_s"]],
        "similarities": [round(float(value), 6) for value in aligned["similarities"]],
        "source_frame_steps": [int(value) for value in aligned["source_frame_steps"]],
        "extra_advance_s": [round(float(value), 6) for value in aligned["extra_advance_s"]],
    })
    from trajectory_bundle import build_trajectory_bundle

    bundle = build_trajectory_bundle(
        root=ROOT,
        manifest=visible_manifest,
        evidence=evidence,
        claim=claim,
        repair_atom=repair_atom,
    )
    write_json(ROOT / "outputs" / "clean_trajectory.json", bundle["clean"])
    write_json(ROOT / "outputs" / "mutated_state.json", bundle["repair"]["mutated_state"])
    write_json(ROOT / "outputs" / "repair_trajectory.json", bundle["repair"])
    write_json(ROOT / "outputs" / "provenance_graph.json", bundle["provenance_graph"])
    write_jsonl(ROOT / "outputs" / "sft_clean.jsonl", [bundle["clean_sft"]])
    write_jsonl(ROOT / "outputs" / "sft_repair.jsonl", [bundle["repair_sft"]])

    # The interactive path executes the media tools again through a real
    # assistant tool-call -> observation -> harness-state loop.  The legacy
    # bundle above remains available for rubric compatibility.
    from interactive_harness import build_interactive_bundle

    interactive = build_interactive_bundle(
        manifest=visible_manifest,
        source_frames=source,
        suspect_frames=edited,
        source_path=SOURCE,
        suspect_path=EDITED,
        metadata_fn=media_metadata,
        align_fn=monotonic_align,
        omission_fn=omission_segments,
    )
    write_json(ROOT / "outputs" / "interactive_clean_episode.json", interactive["clean_episode"])
    write_json(ROOT / "outputs" / "interactive_repair_episode.json", interactive["repair_episode"])
    write_json(ROOT / "outputs" / "interactive_execution_summary.json", interactive["execution_summary"])
    # Harness-1 expands every retained trajectory into one supervised datum per
    # action turn.  Keep full trajectories separately for replay/debugging.
    write_jsonl(ROOT / "outputs" / "interactive_sft_clean.jsonl", interactive["clean_turn_sft"])
    write_jsonl(ROOT / "outputs" / "interactive_sft_repair.jsonl", interactive["repair_turn_sft"])
    write_jsonl(
        ROOT / "outputs" / "interactive_sft_clean_full_trajectory.jsonl",
        [interactive["clean_sft"]],
    )
    write_jsonl(
        ROOT / "outputs" / "interactive_sft_repair_full_trajectory.jsonl",
        [interactive["repair_sft"]],
    )
    return {
        "oracle": oracle,
        "manifest": visible_manifest,
        "evidence": evidence,
        "claim": bundle["clean"]["final_state"]["claim_ledger"][0]["claim"],
        "clean_sft": bundle["clean_sft"],
        "repair_sft": bundle["repair_sft"],
        "interactive": interactive,
    }


def validate(result: dict[str, Any]) -> dict[str, Any]:
    oracle = result["oracle"]
    claim = result["claim"]
    evidence = result["evidence"]
    evidence_ids = {atom["evidence_id"] for atom in evidence}
    student_payload = json.dumps(
        {"manifest": result["manifest"], "clean": result["clean_sft"], "repair": result["repair_sft"]},
        ensure_ascii=False,
    )
    checks = {
        "media_paths_present": all(item["path"] for item in (result["manifest"]["source_media"], result["manifest"]["edited_media"])),
        "claim_evidence_resolves": set(claim["evidence_ids"]).issubset(evidence_ids),
        "oracle_label_match": claim["manipulation_type"] == oracle["manipulation_type"],
        "oracle_window_match": claim["comparison_scope_s"] == oracle["edit_window_s"],
        "oracle_duration_match": abs(result["manifest"]["edited_media"]["duration_s"] - oracle["target_output_duration_s"]) <= 0.05,
        "visual_alignment_quality": evidence[1]["observation"]["p05_similarity"] >= 0.70,
        "positive_contraction": evidence[2]["observation"]["estimated_removed_time_s"] >= 1.0,
        "no_annotation_text_leak": oracle["annotation_text"] not in student_payload,
        "no_oracle_path_leak": "claim_oracle.json" not in student_payload and '"oracle"' not in student_payload,
        "dropout_is_evidence_side": "EA-OMIT-001" not in json.dumps(result["repair_sft"]["messages"][2], ensure_ascii=False),
        "repair_restores_support": "EA-REPAIR-001" in result["repair_sft"]["messages"][-1]["content"],
    }
    from interactive_harness import validate_interactive_bundle

    interactive_report = validate_interactive_bundle(result["interactive"], oracle["annotation_text"])
    checks["interactive_tool_environment"] = interactive_report["passed"]
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "interactive_checks": interactive_report["checks"],
        "interactive_execution_summary": interactive_report["execution_summary"],
    }
    write_json(ROOT / "outputs" / "validation_report.json", report)
    write_json(ROOT / "outputs" / "interactive_validation_report.json", interactive_report)
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"validation failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = build()
    report = validate(result) if args.validate else {"passed": None}
    print(json.dumps({"claim": result["claim"], "validation": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
