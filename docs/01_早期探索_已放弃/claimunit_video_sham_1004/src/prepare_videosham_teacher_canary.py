"""Prepare a small, privacy-audited VideoSham teacher canary without external calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl


DEFAULT_MEASUREMENTS = (
    ROOT
    / "outputs"
    / "selection"
    / "videosham_measurement_probe_v7_streaming_rgb_all_local_n30"
    / "student_measurements.jsonl"
)
DEFAULT_VERIFIER = DEFAULT_MEASUREMENTS.with_name("verifier_audit.jsonl")
DEFAULT_ACQUISITION = ROOT / "outputs" / "scale_readiness" / "videosham_subset_acquisition.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "selection" / "videosham_teacher_canary_v1_n7"
REPAIR_OPERATORS = ("evidence_dropout", "binding_invalidation", "decision_reopen")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _signal_margin(row: dict[str, Any]) -> float:
    triage = row["triage"]
    tool = row["selected_tool"]
    metrics = row["measurement"]["result"]["measurements"]
    if tool == "detect_temporal_edit":
        duration = abs(float(triage["duration_ratio"]) - 1.0) - 0.08
        negative = float(metrics.get("negative_step_fraction", 0.0)) - 0.02
        reused_and_variable = min(
            float(metrics.get("reused_or_reversed_step_fraction", 0.0)) - 0.08,
            float(metrics.get("positive_step_cv", 0.0)) - 0.8,
        )
        return round(max(duration, negative, reused_and_variable), 6)
    if tool == "compare_audio_tracks":
        return round(0.72 - float(triage["audio_similarity"]), 6)
    if tool == "compare_spatial_regions":
        return round(
            max(
                0.9995 - float(triage["visual_p05_similarity"]),
                float(triage["rgb_tile_p95_mae"]) - 0.025,
                float(triage["rgb_tile_max_mae"]) - 0.025,
                float(triage["rgb_changed_tile_fraction"]) - 0.04,
            ),
            6,
        )
    raise ValueError(f"unsupported measurement tool: {tool}")


def _teacher_preview(row: dict[str, Any], repair_operator: str) -> dict[str, Any]:
    case_id = str(row["case_id"])
    return {
        "case_id": case_id,
        "teacher_transport": "text_json_only",
        "media_bytes_sent": False,
        "frames_sent": False,
        "audio_sent": False,
        "initial_task": {
            "case_id": case_id,
            "dataset": "VideoSham",
            "task": (
                "Determine which media modality contains a material source-to-suspect difference, "
                "then localize the compared source and suspect intervals."
            ),
            "source_media": {"ref": "source"},
            "edited_media": {"ref": "suspect"},
            "comparison_scope_s": row["comparison_scope_s"],
            "allowed_inputs": ["logical media refs", "neutral comparison scope", "executed tool observations"],
            "forbidden_inputs": ["dataset answer fields", "editor instruction", "filesystem or network locator"],
        },
        "staged_student_visible_observations": [
            {
                "tool_name": "inspect_media",
                "result": {"modality_triage": row["triage"]},
            },
            row["measurement"],
        ],
        "repair_operator": repair_operator,
    }


def prepare(measurements: Path, verifier: Path, acquisition: Path, output_root: Path) -> dict[str, Any]:
    student_rows = {row["case_id"]: row for row in _jsonl(measurements)}
    verifier_rows = {row["case_id"]: row for row in _jsonl(verifier)}
    acquisition_payload = json.loads(acquisition.read_text(encoding="utf-8"))
    acquired_ids = {f"videosham-{item['case_number']}" for item in acquisition_payload["selection"]}

    by_family: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for case_id in sorted(acquired_ids):
        student = student_rows[case_id]
        audit = verifier_rows[case_id]
        if student["status"] != "measurement_reachable" or not audit["route_matches_dataset_family"]:
            continue
        by_family[str(audit["dataset_family"])].append((_signal_margin(student), student))

    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for family in sorted(by_family):
        margin, student = min(by_family[family], key=lambda item: (item[0], item[1]["case_id"]))
        selected.append(student)
        audit_rows.append({
            "case_id": student["case_id"],
            "dataset_family": family,
            "selection_policy": "lowest_positive_label_blind_signal_margin_within_family",
            "signal_margin": margin,
            "selected_tool": student["selected_tool"],
        })

    previews = [
        _teacher_preview(row, REPAIR_OPERATORS[index % len(REPAIR_OPERATORS)])
        for index, row in enumerate(selected)
    ]
    preview_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in previews)
    forbidden_patterns = {
        "windows_absolute_path": r"(?i)[a-z]:\\\\",
        "unix_home_path": r"/(?:home|users)/[^\"\s]+",
        "media_filename": r"(?i)\.(?:mp4|mov|avi|mkv|webm)(?=[\"\\/,\s]|$)",
        "raw_feature_filename": r"(?i)\.npy(?=[\"\\/,\s]|$)",
        "url": r"(?i)(?:https?|file)://",
        "locator_field": r'(?i)"(?:local_)?(?:path|root|uri|url)"\s*:',
        "dataset_family_field": r'"dataset_family"\s*:',
        "annotation_text_field": r'"annotation_text"\s*:',
        "oracle_field": r'"oracle[^\"]*"\s*:',
        "verifier_field": r'"verifier[^\"]*"\s*:',
    }
    scan = {name: len(re.findall(pattern, preview_text)) for name, pattern in forbidden_patterns.items()}
    report = {
        "selection_policy": "one_hard_reachable_case_per_new_videosham_family",
        "selected_cases": len(selected),
        "families_covered": len(by_family),
        "repair_distribution": {
            operator: sum(preview["repair_operator"] == operator for preview in previews)
            for operator in REPAIR_OPERATORS
        },
        "outbound_contract": {
            "text_json_only": True,
            "media_bytes_sent": False,
            "frames_sent": False,
            "audio_sent": False,
            "neutral_task_instruction_sent": True,
            "dataset_name_sent": True,
            "editor_task_text_sent": False,
            "manipulation_label_sent": False,
            "filesystem_path_sent": False,
            "url_sent": False,
            "oracle_or_verifier_values_sent": False,
        },
        "forbidden_pattern_hits": scan,
        "privacy_scan_passed": not any(scan.values()),
        "teacher_calls_made": 0,
        "selected_case_ids": [row["case_id"] for row in selected],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "teacher_payload_preview.student.jsonl", previews)
    write_jsonl(output_root / "selection_audit.local_verifier.jsonl", audit_rows)
    write_json(output_root / "preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.measurements, args.verifier, args.acquisition, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
