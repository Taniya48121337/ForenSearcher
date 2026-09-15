"""Locally screen selected VideoSham pairs before any teacher-service call.

The student artifact contains only logical media references and measurements that
the executable environment can reproduce. Dataset labels and editor instructions
are written only to a separate verifier audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, decode, write_json, write_jsonl
from videosham_batch import (
    EDITED_ROOT,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
    _case_number,
    _rows,
    _source_id,
    capability_family,
    localize_comparison_scope,
    preflight_case,
)


DEFAULT_ACQUISITION = ROOT / "outputs" / "scale_readiness" / "videosham_subset_acquisition.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "selection" / "videosham_measurement_probe_seed_20260820_n20"
MODALITY_BY_FAMILY = {
    "temporal_omission": "temporal",
    "temporal_warp": "temporal",
    "audio_replacement": "audio",
    "spatial_removal": "spatial",
    "spatial_insertion": "spatial",
    "appearance_change": "spatial",
    "text_edit": "spatial",
}
TOOL_BY_MODALITY = {
    "temporal": "detect_temporal_edit",
    "audio": "compare_audio_tracks",
    "spatial": "compare_spatial_regions",
}


def _selected_case_numbers(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["case_number"]) for item in payload["selection"]]


def _manifest(case_number: str, row: dict[str, str], scope: list[float]) -> dict[str, Any]:
    return {
        "case_id": f"videosham-{case_number}",
        "dataset": "VideoSham",
        "task": (
            "Determine which media modality contains a material source-to-suspect difference, "
            "then localize the compared source and suspect intervals."
        ),
        "source_id": _source_id(row),
        "comparison_scope_s": scope,
        "source_media": {"ref": "source"},
        "edited_media": {"ref": "suspect"},
        "allowed_inputs": ["the two locally resolved media refs", "neutral identifiers", "comparison scope"],
        "forbidden_inputs": ["editor task text", "manipulation label", "oracle fields", "filesystem paths"],
    }


def run(acquisition_path: Path, output_root: Path, *, all_local: bool = False) -> dict[str, Any]:
    rows_by_number = {_case_number(row): row for row in _rows()}
    selected = (
        [number for number, row in rows_by_number.items() if preflight_case(row)["cli_eligible"]]
        if all_local
        else _selected_case_numbers(acquisition_path)
    )
    student_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for case_number in selected:
        row = rows_by_number[case_number]
        preflight = preflight_case(row)
        student: dict[str, Any] = {
            "case_id": f"videosham-{case_number}",
            "status": "preflight_blocked",
            "preflight_reasons": list(preflight["preflight_reasons"]),
        }
        audit: dict[str, Any] = {
            "case_id": student["case_id"],
            "dataset_family": capability_family(row),
            "expected_modality": MODALITY_BY_FAMILY[capability_family(row)],
        }
        if preflight["cli_eligible"]:
            try:
                source_path = SOURCE_ROOT / f"{case_number}_unedited.mp4"
                suspect_path = EDITED_ROOT / f"{case_number}_processed.mp4"
                source_frames = decode(source_path)
                suspect_frames = decode(suspect_path)
                scope = localize_comparison_scope(
                    list(preflight["comparison_scope_s"]), source_frames.duration_s
                )
                manifest = _manifest(case_number, row, scope)
                environment = MultiFamilyVideoEnvironment(
                    manifest=manifest,
                    source_frames=source_frames,
                    suspect_frames=suspect_frames,
                    source_path=source_path,
                    suspect_path=suspect_path,
                )
                inspect = environment.execute("inspect_media", {"media_refs": ["source", "suspect"]})
                triage = inspect["result"]["modality_triage"]
                modality = triage["primary_modality"]
                tool_name = TOOL_BY_MODALITY.get(modality)
                measurement = None
                if tool_name:
                    measurement = environment.execute(
                        tool_name,
                        {
                            "source_ref": "source",
                            "suspect_ref": "suspect",
                            "comparison_scope_s": scope,
                        },
                    )
                    measurement = {
                        **measurement,
                        "evidence_atoms": [asdict(atom) for atom in measurement["evidence_atoms"]],
                    }
                student.update({
                    "status": "measurement_reachable" if measurement else "measurement_unreachable",
                    "comparison_scope_s": scope,
                    "triage": triage,
                    "selected_tool": tool_name,
                    "measurement": measurement,
                })
                audit.update({
                    "selected_modality": modality,
                    "selected_tool": tool_name,
                    "route_matches_dataset_family": modality == audit["expected_modality"],
                    "measurement_reachable": measurement is not None,
                })
            except Exception as error:
                student.update({
                    "status": "measurement_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                audit.update({
                    "route_matches_dataset_family": False,
                    "measurement_reachable": False,
                    "error_type": type(error).__name__,
                })
        student_rows.append(student)
        audit_rows.append(audit)

    statuses = Counter(item["status"] for item in student_rows)
    family_audit = {}
    for family in sorted({item["dataset_family"] for item in audit_rows}):
        members = [item for item in audit_rows if item["dataset_family"] == family]
        family_audit[family] = {
            "cases": len(members),
            "measurement_reachable": sum(bool(item.get("measurement_reachable")) for item in members),
            "route_matches_dataset_family": sum(bool(item.get("route_matches_dataset_family")) for item in members),
        }
    report = {
        "dataset": "VideoSham",
        "selected_cases": len(selected),
        "student_artifact": "student_measurements.jsonl",
        "verifier_artifact": "verifier_audit.jsonl",
        "privacy_contract": {
            "student_contains_editor_task": False,
            "student_contains_manipulation_label": False,
            "student_contains_filesystem_paths": False,
            "student_contains_oracle_fields": False,
        },
        "status_counts": dict(statuses),
        "family_audit": family_audit,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "student_measurements.jsonl", student_rows)
    write_jsonl(output_root / "verifier_audit.jsonl", audit_rows)
    write_json(output_root / "measurement_probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-path", type=Path, default=DEFAULT_ACQUISITION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--all-local", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.acquisition_path, args.output_root, all_local=args.all_local), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
