"""Build teacher-free VideoSham clean seeds for cases absent from every V6 split."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl, _payload
from videosham_batch import EDITED_ROOT, METADATA_PATH, SOURCE_ROOT


DEFAULT_V6 = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_MEASUREMENTS = (
    ROOT
    / "outputs/selection/videosham_measurement_probe_v7_streaming_rgb_all_local_n30"
    / "student_measurements.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/videosham_external_holdout_v1"
)


def _source_id(value: str, dataset: str) -> str:
    token = value.rstrip("/").rsplit("/", 1)[-1]
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", token).strip("-").lower()
    prefix = re.sub(r"[^a-zA-Z0-9]+", "-", dataset).strip("-").lower()
    return f"{prefix}-{token}"


def _metadata_identities(path: Path) -> dict[str, str]:
    identities: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            number = Path(str(row["Video-Name"])).stem
            identities[number] = _source_id(
                str(row["Name/Id in source dataset"]),
                str(row["Source-Dataset"]),
            )
    return identities


def _v6_rows(root: Path) -> list[dict[str, Any]]:
    return [
        row
        for split in ("train", "validation", "test")
        for row in _jsonl(root / split / "sft_all.jsonl")
    ]


def build(v6_root: Path, measurements_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v6_rows = _v6_rows(v6_root)
    used = {
        str(row["metadata"]["case_id"])
        for row in v6_rows
        if row["metadata"]["dataset"] == "VideoSham"
    }
    template = next(
        copy.deepcopy(row)
        for row in v6_rows
        if row["metadata"]["dataset"] == "VideoSham"
        and str(row["metadata"]["row_id"]).endswith(":clean:1")
    )
    identities = _metadata_identities(METADATA_PATH)
    measurements = [
        item
        for item in _jsonl(measurements_path)
        if item.get("status") == "measurement_reachable"
        and str(item["case_id"]) not in used
    ]
    rows: list[dict[str, Any]] = []
    modality_counts: Counter[str] = Counter()
    for item in sorted(measurements, key=lambda value: str(value["case_id"])):
        case_id = str(item["case_id"])
        number = case_id.removeprefix("videosham-")
        source_path = SOURCE_ROOT / f"{number}_unedited.mp4"
        suspect_path = EDITED_ROOT / f"{number}_processed.mp4"
        if (
            not source_path.exists()
            or source_path.stat().st_size < 1024
            or not suspect_path.exists()
            or suspect_path.stat().st_size < 1024
        ):
            continue
        scope = [float(value) for value in item["comparison_scope_s"]]
        row = copy.deepcopy(template)
        payload = _payload(row)
        contract = payload["task"]["task_contract"]
        contract["contract_id"] = f"TASK-VIDEOSHAM-{number}"
        contract["visible_assets"] = {
            "source": "source",
            "suspect": "suspect",
            "comparison_scope_s": scope,
        }
        payload["current_harness_state"]["task_contract"] = copy.deepcopy(contract)
        row["messages"] = [
            copy.deepcopy(row["messages"][0]),
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        trajectory_id = f"STUDENT-HOLDOUT-{number}-CLEAN-001"
        row["metadata"] = {
            "dataset": "VideoSham",
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "trajectory_type": "teacher_free_external_holdout_clean",
            "turn_index": 1,
            "row_id": f"VideoSham:{case_id}:{trajectory_id}:clean:1",
            "data_split": "external_holdout",
            "source_kind": "clean",
            "entity_group": f"VideoSham:{case_id}",
            "seed_contains_teacher_action": False,
            "seed_contains_oracle_or_verifier": False,
            "seed_source": "student_visible_measurement_pool_and_neutral_media_metadata",
        }
        row["_batch_public_video_identity"] = [identities[number], scope]
        rows.append(row)
        modality_counts[str(item.get("selected_tool"))] += 1
    manifest = {
        "artifact": "videosham_teacher_free_external_holdout_seeds_v1",
        "cases": len(rows),
        "case_ids": [row["metadata"]["case_id"] for row in rows],
        "excluded_v6_cases": len(used),
        "selected_tool_counts": dict(modality_counts),
        "teacher_actions_used": False,
        "oracle_or_verifier_used": False,
        "original_edit_tasks_exported": False,
        "raw_urls_exported": False,
        "seed_format": "system + neutral student-visible pre-action user state; no assistant target",
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-root", type=Path, default=DEFAULT_V6)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, manifest = build(args.v6_root, args.measurements)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "seeds.jsonl", rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
