"""Build teacher-free VCSL seeds for cases newly reachable by official SPD."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl, _payload
from vcsl_batch import VCSL_SCHEMA
from vcsl_probe import DEFAULT_FEATURE_ROOT


DEFAULT_V6 = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_SPD_REPORT = (
    ROOT / "outputs/student_external_contract_v1/vcsl_official_spd_experiment_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/vcsl_spd_external_holdout_v1"
)


def _v6_rows(root: Path) -> list[dict[str, Any]]:
    return [
        row
        for split in ("train", "validation", "test")
        for row in _jsonl(root / split / "sft_all.jsonl")
    ]


def build(v6_root: Path, spd_report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v6_rows = _v6_rows(v6_root)
    used = {
        str(row["metadata"]["case_id"])
        for row in v6_rows
        if row["metadata"]["dataset"] == "VCSL"
    }
    template = next(
        copy.deepcopy(row)
        for row in v6_rows
        if row["metadata"]["dataset"] == "VCSL"
        and str(row["metadata"]["row_id"]).endswith(":clean:1")
    )
    report = json.loads(spd_report_path.read_text(encoding="utf-8"))
    candidates = [item for item in report["cases"] if item.get("newly_recovered")]

    rows: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: str(value["pair_id"])):
        pair_id = str(item["pair_id"])
        case_id = f"vcsl-{pair_id}"
        query_id, reference_id = pair_id.split("-", 1)
        if case_id in used:
            continue
        frame_counts = []
        for media_id in (query_id, reference_id):
            feature_path = DEFAULT_FEATURE_ROOT / f"{media_id}.npy"
            if not feature_path.is_file():
                raise FileNotFoundError(f"released VCSL feature is unavailable: {feature_path}")
            frame_counts.append(int(np.load(feature_path, mmap_mode="r").shape[0]))

        row = copy.deepcopy(template)
        payload = _payload(row)
        contract = payload["task"]["task_contract"]
        contract["contract_id"] = f"TASK-VCSL-{query_id.upper()}-{reference_id.upper()}"
        contract["visible_assets"] = {
            "source": "query",
            "suspect": "reference",
            "comparison_scope_s": [0.0, float(max(frame_counts))],
        }
        payload["task"]["claim_schema"] = copy.deepcopy(VCSL_SCHEMA)
        state = payload["current_harness_state"]
        state["task_contract"] = copy.deepcopy(contract)
        state["candidate_claims"] = []
        state["field_bindings"] = {}
        state["evidence_store"] = {}
        state["available_evidence_ids"] = []
        state["dropped_evidence_ids"] = []
        state["open_high_priority_obligations"] = ["G-CONTEXT", "G-DIFFERENCE", "G-REPORT"]
        state["eligible_tools"] = ["inspect_pair_features"]
        state["terminal"] = False
        state["stop_decision"] = None
        state["state_action_affordances"] = {
            "derivation": "Compiled from student-visible state.",
            "binding_candidates": {},
            "executable_binding_actions": [],
            "repair_restoration_candidates": [],
            "meaningful_decision_transitions": [],
            "executable_decision_actions": [],
        }
        row["messages"] = [
            copy.deepcopy(row["messages"][0]),
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        trajectory_id = f"STUDENT-SPD-HOLDOUT-{pair_id.upper()}-CLEAN-001"
        row["metadata"] = {
            "dataset": "VCSL",
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "trajectory_type": "teacher_free_spd_external_holdout_clean",
            "turn_index": 1,
            "row_id": f"VCSL:{case_id}:{trajectory_id}:clean:1",
            "data_split": "external_holdout",
            "source_kind": "clean",
            "entity_group": f"VCSL:{case_id}",
            "routing_variant": "spd_official",
            "resolved_routing_variant": "spd_official",
            "seed_contains_teacher_action": False,
            "seed_contains_oracle_or_verifier": False,
            "seed_source": "neutral_visible_pair_identity_after_verifier_only_candidate_admission",
        }
        row["tools"] = copy.deepcopy(template["tools"])
        rows.append(row)

    manifest = {
        "artifact": "vcsl_teacher_free_spd_external_holdout_seeds_v1",
        "cases": len(rows),
        "case_ids": [row["metadata"]["case_id"] for row in rows],
        "excluded_v6_cases": len(used),
        "routing_variant": "spd_official",
        "teacher_actions_used": False,
        "oracle_or_verifier_in_student_seed": False,
        "verifier_used_only_for_candidate_admission": True,
        "raw_features_embedded": False,
        "local_paths_embedded": False,
        "seed_format": "system + neutral student-visible pre-action state; no assistant target",
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-root", type=Path, default=DEFAULT_V6)
    parser.add_argument("--spd-report", type=Path, default=DEFAULT_SPD_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, manifest = build(args.v6_root, args.spd_report)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "seeds.jsonl", rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
