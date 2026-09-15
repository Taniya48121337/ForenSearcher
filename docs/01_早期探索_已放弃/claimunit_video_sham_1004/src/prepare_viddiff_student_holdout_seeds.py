"""Build teacher-free VidDiff seeds from verified, student-visible candidate sets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl, _payload
from viddiff_batch import DEFAULT_ADAPTIVE, VIDDIFF_SCHEMA


DEFAULT_V6 = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances"
)
DEFAULT_CANDIDATES = DEFAULT_ADAPTIVE
DEFAULT_SELECTION_AUDIT = DEFAULT_CANDIDATES.parent / "selection_audit.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_external_holdout_v1"
)
_FORBIDDEN_STUDENT_TOKENS = (
    '"candidate_report"',
    '"differences_gt"',
    '"ground_truth"',
    '"local_path"',
    '"oracle_direction"',
    '"retrieval_frames"',
    '"url"',
    '"urls"',
    '"verifier"',
    '"video_path"',
)


def _v6_rows(root: Path) -> list[dict[str, Any]]:
    return [
        row
        for split in ("train", "validation", "test")
        for row in _jsonl(root / split / "sft_all.jsonl")
    ]


def _duration(item: dict[str, Any]) -> float:
    durations: list[float] = []
    assets = item.get("task", {}).get("visible_assets")
    for asset in assets if isinstance(assets, list) else []:
        if not isinstance(asset, dict):
            continue
        direct = asset.get("local_duration_s")
        if isinstance(direct, (int, float)):
            durations.append(float(direct))
        global_asset = asset.get("global")
        if isinstance(global_asset, dict):
            nested = global_asset.get("local_duration_s")
            if isinstance(nested, (int, float)):
                durations.append(float(nested))
    return max(durations, default=1.0)


def build(
    v6_root: Path,
    candidates_path: Path,
    selection_audit_path: Path,
    routing_variant: str = "balanced",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v6_rows = _v6_rows(v6_root)
    used = {
        str(row["metadata"]["case_id"])
        for row in v6_rows
        if row["metadata"]["dataset"] == "VidDiffBench"
    }
    template = next(
        copy.deepcopy(row)
        for row in v6_rows
        if row["metadata"]["dataset"] == "VidDiffBench"
        and str(row["metadata"]["row_id"]).endswith(":clean:1")
    )
    candidates = {
        str(item["sample_key"]): item for item in _jsonl(candidates_path)
    }
    audit = json.loads(selection_audit_path.read_text(encoding="utf-8"))
    accepted = {
        str(item["sample_key"])
        for item in audit.get("cases", [])
        if item.get("accepted") is True
    }

    rows: list[dict[str, Any]] = []
    excluded_used = 0
    for sample_key in sorted(accepted):
        item = candidates.get(sample_key)
        if item is None:
            raise ValueError(f"accepted VidDiff case lacks a candidate set: {sample_key}")
        case_id = f"viddiff-{sample_key}"
        if case_id in used:
            excluded_used += 1
            continue
        serialized_item = json.dumps(item, ensure_ascii=False)
        leaks = [token for token in _FORBIDDEN_STUDENT_TOKENS if token in serialized_item]
        if leaks:
            raise ValueError(f"student-visible candidate contains forbidden fields: {sample_key}:{leaks}")

        row = copy.deepcopy(template)
        payload = _payload(row)
        contract = payload["task"]["task_contract"]
        contract.update(
            {
                "contract_id": f"TASK-VIDDIFF-{sample_key.upper()}",
                "episode_mode": "paired_compare",
                "task": "Produce one evidence-bound comparative claim for the visible video pair.",
                "visible_assets": {
                    "source": "video_a",
                    "suspect": "video_b",
                    "comparison_scope_s": [0.0, _duration(item)],
                },
                "claim_schema_id": VIDDIFF_SCHEMA["schema_id"],
            }
        )
        payload["task"]["claim_schema"] = copy.deepcopy(VIDDIFF_SCHEMA)
        state = payload["current_harness_state"]
        state["task_contract"] = copy.deepcopy(contract)
        state["candidate_claims"] = []
        state["field_bindings"] = {}
        state["evidence_store"] = {}
        state["available_evidence_ids"] = []
        state["dropped_evidence_ids"] = []
        state["open_high_priority_obligations"] = [
            "G-CONTEXT",
            "G-DIFFERENCE",
            "G-REPORT",
        ]
        state["eligible_tools"] = ["inspect_comparison_task"]
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
        trajectory_id = f"STUDENT-HOLDOUT-{sample_key.upper()}-CLEAN-001"
        row["metadata"] = {
            "dataset": "VidDiffBench",
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "trajectory_type": "teacher_free_external_holdout_clean",
            "turn_index": 1,
            "row_id": f"VidDiffBench:{case_id}:{trajectory_id}:clean:1",
            "data_split": "external_holdout",
            "source_kind": "clean",
            "entity_group": f"VidDiffBench:{case_id}",
            "routing_variant": routing_variant,
            "resolved_routing_variant": routing_variant,
            "seed_contains_teacher_action": False,
            "seed_contains_oracle_or_verifier": False,
            "seed_source": "neutral_task_identity_after_verifier_only_candidate_admission",
        }
        row["tools"] = copy.deepcopy(template["tools"])
        rows.append(row)

    manifest = {
        "artifact": "viddiff_teacher_free_external_holdout_seeds_v1",
        "cases": len(rows),
        "case_ids": [row["metadata"]["case_id"] for row in rows],
        "accepted_candidates": len(accepted),
        "excluded_v6_cases": excluded_used,
        "teacher_actions_used": False,
        "oracle_or_verifier_in_student_seed": False,
        "raw_assets_embedded": False,
        "urls_embedded": False,
        "local_paths_embedded": False,
        "seed_format": "system + neutral student-visible pre-action state; no assistant target",
        "routing_variant": routing_variant,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6-root", type=Path, default=DEFAULT_V6)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--selection-audit", type=Path, default=DEFAULT_SELECTION_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--routing-variant",
        choices=["balanced", "uniform_first", "global_first", "action_first", "predicate_first"],
        default="balanced",
    )
    args = parser.parse_args()
    rows, manifest = build(
        args.v6_root,
        args.candidates,
        args.selection_audit,
        routing_variant=args.routing_variant,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "seeds.jsonl", rows)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
