"""Create label-blind evidence-dropout repair seeds from accepted clean states.

The compiler copies a clean terminal pre-stop state and removes only the active
measurement EvidenceAtom that supports the claim's target fields.  Media, task
identity, claim values, and independently supported invariant fields are left
unchanged.  The output contains no assistant target and can be free-run by the
real student harness as a repair episode.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_teacher_rollout import _teacher_system_prompt
from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl, _payload


DEFAULT_SOURCE = (
    ROOT
    / "outputs/student_external_contract_v1/semantic_repair_increment_v1"
    / "decision_sufficient_state_v6_executable_affordances/train/sft_all.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/viddiff_counterfactual_repair_seeds_v1"
)
DEFAULT_PROTECTED = ["viddiff-surgery_26"]


def _assistant_tool(row: dict[str, Any]) -> str:
    assistant = next(
        (message for message in reversed(row.get("messages") or []) if message.get("role") == "assistant"),
        {},
    )
    calls = assistant.get("tool_calls") or []
    return str(calls[0]["function"]["name"]) if len(calls) == 1 else ""


def _claim_and_atom(
    state: dict[str, Any], claim_schema: dict[str, Any]
) -> tuple[dict[str, Any], str, list[str]]:
    claims = state.get("candidate_claims") or []
    if len(claims) != 1:
        raise ValueError("counterfactual repair requires exactly one candidate claim")
    claim = claims[0]
    target_fields = [str(field) for field in claim_schema.get("target_fields") or []]
    if not target_fields:
        raise ValueError("claim schema has no target fields")
    support_counts: Counter[str] = Counter()
    for field in target_fields:
        field_state = (claim.get("fields") or {}).get(field) or {}
        support_counts.update(str(item) for item in field_state.get("support_ids") or [])
    evidence_store = state.get("evidence_store") or {}
    candidates = [
        evidence_id
        for evidence_id, _count in support_counts.most_common()
        if evidence_id in evidence_store
        and (evidence_store[evidence_id] or {}).get("active") is not False
        and (evidence_store[evidence_id] or {}).get("kind")
        == "action_comparison_measurement"
    ]
    if not candidates:
        raise ValueError("no active action-comparison atom supports the target fields")
    evidence_id = candidates[0]
    affected = [
        field
        for field in target_fields
        if evidence_id
        in set(str(item) for item in ((claim.get("fields") or {}).get(field) or {}).get("support_ids") or [])
    ]
    if not affected:
        raise ValueError("selected measurement atom affects no target field")
    return claim, evidence_id, affected


def mutate_clean_stop_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if str(metadata.get("dataset")) != "VidDiffBench":
        raise ValueError("v1 counterfactual repair compiler supports VidDiffBench only")
    if _assistant_tool(row) != "stop":
        raise ValueError("source row must be the accepted clean stop turn")
    payload = copy.deepcopy(_payload(row))
    original_task = payload["task"]
    claim_schema = copy.deepcopy(original_task["claim_schema"])
    state = payload["current_harness_state"]
    claim, evidence_id, affected = _claim_and_atom(state, claim_schema)
    claim_id = str(claim["claim_id"])

    state.get("evidence_store", {}).pop(evidence_id, None)
    state["available_evidence_ids"] = [
        item for item in state.get("available_evidence_ids") or [] if str(item) != evidence_id
    ]
    state["dropped_evidence_ids"] = list(
        dict.fromkeys([*(state.get("dropped_evidence_ids") or []), evidence_id])
    )
    state["evidence_assessments"] = {}
    state["evidence_adjudications"] = {}
    for field in affected:
        field_state = claim["fields"][field]
        field_state["support_ids"] = [
            item for item in field_state.get("support_ids") or [] if str(item) != evidence_id
        ]
        field_state["evidence_status"] = (
            "supported" if field_state["support_ids"] else "unsupported"
        )
        # Evidence dropout does not silently edit the policy's decision state.
        field_state["decision_status"] = "committed"
        binding = ((state.get("field_bindings") or {}).get(claim_id) or {}).get(field)
        if binding is not None:
            binding["support_ids"] = [
                item for item in binding.get("support_ids") or [] if str(item) != evidence_id
            ]
            binding["validation_status"] = "invalidated_by_environment_mutation"
    for goal in state.get("case_goals") or []:
        if goal.get("goal_id") in {"G-DIFFERENCE", "G-REPORT"}:
            goal["status"] = "open"
    state["open_high_priority_obligations"] = ["G-DIFFERENCE", "G-REPORT"]
    state["eligible_tools"] = ["set_field_decision"]
    state["terminal"] = False
    state["stop_decision"] = None

    case_id = str(metadata["case_id"])
    trajectory_id = f"COUNTERFACTUAL-{case_id.upper().replace('-', '_')}-EVIDENCE-DROPOUT-001"
    payload["task"] = {
        "task": "Repair the restored state after an evidence-store mutation without changing independent fields.",
        "shared_clean_snapshot_id": f"SNAPSHOT-{case_id.upper().replace('-', '_')}-CLEAN-FINAL",
        "mutation": {
            "operator": "evidence_dropout",
            "removed_evidence_ids": [evidence_id],
            "media_changed": False,
            "claim_values_changed": False,
        },
        "claim_schema": claim_schema,
    }
    return {
        "messages": [
            {"role": "system", "content": _teacher_system_prompt("repair")},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "metadata": {
            "dataset": "VidDiffBench",
            "case_id": case_id,
            "trajectory_id": trajectory_id,
            "trajectory_type": "student_visible_counterfactual_evidence_dropout_repair",
            "turn_index": 1,
            "row_id": f"VidDiffBench:{case_id}:{trajectory_id}:repair:1",
            "data_split": str(metadata.get("source_split") or metadata.get("data_split") or "train"),
            "source_kind": "repair",
            "entity_group": f"VidDiffBench:{case_id}",
            "routing_variant": str(metadata.get("routing_variant") or "balanced"),
            "resolved_routing_variant": str(
                metadata.get("resolved_routing_variant") or metadata.get("routing_variant") or "balanced"
            ),
            "seed_contains_teacher_action": False,
            "seed_contains_oracle_or_verifier": False,
            "seed_source": "accepted_clean_terminal_state_plus_local_evidence_dropout",
            "source_clean_row_id": str(metadata.get("row_id") or ""),
            "mutation_operator": "evidence_dropout",
            "removed_evidence_id": evidence_id,
            "affected_fields": affected,
        },
        "tools": [],
    }


def build(
    source: Path, case_ids: list[str], protected_cases: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = set(case_ids)
    protected = set(protected_cases)
    if requested & protected:
        raise ValueError(f"protected cases requested: {sorted(requested & protected)}")
    stop_by_case: dict[str, dict[str, Any]] = {}
    for row in _jsonl(source):
        metadata = row.get("metadata") or {}
        case_id = str(metadata.get("case_id") or "")
        if case_id in requested and _assistant_tool(row) == "stop":
            stop_by_case[case_id] = row
    missing = sorted(requested - set(stop_by_case))
    if missing:
        raise ValueError(f"accepted clean stop rows not found: {missing}")
    seeds = [mutate_clean_stop_row(stop_by_case[case_id]) for case_id in sorted(requested)]
    manifest = {
        "artifact": "viddiff_counterfactual_repair_seeds_v1",
        "source": str(source),
        "cases": sorted(requested),
        "protected_cases": sorted(protected),
        "rows": len(seeds),
        "mutation_operator": "evidence_dropout",
        "teacher_actions_used": False,
        "oracle_or_verifier_used": False,
        "media_changed": False,
        "claim_values_changed": False,
        "intent": (
            "Create case-disjoint repair-policy supervision by mutating only accepted clean "
            "harness state; no video generation, teacher target, or hidden label is used."
        ),
    }
    return seeds, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--protected-case", action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protected = list(args.protected_case or DEFAULT_PROTECTED)
    seeds, manifest = build(args.source, list(args.case_id), protected)
    write_jsonl(args.output / "seeds.jsonl", seeds)
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
