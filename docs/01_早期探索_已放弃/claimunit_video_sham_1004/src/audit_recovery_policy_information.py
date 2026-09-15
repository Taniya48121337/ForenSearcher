"""Separate semantic policy supervision from control-plane determinism.

This audit deliberately keeps two questions distinct:

1. Does the visible state leave a choice between tools?
2. After a tool is chosen, does its schema uniquely determine the arguments?

The second condition does not imply the first.  In particular, ``stop({})`` has
unique arguments but choosing stop over challenge_evidence remains an
evidence-sensitive policy decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_control_plane_exposure import (
    _compiled_actions,
    _preaction_payload,
    _target_call,
    _tool_schema,
    _unique_schema_value,
)
from pipeline import ROOT, write_json, write_jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike"
DEFAULT_MIX = BASE / "recovery_sft_training_mix_v4_edgeaware55/recovery_sft_weighted.jsonl"


CORE_STAGES = {
    "challenge",
    "fresh_measurement",
    "claim_revision",
    "evidence_assessment",
    "evidence_adjudication",
}
SUPPORTING_STAGES = {"local_reopen", "fresh_binding", "recommit"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tier(stage: str) -> str:
    if stage in CORE_STAGES:
        return "semantic_policy_core"
    if stage in SUPPORTING_STAGES:
        return "evidence_grounded_transition"
    if stage == "terminal_verification":
        return "terminal_evidence_decision"
    return "unclassified"


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_flatten_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    return []


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = len(records)
    flags = (
        "target_tool_is_eligible",
        "tool_choice_available",
        "arguments_unique_given_tool",
        "whole_call_unique_from_control_plane",
        "exact_compiled_target_visible",
        "choice_bearing_supervision",
    )
    counts = {key: sum(bool(row[key]) for row in records) for key in flags}
    return {
        "turns": turns,
        "cases": len({row["case_id"] for row in records}),
        "counts": counts,
        "rates": {
            key: round(value / max(turns, 1), 6) for key, value in counts.items()
        },
        "eligible_tool_count_histogram": dict(
            sorted(Counter(str(row["eligible_tool_count"]) for row in records).items())
        ),
        "target_tools": dict(sorted(Counter(row["target_tool"] for row in records).items())),
    }


def audit(
    rows: list[dict[str, Any]],
    weighted_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weights = {
        str(row["metadata"]["row_id"]): float(
            row["metadata"].get("sft_sampling_weight", 0.0)
        )
        for row in (weighted_rows or [])
    }
    records: list[dict[str, Any]] = []
    for row in rows:
        metadata = row["metadata"]
        stage = str(metadata["causal_recovery_stage"])
        payload, _ = _preaction_payload(row)
        state = payload.get("current_harness_state") or payload.get("harness_state") or {}
        eligible = [str(tool) for tool in state.get("eligible_tools") or []]
        target_tool, target_arguments = _target_call(row)
        schema = _tool_schema(row, target_tool)
        singleton_schema, singleton_arguments = _unique_schema_value(schema)
        arguments_unique = bool(
            singleton_schema and singleton_arguments == target_arguments
        )
        compiled_actions = _compiled_actions(payload)
        exact_compiled = (target_tool, target_arguments) in compiled_actions
        target_is_eligible = target_tool in eligible
        whole_unique = bool(
            len(eligible) == 1
            and eligible[0] == target_tool
            and arguments_unique
        )
        # A turn still carries policy information when either the tool itself or
        # at least one required argument must be selected from semantic state.
        choice_bearing = bool(len(eligible) > 1 or not arguments_unique)
        strings = _flatten_strings(target_arguments)
        evidence_refs = sorted({value for value in strings if value.startswith("EA-")})
        target_fields = target_arguments.get("target_fields")
        if not isinstance(target_fields, list):
            target_fields = target_arguments.get("fields")
        if not isinstance(target_fields, list):
            target_fields = []
        row_id = str(metadata["row_id"])
        records.append(
            {
                "row_id": row_id,
                "case_id": str(metadata["case_id"]),
                "dataset": str(metadata["dataset"]),
                "stage": stage,
                "policy_information_tier": _tier(stage),
                "target_tool": target_tool,
                "eligible_tools": eligible,
                "eligible_tool_count": len(eligible),
                "target_tool_is_eligible": target_is_eligible,
                "tool_choice_available": len(eligible) > 1,
                "arguments_unique_given_tool": arguments_unique,
                "whole_call_unique_from_control_plane": whole_unique,
                "exact_compiled_target_visible": exact_compiled,
                "choice_bearing_supervision": choice_bearing,
                "target_field_count": len(target_fields),
                "target_evidence_reference_count": len(evidence_refs),
                "learning_value_weight": float(metadata["learning_value_weight"]),
                "sft_sampling_weight": weights.get(row_id),
            }
        )

    stages = sorted({row["stage"] for row in records})
    tiers = sorted({row["policy_information_tier"] for row in records})
    datasets = sorted({row["dataset"] for row in records})
    by_stage = {
        stage: _summarize([row for row in records if row["stage"] == stage])
        for stage in stages
    }
    by_tier = {
        tier: _summarize(
            [row for row in records if row["policy_information_tier"] == tier]
        )
        for tier in tiers
    }
    by_dataset = {
        dataset: _summarize([row for row in records if row["dataset"] == dataset])
        for dataset in datasets
    }

    weighted_mass: dict[str, float] = defaultdict(float)
    total_weight = 0.0
    for row in records:
        weight = row["sft_sampling_weight"]
        if weight is None:
            continue
        weighted_mass[row["policy_information_tier"]] += weight
        total_weight += weight

    report = {
        "artifact": "recovery_policy_information_audit_v1",
        "source_rows": len(rows),
        "definitions": {
            "arguments_unique_given_tool": (
                "Conditional on already choosing the target tool, its visible JSON schema "
                "has exactly one materialization and it equals the executed target arguments."
            ),
            "whole_call_unique_from_control_plane": (
                "Exactly one tool is eligible and that tool's complete argument object is "
                "uniquely materialized by its schema."
            ),
            "choice_bearing_supervision": (
                "Either multiple tools are eligible or the chosen tool still requires a "
                "non-singleton argument decision."
            ),
            "semantic_policy_core": (
                "Evidence diagnosis, acquisition, claim revision, assessment, or adjudication."
            ),
            "evidence_grounded_transition": (
                "Local reopening, evidence attribution, or recommitment whose scope must remain "
                "consistent with visible evidence."
            ),
            "terminal_evidence_decision": (
                "Stop versus continued investigation after checking all active evidence."
            ),
        },
        "overall": _summarize(records),
        "by_stage": by_stage,
        "by_policy_information_tier": by_tier,
        "by_dataset": by_dataset,
        "weighted_training_mass": {
            "available": bool(weighted_rows),
            "total": round(total_weight, 6),
            "by_tier": {
                key: round(value, 6) for key, value in sorted(weighted_mass.items())
            },
            "share_by_tier": {
                key: round(value / total_weight, 6)
                for key, value in sorted(weighted_mass.items())
            }
            if total_weight
            else {},
        },
        "interpretation": {
            "control_plane_can_copy_whole_call": sum(
                row["whole_call_unique_from_control_plane"] for row in records
            ),
            "compiled_target_is_visible": sum(
                row["exact_compiled_target_visible"] for row in records
            ),
            "choice_bearing_turns": sum(
                row["choice_bearing_supervision"] for row in records
            ),
            "do_not_equate_unique_stop_arguments_with_preexposed_stop_decision": True,
        },
        "uses_oracle_or_verifier": False,
        "changes_messages_or_targets": False,
        "training_qualification_note": (
            "This is a diagnostic decomposition, not evidence of student free-run gain and "
            "not a replacement for blind review or sealed evaluation."
        ),
    }
    return report, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BATCH / "recovery_sft_candidate.jsonl",
    )
    parser.add_argument("--weighted-input", type=Path, default=DEFAULT_MIX)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BATCH / "policy_information_audit.json",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=DEFAULT_BATCH / "policy_information_rows.jsonl",
    )
    args = parser.parse_args()
    weighted_rows = _jsonl(args.weighted_input) if args.weighted_input.exists() else None
    report, records = audit(_jsonl(args.input), weighted_rows)
    write_json(args.output, report)
    write_jsonl(args.rows_output, records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
