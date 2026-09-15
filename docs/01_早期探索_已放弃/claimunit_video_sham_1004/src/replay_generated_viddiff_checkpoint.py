"""Replay a generated VidDiff external branch from a public repair checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluate_student_context_arm import _parse_generated_call
from llm_teacher_rollout import _normalize_harness_params
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl, _payload, _restore_harness
from viddiff_batch import (
    DEFAULT_ADAPTIVE,
    VIDDIFF_TOOL_REGISTRY,
    VidDiffAdaptiveMeasurementEnvironment,
)


DEFAULT_REPORT = ROOT / "outputs/student_external_contract_v1/eval_v6_repair16_novel_validation24.json"
DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances"
    / "validation/sft_all.jsonl"
)
DEFAULT_ROW_ID = (
    "VidDiffBench:viddiff-surgery_155:LLM-TRJ-VIDDIFF-SURGERY_155-"
    "REPAIR-001:repair:2"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "viddiff_surgery155_generated_branch_checkpoint_replay.json"
)


def _candidate_item(sample_key: str) -> dict[str, Any]:
    items = _jsonl(DEFAULT_ADAPTIVE)
    return next(item for item in items if item["sample_key"] == sample_key)


def _seed_environment(
    environment: VidDiffAdaptiveMeasurementEnvironment,
    state: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    arguments = item["candidate_tools"][0]["arguments"]
    environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
    called_sequences = [
        atom.get("observation", {}).get("routing_state", {}).get("called_methods", [])
        for atom in (state.get("evidence_store") or {}).values()
        if atom.get("kind") == "action_comparison_measurement"
    ]
    clean_methods = list(max(called_sequences, key=len, default=[]))
    for evidence_id in state.get("dropped_evidence_ids") or []:
        upper = str(evidence_id).upper()
        for method in environment.available_methods:
            if method.upper() in upper and method not in clean_methods:
                clean_methods.append(method)
    for method in clean_methods:
        environment.execute(method, arguments)
    environment.on_evidence_dropout(
        str((state.get("dropped_evidence_ids") or ["unknown"])[0]),
        ["preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method"],
    )
    return {"clean_methods_replayed": clean_methods, "arguments": arguments}


def replay(report_path: Path, sft_path: Path, row_id: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated_row = next(item for item in report["rows"] if str(item["row_id"]) == row_id)
    row = next(item for item in _jsonl(sft_path) if str(item["metadata"]["row_id"]) == row_id)
    payload = _payload(row)
    state = payload["current_harness_state"]
    sample_key = str(
        state["candidate_claims"][0]["fields"]["sample_key"]["value"]
    )
    item = _candidate_item(sample_key)
    environment = VidDiffAdaptiveMeasurementEnvironment(item)
    seed = _seed_environment(environment, state, item)
    harness = _restore_harness(row)
    harness.environment = environment
    harness.tool_registry = copy.deepcopy(VIDDIFF_TOOL_REGISTRY)

    generated_name, generated_arguments = _parse_generated_call(generated_row["generated_text"])
    if generated_name is None or generated_arguments is None:
        raise ValueError("generated row has no parseable tool call")
    generated_result = harness.execute_tool(
        generated_name,
        _normalize_harness_params(generated_name, generated_arguments),
        "generated-checkpoint-branch",
    )
    state_after_generated = harness.render_state()

    reference_function = row["messages"][-1]["tool_calls"][0]["function"]
    reference_name = str(reference_function["name"])
    reference_arguments = reference_function["arguments"]
    if isinstance(reference_arguments, str):
        reference_arguments = json.loads(reference_arguments)
    followup_executed = reference_name in state_after_generated["eligible_tools"]
    followup_result = None
    state_after_followup = state_after_generated
    recovery_steps: list[dict[str, Any]] = []
    if followup_executed:
        followup_result = harness.execute_tool(
            reference_name,
            _normalize_harness_params(reference_name, reference_arguments),
            "compatible-followup-branch",
        )
        recovery_steps.append(
            {
                "tool_name": reference_name,
                "status": followup_result["status"],
                "claim_gap_compatibility": followup_result["observation"][
                    "claim_gap_compatibility"
                ],
            }
        )
        state_after_followup = harness.render_state()
    external_names = set(environment.available_methods)
    while True:
        current_state = harness.render_state()
        remaining = [
            name for name in current_state["eligible_tools"] if name in external_names
        ]
        if not remaining:
            break
        name = remaining[0]
        candidate = next(item for item in item["candidate_tools"] if item["name"] == name)
        result = harness.execute_tool(
            name,
            candidate["arguments"],
            f"recovery-{len(recovery_steps) + 1}",
        )
        recovery_steps.append(
            {
                "tool_name": name,
                "status": result["status"],
                "claim_gap_compatibility": result["observation"][
                    "claim_gap_compatibility"
                ],
            }
        )
    final_state = harness.render_state()
    compatibility = generated_result["observation"]["claim_gap_compatibility"]
    return {
        "audit_type": "recorded_label_blind_measurement_checkpoint_replay",
        "row_id": row_id,
        "sample_key": sample_key,
        "raw_video_reprocessed": False,
        "recorded_tool_observations_reexecuted_through_environment": True,
        "hidden_label_used_for_routing_or_execution": False,
        "seed": seed,
        "generated_branch": {
            "tool_name": generated_name,
            "arguments": generated_arguments,
            "status": generated_result["status"],
            "created_evidence_ids": generated_result["created_evidence_ids"],
            "claim_gap_compatibility": compatibility,
            "difference_goal_after": next(
                goal["status"]
                for goal in state_after_generated["case_goals"]
                if goal["goal_id"] == "G-DIFFERENCE"
            ),
            "eligible_tools_after": state_after_generated["eligible_tools"],
            "recoverable_not_dead_end": bool(state_after_generated["eligible_tools"]),
        },
        "compatible_followup": {
            "tool_name": reference_name,
            "executed": followup_executed,
            "status": followup_result["status"] if followup_result else None,
            "claim_gap_compatibility": (
                followup_result["observation"]["claim_gap_compatibility"]
                if followup_result
                else None
            ),
            "eligible_tools_after": state_after_followup["eligible_tools"],
            "binding_candidates_after": state_after_followup["state_action_affordances"][
                "binding_candidates"
            ],
        },
        "complete_recovery": {
            "steps_after_generated_branch": recovery_steps,
            "extra_measurement_count": len(recovery_steps),
            "difference_goal_after": next(
                goal["status"]
                for goal in final_state["case_goals"]
                if goal["goal_id"] == "G-DIFFERENCE"
            ),
            "eligible_tools_after": final_state["eligible_tools"],
            "reached_bindable_state": bool(
                final_state["state_action_affordances"]["binding_candidates"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--row-id", default=DEFAULT_ROW_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = replay(args.report, args.sft, args.row_id)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
