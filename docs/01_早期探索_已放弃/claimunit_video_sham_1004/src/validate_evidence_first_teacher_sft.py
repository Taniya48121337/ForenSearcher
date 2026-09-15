"""Validate evidence-first teacher SFT without pretending it is student rollout data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from audit_control_plane_exposure import audit as audit_control_plane
from compile_evidence_first_sft import _reasoning_audit, _sensitive_key_hits
from pipeline import write_json
from shadow_execute_generated_actions import _jsonl


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    row_ids: list[str] = []
    tools_used: Counter[str] = Counter()
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        row_id = str(metadata.get("row_id") or f"missing-row-id-{index}")
        row_ids.append(row_id)
        row_failures: list[str] = []
        messages = row.get("messages") or []
        if len(messages) != 3 or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            row_failures.append("messages_must_be_system_user_assistant")
            payload: dict[str, Any] = {}
            assistant: dict[str, Any] = {}
        else:
            try:
                payload = json.loads(messages[1]["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                payload = {}
                row_failures.append("user_payload_not_valid_json")
            assistant = messages[2]

        if not str(assistant.get("content") or "").strip():
            row_failures.append("missing_natural_language_reasoning")
        calls = assistant.get("tool_calls") or []
        function: dict[str, Any] = {}
        if len(calls) != 1:
            row_failures.append("assistant_must_have_exactly_one_tool_call")
        else:
            function = calls[0].get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if name:
            tools_used[name] += 1
        if name != str(metadata.get("target_tool") or ""):
            row_failures.append("target_tool_metadata_mismatch")

        state = payload.get("current_harness_state") or payload.get("harness_state") or {}
        if name not in (state.get("eligible_tools") or []):
            row_failures.append("target_tool_not_visible_as_eligible")
        turn_index = int(metadata.get("turn_index") or 0)
        history = payload.get("past_execution_history") or []
        if len(history) != max(turn_index - 1, 0):
            row_failures.append("pre_action_history_length_mismatch")

        schema = next(
            (
                item.get("function")
                for item in row.get("tools") or []
                if (item.get("function") or {}).get("name") == name
            ),
            None,
        )
        if schema is None:
            row_failures.append("target_tool_missing_from_native_tools")
        elif not isinstance(arguments, dict):
            row_failures.append("tool_arguments_not_object")
        elif list(Draft202012Validator(schema["parameters"]).iter_errors(arguments)):
            row_failures.append("tool_arguments_fail_native_schema")

        if metadata.get("policy_view") != "evidence_first_reduced_control_v1":
            row_failures.append("evidence_first_policy_view_missing")
        if metadata.get("teacher_action_used") is not True:
            row_failures.append("teacher_provenance_not_declared")
        if metadata.get("oracle_or_verifier_used_during_generation") is not False:
            row_failures.append("generation_oracle_separation_not_declared")
        if metadata.get("verified_gold_bundle_accepted") is not True:
            row_failures.append("gold_bundle_acceptance_not_declared")
        if row_failures:
            failures.append({"row_id": row_id, "failures": row_failures})

    duplicate_row_ids = sorted(
        row_id for row_id, count in Counter(row_ids).items() if count > 1
    )
    reasoning_audit = _reasoning_audit(rows)
    reasoning_passes = sum(
        bool(item["passes_causal_reasoning"]) for item in reasoning_audit
    )
    sensitive_hits = _sensitive_key_hits(rows)
    exposure, _exposure_rows = audit_control_plane(rows)
    semantic_preexposure = exposure["overall"]["counts"][
        "has_singleton_answer_like_field"
    ]
    return {
        "validator": "evidence_first_teacher_sft_v1",
        "rows": len(rows),
        "structural_all_passed": not failures and not duplicate_row_ids,
        "structural_failed_rows": len(failures),
        "failures": failures,
        "duplicate_row_ids": duplicate_row_ids,
        "causal_reasoning_passed_rows": reasoning_passes,
        "causal_reasoning_failed_rows": len(rows) - reasoning_passes,
        "semantic_target_preexposure_rows": semantic_preexposure,
        "sensitive_model_visible_key_hits": len(sensitive_hits),
        "tool_rows": dict(tools_used),
        "training_qualified": False,
        "qualification_note": (
            "Structural validity does not qualify these teacher rows for training. All causal reasoning rows, "
            "closed-loop evidence sensitivity, and sealed cross-dataset evaluation must still pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = validate_rows(_jsonl(args.input))
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
