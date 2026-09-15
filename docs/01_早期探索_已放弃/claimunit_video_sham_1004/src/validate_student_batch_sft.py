"""Validate closed-loop student SFT rows before they become training input."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from pipeline import write_json
from shadow_execute_generated_actions import _jsonl


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    row_ids: list[str] = []
    reasoning_texts: list[str] = []
    datasets: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    tools_used: Counter[str] = Counter()

    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        row_id = str(metadata.get("row_id") or f"missing-row-id-{index}")
        row_ids.append(row_id)
        datasets[str(metadata.get("dataset"))] += 1
        modes[str(metadata.get("trajectory_mode"))] += 1
        row_failures: list[str] = []

        messages = row.get("messages") or []
        if len(messages) != 3 or [item.get("role") for item in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            row_failures.append("messages_must_be_system_user_assistant")
            payload = {}
            assistant = {}
        else:
            try:
                payload = json.loads(messages[1]["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                payload = {}
                row_failures.append("user_payload_not_valid_json")
            assistant = messages[2]

        reasoning = str(assistant.get("content") or "").strip()
        reasoning_texts.append(reasoning)
        if not reasoning:
            row_failures.append("missing_natural_language_reasoning")

        calls = assistant.get("tool_calls") or []
        if len(calls) != 1:
            row_failures.append("assistant_must_have_exactly_one_tool_call")
            function = {}
        else:
            function = calls[0].get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if name:
            tools_used[str(name)] += 1
        if name != metadata.get("target_tool"):
            row_failures.append("target_tool_metadata_mismatch")

        state = payload.get("current_harness_state") or {}
        if name not in (state.get("eligible_tools") or []):
            row_failures.append("target_tool_not_visible_as_eligible")
        turn_index = int(metadata.get("turn_index") or 0)
        history = payload.get("past_execution_history") or []
        if len(history) != max(turn_index - 1, 0):
            row_failures.append("pre_action_history_length_mismatch")

        function_schema = next(
            (
                tool.get("function")
                for tool in row.get("tools") or []
                if (tool.get("function") or {}).get("name") == name
            ),
            None,
        )
        if function_schema is None:
            row_failures.append("target_tool_missing_from_native_tools")
        elif not isinstance(arguments, dict):
            row_failures.append("tool_arguments_not_object")
        elif list(
            Draft202012Validator(function_schema["parameters"]).iter_errors(arguments)
        ):
            row_failures.append("tool_arguments_fail_native_schema")

        if metadata.get("teacher_action_used") is not False:
            row_failures.append("teacher_action_flag_not_false")
        if metadata.get("oracle_or_verifier_used") is not False:
            row_failures.append("oracle_or_verifier_flag_not_false")
        if metadata.get("strict_closed_loop_accepted") is not True:
            row_failures.append("trajectory_not_strictly_accepted")

        if row_failures:
            failures.append({"row_id": row_id, "failures": row_failures})

    duplicate_row_ids = sorted(
        row_id for row_id, count in Counter(row_ids).items() if count > 1
    )
    reasoning_rows: dict[str, list[str]] = {}
    for row_id, reasoning in zip(row_ids, reasoning_texts):
        if reasoning:
            reasoning_rows.setdefault(reasoning, []).append(row_id)
    duplicate_reasoning_groups = [
        {"row_ids": ids, "reasoning": reasoning}
        for reasoning, ids in reasoning_rows.items()
        if len(ids) > 1
    ]
    exact_reasoning_duplicates = sum(
        len(group["row_ids"]) - 1 for group in duplicate_reasoning_groups
    )
    return {
        "validator": "student_closed_loop_sft_v1",
        "rows": len(rows),
        "all_passed": not failures and not duplicate_row_ids,
        "failed_rows": len(failures),
        "failures": failures,
        "duplicate_row_ids": duplicate_row_ids,
        "exact_reasoning_duplicates": exact_reasoning_duplicates,
        "exact_reasoning_duplicate_groups": duplicate_reasoning_groups,
        "dataset_rows": dict(datasets),
        "mode_rows": dict(modes),
        "tool_rows": dict(tools_used),
    }


def diversity_balance_rows(
    rows: list[dict[str, Any]],
    *,
    max_exact_reasoning_per_tool: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    kept: list[dict[str, Any]] = []
    removed_by_tool: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        assistant = (row.get("messages") or [{}, {}, {}])[-1]
        tool = str(metadata.get("target_tool") or "unknown")
        reasoning = str(assistant.get("content") or "").strip()
        key = (tool, reasoning)
        counts[key] += 1
        if counts[key] <= max_exact_reasoning_per_tool:
            kept.append(row)
        else:
            removed_by_tool[tool] += 1
    return kept, {
        "policy": "cap_exact_assistant_reasoning_per_target_tool",
        "max_exact_reasoning_per_tool": max_exact_reasoning_per_tool,
        "input_rows": len(rows),
        "output_rows": len(kept),
        "removed_rows": len(rows) - len(kept),
        "removed_by_tool": dict(removed_by_tool),
        "full_audit_export_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_rows(_jsonl(args.input))
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
