"""Audit student-visible control-plane leakage in an SFT trajectory file."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COMPILED_ACTION_KEYS = {
    "executable_binding_actions",
    "executable_decision_actions",
    "executable_evidence_assessment_actions",
    "executable_adjudication_actions",
}
ANSWER_LIKE_FIELDS = {
    "preferred_video",
    "direction",
    "relationship",
    "claim_family",
    "manipulation_type",
    "localization_method",
    "measurement_method",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, path + (str(index),))


def _preaction_payload(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    visible_texts: list[str] = []
    for message in row.get("messages", [])[:-1]:
        content = message.get("content")
        if isinstance(content, str):
            visible_texts.append(content)
    for content in reversed(visible_texts):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            isinstance(payload.get("current_harness_state"), dict)
            or isinstance(payload.get("harness_state"), dict)
        ):
            return payload, "\n".join(visible_texts)
    raise ValueError(f"no pre-action harness payload in {row.get('metadata', {}).get('row_id')}")


def _target_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    call = row["messages"][-1]["tool_calls"][0]["function"]
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return str(call["name"]), arguments


def _values_for_key(root: Any, key: str) -> list[Any]:
    return [value for path, value in _walk(root) if path and path[-1] == key]


def _has_nonempty_key(root: Any, key: str) -> bool:
    return any(value not in (None, [], {}, "") for value in _values_for_key(root, key))


def _action_signature(value: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        function = value["function"]
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name") or value.get("tool_name") or value.get("tool")
        arguments = value.get("arguments") or value.get("params")
    if name and isinstance(arguments, dict):
        return str(name), arguments
    return None


def _compiled_actions(root: Any) -> list[tuple[str, dict[str, Any]]]:
    actions: list[tuple[str, dict[str, Any]]] = []
    for path, value in _walk(root):
        if not path or path[-1] not in COMPILED_ACTION_KEYS or not isinstance(value, list):
            continue
        for item in value:
            signature = _action_signature(item)
            if signature is not None:
                actions.append(signature)
    return actions


def _unique_schema_value(schema: Any) -> tuple[bool, Any]:
    if not isinstance(schema, dict):
        return False, None
    if "const" in schema:
        return True, schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and len(enum) == 1:
        return True, enum[0]
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        required = schema.get("required") or []
        result: dict[str, Any] = {}
        for key in required:
            unique, value = _unique_schema_value(schema["properties"].get(key))
            if not unique:
                return False, None
            result[str(key)] = value
        return True, result
    if schema.get("type") == "array":
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and minimum == maximum:
            unique, value = _unique_schema_value(schema.get("items"))
            if unique:
                return True, [value for _ in range(minimum)]
    return False, None


def _tool_schema(row: dict[str, Any], name: str) -> dict[str, Any] | None:
    for tool in row.get("tools") or []:
        function = tool.get("function") or {}
        if function.get("name") == name:
            return function.get("parameters") or {}
    return None


def _claim_schema(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload.get("task") or {}
    if isinstance(task.get("claim_schema"), dict):
        return task["claim_schema"]
    state = payload.get("current_harness_state") or payload.get("harness_state") or {}
    if isinstance(state.get("claim_schema"), dict):
        return state["claim_schema"]
    return {}


def _action_family(tool: str) -> str:
    if tool == "create_candidate_claim":
        return "create_claim"
    if tool == "bind_evidence":
        return "bind_evidence"
    if tool == "set_field_decision":
        return "field_decision"
    if tool == "stop":
        return "stop"
    if tool in {"assess_evidence_candidate", "adjudicate_evidence"}:
        return "evidence_review"
    return "measurement"


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    flags = (
        "eligible_zero",
        "eligible_one",
        "eligible_many",
        "has_executable_binding_actions",
        "has_executable_decision_actions",
        "has_repair_restoration_candidates",
        "has_must_reopen_before_rebinding",
        "has_compiled_full_action",
        "has_fully_singleton_target_tool_schema",
        "target_call_exactly_preexposed",
        "contains_only_eligible_tool_phrase",
        "has_singleton_target_field_enum",
        "has_singleton_answer_like_field",
        "teacher_target_value_unique_before_execution",
    )
    return {
        "turns": len(records),
        "cases": len({item["case_id"] for item in records}),
        "counts": {flag: sum(bool(item[flag]) for item in records) for flag in flags},
        "rates": {
            flag: round(sum(bool(item[flag]) for item in records) / max(len(records), 1), 6)
            for flag in flags
        },
        "eligible_tool_count_histogram": dict(
            sorted(Counter(str(item["eligible_tool_count"]) for item in records).items())
        ),
        "target_tools": dict(sorted(Counter(item["target_tool"] for item in records).items())),
    }


def audit(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        payload, visible_text = _preaction_payload(row)
        state = payload.get("current_harness_state") or payload.get("harness_state") or {}
        eligible = list(state.get("eligible_tools") or [])
        target_tool, target_arguments = _target_call(row)
        actions = _compiled_actions(payload)
        schema = _tool_schema(row, target_tool)
        singleton_tool, singleton_arguments = _unique_schema_value(schema)
        exact_preexposed = (target_tool, target_arguments) in actions or (
            singleton_tool and singleton_arguments == target_arguments
        )
        claim_schema = _claim_schema(payload)
        fields = claim_schema.get("fields") or {}
        target_fields = claim_schema.get("target_fields") or []
        singleton_fields = []
        singleton_answer_fields = []
        for field in target_fields:
            field_schema = fields.get(field) or {}
            enum = field_schema.get("enum")
            if isinstance(enum, list) and len(enum) == 1:
                singleton_fields.append(str(field))
                if str(field) in ANSWER_LIKE_FIELDS:
                    singleton_answer_fields.append(str(field))
        record = {
            "row_id": row["metadata"]["row_id"],
            "case_id": row["metadata"]["case_id"],
            "dataset": row["metadata"]["dataset"],
            "trajectory_mode": row["metadata"].get("trajectory_mode"),
            "target_tool": target_tool,
            "action_family": _action_family(target_tool),
            "eligible_tool_count": len(eligible),
            "eligible_tools": eligible,
            "eligible_zero": len(eligible) == 0,
            "eligible_one": len(eligible) == 1,
            "eligible_many": len(eligible) > 1,
            "has_executable_binding_actions": _has_nonempty_key(
                payload, "executable_binding_actions"
            ),
            "has_executable_decision_actions": _has_nonempty_key(
                payload, "executable_decision_actions"
            ),
            "has_repair_restoration_candidates": _has_nonempty_key(
                payload, "repair_restoration_candidates"
            ),
            "has_must_reopen_before_rebinding": _has_nonempty_key(
                payload, "must_reopen_before_rebinding"
            ),
            "has_compiled_full_action": bool(actions),
            "compiled_action_count": len(actions),
            "has_fully_singleton_target_tool_schema": singleton_tool,
            "target_call_exactly_preexposed": exact_preexposed,
            "contains_only_eligible_tool_phrase": bool(
                re.search(r"\b(?:the\s+)?only\s+eligible\s+tool\b", visible_text, re.I)
            ),
            "singleton_target_fields": singleton_fields,
            "has_singleton_target_field_enum": bool(singleton_fields),
            "singleton_answer_like_fields": singleton_answer_fields,
            "has_singleton_answer_like_field": bool(singleton_answer_fields),
            "teacher_target_value_unique_before_execution": exact_preexposed
            or bool(singleton_fields),
        }
        records.append(record)

    by_dataset = {
        key: _summarize([item for item in records if item["dataset"] == key])
        for key in sorted({item["dataset"] for item in records})
    }
    by_action = {
        key: _summarize([item for item in records if item["action_family"] == key])
        for key in sorted({item["action_family"] for item in records})
    }
    repair_records = [item for item in records if item["trajectory_mode"] == "repair"]
    report = {
        "artifact": "control_plane_exposure_audit_v1",
        "definitions": {
            "compiled_full_action": (
                "A non-empty executable_*_actions entry containing both a tool name and a complete "
                "arguments object in the pre-action student-visible JSON."
            ),
            "fully_singleton_target_tool_schema": (
                "Every required argument of the target tool is recursively fixed by const or a "
                "single-value enum (and fixed-length arrays when applicable)."
            ),
            "target_call_exactly_preexposed": (
                "The teacher target name+arguments exactly equals a visible compiled action or the "
                "unique materialization of the visible target-tool schema."
            ),
            "singleton_target_field_enum": (
                "At least one claim_schema.target_fields entry has an enum of length one; this is "
                "reported separately from exact action pre-exposure."
            ),
            "repair_rollup": "All turns whose trajectory_mode is repair, independent of target tool.",
        },
        "overall": _summarize(records),
        "by_dataset": by_dataset,
        "by_action_family": by_action,
        "repair_rollup": _summarize(repair_records),
        "case_coverage": {
            "singleton_target_field_enum_cases": len(
                {item["case_id"] for item in records if item["has_singleton_target_field_enum"]}
            ),
            "compiled_full_action_cases": len(
                {item["case_id"] for item in records if item["has_compiled_full_action"]}
            ),
            "exact_preexposed_target_cases": len(
                {item["case_id"] for item in records if item["target_call_exactly_preexposed"]}
            ),
        },
        "initial_estimate_review": {
            "eligible_one_claimed_approx": 165,
            "compiled_full_action_claimed_approx": 90,
            "singleton_target_field_turns_claimed_approx": 117,
            "singleton_target_field_cases_claimed_approx": 19,
            "actual": {
                "eligible_one": sum(item["eligible_one"] for item in records),
                "compiled_full_action": sum(item["has_compiled_full_action"] for item in records),
                "singleton_target_field_turns": sum(
                    item["has_singleton_target_field_enum"] for item in records
                ),
                "singleton_target_field_cases": len(
                    {item["case_id"] for item in records if item["has_singleton_target_field_enum"]}
                ),
            },
        },
    }
    return report, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()
    report, records = audit(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    print(json.dumps(report["initial_estimate_review"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
