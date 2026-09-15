"""Build decision-sufficient SFT views from executable full transcripts.

The full transcript remains the audit source of truth.  The compact view keeps
exactly one authoritative pre-action harness state, a lossless task summary, a
small action/outcome history, and the unchanged assistant target.  This matches
the state-externalized context used by the CLI teacher without training on many
duplicated historical harness snapshots.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from student_tool_contract import (
    build_student_tools,
    canonical_student_arguments,
    canonicalize_student_target,
    student_tool_names,
)


EVIDENCE_ID_PATTERN = re.compile(r"\bEA-[A-Za-z0-9][A-Za-z0-9_-]*\b")
PATH_OR_LOCATOR_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?i)[a-z]:\\+[a-z0-9_. -]"),
    "unix_home_path": re.compile(r"/(?:home|users)/[^\"\s]+", re.I),
    "media_filename": re.compile(r"\.(?:mp4|mov|avi|mkv|webm)(?=[\"\\/,\s]|$)", re.I),
    "raw_feature_filename": re.compile(r"\.npy(?=[\"\\/,\s]|$)", re.I),
    "url": re.compile(r"(?:https?|file)://", re.I),
    "locator_field": re.compile(r'"(?:local_)?(?:path|root|uri|url)"\s*:', re.I),
    "oracle_field": re.compile(r'"oracle[^\"]*"\s*:', re.I),
    "verifier_field": re.compile(r'"verifier[^\"]*"\s*:', re.I),
}
OBSERVATION_SUMMARY_KEYS = (
    "status",
    "authority",
    "reason",
    "error_type",
    "error",
    "accepted_fields",
    "rejected_fields",
    "updated_fields",
    "decision_status",
    "created_evidence_ids",
    "claim_id",
    "stop_decision",
)
CONTEXT_VIEW = "decision_sufficient_state_v4_external_tool_contract"
TARGET_ENVELOPE_KEYS = {"tool_name", "arguments", "tool_calls", "function", "call_id"}


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _assistant_call(message: dict[str, Any]) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    call = calls[0]
    function = call.get("function") or {}
    return {
        "tool_name": function.get("name"),
        "arguments": _json_object(function.get("arguments")) or {},
        "call_id": call.get("id"),
    }


def _last_assistant_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("SFT row has no assistant target")


def _visible_chars(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    return len(
        json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False, separators=(",", ":"))
    )


def _logical_refs(row: dict[str, Any]) -> tuple[str, str]:
    dataset = str((row.get("metadata") or {}).get("dataset", ""))
    if dataset == "VCSL":
        return "query", "reference"
    if dataset == "VidDiffBench":
        return "video_a", "video_b"
    return "source", "suspect"


def _normalize_private_locators(value: Any, row: dict[str, Any]) -> Any:
    """Replace media locators with stable logical refs without changing state semantics."""
    source_ref, suspect_ref = _logical_refs(row)

    def find_assets(item: Any) -> dict[str, Any] | None:
        if isinstance(item, dict):
            if isinstance(item.get("visible_assets"), dict):
                return item["visible_assets"]
            for child in item.values():
                found = find_assets(child)
                if found is not None:
                    return found
        elif isinstance(item, list):
            for child in item:
                found = find_assets(child)
                if found is not None:
                    return found
        return None

    raw_assets = find_assets(value) or {}
    replacements = {
        str(raw_assets["source"]): source_ref
        for _ in (0,)
        if isinstance(raw_assets.get("source"), str)
    }
    if isinstance(raw_assets.get("suspect"), str):
        replacements[str(raw_assets["suspect"])] = suspect_ref

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            if item in replacements:
                return replacements[item]
            if any(pattern.search(item) for pattern in PATH_OR_LOCATOR_PATTERNS.values()):
                return "opaque_artifact"
            return item
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, dict):
            return copy.deepcopy(item)
        result: dict[str, Any] = {}
        role = str(item.get("role", "")).upper()
        for key, child in item.items():
            if re.search(r"(?i)(?:^|_)(?:path|root|uri|url)$", key) or key == "candidate_report":
                if key == "video_path" and role in {"A", "B"}:
                    result["media_ref"] = "video_a" if role == "A" else "video_b"
                continue
            result[key] = visit(child)
        assets = result.get("visible_assets")
        if isinstance(assets, dict):
            if "source" in assets:
                assets["source"] = source_ref
            if "suspect" in assets:
                assets["suspect"] = suspect_ref
        return result

    return visit(value)


def compact_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a compact training view while preserving the exact target action."""
    messages = copy.deepcopy(row.get("messages") or [])
    target_index = _last_assistant_index(messages)
    source_target = copy.deepcopy(messages[target_index])
    target = canonicalize_student_target(source_target)
    prefix = messages[:target_index]
    system = next((copy.deepcopy(item) for item in prefix if item.get("role") == "system"), None)
    user = next((item for item in prefix if item.get("role") == "user"), None)
    if system is None or user is None:
        raise ValueError("SFT row must contain system and initial user messages")
    initial = _json_object(user.get("content"))
    if initial is None or not isinstance(initial.get("harness_state"), dict):
        raise ValueError("initial SFT user payload must contain a harness_state object")

    claim_schema = initial.get("claim_schema")
    tool_registry = initial.get("tool_registry")
    if not isinstance(claim_schema, dict) or not isinstance(tool_registry, list):
        raise ValueError("initial SFT user payload must contain claim_schema and tool_registry")

    current_state = _normalize_private_locators(initial["harness_state"], row)
    task_summary = {
        key: _normalize_private_locators(value, row)
        for key, value in initial.items()
        if key not in {"harness_state", "tool_registry"}
    }
    history: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    latest_summary: dict[str, Any] = {}
    for message in prefix:
        role = message.get("role")
        if role == "assistant":
            pending = _assistant_call(message)
        elif role == "tool":
            observation = _json_object(message.get("content")) or {}
            state = observation.get("harness_state")
            if isinstance(state, dict):
                current_state = _normalize_private_locators(state, row)
            latest_summary = {
                key: copy.deepcopy(observation[key])
                for key in OBSERVATION_SUMMARY_KEYS
                if key in observation
            }
            history.append({
                "step_index": len(history),
                "past_action": {
                    "executed_tool": (pending or {}).get("tool_name", message.get("name")),
                    "past_input": canonical_student_arguments(
                        str((pending or {}).get("tool_name", message.get("name"))),
                        copy.deepcopy((pending or {}).get("arguments", {})),
                    ),
                },
                "environment_outcome": copy.deepcopy(latest_summary),
            })
            pending = None

    compact_payload = {
        "task": task_summary,
        "current_harness_state": current_state,
        "past_execution_history": history,
        "latest_environment_observation": latest_summary,
        "compression_contract": (
            "The full executable trajectory is retained in the audit source. "
            "This training view removes duplicate historical states only; "
            "current_harness_state is the authoritative pre-action state. "
            "past_execution_history is environment history, not an output template; "
            "the assistant must emit natural-language reasoning plus exactly one native tool call."
        ),
    }
    compact_messages = [
        system,
        {"role": "user", "content": json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))},
        target,
    ]
    eligible_tools = [str(name) for name in current_state.get("eligible_tools", [])]
    student_tools = build_student_tools(claim_schema, tool_registry, eligible_tools)
    result = copy.deepcopy(row)
    result["messages"] = compact_messages
    result["tools"] = student_tools
    metadata = result.setdefault("metadata", {})
    source_target_index = metadata.get("target_message_index", target_index)
    source_chars = _visible_chars(messages, student_tools)
    compact_chars = _visible_chars(compact_messages, student_tools)
    metadata.update({
        "context_view": CONTEXT_VIEW,
        "source_target_message_index": source_target_index,
        "target_message_index": 2,
        "source_message_count": len(messages),
        "compact_message_count": 3,
        "source_visible_chars": source_chars,
        "compact_visible_chars": compact_chars,
        "context_char_ratio": round(compact_chars / max(source_chars, 1), 6),
        "private_locator_normalized": True,
        "tool_contract": "eligible_qwen_function_schemas_v1",
        "target_contract": "model_visible_external_arguments_v1",
        "target_interface_normalized": (
            (_assistant_call(target) or {}).get("arguments")
            != (_assistant_call(source_target) or {}).get("arguments")
        ),
    })
    return result


def validate_compact_sft_row(source: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    """Prove state/target fidelity and report trainability properties."""
    source_messages = source.get("messages") or []
    compact_messages = compact.get("messages") or []
    source_target = source_messages[_last_assistant_index(source_messages)]
    expected_student_target = canonicalize_student_target(source_target)
    compact_target = compact_messages[_last_assistant_index(compact_messages)]
    initial_message = next(item for item in source_messages if item.get("role") == "user")
    initial = _json_object(initial_message.get("content")) or {}
    compact_payload = _json_object(compact_messages[1].get("content")) if len(compact_messages) >= 2 else None
    compact_payload = compact_payload or {}

    expected_state = _normalize_private_locators(initial.get("harness_state"), source)
    expected_history: list[str] = []
    last_tool_payload: dict[str, Any] = {}
    target_index = _last_assistant_index(source_messages)
    for message in source_messages[:target_index]:
        if message.get("role") == "assistant":
            call = _assistant_call(message)
            if call:
                expected_history.append(str(call.get("tool_name")))
        elif message.get("role") == "tool":
            last_tool_payload = _json_object(message.get("content")) or {}
            if isinstance(last_tool_payload.get("harness_state"), dict):
                expected_state = _normalize_private_locators(last_tool_payload["harness_state"], source)

    history = compact_payload.get("past_execution_history") or []
    actual_history = [str((item.get("past_action") or {}).get("executed_tool")) for item in history]
    current_state = compact_payload.get("current_harness_state")
    reasoning = str(compact_target.get("content", ""))
    visible_ids = set((current_state or {}).get("available_evidence_ids", [])).union(
        (current_state or {}).get("dropped_evidence_ids", [])
    )
    referenced_ids = set(EVIDENCE_ID_PATTERN.findall(reasoning))
    target_call = _assistant_call(compact_target) or {}
    eligible_tools = set((current_state or {}).get("eligible_tools", []))
    compact_tools = compact.get("tools") or []
    compact_text = json.dumps(
        {"messages": compact_messages, "tools": compact_tools},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    privacy_hits = {
        name: len(pattern.findall(compact_text))
        for name, pattern in PATH_OR_LOCATOR_PATTERNS.items()
    }
    expected_tools = build_student_tools(
        initial.get("claim_schema") or {},
        initial.get("tool_registry") or [],
        [str(name) for name in (current_state or {}).get("eligible_tools", [])],
    )
    source_chars = _visible_chars(source_messages, expected_tools)
    compact_chars = _visible_chars(compact_messages, compact_tools)
    prior_tool_count = len(expected_history)
    efficiency_ok = compact_chars <= source_chars * (0.80 if prior_tool_count >= 2 else 1.05)
    def nested_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(nested_keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(nested_keys(child) for child in value)) if value else set()
        return set()

    history_key_collisions = sorted(nested_keys(history).intersection(TARGET_ENVELOPE_KEYS))
    schema_names = student_tool_names(compact_tools)
    schema_by_name = {
        str(tool["function"]["name"]): tool["function"]["parameters"]
        for tool in compact_tools
    }
    target_schema_errors: list[str] = []
    target_name = str(target_call.get("tool_name"))
    if target_name in schema_by_name:
        target_schema_errors = [
            error.message
            for error in Draft202012Validator(schema_by_name[target_name]).iter_errors(
                target_call.get("arguments") or {}
            )
        ]
    all_registered_names = [str(item["name"]) for item in initial.get("tool_registry", [])]
    all_student_tools = build_student_tools(
        initial.get("claim_schema") or {}, initial.get("tool_registry") or [], all_registered_names
    )
    all_schema_by_name = {
        str(tool["function"]["name"]): tool["function"]["parameters"]
        for tool in all_student_tools
    }
    history_schema_errors: list[str] = []
    for item in history:
        past_action = item.get("past_action") or {}
        name = str(past_action.get("executed_tool"))
        parameters = all_schema_by_name.get(name)
        if parameters is None:
            history_schema_errors.append(f"missing_schema:{name}")
            continue
        history_schema_errors.extend(
            f"{name}:{error.message}"
            for error in Draft202012Validator(parameters).iter_errors(past_action.get("past_input") or {})
        )
    checks = {
        "assistant_reasoning_and_tool_choice_preserved": (
            compact_target.get("content") == source_target.get("content")
            and (_assistant_call(compact_target) or {}).get("tool_name")
            == (_assistant_call(source_target) or {}).get("tool_name")
        ),
        "assistant_target_normalized_to_visible_tool_contract": compact_target == expected_student_target,
        "decision_state_preserved_after_locator_normalization": current_state == expected_state,
        "task_payload_preserved": compact_payload.get("task") == {
            key: _normalize_private_locators(value, source)
            for key, value in initial.items()
            if key not in {"harness_state", "tool_registry"}
        },
        "prior_action_order_preserved": actual_history == expected_history,
        "latest_outcome_status_preserved": (
            not last_tool_payload
            or compact_payload.get("latest_environment_observation", {}).get("status") == last_tool_payload.get("status")
        ),
        "single_authoritative_state_only": all("harness_state" not in item for item in history),
        "history_is_role_separated_from_target_envelope": not history_key_collisions,
        "target_tool_is_prestate_eligible": target_call.get("tool_name") in eligible_tools,
        "tool_schemas_exactly_match_prestate_eligibility": schema_names == list((current_state or {}).get("eligible_tools", [])),
        "tool_schemas_match_executable_teacher_contract": compact_tools == expected_tools,
        "target_tool_has_student_schema": target_call.get("tool_name") in schema_names,
        "target_arguments_validate_student_schema": not target_schema_errors,
        "past_inputs_match_external_tool_contract": not history_schema_errors,
        "reasoning_references_only_visible_evidence": referenced_ids.issubset(visible_ids),
        "no_private_locator_or_oracle_fields": not any(privacy_hits.values()),
        "context_is_not_larger_than_allowed": efficiency_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "privacy_hits": privacy_hits,
        "source_visible_chars": source_chars,
        "compact_visible_chars": compact_chars,
        "context_char_ratio": round(compact_chars / max(source_chars, 1), 6),
        "prior_tool_count": prior_tool_count,
        "history_target_envelope_key_collisions": history_key_collisions,
        "student_tool_names": schema_names,
        "target_schema_errors": target_schema_errors,
        "history_schema_errors": history_schema_errors,
    }


def audit_jsonl(paths: list[Path]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = [validate_compact_sft_row(row, compact_sft_row(row)) for row in rows]
    ratios = sorted(result["context_char_ratio"] for result in results)
    failures = Counter(
        check
        for result in results
        for check, passed in result["checks"].items()
        if not passed
    )
    privacy_hits = Counter(
        name
        for result in results
        for name, count in result["privacy_hits"].items()
        for _ in range(count)
    )
    privacy_failure_examples = [
        str((row.get("metadata") or {}).get("row_id", (row.get("metadata") or {}).get("case_id")))
        for row, result in zip(rows, results)
        if any(result["privacy_hits"].values())
    ][:10]

    def percentile(q: float) -> float:
        if not ratios:
            return 0.0
        return ratios[round((len(ratios) - 1) * q)]

    return {
        "rows": len(rows),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "failure_counts": dict(failures),
        "privacy_hit_counts": dict(privacy_hits),
        "privacy_failure_examples": privacy_failure_examples,
        "context_char_ratio": {
            "mean": round(sum(ratios) / max(len(ratios), 1), 6),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "max": max(ratios, default=0.0),
        },
        "source_visible_chars": sum(result["source_visible_chars"] for result in results),
        "compact_visible_chars": sum(result["compact_visible_chars"] for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_jsonl(args.paths), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
