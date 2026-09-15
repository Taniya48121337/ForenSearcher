"""Compile accepted trajectories into an evidence-first, reduced-control SFT view.

The source trajectory and executed target stay unchanged.  Only the student-visible
pre-action context is rewritten: compiled actions, repair recipes, instance-level
tool constants, and singleton answer enums are removed.  The compiler also emits
sampling weights and matched causal pairs so bookkeeping turns cannot dominate
semantic supervision.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from audit_control_plane_exposure import audit as audit_control_plane
from causal_trajectory_generation import (
    CAUSAL_SYSTEM_PROMPT,
    COMPILED_CONTROL_KEYS,
    EVIDENCE_ID,
    REPAIR_RECIPE_KEYS,
    _reasoning_score,
)
from llm_teacher_rollout import TeacherAction
from pipeline import ROOT, write_json, write_jsonl
from student_tool_contract import build_student_tools, canonicalize_student_target
from vcsl_batch import VCSL_CAUSAL_TOOL_REGISTRY
from viddiff_batch import VIDDIFF_CAUSAL_TOOL_REGISTRY
from videosham_batch import MULTIFAMILY_CAUSAL_TOOL_REGISTRY


DEFAULT_INPUT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_current40"
    / "current40_sft_v5_diverse.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/evidence_first_sft_v6"
)

REGISTRIES = {
    "VideoSham": MULTIFAMILY_CAUSAL_TOOL_REGISTRY,
    "VCSL": VCSL_CAUSAL_TOOL_REGISTRY,
    "VidDiffBench": VIDDIFF_CAUSAL_TOOL_REGISTRY,
}

DROP_KEYS = COMPILED_CONTROL_KEYS | REPAIR_RECIPE_KEYS | {
    "action_constraints",
    "compression_contract",
}

SUPERVISION = {
    "create_claim": ("evidence_interpretation", 1.0),
    "evidence_challenge": ("uncertainty_or_repair_detection", 1.2),
    "evidence_review": ("evidence_interpretation", 1.0),
    "measurement": ("semantic_tool_decision", 1.0),
    "bind_evidence": ("evidence_field_mapping", 0.8),
    "field_decision": ("state_transition", 0.25),
    "stop": ("control_only", 0.1),
}

EVIDENCE_FIRST_CONTRACT = (
    "Use the current task, claim state, and complete EvidenceAtom observations to choose the next operation. "
    "No compiled action, binding recipe, restoration recipe, or instance-fixed target arguments are supplied. "
    "Derive claim values and evidence-field links from visible semantics; preserve task identity and independent "
    "fields, and investigate or abstain when evidence is incompatible, partial, empty, stale, or conflicting."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action_family(tool_name: str) -> str:
    if tool_name == "create_candidate_claim":
        return "create_claim"
    if tool_name == "bind_evidence":
        return "bind_evidence"
    if tool_name == "challenge_evidence":
        return "evidence_challenge"
    if tool_name == "set_field_decision":
        return "field_decision"
    if tool_name == "stop":
        return "stop"
    if tool_name in {"assess_evidence_candidate", "adjudicate_evidence"}:
        return "evidence_review"
    return "measurement"


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(child)
            for key, child in value.items()
            if str(key) not in DROP_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return copy.deepcopy(value)


def _sanitize_claim_schema(schema: dict[str, Any]) -> dict[str, Any]:
    revised = copy.deepcopy(schema)
    target_fields = {str(item) for item in revised.get("target_fields") or []}
    for field_name, field_schema in (revised.get("fields") or {}).items():
        if str(field_name) not in target_fields or not isinstance(field_schema, dict):
            continue
        enum = field_schema.get("enum")
        if isinstance(enum, list) and len(enum) == 1:
            field_schema.pop("enum", None)
        for key in ("const", "default", "examples"):
            field_schema.pop(key, None)
    return revised


def _claim_schema(
    payload: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    task = payload.get("task") or {}
    if isinstance(task.get("claim_schema"), dict):
        return task["claim_schema"]
    state = payload.get("current_harness_state") or payload.get("harness_state") or {}
    if isinstance(state.get("claim_schema"), dict):
        return state["claim_schema"]
    if isinstance(fallback, dict):
        return fallback
    raise ValueError("student row has no visible claim_schema")


def _state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("current_harness_state") or payload.get("harness_state")
    if not isinstance(state, dict):
        raise ValueError("student row has no pre-action harness state")
    return state


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    revised = _sanitize_value(payload)
    task = revised.get("task") or {}
    if isinstance(task.get("claim_schema"), dict):
        task["claim_schema"] = _sanitize_claim_schema(task["claim_schema"])
    state = revised.get("current_harness_state") or revised.get("harness_state") or {}
    if isinstance(state.get("claim_schema"), dict):
        state["claim_schema"] = _sanitize_claim_schema(state["claim_schema"])
    revised["evidence_first_contract"] = EVIDENCE_FIRST_CONTRACT
    return revised


def _target(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assistant = row["messages"][-1]
    call = assistant["tool_calls"][0]["function"]
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("target tool arguments must be an object")
    return str(call["name"]), arguments


def _preaction_message_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 2, -1, -1):
        content = messages[index].get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            isinstance(payload.get("current_harness_state"), dict)
            or isinstance(payload.get("harness_state"), dict)
        ):
            return index
    raise ValueError("no pre-action harness payload")


def _generic_tools(
    *, dataset: str, claim_schema: dict[str, Any], eligible_tools: list[str]
) -> list[dict[str, Any]]:
    registry = REGISTRIES.get(dataset)
    if registry is None:
        raise ValueError(f"unsupported dataset: {dataset}")
    return build_student_tools(
        claim_schema,
        registry,
        eligible_tools,
        executable_actions=[],
    )


def _validate_target(tools: list[dict[str, Any]], name: str, arguments: dict[str, Any]) -> None:
    matches = [tool for tool in tools if (tool.get("function") or {}).get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"target tool is not uniquely visible: {name}")
    schema = (matches[0].get("function") or {}).get("parameters") or {}
    errors = sorted(Draft202012Validator(schema).iter_errors(arguments), key=lambda item: list(item.path))
    if errors:
        raise ValueError(f"target arguments fail generic schema for {name}: {errors[0].message}")


def compile_row(
    row: dict[str, Any], *, fallback_claim_schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    revised = copy.deepcopy(row)
    messages = revised.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("row must end in the supervised assistant action")
    payload_index = _preaction_message_index(messages)
    payload = json.loads(messages[payload_index]["content"])
    sanitized_payload = sanitize_payload(payload)
    claim_schema = _sanitize_claim_schema(
        _claim_schema(payload, fallback=fallback_claim_schema)
    )
    state = _state(sanitized_payload)
    eligible_tools = [str(item) for item in state.get("eligible_tools") or []]
    dataset = str(revised.get("metadata", {}).get("dataset") or "")

    messages[0]["content"] = CAUSAL_SYSTEM_PROMPT
    messages[payload_index]["content"] = json.dumps(
        sanitized_payload, ensure_ascii=False, separators=(",", ":")
    )
    messages[-1] = canonicalize_student_target(messages[-1])
    revised["messages"] = messages
    revised["tools"] = _generic_tools(
        dataset=dataset,
        claim_schema=claim_schema,
        eligible_tools=eligible_tools,
    )
    target_name, target_arguments = _target(revised)
    _validate_target(revised["tools"], target_name, target_arguments)

    family = _action_family(target_name)
    supervision_class, weight = SUPERVISION[family]
    metadata = revised.setdefault("metadata", {})
    metadata.update(
        {
            "source_row_id": metadata.get("row_id"),
            "policy_view": "evidence_first_reduced_control_v1",
            "target_interface": "generic_native_tool_schema",
            "action_family": family,
            "supervision_class": supervision_class,
            "training_weight": weight,
            "compiled_actions_visible": False,
            "repair_recipe_visible": False,
        }
    )
    return revised


def _pair_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_name, arguments = _target(row)
        if _action_family(target_name) not in {
            "create_claim",
            "evidence_challenge",
            "evidence_review",
            "measurement",
        }:
            continue
        payload_index = _preaction_message_index(row["messages"])
        payload = json.loads(row["messages"][payload_index]["content"])
        eligible = tuple(_state(payload).get("eligible_tools") or [])
        groups[(row["metadata"]["dataset"], target_name, eligible)].append(
            {"row": row, "arguments": arguments}
        )

    pairs: list[dict[str, Any]] = []
    used: set[str] = set()
    for key, candidates in sorted(groups.items(), key=lambda item: repr(item[0])):
        candidates.sort(key=lambda item: str(item["row"]["metadata"]["row_id"]))
        for left in candidates:
            left_id = str(left["row"]["metadata"]["row_id"])
            if left_id in used:
                continue
            right = next(
                (
                    item
                    for item in candidates
                    if str(item["row"]["metadata"]["row_id"]) not in used
                    and item["row"]["metadata"]["case_id"]
                    != left["row"]["metadata"]["case_id"]
                    and item["arguments"] != left["arguments"]
                ),
                None,
            )
            if right is None:
                continue
            right_id = str(right["row"]["metadata"]["row_id"])
            used.update({left_id, right_id})
            pairs.append(
                {
                    "pair_id": f"causal-pair-{len(pairs) + 1:04d}",
                    "dataset": key[0],
                    "target_tool": key[1],
                    "shared_eligible_tools": list(key[2]),
                    "left_row_id": left_id,
                    "right_row_id": right_id,
                    "left_case_id": left["row"]["metadata"]["case_id"],
                    "right_case_id": right["row"]["metadata"]["case_id"],
                    "target_arguments_differ": True,
                    "purpose": "matched-control different-evidence/different-target minibatch",
                }
            )
    return pairs


def _reasoning(row: dict[str, Any]) -> str:
    return str((row.get("messages") or [{}])[-1].get("content") or "").strip()


def _deduplicate_reasoning(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_reasoning(row)].append(row)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for reasoning, candidates in groups.items():
        candidates.sort(
            key=lambda row: (
                -float(row["metadata"].get("training_weight", 1.0)),
                str(row["metadata"]["row_id"]),
            )
        )
        kept.append(candidates[0])
        for duplicate in candidates[1:]:
            excluded.append(
                {
                    "row_id": duplicate["metadata"]["row_id"],
                    "kept_row_id": candidates[0]["metadata"]["row_id"],
                    "dataset": duplicate["metadata"]["dataset"],
                    "case_id": duplicate["metadata"]["case_id"],
                    "target_tool": duplicate["metadata"]["target_tool"],
                    "reason": "exact_duplicate_natural_language_reasoning",
                    "reasoning": reasoning,
                }
            )
    kept.sort(key=lambda row: str(row["metadata"]["row_id"]))
    excluded.sort(key=lambda row: str(row["row_id"]))
    return kept, excluded


def _reasoning_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recent_by_trajectory: dict[str, list[str]] = defaultdict(list)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(
                row["metadata"].get("trajectory_id")
                or f"{row['metadata'].get('case_id')}:{row['metadata'].get('trajectory_mode')}"
            ),
            int(row["metadata"].get("turn_index") or 0),
            str(row["metadata"]["row_id"]),
        ),
    )
    audit_rows: list[dict[str, Any]] = []
    for row in ordered:
        metadata = row["metadata"]
        trajectory_id = str(
            metadata.get("trajectory_id")
            or f"{metadata.get('case_id')}:{metadata.get('trajectory_mode')}"
        )
        payload_index = _preaction_message_index(row["messages"])
        payload = json.loads(row["messages"][payload_index]["content"])
        state = _state(payload)
        visible_evidence_ids = {
            str(item) for item in (state.get("evidence_store") or {}).keys()
        }
        name, arguments = _target(row)
        reasoning = _reasoning(row)
        action = TeacherAction(
            response_id="offline-audit",
            call_id="offline-audit",
            tool_name=name,
            reasoning=reasoning,
            params=arguments,
            raw_arguments=json.dumps(arguments, ensure_ascii=False),
            usage={},
            replay_items=[],
        )
        score, failures = _reasoning_score(
            action,
            visible_evidence_ids=visible_evidence_ids,
            eligible_count=len(row.get("tools") or []),
            recent_reasoning=recent_by_trajectory[trajectory_id][-4:],
        )
        # An empty object is the complete ordinary interface for stop, not a
        # missing semantic argument.
        if name == "stop" and "empty_arguments" in failures:
            failures = [item for item in failures if item != "empty_arguments"]
        critical = [
            item for item in failures if item != "decision_note_length_outside_35_140"
        ]
        audit_rows.append(
            {
                "row_id": metadata["row_id"],
                "dataset": metadata["dataset"],
                "case_id": metadata["case_id"],
                "trajectory_id": trajectory_id,
                "turn_index": metadata.get("turn_index"),
                "target_tool": name,
                "reasoning_score": round(score, 4),
                "failures": failures,
                "critical_failures": critical,
                "passes_causal_reasoning": score >= 0.875 and not critical,
                "visible_evidence_ids": sorted(visible_evidence_ids),
                "cited_visible_evidence_ids": sorted(
                    visible_evidence_ids.intersection(EVIDENCE_ID.findall(reasoning))
                ),
            }
        )
        recent_by_trajectory[trajectory_id].append(reasoning)
    audit_rows.sort(key=lambda item: str(item["row_id"]))
    return audit_rows


def _strict_reasoning_core(
    rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return only rows whose original reasoning passes the causal audit.

    This is deliberately a projection, not a rewrite: a failed old decision
    note is never repaired with a template and silently promoted to training
    data.
    """

    passed_ids = {
        str(item["row_id"])
        for item in audit_rows
        if bool(item["passes_causal_reasoning"])
    }
    return [
        row for row in rows if str(row["metadata"]["row_id"]) in passed_ids
    ]


def _sensitive_key_hits(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    forbidden = {
        "oracle",
        "verifier",
        "differences_gt",
        "source_path",
        "suspect_path",
        "local_path",
        "video_url",
        "url",
    }
    hits: list[dict[str, str]] = []

    def walk(value: Any, *, path: tuple[str, ...], row_id: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                name = str(key).lower()
                if name in forbidden:
                    hits.append({"row_id": row_id, "path": ".".join((*path, str(key)))})
                walk(child, path=(*path, str(key)), row_id=row_id)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path=(*path, str(index)), row_id=row_id)

    for row in rows:
        row_id = str(row["metadata"]["row_id"])
        walk(row.get("messages") or [], path=("messages",), row_id=row_id)
        walk(row.get("tools") or [], path=("tools",), row_id=row_id)
    return hits


def _apply_exposure_weights(
    rows: list[dict[str, Any]], exposure_rows: list[dict[str, Any]]
) -> dict[str, int]:
    """Downweight deterministic interface rehearsal without hiding it.

    An empty stop call or a fixed logical media reference is still a valid
    executable target, but it is not evidence reasoning.  We retain those rows
    for interface fluency and mark them explicitly instead of broadening the
    schema just to make an exposure metric look better.
    """

    by_id = {str(item["row_id"]): item for item in exposure_rows}
    counts = Counter()
    for row in rows:
        metadata = row["metadata"]
        exposure = by_id[str(metadata["row_id"])]
        if not exposure["target_call_exactly_preexposed"]:
            counts["not_preexposed"] += 1
            continue
        if metadata["action_family"] in {"measurement", "stop"}:
            metadata["supervision_class"] = (
                "control_only" if metadata["action_family"] == "stop" else "interface_deterministic"
            )
            metadata["training_weight"] = 0.1
            metadata["preexposure_interpretation"] = "ordinary_static_interface_arguments"
            counts[metadata["supervision_class"]] += 1
        else:
            metadata["preexposure_interpretation"] = "semantic_target_preexposure"
            counts["semantic_target_preexposure"] += 1
    return dict(sorted(counts.items()))


def compile_dataset(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    schemas_by_case: dict[str, dict[str, Any]] = {}
    schemas_by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        messages = row.get("messages") or []
        try:
            payload_index = _preaction_message_index(messages)
            payload = json.loads(messages[payload_index]["content"])
            schema = _claim_schema(payload)
        except (ValueError, json.JSONDecodeError):
            continue
        case_id = str(row.get("metadata", {}).get("case_id") or "")
        dataset = str(row.get("metadata", {}).get("dataset") or "")
        schemas_by_case.setdefault(case_id, copy.deepcopy(schema))
        schemas_by_dataset.setdefault(dataset, copy.deepcopy(schema))

    compiled: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        case_id = str(metadata.get("case_id") or "")
        dataset = str(metadata.get("dataset") or "")
        fallback = schemas_by_case.get(case_id) or schemas_by_dataset.get(dataset)
        try:
            compiled.append(compile_row(row, fallback_claim_schema=fallback))
        except Exception as error:
            raise ValueError(
                f"failed to compile evidence-first row {metadata.get('row_id')}: {error}"
            ) from error
    source_exposure_report, _source_exposure_rows = audit_control_plane(rows)
    exposure_report, exposure_rows = audit_control_plane(compiled)
    exposure_interpretation = _apply_exposure_weights(compiled, exposure_rows)
    curated, duplicate_exclusions = _deduplicate_reasoning(compiled)
    reasoning_audit = _reasoning_audit(curated)
    strict_core = _strict_reasoning_core(curated, reasoning_audit)
    sensitive_hits = _sensitive_key_hits(curated)
    pairs = _pair_index(curated)
    weighted_total = sum(float(row["metadata"]["training_weight"]) for row in compiled)
    weighted_control = sum(
        float(row["metadata"]["training_weight"])
        for row in compiled
        if row["metadata"]["supervision_class"] == "control_only"
    )
    report = {
        "artifact": "evidence_first_sft_compile_report_v2",
        "artifact_status": "structural_candidate_only",
        "training_qualified": False,
        "source_rows": len(rows),
        "compiled_full_rows": len(compiled),
        "deduplicated_candidate_rows": len(curated),
        "strict_causal_core_rows": len(strict_core),
        "exact_duplicate_reasoning_rows_excluded": len(duplicate_exclusions),
        "causal_reasoning_rows_passed": sum(
            bool(item["passes_causal_reasoning"]) for item in reasoning_audit
        ),
        "causal_reasoning_rows_failed": sum(
            not bool(item["passes_causal_reasoning"]) for item in reasoning_audit
        ),
        "sensitive_model_visible_key_hits": len(sensitive_hits),
        "datasets": dict(sorted(Counter(row["metadata"]["dataset"] for row in compiled).items())),
        "actions": dict(sorted(Counter(row["metadata"]["target_tool"] for row in compiled).items())),
        "supervision_classes": dict(
            sorted(Counter(row["metadata"]["supervision_class"] for row in compiled).items())
        ),
        "weighted_control_only_fraction": round(weighted_control / max(weighted_total, 1e-9), 6),
        "causal_pairs": len(pairs),
        "control_plane_exposure_before_compile": source_exposure_report["overall"],
        "control_plane_exposure_after_compile": exposure_report["overall"],
        "remaining_exact_preexposure_interpretation": exposure_interpretation,
        "semantic_target_preexposure_after_compile": exposure_interpretation.get(
            "semantic_target_preexposure", 0
        ),
        "qualification_note": (
            "Not training-qualified. The deduplicated file is a structurally repaired candidate set, not an SFT "
            "training set. Only the strict causal core passes the offline reasoning audit, and that small core is "
            "also not production-ready; closed-loop evidence-sensitivity and sealed-case evaluation remain required."
        ),
        "qualification_blockers": [
            "causal_reasoning_failures_present",
            "closed_loop_evidence_sensitivity_not_established",
            "sealed_cross_dataset_evaluation_not_available",
        ],
        "exposure_rows": exposure_rows,
        "duplicate_exclusions": duplicate_exclusions,
        "reasoning_audit": reasoning_audit,
        "sensitive_hits": sensitive_hits,
    }
    return compiled, curated, strict_core, report, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_rows = _read_jsonl(args.input)
    compiled, curated, strict_core, report, pairs = compile_dataset(source_rows)
    output_root = Path(args.output_root)
    write_jsonl(output_root / "sft_evidence_first_full_candidate.jsonl", compiled)
    write_jsonl(output_root / "sft_evidence_first_deduplicated_candidate.jsonl", curated)
    write_jsonl(
        output_root / "sft_strict_causal_core_not_production_ready.jsonl",
        strict_core,
    )
    write_jsonl(
        output_root / "candidate_sampler_index_not_for_training.jsonl",
        [
            {
                "row_id": row["metadata"]["row_id"],
                "dataset": row["metadata"]["dataset"],
                "case_id": row["metadata"]["case_id"],
                "target_tool": row["metadata"]["target_tool"],
                "supervision_class": row["metadata"]["supervision_class"],
                "sampling_weight": row["metadata"]["training_weight"],
            }
            for row in curated
        ],
    )
    write_jsonl(output_root / "causal_pair_index.jsonl", pairs)
    write_jsonl(output_root / "control_plane_exposure_rows.jsonl", report.pop("exposure_rows"))
    write_jsonl(output_root / "excluded_duplicate_reasoning_rows.jsonl", report.pop("duplicate_exclusions"))
    write_jsonl(output_root / "causal_reasoning_audit.jsonl", report.pop("reasoning_audit"))
    write_jsonl(output_root / "sensitive_key_hits.jsonl", report.pop("sensitive_hits"))
    write_json(output_root / "compile_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
