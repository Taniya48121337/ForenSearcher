"""Run a non-LLM policy that reads only Harness control-plane fields.

This is a diagnostic lower bound, not a production policy.  It never inspects
EvidenceAtom observations, video measurements, rationales, or oracle values.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from llm_teacher_rollout import _normalize_harness_params
from pipeline import ROOT, write_json
from run_student_closed_loop_canary import (
    VCSL_FIRST,
    VCSL_REPAIR_FIRST,
    VIDEO_FIRST,
    VIDEO_REPAIR_FIRST,
    VIDDIFF_FIRST,
    VIDDIFF_REPAIR_FIRST,
    _compile_visible_decision_actions,
    _materialize_visible_action_arguments,
    _result_summary,
    _schema_valid,
    _vcsl_environment,
    _vcsl_repair_environment,
    _video_environment,
    _video_repair_environment,
    _viddiff_item,
    _viddiff_repair_environment,
)
from shadow_execute_generated_actions import _jsonl, _restore_harness
from student_tool_contract import build_student_tools, constrain_tools_with_visible_invariants
from vcsl_batch import VCSL_TOOL_REGISTRY
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, VidDiffAdaptiveMeasurementEnvironment
from videosham_batch import MULTIFAMILY_TOOL_REGISTRY


DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_sft_gain_experiment_v1"
    / "decision_sufficient_state_v6_executable_affordances/validation/sft_all.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1"
    / "arms/scripted_control_baseline/diagnostic_clean.json"
)


PUBLIC_PRIORITY = (
    "inspect_media",
    "inspect_pair_features",
    "inspect_comparison_task",
    "detect_temporal_edit",
    "compare_spatial_regions",
    "localize_copy_segment_spd_official",
    "localize_copy_segment_dp",
    "localize_copy_segment_local_alignment",
    "localize_copy_segment_temporal_stability",
    "compare_action_refined_symmetric",
    "compare_uniform_timeline",
    "compare_global_motion_symmetric",
    "compare_symmetry_residual_abstention",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "create_candidate_claim",
    "bind_evidence",
    "set_field_decision",
    "stop",
)


def _affordances(state: dict[str, Any]) -> dict[str, Any]:
    value = state.get("state_action_affordances")
    return value if isinstance(value, dict) else {}


def _actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    affordances = _affordances(state)
    result: list[dict[str, Any]] = []
    for key in (
        "executable_binding_actions",
        "executable_decision_actions",
        "executable_evidence_assessment_actions",
        "executable_adjudication_actions",
    ):
        for action in affordances.get(key) or []:
            if not isinstance(action, dict):
                continue
            name = action.get("tool_name") or action.get("name")
            arguments = action.get("arguments")
            if name and isinstance(arguments, dict):
                result.append({"tool_name": str(name), "arguments": copy.deepcopy(arguments)})
    return result


def _first_legal_compiled(
    actions: list[dict[str, Any]], eligible: list[str]
) -> tuple[str, dict[str, Any]] | None:
    for action in actions:
        if action["tool_name"] in eligible:
            return action["tool_name"], copy.deepcopy(action["arguments"])
    return None


_UNMATERIALIZED = object()


def _materialize_public_schema_value(schema: dict[str, Any]) -> Any:
    """Instantiate only deterministic values exposed by a public tool schema.

    This deliberately does not consult the Harness state.  Besides const and
    singleton enums, it handles required nested objects and fixed-length arrays.
    An array whose fixed length equals its item enum is emitted once in the
    schema's public order; this is a deterministic interface convention, not an
    evidence-dependent choice.
    """

    if "const" in schema:
        return copy.deepcopy(schema["const"])
    enum = list(schema.get("enum") or [])
    if len(enum) == 1:
        return copy.deepcopy(enum[0])

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        result: dict[str, Any] = {}
        for field_name in schema.get("required") or []:
            child = _materialize_public_schema_value(properties.get(field_name) or {})
            if child is _UNMATERIALIZED:
                return _UNMATERIALIZED
            result[str(field_name)] = child
        return result

    if schema_type == "array":
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if not isinstance(minimum, int) or minimum != maximum:
            return _UNMATERIALIZED
        length = minimum
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list) and len(prefix_items) == length:
            values = [_materialize_public_schema_value(item or {}) for item in prefix_items]
            return _UNMATERIALIZED if any(value is _UNMATERIALIZED for value in values) else values
        item_schema = schema.get("items") or {}
        item_enum = list(item_schema.get("enum") or [])
        if length > 0 and len(item_enum) == length:
            return copy.deepcopy(item_enum)
        item = _materialize_public_schema_value(item_schema)
        if item is not _UNMATERIALIZED:
            return [copy.deepcopy(item) for _ in range(length)]

    return _UNMATERIALIZED


def _materialize_control_arguments(
    name: str,
    *,
    state: dict[str, Any],
    tools: list[dict[str, Any]],
    executable_actions: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    arguments = _materialize_visible_action_arguments(
        name, state=state, tools=tools, executable_actions=executable_actions
    )
    if arguments is not None:
        return arguments, "visible_control_plane"
    matching_tools = [
        tool for tool in tools if (tool.get("function") or {}).get("name") == name
    ]
    if len(matching_tools) != 1:
        return None, "unmaterialized"
    parameters = (matching_tools[0].get("function") or {}).get("parameters") or {}
    candidate = _materialize_public_schema_value(parameters)
    if (
        candidate is _UNMATERIALIZED
        or not isinstance(candidate, dict)
        or not _schema_valid(matching_tools, name, candidate)
    ):
        return None, "unmaterialized"
    return candidate, "public_schema_order"


def choose_control_action(
    state: dict[str, Any], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Choose without reading evidence_store or any observation payload."""

    eligible = [str(item) for item in state.get("eligible_tools") or []]
    actions = _actions(state)
    affordances = _affordances(state)

    for candidate in affordances.get("repair_restoration_candidates") or []:
        name = str(candidate.get("tool_name") or "")
        if name in eligible:
            arguments, materialization_rule = _materialize_control_arguments(
                name, state=state, tools=tools, executable_actions=actions
            )
            return {
                "tool_name": name,
                "arguments": arguments,
                "selection_rule": "repair_restoration_candidate",
                "materialization_rule": materialization_rule,
            }

    compiled = _first_legal_compiled(actions, eligible)
    if compiled is not None:
        name, arguments = compiled
        return {
            "tool_name": name,
            "arguments": arguments,
            "selection_rule": "first_legal_compiled_action",
            "materialization_rule": "compiled_action",
        }

    if len(eligible) == 1:
        name = eligible[0]
        arguments, materialization_rule = _materialize_control_arguments(
            name, state=state, tools=tools, executable_actions=actions
        )
        return {
            "tool_name": name,
            "arguments": arguments,
            "selection_rule": "sole_eligible_tool",
            "materialization_rule": materialization_rule,
        }

    priority = {name: index for index, name in enumerate(PUBLIC_PRIORITY)}
    name = min(eligible, key=lambda item: (priority.get(item, len(priority)), item))
    arguments, materialization_rule = _materialize_control_arguments(
        name, state=state, tools=tools, executable_actions=actions
    )
    return {
        "tool_name": name,
        "arguments": arguments,
        "selection_rule": "fixed_public_priority",
        "materialization_rule": materialization_rule,
    }


def run_episode(
    *,
    seed_row: dict[str, Any],
    environment: Any,
    tool_registry: list[dict[str, Any]],
    max_turns: int,
) -> dict[str, Any]:
    harness = _restore_harness(seed_row)
    harness.environment = environment
    harness.tool_registry = copy.deepcopy(tool_registry)
    latest: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []
    started = time.perf_counter()
    terminal_success = False
    for turn_index in range(1, max_turns + 1):
        state = harness.render_state()
        _compile_visible_decision_actions(state)
        eligible = [str(item) for item in state.get("eligible_tools") or []]
        if not eligible:
            turns.append({"turn": turn_index, "status": "no_eligible_action"})
            break
        executable_actions = _actions(state)
        tools = build_student_tools(
            harness.claim_schema,
            harness.tool_registry,
            eligible,
            executable_actions=executable_actions,
        )
        tools = constrain_tools_with_visible_invariants(tools, state)
        decision = choose_control_action(state, tools)
        name = decision["tool_name"]
        arguments = decision["arguments"]
        if arguments is None:
            turns.append(
                {
                    "turn": turn_index,
                    "status": "no_control_plane_materialization",
                    "visible_eligible_tools": eligible,
                    "selected_tool": name,
                    "selection_rule": decision["selection_rule"],
                    "materialization_rule": decision["materialization_rule"],
                }
            )
            break
        call_id = f"scripted-control-{turn_index:03d}"
        reasoning = (
            f"Diagnostic control-only policy selected {name} by "
            f"{decision['selection_rule']}."
        )
        internal_arguments = _normalize_harness_params(name, arguments)
        harness.record_policy_call(name, internal_arguments, call_id, reasoning)
        try:
            result = harness.execute_tool(name, internal_arguments, call_id)
        except (ValueError, RuntimeError) as error:
            result = {
                "status": "rejected",
                "authority": "tool_or_harness_precondition",
                "error": str(error),
            }
            harness.record_rejected_call(name, call_id, result)
        latest = _result_summary(result)
        turns.append(
            {
                "turn": turn_index,
                "status": result.get("status"),
                "visible_eligible_tools": eligible,
                "selected_tool": name,
                "selected_arguments": copy.deepcopy(arguments),
                "selection_rule": decision["selection_rule"],
                "materialization_rule": decision["materialization_rule"],
                "environment_outcome": latest,
            }
        )
        if name == "stop" and bool((result.get("stop_decision") or {}).get("allowed")):
            terminal_success = True
            break
    final_state = harness.render_state()
    return {
        "case_id": harness.contract.contract_id,
        "dataset": seed_row["metadata"]["dataset"],
        "policy": "scripted_control_baseline_v2_schema_materializer",
        "evidence_semantics_read_by_policy": False,
        "video_or_feature_content_read_by_policy": False,
        "oracle_or_verifier_read_by_policy": False,
        "policy_visible_fields": [
            "eligible_tools",
            "state_action_affordances",
            "claim_structure",
            "open_obligations",
            "tool_schemas_constrained_by_harness",
        ],
        "terminal_success": terminal_success,
        "turn_count": len(turns),
        "rejection_count": sum(item.get("status") == "rejected" for item in turns),
        "tool_sequence": [item.get("selected_tool") for item in turns if item.get("selected_tool")],
        "episode_seconds": round(time.perf_counter() - started, 3),
        "generated_tokens": 0,
        "turns": turns,
        "final_state": final_state,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = {str(item["metadata"]["row_id"]): item for item in _jsonl(args.sft)}
    clean = {
        "VideoSham": (VIDEO_FIRST, lambda: _video_environment(rows), MULTIFAMILY_TOOL_REGISTRY),
        "VCSL": (VCSL_FIRST, _vcsl_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (
            VIDDIFF_FIRST,
            lambda: VidDiffAdaptiveMeasurementEnvironment(_viddiff_item("surgery_70")),
            VIDDIFF_TOOL_REGISTRY,
        ),
    }
    repair = {
        "VideoSham": (
            VIDEO_REPAIR_FIRST,
            lambda: _video_repair_environment(rows),
            MULTIFAMILY_TOOL_REGISTRY,
        ),
        "VCSL": (VCSL_REPAIR_FIRST, _vcsl_repair_environment, VCSL_TOOL_REGISTRY),
        "VidDiffBench": (
            VIDDIFF_REPAIR_FIRST,
            _viddiff_repair_environment,
            VIDDIFF_TOOL_REGISTRY,
        ),
    }
    builders = clean if args.mode == "clean" else repair
    selected = args.dataset or list(builders)
    episodes = []
    for dataset in selected:
        row_id, environment_builder, registry = builders[dataset]
        episodes.append(
            run_episode(
                seed_row=rows[row_id],
                environment=environment_builder(),
                tool_registry=registry,
                max_turns=args.max_turns,
            )
        )
    return {
        "artifact": "scripted_control_baseline_diagnostic_v2_schema_materializer",
        "mode": args.mode,
        "summary": {
            "episodes": len(episodes),
            "terminal_successes": sum(item["terminal_success"] for item in episodes),
            "total_turns": sum(item["turn_count"] for item in episodes),
            "total_generated_tokens": 0,
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("clean", "repair"), default="clean")
    parser.add_argument("--dataset", action="append", choices=("VideoSham", "VCSL", "VidDiffBench"))
    parser.add_argument("--max-turns", type=int, default=20)
    args = parser.parse_args()
    report = run(args)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for episode in report["episodes"]:
        print(
            json.dumps(
                {
                    "dataset": episode["dataset"],
                    "terminal_success": episode["terminal_success"],
                    "turn_count": episode["turn_count"],
                    "tool_sequence": episode["tool_sequence"],
                    "last_status": episode["turns"][-1]["status"] if episode["turns"] else None,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
