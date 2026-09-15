"""Live LLM teacher rollout for the executable VideoSham harness.

The teacher chooses one tool per turn from the same public state and schemas the
student will see.  Every proposed action is executed by the real environment;
the model never authors observations or EvidenceAtoms.  API credentials are
read from the environment and are never serialized.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from interactive_harness import (
    CLAIM_SCHEMA,
    EvidenceAtom,
    INVARIANT_FIELDS,
    REQUIRED_FIELDS,
    TARGET_FIELDS,
    TOOL_REGISTRY,
    RealVideoToolEnvironment,
    TaskContract,
    VideoForensicHarness,
    _assistant_tool_call,
    _expand_turn_datums,
    _final_claim,
    _tool_message,
    evidence_interval_scope_issues,
)
from pipeline import (
    EDITED,
    ROOT,
    SOURCE,
    decode,
    media_metadata,
    monotonic_align,
    omission_segments,
    write_json,
    write_jsonl,
)
from sft_v2_sidecar import compile_transition_sidecar


class TeacherRolloutIncomplete(RuntimeError):
    """A bounded teacher episode ended without an accepted terminal action."""

    def __init__(self, message: str, *, failure_artifact: dict[str, Any]) -> None:
        super().__init__(message)
        self.failure_artifact = failure_artifact


def _incomplete_rollout_artifact(
    *,
    case_id: str,
    phase: str,
    episode: dict[str, Any],
    harness: VideoForensicHarness,
    clean_episode: dict[str, Any] | None = None,
    mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": "teacher_rollout_failure_v1",
        "case_id": case_id,
        "failure_phase": phase,
        "reason": f"{phase} LLM teacher rollout did not reach an accepted stop",
        "turns_executed": len(episode.get("turn_history", [])),
        "max_turns": episode.get("max_turns"),
        "eligible_tools_at_failure": [
            item["name"] for item in harness.eligible_tool_registry()
        ],
        "failed_episode": copy.deepcopy(episode),
        "excluded_from_sft": True,
    }
    if clean_episode is not None:
        artifact["clean_episode"] = copy.deepcopy(clean_episode)
    if mutation is not None:
        artifact["student_visible_mutation"] = copy.deepcopy(mutation)
    return artifact


REASONING_PROPERTY = {
    "type": "string",
    "description": (
        "Natural-language analysis before acting. Explain what the latest visible observation or state established, "
        "which evidence or decision gap remains, and why this tool and these parameters are the best next action. "
        "Do not mention hidden labels, oracle data, or facts not present in the visible state."
    ),
}


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def teacher_tool_schemas(
    claim_schema: dict[str, Any] = CLAIM_SCHEMA,
    tool_registry: list[dict[str, Any]] = TOOL_REGISTRY,
) -> list[dict[str, Any]]:
    interval = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
    }
    omitted_intervals = {
        "type": "array",
        "items": interval,
    }
    field_properties: dict[str, Any] = {}
    for field_name, spec in claim_schema["fields"].items():
        if "enum" in spec:
            field_schema = {"type": "string", "enum": spec["enum"]}
        elif field_name == "omitted_source_intervals_s":
            field_schema = omitted_intervals
        elif field_name.endswith("_interval_s") or field_name == "comparison_scope_s":
            field_schema = interval
        elif any(name in spec["types"] for name in ("float", "int")):
            field_schema = {"type": "number"}
        elif "list" in spec["types"]:
            field_schema = {"type": "array", "items": {}}
        else:
            field_schema = {"type": "string"}
        field_schema = {**field_schema, "description": spec.get("definition", field_name)}
        field_properties[field_name] = field_schema
    required_fields = list(claim_schema["required_fields"])
    field_values = _object_schema(field_properties, required_fields)
    binding_item = _object_schema(
        {
            "field": {"type": "string", "enum": required_fields},
            "support_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        ["field", "support_ids"],
    )
    pair_compare_parameters = _object_schema(
        {
            "reasoning": REASONING_PROPERTY,
            "source_ref": {"type": "string", "enum": ["source"]},
            "suspect_ref": {"type": "string", "enum": ["suspect"]},
            "comparison_scope_s": interval,
        },
        ["reasoning", "source_ref", "suspect_ref", "comparison_scope_s"],
    )
    schemas = [
        {
            "type": "function",
            "name": "inspect_media",
            "description": "Decode metadata for the visible source and suspect media and read the neutral scope.",
            "parameters": _object_schema(
                {
                    "reasoning": REASONING_PROPERTY,
                    "media_refs": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["source", "suspect"]},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                ["reasoning", "media_refs"],
            ),
            "strict": True,
        },
        {
            "type": "function",
            "name": "align_videos_dense",
            "description": "Execute dense monotonic visual alignment on the real source-suspect video pair.",
            "parameters": pair_compare_parameters,
            "strict": True,
        },
        {
            "type": "function",
            "name": "detect_temporal_edit",
            "description": "Measure frame-order discontinuities, duplicated-frame reuse, and local playback-rate shifts between the real pair.",
            "parameters": pair_compare_parameters,
            "strict": True,
        },
        {
            "type": "function",
            "name": "compare_audio_tracks",
            "description": "Compare decoded audio spectral signatures and temporal-alignment residuals inside the visible scope.",
            "parameters": pair_compare_parameters,
            "strict": True,
        },
        {
            "type": "function",
            "name": "compare_spatial_regions",
            "description": "Measure localized visual residuals under temporal correspondence and region-appearance disagreement in the real pair.",
            "parameters": pair_compare_parameters,
            "strict": True,
        },
        {
            "type": "function",
            "name": "create_candidate_claim",
            "description": (
                "Propose one atomic claim from visible tool evidence. This creates public field obligations but does "
                "not support or commit any field."
            ),
            "parameters": _object_schema(
                {
                    "reasoning": REASONING_PROPERTY,
                    "claim_id": {"type": "string"},
                    "field_values": field_values,
                },
                ["reasoning", "claim_id", "field_values"],
            ),
            "strict": True,
        },
        {
            "type": "function",
            "name": "bind_evidence",
            "description": (
                "Propose evidence-to-field support links. Bind only active EvidenceAtom IDs whose visible kind and "
                "supports_fields are compatible. A repair may bind only a subset of fields. HARD PRECONDITION: if "
                "the visible action_constraints.must_reopen_before_rebinding lists a field, first call "
                "set_field_decision with decision_status=open for that field; otherwise this bind is rejected."
            ),
            "parameters": _object_schema(
                {
                    "reasoning": REASONING_PROPERTY,
                    "claim_id": {"type": "string"},
                    "binding_items": {"type": "array", "items": binding_item, "minItems": 1},
                },
                ["reasoning", "claim_id", "binding_items"],
            ),
            "strict": True,
        },
        {
            "type": "function",
            "name": "set_field_decision",
            "description": (
                "Set policy decision status for selected fields. Commit only visibly supported fields. After evidence "
                "loss, reopen only fields whose support was invalidated."
            ),
            "parameters": _object_schema(
                {
                    "reasoning": REASONING_PROPERTY,
                    "claim_id": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": required_fields},
                        "minItems": 1,
                    },
                    "decision_status": {"type": "string", "enum": ["open", "committed", "abstained"]},
                },
                ["reasoning", "claim_id", "fields", "decision_status"],
            ),
            "strict": True,
        },
        {
            "type": "function",
            "name": "revise_claim_fields",
            "description": (
                "Revise only claim fields that are already open and whose proposed new "
                "values are directly supported by active student-visible evidence. "
                "This changes claim values but does not bind evidence or commit fields."
            ),
            "parameters": _object_schema(
                {
                    "reasoning": REASONING_PROPERTY,
                    "claim_id": {"type": "string"},
                    "field_updates": {
                        "type": "array",
                        "minItems": 1,
                        "items": _object_schema(
                            {
                                "field": {
                                    "type": "string",
                                    "enum": required_fields,
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "array", "items": {}},
                                    ]
                                },
                            },
                            ["field", "value"],
                        ),
                    },
                },
                ["reasoning", "claim_id", "field_updates"],
            ),
            "strict": True,
        },
        {
            "type": "function",
            "name": "stop",
            "description": (
                "Request terminal validation. Use only after all visible high-priority goals and field obligations are "
                "closed; a rejected stop does not end the episode."
            ),
            "parameters": _object_schema({"reasoning": REASONING_PROPERTY}, ["reasoning"]),
            "strict": True,
        },
    ]
    by_name = {schema["name"]: schema for schema in schemas}
    for registry_entry in tool_registry:
        custom_schema = registry_entry.get("teacher_schema")
        if custom_schema is not None:
            by_name[registry_entry["name"]] = copy.deepcopy(custom_schema)
    enabled = [item["name"] for item in tool_registry]
    missing = [name for name in enabled if name not in by_name]
    if missing:
        raise ValueError(f"teacher schemas are missing for registered tools: {missing}")
    return [by_name[name] for name in enabled]


@dataclass
class TeacherAction:
    response_id: str
    call_id: str
    tool_name: str
    reasoning: str
    params: dict[str, Any]
    raw_arguments: str
    usage: dict[str, Any]
    replay_items: list[dict[str, Any]]


class OpenAIResponsesTeacher:
    """Small standard-library Responses API client with no credential logging."""

    source = "openai_responses_api"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.4",
        base_url: str = "https://api.openai.com/v1",
        reasoning_effort: str = "medium",
        timeout_s: int = 180,
        max_retries: int = 3,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY is empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def next_action(self, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> TeacherAction:
        payload = {
            "model": self.model,
            "input": input_items,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "reasoning": {"effort": self.reasoning_effort},
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 2048,
            "store": False,
        }
        body: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                retryable = error.code in {408, 409, 429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    retry_after = error.headers.get("Retry-After") if error.headers else None
                    delay = min(float(retry_after), 20.0) if retry_after and retry_after.isdigit() else 2.0**attempt
                    time.sleep(delay)
                    continue
                detail = error.read().decode("utf-8", errors="replace")[:2000]
                raise RuntimeError(f"OpenAI Responses API returned HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt < self.max_retries:
                    time.sleep(2.0**attempt)
                    continue
                reason = getattr(error, "reason", str(error))
                raise RuntimeError(f"OpenAI Responses API connection failed: {reason}") from error
        if body is None:
            raise RuntimeError("OpenAI Responses API returned no response")

        calls = [item for item in body.get("output", []) if item.get("type") == "function_call"]
        if len(calls) != 1:
            types = [item.get("type") for item in body.get("output", [])]
            raise RuntimeError(f"teacher must emit exactly one function_call; got {len(calls)} output_types={types}")
        call = calls[0]
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, json.JSONDecodeError) as error:
            raise RuntimeError("teacher returned invalid function arguments") from error
        reasoning = arguments.pop("reasoning", "")
        if not _substantive_reasoning(reasoning):
            raise RuntimeError("teacher reasoning is missing or too short")
        return TeacherAction(
            response_id=body.get("id", ""),
            call_id=call["call_id"],
            tool_name=call["name"],
            reasoning=reasoning,
            params=arguments,
            raw_arguments=call["arguments"],
            usage=body.get("usage", {}),
            # Reasoning models require their returned reasoning items to be
            # replayed alongside the function call on the next request.  With
            # store=False, encrypted_content keeps that state client-side.
            replay_items=copy.deepcopy(body.get("output", [])),
        )


def _teacher_system_prompt(mode: str) -> str:
    mode_instruction = (
        "Build one supported atomic claim from the paired evidence assets."
        if mode == "clean"
        else (
            "Repair the restored mutated state. Infer affected fields from visible support status; preserve every "
            "independently supported field. Distinguish the mutation operators exactly: evidence_dropout removes an "
            "EvidenceAtom and forbids reusing its ID; binding_invalidation removes only claim-to-evidence links while "
            "the same active atom remains valid and is intended to be rebound; decision_reopen preserves both evidence "
            "and bindings and requires only recommitting the reopened decisions."
        )
    )
    return (
        "You are the semantic policy inside a state-externalizing video-evidence harness. "
        f"{mode_instruction} Every turn you must call exactly one available function. Put grounded natural-language "
        "analysis in the required reasoning argument. Use 2-4 compact sentences, normally 45-120 words: cite at least "
        "one concrete fact or EvidenceAtom from the latest visible observation (or explicitly state that none exists on "
        "the first turn), identify the single decision/evidence gap that blocks progress, and name why the chosen tool "
        "and parameters close that gap. Do not recite the whole harness state, repeat a fixed template, or discuss actions "
        "that are not decision-relevant. You choose tools, claim values, bindings, decisions, "
        "and when to stop. The environment alone creates observations and EvidenceAtoms; never invent either. The "
        "harness validates state edits and may reject them. A rejected action is an observation to diagnose and recover "
        "from. Do not assume hidden dataset labels or oracle values. Do not clear unaffected fields during repair. "
        "When visible evidence is inconsistent, incomplete, stale, or belongs to another task, use "
        "challenge_evidence to name the suspect atom and request a generic fresh check; do not create or bind a "
        "claim merely because those bookkeeping actions are also eligible. "
        "For evidence assessment or adjudication, target_fields must exactly match the current visible "
        "evidence_adjudication.requirement.target_fields; do not add other fields supported by the same atom. "
        "Use evidence defect codes literally: claim_value_conflict only when the candidate cannot support the proposed "
        "field value; order_inconsistent only when observation.order_consistent is explicitly false or a visible "
        "interval/box has its start after its end; "
        "role_assignment_mismatch only when provenance marks the current role assignment invalid; "
        "abstention_conflicts_with_claim only for an explicit abstention against a non-abstaining claim; and "
        "redundant_same_conclusion or lower_confidence only relative to a named selected competitor. Do not infer one "
        "defect code from another. "
        "When multiple active measurement atoms disagree, never choose the newest atom merely because it is newest, "
        "and never treat a count majority or raw confidence alone as adjudication. Before creating a claim, compare "
        "each method's fit to the exact statement, order-consistency, direct visual counterexamples, confidence, and "
        "whether preferred_video=c means a genuine equal/insufficient-evidence abstention. Choose claim values from "
        "the single active atom with the strongest visible justification, and explicitly explain why conflicting "
        "atoms are weaker. If the selected atom reports confidence below 0.65, order inconsistency, disagreement, or "
        "an abstaining c result, the create-or-bind reasoning must explicitly name the remaining uncertainty, fit to "
        "the exact statement, a visible cue or counterexample, and why the observed decision rule still warrants the "
        "chosen direction or abstention; confidence alone is never enough. State the resolution as a causal sentence "
        "of the form 'I select/abstain because ... despite ...', filled with concrete observation content rather than "
        "those placeholder words. If no active atom justifies a conclusion, "
        "continue measuring or abstain; do not fabricate."
    )


def _substantive_reasoning(value: Any) -> bool:
    """Language-agnostic minimum: enough visible text to state evidence, gap, and action rationale."""
    return isinstance(value, str) and len(value.strip()) >= 48


def _api_initial_input(
    *,
    system_prompt: str,
    task_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(task_payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def _normalize_harness_params(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(params)
    if tool_name == "bind_evidence":
        items = normalized.pop("binding_items", [])
        bindings: dict[str, list[str]] = {}
        for item in items:
            field_name = item.get("field")
            if field_name in bindings:
                raise ValueError(f"duplicate binding item for field: {field_name}")
            bindings[field_name] = item.get("support_ids", [])
        normalized["bindings"] = bindings
    return normalized


def _api_safe_tool_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _accepted_turn_datums(
    messages: list[dict[str, Any]],
    turn_history: list[dict[str, Any]],
    *,
    case_id: str,
    trajectory_id: str,
    trajectory_type: str,
    turn_audits: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Keep recoveries as context, but never train a rejected proposal as the positive target."""
    expanded = _expand_turn_datums(
        messages,
        case_id=case_id,
        trajectory_id=trajectory_id,
        trajectory_type=trajectory_type,
    )
    if len(expanded) != len(turn_history):
        raise RuntimeError("assistant action count does not match teacher turn history")
    audits = turn_audits or []
    if audits and len(audits) != len(turn_history):
        raise RuntimeError("teacher audit count does not match teacher turn history")
    kept: list[dict[str, Any]] = []
    excluded: list[int] = []
    for index, (datum, turn) in enumerate(zip(expanded, turn_history)):
        status = turn["observation"].get("status")
        if status not in {"accepted", "executed"}:
            excluded.append(turn["turn_idx"])
            continue
        datum["metadata"]["target_execution_status"] = status
        datum["metadata"]["source_turn_index"] = turn["turn_idx"]
        audit = audits[index] if audits else None
        choice = _policy_choice_turn_metadata(audit)
        datum["metadata"].update(choice)
        kept.append(datum)
    return kept, excluded


def _policy_choice_turn_metadata(
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project search evidence onto one SFT row without copying branch results."""

    if not audit:
        return {
            "policy_choice_audit_status": "missing",
            "policy_choice_training_eligible": False,
        }
    exposure = audit.get("control_plane_exposure") or {}
    selected = audit.get("selected_action") or {}
    selected_name = str(selected.get("tool_name") or selected.get("name") or "")
    semantic_names = set(exposure.get("semantic_measurement_tool_names") or [])
    semantic_turn = selected_name in semantic_names
    choice_count = int(exposure.get("semantic_measurement_choice_count", 0))
    search = audit.get("evidence_choice_search") or {}
    supervision = copy.deepcopy(audit.get("supervision") or {})
    if not semantic_turn:
        eligible = True
        status = "not_applicable_state_or_context_action"
    elif choice_count < 2:
        eligible = False
        status = "singleton_semantic_choice"
    else:
        eligible = bool(
            search.get("applicable") is True
            and search.get("proposal_coverage_passed") is True
            and search.get("selection_ambiguous") is False
            and search.get("selection_frozen_before_branch_execution") is True
            and search.get("hindsight_outcome_used_for_selection") is False
            and search.get("selected_branch_execution_admissible") is True
            and float(search.get("selected_preaction_method_fit_score", 0.0)) >= 1.0
            and search.get("selected_counterfactual_tool_swap_passed") is True
            and search.get("committed_replay_checked") is True
            and search.get("deterministic_replay_match") is True
        )
        status = "passed" if eligible else "failed_evidence_choice_search"
    return {
        "policy_choice_audit_status": status,
        "policy_choice_training_eligible": eligible,
        "semantic_measurement_turn": semantic_turn,
        "visible_semantic_choice_count": choice_count,
        "latest_observation_uptake_expected": bool(
            audit.get("latest_observation_uptake_expected")
        ),
        "latest_observation_uptake_passed": bool(
            audit.get("latest_observation_uptake_passed")
        ),
        "supervision": supervision,
        "supervision_class": supervision.get("class"),
        "base_training_weight": supervision.get("training_weight"),
        "evidence_choice_search": {
            key: copy.deepcopy(search.get(key))
            for key in (
                "applicable",
                "visible_choice_count",
                "proposed_choice_count",
                "proposal_coverage_passed",
                "selection_ambiguous",
                "top_rank_tie_count",
                "selected_utility_margin",
                "selected_preaction_method_fit_score",
                "selected_preaction_method_fit_ranking_score",
                "selected_counterfactual_tool_swap_passed",
                "selected_directional_contrast",
                "selection_frozen_before_branch_execution",
                "hindsight_outcome_used_for_selection",
                "selected_branch_execution_admissible",
                "higher_utility_branch_was_not_allowed_to_reroute_selection",
                "proposal_rounds_attempted",
                "proposal_retry_outcome_feedback_used",
                "selected_tool",
                "committed_replay_checked",
                "deterministic_replay_match",
            )
            if key in search
        },
    }


def _usage_totals(episodes: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for episode in episodes:
        for item in episode["api_trace"]:
            usage = item.get("usage", {})
            for key in totals:
                value = usage.get(key, 0)
                if isinstance(value, int):
                    totals[key] += value
    return totals


def _replay_trajectory_path(output_root: Path, phase: str) -> Path:
    """Use the descriptive filename unless it would cross Windows MAX_PATH."""

    long_name = f"llm_teacher_sft_{phase}_full_trajectory.jsonl"
    candidate = output_root / long_name
    # Keep margin for Windows runtimes that do not opt into long-path support.
    if len(str(candidate.resolve())) >= 240:
        return output_root / f"llm_teacher_{phase}_full.jsonl"
    return candidate


def run_teacher_episode(
    *,
    harness: VideoForensicHarness,
    teacher: Any,
    mode: str,
    task_payload: dict[str, Any],
    max_turns: int,
) -> dict[str, Any]:
    bind_harness = getattr(teacher, "bind_harness", None)
    if callable(bind_harness):
        bind_harness(harness)
    system_prompt = _teacher_system_prompt(mode)
    api_input = _api_initial_input(system_prompt=system_prompt, task_payload=task_payload)
    messages: list[dict[str, Any]] = copy.deepcopy(api_input)
    turn_history: list[dict[str, Any]] = []
    api_trace: list[dict[str, Any]] = []
    for turn_index in range(1, max_turns + 1):
        eligible_registry = harness.eligible_tool_registry()
        schemas = teacher_tool_schemas(harness.claim_schema, eligible_registry)
        action = teacher.next_action(copy.deepcopy(api_input), schemas)
        if action.tool_name not in {item["name"] for item in eligible_registry}:
            raise RuntimeError(f"teacher selected unknown tool: {action.tool_name}")
        harness_params = _normalize_harness_params(action.tool_name, action.params)
        messages.append(_assistant_tool_call(action.call_id, action.tool_name, harness_params, action.reasoning))
        harness.record_policy_call(action.tool_name, harness_params, action.call_id, action.reasoning)
        try:
            result = harness.execute_tool(action.tool_name, harness_params, action.call_id)
        except (ValueError, RuntimeError) as error:
            result = {
                "status": "rejected",
                "authority": "tool_or_harness_precondition",
                "error_type": type(error).__name__,
                "error": str(error),
                "harness_state": harness.render_state(),
            }
            harness.record_rejected_call(action.tool_name, action.call_id, result)
        committed_hook = getattr(teacher, "record_committed_result", None)
        if callable(committed_hook):
            committed_hook(action, copy.deepcopy(result))
        messages.append(_tool_message(action.call_id, action.tool_name, result))
        turn_history.append(
            {
                "turn_idx": turn_index,
                "reasoning": action.reasoning,
                "tool_name": action.tool_name,
                "params": copy.deepcopy(harness_params),
                "observation": copy.deepcopy(result),
                "teacher_response_id": action.response_id,
            }
        )
        api_trace.append(
            {
                "turn_idx": turn_index,
                "response_id": action.response_id,
                "call_id": action.call_id,
                "model": getattr(teacher, "model", "fake_teacher"),
                "usage": copy.deepcopy(action.usage),
            }
        )
        api_input.extend(copy.deepcopy(action.replay_items))
        api_input.append(
            {
                "type": "function_call_output",
                "call_id": action.call_id,
                "output": _api_safe_tool_result(result),
            }
        )
        if action.tool_name == "stop" and result.get("stop_decision", {}).get("allowed"):
            break

    return {
        "mode": mode,
        "model": getattr(teacher, "model", "fake_teacher"),
        "max_turns": max_turns,
        "terminated": harness.terminal,
        "messages": messages,
        "turn_history": turn_history,
        "api_trace": api_trace,
        "events": copy.deepcopy(harness.events),
        "final_state": harness.render_state(),
        "final_claim": _final_claim(harness, next(iter(harness.claims))) if len(harness.claims) == 1 else None,
    }


def _select_dropout_evidence(harness: VideoForensicHarness) -> str:
    if len(harness.claims) != 1:
        raise RuntimeError("clean teacher must produce exactly one claim before mutation")
    claim_id = next(iter(harness.claims))
    candidates: list[tuple[int, str]] = []
    for evidence_id, atom in harness.evidence_store.items():
        if not atom.active:
            continue
        supported = {
            field_name
            for field_name, binding in harness.bindings[claim_id].items()
            if evidence_id in binding.support_ids
        }
        candidates.append((len(supported.intersection(harness.target_fields)), evidence_id))
    if not candidates:
        raise RuntimeError("no active gap evidence is bound to the clean claim")
    count, evidence_id = max(candidates)
    if count != len(harness.target_fields):
        raise RuntimeError(
            f"no single gap atom supports the complete mutation slice; best={evidence_id} fields={count}"
        )
    return evidence_id


def _contract_and_environment(
    *,
    manifest: dict[str, Any],
    source_frames: Any,
    suspect_frames: Any,
    source_path: Path = SOURCE,
    suspect_path: Path = EDITED,
    claim_schema: dict[str, Any] = CLAIM_SCHEMA,
    tool_registry: list[dict[str, Any]] = TOOL_REGISTRY,
    external_environment: Any | None = None,
    target_fields: list[str] | None = None,
    invariant_fields: list[str] | None = None,
) -> tuple[TaskContract, RealVideoToolEnvironment]:
    case_id = manifest["case_id"]
    # The policy teacher needs stable asset identities, not host filesystem
    # locators. Media paths remain local to the executable environment passed
    # below; the task contract exposes only the manifest's logical refs.
    source_ref = manifest.get("source_media", {}).get("ref", "source")
    suspect_ref = manifest.get("edited_media", {}).get("ref", "suspect")
    contract = TaskContract(
        contract_id=f"TASK-{case_id.upper()}",
        episode_mode="paired_compare",
        task=manifest.get("task", "Determine whether and how the suspect clip omits source content."),
        visible_assets={
            "source": source_ref,
            "suspect": suspect_ref,
            "comparison_scope_s": manifest["comparison_scope_s"],
        },
        claim_schema_id=claim_schema["schema_id"],
    )
    environment = external_environment or RealVideoToolEnvironment(
        manifest=manifest,
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        source_path=Path(source_path),
        suspect_path=Path(suspect_path),
        metadata_fn=media_metadata,
        align_fn=monotonic_align,
        omission_fn=omission_segments,
    )
    return contract, environment


def build_llm_teacher_bundle(
    *,
    manifest: dict[str, Any],
    source_frames: Any,
    suspect_frames: Any,
    teacher: Any,
    source_path: Path = SOURCE,
    suspect_path: Path = EDITED,
    claim_schema: dict[str, Any] = CLAIM_SCHEMA,
    tool_registry: list[dict[str, Any]] = TOOL_REGISTRY,
    external_environment: Any | None = None,
    target_fields: list[str] | None = None,
    invariant_fields: list[str] | None = None,
    max_clean_turns: int = 12,
    max_repair_turns: int = 10,
    repair_operator: str = "evidence_dropout",
) -> dict[str, Any]:
    causal_audit_start = len(getattr(teacher, "audit_log", []))
    contract, environment = _contract_and_environment(
        manifest=manifest,
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        source_path=source_path,
        suspect_path=suspect_path,
        claim_schema=claim_schema,
        tool_registry=tool_registry,
        external_environment=external_environment,
        target_fields=target_fields,
        invariant_fields=invariant_fields,
    )
    case_id = manifest["case_id"]
    case_token = case_id.replace("videosham-", "").upper()
    clean_harness = VideoForensicHarness(
        contract,
        environment,
        claim_schema=claim_schema,
        tool_registry=tool_registry,
        target_fields=target_fields,
        invariant_fields=invariant_fields,
    )
    clean_payload = {
        "task_contract": asdict(contract),
        "claim_schema": claim_schema,
        "tool_registry": tool_registry,
        "harness_state": clean_harness.render_state(),
    }
    clean = run_teacher_episode(
        harness=clean_harness,
        teacher=teacher,
        mode="clean",
        task_payload=clean_payload,
        max_turns=max_clean_turns,
    )
    clean_audit_end = len(getattr(teacher, "audit_log", []))
    if not clean_harness.terminal:
        message = "clean LLM teacher rollout did not reach an accepted stop"
        raise TeacherRolloutIncomplete(
            message,
            failure_artifact=_incomplete_rollout_artifact(
                case_id=case_id,
                phase="clean",
                episode=clean,
                harness=clean_harness,
            ),
        )

    clean_snapshot = clean_harness.render_state()
    invariant_before = {
        name: copy.deepcopy(clean_snapshot["candidate_claims"][0]["fields"][name])
        for name in clean_harness.invariant_fields
    }
    repair_harness = clean_harness.clone_state()
    # Select from the state that will actually receive the mutation. Dataset
    # adapters can carry stateful external environments, so choosing from the
    # pre-clone harness can leave a repair operator holding an ID that is not
    # available in the restored repair state.
    mutation_evidence_id = _select_dropout_evidence(repair_harness)
    if repair_operator == "evidence_dropout":
        mutation = repair_harness.inject_evidence_dropout(mutation_evidence_id)
        visible_mutation = {
            "operator": mutation["operator"],
            "removed_evidence_ids": mutation["removed_evidence_ids"],
            "media_changed": mutation["media_changed"],
            "claim_values_changed": mutation["claim_values_changed"],
        }
    elif repair_operator == "binding_invalidation":
        mutation = repair_harness.inject_binding_invalidation(mutation_evidence_id)
        visible_mutation = {
            "operator": mutation["operator"],
            "invalidated_binding_evidence_ids": mutation["invalidated_binding_evidence_ids"],
            "evidence_atoms_remain_active": True,
            "evidence_observations_remain_valid": True,
            "same_evidence_rebinding_allowed": True,
            "only_claim_evidence_edges_removed": True,
            "media_changed": mutation["media_changed"],
            "claim_values_changed": mutation["claim_values_changed"],
        }
    elif repair_operator == "decision_reopen":
        mutation = repair_harness.inject_decision_reopen()
        visible_mutation = {
            "operator": mutation["operator"],
            "reopened_decision_count": mutation["reopened_decision_count"],
            "media_changed": mutation["media_changed"],
            "claim_values_changed": mutation["claim_values_changed"],
        }
    else:
        raise ValueError(f"unsupported repair operator: {repair_operator}")
    repair_payload = {
        "task": "Repair the restored state after an evidence-store mutation without changing independent fields.",
        "shared_clean_snapshot_id": f"SNAPSHOT-{case_token}-LLM-CLEAN-FINAL",
        "mutation": visible_mutation,
        "claim_schema": claim_schema,
        "tool_registry": tool_registry,
        "harness_state": repair_harness.render_state(),
    }
    repair = run_teacher_episode(
        harness=repair_harness,
        teacher=teacher,
        mode="repair",
        task_payload=repair_payload,
        max_turns=max_repair_turns,
    )
    repair_audit_end = len(getattr(teacher, "audit_log", []))
    repair["mutation"] = {key: value for key, value in mutation.items() if key != "harness_state"}
    repair["student_visible_mutation"] = visible_mutation
    repair["shared_clean_snapshot_id"] = f"SNAPSHOT-{case_token}-LLM-CLEAN-FINAL"
    if not repair_harness.terminal:
        message = "repair LLM teacher rollout did not reach an accepted stop"
        raise TeacherRolloutIncomplete(
            message,
            failure_artifact=_incomplete_rollout_artifact(
                case_id=case_id,
                phase="repair",
                episode=repair,
                harness=repair_harness,
                clean_episode=clean,
                mutation=visible_mutation,
            ),
        )
    repair["invariant_fields_before"] = invariant_before
    repair["invariant_fields_after"] = {
        name: copy.deepcopy(repair["final_state"]["candidate_claims"][0]["fields"][name])
        for name in clean_harness.invariant_fields
    }
    clean_id = f"LLM-TRJ-{case_token}-CLEAN-001"
    repair_id = f"LLM-TRJ-{case_token}-REPAIR-001"
    repair_trajectory_type = f"llm_teacher_{repair_operator}_repair"
    clean["trajectory_type"] = "llm_teacher_clean_acquisition"
    repair["trajectory_type"] = repair_trajectory_type
    clean_turn_sft, clean_excluded = _accepted_turn_datums(
        clean["messages"],
        clean["turn_history"],
        case_id=manifest["case_id"],
        trajectory_id=clean_id,
        trajectory_type="llm_teacher_clean_acquisition",
        turn_audits=copy.deepcopy(
            getattr(teacher, "audit_log", [])[causal_audit_start:clean_audit_end]
        ),
    )
    repair_turn_sft, repair_excluded = _accepted_turn_datums(
        repair["messages"],
        repair["turn_history"],
        case_id=manifest["case_id"],
        trajectory_id=repair_id,
        trajectory_type=repair_trajectory_type,
        turn_audits=copy.deepcopy(
            getattr(teacher, "audit_log", [])[clean_audit_end:repair_audit_end]
        ),
    )
    return {
        "case_id": case_id,
        "harness_config": {
            "claim_schema": copy.deepcopy(claim_schema),
            "tool_registry": copy.deepcopy(tool_registry),
            "target_fields": list(clean_harness.target_fields),
            "invariant_fields": list(clean_harness.invariant_fields),
        },
        "clean_episode": {"trajectory_id": clean_id, **clean},
        "repair_episode": {"trajectory_id": repair_id, **repair},
        "clean_turn_sft": clean_turn_sft,
        "repair_turn_sft": repair_turn_sft,
        "execution_summary": {
            "teacher_model": getattr(teacher, "model", "fake_teacher"),
            "teacher_source": getattr(
                teacher,
                "source",
                "openai_responses_api" if isinstance(teacher, OpenAIResponsesTeacher) else "test_double",
            ),
            "policy_view": getattr(teacher, "policy_view", "legacy_full_control"),
            "causal_generation_audit": copy.deepcopy(
                getattr(teacher, "audit_log", [])[causal_audit_start:]
            ),
            "policy_choice_training_eligible_turns": sum(
                row["metadata"].get("policy_choice_training_eligible") is True
                for row in clean_turn_sft + repair_turn_sft
            ),
            "policy_choice_training_ineligible_turns": sum(
                row["metadata"].get("policy_choice_training_eligible") is not True
                for row in clean_turn_sft + repair_turn_sft
            ),
            "external_tool_call_counts": copy.deepcopy(environment.call_counts),
            "api_usage": _usage_totals([clean, repair]),
            "excluded_rejected_sft_turns": {"clean": clean_excluded, "repair": repair_excluded},
            "repair_operator": repair_operator,
            "mutation_evidence_id": mutation_evidence_id,
            "dropped_evidence_id": mutation_evidence_id if repair_operator == "evidence_dropout" else None,
            "difference_tool_name": next(
                item["name"]
                for item in tool_registry
                if item.get("domain") == "external_media"
                and item.get("satisfies_goal", "G-CONTEXT" if item["name"] == "inspect_media" else "G-DIFFERENCE")
                == "G-DIFFERENCE"
                and environment.call_counts.get(item["name"], 0) > 0
            ),
            "repair_active_gap_ids": [
                evidence_id
                for evidence_id, atom in repair_harness.evidence_store.items()
                if atom.active and set(atom.supports_fields).intersection(repair_harness.target_fields)
            ],
        },
    }


def validate_llm_teacher_bundle(
    bundle: dict[str, Any],
    *,
    oracle: dict[str, Any],
) -> dict[str, Any]:
    clean = bundle["clean_episode"]
    repair = bundle["repair_episode"]
    summary = bundle["execution_summary"]
    student_payload = json.dumps(
        {"clean": bundle["clean_turn_sft"], "repair": bundle["repair_turn_sft"]}, ensure_ascii=False
    )
    repair_user = json.loads(repair["messages"][1]["content"])
    final_claim = repair["final_claim"] or {}
    target_states = repair["final_state"]["candidate_claims"][0]["fields"]
    target_fields = bundle.get("harness_config", {}).get("target_fields", TARGET_FIELDS)
    api_turns = clean["api_trace"] + repair["api_trace"]
    difference_atoms = [
        atom
        for episode in (clean, repair)
        for event in episode["events"]
        if event["action_type"] == "external_tool_execution"
        for atom in event["result"]["evidence_atoms"]
        if set(atom["supports_fields"]).intersection(target_fields)
    ]
    final_state = repair["final_state"]
    active_atoms = final_state["evidence_store"]
    comparison_scope = (
        final_state.get("task_contract", {}).get("visible_assets", {}).get("comparison_scope_s")
    )
    interval_scope_issues = evidence_interval_scope_issues(active_atoms, comparison_scope)
    semantic_bindings_valid = True
    for field_name, field_state in target_states.items():
        for evidence_id in field_state["support_ids"]:
            atom_payload = active_atoms.get(evidence_id)
            if atom_payload is None:
                semantic_bindings_valid = False
                continue
            atom = EvidenceAtom(**atom_payload)
            semantic_bindings_valid = semantic_bindings_valid and VideoForensicHarness._atom_supports_claim_value(
                field_name, field_state["value"], atom
            )
    checks = {
        "clean_terminated": clean["terminated"],
        "repair_terminated": repair["terminated"],
        "live_llm_response_per_turn": bool(api_turns)
        and (
            (
                summary["teacher_source"] == "openai_responses_api"
                and all(
                    item["response_id"].startswith("resp_")
                    and item["call_id"].startswith("call_")
                    and item["model"] != "fake_teacher"
                    for item in api_turns
                )
            )
            or (
                summary["teacher_source"] == "codex_cli"
                and all(
                    item["response_id"].startswith("cli_resp_")
                    and item["call_id"].startswith("cli_call_")
                    and item["model"] != "fake_teacher"
                    for item in api_turns
                )
            )
        ),
        "real_alignment_reexecuted": summary.get("repair_operator") != "evidence_dropout"
        or summary["external_tool_call_counts"].get(
            summary.get("difference_tool_name", "align_videos_dense"), 0
        )
        >= 2,
        "target_fields_repaired": all(
            target_states[name]["evidence_status"] == "supported"
            and target_states[name]["decision_status"] == "committed"
            for name in target_fields
        ),
        "off_target_invariance": repair["invariant_fields_before"] == repair["invariant_fields_after"],
        "dropped_evidence_not_reused": summary.get("repair_operator") != "evidence_dropout"
        or summary["dropped_evidence_id"] not in repair["final_state"]["available_evidence_ids"],
        "oracle_label_match_post_execution": final_claim.get("manipulation_type") == oracle["manipulation_type"],
        "oracle_window_match_post_execution": final_claim.get("comparison_scope_s") == oracle["edit_window_s"],
        "natural_language_reasoning": all(
            _substantive_reasoning(turn.get("reasoning"))
            for turn in clean["turn_history"] + repair["turn_history"]
        ),
        "tool_measurements_do_not_emit_claim_answers": all(
            not {"claim_family", "manipulation_type", "direction"}.intersection(atom["observation"])
            for atom in difference_atoms
        ),
        "final_field_values_match_bound_evidence": semantic_bindings_valid,
        "evidence_intervals_within_visible_scope": not interval_scope_issues,
        "sft_targets_are_accepted_actions": all(
            datum["metadata"].get("target_execution_status") in {"accepted", "executed"}
            for datum in bundle["clean_turn_sft"] + bundle["repair_turn_sft"]
        ),
        "step_utility_counterfactual_audit": bundle.get("step_utility_report", {}).get("passed") is True,
        "no_oracle_leak": oracle["annotation_text"] not in student_payload
        and "claim_oracle.json" not in student_payload,
        "no_target_hint_leak": "affected_fields" not in repair_user["mutation"]
        and "target_fields" not in repair_user["mutation"]
        and "invariant_fields" not in repair_user["mutation"],
    }
    return {"passed": all(checks.values()), "checks": checks, "execution_summary": summary}


def _write_bundle(
    bundle: dict[str, Any],
    report: dict[str, Any],
    *,
    promote: bool,
    output_dir: Path | None = None,
) -> None:
    output_root = Path(output_dir) if output_dir is not None else ROOT / "outputs"
    case_id = bundle.get("case_id", "videosham-1004")
    generator = f"{bundle['execution_summary'].get('teacher_source', 'llm_teacher')}_v1"
    write_json(output_root / "llm_teacher_clean_episode.json", bundle["clean_episode"])
    write_json(output_root / "llm_teacher_repair_episode.json", bundle["repair_episode"])
    write_json(output_root / "llm_teacher_execution_summary.json", bundle["execution_summary"])
    write_json(output_root / "llm_teacher_validation_report.json", report)
    write_json(output_root / "llm_teacher_step_utility_report.json", bundle["step_utility_report"])
    write_jsonl(output_root / "llm_teacher_sft_clean.jsonl", bundle["clean_turn_sft"])
    write_jsonl(output_root / "llm_teacher_sft_repair.jsonl", bundle["repair_turn_sft"])
    write_jsonl(
        output_root / "llm_teacher_transition_sidecar.jsonl",
        compile_transition_sidecar(bundle),
    )
    write_jsonl(
        _replay_trajectory_path(output_root, "clean"),
        [{"messages": bundle["clean_episode"]["messages"], "metadata": {
            "case_id": case_id,
            "trajectory_id": bundle["clean_episode"]["trajectory_id"],
            "trajectory_type": "llm_teacher_clean_acquisition",
            "generator": generator,
            "loss_roles": [],
            "training_eligible": False,
            "purpose": "replay_and_audit_only; use the accepted per-turn JSONL for SFT",
        }}],
    )
    write_jsonl(
        _replay_trajectory_path(output_root, "repair"),
        [{"messages": bundle["repair_episode"]["messages"], "metadata": {
            "case_id": case_id,
            "trajectory_id": bundle["repair_episode"]["trajectory_id"],
            "trajectory_type": bundle["repair_episode"].get(
                "trajectory_type", "llm_teacher_evidence_dropout_repair"
            ),
            "generator": generator,
            "loss_roles": [],
            "training_eligible": False,
            "purpose": "replay_and_audit_only; use the accepted per-turn JSONL for SFT",
        }}],
    )
    if promote:
        if not report["passed"]:
            raise RuntimeError("refusing to promote an invalid LLM teacher bundle")
        write_json(output_root / "interactive_clean_episode.json", bundle["clean_episode"])
        write_json(output_root / "interactive_repair_episode.json", bundle["repair_episode"])
        write_json(output_root / "interactive_execution_summary.json", bundle["execution_summary"])
        write_json(output_root / "interactive_validation_report.json", report)
        write_json(output_root / "interactive_step_utility_report.json", bundle["step_utility_report"])
        write_jsonl(output_root / "interactive_sft_clean.jsonl", bundle["clean_turn_sft"])
        write_jsonl(output_root / "interactive_sft_repair.jsonl", bundle["repair_turn_sft"])
        write_jsonl(
            output_root / "interactive_transition_sidecar.jsonl",
            compile_transition_sidecar(bundle),
        )
        write_jsonl(
            output_root / "interactive_sft_clean_full_trajectory.jsonl",
            [{"messages": bundle["clean_episode"]["messages"], "metadata": {
                "case_id": case_id,
                "trajectory_id": bundle["clean_episode"]["trajectory_id"],
                "trajectory_type": "llm_teacher_clean_acquisition",
                "generator": generator,
                "loss_roles": [],
                "training_eligible": False,
                "purpose": "replay_and_audit_only; use the accepted per-turn JSONL for SFT",
            }}],
        )
        write_jsonl(
            output_root / "interactive_sft_repair_full_trajectory.jsonl",
            [{"messages": bundle["repair_episode"]["messages"], "metadata": {
                "case_id": case_id,
                "trajectory_id": bundle["repair_episode"]["trajectory_id"],
                "trajectory_type": bundle["repair_episode"].get(
                    "trajectory_type", "llm_teacher_evidence_dropout_repair"
                ),
                "generator": generator,
                "loss_roles": [],
                "training_eligible": False,
                "purpose": "replay_and_audit_only; use the accepted per-turn JSONL for SFT",
            }}],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live OpenAI LLM teacher inside the VideoSham harness")
    parser.add_argument("--model", default=os.environ.get("CLAIMUNIT_TEACHER_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-clean-turns", type=int, default=12)
    parser.add_argument("--max-repair-turns", type=int, default=10)
    parser.add_argument("--candidates-per-turn", type=int, default=3)
    parser.add_argument("--proposal-rounds-per-turn", type=int, default=2)
    parser.add_argument(
        "--legacy-full-control",
        action="store_true",
        help="Diagnostic compatibility mode. The default causal view removes answer-like control affordances.",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Inject it into the current process environment; do not put it in JSONL or source."
        )
    manifest = json.loads((ROOT / "student_view" / "case_manifest.json").read_text(encoding="utf-8"))
    oracle = json.loads((ROOT / "oracle" / "claim_oracle.json").read_text(encoding="utf-8"))
    source_frames = decode(SOURCE)
    suspect_frames = decode(EDITED)
    base_teacher = OpenAIResponsesTeacher(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
    )
    from evidence_choice_search import build_evidence_choice_teacher

    teacher = (
        base_teacher
        if args.legacy_full_control
        else build_evidence_choice_teacher(
            base_teacher,
            candidates_per_turn=args.candidates_per_turn,
            proposal_rounds_per_turn=args.proposal_rounds_per_turn,
        )
    )
    bundle = build_llm_teacher_bundle(
        manifest=manifest,
        source_frames=source_frames,
        suspect_frames=suspect_frames,
        teacher=teacher,
        max_clean_turns=args.max_clean_turns,
        max_repair_turns=args.max_repair_turns,
    )
    from step_utility_auditor import apply_step_utility_filter, audit_step_utility

    step_report = audit_step_utility(bundle)
    apply_step_utility_filter(bundle, step_report)
    report = validate_llm_teacher_bundle(bundle, oracle=oracle)
    _write_bundle(bundle, report, promote=args.promote)
    if not report["passed"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise RuntimeError(f"LLM teacher validation failed: {failed}")
    print(
        json.dumps(
            {
                "passed": True,
                "model": args.model,
                "policy_view": getattr(teacher, "policy_view", "legacy_full_control"),
                "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                "promoted": args.promote,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
