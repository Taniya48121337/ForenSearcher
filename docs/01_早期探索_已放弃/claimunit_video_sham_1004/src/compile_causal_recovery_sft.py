"""Compile complete local causal-recovery trajectories for three video datasets.

Every episode starts from an unannounced, student-visible evidence defect and
executes the same real Harness used at inference time:

challenge -> fresh external measurement(s) -> local reopen -> fresh binding ->
commit -> stop.

Private intervention metadata selects construction cases but is never serialized
into model-visible messages.  The output remains a structural candidate because
the deterministic analyst notes have not been naturalized by a local teacher.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from audit_recovery_step_value import audit_episode_step_value
from compile_evidence_first_sft import sanitize_payload
from compile_unannounced_challenge_sft import (
    OPERATOR_ACTION,
    challenge_reasoning,
    challenged_evidence_ids,
)
from causal_trajectory_generation import CAUSAL_SYSTEM_PROMPT
from interactive_harness import VideoForensicHarness
from llm_teacher_rollout import _normalize_harness_params
from pipeline import ROOT, write_json, write_jsonl
from run_unannounced_repair_falsification import (
    DEFAULT_SFT,
    STOP_ROWS,
    _active_registry,
    _builder,
)
from shadow_execute_generated_actions import _jsonl, _restore_harness
from student_tool_contract import build_student_tools
from unannounced_repair_interventions import INTERVENTIONS, apply_intervention


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/causal_recovery_curriculum_v2"
)


def _external_names(registry: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["name"])
        for item in registry
        if item.get("domain") == "external_media"
    }


def _localizer_name(
    method: str, registry: list[dict[str, Any]]
) -> str | None:
    for item in registry:
        parameters = ((item.get("teacher_schema") or {}).get("parameters") or {})
        schema = (parameters.get("properties") or {}).get("method") or {}
        if method in (schema.get("enum") or []):
            return str(item["name"])
    return None


def clean_measurement_methods(
    state: dict[str, Any],
    registry: list[dict[str, Any]],
    target_fields: list[str],
) -> list[str]:
    external = _external_names(registry)
    result: list[str] = []
    for atom in (state.get("evidence_store") or {}).values():
        if not set(atom.get("supports_fields") or []).intersection(target_fields):
            continue
        observation = atom.get("observation") or {}
        provenance = atom.get("provenance") or {}
        candidates = [
            provenance.get("method"),
            observation.get("measurement_method"),
        ]
        localization_method = observation.get("localization_method")
        if isinstance(localization_method, str):
            candidates.append(_localizer_name(localization_method, registry))
        for name in candidates:
            if name in external and name not in result:
                result.append(str(name))
    if not result:
        raise ValueError("clean checkpoint does not identify an executable measurement")
    if "measurement_method" in target_fields:
        claims = state.get("candidate_claims") or []
        current_method = (
            ((claims[0].get("fields") or {}).get("measurement_method") or {}).get(
                "value"
            )
            if claims
            else None
        )
        if current_method in result:
            # A method-provenance or binding challenge should first repeat the
            # method named by the visible claim.  This is the most direct test
            # of whether the value can be preserved, and avoids changing a
            # sound field merely because a different branch appears first in
            # evidence-store insertion order.
            result.remove(current_method)
            result.insert(0, str(current_method))
    return result


def sync_environment_checkpoint(
    environment: Any,
    clean_state: dict[str, Any],
    measurement_methods: list[str],
) -> None:
    """Give a reconstructed environment the call counters of its restored Harness."""

    call_counts = getattr(environment, "call_counts", None)
    if not isinstance(call_counts, dict):
        return
    for item in clean_state.get("evidence_store", {}).values():
        method = str((item.get("provenance") or {}).get("method") or "")
        generated_by = str((item.get("provenance") or {}).get("generated_by") or "")
        observation = item.get("observation") or {}
        direct = str(observation.get("measurement_method") or "")
        for name in call_counts:
            if name == method or name == direct or name.upper().replace("_", "-") in generated_by:
                call_counts[name] = max(int(call_counts[name]), 1)
    for name in measurement_methods:
        if name in call_counts:
            call_counts[name] = max(int(call_counts[name]), 1)
    for name in call_counts:
        if name.startswith("inspect_") or name == "inspect_media":
            call_counts[name] = max(int(call_counts[name]), 1)


def _single_schema_value(
    name: str,
    schema: dict[str, Any],
    state: dict[str, Any],
) -> Any:
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    enum = schema.get("enum") or []
    if len(enum) == 1:
        return copy.deepcopy(enum[0])
    visible_assets = (state.get("task_contract") or {}).get("visible_assets") or {}
    if name in visible_assets:
        return copy.deepcopy(visible_assets[name])
    if schema.get("type") == "array":
        item_enum = (schema.get("items") or {}).get("enum") or []
        minimum = int(schema.get("minItems", 0))
        maximum = int(schema.get("maxItems", len(item_enum)))
        if item_enum and minimum == maximum == len(item_enum):
            return copy.deepcopy(item_enum)
    claims = state.get("candidate_claims") or []
    if claims:
        field = ((claims[0].get("fields") or {}).get(name) or {})
        if "value" in field:
            return copy.deepcopy(field["value"])
    raise ValueError(f"cannot materialize visible argument: {name}")


def materialize_external_arguments(
    harness: VideoForensicHarness, tool_name: str, state: dict[str, Any]
) -> dict[str, Any]:
    tools = build_student_tools(
        harness.claim_schema,
        harness.tool_registry,
        [tool_name],
        executable_actions=[],
    )
    parameters = tools[0]["function"]["parameters"]
    properties = parameters.get("properties") or {}
    return {
        name: _single_schema_value(name, properties[name], state)
        for name in parameters.get("required") or []
    }


def _claim_id(state: dict[str, Any]) -> str:
    claims = state.get("candidate_claims") or []
    if len(claims) != 1:
        raise ValueError("recovery compiler requires exactly one existing claim")
    return str(claims[0]["claim_id"])


def _fresh_bindings(
    harness: VideoForensicHarness,
    claim_id: str,
    target_fields: list[str],
    fresh_evidence_ids: list[str],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    claim = harness.claims[claim_id]
    adjudication = harness.evidence_adjudications.get(claim_id)
    for field_name in target_fields:
        matches = [
            evidence_id
            for evidence_id in fresh_evidence_ids
            if evidence_id in harness.evidence_store
            and harness.evidence_store[evidence_id].active
            and (
                adjudication is None
                or field_name not in adjudication.target_fields
                or evidence_id in adjudication.selected_evidence_ids
            )
            and field_name in harness.evidence_store[evidence_id].supports_fields
            and not harness._challenge_blocks_field_support(evidence_id, field_name)
            and harness._atom_supports_claim_value(
                field_name,
                claim.fields[field_name].value,
                harness.evidence_store[evidence_id],
            )
        ]
        if not matches:
            diagnostics = {
                evidence_id: {
                    "active": bool(
                        harness.evidence_store.get(evidence_id)
                        and harness.evidence_store[evidence_id].active
                    ),
                    "supports_field": bool(
                        harness.evidence_store.get(evidence_id)
                        and field_name
                        in harness.evidence_store[evidence_id].supports_fields
                    ),
                    "challenge_blocked": bool(
                        harness.evidence_store.get(evidence_id)
                        and harness._challenge_blocks_field_support(
                            evidence_id, field_name
                        )
                    ),
                    "value_supported": bool(
                        harness.evidence_store.get(evidence_id)
                        and harness._atom_supports_claim_value(
                            field_name,
                            claim.fields[field_name].value,
                            harness.evidence_store[evidence_id],
                        )
                    ),
                    "adjudication_selected": bool(
                        adjudication is None
                        or field_name not in adjudication.target_fields
                        or evidence_id in adjudication.selected_evidence_ids
                    ),
                    "observation": copy.deepcopy(
                        (harness.evidence_store.get(evidence_id).observation or {})
                        if harness.evidence_store.get(evidence_id) is not None
                        else None
                    ),
                }
                for evidence_id in fresh_evidence_ids
            }
            raise ValueError(
                f"fresh evidence does not support {claim_id}:{field_name}; "
                f"claim_value={claim.fields[field_name].value!r}; fresh={diagnostics}; "
                f"revision_candidates={harness._claim_revision_candidates().get(claim_id, {})}; "
                f"adjudication={copy.deepcopy(adjudication.__dict__) if adjudication else None}"
            )
        result[field_name] = [matches[0]]
    return result


def _assistant_message(
    call_id: str, tool_name: str, arguments: dict[str, Any], reasoning: str
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": reasoning,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": copy.deepcopy(arguments),
                },
            }
        ],
    }


def _schema_for(tools: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [
        tool["function"]["parameters"]
        for tool in tools
        if tool["function"]["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"student schema is not unique: {name}")
    return matches[0]


def execute_sft_step(
    *,
    harness: VideoForensicHarness,
    conversation: list[dict[str, Any]],
    episode_id: str,
    dataset: str,
    case_id: str,
    stage: str,
    step_index: int,
    tool_name: str,
    visible_arguments: dict[str, Any],
    reasoning: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preaction_state = harness.render_state()
    eligible = [str(item) for item in preaction_state.get("eligible_tools") or []]
    if tool_name not in eligible:
        raise ValueError(
            f"{episode_id}:{stage} chose {tool_name}, eligible={eligible}, "
            f"revision_candidates={harness._claim_revision_candidates()}"
        )
    tools = build_student_tools(
        harness.claim_schema,
        harness.tool_registry,
        eligible,
        executable_actions=[],
    )
    errors = list(
        Draft202012Validator(_schema_for(tools, tool_name)).iter_errors(
            visible_arguments
        )
    )
    if errors:
        raise ValueError(
            f"{episode_id}:{stage} target schema error: {errors[0].message}"
        )
    call_id = f"CALL-{episode_id.upper()}-{step_index:02d}"
    assistant = _assistant_message(
        call_id, tool_name, visible_arguments, reasoning
    )
    row = {
        "messages": copy.deepcopy(conversation) + [copy.deepcopy(assistant)],
        "tools": tools,
        "metadata": {
            "row_id": f"{episode_id}:{step_index:02d}:{stage}",
            "episode_id": episode_id,
            "dataset": dataset,
            "case_id": case_id,
            "trajectory_mode": "complete_unannounced_causal_recovery",
            "causal_recovery_stage": stage,
            "target_tool": tool_name,
            "loss_scope": "last_assistant_action_only",
            "policy_view": "evidence_first_reduced_control_v2",
            "target_was_executed": True,
            "compiled_actions_visible": False,
            "repair_recipe_visible": False,
            "private_intervention_visible": False,
            "structural_candidate_only": True,
        },
    }
    internal_arguments = _normalize_harness_params(
        tool_name, copy.deepcopy(visible_arguments)
    )
    harness.record_policy_call(
        tool_name,
        copy.deepcopy(internal_arguments),
        call_id,
        reasoning,
    )
    result = harness.execute_tool(
        tool_name,
        copy.deepcopy(internal_arguments),
        call_id,
    )
    if result.get("status") not in {"accepted", "executed"}:
        raise ValueError(
            f"{episode_id}:{stage} execution status={result.get('status')}"
        )
    visible_result = sanitize_payload(result)
    conversation.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": json.dumps(
                    visible_result, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
    )
    return row, result


def _style_variant(episode_id: str, stage: str, count: int = 4) -> int:
    return sum(ord(char) for char in f"{episode_id}:{stage}") % count


def _display_fact(key: str, value: Any) -> str:
    if key == "copied_segments_frames" and isinstance(value, list) and len(value) > 3:
        value = {
            "segment_count": len(value),
            "first_box": value[0],
            "last_box": value[-1],
        }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _display_semantic_fact(key: str, value: Any) -> str:
    rendered = _display_fact(key, value)
    if key == "preferred_video" and value == "c":
        return f'{rendered} [abstention: neither/equal/insufficient evidence]'
    return rendered


def _compact_atom_fact(state: dict[str, Any], evidence_id: str) -> str:
    atom = (state.get("evidence_store") or {}).get(evidence_id) or {}
    observation = atom.get("observation") or {}
    keys = (
        "relationship",
        "copied_segments_frames",
        "localization_method",
        "preferred_video",
        "video_a_interval_s",
        "video_b_interval_s",
        "measurement_method",
        "source_role",
        "suspect_role",
        "affected_source_interval_s",
        "affected_suspect_interval_s",
        "material_difference_detected",
        "confidence",
        "order_consistent",
    )
    facts = [
        f"{key}={_display_semantic_fact(key, observation[key])}"
        for key in keys
        if key in observation
    ]
    if not facts:
        facts = [
            f"{key}={_display_semantic_fact(key, value)}"
            for key, value in list(observation.items())[:3]
        ]
    return f"{evidence_id} ({'; '.join(facts)})"


def _claim_field_facts(state: dict[str, Any], fields: list[str]) -> str:
    claims = state.get("candidate_claims") or []
    values = (claims[0].get("fields") or {}) if claims else {}
    return "; ".join(
        f"{field_name}={_display_semantic_fact(field_name, values[field_name].get('value'))}"
        for field_name in fields
        if field_name in values
    )


def _measurement_reasoning(
    tool_name: str,
    state: dict[str, Any],
    previous_fresh_ids: list[str],
    style: int,
) -> str:
    open_challenges = state.get("open_evidence_challenges") or []
    if previous_fresh_ids:
        store = state.get("evidence_store") or {}
        prior_observations = [
            (store.get(evidence_id) or {}).get("observation") or {}
            for evidence_id in previous_fresh_ids[-2:]
        ]
        prior = "; ".join(
            _compact_atom_fact(state, evidence_id)
            for evidence_id in previous_fresh_ids[-2:]
        )
        if any(
            observation.get("preferred_video") == "c"
            and observation.get("order_consistent") is False
            for observation in prior_observations
        ):
            return (
                f"{prior} abstains because the A/B order check is inconsistent; it does "
                "not establish a directional preference. "
                f"Running {tool_name} supplies an independent comparison that can test "
                "whether any A/B conclusion is stable before binding."
            )
        if any(
            observation.get("preferred_video") == "c"
            for observation in prior_observations
        ):
            return (
                f"{prior} returns c, meaning neither/equal/insufficient evidence rather "
                "than a third video choice. "
                f"I will run {tool_name} to test whether another executable view can "
                "resolve the A/B comparison without presuming that it will."
            )
        variants = (
            f"The first fresh result, {prior}, still leaves the difference goal open. "
            f"Running {tool_name} adds a distinct visible comparison before any claim repair.",
            f"Although {prior} is informative, it does not yet settle the disputed comparison. "
            f"I will use {tool_name} to test the provisional conclusion through another executable route.",
            f"Because {prior} did not settle every disputed field, binding now would be premature. "
            f"{tool_name} provides an independent comparison that can separate the competing conclusions.",
            f"The remaining open goal is explained by {prior}, not by a missing state edit. "
            f"A real {tool_name} measurement can determine whether the current direction survives additional evidence.",
        )
        return variants[style]
    challenged = sorted(
        {
            evidence_id
            for challenge in open_challenges
            for evidence_id in challenge.get("evidence_ids") or []
        }
    )
    challenged_fact = "; ".join(
        _compact_atom_fact(state, evidence_id) for evidence_id in challenged
    )
    variants = (
        f"The open challenge concerns {challenged_fact}, so its current active flag is not enough for reuse. "
        f"{tool_name} re-executes the disputed comparison and must produce fresh evidence before binding or stop.",
        f"The cited support is {challenged_fact}, and the verification obligation is still open. "
        f"I will call {tool_name} to obtain a new measurement rather than carry the old conclusion forward.",
        f"No repaired binding can rely on {challenged_fact} after the integrity challenge. "
        f"A fresh {tool_name} result is the direct way to retest the same visible media relation.",
        f"The visible conflict in {challenged_fact} requires another observation. "
        f"Executing {tool_name} tests the evidence itself before any field decision is changed.",
    )
    return variants[style]


def _reopen_reasoning(
    target_fields: list[str], fresh_ids: list[str], state: dict[str, Any], style: int
) -> str:
    summary = "; ".join(_compact_atom_fact(state, item) for item in fresh_ids)
    fields = ", ".join(target_fields)
    variants = (
        f"Fresh evidence is now available as {summary}. It supersedes the challenged support for {fields}, so I will reopen only those dependent fields and leave every other commitment unchanged.",
        f"The recheck produced {summary}, while {fields} still carry decisions made from the challenged atom. Reopening exactly that subset makes rebinding legal without disturbing independent fields.",
        f"The visible replacement evidence is {summary}. Since only {fields} depended on the disputed support, a local reopen is sufficient and broader invalidation would be unsupported.",
        f"The challenge has been resolved by {summary}, but the affected committed decisions are now stale. I will set {fields} to open and preserve all fields with independent evidence.",
    )
    return variants[style]


def _revision_reasoning(
    field_updates: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> str:
    facts = "; ".join(
        f"{item['field']}={_display_semantic_fact(item['field'], item['value'])} from "
        f"{', '.join(candidates[item['field']]['evidence_ids'])}"
        for item in field_updates
    )
    return (
        f"The fresh active evidence directly reports {facts}, which differs from the "
        "reopened claim value. Revising only these fields makes the claim match the "
        "new observation; evidence binding and commitment remain separate checks."
    )


def _bind_reasoning(
    bindings: dict[str, list[str]], state: dict[str, Any], style: int
) -> str:
    links = "; ".join(
        f"{field}<-{','.join(evidence_ids)}"
        for field, evidence_ids in bindings.items()
    )
    facts = _claim_field_facts(state, list(bindings))
    variants = (
        f"The current reopened values are {facts}, and the visible support map is {links}. Binding those links consumes the fresh measurement while leaving commitment for the next check.",
        f"For {facts}, each proposed link in {links} points to fresh compatible evidence. I will bind only this subset; independent fields stay untouched and commitment remains separate.",
        f"The fresh atoms match the claim facts {facts}. Recording {links} as bindings restores evidence attribution without silently committing the repaired fields.",
        f"The field-level comparison now supports {facts} through {links}. This binding is local to the reopened obligations, and decision commitment remains a separate action.",
    )
    return variants[style]


def _recommit_reasoning(
    target_fields: list[str], state: dict[str, Any], style: int
) -> str:
    facts = _claim_field_facts(state, target_fields)
    variants = (
        f"The accepted fresh bindings now support {facts}. Committing exactly these reopened fields closes the local decision gap while preserving every independent commitment.",
        f"The repaired subset has supported values {facts}, and no challenged atom remains in its bindings. I will commit that subset without reopening or changing unrelated fields.",
        f"Fresh support for {facts} is now attached to the reopened fields. Committing only that subset completes the repair without changing any independent conclusion.",
        f"Each reopened obligation is now backed by fresh evidence for {facts}. A local commit is justified, whereas changing any independently supported field is not.",
    )
    return variants[style]


def _terminal_reasoning(
    target_fields: list[str], state: dict[str, Any], style: int
) -> str:
    facts = _claim_field_facts(state, target_fields)
    variants = (
        f"The repaired fields are committed as {facts}, their fresh supports are bound, and every high-priority obligation is closed. Stopping is now justified because no unsupported or conflicted field remains.",
        f"The visible record now supports and commits {facts}, with no open challenge or repair obligation. I will stop because another measurement would not resolve any remaining evidential question.",
        f"Fresh evidence supports and commits {facts}, while independently supported fields remain unchanged. All required fields are closed, so stop is the only non-redundant action.",
        f"The recovery ends with {facts} supported, bound, and committed, and the obligation list is empty. A stop is warranted because the repaired conclusion no longer depends on the challenged evidence-field link.",
    )
    return variants[style]


def _retention_hard_defects(
    harness: VideoForensicHarness,
    requirement: dict[str, Any],
    evidence_id: str,
) -> set[str]:
    observation = harness.evidence_store[evidence_id].observation or {}
    hard_defects = {"role_assignment_mismatch"}
    if observation.get("preferred_video") != "c":
        hard_defects.add("order_inconsistent")
    return hard_defects.intersection(
        harness._visible_evidence_defects(requirement, evidence_id)
    )


def _fresh_adjudication_cover(
    harness: VideoForensicHarness,
    requirement: dict[str, Any],
    fresh_ids: list[str],
) -> list[str]:
    selected = harness._evidence_first_selection(requirement)
    if not set(selected).issubset(set(fresh_ids)):
        competitor_observations = {
            evidence_id: {
                "kind": harness.evidence_store[evidence_id].kind,
                "observation": copy.deepcopy(
                    harness.evidence_store[evidence_id].observation or {}
                ),
                "provenance": copy.deepcopy(
                    harness.evidence_store[evidence_id].provenance or {}
                ),
            }
            for evidence_id in requirement.get("competing_evidence_ids", [])
            if evidence_id in harness.evidence_store
        }
        raise ValueError(
            "evidence-first selection escaped the fresh measurement set; "
            f"selected={selected}, fresh={fresh_ids}, "
            f"competitors={requirement.get('competing_evidence_ids')}, "
            f"target_fields={requirement.get('target_fields')}, "
            f"competitor_observations={competitor_observations}"
        )
    return selected


def _minimal_rejection_defects(
    harness: VideoForensicHarness,
    requirement: dict[str, Any],
    evidence_id: str,
    selected: list[str],
) -> list[str]:
    """Choose the smallest visible reason that actually distinguishes a rejection."""

    supported = harness._visible_evidence_defects(
        requirement, evidence_id, selected_evidence_ids=selected
    )
    for code in (
        "role_assignment_mismatch",
        "cross_method_direction_conflict",
    ):
        if code in supported:
            return [code]
    if "order_inconsistent" in supported:
        selected_order_safe = all(
            (harness.evidence_store[item].observation or {}).get(
                "order_consistent"
            )
            is not False
            for item in selected
        )
        if selected_order_safe:
            return ["order_inconsistent"]
    if "redundant_same_conclusion" in supported:
        return ["redundant_same_conclusion"]
    if "lower_confidence" in supported:
        return ["lower_confidence"]
    if "claim_value_conflict" in supported:
        return ["claim_value_conflict"]
    raise ValueError(f"no discriminative visible defect for {evidence_id}")


def _assessment_reasoning(
    evidence_id: str,
    disposition: str,
    defect_codes: list[str],
    state: dict[str, Any],
) -> str:
    observation = _compact_atom_fact(state, evidence_id)
    atom = (state.get("evidence_store") or {}).get(evidence_id) or {}
    values = atom.get("observation") or {}
    requirement = (state.get("evidence_adjudication") or {}).get("requirement") or {}
    target_fields = list(requirement.get("target_fields") or [])
    claim_facts = _claim_field_facts(state, target_fields)
    if disposition == "retain":
        if values.get("preferred_video") == "c":
            if values.get("order_consistent") is False:
                return (
                    f"{observation} is complete for the fields under review. The A/B "
                    "order check disagrees, so c is an abstention rather than a third "
                    "video; here that inconsistency is the reason not to assert A or B, "
                    "not a directional result being trusted. I will retain the abstaining "
                    "candidate with no defect code; retention still does not bind or commit it."
                )
            return (
                f"{observation} is complete for the fields under review. Its c value "
                "means neither/equal/insufficient evidence, so retaining it records an "
                "abstention rather than inventing an A/B preference. Retention carries no "
                "defect code and does not yet bind or commit the claim."
            )
        return (
            f"{observation} is complete for the fields under review and survives the "
            "label-blind cross-method rule. I will retain it with no defect code even if "
            f"it differs from the reopened claim facts {claim_facts}; retention does not "
            "yet revise, bind, or commit the claim."
        )

    code = defect_codes[0] if defect_codes else ""
    assessments = state.get("evidence_assessments") or {}
    retained_ids = [
        candidate_id
        for candidate_id, assessment in assessments.items()
        if assessment.get("disposition") == "retain"
    ]
    retained_facts = "; ".join(
        _compact_atom_fact(state, candidate_id) for candidate_id in retained_ids
    )
    if code == "redundant_same_conclusion":
        reference = retained_facts or claim_facts
        return (
            f"{observation} repeats the same field conclusion already represented by "
            f"{reference}; it is redundant rather than contradictory. I will reject the "
            "duplicate so the later binding does not double-count one conclusion."
        )
    if code == "claim_value_conflict":
        return (
            f"{observation} does not match the reopened claim facts {claim_facts}. "
            "I will reject it so that concrete value conflict cannot enter the next "
            "evidence binding."
        )
    if code == "cross_method_direction_conflict":
        return (
            f"{observation} gives a different A/B/c direction from the independently "
            "supported ensemble result. I will reject it as a cross-method direction "
            "conflict so it cannot enter the adjudicated support partition or later "
            "binding; the original claim value is not being used to pick the winner."
        )
    if code == "abstention_conflicts_with_claim":
        return (
            f"{observation} abstains with preferred_video=c, whereas the reopened claim "
            f"asserts {claim_facts}. I will reject it because an abstention cannot support "
            "or be bound to that directional claim."
        )
    if code == "order_inconsistent":
        return (
            f"{observation} reports an A/B direction while order_consistent=false. The "
            "direction changes under presentation order, so I will reject it rather than "
            "bind the unstable preference as support."
        )
    if code == "lower_confidence":
        reference = retained_facts or "the retained fresh candidate"
        return (
            f"{observation} is lower-confidence than {reference}. I will reject the "
            "weaker competitor so it is not used as interchangeable support in binding."
        )
    if code == "role_assignment_mismatch":
        return (
            f"{observation} was measured under a role assignment incompatible with the "
            "current A/B contract. I will reject it instead of binding values across roles."
        )
    return (
        f"{observation} has the visible defect {', '.join(defect_codes)} relative to "
        f"{claim_facts}. I will reject it so incompatible evidence is not used in binding."
    )


def _adjudication_reasoning(
    selected: list[str], rejected: list[dict[str, Any]], state: dict[str, Any]
) -> str:
    selected_text = "; ".join(
        _compact_atom_fact(state, evidence_id) for evidence_id in selected
    )
    selected_store = state.get("evidence_store") or {}
    rejected_parts = []
    for item in rejected:
        evidence_id = item["evidence_id"]
        observation = ((selected_store.get(evidence_id) or {}).get("observation") or {})
        summary_fields = (
            "preferred_video",
            "measurement_method",
            "confidence",
            "order_consistent",
        )
        facts = ", ".join(
            f"{key}={_display_semantic_fact(key, observation[key])}"
            for key in summary_fields
            if key in observation
        )
        rejected_parts.append(
            f"{evidence_id} ({facts}) -> {','.join(item['defect_codes'])}"
        )
    rejected_text = "; ".join(rejected_parts)
    abstention_note = ""
    if any(
        ((selected_store.get(evidence_id) or {}).get("observation") or {}).get(
            "preferred_video"
        )
        == "c"
        for evidence_id in selected
    ):
        abstention_note = (
            " The selected c conclusion is an explicit abstention, not a third video."
        )
    return (
        f"Every fresh competitor has now been compared. The selected support is "
        f"{selected_text}.{abstention_note} The rejected set is {rejected_text}. "
        "Recording this evidence-level partition prevents the next binding from mixing "
        "conflicting or duplicate conclusions."
    )


STAGE_WEIGHTS = {
    "challenge": ("belief_repair", 1.2),
    "fresh_measurement": ("evidence_measurement", 1.0),
    "local_reopen": ("belief_repair", 1.2),
    "claim_revision": ("belief_repair", 1.3),
    "post_adjudication_claim_revision": ("belief_repair", 1.3),
    "final_visible_claim_revision": ("belief_repair", 1.3),
    "evidence_assessment": ("evidence_assessment_or_adjudication", 1.1),
    "evidence_adjudication": ("evidence_assessment_or_adjudication", 1.1),
    "fresh_binding": ("evidence_attribution", 0.7),
    "recommit": ("belief_commitment", 0.6),
    "terminal_verification": ("terminal_verification", 0.2),
}


def compile_recovery_episodes(
    source_rows: dict[str, dict[str, Any]],
    *,
    intervention_specs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = INTERVENTIONS,
    job_builder: Any | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    sft_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    step_value_reports: list[dict[str, Any]] = []
    for episode_index, spec in enumerate(intervention_specs, start=1):
        if job_builder is None:
            environment, base_registry = _builder(spec["dataset"], source_rows)
            seed_row = source_rows[STOP_ROWS[spec["dataset"]]]
        else:
            environment, base_registry, seed_row = job_builder(spec)
        harness = _restore_harness(seed_row)
        harness.environment = environment
        harness.tool_registry = _active_registry(base_registry)
        clean_state = harness.render_state()
        clean_control = harness.clone_state(environment=environment)
        clean_stop = clean_control.execute_tool(
            "stop", {}, f"CALL-CLEAN-CONTROL-{episode_index:03d}"
        )
        clean_control_suppresses_challenge = (
            clean_stop.get("status") == "accepted"
        )
        normal_eligible = harness.eligible_tool_registry
        private_metadata = apply_intervention(harness, spec)
        # The intervention preflight deliberately exposes a broad audit surface.
        # Complete trajectories use the normal dependency-ordered Harness after
        # the defect has been injected.
        harness.eligible_tool_registry = normal_eligible
        preaction_state = harness.render_state()
        evidence_ids = challenged_evidence_ids(clean_state, preaction_state)
        moved_state = copy.deepcopy(preaction_state)
        for evidence_id in evidence_ids:
            if evidence_id in (clean_state.get("evidence_store") or {}):
                moved_state["evidence_store"][evidence_id] = copy.deepcopy(
                    clean_state["evidence_store"][evidence_id]
                )
            else:
                moved_state["evidence_store"].pop(evidence_id, None)
        moved_state["field_bindings"] = copy.deepcopy(
            clean_state.get("field_bindings") or {}
        )
        moved_id = f"EA-MOVED-DEFECT-{episode_index:03d}"
        moved_atom = copy.deepcopy(
            preaction_state["evidence_store"][evidence_ids[0]]
        )
        moved_atom["evidence_id"] = moved_id
        moved_state["evidence_store"][moved_id] = moved_atom
        moved_targets = challenged_evidence_ids(clean_state, moved_state)
        defect_target_moves = (
            moved_id in moved_targets
            and not set(evidence_ids).intersection(moved_targets)
        )
        target_fields = [
            str(field_name)
            for field_name in private_metadata.get(
                "affected_fields_verifier_only", []
            )
        ]
        defect_code, requested_check = OPERATOR_ACTION[spec["operator"]]
        measurement_methods = clean_measurement_methods(
            clean_state, harness.tool_registry, target_fields
        )
        sync_environment_checkpoint(
            environment, clean_state, measurement_methods
        )

        episode_id = str(
            spec.get("episode_id") or f"causal-recovery-{episode_index:03d}"
        )
        case_id = str(harness.contract.contract_id)
        initial_payload = sanitize_payload(
            {
                "task": {
                    "instruction": harness.contract.task,
                    "claim_schema": copy.deepcopy(harness.claim_schema),
                },
                "current_harness_state": preaction_state,
                "trajectory_contract": (
                    "Detect evidence-integrity failures from visible content, request "
                    "a real fresh check, and repair only the dependent fields. No error "
                    "location, restoration value, or next-tool answer is supplied."
                ),
            }
        )
        conversation: list[dict[str, Any]] = [
            {"role": "system", "content": CAUSAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    initial_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        episode_rows: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        step_index = 1

        reasoning = challenge_reasoning(
            state=preaction_state,
            evidence_ids=evidence_ids,
            defect_code=defect_code,
            target_fields=target_fields,
            style=_style_variant(episode_id, "challenge"),
        ) + (
            f" This puts only {', '.join(target_fields)} in doubt; unrelated committed "
            "fields retain their existing support."
        )
        row, result = execute_sft_step(
            harness=harness,
            conversation=conversation,
            episode_id=episode_id,
            dataset=spec["dataset"],
            case_id=case_id,
            stage="challenge",
            step_index=step_index,
            tool_name="challenge_evidence",
            visible_arguments={
                "evidence_ids": evidence_ids,
                "defect_codes": [defect_code],
                "target_fields": target_fields,
                "requested_check": requested_check,
            },
            reasoning=reasoning,
        )
        episode_rows.append(row)
        executions.append(result)
        step_index += 1

        fresh_ids: list[str] = []
        executed_measurement_methods: list[str] = []
        while harness.goals["G-DIFFERENCE"].status != "satisfied":
            eligible = [
                str(item["name"])
                for item in harness.eligible_tool_registry()
                if item.get("domain") == "external_media"
            ]
            remaining = [
                name
                for name in measurement_methods
                if name in eligible and name not in executed_measurement_methods
            ]
            if not remaining:
                # The current environment may require one more visible branch than
                # the historical checkpoint (for example, after a stricter routing
                # rule was introduced).  Continue only through a currently eligible
                # real measurement, never through a private restoration recipe.
                remaining = [
                    name
                    for name in eligible
                    if name not in executed_measurement_methods
                ]
            if not remaining:
                # A conclusive fresh measurement can legitimately invalidate the
                # old claim value (for example after measurement semantics improve).
                # Stop searching when active visible evidence offers one unique
                # alternative; the policy must reopen and explicitly revise it.
                if harness._claim_revision_candidates():
                    break
                raise ValueError(f"{episode_id} exhausted every visible measurement")
            tool_name = remaining[0]
            executed_measurement_methods.append(tool_name)
            state = harness.render_state()
            arguments = materialize_external_arguments(harness, tool_name, state)
            row, result = execute_sft_step(
                harness=harness,
                conversation=conversation,
                episode_id=episode_id,
                dataset=spec["dataset"],
                case_id=case_id,
                stage="fresh_measurement",
                step_index=step_index,
                tool_name=tool_name,
                visible_arguments=arguments,
                reasoning=_measurement_reasoning(
                    tool_name,
                    state,
                    fresh_ids,
                    _style_variant(episode_id, f"measurement-{step_index}"),
                ),
            )
            episode_rows.append(row)
            executions.append(result)
            fresh_ids.extend(str(item) for item in result.get("created_evidence_ids") or [])
            step_index += 1
            if step_index > 8:
                raise ValueError(f"{episode_id} did not close the fresh measurement goal")

        state = harness.render_state()
        claim_id = _claim_id(state)
        row, result = execute_sft_step(
            harness=harness,
            conversation=conversation,
            episode_id=episode_id,
            dataset=spec["dataset"],
            case_id=case_id,
            stage="local_reopen",
            step_index=step_index,
            tool_name="set_field_decision",
            visible_arguments={
                "claim_id": claim_id,
                "fields": target_fields,
                "decision_status": "open",
            },
            reasoning=_reopen_reasoning(
                target_fields,
                fresh_ids,
                state,
                _style_variant(episode_id, "reopen"),
            ),
        )
        episode_rows.append(row)
        executions.append(result)
        step_index += 1

        revision_candidates = harness._claim_revision_candidates().get(claim_id, {})
        if revision_candidates:
            pre_revision_requirement = harness._evidence_adjudication_requirement()
            current_claim = harness.claims[claim_id]

            def fresh_preserves_current(field_name: str) -> bool:
                current_value = current_claim.fields[field_name].value
                for evidence_id in fresh_ids:
                    atom = harness.evidence_store.get(evidence_id)
                    if (
                        atom is None
                        or not atom.active
                        or field_name not in atom.supports_fields
                        or atom.observation.get(field_name) != current_value
                    ):
                        continue
                    if pre_revision_requirement is None or not _retention_hard_defects(
                        harness, pre_revision_requirement, evidence_id
                    ):
                        return True
                return False

            revision_fields = [
                field_name
                for field_name in target_fields
                if field_name in revision_candidates
                and not fresh_preserves_current(field_name)
            ]
            if revision_fields:
                field_updates = [
                    {
                        "field": field_name,
                        "value": copy.deepcopy(revision_candidates[field_name]["value"]),
                    }
                    for field_name in revision_fields
                ]
                state = harness.render_state()
                row, result = execute_sft_step(
                    harness=harness,
                    conversation=conversation,
                    episode_id=episode_id,
                    dataset=spec["dataset"],
                    case_id=case_id,
                    stage="claim_revision",
                    step_index=step_index,
                    tool_name="revise_claim_fields",
                    visible_arguments={
                        "claim_id": claim_id,
                        "field_updates": field_updates,
                    },
                    reasoning=_revision_reasoning(
                        field_updates, revision_candidates
                    ),
                )
                episode_rows.append(row)
                executions.append(result)
                step_index += 1
                if harness.goals["G-DIFFERENCE"].status != "satisfied":
                    raise ValueError(
                        f"{episode_id} claim revision did not close the measurement goal"
                    )

        requirement = harness._evidence_adjudication_requirement()
        if requirement is not None:
            selected = _fresh_adjudication_cover(harness, requirement, fresh_ids)
            rejected = []
            for evidence_id in requirement["competing_evidence_ids"]:
                if evidence_id in selected:
                    continue
                defects = _minimal_rejection_defects(
                    harness, requirement, evidence_id, selected
                )
                rejected.append(
                    {"evidence_id": evidence_id, "defect_codes": defects}
                )
            while any(
                item["name"] == "assess_evidence_candidate"
                for item in harness.eligible_tool_registry()
            ):
                state = harness.render_state()
                current_id = str(
                    (state.get("evidence_assessment") or {}).get(
                        "current_candidate_id"
                    )
                    or ""
                )
                if not current_id:
                    raise ValueError(f"{episode_id} assessment queue has no candidate")
                rejected_map = {
                    item["evidence_id"]: item["defect_codes"] for item in rejected
                }
                disposition = "retain" if current_id in selected else "reject"
                defect_codes = [] if disposition == "retain" else rejected_map[current_id]
                supported_now = harness._visible_evidence_defects(
                    requirement, current_id
                )
                if disposition == "reject" and not set(defect_codes).issubset(
                    supported_now
                ):
                    raise ValueError(
                        f"{episode_id} assessment defects drifted for {current_id}: "
                        f"chosen={defect_codes}, supported={sorted(supported_now)}"
                    )
                row, result = execute_sft_step(
                    harness=harness,
                    conversation=conversation,
                    episode_id=episode_id,
                    dataset=spec["dataset"],
                    case_id=case_id,
                    stage="evidence_assessment",
                    step_index=step_index,
                    tool_name="assess_evidence_candidate",
                    visible_arguments={
                        "claim_id": claim_id,
                        "target_fields": list(requirement["target_fields"]),
                        "evidence_id": current_id,
                        "disposition": disposition,
                        "defect_codes": defect_codes,
                    },
                    reasoning=_assessment_reasoning(
                        current_id, disposition, defect_codes, state
                    ),
                )
                episode_rows.append(row)
                executions.append(result)
                step_index += 1
            if any(
                item["name"] == "adjudicate_evidence"
                for item in harness.eligible_tool_registry()
            ):
                state = harness.render_state()
                row, result = execute_sft_step(
                    harness=harness,
                    conversation=conversation,
                    episode_id=episode_id,
                    dataset=spec["dataset"],
                    case_id=case_id,
                    stage="evidence_adjudication",
                    step_index=step_index,
                    tool_name="adjudicate_evidence",
                    visible_arguments={
                        "claim_id": claim_id,
                        "target_fields": list(requirement["target_fields"]),
                        "selected_evidence_ids": selected,
                        "rejected_evidence": rejected,
                    },
                    reasoning=_adjudication_reasoning(selected, rejected, state),
                )
                episode_rows.append(row)
                executions.append(result)
                step_index += 1

            # Evidence is selected before claim compatibility is considered.  If the
            # label-blind ensemble disagrees with the reopened claim, make that change
            # an explicit tool turn before binding the selected atom.
            revision_candidates = harness._claim_revision_candidates().get(claim_id, {})
            revision_fields = [
                field_name
                for field_name in target_fields
                if field_name in revision_candidates
            ]
            if revision_fields:
                field_updates = [
                    {
                        "field": field_name,
                        "value": copy.deepcopy(revision_candidates[field_name]["value"]),
                    }
                    for field_name in revision_fields
                ]
                row, result = execute_sft_step(
                    harness=harness,
                    conversation=conversation,
                    episode_id=episode_id,
                    dataset=spec["dataset"],
                    case_id=case_id,
                    stage="claim_revision",
                    step_index=step_index,
                    tool_name="revise_claim_fields",
                    visible_arguments={
                        "claim_id": claim_id,
                        "field_updates": field_updates,
                    },
                    reasoning=(
                        _revision_reasoning(field_updates, revision_candidates)
                        + " This revision follows the selected cross-method evidence "
                        "partition rather than treating the old claim as a selection prior."
                    ),
                )
                episode_rows.append(row)
                executions.append(result)
                step_index += 1

        # A field-level alternative can become unambiguous only after the full
        # evidence partition is recorded (or after another target field preserves
        # its old value).  Resolve that final visible revision surface here as well.
        remaining_revision_candidates = (
            harness._claim_revision_candidates().get(claim_id, {})
            if claim_id not in harness.evidence_adjudications
            else {}
        )
        remaining_revision_fields = [
            field_name
            for field_name in target_fields
            if field_name in remaining_revision_candidates
        ]
        if remaining_revision_fields:
            field_updates = [
                {
                    "field": field_name,
                    "value": copy.deepcopy(
                        remaining_revision_candidates[field_name]["value"]
                    ),
                }
                for field_name in remaining_revision_fields
            ]
            row, result = execute_sft_step(
                harness=harness,
                conversation=conversation,
                episode_id=episode_id,
                dataset=spec["dataset"],
                case_id=case_id,
                stage="final_visible_claim_revision",
                step_index=step_index,
                tool_name="revise_claim_fields",
                visible_arguments={
                    "claim_id": claim_id,
                    "field_updates": field_updates,
                },
                reasoning=_revision_reasoning(
                    field_updates, remaining_revision_candidates
                ),
            )
            episode_rows.append(row)
            executions.append(result)
            step_index += 1

        no_fresh = harness.clone_state(environment=environment)
        for evidence_id in fresh_ids:
            if evidence_id in no_fresh.evidence_store:
                no_fresh.evidence_store[evidence_id].active = False
        try:
            _fresh_bindings(no_fresh, claim_id, target_fields, fresh_ids)
            fresh_removal_blocks_repair = False
        except ValueError:
            fresh_removal_blocks_repair = True

        bindings = _fresh_bindings(
            harness, claim_id, target_fields, fresh_ids
        )
        row, result = execute_sft_step(
            harness=harness,
            conversation=conversation,
            episode_id=episode_id,
            dataset=spec["dataset"],
            case_id=case_id,
            stage="fresh_binding",
            step_index=step_index,
            tool_name="bind_evidence",
            visible_arguments={
                "claim_id": claim_id,
                "binding_items": [
                    {"field": field_name, "support_ids": support_ids}
                    for field_name, support_ids in bindings.items()
                ],
            },
            reasoning=_bind_reasoning(
                bindings,
                harness.render_state(),
                _style_variant(episode_id, "binding"),
            ),
        )
        episode_rows.append(row)
        executions.append(result)
        step_index += 1

        row, result = execute_sft_step(
            harness=harness,
            conversation=conversation,
            episode_id=episode_id,
            dataset=spec["dataset"],
            case_id=case_id,
            stage="recommit",
            step_index=step_index,
            tool_name="set_field_decision",
            visible_arguments={
                "claim_id": claim_id,
                "fields": target_fields,
                "decision_status": "committed",
            },
            reasoning=_recommit_reasoning(
                target_fields,
                harness.render_state(),
                _style_variant(episode_id, "recommit"),
            ),
        )
        episode_rows.append(row)
        executions.append(result)
        step_index += 1

        row, result = execute_sft_step(
            harness=harness,
            conversation=conversation,
            episode_id=episode_id,
            dataset=spec["dataset"],
            case_id=case_id,
            stage="terminal_verification",
            step_index=step_index,
            tool_name="stop",
            visible_arguments={},
            reasoning=_terminal_reasoning(
                target_fields,
                harness.render_state(),
                _style_variant(episode_id, "terminal"),
            ),
        )
        episode_rows.append(row)
        executions.append(result)

        for current in episode_rows:
            subtype, weight = STAGE_WEIGHTS[
                current["metadata"]["causal_recovery_stage"]
            ]
            current["metadata"]["supervision_subtype"] = subtype
            current["metadata"]["learning_value_weight"] = weight
        final_state = harness.render_state()
        final_bindings = final_state["field_bindings"][claim_id]
        independent_fields = [
            field_name
            for field_name in harness.required_fields
            if field_name not in target_fields
        ]
        independent_preserved = all(
            (clean_state["field_bindings"][claim_id].get(field_name) or {}).get(
                "support_ids"
            )
            == (final_bindings.get(field_name) or {}).get("support_ids")
            for field_name in independent_fields
        )
        fresh_consumed = all(
            set((final_bindings.get(field_name) or {}).get("support_ids") or []).intersection(
                fresh_ids
            )
            for field_name in target_fields
        )
        challenged_rebound = any(
            set((final_bindings.get(field_name) or {}).get("support_ids") or []).intersection(
                evidence_ids
            )
            for field_name in target_fields
        )
        if not final_state.get("terminal"):
            raise ValueError(f"{episode_id} failed terminal validation")
        if not independent_preserved or not fresh_consumed or challenged_rebound:
            raise ValueError(f"{episode_id} failed recovery locality/consumption checks")

        visible_blob = json.dumps(
            [row["messages"] for row in episode_rows], ensure_ascii=False
        )
        forbidden = [str(spec["id"]), str(spec["operator"]), "affected_fields_verifier_only"]
        leakage_hits = [value for value in forbidden if value in visible_blob]
        if leakage_hits:
            raise ValueError(f"{episode_id} student-visible leakage: {leakage_hits}")
        episode_record = {
            "episode_id": episode_id,
            "dataset": spec["dataset"],
            "case_id": case_id,
            "messages": conversation,
            "turns": len(episode_rows),
            "target_fields": target_fields,
            "fresh_evidence_ids": fresh_ids,
            "terminal": True,
            "independent_fields_preserved": independent_preserved,
            "fresh_evidence_consumed": fresh_consumed,
            "challenged_evidence_rebound": challenged_rebound,
            "clean_control_suppresses_challenge": clean_control_suppresses_challenge,
            "defect_target_moves_with_atom": defect_target_moves,
            "fresh_removal_blocks_repair": fresh_removal_blocks_repair,
            "student_visible_leakage_hits": leakage_hits,
        }
        step_value = audit_episode_step_value(episode_record, episode_rows)
        if not step_value["passed"]:
            failures = [
                {
                    "turn": item["turn"],
                    "stage": item["stage"],
                    "failures": item["failures"],
                }
                for item in step_value["details"]
                if not item["passed"]
            ]
            raise ValueError(
                f"{episode_id} contains executed steps without a downstream decision "
                f"role: {failures}"
            )
        by_turn = {item["turn"]: item for item in step_value["details"]}
        for turn, current in enumerate(episode_rows, start=1):
            current["metadata"]["step_value_audited"] = True
            current["metadata"]["step_value_passed"] = by_turn[turn]["passed"]
            current["metadata"]["step_value_role"] = by_turn[turn]["stage"]
        step_value_reports.append(step_value)
        sft_rows.extend(episode_rows)
        episodes.append(episode_record)
        sidecars.append(
            {
                "episode_id": episode_id,
                "intervention_id": spec["id"],
                "operator": spec["operator"],
                "private_generation_metadata": private_metadata,
                "challenged_evidence_ids": evidence_ids,
                "target_fields": target_fields,
                "measurement_methods": measurement_methods,
                "fresh_evidence_ids": fresh_ids,
            }
        )

    stage_counts = Counter(
        row["metadata"]["causal_recovery_stage"] for row in sft_rows
    )
    report = {
        "artifact": "complete_causal_recovery_sft_candidate_v2",
        "artifact_status": "structural_candidate_not_training_qualified",
        "episodes": len(episodes),
        "sft_rows": len(sft_rows),
        "datasets": dict(Counter(item["dataset"] for item in episodes)),
        "stage_counts": dict(stage_counts),
        "terminal_success": sum(item["terminal"] for item in episodes),
        "fresh_evidence_consumed": sum(
            item["fresh_evidence_consumed"] for item in episodes
        ),
        "independent_fields_preserved": sum(
            item["independent_fields_preserved"] for item in episodes
        ),
        "challenged_evidence_rebound": sum(
            item["challenged_evidence_rebound"] for item in episodes
        ),
        "clean_control_suppresses_challenge": sum(
            item["clean_control_suppresses_challenge"] for item in episodes
        ),
        "defect_target_moves_with_atom": sum(
            item["defect_target_moves_with_atom"] for item in episodes
        ),
        "fresh_removal_blocks_repair": sum(
            item["fresh_removal_blocks_repair"] for item in episodes
        ),
        "student_visible_leakage_episodes": sum(
            bool(item["student_visible_leakage_hits"]) for item in episodes
        ),
        "step_value_audit": {
            "episodes_passed": sum(item["passed"] for item in step_value_reports),
            "turns_passed": sum(
                item["turns"] - item["failed_turns"] for item in step_value_reports
            ),
            "all_steps_have_downstream_decision_role": all(
                item["passed"] for item in step_value_reports
            ),
        },
        "qualification_blockers": [
            "deterministic_notes_require_local_teacher_naturalization",
            "closed_loop_student_gain_not_evaluated",
            "sealed_cross_dataset_evaluation_not_available",
        ],
    }
    return sft_rows, episodes, sidecars, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_rows = {
        str(row["metadata"]["row_id"]): row for row in _jsonl(args.sft)
    }
    rows, episodes, sidecars, report = compile_recovery_episodes(source_rows)
    write_jsonl(args.output_root / "recovery_sft_candidate.jsonl", rows)
    write_jsonl(args.output_root / "recovery_episodes.jsonl", episodes)
    write_jsonl(args.output_root / "private_intervention_sidecar.jsonl", sidecars)
    write_json(args.output_root / "compile_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
