"""Strict unified quality audit for student-generated clean and repair loops.

The audit is target-hidden.  It checks the student's own executed trajectory
against its public initial state and does not compare with a teacher action.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from run_student_closed_loop_canary import (
    VCSL_FIRST,
    VCSL_REPAIR_FIRST,
    VIDDIFF_FIRST,
    VIDDIFF_REPAIR_FIRST,
    VIDEO_FIRST,
    VIDEO_REPAIR_FIRST,
    _adjudication_reasoning_problems,
    _bind_evidence_reasoning_problem,
    _bind_latest_measurement_problem,
    _chosen_measurement_tool_problem,
    _evidence_assessment_reasoning_problems,
    _eligible_tool_claim_problems,
    _measurement_need_problem,
    _measurement_outcome_presupposition_problem,
    _minimum_reasoning_words,
    _reasoning_process_artifact_problems,
    _visible_observation_fact_problems,
)
from shadow_execute_generated_actions import _jsonl, _payload


DEFAULT_CLEAN = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_canary_three_datasets_restoration_hint_6144.json"
)
DEFAULT_REPAIR = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_repair_three_datasets_6144.json"
)
DEFAULT_SFT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "decision_sufficient_state_v6_executable_affordances"
    / "validation/sft_all.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_unified_quality_audit.json"
)

EVIDENCE_PATTERN = re.compile(r"\bEA-[A-Z0-9_-]+-\d{3}\b", re.I)
STATE_TOOLS = {
    "create_candidate_claim",
    "assess_evidence_candidate",
    "adjudicate_evidence",
    "bind_evidence",
    "set_field_decision",
    "stop",
}
SEED_ROWS = {
    "clean": {
        "VideoSham": VIDEO_FIRST,
        "VCSL": VCSL_FIRST,
        "VidDiffBench": VIDDIFF_FIRST,
    },
    "repair": {
        "VideoSham": VIDEO_REPAIR_FIRST,
        "VCSL": VCSL_REPAIR_FIRST,
        "VidDiffBench": VIDDIFF_REPAIR_FIRST,
    },
}

WEIGHTS = {
    "terminal_integrity": 15,
    "execution_integrity": 15,
    "evidence_reference_integrity": 12,
    "cross_turn_entity_consistency": 10,
    "evidence_consumption": 10,
    "mutation_locality_or_clean_scope": 10,
    "dependency_ordering": 10,
    "route_efficiency": 8,
    "human_like_reasoning": 5,
    "context_integrity": 5,
}
CRITICAL = {
    "terminal_integrity",
    "execution_integrity",
    "evidence_reference_integrity",
    "cross_turn_entity_consistency",
    "evidence_consumption",
    "mutation_locality_or_clean_scope",
    "context_integrity",
}


def _meta_abstention_reasoning_problems(
    turn_index: int,
    reasoning: str,
    cited_evidence_ids: set[str],
    trajectory_grounded_evidence_ids: set[str] | None = None,
) -> list[str]:
    """Check that the derived abstention is described as aggregation, not sensing."""

    required_sources = {
        "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC": r"global[-_ ]motion",
        "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC": r"action[-_ ]refined",
    }
    grounded = trajectory_grounded_evidence_ids or cited_evidence_ids
    problems: list[str] = []
    for source, method_pattern in required_sources.items():
        exact_current = any(
            evidence_id.startswith(source) for evidence_id in cited_evidence_ids
        )
        grounded_earlier = any(evidence_id.startswith(source) for evidence_id in grounded)
        method_named_now = re.search(method_pattern, reasoning, re.I) is not None
        fixed_pair_named_now = re.search(
            r"both (?:fixed )?(?:symmetric )?(?:source )?routes|"
            r"(?:two|both) (?:already executed )?symmetric (?:observations|measurements|routes)|"
            r"already executed symmetric (?:observations|measurements|routes)",
            reasoning,
            re.I,
        ) is not None
        if not exact_current and not (
            grounded_earlier and (method_named_now or fixed_pair_named_now)
        ):
            problems.append(f"turn_{turn_index}_meta_abstention_missing_source_citation:{source}")
    if not re.search(
        r"aggregat|combin|already executed|prior (?:route|measurement)|two (?:route|measurement)",
        reasoning,
        re.I,
    ):
        problems.append(f"turn_{turn_index}_meta_abstention_not_described_as_aggregation")
    if not re.search(
        r"(?:without|does not|doesn't|cannot|no new).{0,45}(?:inspect|pixel|frame|image|visual|sens)",
        reasoning,
        re.I,
    ):
        problems.append(f"turn_{turn_index}_meta_abstention_omits_no_new_sensing_contract")
    asserted_sensing = re.sub(
        r"(?:without|does not|doesn't|cannot|can't|will not|won't|never)\s+"
        r"(?:inspect|examine|look at|re-?inspect)(?:ing)?",
        "",
        reasoning,
        flags=re.I,
    )
    asserted_sensing = re.sub(
        r"\bno\b.{0,40}\b(?:can|will|may|would)\s+"
        r"(?:inspect|examine|look at|re-?inspect)(?:ing)?",
        "",
        asserted_sensing,
        flags=re.I,
    )
    if re.search(
        r"(?:inspect|examine|look at|re-?inspect).{0,45}(?:new|direct|pixel|frame|image|strip|visual|visible)",
        asserted_sensing,
        re.I,
    ):
        problems.append(f"turn_{turn_index}_meta_abstention_claims_new_visual_inspection")
    return problems


def _meta_observation_fact_problem(turn_index: int, reasoning: str) -> str | None:
    meta_reference = (
        r"(?:EA-VIDDIFF-COMPARE-SYMMETRY-RESIDUAL-ABSTENTION-\d{3}"
        r"|EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-\d{3}"
        r"|meta[-_ ](?:atom|evidence|abstention|result))"
    )
    false_order = (
        r"(?:order[-_ ]inconsisten\w*|order[-_ ]consisten(?:t|cy)"
        r"\s*[`*]*\s*(?:(?::|=|is|was|of)\s*)?[`*]*\s*false)"
    )
    if re.search(rf"{meta_reference}.{{0,220}}{false_order}", reasoning, re.I):
        return f"turn_{turn_index}_role_invariant_meta_described_as_order_inconsistent"
    return None


def _meta_evidence_scope_problem(turn_index: int, reasoning: str) -> str | None:
    meta_reference = (
        r"(?:EA-VIDDIFF-COMPARE[-_]SYMMETRY[-_]RESIDUAL[-_]ABSTENTION-\d{3}"
        r"|meta[-_ ](?:atom|evidence|abstention|result))"
    )
    all_fields = (
        r"(?:all\s+(?:six|6)\s+(?:required\s+)?(?:claim\s+)?fields"
        r"|the\s+(?:six|6)\s+required\s+(?:claim\s+)?fields"
        r"|all\s+required\s+(?:claim\s+)?fields|every\s+(?:claim\s+)?field)"
    )
    task_identity = (
        r"(?:EA-VIDDIFF-(?:ADAPTIVE-)?TASK-\d{3}"
        r"|(?:the\s+)?task[-_ ](?:identity[-_ ])?(?:atom|evidence))"
    )
    direct_meta_scope = re.search(
        rf"(?:{all_fields}.{{0,35}}\b(?:via|from|by|through)\b\s*(?:the\s+)?{meta_reference}"
        rf"|{meta_reference}.{{0,80}}\b(?:supports?|covers?|handles?)\b.{{0,50}}{all_fields}"
        rf"|{meta_reference}.{{0,100}}\b(?:support|evidence)\s+for\s+{all_fields})",
        reasoning,
        re.I,
    )
    if (
        re.search(meta_reference, reasoning, re.I)
        and re.search(all_fields, reasoning, re.I)
        and (direct_meta_scope or not re.search(task_identity, reasoning, re.I))
    ):
        return f"turn_{turn_index}_meta_evidence_overclaims_identity_field_support"
    return None


def _meta_claim_alignment_problem(
    turn_index: int,
    reasoning: str,
    selected_method: str,
) -> str | None:
    if selected_method == "compare_symmetry_residual_abstention":
        return None
    explicit_meta_rejection = re.search(
        r"(?:reject|weaker|defect|not suitable|not reliable|cannot use).{0,80}(?:meta|symmetry[-_ ]residual)",
        reasoning,
        re.I,
    ) or re.search(
        r"(?:meta|symmetry[-_ ]residual).{0,80}(?:reject|weaker|defect|not suitable|not reliable|cannot use)",
        reasoning,
        re.I,
    )
    if explicit_meta_rejection:
        return None
    return f"turn_{turn_index}_meta_reasoning_and_claim_arguments_disagree:{selected_method}"


def _tool_fields(arguments: dict[str, Any]) -> list[str]:
    if isinstance(arguments.get("fields"), list):
        return [str(item) for item in arguments["fields"]]
    if isinstance(arguments.get("binding_items"), list):
        return [str(item.get("field")) for item in arguments["binding_items"]]
    return []


def _support_ids(arguments: dict[str, Any]) -> set[str]:
    return {
        str(support_id).upper()
        for item in arguments.get("binding_items") or []
        for support_id in item.get("support_ids") or []
    }


def _measurement_tools(turns: list[dict[str, Any]]) -> list[str]:
    return [
        str(turn.get("generated_tool"))
        for turn in turns
        if turn.get("generated_tool")
        and turn.get("generated_tool") not in STATE_TOOLS
        and not str(turn.get("generated_tool", "")).startswith("inspect")
    ]


def _repair_profile(state: dict[str, Any]) -> tuple[str, set[str]]:
    claims = state.get("candidate_claims") or []
    if len(claims) != 1:
        return "invalid", set()
    fields = claims[0].get("fields") or {}
    damaged = {
        str(name)
        for name, field in fields.items()
        if field.get("evidence_status") in {"unsupported", "conflicted"}
    }
    if damaged:
        return "evidence_dropout", damaged
    decision_open = {
        str(name)
        for name, field in fields.items()
        if field.get("evidence_status") == "supported"
        and field.get("decision_status") == "open"
    }
    if decision_open:
        return "decision_reopen", decision_open
    return "invalid", set()


def _progressive_assessment_integrity(
    turns: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Verify that an assessment ledger is atomic and matches final adjudication."""

    assessment_turns = [
        turn
        for turn in turns
        if turn.get("generated_tool") == "assess_evidence_candidate"
    ]
    if not assessment_turns:
        return True, []
    reasons: list[str] = []
    adjudication_turns = [
        turn for turn in turns if turn.get("generated_tool") == "adjudicate_evidence"
    ]
    if len(adjudication_turns) != 1:
        return False, [
            f"progressive_assessment_requires_one_adjudication:{len(adjudication_turns)}"
        ]
    adjudication_turn = adjudication_turns[0]
    adjudication_index = int(adjudication_turn.get("turn", 0))
    if any(int(turn.get("turn", 0)) >= adjudication_index for turn in assessment_turns):
        reasons.append("candidate_assessment_not_before_adjudication")

    assessment_arguments = [turn.get("generated_arguments") or {} for turn in assessment_turns]
    assessed_ids = [str(arguments.get("evidence_id") or "") for arguments in assessment_arguments]
    if any(not evidence_id for evidence_id in assessed_ids):
        reasons.append("candidate_assessment_missing_evidence_id")
    if len(assessed_ids) != len(set(assessed_ids)):
        reasons.append("candidate_assessment_repeats_evidence_id")

    adjudication_arguments = adjudication_turn.get("generated_arguments") or {}
    selected = {str(item) for item in adjudication_arguments.get("selected_evidence_ids") or []}
    rejected = {
        str(item.get("evidence_id") or "")
        for item in adjudication_arguments.get("rejected_evidence") or []
    }
    if set(assessed_ids) != selected | rejected:
        reasons.append("candidate_assessment_partition_mismatch")
    dispositions = {
        str(arguments.get("evidence_id") or ""): str(arguments.get("disposition") or "")
        for arguments in assessment_arguments
    }
    if any(dispositions.get(evidence_id) != "retain" for evidence_id in selected):
        reasons.append("selected_evidence_not_retained_in_assessment")
    if any(dispositions.get(evidence_id) != "reject" for evidence_id in rejected):
        reasons.append("rejected_evidence_not_rejected_in_assessment")
    if any(
        arguments.get("disposition") == "reject"
        and not list(arguments.get("defect_codes") or [])
        for arguments in assessment_arguments
    ):
        reasons.append("rejected_assessment_has_no_visible_defect")
    return not reasons, reasons


def _dependency_ordering(
    mode: str,
    state: dict[str, Any],
    turns: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    tools = [str(turn.get("generated_tool")) for turn in turns]
    reasons: list[str] = []
    if not tools or tools[-1] != "stop":
        reasons.append("stop_is_not_the_last_action")
    if tools.count("stop") != 1:
        reasons.append("terminal_action_count_is_not_one")
    if mode == "clean":
        if not tools or not tools[0].startswith("inspect"):
            reasons.append("clean_episode_does_not_start_with_context_inspection")
        for required in ("create_candidate_claim", "bind_evidence", "set_field_decision"):
            if tools.count(required) != 1:
                reasons.append(f"clean_episode_requires_exactly_one:{required}")
        if not reasons:
            clean_order_ok = (
                tools.index("create_candidate_claim")
                < tools.index("bind_evidence")
                < tools.index("set_field_decision")
                < tools.index("stop")
            )
            if "adjudicate_evidence" in tools:
                clean_order_ok = clean_order_ok and (
                    tools.index("create_candidate_claim")
                    < tools.index("adjudicate_evidence")
                    < tools.index("bind_evidence")
                )
            assessment_indexes = [
                index
                for index, name in enumerate(tools)
                if name == "assess_evidence_candidate"
            ]
            if assessment_indexes:
                clean_order_ok = clean_order_ok and (
                    tools.index("create_candidate_claim")
                    < min(assessment_indexes)
                    <= max(assessment_indexes)
                    < tools.index("adjudicate_evidence")
                )
            if not clean_order_ok:
                reasons.append("clean_state_actions_violate_dependency_order")
    else:
        if "create_candidate_claim" in tools:
            reasons.append("repair_episode_unnecessarily_creates_a_new_claim")
        subtype, _target_fields = _repair_profile(state)
        if subtype == "evidence_dropout":
            assessment_tools = [
                "assess_evidence_candidate"
                for name in tools
                if name == "assess_evidence_candidate"
            ]
            expected_state = [
                "set_field_decision",
                *assessment_tools,
                *(
                    ["adjudicate_evidence"]
                    if "adjudicate_evidence" in tools
                    else []
                ),
                "bind_evidence",
                "set_field_decision",
                "stop",
            ]
        elif subtype == "decision_reopen":
            expected_state = ["set_field_decision", "stop"]
        else:
            expected_state = []
            reasons.append("repair_seed_has_no_repairable_field_slice")
        actual_state = [name for name in tools if name in STATE_TOOLS]
        if actual_state != expected_state:
            reasons.append(f"repair_state_action_order_is:{actual_state}")
        first_arguments = (turns[0].get("generated_arguments") or {}) if turns else {}
        expected_first_status = "open" if subtype == "evidence_dropout" else "committed"
        if first_arguments.get("decision_status") != expected_first_status:
            reasons.append(
                f"repair_first_decision_is_not_{expected_first_status}"
            )
        commit_arguments = next(
            (
                turn.get("generated_arguments") or {}
                for turn in turns
                if turn.get("generated_tool") == "set_field_decision"
                and (turn.get("generated_arguments") or {}).get("decision_status") == "committed"
            ),
            {},
        )
        if not commit_arguments:
            reasons.append("repair_does_not_recommit_restored_fields")
    return not reasons, reasons


def _repair_locality(
    mode: str,
    state: dict[str, Any],
    turns: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    if mode == "clean":
        return True, []
    claims = state.get("candidate_claims") or []
    if len(claims) != 1:
        return False, ["repair_seed_does_not_have_exactly_one_claim"]
    subtype, target_fields = _repair_profile(state)
    if subtype == "invalid":
        return False, ["repair_seed_has_no_repairable_field_slice"]
    decision_turns = [
        turn for turn in turns if turn.get("generated_tool") == "set_field_decision"
    ]
    commit_turn = next(
        (turn
        for turn in turns
        if turn.get("generated_tool") == "set_field_decision"
        and (turn.get("generated_arguments") or {}).get("decision_status") == "committed"
        ),
        None,
    )
    if commit_turn is None:
        return False, ["repair_has_no_commit_action"]
    if subtype == "decision_reopen":
        actual = {
            "commit": set(_tool_fields(commit_turn.get("generated_arguments") or {})),
        }
    else:
        bind_turn = next(
            (turn for turn in turns if turn.get("generated_tool") == "bind_evidence"),
            None,
        )
        if not decision_turns or bind_turn is None:
            return False, ["evidence_dropout_repair_is_missing_reopen_or_bind"]
        actual = {
            "reopen": set(_tool_fields(decision_turns[0].get("generated_arguments") or {})),
            "rebind": set(_tool_fields(bind_turn.get("generated_arguments") or {})),
            "recommit": set(_tool_fields(commit_turn.get("generated_arguments") or {})),
        }
    reasons = [
        f"{stage}_fields_do_not_equal_repair_slice:{sorted(values)}!={sorted(target_fields)}"
        for stage, values in actual.items()
        if values != target_fields
    ]
    return not reasons, reasons


def _audit_episode(
    mode: str,
    episode: dict[str, Any],
    seed_row: dict[str, Any],
    generation_limit: int,
) -> dict[str, Any]:
    state = _payload(seed_row)["current_harness_state"]
    turns = list(episode.get("turns") or [])
    visible_ids = {
        str(item).upper() for item in (state.get("evidence_store") or {})
    }
    visible_ids.update(str(item).upper() for item in state.get("dropped_evidence_ids") or [])
    unknown_citations: list[dict[str, Any]] = []
    created_at: dict[str, int] = {}
    cited_at: dict[str, list[int]] = {}
    bound_at: dict[str, list[int]] = {}
    claim_ids = {
        str(claim["claim_id"]) for claim in state.get("candidate_claims") or []
    }
    entity_problems: list[str] = []
    reasoning_problems: list[str] = []
    meta_abstention_seen = False
    for turn in turns:
        index = int(turn["turn"])
        reasoning = str(turn.get("reasoning") or "")
        cited = {item.upper() for item in EVIDENCE_PATTERN.findall(reasoning)}
        unknown = sorted(cited - visible_ids)
        if unknown:
            unknown_citations.append({"turn": index, "unknown_ids": unknown})
        for evidence_id in cited:
            cited_at.setdefault(evidence_id, []).append(index)
        arguments = turn.get("generated_arguments") or {}
        name = str(turn.get("generated_tool"))
        if name == "create_candidate_claim":
            new_claim = str(arguments.get("claim_id"))
            if not new_claim or new_claim in claim_ids:
                entity_problems.append(f"turn_{index}_claim_id_is_missing_or_duplicate")
            claim_ids.add(new_claim)
            selected_method = str(
                (arguments.get("field_values") or {}).get("measurement_method", "")
            )
            if meta_abstention_seen:
                alignment_problem = _meta_claim_alignment_problem(
                    index, reasoning, selected_method
                )
                if alignment_problem:
                    reasoning_problems.append(alignment_problem)
        if name in {
            "assess_evidence_candidate",
            "adjudicate_evidence",
            "bind_evidence",
            "set_field_decision",
        }:
            claim_id = str(arguments.get("claim_id"))
            if claim_id not in claim_ids:
                entity_problems.append(f"turn_{index}_references_unknown_claim:{claim_id}")
        for evidence_id in _support_ids(arguments):
            bound_at.setdefault(evidence_id, []).append(index)
        words = re.findall(r"\b\w+[\w'-]*\b", reasoning)
        minimum_words = _minimum_reasoning_words(name)
        if not minimum_words <= len(words) <= 120:
            reasoning_problems.append(f"turn_{index}_word_count:{len(words)}")
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", reasoning))
        if not 2 <= sentence_count <= 4:
            reasoning_problems.append(
                f"turn_{index}_sentence_count:{sentence_count}"
            )
        if not re.search(r"[.!?][\"'`)\]]?\s*$", reasoning):
            reasoning_problems.append(f"turn_{index}_missing_terminal_punctuation")
        measurement_uncertainty_named = name.startswith(
            ("compare_", "detect_", "localize_")
        ) and re.search(
            r"low[-_ ]confidence|confidence\s*(?::|=|is|of|\()?\s*0?\.[0-6]|"
            r"uncertain|disagree|inconsisten|abstain|independent\s+(?:check|evidence|"
            r"measurement|corroboration)|cross[-_ ]method|occlud|sparse|limitation|"
            r"unresolved",
            reasoning,
            re.I,
        )
        names_decision_gap = bool(
            re.search(
                r"gap|block|remaining|no observation|no available", reasoning, re.I
            )
            or (
                name == "stop"
                and re.search(
                    r"terminal\s+(?:validation|finalization)", reasoning, re.I
                )
            )
        )
        if not names_decision_gap and not measurement_uncertainty_named:
            reasoning_problems.append(f"turn_{index}_does_not_name_the_decision_gap")
        if re.search(r"\boracle\b|\bverifier\b|teacher target", reasoning, re.I):
            reasoning_problems.append(f"turn_{index}_mentions_hidden_evaluation_information")
        chosen_tool_problem = _chosen_measurement_tool_problem(name, reasoning)
        if chosen_tool_problem:
            reasoning_problems.append(f"turn_{index}_{chosen_tool_problem}")
        measurement_need_problem = _measurement_need_problem(name, reasoning)
        if measurement_need_problem:
            reasoning_problems.append(f"turn_{index}_{measurement_need_problem}")
        outcome_presupposition_problem = _measurement_outcome_presupposition_problem(
            name, reasoning
        )
        if outcome_presupposition_problem:
            reasoning_problems.append(
                f"turn_{index}_{outcome_presupposition_problem}"
            )
        bind_scope_problem = _bind_evidence_reasoning_problem(
            name, arguments, reasoning
        )
        if bind_scope_problem:
            reasoning_problems.append(f"turn_{index}_{bind_scope_problem}")
        sft_datum = turn.get("sft_datum") or {}
        visible_turn_state: dict[str, Any] = {}
        for message in sft_datum.get("messages") or []:
            if message.get("role") != "user":
                continue
            try:
                turn_payload = json.loads(str(message.get("content") or "{}"))
                task_payload = turn_payload.get("task") or {}
                visible_turn_state = (
                    turn_payload.get("current_harness_state")
                    or task_payload.get("current_harness_state")
                    or {}
                )
                visible_turn_state = {
                    **visible_turn_state,
                    "latest_environment_observation": (
                        turn_payload.get("latest_environment_observation")
                        or task_payload.get("latest_environment_observation")
                        or visible_turn_state.get("latest_environment_observation")
                    )
                    or {},
                }
            except (TypeError, ValueError):
                visible_turn_state = {}
            break
        reasoning_problems.extend(
            f"turn_{index}_{problem}"
            for problem in _visible_observation_fact_problems(reasoning, visible_turn_state)
        )
        reasoning_problems.extend(
            f"turn_{index}_{problem}"
            for problem in _eligible_tool_claim_problems(reasoning, visible_turn_state)
        )
        reasoning_problems.extend(
            f"turn_{index}_{problem}"
            for problem in _reasoning_process_artifact_problems(
                name, reasoning, visible_turn_state
            )
        )
        reasoning_problems.extend(
            f"turn_{index}_{problem}"
            for problem in _adjudication_reasoning_problems(
                name,
                arguments,
                reasoning,
                visible_state=visible_turn_state,
            )
        )
        reasoning_problems.extend(
            f"turn_{index}_{problem}"
            for problem in _evidence_assessment_reasoning_problems(
                name,
                arguments,
                reasoning,
                visible_state=visible_turn_state,
            )
        )
        latest_measurement_problem = _bind_latest_measurement_problem(
            name, arguments, reasoning, visible_turn_state
        )
        if latest_measurement_problem:
            reasoning_problems.append(
                f"turn_{index}_{latest_measurement_problem}"
            )
        if meta_abstention_seen:
            meta_fact_problem = _meta_observation_fact_problem(index, reasoning)
            if meta_fact_problem:
                reasoning_problems.append(meta_fact_problem)
            meta_scope_problem = _meta_evidence_scope_problem(index, reasoning)
            if meta_scope_problem:
                reasoning_problems.append(meta_scope_problem)
        if name == "compare_symmetry_residual_abstention":
            reasoning_problems.extend(
                _meta_abstention_reasoning_problems(
                    index,
                    reasoning,
                    cited,
                    set(cited_at),
                )
            )
            meta_abstention_seen = True
        for evidence_id in turn.get("environment_outcome", {}).get("created_evidence_ids") or []:
            upper = str(evidence_id).upper()
            created_at[upper] = index
            visible_ids.add(upper)

    unconsumed = []
    for evidence_id, created_turn in created_at.items():
        later_citations = [turn for turn in cited_at.get(evidence_id, []) if turn > created_turn]
        later_bindings = [turn for turn in bound_at.get(evidence_id, []) if turn > created_turn]
        if not later_citations and not later_bindings:
            unconsumed.append(evidence_id)

    execution_problems = [
        f"turn_{turn['turn']}_status:{turn.get('status')}"
        for turn in turns
        if turn.get("status") not in {"accepted", "executed"}
    ]
    assessment_integrity_ok, assessment_integrity_problems = (
        _progressive_assessment_integrity(turns)
    )
    if not assessment_integrity_ok:
        execution_problems.extend(assessment_integrity_problems)
    terminal_ok = bool(
        episode.get("terminal_success")
        and episode.get("final_terminal")
        and (episode.get("final_stop_decision") or {}).get("allowed")
    )
    ordering_ok, ordering_problems = _dependency_ordering(mode, state, turns)
    locality_ok, locality_problems = _repair_locality(mode, state, turns)
    measurements = _measurement_tools(turns)
    efficiency_problems = []
    if len(measurements) != len(set(measurements)):
        efficiency_problems.append("measurement_tool_repeated_within_episode")
    if mode == "repair":
        subtype, _target_fields = _repair_profile(state)
        adjudication_turns = sum(
            turn.get("generated_tool") == "adjudicate_evidence" for turn in turns
        )
        assessment_turns = sum(
            turn.get("generated_tool") == "assess_evidence_candidate"
            for turn in turns
        )
        expected_turns = (
            2
            if subtype == "decision_reopen"
            else 4 + len(measurements) + assessment_turns + adjudication_turns
        )
        if len(turns) != expected_turns:
            efficiency_problems.append(
                f"repair_action_count_is_not_minimal:{len(turns)}!={expected_turns}"
            )
    context_problems = []
    for turn in turns:
        if int(turn.get("prompt_tokens_kept", 0)) < int(turn.get("prompt_tokens_source", 0)):
            context_problems.append(f"turn_{turn['turn']}_prompt_truncated")
        if int(turn.get("generated_tokens", 0)) >= generation_limit:
            context_problems.append(f"turn_{turn['turn']}_generation_hit_limit")

    checks = {
        "terminal_integrity": (terminal_ok, [] if terminal_ok else ["terminal_validation_failed"]),
        "execution_integrity": (not execution_problems, execution_problems),
        "evidence_reference_integrity": (not unknown_citations, unknown_citations),
        "cross_turn_entity_consistency": (not entity_problems, entity_problems),
        "evidence_consumption": (not unconsumed, unconsumed),
        "mutation_locality_or_clean_scope": (locality_ok, locality_problems),
        "dependency_ordering": (ordering_ok, ordering_problems),
        "route_efficiency": (not efficiency_problems, efficiency_problems),
        "human_like_reasoning": (not reasoning_problems, reasoning_problems),
        "context_integrity": (not context_problems, context_problems),
    }
    scores = {
        name: WEIGHTS[name] if passed else 0
        for name, (passed, _reasons) in checks.items()
    }
    total = sum(scores.values())
    critical_pass = all(checks[name][0] for name in CRITICAL)
    accepted = critical_pass and total >= 98
    return {
        "dataset": episode["dataset"],
        "case_id": episode.get("case_id") or episode.get("case_name"),
        "seed_row_id": episode.get("seed_row_id"),
        "mode": mode,
        "repair_subtype": _repair_profile(state)[0] if mode == "repair" else None,
        "turns": len(turns),
        "score": total,
        "max_score": 100,
        "critical_checks_passed": critical_pass,
        "accepted_for_high_quality_sft": accepted,
        "dimension_scores": scores,
        "failures": {
            name: reasons for name, (passed, reasons) in checks.items() if not passed
        },
        "tool_sequence": episode.get("tool_sequence"),
    }


def _candidate_episode_selected(
    episode: dict[str, Any], accepted_candidates_only: bool
) -> bool:
    return not accepted_candidates_only or bool(
        ((episode.get("quality") or {}).get("accepted_for_high_quality_sft"))
    )


def audit_reports(
    report_paths: list[Path],
    sft_path: Path | list[Path],
    generation_limit: int,
    accepted_candidates_only: bool = False,
) -> dict[str, Any]:
    sft_paths = [sft_path] if isinstance(sft_path, Path) else list(sft_path)
    rows: dict[str, dict[str, Any]] = {}
    turn_rows_by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sft_paths:
        for item in _jsonl(path):
            metadata = item.get("metadata") or {}
            row_id = metadata.get("row_id")
            if row_id:
                rows[str(row_id)] = item
            source_seed_row_id = metadata.get("source_seed_row_id")
            if source_seed_row_id:
                turn_rows_by_seed[str(source_seed_row_id)].append(item)
                if int(metadata.get("turn_index") or 0) == 1:
                    rows.setdefault(str(source_seed_row_id), item)
    episodes = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audit_name = str(report.get("audit") or "")
        report_mode = report.get("episode_mode")
        if report_mode not in {"clean", "repair"}:
            if "repair" in audit_name:
                report_mode = "repair"
            elif audit_name == "student_closed_loop_three_dataset_canary":
                report_mode = "clean"
        for episode in report["episodes"]:
            if not _candidate_episode_selected(episode, accepted_candidates_only):
                continue
            mode = episode.get("mode") or report_mode
            if mode not in {"clean", "repair"}:
                raise ValueError(
                    f"episode lacks a clean/repair mode in {report_path}: {episode.get('case_name')}"
                )
            seed_row_id = episode.get("seed_row_id") or SEED_ROWS[mode][episode["dataset"]]
            seed = _seed_row_for_episode(episode, rows, str(seed_row_id))
            hydrated_episode = _hydrate_episode_from_sft(
                episode, turn_rows_by_seed.get(str(seed_row_id), [])
            )
            audited_episode = _audit_episode(
                mode, hydrated_episode, seed, generation_limit
            )
            audited_episode["source_report"] = str(report_path)
            episodes.append(audited_episode)
    accepted = sum(item["accepted_for_high_quality_sft"] for item in episodes)
    return {
        "audit": "unified_student_closed_loop_quality_v1",
        "teacher_target_used": False,
        "candidate_selection": (
            "report-marked accepted candidates, independently rescored"
            if accepted_candidates_only
            else "all report episodes"
        ),
        "rubric": {
            "weights": WEIGHTS,
            "critical_dimensions": sorted(CRITICAL),
            "acceptance_rule": "all critical dimensions pass and score >= 98/100",
            "quality_intent": (
                "Require executable terminal success, grounded cross-turn evidence and entity use, "
                "repair locality, dependency order, efficient routing, human-readable gap reasoning, "
                "and intact model context."
            ),
        },
        "summary": {
            "episodes": len(episodes),
            "accepted": accepted,
            "rejected": len(episodes) - accepted,
            "acceptance_rate": round(accepted / len(episodes), 4),
            "mean_score": round(sum(item["score"] for item in episodes) / len(episodes), 3),
            "minimum_score": min(item["score"] for item in episodes),
            "all_accepted": accepted == len(episodes),
        },
        "episodes": episodes,
    }


def _seed_row_for_episode(
    episode: dict[str, Any], rows: dict[str, dict[str, Any]], seed_row_id: str
) -> dict[str, Any]:
    """Use the original seed when available, otherwise the episode's first pre-action view."""
    if seed_row_id in rows:
        return rows[seed_row_id]
    turns = list(episode.get("turns") or [])
    embedded = (turns[0].get("sft_datum") or {}) if turns else {}
    if embedded.get("messages"):
        return embedded
    raise KeyError(
        f"missing seed row {seed_row_id!r} and no embedded first-turn sft_datum"
    )


def _state_from_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        payload = json.loads(str(message.get("content") or "{}"))
        task_payload = payload.get("task") or {}
        return (
            payload.get("current_harness_state")
            or task_payload.get("current_harness_state")
            or {}
        )
    return {}


def _hydrate_episode_from_sft(
    episode: dict[str, Any], turn_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Restore compact batch-report turns from the accepted native-tool SFT rows."""
    if episode.get("turns") or not turn_rows:
        return episode
    ordered = sorted(
        turn_rows,
        key=lambda row: int((row.get("metadata") or {}).get("turn_index") or 0),
    )
    turns: list[dict[str, Any]] = []
    for offset, row in enumerate(ordered):
        metadata = row.get("metadata") or {}
        assistant = next(
            (
                message
                for message in reversed(row.get("messages") or [])
                if message.get("role") == "assistant"
            ),
            {},
        )
        calls = assistant.get("tool_calls") or []
        function = ((calls[0] if calls else {}).get("function") or {})
        current_state = _state_from_sft_row(row)
        next_state = (
            _state_from_sft_row(ordered[offset + 1])
            if offset + 1 < len(ordered)
            else {}
        )
        current_ids = set((current_state.get("evidence_store") or {}).keys())
        next_ids = set((next_state.get("evidence_store") or {}).keys())
        created_ids = sorted(next_ids - current_ids)
        turns.append(
            {
                "turn": int(metadata.get("turn_index") or offset + 1),
                "status": "executed",
                "generated_tool": function.get("name"),
                "generated_arguments": function.get("arguments") or {},
                "reasoning": str(assistant.get("content") or ""),
                "environment_outcome": {
                    "status": "executed",
                    "created_evidence_ids": created_ids,
                },
                "prompt_tokens_source": 0,
                "prompt_tokens_kept": 0,
                "generated_tokens": 0,
                "sft_datum": row,
            }
        )
    hydrated = copy.deepcopy(episode)
    hydrated["turns"] = turns
    last_tool = turns[-1]["generated_tool"] if turns else None
    original_quality = episode.get("quality") or {}
    original_terminal_ok = bool(
        (original_quality.get("dimension_scores") or {}).get("terminal_integrity")
        == WEIGHTS["terminal_integrity"]
    )
    terminal_ok = last_tool == "stop" and original_terminal_ok
    hydrated["terminal_success"] = terminal_ok
    hydrated["final_terminal"] = terminal_ok
    hydrated["final_stop_decision"] = {"allowed": terminal_ok}
    return hydrated


def audit(clean_path: Path, repair_path: Path, sft_path: Path, generation_limit: int) -> dict[str, Any]:
    return audit_reports([clean_path, repair_path], sft_path, generation_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument(
        "--sft",
        action="append",
        type=Path,
        help="One or more accepted SFT files used to restore source seed rows.",
    )
    parser.add_argument("--generation-limit", type=int, default=384)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        help="Audit one or more mixed/expanded reports instead of --clean and --repair.",
    )
    parser.add_argument(
        "--accepted-candidates-only",
        action="store_true",
        help=(
            "Select only report-marked accepted candidates, then independently rescore them."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = (
        audit_reports(
            args.report,
            args.sft or [DEFAULT_SFT],
            args.generation_limit,
            accepted_candidates_only=args.accepted_candidates_only,
        )
        if args.report
        else audit(
            args.clean,
            args.repair,
            (args.sft or [DEFAULT_SFT])[0],
            args.generation_limit,
        )
    )
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
