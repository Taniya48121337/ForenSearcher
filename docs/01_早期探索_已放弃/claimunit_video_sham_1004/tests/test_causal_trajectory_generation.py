from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_trajectory_generation import (  # noqa: E402
    CAUSAL_SYSTEM_PROMPT,
    CausalPolicyViewTeacher,
    _latest_created_evidence_ids,
    _recent_reasoning,
    _reasoning_score,
    _style_frame_signature,
    classify_supervision,
    control_plane_exposure_audit,
    prospective_method_fit_audit,
    sanitize_teacher_transcript,
    sanitize_tool_schemas,
    visible_integrity_issues,
)
from llm_teacher_rollout import TeacherAction  # noqa: E402


def _action(index: int, reasoning: str) -> TeacherAction:
    return TeacherAction(
        response_id=f"response-{index}",
        call_id=f"call-{index}",
        tool_name="create_candidate_claim",
        reasoning=reasoning,
        params={"claim_id": "claim-1", "field_values": {"preferred_video": "a"}},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )


class CandidateTeacher:
    model = "fake-candidate-teacher"

    def __init__(self, actions: list[TeacherAction]) -> None:
        self.actions = list(actions)
        self.seen_inputs = []
        self.seen_tools = []

    def next_action(self, input_items, tools):
        self.seen_inputs.append(copy.deepcopy(input_items))
        self.seen_tools.append(copy.deepcopy(tools))
        return self.actions.pop(0)


def test_style_memory_reads_reasoning_from_responses_function_calls() -> None:
    notes = _recent_reasoning(
        [
            {
                "type": "function_call",
                "name": "inspect_pair_features",
                "arguments": json.dumps(
                    {"reasoning": "The pair identity is unknown, so inspect the visible headers first."}
                ),
            }
        ]
    )
    assert notes == ["The pair identity is unknown, so inspect the visible headers first."]


def test_style_signature_catches_relabelled_gap_template() -> None:
    first = (
        "The pair identity is visible. That matters because later localization must use the same pair, "
        "so inspection is warranted now rather than guessing."
    )
    second = (
        "A localization atom is now present. Because its values match the pair, claim creation is "
        "warranted now rather than another measurement."
    )
    assert {"warranted_frame", "because_connector", "rather_than_frame"}.issubset(
        _style_frame_signature(first)
    )
    action = _action(1, second)
    _score, failures = _reasoning_score(
        action,
        visible_evidence_ids=set(),
        eligible_count=1,
        recent_reasoning=[first],
    )
    assert any(item.startswith("repeated_discourse_frame:") for item in failures)


def test_exploratory_note_can_state_what_measurement_can_establish() -> None:
    action = TeacherAction(
        response_id="response-establish",
        call_id="call-establish",
        tool_name="compare_uniform_timeline",
        reasoning=(
            "The preferred-video direction is unresolved because no temporal measurement is visible yet. "
            "Uniform comparison can establish whether one role finishes the action earlier while preserving "
            "the alternative that their timing is effectively tied across the requested interval."
        ),
        params={"source_id": "A", "suspect_id": "B"},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )
    _score, failures = _reasoning_score(
        action,
        visible_evidence_ids=set(),
        eligible_count=2,
    )
    assert "prospective_discrimination_missing" not in failures


def test_method_fit_uses_visible_diagnostics_and_tool_mechanism_without_router() -> None:
    tools = [
        {
            "name": "localize_copy_segment_dp",
            "description": "Run the non-learned dynamic-programming temporal localizer on the real similarity map.",
        },
        {
            "name": "localize_copy_segment_axis_saliency",
            "description": "Bound broad speed-changed copies using robust row/column upper-tail saliency anchored by the strongest monotonic path.",
        },
    ]
    input_items = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "localizer_routing_diagnostics": {
                        "aspect_ratio": 1.2,
                        "dp_candidate_count": 1,
                        "stability_candidate_count": 1,
                    }
                }
            ),
        }
    ]
    action = TeacherAction(
        response_id="response-fit",
        call_id="call-fit",
        tool_name="localize_copy_segment_dp",
        reasoning=(
            "The visible aspect ratio is 1.2 and dp candidate count is one. Because a compact path remains "
            "plausible, dynamic programming can preserve temporal alignment, unlike "
            "localize_copy_segment_axis_saliency, which targets broad row and column saliency."
        ),
        params={"query_ref": "query", "reference_ref": "reference", "method": "DP"},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )
    audit = prospective_method_fit_audit(
        action, input_items=input_items, tools=tools
    )
    assert audit["score"] == 1.0
    assert audit["matched_diagnostic_keys"] == ["aspect_ratio", "dp_candidate_count"]
    assert audit["named_alternative_tools"] == [
        "localize_copy_segment_axis_saliency"
    ]
    assert audit["oracle_or_verifier_used"] is False


def test_generic_tool_mention_does_not_pass_method_fit() -> None:
    tools = [
        {
            "name": "localize_copy_segment_dp",
            "description": "Run dynamic-programming temporal localization on a similarity map.",
        },
        {
            "name": "localize_copy_segment_axis_saliency",
            "description": "Use row and column saliency to bound broad copied regions.",
        },
    ]
    action = TeacherAction(
        response_id="response-generic",
        call_id="call-generic",
        tool_name="localize_copy_segment_dp",
        reasoning="The relationship remains uncertain, so localize_copy_segment_dp can help answer it.",
        params={"query_ref": "query", "reference_ref": "reference", "method": "DP"},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )
    audit = prospective_method_fit_audit(
        action,
        input_items=[{"role": "user", "content": '{"aspect_ratio": 1.2}'}],
        tools=tools,
    )
    assert audit["score"] < 0.5
    assert "visible_diagnostic_grounded" in audit["failures"]
    assert "chosen_method_mechanism_grounded" in audit["failures"]


def test_keyword_stuffed_method_inventory_fails_counterfactual_tool_swap() -> None:
    tools = [
        {
            "name": "localize_copy_segment_dp",
            "description": "Run dynamic-programming temporal localization on a similarity map.",
        },
        {
            "name": "localize_copy_segment_axis_saliency",
            "description": "Use row and column upper-tail saliency to bound broad copied regions.",
        },
    ]
    action = TeacherAction(
        response_id="response-stuffed",
        call_id="call-stuffed",
        tool_name="localize_copy_segment_dp",
        reasoning=(
            "Given aspect ratio 1.2, the alternative mechanisms include dynamic programming temporal localization "
            "and row column upper tail saliency; either can test the visible similarity map diagnostics."
        ),
        params={"method": "DP"},
        raw_arguments="{}",
        usage={},
        replay_items=[],
    )
    audit = prospective_method_fit_audit(
        action,
        input_items=[{"role": "user", "content": '{"aspect_ratio": 1.2}'}],
        tools=tools,
    )
    assert audit["counterfactual_tool_swap_passed"] is False
    assert "counterfactual_tool_swap_specificity" in audit["failures"]
    assert audit["score"] < 1.0


def test_method_fit_allows_different_tools_when_each_has_visible_warrant() -> None:
    tools = [
        {
            "name": "localize_copy_segment_dp",
            "description": "Run dynamic-programming temporal localization on a similarity map.",
        },
        {
            "name": "localize_copy_segment_axis_saliency",
            "description": "Bound broad speed-changed copies using robust row and column upper-tail saliency.",
        },
    ]
    input_items = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "aspect_ratio": 1.2,
                    "dp_candidate_count": 1,
                    "stability_candidate_count": 4,
                }
            ),
        }
    ]
    actions = [
        TeacherAction(
            response_id="response-dp",
            call_id="call-dp",
            tool_name="localize_copy_segment_dp",
            reasoning=(
                "Given the dp candidate count of one, dynamic programming can test the compact temporal path, "
                "unlike row and column saliency, which would emphasize broad support."
            ),
            params={"method": "DP"}, raw_arguments="{}", usage={}, replay_items=[],
        ),
        TeacherAction(
            response_id="response-axis",
            call_id="call-axis",
            tool_name="localize_copy_segment_axis_saliency",
            reasoning=(
                "Given the stability candidate count of four, robust upper tail saliency can test broad row and "
                "column support, unlike dynamic programming, which follows a compact path."
            ),
            params={"method": "AXIS-SALIENCY"}, raw_arguments="{}", usage={}, replay_items=[],
        ),
    ]
    audits = [
        prospective_method_fit_audit(action, input_items=input_items, tools=tools)
        for action in actions
    ]
    assert [audit["score"] for audit in audits] == [1.0, 1.0]


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "create_candidate_claim",
            "description": "Create a claim; do not copy these arguments exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_values": {
                        "type": "object",
                        "properties": {
                            "preferred_video": {"type": "string", "enum": ["a"]},
                            "comparison_statement": {"type": "string"},
                        },
                    }
                },
            },
        },
        {
            "type": "function",
            "name": "set_field_decision",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision_status": {"type": "string", "enum": ["open", "committed"]}
                },
            },
        },
    ]


def _input() -> list[dict]:
    state = {
        "evidence_store": {
            "EA-TEST-001": {
                "evidence_id": "EA-TEST-001",
                "observation": {"preferred_video": "a", "visible_rationale": "A is faster"},
            }
        },
        "state_action_affordances": {
            "executable_binding_actions": [{"name": "bind_evidence", "arguments": {"x": 1}}]
        },
        "action_constraints": {
            "must_reopen_before_rebinding": ["preferred_video"]
        },
        "repair_restoration_candidates": [{"tool": "compare_uniform_timeline"}],
        "evidence_adjudication": {
            "required": True,
            "requirement": {
                "claim_id": "claim-1",
                "target_fields": ["preferred_video"],
                "competing_evidence_ids": ["EA-TEST-001", "EA-TEST-002"],
                "compatible_evidence_ids_by_field": {
                    "preferred_video": ["EA-TEST-001"]
                },
            },
        },
    }
    return [
        {"role": "system", "content": "Use the only eligible tool and exact repair recipe."},
        {"role": "user", "content": json.dumps({"harness_state": state})},
    ]


def test_causal_view_removes_compiled_actions_and_repair_recipe() -> None:
    revised = sanitize_teacher_transcript(_input())
    text = json.dumps(revised)
    assert "executable_binding_actions" not in text
    assert "must_reopen_before_rebinding" not in text
    assert "repair_restoration_candidates" not in text
    assert "compatible_evidence_ids_by_field" not in text
    assert "competing_evidence_ids" in text
    assert "EA-TEST-001" in text
    assert "A is faster" in text


def test_singleton_semantic_enum_is_removed_but_control_domain_remains() -> None:
    revised = sanitize_tool_schemas(_tools())
    claim = revised[0]["parameters"]["properties"]["field_values"]["properties"]
    assert "enum" not in claim["preferred_video"]
    decision = revised[1]["parameters"]["properties"]["decision_status"]
    assert decision["enum"] == ["open", "committed"]
    assert "copy these arguments exactly" not in revised[0]["description"].lower()


def test_sanitized_view_has_no_materializable_control_answer() -> None:
    audit = control_plane_exposure_audit(
        sanitize_teacher_transcript(_input()), sanitize_tool_schemas(_tools())
    )
    assert audit["compiled_semantic_actions_visible"] is False
    assert audit["repair_recipe_visible"] is False
    assert audit["singleton_target_values_visible"] is False
    assert audit["full_target_call_materializable"] is False
    assert audit["semantic_measurement_choice_count"] == 0
    assert audit["categorical_router_answer_visible"] is False


def test_choice_surface_audit_counts_real_measurement_tools_and_router_answers() -> None:
    audit = control_plane_exposure_audit(
        [{"role": "user", "content": json.dumps({"primary_modality": "audio"})}],
        [
            {"name": "inspect_media", "parameters": {"type": "object"}},
            {"name": "compare_audio_tracks", "parameters": {"type": "object"}},
            {"name": "detect_temporal_edit", "parameters": {"type": "object"}},
            {"name": "create_candidate_claim", "parameters": {"type": "object"}},
        ],
    )
    assert audit["semantic_measurement_tool_names"] == [
        "compare_audio_tracks",
        "detect_temporal_edit",
    ]
    assert audit["semantic_measurement_choice_count"] == 2
    assert audit["categorical_router_answer_visible"] is True


def test_rejected_interval_atom_does_not_overinvalidate_unrelated_bound_fields() -> None:
    state = {
        "task_contract": {
            "contract_id": "TASK-1",
            "visible_assets": {"comparison_scope_s": [0.0, 5.0]},
        },
        "evidence_store": {
            "EA-BAD": {
                "observation": {
                    "affected_source_interval_s": [7.0, 8.0],
                    "material_difference_detected": True,
                },
                "provenance": {"task_contract_id": "TASK-1"},
            },
            "EA-GOOD": {
                "observation": {"affected_source_interval_s": [0.0, 5.0]},
                "provenance": {"task_contract_id": "TASK-1"},
            },
        },
        "candidate_claims": [
            {
                "claim_id": "CC-1",
                "fields": {
                    "claim_family": {"value": "paired_media_difference"},
                    "affected_source_interval_s": {"value": [0.0, 5.0]},
                },
            }
        ],
        "field_bindings": {
            "CC-1": {
                "claim_family": {"support_ids": ["EA-BAD"]},
                "affected_source_interval_s": {"support_ids": ["EA-GOOD"]},
            }
        },
        "evidence_adjudications": {
            "CC-1": {
                "target_fields": ["affected_source_interval_s"],
                "selected_evidence_ids": ["EA-GOOD"],
                "rejected_evidence": [
                    {"evidence_id": "EA-BAD", "defect_codes": ["claim_value_conflict"]}
                ],
            }
        },
    }
    items = [{"role": "user", "content": json.dumps({"harness_state": state})}]
    assert visible_integrity_issues(items) == []


def test_best_of_n_prefers_grounded_prospective_decision_note() -> None:
    generic = _action(
        1,
        "This is generally the next action and it should make progress because the current state needs work.",
    )
    grounded = _action(
        2,
        "EA-TEST-001 reports that A is faster, but the current claim is still missing and the alternative of no "
        "meaningful difference remains unresolved. Creating the candidate now can record the measured direction "
        "without committing it; if later scope checks disagree, the evidence can still be rejected before binding.",
    )
    delegate = CandidateTeacher([generic, grounded])
    teacher = CausalPolicyViewTeacher(delegate, candidates_per_turn=2)
    selected = teacher.next_action(_input(), _tools())
    assert selected.response_id == "response-2"
    assert teacher.audit_log[0]["selected_candidate_index"] == 1
    assert teacher.audit_log[0]["selected_score"] > teacher.audit_log[0]["candidate_scores"][0]
    sent = json.dumps(delegate.seen_inputs)
    assert "executable_binding_actions" not in sent
    assert "must_reopen_before_rebinding" not in sent


def test_best_of_n_stops_early_when_first_candidate_is_strictly_grounded() -> None:
    grounded = _action(
        3,
        "EA-TEST-001 currently favors A, while the alternative of no meaningful difference remains unresolved. "
        "Creating a candidate claim can record this measured direction without committing it; if the later scope "
        "or identity check conflicts, the atom can still be rejected before any binding is accepted.",
    )
    unused = _action(4, "This extra candidate should never be requested because the first one already passes.")
    delegate = CandidateTeacher([grounded, unused])
    teacher = CausalPolicyViewTeacher(delegate, candidates_per_turn=3, early_accept_score=0.875)
    selected = teacher.next_action(_input(), _tools())
    assert selected.response_id == "response-3"
    assert len(delegate.seen_inputs) == 1
    assert teacher.audit_log[0]["early_stopped"] is True
    assert teacher.audit_log[0]["candidates_attempted"] == 1


def test_singleton_materialized_turn_is_low_weight_control_supervision() -> None:
    action = _action(1, "A sufficiently long decision note that explains the next ordinary state transition.")
    result = classify_supervision(
        eligible_tool_count=1,
        action=action,
        exposure={"full_target_call_materializable": True},
        evidence_ids_in_reasoning=0,
    )
    assert result["class"] == "evidence_interpretation"
    # Claim construction stays semantic even when the interface leaks it; the
    # exposure is rejected by UTQR-7 rather than silently relabeled bookkeeping.
    assert result["training_weight"] == 1.0

    stop = copy.deepcopy(action)
    stop.tool_name = "stop"
    result = classify_supervision(
        eligible_tool_count=1,
        action=stop,
        exposure={"full_target_call_materializable": True},
        evidence_ids_in_reasoning=0,
    )
    assert result["class"] == "control_only"
    assert result["training_weight"] == 0.2
    assert result["subtype"] == "terminal_verification"


def test_visible_claim_evidence_conflict_blocks_stop_candidate() -> None:
    state = {
        "task_contract": {
            "contract_id": "TASK-1",
            "visible_assets": {"comparison_scope_s": [0.0, 5.0]},
        },
        "candidate_claims": [
            {
                "claim_id": "CL-1",
                "fields": {
                    "affected_source_interval_s": {
                        "value": [0.0, 5.0],
                        "decision_status": "committed",
                    }
                },
            }
        ],
        "field_bindings": {
            "CL-1": {
                "affected_source_interval_s": {"support_ids": ["EA-1"]}
            }
        },
        "evidence_store": {
            "EA-1": {
                "observation": {"affected_source_interval_s": [7.0, 8.0]},
                "provenance": {"task_contract_id": "TASK-1"},
            }
        },
    }
    input_items = [{"role": "user", "content": json.dumps({"harness_state": state})}]
    issues = visible_integrity_issues(input_items)
    assert any("claim_evidence_conflict" in issue for issue in issues)
    assert any("outside_scope" in issue for issue in issues)

    stop = _action(
        9,
        "EA-1 is active and the report looks complete, while the only unresolved question is terminal status. "
        "Stopping can test whether the current obligations are closed; if not, the harness would reject it and "
        "the claim could be revisited.",
    )
    stop.tool_name = "stop"
    score, failures = _reasoning_score(
        stop,
        visible_evidence_ids={"EA-1"},
        eligible_count=2,
        integrity_issues=issues,
    )
    assert "semantic_commit_with_visible_integrity_issue" in failures
    assert score < 1.0
    assert "Immediately before stop" in CAUSAL_SYSTEM_PROMPT


def test_state_transition_requires_evidence_consequence_not_fake_prospective_language() -> None:
    action = _action(
        10,
        "EA-TEST-001 supports the measured preferred-video field and matches the current task identity and scope. "
        "Creating the candidate records that supported value while leaving every field open for later binding and "
        "commitment, so no independent invariant is changed.",
    )
    score, failures = _reasoning_score(
        action,
        visible_evidence_ids={"EA-TEST-001"},
        eligible_count=2,
    )
    assert "prospective_discrimination_missing" not in failures
    assert "state_consequence_missing" not in failures
    assert score > 0.85


def test_latest_tool_observation_requires_named_atom_and_semantic_update() -> None:
    transcript = [
        {
            "type": "function_call_output",
            "output": json.dumps(
                {
                    "status": "executed",
                    "created_evidence_ids": ["EA-NEW-001"],
                    "evidence_atoms": [{"evidence_id": "EA-NEW-001"}],
                }
            ),
        }
    ]
    assert _latest_created_evidence_ids(transcript) == {"EA-NEW-001"}
    action = _action(
        11,
        "EA-OLD-001 supports a prior field. Creating the claim follows from the current state and preserves scope.",
    )
    _score, failures = _reasoning_score(
        action,
        visible_evidence_ids={"EA-OLD-001", "EA-NEW-001"},
        latest_observation_evidence_ids={"EA-NEW-001"},
        eligible_count=2,
    )
    assert "does_not_name_newly_created_evidence" in failures

    action.reasoning = (
        "EA-NEW-001 reports a localized temporal difference and supports the target interval fields. "
        "Creating the claim records that measured result while preserving the current identity and scope."
    )
    _score, failures = _reasoning_score(
        action,
        visible_evidence_ids={"EA-OLD-001", "EA-NEW-001"},
        latest_observation_evidence_ids={"EA-NEW-001"},
        eligible_count=2,
    )
    assert "does_not_name_newly_created_evidence" not in failures
    assert "new_observation_semantic_update_missing" not in failures
