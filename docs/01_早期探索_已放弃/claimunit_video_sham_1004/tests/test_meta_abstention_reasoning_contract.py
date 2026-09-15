import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_student_closed_loop_quality import (
    _meta_abstention_reasoning_problems,
    _meta_claim_alignment_problem,
    _meta_evidence_scope_problem,
    _meta_observation_fact_problem,
)
from run_student_closed_loop_canary import (
    _bind_evidence_reasoning_problem,
    _bind_latest_measurement_problem,
    _eligibility_retry_feedback,
    _eligible_tool_claim_problems,
    _generation_hit_limit_problem,
    _generation_retry_feedback,
    _chosen_measurement_tool_problem,
    _evidence_assessment_reasoning_problems,
    _measurement_need_problem,
    _measurement_outcome_presupposition_problem,
    _next_action_lock_tool,
    _preexecution_reasoning_problems,
    _reasoning_process_artifact_problems,
    _schema_retry_feedback,
    _schema_valid,
    _visible_observation_fact_problems,
)


GLOBAL_ID = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
ACTION_ID = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
META_ID = "EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-001"


def test_accepts_explicit_two_route_aggregation() -> None:
    reasoning = (
        f"The two already executed routes {GLOBAL_ID} and {ACTION_ID} both return c and fail "
        "A/B-swap consistency. The remaining gap is whether their independent instability "
        "supports a role-invariant abstention, so this tool aggregates those prior measurements "
        "without inspecting new pixels."
    )
    assert _meta_abstention_reasoning_problems(5, reasoning, {GLOBAL_ID, ACTION_ID}) == []


def test_rejects_claim_of_new_visual_inspection() -> None:
    reasoning = (
        f"The latest observations are {GLOBAL_ID} and {ACTION_ID}. The blocking gap is a stable "
        "direction, so the tool will inspect the visible motion residuals directly."
    )
    problems = _meta_abstention_reasoning_problems(5, reasoning, {GLOBAL_ID, ACTION_ID})
    assert "turn_5_meta_abstention_not_described_as_aggregation" in problems
    assert "turn_5_meta_abstention_claims_new_visual_inspection" in problems


def test_rejects_missing_source_citation() -> None:
    reasoning = (
        f"The prior route {GLOBAL_ID} is unstable. The remaining gap is whether the two "
        "measurements support abstention, so this tool aggregates them without new sensing."
    )


def test_accepts_explicitly_negated_new_pixel_inspection() -> None:
    reasoning = (
        f"The already executed symmetric routes {GLOBAL_ID} and {ACTION_ID} remain unstable. "
        "The blocking gap is a role-invariant result, and no later measurement can inspect new "
        "pixels because this meta action only combines prior observations without re-inspecting "
        "video frames."
    )
    assert _meta_abstention_reasoning_problems(
        5, reasoning, {GLOBAL_ID, ACTION_ID}, {GLOBAL_ID, ACTION_ID}
    ) == []
    problems = _meta_abstention_reasoning_problems(5, reasoning, {GLOBAL_ID})
    assert any("missing_source_citation" in problem for problem in problems)


def test_accepts_prior_exact_grounding_plus_current_method_name() -> None:
    reasoning = (
        f"The latest global-motion observation {GLOBAL_ID} and the earlier action-refined atom "
        "both fail swap consistency. The blocking gap is stable direction, so this tool "
        "aggregates the two prior measurements without inspecting new pixels."
    )
    problems = _meta_abstention_reasoning_problems(
        5,
        reasoning,
        {GLOBAL_ID},
        {GLOBAL_ID, ACTION_ID},
    )
    assert problems == []


def test_accepts_prior_exact_grounding_plus_unambiguous_fixed_pair_reference() -> None:
    reasoning = (
        f"The latest observation {GLOBAL_ID} and the other active measurements disagree. "
        "The blocking gap is stable direction, so this tool aggregates both fixed symmetric "
        "source routes without inspecting new pixels."
    )
    assert _meta_abstention_reasoning_problems(
        5,
        reasoning,
        {GLOBAL_ID},
        {GLOBAL_ID, ACTION_ID},
    ) == []


def test_accepts_plural_already_executed_symmetric_routes_reference() -> None:
    reasoning = (
        f"The latest observation {GLOBAL_ID} and the other active measurements disagree. "
        "The blocking gap is stable direction, so this tool aggregates the already executed "
        "symmetric routes without inspecting new pixels."
    )
    assert _meta_abstention_reasoning_problems(
        5,
        reasoning,
        {GLOBAL_ID},
        {GLOBAL_ID, ACTION_ID},
    ) == []


def test_rejects_meta_call_that_omits_no_new_sensing_statement() -> None:
    reasoning = (
        f"The two routes {GLOBAL_ID} and {ACTION_ID} disagree. The blocking gap is stable "
        "direction, so this tool aggregates those prior measurements."
    )
    problems = _meta_abstention_reasoning_problems(
        5, reasoning, {GLOBAL_ID, ACTION_ID}
    )
    assert "turn_5_meta_abstention_omits_no_new_sensing_contract" in problems


def test_rejects_meta_reasoning_with_older_source_arguments() -> None:
    problem = _meta_claim_alignment_problem(
        6,
        "The meta atom is the strongest evidence and resolves the remaining gap.",
        "compare_action_refined_symmetric",
    )
    assert problem == (
        "turn_6_meta_reasoning_and_claim_arguments_disagree:"
        "compare_action_refined_symmetric"
    )


def test_allows_explicit_visible_rejection_of_meta_atom() -> None:
    assert (
        _meta_claim_alignment_problem(
            6,
            "The symmetry-residual meta atom is not reliable because its retained interval is outside the task scope.",
            "compare_uniform_timeline",
        )
        is None
    )


def test_rejects_order_inconsistency_attributed_to_role_invariant_meta() -> None:
    assert _meta_observation_fact_problem(
        6,
        "EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-001 is the strongest meta evidence with order_consistent: false.",
    ) == "turn_6_role_invariant_meta_described_as_order_inconsistent"


def test_rejects_hyphenated_order_consistent_false_attributed_to_meta() -> None:
    reasoning = (
        f"The latest observation created {META_ID}, which is the strongest visible support "
        "and is order-consistent `false`. The next step creates the claim from this meta atom."
    )
    assert _meta_observation_fact_problem(6, reasoning) == (
        "turn_6_role_invariant_meta_described_as_order_inconsistent"
    )
    assert "role_invariant_meta_described_as_order_inconsistent" in (
        _preexecution_reasoning_problems(
            reasoning,
            "create_candidate_claim",
            {"field_values": {"measurement_method": "compare_symmetry_residual_abstention"}},
            meta_visible=True,
        )
    )


def test_allows_source_instability_contrasted_with_consistent_meta() -> None:
    assert _meta_observation_fact_problem(
        6,
        "The source routes are order-inconsistent, while the meta result is role-invariant and order_consistent: true.",
    ) is None


def test_rejects_meta_atom_claimed_as_supporting_all_six_fields() -> None:
    reasoning = (
        f"The latest state shows all six claim fields are supported by {META_ID}. "
        "The only remaining gap is commitment, so the next action commits them."
    )
    assert _meta_evidence_scope_problem(8, reasoning) == (
        "turn_8_meta_evidence_overclaims_identity_field_support"
    )
    assert "meta_evidence_overclaims_identity_field_support" in (
        _preexecution_reasoning_problems(
            reasoning,
            "set_field_decision",
            {"claim_id": "claim_1", "field_decisions": []},
            meta_visible=True,
        )
    )


def test_allows_all_six_fields_when_task_and_meta_atoms_are_both_named() -> None:
    reasoning = (
        f"All six claim fields are supported jointly by EA-VIDDIFF-ADAPTIVE-TASK-001 and {META_ID}. "
        "The remaining gap is commitment, so the next action commits them."
    )
    assert _meta_evidence_scope_problem(8, reasoning) is None


def test_allows_unambiguous_task_atom_reference_after_grounding() -> None:
    reasoning = (
        f"All six claim fields are supported, with {META_ID} covering the comparative fields "
        "and the task atom supporting sample_key and comparison_statement. The remaining gap "
        "is terminal validation, so stop is the correct action."
    )
    assert _meta_evidence_scope_problem(9, reasoning) is None


def test_rejects_direct_all_six_via_meta_even_if_task_is_mentioned_later() -> None:
    reasoning = (
        f"All six claim fields are supported via {META_ID}, and the only remaining gap is "
        "commitment. The meta atom supports the same values as the prior task-identity evidence."
    )
    assert _meta_evidence_scope_problem(8, reasoning) == (
        "turn_8_meta_evidence_overclaims_identity_field_support"
    )
    assert "meta_evidence_overclaims_identity_field_support" in (
        _preexecution_reasoning_problems(
            reasoning,
            "set_field_decision",
            {"claim_id": "claim_1", "field_decisions": []},
            meta_visible=True,
        )
    )


def test_allows_all_six_by_active_evidence_then_two_sources_enumerated() -> None:
    reasoning = (
        f"All six claim fields are committed and supported by active evidence, including {META_ID} "
        "for the comparative fields and EA-VIDDIFF-ADAPTIVE-TASK-001 for the identity fields. "
        "The remaining gap is terminal validation."
    )
    assert _meta_evidence_scope_problem(9, reasoning) is None


def test_rejects_meta_called_support_for_all_six_even_if_task_is_added_later() -> None:
    reasoning = (
        f"{META_ID} is the strongest active support for all six required fields, including the "
        "comparative fields, while EA-VIDDIFF-ADAPTIVE-TASK-001 covers identity. The blocking "
        "gap is binding."
    )
    assert _meta_evidence_scope_problem(7, reasoning) == (
        "turn_7_meta_evidence_overclaims_identity_field_support"
    )
    assert "meta_evidence_overclaims_identity_field_support" in (
        _preexecution_reasoning_problems(
            reasoning,
            "bind_evidence",
            {"claim_id": "claim_1", "binding_items": []},
            meta_visible=True,
        )
    )


def test_preexecution_rejects_one_long_sentence_before_tool_execution() -> None:
    reasoning = (
        "The latest atom is visible and the blocking gap is still unresolved, so this action "
        "uses the only eligible comparison method to produce grounded evidence for every "
        "remaining claim field before any binding or commitment can occur."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "compare_uniform_timeline",
        {"video_a_ref": "video_a"},
        meta_visible=False,
    )
    assert "reasoning_sentence_count:1" in problems


def test_preexecution_accepts_two_sentence_grounded_reasoning() -> None:
    reasoning = (
        "The latest atom shows only task identity, so no comparative evidence is available yet. "
        "The blocking gap is the preferred video and intervals, and the uniform-timeline method is "
        "the best next measurement because it can inspect them before claim creation while preserving "
        "the visible comparison statement and avoiding unsupported field commitments."
    )
    assert _preexecution_reasoning_problems(
        reasoning,
        "compare_uniform_timeline",
        {"video_a_ref": "video_a"},
        meta_visible=False,
    ) == []


def test_measurement_uncertainty_need_not_use_literal_gap_keyword() -> None:
    reasoning = (
        "The latest active atom reports an abstaining direction with confidence=0.495 and "
        "order inconsistency, so its visual conclusion remains uncertain. "
        "compare_global_motion_symmetric is the best next route because it tests whether "
        "a whole-video motion direction survives both A/B role orders without forcing a winner."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "compare_global_motion_symmetric",
        {"video_a_ref": "video_a"},
        meta_visible=False,
    )
    assert "reasoning_does_not_name_decision_gap" not in problems


def test_preexecution_rejects_truncated_reasoning_even_when_sentence_count_passes() -> None:
    reasoning = (
        "The latest atom leaves a comparison gap, so the visible motion route is still required. "
        "I will run compare_uniform_timeline to obtain a role-symmetric observation and determine whether"
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "compare_uniform_timeline",
        {"video_a_ref": "video_a"},
        meta_visible=False,
    )
    assert "reasoning_missing_terminal_punctuation" in problems
    assert "Finish the final reasoning sentence" in _generation_retry_feedback(problems)


def test_schema_valid_accepts_required_argument_and_rejects_unknown_argument() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_comparison_task",
                "parameters": {
                    "type": "object",
                    "properties": {"sample_ref": {"type": "string"}},
                    "required": ["sample_ref"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert _schema_valid(tools, "inspect_comparison_task", {"sample_ref": "current_sample"})
    assert not _schema_valid(
        tools,
        "inspect_comparison_task",
        {"sample_ref": "current_sample", "oracle": "hidden"},
    )


def test_schema_retry_feedback_names_native_shape_errors() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bind_evidence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "binding_items": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["claim_id", "binding_items"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    feedback = _schema_retry_feedback(
        tools,
        "bind_evidence",
        {"claim_id": "claim-1", "field_bindings": []},
    )
    assert "'binding_items' is a required property" in feedback
    assert "field_bindings" in feedback
    assert "allowed top-level keys" in feedback
    assert "Rename the top-level key `field_bindings` to exactly `binding_items`" in feedback
    assert '"binding_items"' in feedback
    assert "legacy or extra keys" in feedback


def test_eligibility_feedback_blocks_premature_binding_and_names_next_tools() -> None:
    feedback = _eligibility_retry_feedback(
        "bind_evidence",
        ["compare_global_motion_symmetric", "compare_action_refined_symmetric"],
    )
    assert "`bind_evidence` is not executable" in feedback
    assert "compare_global_motion_symmetric" in feedback
    assert "compare_action_refined_symmetric" in feedback
    assert "remaining measurement or dependency step" in feedback


def test_retry_feedback_explains_meta_scope_and_generation_limit() -> None:
    feedback = _generation_retry_feedback(
        ["meta_evidence_overclaims_identity_field_support", "generation_hit_limit"]
    )
    assert "task atom for sample_key/comparison_statement" in feedback
    assert "meta atom only for preferred_video" in feedback
    assert "45-65 words" in feedback
    assert "meta_evidence_overclaims_identity_field_support" not in feedback
    assert "generation_hit_limit" not in feedback
    method_feedback = _generation_retry_feedback(
        ["chosen_measurement_tool_not_justified:compare_uniform_timeline"]
    )
    assert "`compare_uniform_timeline`" in method_feedback


def test_fact_error_prevents_action_lock_on_retry() -> None:
    tool = "compare_action_refined_symmetric"
    surface_only = [f"chosen_measurement_tool_not_justified:{tool}"]
    assert _next_action_lock_tool(
        None,
        tool,
        tool,
        parseable=True,
        eligible_call=True,
        problems=surface_only,
    ) == tool
    factual_error = [
        *surface_only,
        "visible_atom_active_status_mismatch:EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-002:true",
    ]
    assert _next_action_lock_tool(
        None,
        tool,
        tool,
        parseable=True,
        eligible_call=True,
        problems=factual_error,
    ) is None
    assert _next_action_lock_tool(
        tool,
        tool,
        tool,
        parseable=True,
        eligible_call=True,
        problems=factual_error,
    ) is None


def test_rejects_only_eligible_measurement_wording_when_multiple_tools_exist() -> None:
    state = {
        "eligible_tools": [
            "compare_uniform_timeline",
            "compare_action_refined_symmetric",
        ]
    }
    for wording in (
        "compare_action_refined_symmetric is the only eligible measurement tool.",
        "compare_action_refined_symmetric is the only remaining eligible measurement method.",
    ):
        assert _eligible_tool_claim_problems(wording, state) == [
            "eligible_tool_count_mismatch:said_only_one!=2"
        ]


def test_complete_parseable_call_is_not_rejected_only_for_hitting_token_cap() -> None:
    assert _generation_hit_limit_problem(
        192,
        192,
        raw_call_parseable=True,
        arguments_materialized=False,
    ) is None
    assert _generation_hit_limit_problem(
        192,
        192,
        raw_call_parseable=False,
        arguments_materialized=False,
    ) == "generation_hit_limit"


def test_measurement_rejects_binding_only_gap_rationale() -> None:
    tool = "compare_action_refined_symmetric"
    bad = (
        "The new global-motion atom is active. The only remaining step is to rebind it, "
        f"so I will call {tool}."
    )
    assert _measurement_need_problem(tool, bad) == (
        f"measurement_action_denies_need_for_measurement:{tool}"
    )
    good = (
        "The global-motion result is active, but cross-method disagreement remains. "
        f"{tool} tests predicate-relevant local behavior in both A/B orders."
    )
    assert _measurement_need_problem(tool, good) is None
    feedback = _generation_retry_feedback(
        [f"measurement_action_denies_need_for_measurement:{tool}"]
    )
    assert "unresolved observation gap" in feedback


def test_measurement_reasoning_must_not_presuppose_order_consistent_result() -> None:
    tool = "compare_action_refined_symmetric"
    for bad in (
        "I use the tool to establish a fresh, order-consistent direction.",
        "This measurement will create order-consistent support for the open fields.",
    ):
        assert _measurement_outcome_presupposition_problem(tool, bad) == (
            f"measurement_result_presupposed_before_execution:{tool}"
        )
    for good in (
        "I use the tool to test whether the direction is order-consistent.",
        "If it reports order-consistent support, the atom may become bindable.",
    ):
        assert _measurement_outcome_presupposition_problem(tool, good) is None
    feedback = _generation_retry_feedback(
        [f"measurement_result_presupposed_before_execution:{tool}"]
    )
    assert "whether/if language" in feedback


def test_rejects_denial_of_intervals_present_in_latest_visible_atom() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "observation": {
                    "preferred_video": "c",
                    "video_a_interval_s": [0.0, 4.3],
                    "video_b_interval_s": [0.0, 4.2],
                    "order_consistent": False,
                }
            }
        },
    }
    reasoning = (
        f"The latest measurement {evidence_id} lacks visible interval evidence despite "
        "preferred_video: c and order_consistent: false. The blocking gap requires another method."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_interval_evidence_denied:{evidence_id}"
    ]


def test_rejects_latest_atom_preferred_video_and_order_mismatches() -> None:
    evidence_id = META_ID
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "observation": {
                    "preferred_video": "c",
                    "video_a_interval_s": [0.0, 4.3],
                    "video_b_interval_s": [0.0, 4.2],
                    "order_consistent": True,
                }
            }
        },
    }
    problems = _visible_observation_fact_problems(
        f"{evidence_id} reports preferred_video: a and order-consistent `false`.",
        state,
    )
    assert problems == [
        f"latest_atom_preferred_video_mismatch:{evidence_id}:a!=c",
        f"latest_atom_order_consistency_mismatch:{evidence_id}:false!=true",
    ]
    natural_preference = _visible_observation_fact_problems(
        f"{evidence_id} established a visible preference for video B. The gap remains open.",
        state,
    )
    assert natural_preference == [
        f"latest_atom_preferred_video_mismatch:{evidence_id}:b!=c"
    ]


def test_rejects_denial_that_latest_measurement_atom_is_active() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": ["preferred_video", "video_a_interval_s"],
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "video_a_interval_s": [0.0, 4.3],
                    "video_b_interval_s": [0.0, 4.2],
                    "order_consistent": False,
                },
            }
        },
    }
    reasoning = (
        f"The latest atom {evidence_id} gives preferred_video: c and valid intervals. "
        "The blocking gap remains because there is no active measurement evidence."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_active_measurement_denied:{evidence_id}"
    ]
    missing_reasoning = (
        f"The latest atom {evidence_id} gives preferred_video: c but is unstable. "
        "The blocking gap requires another route to address the missing measurement evidence."
    )
    assert _visible_observation_fact_problems(missing_reasoning, state) == [
        f"latest_atom_active_measurement_denied:{evidence_id}"
    ]

    dropped_reasoning = (
        f"The latest atom {evidence_id} is active and supports the comparison. The blocking "
        "gap remains because this atom was dropped before binding."
    )
    assert (
        f"visible_atom_active_status_mismatch:{evidence_id}:true"
        in _visible_observation_fact_problems(dropped_reasoning, state)
    )
    feedback = _generation_retry_feedback(
        [f"visible_atom_active_status_mismatch:{evidence_id}:true"]
    )
    assert "do not describe an active new atom as dropped" in feedback


def test_rejects_order_inconsistent_when_latest_atom_order_is_unobserved() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                    "video_a_interval_s": [3.3, 11.5],
                    "video_b_interval_s": [7.1, 15.6],
                    "order_consistent": None,
                },
            }
        },
    }
    reasoning = (
        f"The latest atom {evidence_id} supports a but is order-inconsistent. "
        "The blocking gap requires a symmetric route."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_order_consistency_unobserved_but_asserted:{evidence_id}"
    ]


def test_rejects_positive_order_consistent_adjective_when_latest_atom_is_false() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "video_a_interval_s": [0.3, 7.9],
                    "video_b_interval_s": [0.1, 10.2],
                    "order_consistent": False,
                },
            }
        },
    }
    reasoning = (
        f"The latest atom {evidence_id} abstains with c and order-consistent intervals. "
        "The blocking gap requires aggregation."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_order_consistency_mismatch:{evidence_id}:true!=false"
    ]


def test_rejects_only_active_measurement_when_two_are_visible() -> None:
    latest_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [latest_id]},
        "evidence_store": {
            latest_id: {
                "active": True,
                "observation": {"measurement_method": "compare_uniform_timeline"},
            },
            ACTION_ID: {
                "active": True,
                "observation": {"measurement_method": "compare_action_refined_symmetric"},
            },
        },
    }
    reasoning = (
        f"The latest atom {latest_id} is the only active measurement. "
        "The blocking gap requires another route."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        "active_measurement_count_mismatch:said_only_one!=2"
    ]


def test_accepts_only_measurement_with_qualified_property_when_three_are_visible() -> None:
    state = {
        "evidence_store": {
            f"evidence-{index}": {
                "active": True,
                "observation": {"measurement_method": f"compare_method_{index}"},
            }
            for index in range(3)
        }
    }
    reasoning = (
        "The uniform atom is the only active measurement that reached a stable conclusion; "
        "the other two active measurements remain inconsistent. The blocking gap is creating "
        "a claim from that stable result."
    )
    assert _visible_observation_fact_problems(reasoning, state) == []

    claim_qualified = (
        "The uniform atom is the only active measurement atom that can support the exact "
        "claim values; the other two active measurements conflict with them. The blocking "
        "gap is resolving that disagreement before binding."
    )
    assert _visible_observation_fact_problems(claim_qualified, state) == []


def test_requires_unambiguous_selected_measurement_method_in_reasoning() -> None:
    generic = (
        "The latest task atom leaves the comparison gap open. Running one of the eligible "
        "visual comparison methods is the necessary next step to create support."
    )
    assert _chosen_measurement_tool_problem(
        "compare_action_refined_symmetric", generic
    ) == "chosen_measurement_tool_not_justified:compare_action_refined_symmetric"
    specific = (
        "The latest task atom leaves the comparison gap open. The fine-grained action route "
        "is selected because it targets the procedural movement rather than broad motion."
    )
    assert _chosen_measurement_tool_problem(
        "compare_action_refined_symmetric", specific
    ) is None


def test_rejects_false_only_eligible_tool_claim_but_allows_best_choice() -> None:
    state = {
        "eligible_tools": [
            "compare_uniform_timeline",
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
        ]
    }
    false_unique = (
        "The blocking gap is a coarse comparison. compare_uniform_timeline is the only "
        "eligible tool that can close it."
    )
    assert _eligible_tool_claim_problems(false_unique, state) == [
        "eligible_tool_count_mismatch:said_only_one!=3"
    ]
    best_choice = (
        "The blocking gap is a coarse comparison. compare_uniform_timeline is the best "
        "first eligible route because it surveys the whole scope."
    )
    assert _eligible_tool_claim_problems(best_choice, state) == []
    feedback = _generation_retry_feedback(
        ["eligible_tool_count_mismatch:said_only_one!=3"]
    )
    assert "contains multiple actions" in feedback


def test_requires_selected_vcsl_localizer_to_be_justified() -> None:
    generic = "The visible pair leaves a localization gap. A detector should run next."
    assert _chosen_measurement_tool_problem(
        "localize_copy_segment_spd_official", generic
    ) == "chosen_measurement_tool_not_justified:localize_copy_segment_spd_official"
    specific = (
        "The visible pair leaves a localization gap. The official SPD detector is selected "
        "because it produces the required frame box."
    )
    assert _chosen_measurement_tool_problem(
        "localize_copy_segment_spd_official", specific
    ) is None


def test_rejects_generation_audit_language_in_accepted_reasoning() -> None:
    reasoning = (
        "The latest uniform timeline atom exposes an active measurement count mismatch. "
        "The blocking gap is unresolved disagreement, so global motion is the next "
        "complementary comparison method."
    )
    assert _reasoning_process_artifact_problems(
        "compare_global_motion_symmetric", reasoning, {}
    ) == ["reasoning_mentions_generation_audit_artifact"]
    assert "do not mention generation attempts" in _generation_retry_feedback(
        ["reasoning_mentions_generation_audit_artifact"]
    )


def test_rejects_supported_then_unsupported_field_contradiction() -> None:
    reasoning = (
        "The visible state has active support for all six claim fields from two atoms. "
        "The blocking gap is that every field is still unsupported and unbound, so binding "
        "the atoms is the next step."
    )
    assert _reasoning_process_artifact_problems(
        "bind_evidence", reasoning, {}
    ) == ["reasoning_self_contradicts_field_support"]
    assert "supported but still unbound" in _generation_retry_feedback(
        ["reasoning_self_contradicts_field_support"]
    )


def test_rejects_unsupported_wording_when_complete_binding_is_executable() -> None:
    state = {
        "state_action_affordances": {
            "executable_binding_actions": [{"tool_name": "bind_evidence"}]
        }
    }
    reasoning = (
        "Two active atoms support the identity and localization fields. The blocking gap is "
        "that every required field is still unsupported and unbound."
    )
    assert "bindable_fields_described_as_unsupported" in (
        _reasoning_process_artifact_problems("bind_evidence", reasoning, state)
    )
    four_field_reasoning = (
        "The new measurement exists, but those four measurement fields are now unsupported. "
        "The blocking gap is binding the replacement atom."
    )
    assert "bindable_fields_described_as_unsupported" in (
        _reasoning_process_artifact_problems(
            "bind_evidence", four_field_reasoning, state
        )
    )
    for count_word in ("three", "five"):
        cross_dataset_reasoning = (
            f"The active atom supports the target fields, but those {count_word} fields are "
            "still unsupported after invalidation. Binding is the next step."
        )
        assert "bindable_fields_described_as_unsupported" in (
            _reasoning_process_artifact_problems(
                "bind_evidence", cross_dataset_reasoning, state
            )
        )


def test_rejects_nonexistent_rebind_tool_name_and_interval_order_semantics() -> None:
    invalid_tool = (
        "The binding invalidation was repaired by rebind_evidence, and the remaining gap is "
        "commitment."
    )
    problems = _reasoning_process_artifact_problems(
        "set_field_decision", invalid_tool, {}
    )
    assert "nonexistent_tool_name_in_reasoning:rebind_evidence" in problems
    assert "not a native tool" in _generation_retry_feedback(problems)

    interval_semantics = (
        "compare_action_refined_symmetric will test whether the intervals are "
        "order-consistent under role swapping."
    )
    problems = _reasoning_process_artifact_problems(
        "compare_action_refined_symmetric", interval_semantics, {}
    )
    assert "order_consistency_misapplied_to_intervals" in problems
    assert "comparison direction" in _generation_retry_feedback(problems)


def test_claim_field_count_is_derived_from_visible_schema() -> None:
    state = {
        "candidate_claims": [
            {
                "claim_id": "CC-VCSL",
                "fields": {
                    "query_id": {},
                    "reference_id": {},
                    "relationship": {},
                    "copied_segments_frames": {},
                    "localization_method": {},
                },
            }
        ]
    }
    bad = "The visible state shows all six claim fields are supported and committed."
    problems = _reasoning_process_artifact_problems("stop", bad, state)
    assert "claim_field_count_mismatch:said_6!=5" in problems
    assert "exactly 5 fields" in _generation_retry_feedback(problems)

    good = "The visible state shows all five claim fields are supported and committed."
    assert _reasoning_process_artifact_problems("stop", good, state) == []


def test_bind_must_disposition_just_executed_measurement() -> None:
    latest_id = "EA-VIDDIFF-COMPARE-GLOBAL-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [latest_id]},
        "evidence_store": {
            latest_id: {
                "active": True,
                "observation": {"measurement_method": "compare_global_motion_symmetric"},
            }
        },
    }
    arguments = {
        "binding_items": [
            {"field": "preferred_video", "support_ids": ["EA-OLDER-001"]}
        ]
    }
    assert _bind_latest_measurement_problem(
        "bind_evidence",
        arguments,
        "The blocking gap is binding the older stable atom.",
        state,
    ) == f"bind_omits_latest_measurement_rationale:{latest_id}"
    cited = (
        f"The latest `{latest_id}` has a visible disagreement, so the blocking gap is binding "
        "the older stable atom instead."
    )
    assert _bind_latest_measurement_problem(
        "bind_evidence", arguments, cited, state
    ) is None
    feedback = _generation_retry_feedback(
        [f"bind_omits_latest_measurement_rationale:{latest_id}"]
    )
    assert "must be consumed" in feedback
    assert latest_id in feedback


def test_rejects_binding_all_fields_as_if_meta_were_the_only_source() -> None:
    arguments = {
        "claim_id": "claim_1",
        "binding_items": [
            {
                "field": "sample_key",
                "support_ids": ["EA-VIDDIFF-ADAPTIVE-TASK-001"],
            },
            {
                "field": "preferred_video",
                "support_ids": [META_ID],
            },
        ],
    }
    reasoning = (
        f"The task atom covers identity and {META_ID} covers comparison. The blocking gap "
        "is binding, so binding that meta atom to every supported field is the next step."
    )
    assert _bind_evidence_reasoning_problem(
        "bind_evidence", arguments, reasoning
    ) == "bind_reasoning_excludes_required_task_atom"
    assert "two-atom evidence set" in _generation_retry_feedback(
        ["bind_reasoning_excludes_required_task_atom"]
    )


def test_rejects_first_use_measurement_described_as_rerun() -> None:
    state = {
        "past_execution_history": [
            {"past_action": {"executed_tool": "compare_uniform_timeline"}}
        ]
    }
    reasoning = (
        "The latest visible atom disagrees with the earlier route. The blocking gap is an "
        "independent view, so rerunning compare_global_motion_symmetric supplies the needed "
        "broad-motion comparison."
    )
    assert _reasoning_process_artifact_problems(
        "compare_global_motion_symmetric", reasoning, state
    ) == [
        "first_use_measurement_described_as_rerun:compare_global_motion_symmetric"
    ]


def test_accepts_explicit_rejection_of_rerun_for_first_use_measurement() -> None:
    reasoning = (
        "The current atom is insufficient, so a new complementary measurement is needed "
        "rather than rerunning the same method. The blocking gap is an independent timeline "
        "view, so compare_uniform_timeline is the next step."
    )
    assert _reasoning_process_artifact_problems(
        "compare_uniform_timeline", reasoning, {"past_execution_history": []}
    ) == []


def test_accepts_new_measurement_described_as_rerunning_a_different_method() -> None:
    reasoning = (
        "The visible uniform result remains low-confidence. The blocking gap needs an "
        "independent view, so rerunning a different eligible method supplies a new "
        "measurement rather than repeating compare_uniform_timeline."
    )
    assert _reasoning_process_artifact_problems(
        "compare_action_refined_symmetric", reasoning, {"past_execution_history": []}
    ) == []


def test_retry_feedback_for_unobserved_order_consistency_forbids_inference() -> None:
    feedback = _generation_retry_feedback(
        [
            "latest_atom_order_consistency_unobserved_but_asserted:"
            "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
        ]
    )
    assert "no observed order_consistent field" in feedback
    assert "Do not infer or assert true/false" in feedback


def test_retry_feedback_names_exact_preferred_video_correction() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    feedback = _generation_retry_feedback(
        [f"latest_atom_preferred_video_mismatch:{evidence_id}:a!=c"]
    )
    assert evidence_id in feedback
    assert "visible preferred_video is `c`, not `a`" in feedback
    assert "concrete visible defect" in feedback


def test_rejects_wrong_fact_about_earlier_atom_named_by_method() -> None:
    action_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    uniform_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [uniform_id]},
        "evidence_store": {
            action_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                },
            },
            uniform_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "b",
                    "order_consistent": None,
                },
            },
        },
    }
    reasoning = (
        "The active routes still disagree: `compare_action_refined_symmetric` and "
        "`compare_uniform_timeline` both support b. The blocking gap needs another route."
    )
    assert (
        f"latest_atom_preferred_video_mismatch:{action_id}:b!=c"
        in _visible_observation_fact_problems(reasoning, state)
    )


def test_does_not_attribute_contrasted_older_preference_to_meta_atom() -> None:
    meta_id = "EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-001"
    uniform_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [meta_id]},
        "evidence_store": {
            meta_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_symmetry_residual_abstention",
                    "preferred_video": "c",
                    "order_consistent": True,
                },
            },
            uniform_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "b",
                    "order_consistent": None,
                },
            },
        },
    }
    reasoning = (
        f"{meta_id} is role-invariant and strongest, whereas the earlier "
        "`compare_uniform_timeline` measurement supports preferred_video: b. "
        "The blocking gap is claim creation."
    )
    assert _visible_observation_fact_problems(reasoning, state) == []


def test_supports_because_is_not_parsed_as_preferred_video_b() -> None:
    meta_id = "EA-VIDDIFF-COMPARE_SYMMETRY_RESIDUAL_ABSTENTION-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [meta_id]},
        "evidence_store": {
            meta_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_symmetry_residual_abstention",
                    "preferred_video": "c",
                    "order_consistent": True,
                },
            }
        },
    }
    reasoning = (
        f"{meta_id} is the strongest support because it is active and directly supports "
        "preferred_video=c. The blocking gap is claim creation."
    )
    assert _visible_observation_fact_problems(reasoning, state) == []


def test_but_it_keeps_fact_attributed_to_current_atom() -> None:
    uniform_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [uniform_id]},
        "evidence_store": {
            uniform_id: {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "b",
                    "order_consistent": None,
                },
            }
        },
    }
    reasoning = (
        f"{uniform_id} supports b, but it is order-inconsistent. The blocking gap needs "
        "another method."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_order_consistency_unobserved_but_asserted:{uniform_id}"
    ]


def test_rejects_generic_scalar_fact_mismatch_for_named_vcsl_atom() -> None:
    evidence_id = "EA-VCSL-LOCALIZATION-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": ["relationship", "localization_method"],
                "observation": {
                    "relationship": "partial_copy",
                    "localization_method": "SPD-OFFICIAL-ISC",
                },
            }
        },
    }
    reasoning = (
        f"{evidence_id} reports relationship=full_copy. The blocking gap is claim creation."
    )
    assert (
        f"visible_atom_scalar_mismatch:{evidence_id}:relationship:full_copy!=partial_copy"
        in _visible_observation_fact_problems(reasoning, state)
    )


def test_confidence_only_phrase_is_not_parsed_as_a_scalar_value() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-UNIFORM-TIMELINE-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "confidence": 0.61,
                },
            }
        },
    }
    reasoning = (
        f"{evidence_id} has limited support, so the decision must not rely on confidence only. "
        "The blocking gap requires a complementary route."
    )
    assert _visible_observation_fact_problems(reasoning, state) == []


def test_assessment_accepts_preferred_video_as_natural_wording() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                    "order_consistent": None,
                },
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I select assess_evidence_candidate for {evidence_id}, which reports "
        "preferred_video as a and order_consistent as null. I retain it because "
        "its visible facts agree with the candidate claim."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    ) == []


def test_assessment_accepts_quoted_assignment_values() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-ACTION-REFINED-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "order_consistent": True,
                },
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I select assess_evidence_candidate for `{evidence_id}`. It reports "
        "`preferred_video=\"c\"` and `order_consistent=true`. I retain it because "
        "these visible facts agree with the candidate claim."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    ) == []


def test_set_field_decision_accepts_concise_complete_reasoning() -> None:
    reasoning = (
        "The remaining gap is that preferred_video and the interval fields are still open "
        "after the evidence dropout, although the retained atom now supports them. Committing "
        "those fields closes the report gap without changing the independently committed "
        "context and statement fields."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "set_field_decision",
        {
            "claim_id": "CC-VIDDIFF-SURGERY_26-001",
            "fields": [
                "preferred_video",
                "video_a_interval_s",
                "video_b_interval_s",
                "measurement_method",
            ],
            "decision_status": "committed",
        },
        meta_visible=False,
        visible_state={"eligible_tools": ["set_field_decision"]},
    )
    assert not any(item.startswith("reasoning_word_count:") for item in problems)


def test_rejects_wrong_vcsl_localization_box() -> None:
    evidence_id = "EA-VCSL-LOCALIZATION-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": ["copied_segments_frames"],
                "observation": {"copied_segments_frames": [[41, 1, 68, 29]]},
            }
        },
    }
    reasoning = (
        f"{evidence_id} supports box [40,1,68,29]. The blocking gap is claim creation."
    )
    assert (
        f"visible_atom_localization_box_mismatch:{evidence_id}:[40, 1, 68, 29]!="
        "[[41, 1, 68, 29]]"
        in _visible_observation_fact_problems(reasoning, state)
    )


def test_allows_rerun_word_when_selected_method_was_executed() -> None:
    state = {
        "past_execution_history": [
            {"past_action": {"executed_tool": "compare_global_motion_symmetric"}}
        ]
    }
    assert _reasoning_process_artifact_problems(
        "compare_global_motion_symmetric",
        "The visible result was dropped. The blocking gap requires rerunning global motion.",
        state,
    ) == []


def test_allows_rerun_for_same_method_evidence_dropout() -> None:
    state = {
        "past_execution_history": [],
        "dropped_evidence_ids": [
            "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
        ],
    }
    assert _reasoning_process_artifact_problems(
        "compare_global_motion_symmetric",
        "The old global-motion atom was dropped, so rerunning that measurement supplies fresh support.",
        state,
    ) == []


def test_allows_rerun_for_generic_dropout_id_when_claim_names_method() -> None:
    state = {
        "past_execution_history": [],
        "dropped_evidence_ids": ["EA-VIDDIFF-MEASUREMENT-001"],
        "candidate_claims": [
            {
                "fields": {
                    "measurement_method": {
                        "value": "compare_uniform_timeline",
                        "evidence_status": "unsupported",
                    }
                }
            }
        ],
    }
    assert _reasoning_process_artifact_problems(
        "compare_uniform_timeline",
        "The prior measurement was dropped, so rerunning the uniform timeline restores visible evidence.",
        state,
    ) == []


def test_rejects_rerun_for_different_method_after_dropout() -> None:
    state = {
        "past_execution_history": [],
        "dropped_evidence_ids": [
            "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
        ],
    }
    assert _reasoning_process_artifact_problems(
        "compare_action_refined_symmetric",
        "The global-motion atom was dropped, so rerunning action-refined comparison closes the gap.",
        state,
    ) == [
        "first_use_measurement_described_as_rerun:compare_action_refined_symmetric"
    ]


def test_allows_rerun_for_active_measurement_from_repair_snapshot() -> None:
    state = {
        "past_execution_history": [],
        "dropped_evidence_ids": ["EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001"],
        "evidence_store": {
            "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric"
                },
            }
        },
    }
    assert _reasoning_process_artifact_problems(
        "compare_action_refined_symmetric",
        "The old action-refined atom remains visible, but the repair round requires rerunning it after the dropout.",
        state,
    ) == []


def test_rejects_measurement_atom_claimed_to_support_task_identity_fields() -> None:
    evidence_id = ACTION_ID
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": [
                    "preferred_video",
                    "video_a_interval_s",
                    "video_b_interval_s",
                    "measurement_method",
                ],
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                },
            }
        },
    }
    reasoning = (
        f"The latest measurement {evidence_id} already supports the task identity fields and "
        "gives a low-confidence abstention. The blocking gap is stronger comparative evidence."
    )
    assert _visible_observation_fact_problems(reasoning, state) == [
        f"latest_atom_support_scope_mismatch:{evidence_id}:claimed_identity_fields"
    ]
