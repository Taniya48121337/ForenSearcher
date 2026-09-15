from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interactive_harness import (  # noqa: E402
    CandidateClaim,
    CLAIM_REVISION_TOOL,
    ClaimFieldState,
    EVIDENCE_ADJUDICATION_TOOL,
    EVIDENCE_ASSESSMENT_TOOL,
    EvidenceAdjudication,
    EvidenceAtom,
    EvidenceCandidateAssessment,
    TaskContract,
    VideoForensicHarness,
    _visible_temporal_order_inconsistent,
)
from viddiff_batch import VIDDIFF_SCHEMA  # noqa: E402
from run_student_closed_loop_canary import (  # noqa: E402
    PROGRESSIVE_VCSL_LOCALIZATION_SYSTEM,
    _action_lock_safe,
    _adjudication_reasoning_problems,
    _adjudication_reasoning_frontier,
    _chosen_measurement_tool_problem,
    _claim_creation_source_problem,
    _typed_universal_claim_choice_problem,
    _compact_recommended_measurement_retry_payload,
    _compact_progressive_decision_retry_payload,
    _evidence_assessment_reasoning_frontier,
    _evidence_assessment_reasoning_problems,
    _execution_frontier,
    _generation_retry_feedback,
    _generated_tool_name,
    _initial_progressive_decision_message,
    _initial_recommended_tool,
    _materialize_visible_action_arguments,
    _measurement_gap_contract,
    _measurement_need_problem,
    _measurement_observation_gap_problem,
    _measurement_recommendation_problem,
    _measurement_route_semantic_problem,
    _minimum_reasoning_words,
    _preexecution_reasoning_problems,
    _reasoning_process_artifact_problems,
    _repetitive_reasoning_style_problems,
    _semantic_tool_name,
    _progressive_turn_generation_budget,
    _repair_restoration_choice_problem,
    _repair_typed_order_abstention_causal_edge,
    _recommended_retry_tool,
    _turn_generation_budget,
    _use_progressive_measurement_system,
    _use_progressive_vcsl_localization_system,
    _use_post_adjudication_system,
    _visible_observation_fact_problems,
)
from run_student_closed_loop_batch import (  # noqa: E402
    DEFAULT_SFT_ROOT,
    _all_viddiff_items,
    _environment,
    _first_rows,
)
from shadow_execute_generated_actions import _restore_harness  # noqa: E402
from llm_teacher_rollout import _normalize_harness_params  # noqa: E402
from audit_student_closed_loop_quality import (  # noqa: E402
    _progressive_assessment_integrity,
)


class _NoExternalCalls:
    def execute(self, _name: str, _arguments: dict) -> dict:
        raise AssertionError("this unit test should not execute an external tool")


def test_visible_temporal_inversion_is_a_defensible_order_defect() -> None:
    assert _visible_temporal_order_inconsistent(
        {"copied_segments_frames": [[170, 152, 26, 8]]},
        ["copied_segments_frames"],
    )
    assert _visible_temporal_order_inconsistent(
        {"video_a_interval_s": [13.1, 5.6]},
        ["video_a_interval_s"],
    )
    assert not _visible_temporal_order_inconsistent(
        {"copied_segments_frames": [[26, 8, 170, 152]]},
        ["copied_segments_frames"],
    )


def test_assessment_word_floor_allows_concise_complete_fact_checks() -> None:
    assert _minimum_reasoning_words("assess_evidence_candidate") == 35
    assert _minimum_reasoning_words("adjudicate_evidence") == 45


def test_restore_harness_preserves_assessment_and_adjudication_ledgers() -> None:
    harness = _harness(include_adjudication=True, include_assessment=True)
    evidence_id = "EA-TASK-001"
    harness.evidence_store[evidence_id] = EvidenceAtom(
        evidence_id=evidence_id,
        kind="task_identity",
        supports_fields=["sample_key"],
        observation={"sample_key": "surgery_26"},
        provenance={"generated_by": "test"},
    )
    harness.evidence_assessments["CC-001"] = {
        evidence_id: EvidenceCandidateAssessment(
            assessment_id="ASSESS-001",
            claim_id="CC-001",
            target_fields=["sample_key"],
            evidence_id=evidence_id,
            disposition="retain",
            defect_codes=[],
            considered_evidence_ids=[evidence_id],
        )
    }
    harness.evidence_adjudications["CC-001"] = EvidenceAdjudication(
        adjudication_id="ADJ-001",
        claim_id="CC-001",
        target_fields=["sample_key"],
        selected_evidence_ids=[evidence_id],
        rejected_evidence=[],
        considered_evidence_ids=[evidence_id],
    )
    row = {
        "messages": [
            {"role": "system", "content": "test"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": {"claim_schema": VIDDIFF_SCHEMA},
                        "current_harness_state": harness.render_state(),
                    }
                ),
            },
        ],
        "tools": [
            {"type": "function", "function": {"name": "adjudicate_evidence"}}
        ],
    }
    restored = _restore_harness(row)
    assert restored.evidence_assessments["CC-001"][evidence_id].disposition == (
        "retain"
    )
    assert restored.evidence_adjudications["CC-001"].selected_evidence_ids == [
        evidence_id
    ]


def _measurement(
    evidence_id: str,
    *,
    preferred: str,
    method: str,
    confidence: float,
    order_consistent: bool | None,
) -> EvidenceAtom:
    return EvidenceAtom(
        evidence_id=evidence_id,
        kind="action_comparison_measurement",
        supports_fields=list(VIDDIFF_SCHEMA["target_fields"]),
        observation={
            "preferred_video": preferred,
            "video_a_interval_s": [5.6, 13.1] if preferred == "a" else [4.5, 8.9],
            "video_b_interval_s": [2.433, 8.567] if preferred == "a" else [3.5, 8.8],
            "measurement_method": method,
            "confidence": confidence,
            "order_consistent": order_consistent,
            "decision_rule": "direct_visual_comparison" if preferred == "a" else "order_disagreement_abstain",
            "visible_rationale": "Visible structured comparison result.",
        },
        provenance={"generated_by": f"EXEC-{evidence_id}"},
    )


def _harness(
    *,
    include_adjudication: bool = True,
    include_assessment: bool = False,
) -> VideoForensicHarness:
    registry = [
        {"name": "bind_evidence", "domain": "harness_state"},
        {"name": "set_field_decision", "domain": "harness_state"},
        {"name": "stop", "domain": "terminal"},
        copy.deepcopy(CLAIM_REVISION_TOOL),
    ]
    if include_adjudication:
        registry.append(copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL))
    if include_assessment:
        registry.append(copy.deepcopy(EVIDENCE_ASSESSMENT_TOOL))
    harness = VideoForensicHarness(
        TaskContract(
            contract_id="TASK-ADJ-001",
            episode_mode="paired_compare",
            task="Compare the visible pair.",
            visible_assets={"source": "video_a", "suspect": "video_b"},
            claim_schema_id=VIDDIFF_SCHEMA["schema_id"],
        ),
        _NoExternalCalls(),
        claim_schema=VIDDIFF_SCHEMA,
        tool_registry=registry,
        target_fields=VIDDIFF_SCHEMA["target_fields"],
        invariant_fields=VIDDIFF_SCHEMA["invariant_fields"],
    )
    fields = {
        "sample_key": ClaimFieldState(
            value="surgery_26", evidence_status="supported", decision_status="committed"
        ),
        "comparison_statement": ClaimFieldState(
            value="The tube moves more", evidence_status="supported", decision_status="committed"
        ),
        "preferred_video": ClaimFieldState(value="a"),
        "video_a_interval_s": ClaimFieldState(value=[5.6, 13.1]),
        "video_b_interval_s": ClaimFieldState(value=[2.433, 8.567]),
        "measurement_method": ClaimFieldState(value="compare_uniform_timeline"),
    }
    harness.claims["CC-001"] = CandidateClaim(
        "CC-001", VIDDIFF_SCHEMA["schema_id"], fields
    )
    harness.bindings["CC-001"] = {}
    harness.evidence_store = {
        "EA-UNIFORM-001": _measurement(
            "EA-UNIFORM-001",
            preferred="a",
            method="compare_uniform_timeline",
            confidence=0.61,
            order_consistent=None,
        ),
        "EA-ACTION-001": _measurement(
            "EA-ACTION-001",
            preferred="c",
            method="compare_action_refined_symmetric",
            confidence=0.72,
            order_consistent=False,
        ),
        "EA-GLOBAL-001": _measurement(
            "EA-GLOBAL-001",
            preferred="c",
            method="compare_global_motion_symmetric",
            confidence=0.70,
            order_consistent=False,
        ),
    }
    harness.goals["G-CONTEXT"].status = "satisfied"
    harness.goals["G-DIFFERENCE"].status = "satisfied"
    harness.goals["G-REPORT"].status = "open"
    return harness


def _apply_visible_revision(harness: VideoForensicHarness) -> None:
    candidates = harness._claim_revision_candidates()["CC-001"]
    harness.execute_tool(
        "revise_claim_fields",
        {
            "claim_id": "CC-001",
            "field_updates": [
                {"field": field_name, "value": item["value"]}
                for field_name, item in candidates.items()
            ],
        },
        "call-revise-selected",
    )


def test_conflicting_visible_atoms_require_exact_adjudication_before_bind() -> None:
    harness = _harness()
    state = harness.render_state()
    assert state["eligible_tools"] == ["adjudicate_evidence"]
    requirement = state["evidence_adjudication"]["requirement"]
    assert requirement["competing_evidence_ids"] == [
        "EA-ACTION-001",
        "EA-GLOBAL-001",
        "EA-UNIFORM-001",
    ]
    actions = state["state_action_affordances"]["executable_adjudication_actions"]
    assert len(actions) == 1
    arguments = actions[0]["arguments"]
    assert arguments["selected_evidence_ids"] == ["EA-ACTION-001"]
    assert {
        item["evidence_id"] for item in arguments["rejected_evidence"]
    } == {"EA-GLOBAL-001", "EA-UNIFORM-001"}

    result = harness.execute_tool("adjudicate_evidence", arguments, "call-adj-001")
    assert result["status"] == "accepted"
    post = harness.render_state()
    assert post["evidence_adjudication"]["required"] is False
    assert post["eligible_tools"] == ["revise_claim_fields"]
    _apply_visible_revision(harness)
    post = harness.render_state()
    assert "bind_evidence" in post["eligible_tools"]
    binding = post["state_action_affordances"]["executable_binding_actions"][0]
    assert {
        item["support_ids"][0]
        for item in binding["arguments"]["binding_items"]
    } == {"EA-ACTION-001"}
    harness.execute_tool(
        "bind_evidence",
        {
            "claim_id": "CC-001",
            "bindings": {
                item["field"]: item["support_ids"]
                for item in binding["arguments"]["binding_items"]
            },
        },
        "call-bind-selected",
    )
    bound = harness.render_state()
    assert bound["eligible_tools"] == ["set_field_decision"]
    assert bound["state_action_affordances"]["executable_binding_actions"] == []


@pytest.mark.parametrize(
    ("measurements", "expected"),
    [
        (
            [("EA-A", "a", "m-a", 0.7), ("EA-B", "b", "m-b", 0.8), ("EA-C", "c", "m-c", 0.6)],
            "EA-C",
        ),
        (
            [("EA-B1", "b", "m-b1", 0.7), ("EA-B2", "b", "m-b2", 0.8), ("EA-C", "c", "m-c", 0.9)],
            "EA-B2",
        ),
        (
            [("EA-C1", "c", "m-c1", 0.6), ("EA-C2", "c", "m-c2", 0.8), ("EA-A", "a", "m-a", 0.9)],
            "EA-C2",
        ),
        (
            [("EA-B1", "b", "same-method", 0.6), ("EA-B2", "b", "same-method", 0.8), ("EA-A", "a", "m-a", 0.9), ("EA-C", "c", "m-c", 0.7)],
            "EA-C",
        ),
    ],
)
def test_label_blind_selection_uses_independent_method_votes(
    measurements: list[tuple[str, str, str, float]], expected: str
) -> None:
    harness = _harness()
    harness.evidence_store = {
        evidence_id: _measurement(
            evidence_id,
            preferred=preferred,
            method=method,
            confidence=confidence,
            order_consistent=False if preferred == "c" else True,
        )
        for evidence_id, preferred, method, confidence in measurements
    }
    requirement = {
        "claim_id": "CC-001",
        "target_fields": list(VIDDIFF_SCHEMA["target_fields"]),
        "competing_evidence_ids": sorted(harness.evidence_store),
        "compatible_evidence_ids_by_field": {},
    }
    assert harness._evidence_first_selection(requirement) == [expected]


def test_open_supported_fields_still_surface_new_visible_conflict() -> None:
    harness = _harness(include_assessment=True)
    for field_name in harness.target_fields:
        field = harness.claims["CC-001"].fields[field_name]
        field.evidence_status = "supported"
        field.decision_status = "open"
        field.support_ids = ["EA-ACTION-001"]
    state = harness.render_state()
    assert state["evidence_adjudication"]["required"] is True
    assert state["eligible_tools"] == ["assess_evidence_candidate"]
    assert set(state["evidence_adjudication"]["requirement"]["target_fields"]) == set(
        harness.target_fields
    )


def test_partial_field_match_cannot_hide_whole_atom_conflict() -> None:
    harness = _harness()
    target_a = harness.claims["CC-001"].fields["video_a_interval_s"].value
    target_b = harness.claims["CC-001"].fields["video_b_interval_s"].value
    harness.evidence_store["EA-ACTION-001"].observation["video_b_interval_s"] = copy.deepcopy(
        target_b
    )
    harness.evidence_store["EA-GLOBAL-001"].observation["video_a_interval_s"] = copy.deepcopy(
        target_a
    )
    state = harness.render_state()
    assert state["eligible_tools"] == ["adjudicate_evidence"]
    requirement = state["evidence_adjudication"]["requirement"]
    assert requirement["competing_evidence_ids"] == [
        "EA-ACTION-001",
        "EA-GLOBAL-001",
        "EA-UNIFORM-001",
    ]
    action = state["state_action_affordances"]["executable_adjudication_actions"][0]
    assert action["arguments"]["selected_evidence_ids"] == ["EA-ACTION-001"]
    assert {
        item["evidence_id"] for item in action["arguments"]["rejected_evidence"]
    } == {"EA-GLOBAL-001", "EA-UNIFORM-001"}


def test_adjudication_rejects_incomplete_competitor_partition() -> None:
    harness = _harness()
    arguments = copy.deepcopy(
        harness.render_state()["state_action_affordances"]
        ["executable_adjudication_actions"][0]["arguments"]
    )
    arguments["rejected_evidence"] = arguments["rejected_evidence"][:1]
    with pytest.raises(ValueError, match="exactly match"):
        harness.execute_tool("adjudicate_evidence", arguments, "call-adj-bad")


def test_bind_cannot_bypass_selected_evidence_after_adjudication() -> None:
    harness = _harness()
    arguments = harness.render_state()["state_action_affordances"][
        "executable_adjudication_actions"
    ][0]["arguments"]
    harness.execute_tool("adjudicate_evidence", arguments, "call-adj-001")
    with pytest.raises(ValueError, match="not eligible"):
        harness.execute_tool(
            "bind_evidence",
            {
                "claim_id": "CC-001",
                "bindings": {"preferred_video": ["EA-UNIFORM-001"]},
            },
            "call-bind-before-revision",
        )
    _apply_visible_revision(harness)
    result = harness.execute_tool(
        "bind_evidence",
        {
            "claim_id": "CC-001",
            "bindings": {"preferred_video": ["EA-UNIFORM-001"]},
        },
        "call-bind-bad",
    )
    assert result["status"] == "rejected"
    assert result["rejected_fields"]["preferred_video"] in {
        "claim_value_not_supported_by_evidence",
        "evidence_not_selected_by_latest_adjudication",
    }


def test_unregistered_adjudication_tool_preserves_existing_binding_surface() -> None:
    state = _harness(include_adjudication=False).render_state()
    assert "evidence_adjudication" not in state
    assert "bind_evidence" in state["eligible_tools"]


def test_progressive_assessment_externalizes_every_candidate_before_adjudication() -> None:
    harness = _harness(include_assessment=True)
    assessed: list[str] = []
    for _ in range(3):
        state = harness.render_state()
        assert state["eligible_tools"] == ["assess_evidence_candidate"]
        assert state["evidence_assessment"]["current_candidate_id"] not in assessed
        actions = state["state_action_affordances"][
            "executable_evidence_assessment_actions"
        ]
        assert len(actions) == 1
        assert actions[0]["arguments"]["evidence_id"] == state[
            "evidence_assessment"
        ]["current_candidate_id"]
        action = actions[0]
        evidence_id = action["arguments"]["evidence_id"]
        assert evidence_id not in assessed
        result = harness.execute_tool(
            "assess_evidence_candidate",
            action["arguments"],
            f"call-assess-{len(assessed) + 1:03d}",
        )
        assert result["status"] == "accepted"
        assessed.append(evidence_id)

    state = harness.render_state()
    assert state["evidence_assessment"]["pending_evidence_ids"] == []
    assert state["eligible_tools"] == ["adjudicate_evidence"]
    action = state["state_action_affordances"]["executable_adjudication_actions"][0]
    result = harness.execute_tool(
        "adjudicate_evidence", action["arguments"], "call-progressive-adj"
    )
    assert result["status"] == "accepted"
    assert harness.render_state()["eligible_tools"] == ["revise_claim_fields"]
    _apply_visible_revision(harness)
    assert "bind_evidence" in harness.render_state()["eligible_tools"]


def test_atomic_assessment_reasoning_is_candidate_specific_and_null_safe() -> None:
    harness = _harness(include_assessment=True)
    state = harness.render_state()
    while state["evidence_assessment"]["current_candidate_id"] != "EA-UNIFORM-001":
        prior_action = state["state_action_affordances"][
            "executable_evidence_assessment_actions"
        ][0]
        harness.execute_tool(
            "assess_evidence_candidate",
            prior_action["arguments"],
            f"advance-{state['evidence_assessment']['current_candidate_id']}",
        )
        state = harness.render_state()
    actions = state["state_action_affordances"][
        "executable_evidence_assessment_actions"
    ]
    uniform = next(
        action
        for action in actions
        if action["arguments"]["evidence_id"] == "EA-UNIFORM-001"
    )
    reasoning = (
        "`EA-UNIFORM-001` reports `preferred_video=a`, while its `order_consistent` "
        "status is unobserved rather than true or false. Two independent methods "
        "abstain with c, so I reject this candidate for cross-method direction conflict "
        "and keep it out of the later evidence binding."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        uniform["arguments"],
        reasoning,
        visible_state=state,
    ) == []
    quoted_reasoning = (
        "I select assess_evidence_candidate for `EA-UNIFORM-001`, which reports "
        "`preferred_video=\"a\"` and `order_consistent=null`, so order consistency is "
        "unobserved. Two independent methods abstain with c, so I reject this candidate "
        "for cross-method direction conflict and keep it out of the later binding."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        uniform["arguments"],
        quoted_reasoning,
        visible_state=state,
    ) == []
    frontier = _evidence_assessment_reasoning_frontier(state, actions)
    assert frontier is not None
    assert len(frontier["pending_candidate_assessments"]) == 1
    assert len(state["evidence_assessment"]["pending_evidence_ids"]) == 1
    assert any(
        item["order_consistent"] is None
        for item in frontier["pending_candidate_assessments"]
    )

    bad = (
        "This candidate has no order inconsistency and should be retained. The "
        "remaining gap is to save the assessment before adjudication."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        uniform["arguments"],
        bad,
        visible_state=state,
    )
    assert "assessment_candidate_not_cited:EA-UNIFORM-001" in problems
    assert "assessment_preferred_video_not_stated:EA-UNIFORM-001" in problems
    assert "assessment_unobserved_order_not_stated:EA-UNIFORM-001" in problems

    retained_but_misframed = (
        "I select assess_evidence_candidate for `EA-UNIFORM-001`, which reports "
        "preferred_video=a and order_consistent is unobserved. I retain it because the visible "
        "values match, although the visible defect is missing order consistency."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        uniform["arguments"],
        retained_but_misframed,
        visible_state=state,
    )
    assert "assessment_reject_disposition_not_explained:EA-UNIFORM-001" in problems
    assert "native retain/reject disposition" in _generation_retry_feedback(problems)
    full_partition_overclaim = (
        "I select assess_evidence_candidate for `EA-UNIFORM-001`, which reports "
        "preferred_video=a and order_consistent is unobserved, and I retain it as usable "
        "support. The native call carries every exact rejected ID and defect code."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        uniform["arguments"],
        full_partition_overclaim,
        visible_state=state,
    )
    assert "assessment_claims_full_adjudication_partition_payload" in problems
    assert "later adjudicate_evidence" in _generation_retry_feedback(problems)


def test_identity_atom_assessment_does_not_require_comparison_fields() -> None:
    evidence_id = "EA-VIDDIFF-ADAPTIVE-TASK-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "kind": "task_identity",
                "supports_fields": ["sample_key", "comparison_statement"],
                "observation": {
                    "sample_key": "surgery_26",
                    "comparison_statement": "The tube moves more",
                },
            }
        }
    }
    arguments = {
        "claim_id": "claim_1",
        "target_fields": ["sample_key", "comparison_statement"],
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I select assess_evidence_candidate for `{evidence_id}`, whose visible facts are "
        "sample_key=surgery_26 and comparison_statement=\"The tube moves more\". I retain "
        "it because those identity values match the claim, closing only this assessment gap."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert not any("order" in item or "preferred_video" in item for item in problems)
    missing_identity_facts = (
        f"I select assess_evidence_candidate for `{evidence_id}`. "
        "I retain the identity atom because it matches the claim."
    )
    missing_problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        missing_identity_facts,
        visible_state=state,
    )
    assert (
        f"assessment_identity_fact_not_stated:{evidence_id}:sample_key"
        in missing_problems
    )
    assert (
        f"assessment_identity_fact_not_stated:{evidence_id}:comparison_statement"
        in missing_problems
    )
    frontier = _evidence_assessment_reasoning_frontier(
        state,
        [{"tool_name": "assess_evidence_candidate", "arguments": arguments}],
    )
    assert frontier is not None
    candidate = frontier["pending_candidate_assessments"][0]
    assert candidate["measurement_semantics_required"] is False
    assert "order_consistent" not in candidate
    assert candidate["supported_observation_facts"] == {
        "sample_key": "surgery_26",
        "comparison_statement": "The tube moves more",
    }


def test_identity_assessment_accepts_exact_statement_ending_in_parenthesis() -> None:
    evidence_id = "EA-VIDDIFF-ADAPTIVE-TASK-001"
    statement = "The tips are never out of view (occluded, or out of frame)"
    state = {
        "evidence_store": {
            evidence_id: {
                "kind": "task_identity",
                "supports_fields": ["sample_key", "comparison_statement"],
                "observation": {
                    "sample_key": "surgery_71",
                    "comparison_statement": statement,
                },
            }
        }
    }
    arguments = {
        "claim_id": "claim_1",
        "target_fields": ["sample_key", "comparison_statement"],
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I assess `{evidence_id}` with sample_key=\"surgery_71\" and "
        f"comparison_statement=\"{statement}\". I retain the atom because both exact "
        "identity facts match the claim, closing only this candidate-assessment gap."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    ) == []


def test_universal_typed_absence_cannot_be_overridden_by_coarse_preference() -> None:
    evidence_id = "EA-TYPED-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_predicate_independent",
                    "predicate_family": "visibility_coverage",
                    "decision_rule": "universal_claim_no_observed_counterexample_abstain",
                    "preferred_video": "c",
                    "usable_a": True,
                    "usable_b": True,
                    "reasons_a": ["sparse_sampling_cannot_prove_never"],
                    "reasons_b": ["sparse_sampling_cannot_prove_never"],
                    "independent_typed_records": {
                        "a": {"violation_count": 0},
                        "b": {"violation_count": 0},
                    },
                },
            }
        }
    }
    bad = {
        "field_values": {
            "preferred_video": "a",
            "measurement_method": "compare_uniform_timeline",
        }
    }
    assert _typed_universal_claim_choice_problem(
        "create_candidate_claim", bad, state
    ) == f"typed_universal_abstention_overridden:{evidence_id}"
    good = {
        "field_values": {
            "preferred_video": "c",
            "measurement_method": "compare_predicate_independent",
        }
    }
    assert _typed_universal_claim_choice_problem(
        "create_candidate_claim", good, state
    ) is None


def test_assessment_rejects_false_ledger_and_premature_support_language() -> None:
    evidence_id = "EA-IDENTITY-001"
    state = {
        "candidate_claims": [
            {
                "claim_id": "claim_1",
                "fields": {
                    "sample_key": {
                        "value": "surgery_26",
                        "evidence_status": "unsupported",
                    }
                },
            }
        ],
        "evidence_store": {
            evidence_id: {
                "kind": "task_identity",
                "supports_fields": ["sample_key"],
                "observation": {"sample_key": "surgery_26"},
            }
        },
        "evidence_assessment": {"pending_evidence_ids": [evidence_id, "EA-NEXT"]},
    }
    arguments = {
        "claim_id": "claim_1",
        "target_fields": ["sample_key"],
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I select assess_evidence_candidate for `{evidence_id}` and record "
        "sample_key=surgery_26. The candidate-assessment ledger is complete, this atom is "
        "the only eligible action, and I retain it because the claim values are already supported."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert "assessment_falsely_claims_complete_ledger" in problems
    assert "assessment_claims_premature_field_support" in problems
    assert "assessment_atom_called_action" in problems

    closes_ledger = (
        f"I select assess_evidence_candidate for `{evidence_id}` and record "
        "sample_key=surgery_26. Retaining it closes the candidate-assessment ledger."
    )
    closing_problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, closes_ledger, visible_state=state
    )
    assert "assessment_falsely_claims_complete_ledger" in closing_problems


def test_progressive_assessment_audit_requires_exact_final_partition() -> None:
    turns = [
        {
            "turn": 1,
            "generated_tool": "assess_evidence_candidate",
            "generated_arguments": {
                "evidence_id": "EA-UNIFORM-001",
                "disposition": "retain",
                "defect_codes": [],
            },
        },
        {
            "turn": 2,
            "generated_tool": "assess_evidence_candidate",
            "generated_arguments": {
                "evidence_id": "EA-ACTION-001",
                "disposition": "reject",
                "defect_codes": ["order_inconsistent"],
            },
        },
        {
            "turn": 3,
            "generated_tool": "adjudicate_evidence",
            "generated_arguments": {
                "selected_evidence_ids": ["EA-UNIFORM-001"],
                "rejected_evidence": [
                    {
                        "evidence_id": "EA-ACTION-001",
                        "defect_codes": ["order_inconsistent"],
                    }
                ],
            },
        },
    ]
    assert _progressive_assessment_integrity(turns) == (True, [])
    turns[1]["generated_arguments"]["disposition"] = "retain"
    passed, problems = _progressive_assessment_integrity(turns)
    assert passed is False
    assert "rejected_evidence_not_rejected_in_assessment" in problems


def test_generation_budget_tracks_native_action_envelope() -> None:
    assert _turn_generation_budget(["set_field_decision"], 512) == 192
    assert _turn_generation_budget(["compare_uniform_timeline"], 512) == 320
    assert _turn_generation_budget(["bind_evidence"], 512) == 320
    assert _turn_generation_budget(["assess_evidence_candidate"], 512) == 320
    assert _turn_generation_budget(["adjudicate_evidence"], 512) == 448
    assert _turn_generation_budget(["adjudicate_evidence"], 192) == 192
    assert _progressive_turn_generation_budget(["set_field_decision"], 384) == 144
    assert _progressive_turn_generation_budget(["assess_evidence_candidate"], 384) == 176
    assert _progressive_turn_generation_budget(["create_candidate_claim"], 384) == 320
    assert _progressive_turn_generation_budget(["adjudicate_evidence"], 384) == 160
    assert _progressive_turn_generation_budget(["stop"], 384) == 128
    four_competitor_action = [
        {
            "tool_name": "adjudicate_evidence",
            "arguments": {
                "selected_evidence_ids": ["EA-4"],
                "rejected_evidence": [
                    {"evidence_id": "EA-1"},
                    {"evidence_id": "EA-2"},
                    {"evidence_id": "EA-3"},
                ],
            },
        }
    ]
    assert (
        _progressive_turn_generation_budget(
            ["adjudicate_evidence"], 384, four_competitor_action
        )
        == 240
    )
    assert (
        _progressive_turn_generation_budget(
            ["adjudicate_evidence"], 200, four_competitor_action
        )
        == 200
    )
    assert _progressive_turn_generation_budget(["compare_uniform_timeline"], 384) == 192


def test_adjudication_coordinated_selected_list_grounds_each_atom() -> None:
    identity_id = "EA-VIDDIFF-ADAPTIVE-TASK-001"
    measurement_id = "EA-VIDDIFF-COMPARE-UNIFORM-TIMELINE-001"
    state = {
        "evidence_store": {
            identity_id: {"observation": {"sample_key": "surgery_26"}},
            measurement_id: {"observation": {"preferred_video": "a"}},
        }
    }
    arguments = {
        "selected_evidence_ids": [identity_id, measurement_id],
        "rejected_evidence": [],
    }
    reasoning = (
        "I select adjudicate_evidence to resolve the final evidence partition. "
        f"The selected atoms are `{identity_id}` because it matches "
        f"`sample_key=\"surgery_26\"`, and `{measurement_id}` because it matches "
        "`preferred_video=\"a\"`."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, reasoning, visible_state=state
    ) == []


def test_short_post_adjudication_system_is_used_only_after_visible_adjudication() -> None:
    state = {"evidence_adjudications": {"CC-1": {"selected_evidence_ids": ["EA-1"]}}}
    assert _use_post_adjudication_system(
        ["bind_evidence"], state, require_progressive_evidence_assessment=True
    )
    assert _use_post_adjudication_system(
        ["set_field_decision"], state, require_progressive_evidence_assessment=True
    )
    assert _use_post_adjudication_system(
        ["stop"], state, require_progressive_evidence_assessment=True
    )
    assert not _use_post_adjudication_system(
        ["adjudicate_evidence"], state, require_progressive_evidence_assessment=True
    )
    assert not _use_post_adjudication_system(
        ["bind_evidence"], {}, require_progressive_evidence_assessment=True
    )
    assert not _use_post_adjudication_system(
        ["bind_evidence"], state, require_progressive_evidence_assessment=False
    )


def test_execution_frontier_externalizes_post_adjudication_state_language() -> None:
    frontier = _execution_frontier(
        ["bind_evidence"],
        [
            {
                "type": "function",
                "function": {"name": "bind_evidence", "description": "Bind support."},
            }
        ],
        [{"tool_name": "bind_evidence", "arguments": {"claim_id": "claim_1"}}],
    )
    contract = frontier["state_transition_language_contract"]
    assert contract["transition"] == "supported_unbound_to_bound"
    assert "supports these claim fields" in contract["required_gap_clause"]
    assert any("unsupported" in item for item in contract["prohibited_descriptions"])


def test_short_progressive_measurement_system_is_used_only_for_sensing_frontiers() -> None:
    assert _use_progressive_measurement_system(
        ["compare_uniform_timeline", "compare_global_motion_symmetric"],
        require_progressive_evidence_assessment=True,
    )
    assert not _use_progressive_measurement_system(
        ["compare_uniform_timeline", "bind_evidence"],
        require_progressive_evidence_assessment=True,
    )
    assert not _use_progressive_measurement_system(
        ["compare_uniform_timeline"],
        require_progressive_evidence_assessment=False,
    )


def test_vcsl_localization_system_is_used_only_for_pure_localizer_frontiers() -> None:
    assert _use_progressive_vcsl_localization_system(
        ["localize_copy_segment_dp"],
        require_progressive_evidence_assessment=True,
    )
    assert not _use_progressive_vcsl_localization_system(
        ["localize_copy_segment_dp", "compare_uniform_timeline"],
        require_progressive_evidence_assessment=True,
    )
    assert not _use_progressive_vcsl_localization_system(
        ["localize_copy_segment_dp"],
        require_progressive_evidence_assessment=False,
    )
    assert "query/reference feature-similarity map" in (
        PROGRESSIVE_VCSL_LOCALIZATION_SYSTEM
    )
    assert "does not test A/B presentation orders" in (
        PROGRESSIVE_VCSL_LOCALIZATION_SYSTEM
    )


def test_visible_action_materialization_requires_unique_state_values() -> None:
    text = (
        "The remaining gap needs a focused comparison. <tool_call>"
        '{"name":"compare_action_refined_symmetric","arguments":{"video_a_ref":'
    )
    assert _generated_tool_name(text) == "compare_action_refined_symmetric"
    assert _semantic_tool_name(
        text,
        ["compare_global_motion_symmetric", "compare_action_refined_symmetric"],
    ) == "compare_action_refined_symmetric"
    assert _semantic_tool_name(
        "Compare compare_global_motion_symmetric with compare_action_refined_symmetric.",
        ["compare_global_motion_symmetric", "compare_action_refined_symmetric"],
    ) is None
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compare_action_refined_symmetric",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_a_ref": {"type": "string", "enum": ["video_a"]},
                        "video_b_ref": {"type": "string", "enum": ["video_b"]},
                        "comparison_statement": {"type": "string"},
                    },
                    "required": [
                        "video_a_ref",
                        "video_b_ref",
                        "comparison_statement",
                    ],
                    "additionalProperties": False,
                },
            },
        }
    ]
    state = {
        "candidate_claims": [
            {
                "fields": {
                    "comparison_statement": {"value": "The tube moves more"}
                }
            }
        ]
    }
    assert _materialize_visible_action_arguments(
        "compare_action_refined_symmetric",
        state=state,
        tools=tools,
        executable_actions=[],
    ) == {
        "video_a_ref": "video_a",
        "video_b_ref": "video_b",
        "comparison_statement": "The tube moves more",
    }
    state["candidate_claims"].append(
        {"fields": {"comparison_statement": {"value": "A different statement"}}}
    )
    assert _materialize_visible_action_arguments(
        "compare_action_refined_symmetric",
        state=state,
        tools=tools,
        executable_actions=[],
    ) is None


def test_adjudication_reasoning_requires_full_prose_partition_without_ledger() -> None:
    state = _harness().render_state()
    arguments = state["state_action_affordances"][
        "executable_adjudication_actions"
    ][0]["arguments"]
    frontier = _adjudication_reasoning_frontier(
        state,
        [{"tool_name": "adjudicate_evidence", "arguments": arguments}],
    )
    assert frontier is not None
    assert frontier["required_atom_mentions"] == [
        {"reasoning_handle": "@E1", "evidence_id": "EA-ACTION-001"},
        {"reasoning_handle": "@E2", "evidence_id": "EA-GLOBAL-001"},
        {"reasoning_handle": "@E3", "evidence_id": "EA-UNIFORM-001"},
    ]
    handle_reasoning = (
        "I compare and resolve the three atoms before binding. I retain `@E1` because "
        "preferred_video=c means abstention and two independent methods support c, not "
        "because it matches the old claim. I reject `@E2` as a redundant c conclusion "
        "and `@E3` for cross-method direction conflict with a; this records the complete "
        "partition before any claim revision or binding."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, handle_reasoning, visible_state=state
    ) == []
    good = (
        "I compare all three visible atoms before binding. I select `EA-ACTION-001` "
        "because preferred_video=c means abstention and two independent methods agree "
        "on c without using the old claim as a prior. `EA-GLOBAL-001` is a redundant "
        "same-conclusion c atom, while `EA-UNIFORM-001` has a cross-method direction "
        "conflict with a, so the remaining gap is an explicit adjudication record."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, good
    ) == []

    bad = (
        "I compare the available evidence and select the strongest one. The remaining "
        "gap is to resolve the alternatives before binding the claim."
    )
    problems = _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, bad
    )
    assert any(item.startswith("adjudication_selected_id_not_cited:") for item in problems)
    assert sum(
        item.startswith("adjudication_rejected_id_not_cited:") for item in problems
    ) == 2
    feedback = _generation_retry_feedback(problems)
    assert "EA-UNIFORM-001" in feedback
    assert "EA-ACTION-001" in feedback

    semantic = (
        "I compare all three visible atoms before binding and retain "
        "compare_action_refined_symmetric because preferred_video=c is the abstention "
        "supported by two independent methods, not because it matches the old claim. I "
        "reject compare_global_motion_symmetric as a redundant c conclusion and "
        "compare_uniform_timeline for a cross-method direction conflict with a. The "
        "remaining gap is to record this complete adjudication before claim revision or binding."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence",
        arguments,
        semantic,
        visible_state=_harness().render_state(),
    ) == []

    misframed_gap = (
        "I select adjudicate_evidence and retain `EA-UNIFORM-001` because it matches the claim. "
        "The visible gap is that `EA-ACTION-001` and `EA-GLOBAL-001` are still unbound; I reject "
        "both because each is order-inconsistent and conflicts with the claim."
    )
    problems = _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, misframed_gap
    )
    assert "adjudication_gap_misframed_as_binding" in problems
    assert "unresolved evidence partition" in _generation_retry_feedback(problems)


def test_adjudication_uses_complete_atomic_ledger_without_repeating_rejections() -> None:
    harness = _harness(include_assessment=True)
    while harness.render_state()["eligible_tools"] == ["assess_evidence_candidate"]:
        state = harness.render_state()
        action = state["state_action_affordances"][
            "executable_evidence_assessment_actions"
        ][0]
        harness.execute_tool(
            "assess_evidence_candidate",
            action["arguments"],
            f"ledger-{action['arguments']['evidence_id']}",
        )
    state = harness.render_state()
    arguments = state["state_action_affordances"][
        "executable_adjudication_actions"
    ][0]["arguments"]
    frontier = _adjudication_reasoning_frontier(
        state, [{"tool_name": "adjudicate_evidence", "arguments": arguments}]
    )
    assert frontier is not None
    assert frontier["rejected_alternatives_externalized_in_complete_ledger"] is True
    assert frontier["required_atom_mentions"] == [
            {"reasoning_handle": "@E1", "evidence_id": "EA-ACTION-001"}
    ]

    concise = (
        "I compare and resolve the visible evidence partition, retaining `@E1` because "
        "its preferred_video=c is the abstaining direction supported by two independent "
        "methods rather than by the old claim. The completed per-candidate assessment "
        "ledger records every rejected alternative and the native call preserves their "
        "exact IDs and codes before claim revision or binding."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, concise, visible_state=state
    ) == []

    missing_ledger_reference = (
        "I compare and resolve the visible evidence partition, retaining `@E1` because its "
        "direction and intervals fit the claim. The native action now records the partition."
    )
    assert "adjudication_complete_ledger_not_referenced" in _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, missing_ledger_reference, visible_state=state
    )


def test_uniform_timeline_cannot_claim_global_motion_or_order_observation() -> None:
    bad = (
        "The remaining gap is complementary coverage, so I select compare_uniform_timeline. "
        "It tests the same global-motion comparison with broader temporal coverage and can "
        "resolve the remaining order-consistency doubt."
    )
    assert (
        _measurement_route_semantic_problem("compare_uniform_timeline", bad)
        == "uniform_timeline_overclaims_motion_or_order_observation"
    )
    good = (
        "The remaining gap is complementary coarse visual coverage, so I select "
        "compare_uniform_timeline. It inspects uniformly sampled chronological frames across "
        "the whole timeline while leaving A/B order consistency unobserved."
    )
    assert _measurement_route_semantic_problem("compare_uniform_timeline", good) is None
    disguised = (
        "The remaining gap needs another view. A new measurement can test whether the "
        "whole-video motion direction is consistent under both A/B orders. I will run "
        "compare_uniform_timeline for coarse coverage."
    )
    assert _measurement_route_semantic_problem(
        "compare_uniform_timeline", disguised
    ) == "uniform_timeline_overclaims_motion_or_order_observation"
    motion_dynamics = (
        "Low confidence warrants an independent route. compare_uniform_timeline will test "
        "whether A's faster-movement direction holds across the whole scope and whether "
        "B's motion is more stop-and-go."
    )
    assert _measurement_route_semantic_problem(
        "compare_uniform_timeline", motion_dynamics
    ) == "uniform_timeline_overclaims_motion_or_order_observation"


def test_vcsl_localizer_does_not_claim_viddiff_order_symmetry() -> None:
    bad = (
        "localize_copy_segment_dp measures temporal copy boxes under both A/B orders "
        "to test whether the direction survives role swapping."
    )
    assert _measurement_route_semantic_problem(
        "localize_copy_segment_dp", bad
    ) == "vcsl_localizer_overclaims_ab_order_symmetry"
    good = (
        "localize_copy_segment_dp runs dynamic programming on the query/reference "
        "similarity map to localize copied temporal segments."
    )
    assert _measurement_route_semantic_problem("localize_copy_segment_dp", good) is None


def test_literal_vcsl_localizer_name_satisfies_method_identification() -> None:
    reasoning = (
        "The missing gap is query/reference temporal localization. I select "
        "localize_copy_segment_dp because it will inspect the feature-similarity map "
        "to estimate copied segment boundaries."
    )
    assert _chosen_measurement_tool_problem(
        "localize_copy_segment_dp", reasoning
    ) is None


def test_latest_identity_atom_cannot_inherit_viddiff_comparison_fields() -> None:
    evidence_id = "EA-VCSL-PAIR-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {"query_id": "q1", "reference_id": "r1"},
            }
        },
    }
    bad = (
        "The latest observation established an active atom with preferred_video=c, "
        "order_consistent=false, and confidence=0.7."
    )
    problems = _visible_observation_fact_problems(bad, state)
    assert (
        f"latest_created_atom_asserts_absent_field:{evidence_id}:preferred_video"
        in problems
    )
    assert (
        f"latest_created_atom_asserts_absent_field:{evidence_id}:order_consistent"
        in problems
    )
    assert f"latest_created_atom_asserts_absent_field:{evidence_id}:confidence" in problems
    assert "has no student-visible" in _generation_retry_feedback(problems)


def test_reasoning_cannot_deny_a_nonempty_visible_evidence_store() -> None:
    state = {
        "evidence_store": {
            "EA-PAIR-001": {
                "active": True,
                "observation": {"source_id": "s1", "suspect_id": "x1"},
            },
            "EA-SCOPE-001": {
                "active": True,
                "observation": {"comparison_scope_s": [0.0, 15.0]},
            },
        }
    }
    bad = (
        "No atom has been established yet, so detect_temporal_edit must inspect "
        "the visible temporal properties."
    )
    assert "visible_evidence_store_denied_nonempty:actual_2" in (
        _visible_observation_fact_problems(bad, state)
    )
    good = (
        "The two context atoms establish identity and scope, but no temporal-edit "
        "measurement exists yet, so detect_temporal_edit closes that observation gap."
    )
    assert not any(
        item.startswith("visible_evidence_store_denied_nonempty:")
        for item in _visible_observation_fact_problems(good, state)
    )
    feedback = _generation_retry_feedback(
        ["visible_evidence_store_denied_nonempty:actual_2"]
    )
    assert "evidence_store is not empty" in feedback


def test_commit_reasoning_cannot_use_an_irrelevant_rebinding_constraint() -> None:
    state = {
        "action_constraints": {"must_reopen_before_rebinding": []},
        "candidate_claims": [
            {
                "claim_id": "claim-001",
                "fields": {
                    "relationship": {
                        "evidence_status": "supported",
                        "decision_status": "open",
                    }
                },
            }
        ],
    }
    bad = (
        "The supported relationship field remains open. Committing it is allowed by "
        "must_reopen_before_rebinding."
    )
    assert "irrelevant_rebinding_constraint_justifies_decision_transition" in (
        _reasoning_process_artifact_problems("set_field_decision", bad, state)
    )
    good = (
        "The relationship field has active support but its decision status remains open. "
        "Committing that supported field closes the decision-state gap."
    )
    assert "irrelevant_rebinding_constraint_justifies_decision_transition" not in (
        _reasoning_process_artifact_problems("set_field_decision", good, state)
    )
    assert "actual repair bind" in _generation_retry_feedback(
        ["irrelevant_rebinding_constraint_justifies_decision_transition"]
    )


def test_reasoning_cannot_quote_prompt_scaffolding_as_evidence() -> None:
    bad = (
        "The required rejection grounding clause names the mismatch. "
        "I reject this atom to close the pending assessment gap."
    )
    problems = _reasoning_process_artifact_problems(
        "assess_evidence_candidate", bad, {}
    )
    assert "reasoning_mentions_prompt_scaffolding" in problems
    compact_bad = (
        "The required compact grounding clauses are Retain @E1 because sample_key=x matches "
        "the claim. I adjudicate the partition to close the remaining decision gap."
    )
    assert "reasoning_mentions_prompt_scaffolding" in (
        _reasoning_process_artifact_problems(
            "adjudicate_evidence", compact_bad, {}
        )
    )
    clause_less_bad = (
        "The required rejection grounding already lists the defect. "
        "I reject the atom and record the assessment."
    )
    assert "reasoning_mentions_prompt_scaffolding" in (
        _reasoning_process_artifact_problems(
            "assess_evidence_candidate", clause_less_bad, {}
        )
    )
    feedback = _generation_retry_feedback(problems)
    assert "State the observed evidence facts" in feedback

    good = (
        "EA-001 reports preferred_video=a, whereas the claim records preferred_video=c. "
        "I reject this conflicting atom to close its pending assessment gap."
    )
    assert "reasoning_mentions_prompt_scaffolding" not in (
        _reasoning_process_artifact_problems(
            "assess_evidence_candidate", good, {}
        )
    )


def test_reasoning_cannot_expose_internal_policy_view_keys() -> None:
    bad = (
        "The current_harness_state lists bind_evidence in eligible_tools, so I select it. "
        "The execution_frontier supplies the arguments."
    )
    problems = _reasoning_process_artifact_problems("bind_evidence", bad, {})
    assert "reasoning_mentions_internal_policy_scaffolding" in problems
    feedback = _generation_retry_feedback(problems)
    assert "Describe the case evidence" in feedback

    good = (
        "EA-001 supports the open relationship field, but that support is not yet linked to "
        "the claim. bind_evidence creates the missing evidence link."
    )
    assert "reasoning_mentions_internal_policy_scaffolding" not in (
        _reasoning_process_artifact_problems("bind_evidence", good, {})
    )


def test_reasoning_cannot_repeat_episode_wide_gap_scaffolds() -> None:
    prior = [
        "The latest visible state contains EA-001. The only remaining gap is binding."
    ]
    current = (
        "The latest visible state now has support. The only remaining gap is commitment."
    )
    problems = _repetitive_reasoning_style_problems(current, prior)
    assert "repeated_reasoning_scaffold:latest_visible_state" in problems
    assert "repeated_reasoning_scaffold:only_remaining_gap" in problems
    assert "different sentence structure" in _generation_retry_feedback(problems)

    natural = (
        "EA-001 now supports the relationship field, but its decision remains open. "
        "set_field_decision commits that supported value."
    )
    assert _repetitive_reasoning_style_problems(natural, prior) == []


def test_pending_assessment_cannot_be_described_as_already_recorded() -> None:
    evidence_id = "EA-PENDING-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "kind": "comparison_task_identity",
                "supports_fields": ["sample_key"],
                "observation": {"sample_key": "case-1"},
            }
        },
        "evidence_assessment": {"pending_evidence_ids": [evidence_id]},
        "evidence_assessments": {"CC-1": {}},
        "candidate_claims": [],
    }
    arguments = {
        "claim_id": "CC-1",
        "evidence_id": evidence_id,
        "target_fields": ["sample_key"],
        "disposition": "retain",
        "defect_codes": [],
    }
    bad = (
        f"{evidence_id} has sample_key=case-1 and was already retained in the ledger. "
        "I record the retain disposition for this identity atom."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        bad,
        visible_state=state,
    )
    assert (
        f"pending_assessment_described_as_already_recorded:{evidence_id}" in problems
    )


def test_stop_can_name_terminal_validation_as_the_final_gap() -> None:
    reasoning = (
        "All six claim fields are committed and supported by active evidence, with no open "
        "high-priority obligations. No new measurement or rebinding remains, so terminal "
        "validation is the only eligible next action."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "stop",
        {},
        meta_visible=False,
        visible_state={},
    )
    assert "reasoning_does_not_name_decision_gap" not in problems


def test_atomic_assessment_decision_does_not_require_checker_like_gap_phrase() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-GLOBAL-MOTION-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                    "decision_rule": "order_disagreement_abstain",
                },
            },
            "EA-VIDDIFF-COMPARE-SYMMETRY-RESIDUAL-ABSTENTION-001": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_symmetry_residual_abstention",
                    "preferred_video": "c",
                    "order_consistent": True,
                },
            },
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "reject",
        "defect_codes": ["order_inconsistent"],
    }
    reasoning = (
        f"I reject `{evidence_id}`, which reports preferred_video=c and "
        "order_consistent=false. Its direction disagrees across the two A/B presentation "
        "orders, so the atom abstains and is order-inconsistent rather than usable support "
        "for the proposed comparison claim."
    )
    problems = _preexecution_reasoning_problems(
        reasoning,
        "assess_evidence_candidate",
        arguments,
        meta_visible=True,
        visible_state=state,
    )
    assert "reasoning_does_not_name_decision_gap" not in problems
    assert not any(item.startswith("latest_atom_order_consistency_mismatch:") for item in problems)
    assert problems == []


def test_typed_causal_repair_fixes_only_visible_order_abstention_edge() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-ACTION-REFINED-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "observation": {
                    "preferred_video": "c",
                    "order_consistent": False,
                    "decision_rule": "targeted_disagreement_abstain",
                }
            }
        }
    }
    reasoning = (
        f"I reject `{evidence_id}` with preferred_video=c and order_consistent=false. "
        "Its interval conflicts with the claim, and the order-disagreement abstention "
        "rule makes its A/B direction unstable."
    )
    repaired, repairs = _repair_typed_order_abstention_causal_edge(
        reasoning,
        "assess_evidence_candidate",
        {"evidence_id": evidence_id},
        state,
    )
    assert repairs == ["typed_order_abstention_causal_edge_repaired"]
    assert "direction disagrees across the two A/B presentation orders" in repaired
    assert "atom abstains with preferred_video=c" in repaired
    unchanged, no_repairs = _repair_typed_order_abstention_causal_edge(
        reasoning,
        "bind_evidence",
        {"evidence_id": evidence_id},
        state,
    )
    assert unchanged == reasoning
    assert no_repairs == []


def test_dropout_reasoning_cannot_reuse_binding_invalidation_semantics() -> None:
    state = {"dropped_evidence_ids": ["EA-DROPPED-001"]}
    bad = (
        "The remaining gap is commitment after the evidence-dropout mutation removed an atom. "
        "The same active atom remains valid and is intended to be rebound."
    )
    assert "evidence_dropout_misdescribed_as_binding_invalidation" in (
        _reasoning_process_artifact_problems("set_field_decision", bad, state)
    )
    subtly_bad = (
        "The evidence_dropout removed the old measurement, so reopening restores the ability "
        "to rebind the same atom or a replacement later."
    )
    problems = _reasoning_process_artifact_problems(
        "set_field_decision", subtly_bad, state
    )
    assert "dropped_atom_described_as_rebindable" in problems
    assert "cannot be rebound" in _generation_retry_feedback(problems)
    good = (
        "The old EvidenceAtom was dropped and remains invalid. The remaining gap is commitment "
        "after binding a newly generated active EvidenceAtom with a distinct ID."
    )
    assert _reasoning_process_artifact_problems("set_field_decision", good, state) == []


def test_visible_repair_restoration_candidate_is_not_bypassed() -> None:
    state = {
        "state_action_affordances": {
            "repair_restoration_candidates": [
                {"tool_name": "compare_global_motion_symmetric"}
            ]
        }
    }
    assert (
        _repair_restoration_choice_problem("compare_action_refined_symmetric", state)
        == "repair_restoration_candidate_bypassed:compare_global_motion_symmetric"
    )
    assert _repair_restoration_choice_problem(
        "compare_global_motion_symmetric", state
    ) is None


def test_visible_atom_support_is_distinct_from_missing_claim_binding() -> None:
    state = {
        "evidence_store": {
            "EA-GLOBAL-002": {
                "active": True,
                "supports_fields": ["preferred_video", "measurement_method"],
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_global_motion_symmetric",
                },
            }
        }
    }
    bad = (
        "`EA-GLOBAL-002` reports preferred_video=a. The remaining gap is that this atom "
        "does not yet support the open claim fields, so another measurement is required."
    )
    assert "visible_atom_declared_support_denied:EA-GLOBAL-002" in (
        _visible_observation_fact_problems(bad, state)
    )
    good = (
        "`EA-GLOBAL-002` reports preferred_video=a and supports the comparison fields. "
        "The remaining gap is independent corroboration before binding because confidence is low."
    )
    assert _visible_observation_fact_problems(good, state) == []


def test_active_measurement_cannot_be_denied_across_later_sentence() -> None:
    state = {
        "evidence_store": {
            "EA-GLOBAL-002": {
                "active": True,
                "supports_fields": ["preferred_video", "measurement_method"],
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_global_motion_symmetric",
                },
            }
        }
    }
    contradictory = (
        "`EA-GLOBAL-002` is active and supports A as faster. The blocking gap is that the "
        "current state has no active evidence for the open fields, so a new measurement is "
        "needed to reestablish support."
    )
    assert "active_measurement_evidence_denied_for_fields" in (
        _visible_observation_fact_problems(contradictory, state)
    )
    replacement_denial = (
        "`EA-GLOBAL-002` is active and supports A as faster, but no new evidence has replaced "
        "the dropped atom. The blocking gap is a missing measurement for the same statement."
    )
    assert "active_measurement_evidence_denied_for_fields" in (
        _visible_observation_fact_problems(replacement_denial, state)
    )
    dropout_denial = (
        "The dropped global-motion atom left the four claim fields unsupported, so a new "
        "measurement must restore support."
    )
    assert "active_measurement_evidence_denied_for_fields" in (
        _visible_observation_fact_problems(dropout_denial, state)
    )


def test_visible_order_consistency_cannot_be_described_as_missing() -> None:
    state = {
        "evidence_store": {
            "EA-GLOBAL-002": {
                "active": True,
                "supports_fields": ["preferred_video"],
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_global_motion_symmetric",
                    "order_consistent": True,
                },
            }
        }
    }
    reasoning = (
        "`EA-GLOBAL-002` supports A but reports low confidence and no order_consistent field."
    )
    assert (
        "latest_atom_order_consistency_mismatch:EA-GLOBAL-002:unobserved!=true"
        in _visible_observation_fact_problems(reasoning, state)
    )
    state["evidence_store"]["EA-GLOBAL-002"]["observation"][
        "order_consistent"
    ] = None
    negated_defect = (
        "`EA-GLOBAL-002` provides a prior view without an observed "
        "order-inconsistency defect."
    )
    assert _visible_observation_fact_problems(negated_defect, state) == []


def test_unbound_claim_method_value_cannot_be_called_a_binding() -> None:
    state = {
        "field_bindings": {
            "claim-1": {
                "measurement_method": {
                    "support_ids": [],
                    "validation_status": "invalidated_by_environment_mutation",
                }
            }
        }
    }
    reasoning = (
        "The remaining gap is fresh evidence after dropout, while the current claim still "
        "binds `compare_uniform_timeline` as the method."
    )
    problems = _reasoning_process_artifact_problems(
        "compare_uniform_timeline", reasoning, state
    )
    assert "unbound_claim_method_described_as_bound" in problems
    assert "value remains" in _generation_retry_feedback(problems)

    state["field_bindings"]["claim-1"]["measurement_method"] = {
        "support_ids": ["EA-MEASUREMENT-001"],
        "validation_status": "accepted",
    }
    assert _reasoning_process_artifact_problems(
        "compare_uniform_timeline", reasoning, state
    ) == []


def test_visible_stable_direction_cannot_be_denied() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_UNIFORM-TIMELINE-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": ["preferred_video", "measurement_method"],
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                    "confidence": 0.61,
                    "order_consistent": None,
                    "routing_state": {"stable_observed_directions": ["a"]},
                },
            }
        }
    }
    reasoning = (
        f"`{evidence_id}` has preferred_video=a but no stable direction. "
        "Its low confidence leaves a need for independent corroboration."
    )
    problems = _visible_observation_fact_problems(reasoning, state)
    assert f"latest_atom_stable_direction_denied:{evidence_id}:a" in problems
    assert "Do not deny that stable direction" in _generation_retry_feedback(problems)


def test_assessment_does_not_confuse_router_stop_with_abstention_cause() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-GLOBAL-MOTION-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "confidence": 0.70,
                    "order_consistent": False,
                    "decision_rule": "order_disagreement_abstain",
                    "routing_state": {"reason_code": "all_candidates_exhausted"},
                },
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "reject",
        "defect_codes": ["order_inconsistent"],
    }
    reasoning = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "It abstains because all candidates are exhausted."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert (
        f"assessment_abstention_cause_confuses_routing_termination:{evidence_id}"
        in problems
    )
    assert "router stopped searching" in _generation_retry_feedback(problems)


def test_assessment_abstention_cause_must_follow_visible_decision_rule() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-GLOBAL-MOTION-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                    "decision_rule": "order_disagreement_abstain",
                    "routing_state": {"reason_code": "cross_method_disagreement"},
                },
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "reject",
        "defect_codes": ["order_inconsistent"],
    }
    reasoning = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "It abstains because of cross-method disagreement."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert (
        f"assessment_abstention_cause_omits_decision_rule:{evidence_id}" in problems
    )
    assert "direct decision-rule cause" in _generation_retry_feedback(problems)

    grounded = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "This order-inconsistent result abstains because the direction disagrees across "
        "A/B presentation orders."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, grounded, visible_state=state
    ) == []


def test_machine_abstention_token_is_not_a_natural_causal_explanation() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-ACTION-REFINED-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                    "decision_rule": "targeted_disagreement_abstain",
                },
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "reject",
        "defect_codes": ["order_inconsistent"],
    }
    reasoning = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "It abstains for targeted_disagreement_abstain and is order-inconsistent."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert (
        f"assessment_abstention_decision_rule_not_explained:{evidence_id}"
        in problems
    )

    causal_noun_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The atom is order-inconsistent, and the visible abstention cause is disagreement "
        "across A/B presentation orders."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        causal_noun_phrase,
        visible_state=state,
    ) == []

    passive_causal_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The atom is order-inconsistent, and its abstention is caused by disagreement "
        "across the two A/B presentation orders."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        passive_causal_phrase,
        visible_state=state,
    ) == []

    subject_first_causal_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The atom is order-inconsistent because A/B presentation-order disagreement "
        "causes the abstention."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        subject_first_causal_phrase,
        visible_state=state,
    ) == []
    order_first_causal_noun_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The direct abstention cause is that its A/B presentation-order disagreement "
        "makes the direction unstable, so the atom is order-inconsistent."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        order_first_causal_noun_phrase,
        visible_state=state,
    ) == []
    due_to_causal_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The atom is order-inconsistent: its A/B direction is unstable due to the "
        "presentation-order disagreement that caused the abstention."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        due_to_causal_phrase,
        visible_state=state,
    ) == []
    structured_rule_phrase = (
        f"I reject `{evidence_id}`: preferred_video=c with order_consistent=false under "
        "an order-disagreement abstention rule. That A/B-order disagreement leaves this "
        "order-inconsistent atom without a stable direction for the claim."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        structured_rule_phrase,
        visible_state=state,
    ) == []
    assert "directed cause and effect" in _generation_retry_feedback(problems)

    abstention_without_cause = (
        f"I reject `{evidence_id}`: preferred_video=c and order_consistent=false. "
        "The abstention conflicts with the claim and the atom is order-inconsistent."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        arguments,
        abstention_without_cause,
        visible_state=state,
    )
    assert (
        f"assessment_abstention_decision_rule_not_explained:{evidence_id}"
        in problems
    )


def test_retained_null_order_is_a_limitation_not_the_action_gap() -> None:
    evidence_id = "EA-UNIFORM-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                    "order_consistent": None,
                }
            }
        }
    }
    arguments = {
        "evidence_id": evidence_id,
        "disposition": "retain",
        "defect_codes": [],
    }
    reasoning = (
        f"I retain `{evidence_id}` because preferred_video=a matches the claim and "
        "order_consistent=null is unobserved. The only remaining gap is the missing "
        "order-consistency signal."
    )
    problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate", arguments, reasoning, visible_state=state
    )
    assert (
        f"assessment_retained_uncertainty_misframed_as_action_gap:{evidence_id}"
        in problems
    )


def test_progressive_adjudication_retry_view_is_compact_and_fact_complete() -> None:
    selected_id = "EA-UNIFORM-001"
    rejected_id = "EA-GLOBAL-001"
    payload = {
        "task": {"task": "repair"},
        "current_harness_state": {
            "candidate_claims": [
                {
                    "claim_id": "CC-1",
                    "fields": {
                        "preferred_video": {
                            "value": "a",
                            "evidence_status": "unsupported",
                            "decision_status": "open",
                            "support_ids": [],
                        }
                    },
                }
            ],
            "evidence_store": {
                selected_id: {
                    "active": True,
                    "supports_fields": ["preferred_video"],
                    "observation": {
                        "preferred_video": "a",
                        "measurement_method": "compare_uniform_timeline",
                        "order_consistent": None,
                        "visible_rationale": "large text that should be omitted",
                    },
                },
                rejected_id: {
                    "active": True,
                    "supports_fields": ["preferred_video"],
                    "observation": {
                        "preferred_video": "c",
                        "measurement_method": "compare_global_motion_symmetric",
                        "order_consistent": False,
                        "decision_rule": "order_disagreement_abstain",
                    },
                },
                "EA-UNRELATED-001": {
                    "active": True,
                    "observation": {"relationship": "unrelated"},
                },
            },
            "evidence_assessments": {"CC-1": {}},
            "evidence_assessment": {"pending_evidence_ids": []},
            "evidence_adjudication": {"required": True},
        },
        "execution_frontier": {
            "eligible_tools": ["adjudicate_evidence"],
            "planning_instruction": "adjudicate",
        },
        "evidence_adjudication_frontier": {
            "selected_support": [
                {
                    "evidence_id": selected_id,
                    "reasoning_handle": "@E1",
                    "preferred_video": "a",
                    "order_consistent": None,
                    "required_minimal_grounding_clause": (
                        f"Retain {selected_id} because preferred_video=\"a\" matches the claim."
                    ),
                    "required_compact_grounding_clause": (
                        "Retain @E1 because preferred_video=\"a\" matches the claim."
                    ),
                }
            ],
            "rejected_alternatives": [
                {
                    "evidence_id": rejected_id,
                    "preferred_video": "c",
                    "order_consistent": False,
                    "validated_defect_codes": ["order_inconsistent"],
                }
            ],
        },
    }
    compact = _compact_progressive_decision_retry_payload(
        payload, "adjudicate_evidence"
    )
    store = compact["current_harness_state"]["evidence_store"]
    assert set(store) == {selected_id, rejected_id}
    assert store[selected_id]["observation"]["order_consistent"] is None
    assert "visible_rationale" not in store[selected_id]["observation"]
    plan = compact["grounded_reasoning_plan"]
    assert plan["selected_support"][0]["evidence_id"] == selected_id
    assert plan["full_selected_support_facts"] == [
        f"Retain {selected_id} because preferred_video=\"a\" matches the claim."
    ]
    assert plan["selected_support_facts"] == [
        "Retain @E1 because preferred_video=\"a\" matches the claim."
    ]
    assert "Never change order_consistent=null" in plan["forbidden_inferences"][0]
    initial_message = _initial_progressive_decision_message(
        {"role": "user", "content": json.dumps(payload)}, "adjudicate_evidence"
    )
    initial = json.loads(initial_message["content"])
    assert initial["initial_decision_view"]["selected_support_facts_to_state"] == [
        "Retain @E1 because preferred_video=\"a\" matches the claim."
    ]
    assert "required_compact_grounding_clauses" not in json.dumps(initial)
    assert set(initial["evidence_adjudication_frontier"]) == {
        "decision_gap",
        "rejected_alternatives_externalized_in_complete_ledger",
    }


def test_initial_assessment_view_preserves_identity_facts() -> None:
    evidence_id = "EA-TASK-001"
    payload = {
        "task": {"claim_schema_id": "schema"},
        "current_harness_state": {
            "candidate_claims": [],
            "evidence_store": {
                evidence_id: {
                    "supports_fields": ["sample_key", "comparison_statement"],
                    "observation": {
                        "sample_key": "surgery_26",
                        "comparison_statement": "The tube moves more",
                    },
                }
            },
            "evidence_assessments": {},
            "evidence_assessment": {"pending_evidence_ids": [evidence_id]},
            "evidence_adjudication": {"required": True},
        },
        "execution_frontier": {"eligible_tools": ["assess_evidence_candidate"]},
        "evidence_assessment_frontier": {
            "pending_candidate_assessments": [
                {
                    "evidence_id": evidence_id,
                    "measurement_semantics_required": False,
                    "supported_observation_facts": {
                        "sample_key": "surgery_26",
                        "comparison_statement": "The tube moves more",
                    },
                }
            ]
        },
    }
    message = _initial_progressive_decision_message(
        {"role": "user", "content": json.dumps(payload)},
        "assess_evidence_candidate",
    )
    compact = json.loads(message["content"])
    observation = compact["current_harness_state"]["evidence_store"][evidence_id][
        "observation"
    ]
    assert observation == {
        "sample_key": "surgery_26",
        "comparison_statement": "The tube moves more",
    }
    assert compact["initial_decision_view"]["unique_eligible_tool"] == (
        "assess_evidence_candidate"
    )


def test_order_disagreement_assessment_exposes_verbatim_causal_clause() -> None:
    evidence_id = "EA-ACTION-001"
    state = {
        "candidate_claims": [],
        "evidence_store": {
            evidence_id: {
                "kind": "action_comparison_measurement",
                "supports_fields": ["preferred_video", "measurement_method"],
                "observation": {
                    "preferred_video": "c",
                    "measurement_method": "compare_action_refined_symmetric",
                    "order_consistent": False,
                    "decision_rule": "targeted_disagreement_abstain",
                },
            }
        },
        "evidence_assessments": {},
        "evidence_assessment": {"pending_evidence_ids": [evidence_id]},
        "evidence_adjudication": {"required": True},
    }
    actions = [
        {
            "tool_name": "assess_evidence_candidate",
            "arguments": {
                "claim_id": "CC-1",
                "target_fields": ["preferred_video", "measurement_method"],
                "evidence_id": evidence_id,
                "disposition": "reject",
                "defect_codes": ["order_inconsistent"],
            },
        }
    ]
    frontier = _evidence_assessment_reasoning_frontier(state, actions)
    assert frontier is not None
    clause = (
        "This atom abstains because its direction disagrees across the two A/B "
        "presentation orders."
    )
    assert frontier["pending_candidate_assessments"][0][
        "required_abstention_causal_clause"
    ] == clause
    assert frontier["pending_candidate_assessments"][0]["causal_edges"] == [
        {
            "cause": "direction_disagreement_across_two_ab_presentation_orders",
            "effect": "preferred_video_c_abstention",
            "forbid_reverse_relation": True,
        }
    ]
    assert frontier["pending_candidate_assessments"][0][
        "required_measurement_semantics_clause"
    ] == (
        f'{evidence_id} reports preferred_video="c" and order_consistent=false.'
    )
    assert frontier["pending_candidate_assessments"][0][
        "required_rejection_grounding_clauses"
    ] == [
        f"Reject {evidence_id} because order_consistent=false makes its A/B direction unstable."
    ]
    decision_gap_clause = (
        f"The visible decision gap is the unrecorded assessment for {evidence_id}."
    )
    assert frontier["pending_candidate_assessments"][0][
        "required_decision_gap_clause"
    ] == decision_gap_clause
    payload = {
        "task": {},
        "current_harness_state": state,
        "execution_frontier": {"eligible_tools": ["assess_evidence_candidate"]},
        "evidence_assessment_frontier": frontier,
    }
    message = _initial_progressive_decision_message(
        {"role": "user", "content": json.dumps(payload)},
        "assess_evidence_candidate",
    )
    compact = json.loads(message["content"])
    assert compact["grounded_reasoning_plan"][
        "required_abstention_causal_clause"
    ] == clause
    assert compact["initial_decision_view"][
        "facts_to_express_naturally_before_tool_call"
    ] == [decision_gap_clause, clause]
    assert "must_copy_verbatim_before_tool_call" not in compact[
        "initial_decision_view"
    ]


def test_nonassessment_reasoning_must_preserve_abstention_cause() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE-GLOBAL-MOTION-SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                    "order_consistent": False,
                    "decision_rule": "order_disagreement_abstain",
                },
            }
        }
    }
    reasoning = (
        f"`{evidence_id}` abstains because B has clearer pixel motion. "
        "A localized route is the next action."
    )
    problems = _visible_observation_fact_problems(reasoning, state)
    assert f"latest_atom_abstention_cause_omits_decision_rule:{evidence_id}" in problems

    cross_method_is_not_the_cause = (
        f"`{evidence_id}` abstains because of cross-method disagreement and "
        "order_consistent=false. A localized route is the next action."
    )
    assert f"latest_atom_abstention_cause_omits_decision_rule:{evidence_id}" in (
        _visible_observation_fact_problems(cross_method_is_not_the_cause, state)
    )
    intervening_fields_do_not_hide_the_wrong_cause = (
        f"`{evidence_id}` abstains with preferred_video=c because both videos have weak "
        "tube motion. It is the strongest basis without order inconsistency."
    )
    intervening_problems = _visible_observation_fact_problems(
        intervening_fields_do_not_hide_the_wrong_cause, state
    )
    assert f"latest_atom_abstention_cause_omits_decision_rule:{evidence_id}" in (
        intervening_problems
    )
    assert f"visible_atom_order_consistency_denied:{evidence_id}:false" in (
        intervening_problems
    )

    grounded = (
        f"`{evidence_id}` abstains because the direction disagrees across A/B presentation "
        "orders. A localized route is the next action."
    )
    assert _visible_observation_fact_problems(grounded, state) == []


def test_visible_multiword_quoted_scalar_is_compared_as_a_whole_value() -> None:
    evidence_id = "EA-TASK-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "comparison_statement": "The tube moves more",
                },
            }
        }
    }
    good = (
        f'Retain `{evidence_id}` because comparison_statement="The tube moves more" '
        "matches the claim."
    )
    assert not any(
        item.startswith("visible_atom_scalar_mismatch:")
        for item in _visible_observation_fact_problems(good, state)
    )
    bad = (
        f'Retain `{evidence_id}` because comparison_statement="The tool moves less" '
        "matches the claim."
    )
    assert any(
        item.startswith(
            f"visible_atom_scalar_mismatch:{evidence_id}:comparison_statement:"
        )
        for item in _visible_observation_fact_problems(bad, state)
    )


def test_claim_creation_cannot_pretend_to_bind_or_commit_support() -> None:
    state = {"candidate_claims": []}
    bad = (
        "The latest measurement provides the candidate values. I create the claim to turn "
        "this measurement into a supported claim."
    )
    assert "claim_creation_described_as_binding_or_commitment" in (
        _reasoning_process_artifact_problems("create_candidate_claim", bad, state)
    )
    good = (
        "The latest measurement provides candidate values. Creating the claim records field "
        "obligations without binding support or committing a decision."
    )
    assert "claim_creation_described_as_binding_or_commitment" not in (
        _reasoning_process_artifact_problems("create_candidate_claim", good, state)
    )
    assert "creates field values and obligations only" in _generation_retry_feedback(
        ["claim_creation_described_as_binding_or_commitment"]
    )


def test_binding_cannot_be_described_as_completing_assessment() -> None:
    bad = (
        "The selected evidence is supported but unbound. Binding those exact atoms closes the "
        "candidate-assessment step without committing the fields."
    )
    problems = _reasoning_process_artifact_problems("bind_evidence", bad, {})
    assert "binding_described_as_assessment_transition" in problems
    good = (
        "The selected evidence is supported but unbound. bind_evidence creates the exact "
        "claim-field-to-EvidenceAtom links without committing any field."
    )
    assert "binding_described_as_assessment_transition" not in (
        _reasoning_process_artifact_problems("bind_evidence", good, {})
    )


def test_claim_creation_reasoning_identifies_the_actual_measurement_source() -> None:
    state = {
        "evidence_store": {
            "EA-UNIFORM-001": {
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                }
            },
            "EA-GLOBAL-001": {
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "c",
                }
            },
        }
    }
    arguments = {
        "field_values": {"measurement_method": "compare_uniform_timeline"}
    }
    ambiguous = (
        "The latest global-motion atom abstains. I create the claim from the strongest "
        "visible measurement without committing fields."
    )
    assert _claim_creation_source_problem(
        "create_candidate_claim", arguments, ambiguous, state
    ) == "claim_creation_source_not_identified:compare_uniform_timeline"
    grounded = (
        "I use `EA-UNIFORM-001` because its visible preferred_video=a supplies the claim "
        "values, then create the candidate without binding support."
    )
    assert _claim_creation_source_problem(
        "create_candidate_claim", arguments, grounded, state
    ) is None


def test_assessment_cannot_claim_that_retain_binds_field_support() -> None:
    bad = (
        "I select assess_evidence_candidate for the pending task atom. Retaining it closes "
        "the sample_key and comparison_statement support."
    )
    assert "assessment_described_as_binding_field_support" in (
        _reasoning_process_artifact_problems(
            "assess_evidence_candidate", bad, {"candidate_claims": []}
        )
    )
    good = (
        "I select assess_evidence_candidate for the pending task atom. Retaining it closes "
        "only this candidate's assessment gap; binding support remains a later transition."
    )
    assert "assessment_described_as_binding_field_support" not in (
        _reasoning_process_artifact_problems(
            "assess_evidence_candidate", good, {"candidate_claims": []}
        )
    )


def test_adjudication_requires_a_grounded_selection_reason() -> None:
    selected_id = "EA-VIDDIFF-COMPARE-UNIFORM-TIMELINE-001"
    arguments = {
        "claim_id": "CC-1",
        "target_fields": ["preferred_video"],
        "selected_evidence_ids": [selected_id],
        "rejected_evidence": [],
    }
    state = {
        "evidence_store": {
            selected_id: {
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                }
            }
        }
    }
    weak = (
        f"I select adjudicate_evidence to resolve the conflict. `{selected_id}` is retained."
    )
    assert "adjudication_selection_not_justified" in _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, weak, visible_state=state
    )
    grounded = (
        f"I select adjudicate_evidence to resolve the conflict. I retain `{selected_id}` "
        "because its visible preferred_video=a matches the claim."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, grounded, visible_state=state
    ) == []
    state["candidate_claims"] = [
        {
            "claim_id": "CC-1",
            "fields": {"preferred_video": {"value": "a"}},
        }
    ]
    feedback = _generation_retry_feedback(
        [f"adjudication_selected_id_not_cited:{selected_id}"],
        visible_state=state,
        arguments=arguments,
    )
    assert (
        f"Retain {selected_id} because preferred_video=\"a\" matches the claim."
        in feedback
    )


def test_adjudication_rejects_calling_rejected_alternatives_retained() -> None:
    selected_id = "EA-UNIFORM-001"
    rejected_id = "EA-GLOBAL-001"
    arguments = {
        "claim_id": "CC-1",
        "target_fields": ["preferred_video"],
        "selected_evidence_ids": [selected_id],
        "rejected_evidence": [
            {
                "evidence_id": rejected_id,
                "defect_codes": ["order_inconsistent"],
            }
        ],
    }
    state = {
        "evidence_store": {
            selected_id: {"observation": {"preferred_video": "a"}},
            rejected_id: {
                "observation": {
                    "preferred_video": "c",
                    "order_consistent": False,
                }
            },
        }
    }
    reasoning = (
        "I retain EA-UNIFORM-001 because preferred_video=a matches the claim. "
        "The rejected alternatives are retained because their dispositions are recorded."
    )
    problems = _adjudication_reasoning_problems(
        "adjudicate_evidence", arguments, reasoning, visible_state=state
    )
    assert "adjudication_rejected_alternatives_called_retained" in problems
    assert "Never call rejected alternatives retained" in _generation_retry_feedback(
        problems
    )


def test_adjudication_frontier_exposes_selected_claim_matching_facts() -> None:
    task_id = "EA-TASK-001"
    measurement_id = "EA-UNIFORM-001"
    state = {
        "candidate_claims": [
            {
                "claim_id": "CC-1",
                "fields": {
                    "sample_key": {"value": "surgery_26"},
                    "preferred_video": {"value": "a"},
                },
            }
        ],
        "evidence_store": {
            task_id: {
                "supports_fields": ["sample_key"],
                "observation": {"sample_key": "surgery_26"},
            },
            measurement_id: {
                "supports_fields": ["preferred_video"],
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_uniform_timeline",
                },
            },
        },
    }
    action = {
        "tool_name": "adjudicate_evidence",
        "arguments": {
            "claim_id": "CC-1",
            "target_fields": ["sample_key", "preferred_video"],
            "selected_evidence_ids": [task_id, measurement_id],
            "rejected_evidence": [],
        },
    }
    frontier = _adjudication_reasoning_frontier(state, [action])
    assert frontier is not None
    selected = {
        item["evidence_id"]: item for item in frontier["selected_support"]
    }
    assert selected[task_id]["claim_matching_facts"] == {
        "sample_key": "surgery_26"
    }
    assert selected[measurement_id]["claim_matching_facts"] == {
        "preferred_video": "a"
    }
    assert "sample_key=\"surgery_26\"" in selected[task_id][
        "required_minimal_grounding_clause"
    ]
    assert "preferred_video=\"a\"" in selected[measurement_id][
        "required_minimal_grounding_clause"
    ]
    assert "final selected/rejected evidence partition" in frontier["decision_gap"]


def test_missing_adjudication_selection_fact_can_keep_the_unique_action() -> None:
    assert _action_lock_safe(["adjudication_selection_not_justified"])
    assert _action_lock_safe(
        [
            "assessment_abstention_decision_rule_not_explained:EA-001",
            "assessment_rejection_defect_not_explained:EA-001",
        ]
    )
    assert _action_lock_safe(
        [
            "adjudication_selected_id_not_cited:EA-001",
            "adjudication_selection_not_justified",
        ]
    )
    assert not _action_lock_safe(
        ["latest_atom_preferred_video_mismatch:EA-UNIFORM-001:b!=a"]
    )


def test_measurement_gap_contract_externalizes_real_visible_uncertainty() -> None:
    state = {
        "evidence_store": {
            "EA-GLOBAL-002": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "a",
                    "confidence": 0.635,
                    "order_consistent": True,
                },
            }
        }
    }
    contract = _measurement_gap_contract(state, ["compare_uniform_timeline"])
    assert contract is not None
    assert contract["valid_observation_uncertainties"][0]["visible_uncertainties"] == [
        "low_confidence=0.635"
    ]
    bad = (
        "The blocking gap is that the claim fields still lack evidence, so "
        "compare_uniform_timeline will create a replacement."
    )
    assert _measurement_observation_gap_problem(
        "compare_uniform_timeline", bad, state
    ) == "measurement_reasoning_omits_visible_observation_uncertainty:compare_uniform_timeline"
    open_field_gap = (
        "The latest atom has low confidence, and the remaining open fields create a "
        "measurement gap. compare_uniform_timeline supplies an independent check."
    )
    assert _measurement_observation_gap_problem(
        "compare_uniform_timeline", open_field_gap, state
    ) == "measurement_action_denies_need_for_measurement:compare_uniform_timeline"
    circular_gap = (
        "The latest atom does not resolve the remaining gap because global motion is not "
        "the required tool. compare_uniform_timeline supplies an independent check."
    )
    assert _measurement_observation_gap_problem(
        "compare_uniform_timeline", circular_gap, state
    ) == "measurement_tool_policy_substituted_for_observation_gap:compare_uniform_timeline"
    good = (
        "Confidence=0.635 leaves uncertainty, so compare_uniform_timeline supplies an "
        "independent coarse chronological check."
    )
    assert _measurement_observation_gap_problem(
        "compare_uniform_timeline", good, state
    ) is None


def test_measurement_gap_contract_frontloads_abstention_causal_rule() -> None:
    evidence_id = "EA-ACTION-001"
    state = {
        "latest_environment_observation": {"created_evidence_ids": [evidence_id]},
        "evidence_store": {
            evidence_id: {
                "active": True,
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "confidence": 0.57,
                    "order_consistent": False,
                    "decision_rule": "order_disagreement_abstain",
                },
            }
        },
    }
    contract = _measurement_gap_contract(
        state, ["compare_global_motion_symmetric"]
    )
    assert contract is not None
    causal = contract["required_latest_atom_causal_explanation"]
    assert causal["decision_rule"] == "order_disagreement_abstain"
    assert "disagrees across the two A/B presentation orders" in (
        causal["required_natural_language"]
    )


def test_gap_contract_recommends_stronger_label_blind_corroboration_route() -> None:
    state = {
        "eligible_tools": [
            "compare_uniform_timeline",
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
        ],
        "latest_environment_observation": {"created_evidence_ids": ["EA-GLOBAL-002"]},
        "evidence_store": {
            "EA-GLOBAL-002": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_global_motion_symmetric",
                    "preferred_video": "a",
                    "confidence": 0.635,
                    "order_consistent": True,
                },
            },
            "EA-UNIFORM-001": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_uniform_timeline",
                    "preferred_video": "a",
                    "confidence": 0.8,
                    "order_consistent": None,
                },
            },
            "EA-ACTION-001": {
                "active": True,
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "confidence": 0.495,
                    "order_consistent": False,
                },
            },
        },
    }
    contract = _measurement_gap_contract(state, state["eligible_tools"])
    assert contract is not None
    assert contract["recommended_next_measurement"]["tool_name"] == (
        "compare_uniform_timeline"
    )
    payload = {
        "execution_frontier": {"measurement_gap_contract": contract}
    }
    assert _initial_recommended_tool(payload, state["eligible_tools"]) == (
        "compare_uniform_timeline"
    )
    assert _measurement_recommendation_problem(
        "compare_action_refined_symmetric", state
    ) == "student_visible_measurement_recommendation_bypassed:compare_uniform_timeline"
    assert _measurement_recommendation_problem("compare_uniform_timeline", state) is None
    assert _recommended_retry_tool(
        ["student_visible_measurement_recommendation_bypassed:compare_uniform_timeline"],
        state["eligible_tools"],
    ) == "compare_uniform_timeline"
    assert _recommended_retry_tool(
        ["student_visible_measurement_recommendation_bypassed:not_eligible"],
        state["eligible_tools"],
    ) is None
    assert _recommended_retry_tool(
        ["repair_restoration_candidate_bypassed:compare_uniform_timeline"],
        state["eligible_tools"],
    ) == "compare_uniform_timeline"
    assert _measurement_need_problem(
        "compare_uniform_timeline",
        "The remaining gap is that claim fields are still unsupported.",
    ) == "measurement_action_denies_need_for_measurement:compare_uniform_timeline"
    assert _measurement_need_problem(
        "compare_uniform_timeline",
        "The remaining gap is independent corroboration rather than claim binding.",
    ) is None
    false_conflict = (
        "compare_uniform_timeline and compare_global_motion_symmetric conflict on "
        "preferred_video, so another route is required."
    )
    assert (
        "measurement_pair_conflict_mismatch:compare_uniform_timeline:"
        "compare_global_motion_symmetric:a==a"
        in _visible_observation_fact_problems(false_conflict, state)
    )
    false_id_conflict = (
        "`EA-UNIFORM-001` and `EA-GLOBAL-002` conflict on preferred_video, "
        "so another measurement is required."
    )
    assert (
        "measurement_pair_conflict_mismatch:EA-UNIFORM-001:EA-GLOBAL-002:a==a"
        in _visible_observation_fact_problems(false_id_conflict, state)
    )


def test_recommended_measurement_retry_view_keeps_facts_and_drops_rationale() -> None:
    payload = {
        "task": {"task": "Compare A and B"},
        "current_harness_state": {
            "candidate_claims": [
                {
                    "claim_id": "CC-1",
                    "fields": {
                        "preferred_video": {
                            "value": "a",
                            "evidence_status": "unsupported",
                            "decision_status": "open",
                            "support_ids": [],
                        }
                    },
                }
            ],
            "evidence_store": {
                "EA-GLOBAL-002": {
                    "active": True,
                    "supports_fields": ["preferred_video"],
                    "observation": {
                        "preferred_video": "a",
                        "measurement_method": "compare_global_motion_symmetric",
                        "confidence": 0.635,
                        "visible_rationale": "large pixel-derived text must be omitted",
                    },
                },
                "EA-UNIFORM-001": {
                    "active": True,
                    "supports_fields": ["preferred_video"],
                    "observation": {
                        "preferred_video": "a",
                        "measurement_method": "compare_uniform_timeline",
                        "confidence": 0.8,
                        "order_consistent": None,
                        "visible_rationale": "another long rationale to omit",
                    },
                }
            },
            "dropped_evidence_ids": ["EA-GLOBAL-001"],
        },
        "latest_environment_observation": {"created_evidence_ids": ["EA-GLOBAL-002"]},
        "execution_frontier": {
            "eligible_tools": [
                "compare_uniform_timeline",
                "compare_action_refined_symmetric",
            ],
            "tool_purposes": {
                "compare_uniform_timeline": "coarse chronological coverage",
                "compare_action_refined_symmetric": "local action coverage",
            },
            "native_action_templates": [],
            "measurement_gap_contract": {
                "valid_observation_uncertainties": [
                    {
                        "evidence_id": "EA-GLOBAL-002",
                        "measurement_method": "compare_global_motion_symmetric",
                        "visible_uncertainties": ["low_confidence=0.635"],
                    }
                ]
            },
        },
    }
    compact = _compact_recommended_measurement_retry_payload(
        payload, "compare_uniform_timeline"
    )
    assert compact["execution_frontier"]["eligible_tools"] == [
        "compare_uniform_timeline",
        "compare_action_refined_symmetric",
    ]
    assert compact["execution_frontier"]["required_tool_for_revision"] == (
        "compare_uniform_timeline"
    )
    assert compact["current_harness_state"]["eligible_tools"] == [
        "compare_uniform_timeline",
        "compare_action_refined_symmetric",
    ]
    observation = compact["current_harness_state"]["evidence_store"][
        "EA-GLOBAL-002"
    ]["observation"]
    assert observation["confidence"] == 0.635
    assert "visible_rationale" not in observation
    assert compact["current_harness_state"]["candidate_claims"][0]["claim_id"] == "CC-1"
    plan = compact["grounded_reasoning_plan"]
    assert plan["latest_atom_exact_facts"]["evidence_id"] == "EA-GLOBAL-002"
    assert plan["recommended_route_prior_exact_facts"]["evidence_id"] == (
        "EA-UNIFORM-001"
    )
    assert plan["preferred_video_relation"] == "same:a"
    assert plan["valid_primary_gap"][0]["evidence_id"] == "EA-GLOBAL-002"
    assert "global-motion statistics" in plan["required_route_scope"]


def test_vcsl_measurement_retry_preserves_identity_without_viddiff_fields() -> None:
    evidence_id = "EA-VCSL-PAIR-001"
    payload = {
        "task": {"instruction": "Localize the visible pair."},
        "current_harness_state": {
            "candidate_claims": [],
            "evidence_store": {
                evidence_id: {
                    "evidence_id": evidence_id,
                    "active": True,
                    "supports_fields": ["query_id", "reference_id"],
                    "observation": {"query_id": "q1", "reference_id": "r1"},
                }
            },
            "eligible_tools": ["localize_copy_segment_dp"],
        },
        "latest_environment_observation": {
            "created_evidence_ids": [evidence_id]
        },
        "execution_frontier": {
            "eligible_tools": ["localize_copy_segment_dp"],
            "tool_purposes": {
                "localize_copy_segment_dp": "localize copied temporal segments"
            },
            "native_action_templates": [],
        },
    }
    compact = _compact_recommended_measurement_retry_payload(
        payload, "localize_copy_segment_dp"
    )
    observation = compact["current_harness_state"]["evidence_store"][evidence_id][
        "observation"
    ]
    assert observation == {"query_id": "q1", "reference_id": "r1"}
    plan = compact["grounded_reasoning_plan"]
    assert plan["latest_atom_exact_facts"]["observed_fields"] == observation
    assert plan["preferred_video_relation"] == "not_applicable_to_vcsl_localization"
    assert "missing_query_reference_temporal_localization" in str(
        plan["valid_primary_gap"]
    )
    assert "A/B presentation-order consistency" in plan["forbidden_inferences"]


def test_measurement_retry_externalizes_order_disagreement_abstention_cause() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    payload = {
        "task": {"instruction": "Compare the visible pair."},
        "current_harness_state": {
            "candidate_claims": [],
            "evidence_store": {
                evidence_id: {
                    "evidence_id": evidence_id,
                    "active": True,
                    "supports_fields": ["preferred_video"],
                    "observation": {
                        "preferred_video": "c",
                        "measurement_method": "compare_action_refined_symmetric",
                        "confidence": 0.57,
                        "order_consistent": False,
                        "decision_rule": "order_disagreement_abstain",
                    },
                }
            },
            "eligible_tools": ["compare_global_motion_symmetric"],
        },
        "latest_environment_observation": {
            "created_evidence_ids": [evidence_id]
        },
        "execution_frontier": {
            "eligible_tools": ["compare_global_motion_symmetric"],
            "tool_purposes": {
                "compare_global_motion_symmetric": "whole-video motion under both orders"
            },
            "native_action_templates": [],
        },
    }
    compact = _compact_recommended_measurement_retry_payload(
        payload, "compare_global_motion_symmetric"
    )
    plan = compact["grounded_reasoning_plan"]
    assert plan["latest_atom_exact_facts"]["decision_rule"] == (
        "order_disagreement_abstain"
    )
    assert "disagrees across the two A/B presentation orders" in (
        plan["required_abstention_causal_explanation"]
    )


def test_claim_conflict_reasoning_must_name_an_actually_mismatched_field() -> None:
    state = {
        "candidate_claims": [
            {
                "claim_id": "CC-001",
                "fields": {
                    "preferred_video": {"value": "a"},
                    "measurement_method": {
                        "value": "compare_global_motion_symmetric"
                    },
                },
            }
        ],
        "evidence_store": {
            "EA-UNIFORM-002": {
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_uniform_timeline",
                    "order_consistent": None,
                }
            },
            "EA-GLOBAL-002": {
                "observation": {
                    "preferred_video": "a",
                    "measurement_method": "compare_global_motion_symmetric",
                    "order_consistent": True,
                }
            },
        },
    }
    assessment_arguments = {
        "claim_id": "CC-001",
        "evidence_id": "EA-UNIFORM-002",
        "target_fields": ["preferred_video", "measurement_method"],
        "disposition": "reject",
        "defect_codes": ["claim_value_conflict"],
    }
    frontier = _evidence_assessment_reasoning_frontier(
        state,
        [
            {
                "tool_name": "assess_evidence_candidate",
                "arguments": assessment_arguments,
            }
        ],
    )
    assert frontier is not None
    assert frontier["pending_candidate_assessments"][0][
        "validated_claim_conflict_fields"
    ] == ["measurement_method"]
    assert frontier["pending_candidate_assessments"][0][
        "required_rejection_grounding_clauses"
    ] == [
        "Reject EA-UNIFORM-002 because measurement_method=\"compare_uniform_timeline\" "
        "conflicts with the claim value "
        "measurement_method=\"compare_global_motion_symmetric\"."
    ]
    bad_assessment = (
        "`EA-UNIFORM-002` reports preferred_video=a and order_consistent is unobserved. "
        "I reject it because preferred_video=a conflicts with the claim, so the remaining "
        "gap is to record this candidate assessment."
    )
    assessment_problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        assessment_arguments,
        bad_assessment,
        visible_state=state,
    )
    assert (
        "assessment_claim_conflict_field_not_named:EA-UNIFORM-002"
        in assessment_problems
    )
    assert "actually differs" in _generation_retry_feedback(assessment_problems)
    grounded_feedback = _generation_retry_feedback(
        ["assessment_rejection_defect_not_explained:EA-UNIFORM-002"],
        visible_state=state,
        arguments=assessment_arguments,
    )
    assert 'measurement_method="compare_uniform_timeline"' in grounded_feedback
    assert 'measurement_method="compare_global_motion_symmetric"' in grounded_feedback
    assert "Use one complete evidence sentence" in grounded_feedback
    order_feedback = _generation_retry_feedback(
        ["assessment_order_consistency_not_stated:EA-GLOBAL-002"],
        visible_state=state,
        arguments=assessment_arguments,
    )
    assert "order_consistent=true" in order_feedback

    good_assessment = (
        "`EA-UNIFORM-002` reports preferred_video=a and order_consistent is unobserved. "
        "I reject it because its measurement_method is compare_uniform_timeline rather "
        "than the claim's compare_global_motion_symmetric, so that field remains mismatched."
    )
    assert _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        assessment_arguments,
        good_assessment,
        visible_state=state,
    ) == []
    support_status_misframe = (
        "`EA-UNIFORM-002` reports preferred_video=a and order_consistent is unobserved. "
        "I reject it because measurement_method differs from the claim; the visible "
        "mismatch is that the claim values remain unsupported after dropout."
    )
    assessment_problems = _evidence_assessment_reasoning_problems(
        "assess_evidence_candidate",
        assessment_arguments,
        support_status_misframe,
        visible_state=state,
    )
    assert (
        "assessment_claim_conflict_misframed_as_support_status:EA-UNIFORM-002"
        in assessment_problems
    )
    assert "not a claim_value_conflict" in _generation_retry_feedback(
        assessment_problems
    )

    adjudication_arguments = {
        "claim_id": "CC-001",
        "target_fields": ["preferred_video", "measurement_method"],
        "selected_evidence_ids": ["EA-GLOBAL-002"],
        "rejected_evidence": [
            {
                "evidence_id": "EA-UNIFORM-002",
                "defect_codes": ["claim_value_conflict"],
            }
        ],
    }
    bad_adjudication = (
        "I compare and resolve both atoms, selecting `EA-GLOBAL-002` as support because "
        "its visible preferred_video=a fits the claim. I reject `EA-UNIFORM-002` because "
        "preferred_video=a conflicts "
        "with the claim; the remaining gap is to record the evidence partition."
    )
    adjudication_problems = _adjudication_reasoning_problems(
        "adjudicate_evidence",
        adjudication_arguments,
        bad_adjudication,
        visible_state=state,
    )
    assert (
        "adjudication_claim_conflict_field_not_named:EA-UNIFORM-002"
        in adjudication_problems
    )

    good_adjudication = (
        "I compare and resolve both atoms, selecting `EA-GLOBAL-002` as support because "
        "its visible preferred_video=a fits the claim. I reject `EA-UNIFORM-002` because "
        "its measurement_method is "
        "compare_uniform_timeline rather than compare_global_motion_symmetric, a claim-value "
        "mismatch; the remaining gap is to record the evidence partition."
    )
    assert _adjudication_reasoning_problems(
        "adjudicate_evidence",
        adjudication_arguments,
        good_adjudication,
        visible_state=state,
    ) == []


def test_real_surgery26_baseline_state_routes_through_adjudication_before_bind() -> None:
    row = next(
        item
        for item in _first_rows(DEFAULT_SFT_ROOT, "test")
        if item["metadata"]["case_id"] == "viddiff-surgery_26"
        and ":repair:" in item["metadata"]["row_id"]
    )
    environment, registry = _environment(row, "repair", _all_viddiff_items())
    harness = _restore_harness(row)
    harness.environment = environment
    harness.tool_registry = [
        *copy.deepcopy(registry),
        copy.deepcopy(CLAIM_REVISION_TOOL),
        copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL),
    ]
    trace_path = (
        ROOT
        / "outputs/student_external_contract_v1"
        / "semantic_repair_target26_baseline_v10_current"
        / "cases/viddiffbench/viddiff-surgery_26/repair.json"
    )
    turns = json.loads(trace_path.read_text(encoding="utf-8"))["episode"]["turns"]
    for index, turn in enumerate(turns[:4], start=1):
        name = str(turn["generated_tool"])
        arguments = _normalize_harness_params(name, turn["generated_arguments"])
        result = harness.execute_tool(name, arguments, f"replay-{index:03d}")
        assert result["status"] in {"accepted", "executed"}

    state = harness.render_state()
    assert state["eligible_tools"] == ["adjudicate_evidence"]
    action = state["state_action_affordances"]["executable_adjudication_actions"][0]
    assert action["arguments"]["selected_evidence_ids"] == [
        "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    ]
    assert {
        item["evidence_id"]
        for item in action["arguments"]["rejected_evidence"]
    } == {
        "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001",
        "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001",
    }
    harness.execute_tool("adjudicate_evidence", action["arguments"], "replay-adj")
    post = harness.render_state()
    assert post["eligible_tools"] == ["revise_claim_fields"]
    candidates = harness._claim_revision_candidates()["CC-VIDDIFF-SURGERY_26-001"]
    harness.execute_tool(
        "revise_claim_fields",
        {
            "claim_id": "CC-VIDDIFF-SURGERY_26-001",
            "field_updates": [
                {"field": field_name, "value": item["value"]}
                for field_name, item in candidates.items()
            ],
        },
        "replay-revise",
    )
    post = harness.render_state()
    assert "bind_evidence" in post["eligible_tools"]
    bind_action = post["state_action_affordances"]["executable_binding_actions"][0]
    assert {
        support_id
        for item in bind_action["arguments"]["binding_items"]
        for support_id in item["support_ids"]
    } == {"EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"}


def test_viddiff_measurement_routes_have_distinct_semantic_justifications() -> None:
    assert _chosen_measurement_tool_problem(
        "compare_uniform_timeline",
        "Use coarse whole-video chronological sampling for broad initial coverage.",
    ) is None
    assert _chosen_measurement_tool_problem(
        "compare_global_motion_symmetric",
        "Use whole-video motion strips under both A/B orders to test role stability.",
    ) is None
    assert _chosen_measurement_tool_problem(
        "compare_action_refined_symmetric",
        "Use dense action-focused local strips to inspect the brief tube movement.",
    ) is None
    assert _chosen_measurement_tool_problem(
        "compare_action_refined_symmetric",
        "Use the remaining eligible measurement method.",
    ) == "chosen_measurement_tool_not_justified:compare_action_refined_symmetric"


def test_c_abstention_article_is_not_misread_as_preferred_video_a() -> None:
    evidence_id = "EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-001"
    state = {
        "evidence_store": {
            evidence_id: {
                "active": True,
                "supports_fields": list(VIDDIFF_SCHEMA["target_fields"]),
                "observation": {
                    "measurement_method": "compare_action_refined_symmetric",
                    "preferred_video": "c",
                    "confidence": 0.72,
                    "order_consistent": False,
                },
            }
        }
    }
    reasoning = (
        f"The latest observation created `{evidence_id}`, which visibly supports a "
        "`c` abstention with higher confidence and reports order_consistent=false."
    )
    assert _visible_observation_fact_problems(reasoning, state) == []
