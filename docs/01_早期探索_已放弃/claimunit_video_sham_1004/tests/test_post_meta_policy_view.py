import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_student_closed_loop_canary import (
    _compact_progressive_decision_view,
    _compact_viddiff_state_transition_view,
    _post_meta_policy_view,
    _post_meta_task_view,
)


def _atom(method: str, rationale: str, order_consistent: bool | None) -> dict:
    return {
        "evidence_id": f"EA-{method.upper()}-001",
        "kind": "action_comparison_measurement",
        "observation": {
            "preferred_video": "c",
            "video_a_interval_s": [1.0, 2.0],
            "video_b_interval_s": [1.5, 2.5],
            "measurement_method": method,
            "confidence": 0.65,
            "order_consistent": order_consistent,
            "decision_rule": "test",
            "visible_rationale": rationale,
            "routing_state": {"large": "payload"},
        },
        "provenance": {"label_visible_to_tool": False},
        "active": True,
    }


def test_no_meta_atom_leaves_policy_view_unchanged() -> None:
    state = {"evidence_store": {"source": _atom("compare_global_motion_symmetric", "full", False)}}
    history = [{"past_action": {"executed_tool": "compare_global_motion_symmetric", "past_input": {}}}]
    assert _post_meta_policy_view(state, history) == (state, history)


def test_multi_measurement_view_compacts_earlier_rationale_but_keeps_latest_full() -> None:
    state = {
        "evidence_store": {
            "global": _atom(
                "compare_global_motion_symmetric", "global " + "detail " * 80, False
            ),
            "uniform": _atom(
                "compare_uniform_timeline", "latest uniform full rationale", None
            ),
        }
    }
    original = copy.deepcopy(state)
    compact, _ = _post_meta_policy_view(state, [])
    assert state == original
    assert compact["policy_view_compaction"]["kind"] == "multi_measurement_rationale_summary_v1"
    assert compact["policy_view_compaction"]["latest_full_measurement_evidence_id"] == "uniform"
    assert compact["evidence_store"]["uniform"]["observation"]["visible_rationale"] == (
        "latest uniform full rationale"
    )
    earlier = compact["evidence_store"]["global"]["observation"]
    assert earlier["visible_rationale"].startswith("Earlier measurement rationale excerpt:")
    assert earlier["preferred_video"] == "c"
    assert earlier["video_a_interval_s"] == [1.0, 2.0]
    assert earlier["order_consistent"] is False


def test_meta_atom_compacts_only_subsumed_symmetric_sources() -> None:
    state = {
        "evidence_store": {
            "global": _atom("compare_global_motion_symmetric", "global full rationale", False),
            "action": _atom("compare_action_refined_symmetric", "action full rationale", False),
            "uniform": _atom("compare_uniform_timeline", "uniform full rationale", None),
            "meta": _atom("compare_symmetry_residual_abstention", "meta full rationale", True),
        }
    }
    history = [
        {
            "past_action": {
                "executed_tool": "compare_global_motion_symmetric",
                "past_input": {"comparison_statement": "statement", "video_a_ref": "video_a"},
            },
            "environment_outcome": {"status": "executed"},
        },
        {
            "past_action": {
                "executed_tool": "bind_evidence",
                "past_input": {
                    "claim_id": "claim-1",
                    "binding_items": [{"field": "preferred_video", "support_ids": ["meta"]}],
                },
            },
            "environment_outcome": {"status": "accepted"},
        },
    ]
    original = copy.deepcopy(state)

    compact_state, compact_history = _post_meta_policy_view(state, history)

    assert state == original
    assert "routing_state" not in compact_state["evidence_store"]["global"]["observation"]
    assert "routing_state" not in compact_state["evidence_store"]["action"]["observation"]
    assert compact_state["evidence_store"]["uniform"]["observation"]["visible_rationale"] == "uniform full rationale"
    assert compact_state["evidence_store"]["meta"]["observation"]["visible_rationale"] == "meta full rationale"
    for key in ("preferred_video", "video_a_interval_s", "video_b_interval_s", "confidence", "order_consistent"):
        assert compact_state["evidence_store"]["global"]["observation"][key] == original["evidence_store"]["global"]["observation"][key]
    assert compact_state["evidence_store"]["global"]["provenance"] == original["evidence_store"]["global"]["provenance"]
    assert "comparison_statement" not in compact_history[0]["past_action"]["past_input"]
    assert compact_history[0]["environment_outcome"] == {"status": "executed"}
    assert compact_history[1]["past_action"]["past_input"] == {
        "claim_id": "claim-1",
        "bound_fields": ["preferred_video"],
    }


def test_post_meta_task_view_removes_schema_duplication() -> None:
    task = {
        "task_contract": {"contract_id": "task-1"},
        "claim_schema": {
            "schema_id": "schema-1",
            "required_fields": ["preferred_video", "measurement_method"],
            "fields": {"preferred_video": {"enum": ["a", "b", "c"]}},
        },
    }
    state = {"policy_view_compaction": {"kind": "post_meta_source_rationale_summary_v1"}}
    compact = _post_meta_task_view(task, state)
    assert compact == {
        "task_contract_reference": "current_harness_state.task_contract",
        "claim_schema_id": "schema-1",
        "required_fields": ["preferred_video", "measurement_method"],
        "claim_field_schemas_reference": "currently eligible native tool schema",
    }
    assert _post_meta_task_view(task, {}) is task


def test_progressive_decision_view_removes_only_duplicate_policy_facts() -> None:
    state = {
        "eligible_tools": ["adjudicate_evidence"],
        "candidate_claims": [
            {
                "claim_id": "claim-1",
                "schema_id": "schema-1",
                "fields": {
                    "preferred_video": {
                        "value": "a",
                        "evidence_status": "unsupported",
                        "decision_status": "open",
                        "support_ids": [],
                        "attack_ids": [],
                    }
                },
            }
        ],
        "evidence_store": {"EA-1": _atom("compare_uniform_timeline", "cue", None)},
        "evidence_adjudication": {
            "required": True,
            "requirement": {
                "claim_id": "claim-1",
                "target_fields": ["preferred_video"],
                "competing_evidence_ids": ["EA-1"],
                "compatible_evidence_ids_by_field": {"preferred_video": ["EA-1"]},
            },
            "rule": "duplicated by the frontier",
        },
        "evidence_assessment": {"required": False, "pending_evidence_ids": []},
        "evidence_assessments": {"claim-1": {"EA-1": {"disposition": "retain"}}},
        "case_goals": [{"goal_id": "G-REPORT", "status": "open"}],
        "task_contract": {"contract_id": "TASK-1"},
        "available_evidence_ids": ["EA-1"],
        "field_bindings_summary": {},
        "action_constraints": {"must_reopen_before_rebinding": []},
        "open_high_priority_obligations": ["G-REPORT"],
        "state_action_affordances": {"current_native_actions_reference": "schema"},
    }
    original = copy.deepcopy(state)
    compact, history = _compact_progressive_decision_view(
        state,
        [
            {
                "past_action": {"executed_tool": "assess_evidence_candidate"},
                "environment_outcome": {
                    "status": "accepted",
                    "assessment_evidence_id": "EA-ASSESS-1",
                    "verbose_duplicate": "remove",
                },
            }
        ],
    )
    assert state == original
    assert compact["evidence_store"] == state["evidence_store"]
    assert compact["evidence_assessments"] == state["evidence_assessments"]
    assert compact["candidate_claims"][0]["fields"]["preferred_video"] == {
        "value": "a",
        "evidence_status": "unsupported",
        "decision_status": "open",
    }
    assert compact["evidence_adjudication"]["requirement"] == {
        "claim_id": "claim-1",
        "target_fields": ["preferred_video"],
        "competing_evidence_ids": ["EA-1"],
    }
    assert "case_goals" not in compact
    assert "task_contract" not in compact
    assert "evidence_assessment" not in compact
    assert history[0]["environment_outcome"] == {
        "status": "accepted",
        "assessment_evidence_id": "EA-ASSESS-1",
    }


def test_viddiff_bind_view_keeps_decision_facts_without_duplicate_rationales() -> None:
    state = {
        "task_contract": {
            "claim_schema_id": "viddiff_comparative_measurement_claim_v1"
        },
        "eligible_tools": ["bind_evidence"],
        "candidate_claims": [
            {
                "claim_id": "claim-1",
                "fields": {
                    "preferred_video": {
                        "value": "a",
                        "evidence_status": "bindable",
                        "decision_status": "open",
                        "support_ids": [],
                        "attack_ids": ["noise"],
                    }
                },
            }
        ],
        "evidence_store": {
            "global": _atom(
                "compare_global_motion_symmetric", "global rationale " * 50, True
            ),
            "uniform": _atom(
                "compare_uniform_timeline", "uniform rationale " * 50, None
            ),
        },
        "dropped_evidence_ids": ["EA-COMPARE_GLOBAL_MOTION_SYMMETRIC-000"],
        "open_high_priority_obligations": ["G-REPORT"],
        "case_goals": [{"goal_id": "G-REPORT", "status": "open"}],
        "available_evidence_ids": ["global", "uniform"],
        "evidence_assessment": {"large": "duplicate"},
        "evidence_adjudication": {"large": "duplicate"},
        "state_action_affordances": {
            "derivation": "visible",
            "binding_candidates": {"duplicate": True},
            "executable_binding_actions": [
                {
                    "tool_name": "bind_evidence",
                    "arguments": {
                        "claim_id": "claim-1",
                        "binding_items": [
                            {"field": "preferred_video", "support_ids": ["global"]}
                        ],
                    },
                }
            ],
        },
    }
    original = copy.deepcopy(state)
    compact, history = _compact_viddiff_state_transition_view(state, [])
    assert state == original
    assert history == []
    assert compact["policy_view_compaction"]["kind"] == "viddiff_state_transition_v1"
    assert "task_contract" not in compact
    assert "case_goals" not in compact
    assert "available_evidence_ids" not in compact
    assert "evidence_assessment" not in compact
    assert "visible_rationale" not in compact["evidence_store"]["global"]["observation"]
    assert "routing_state" not in compact["evidence_store"]["global"]["observation"]
    assert compact["evidence_store"]["global"]["observation"]["preferred_video"] == "c"
    assert compact["candidate_claims"][0]["fields"]["preferred_video"] == {
        "value": "a",
        "evidence_status": "bindable",
        "decision_status": "open",
        "support_ids": [],
    }
    assert compact["state_action_affordances"]["executable_binding_actions"][0][
        "tool_name"
    ] == "bind_evidence"
    assert compact["dropped_evidence_ids"] == [
        "EA-COMPARE_GLOBAL_MOTION_SYMMETRIC-000"
    ]
    task = {
        "claim_schema": {
            "schema_id": "viddiff_comparative_measurement_claim_v1",
            "required_fields": ["preferred_video"],
        }
    }
    assert "task_contract_reference" not in _post_meta_task_view(task, compact)


def test_post_claim_view_keeps_selected_meta_and_summarizes_nonselected_atoms() -> None:
    state = {
        "candidate_claims": [
            {
                "claim_id": "claim-1",
                "fields": {
                    "measurement_method": {
                        "value": "compare_symmetry_residual_abstention"
                    }
                },
            }
        ],
        "evidence_store": {
            "uniform": _atom("compare_uniform_timeline", "uniform rationale", None),
            "meta": _atom("compare_symmetry_residual_abstention", "meta rationale", True),
        },
        "field_bindings": {
            "claim-1": {
                "preferred_video": {
                    "support_ids": ["meta"],
                    "validation_status": "accepted",
                    "proposed_by": "policy",
                }
            }
        },
    }
    compact, _ = _post_meta_policy_view(state, [])
    uniform = compact["evidence_store"]["uniform"]["observation"]
    assert set(uniform) == {
        "preferred_video",
        "measurement_method",
        "confidence",
        "order_consistent",
        "decision_rule",
        "summary",
    }
    assert compact["evidence_store"]["meta"]["observation"]["visible_rationale"] == "meta rationale"
    assert compact["field_bindings"] == {}
    assert compact["field_bindings_summary"]["claim-1"]["preferred_video"] == {
        "support_ids": ["meta"],
        "validation_status": "accepted",
    }
