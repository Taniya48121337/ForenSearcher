from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_choice_search import (  # noqa: E402
    EvidenceChoiceSearchTeacher,
    EvidenceChoiceProposalRejected,
    commit_selected_branch,
    label_blind_branch_score,
    select_evidence_branch,
)
from llm_teacher_rollout import TeacherAction  # noqa: E402


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, name, _arguments):
        self.calls.append(name)
        positive = name != "measure_a"
        return {
            "status": "executed",
            "tool_name": name,
            "result": {
                "tested_modality": name,
                "material_difference_detected": positive,
                "measurement": 0.8 if positive else 0.1,
            },
            "evidence_atoms": [
                {
                    "evidence_id": f"EA-{name}",
                    "kind": "measurement" if positive else "negative_measurement",
                    "supports_fields": ["answer"] if positive else [],
                    "observation": {
                        "material_difference_detected": positive,
                        "measurement": 0.8 if positive else 0.1,
                    },
                    "provenance": {"method": name, "generated_by": f"EXEC-{name}"},
                }
            ],
        }


class FakeHarness:
    def __init__(self, environment=None) -> None:
        self.environment = environment or FakeEnvironment()
        self.target_fields = ["answer"]
        self.policy_calls = []

    def render_state(self):
        return {"open_high_priority_obligations": ["G-DIFFERENCE"]}

    def clone_state(self, *, environment=None):
        return FakeHarness(environment=environment)

    def record_policy_call(self, *args):
        self.policy_calls.append(args)

    def execute_tool(self, name, arguments, _call_id):
        result = self.environment.execute(name, arguments)
        result["harness_state"] = {
            "open_high_priority_obligations": []
            if name != "measure_a"
            else ["G-DIFFERENCE"]
        }
        return result


class CyclingDelegate:
    model = "local-test"
    source = "test-double"

    def __init__(self) -> None:
        self.index = 0
        self.audit_log = []

    def next_action(self, _input, _tools):
        name = ["measure_a", "measure_b"][self.index % 2]
        self.index += 1
        action = _action(name, str(self.index))
        action.usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        self.audit_log.append(
            {
                "control_plane_exposure": {
                    "semantic_measurement_tool_names": ["measure_a", "measure_b"],
                    "semantic_measurement_choice_count": 2,
                },
                "selected_score": 0.9 if name == "measure_b" else 0.8,
                "selected_method_fit_score": 1.0,
                "selected_method_fit_ranking_score": 0.95 if name == "measure_b" else 0.9,
                "selected_method_fit": {
                    "counterfactual_tool_swap_passed": True,
                    "directional_contrast": {"alternative_tool": "other"},
                },
                "selected_action": {"tool_name": name},
            }
        )
        return action


class ScriptedProposalDelegate:
    model = "local-scripted"
    source = "test-double"

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self.audit_log = []
        self.seen_inputs = []
        self.index = 0

    def next_action(self, input_items, _tools):
        self.seen_inputs.append(copy.deepcopy(input_items))
        name = self.names[self.index]
        self.index += 1
        action = _action(name, f"SCRIPT-{self.index}")
        action.usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        rank = 0.95 if name == "measure_b" else 0.9
        self.audit_log.append(
            {
                "selected_score": 0.9 if name == "measure_b" else 0.8,
                "selected_method_fit_score": 1.0,
                "selected_method_fit_ranking_score": rank,
                "selected_method_fit": {
                    "counterfactual_tool_swap_passed": True,
                    "directional_contrast": {"alternative_tool": "other"},
                },
                "selected_action": {"tool_name": name},
                "control_plane_exposure": {
                    "semantic_measurement_tool_names": ["measure_a", "measure_b"],
                    "semantic_measurement_choice_count": 2,
                },
            }
        )
        return action


class ScriptedStateDelegate:
    model = "local-state"
    source = "test-double"

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.audit_log = []
        self.seen_inputs = []
        self.index = 0

    def next_action(self, input_items, _tools):
        self.seen_inputs.append(copy.deepcopy(input_items))
        self.index += 1
        score = self.scores[self.index - 1]
        action = _action("stop", f"STATE-{self.index}")
        action.usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        self.audit_log.append(
            {
                "selected_score": score,
                "selected_failures": (
                    [] if score >= 0.875 else ["state_consequence_missing"]
                ),
                "selected_action": {"tool_name": "stop"},
                "control_plane_exposure": {
                    "compiled_semantic_actions_visible": False,
                    "repair_recipe_visible": False,
                    "full_target_call_materializable": False,
                    "categorical_router_answer_visible": False,
                    "semantic_measurement_tool_names": [],
                    "semantic_measurement_choice_count": 0,
                },
            }
        )
        return action


def _action(name: str, suffix: str) -> TeacherAction:
    return TeacherAction(
        reasoning=f"The unresolved modality question makes {name} discriminating.",
        tool_name=name,
        params={"media_ref": "current"},
        raw_arguments={"reasoning": f"Choose {name}.", "media_ref": "current"},
        call_id=f"CALL-{suffix}",
        response_id=f"RESP-{suffix}",
        replay_items=[],
        usage={},
    )


def test_isolated_branching_does_not_mutate_parent_and_commits_one_branch() -> None:
    harness = FakeHarness()
    selection = select_evidence_branch(
        harness=harness,
        actions=[_action("measure_a", "A"), _action("measure_b", "B")],
        visible_semantic_tools=["measure_a", "measure_b"],
        proposal_method_fit_by_call_id={"CALL-A": 1.0, "CALL-B": 1.0},
        proposal_method_fit_ranking_by_call_id={"CALL-A": 0.95, "CALL-B": 0.9},
    )
    assert harness.environment.calls == []
    assert selection["audit"]["proposal_coverage_passed"] is True
    assert selection["audit"]["preview_observations_student_visible"] is False
    assert selection["selected"].action.tool_name == "measure_a"
    assert selection["audit"]["selection_frozen_before_branch_execution"] is True
    assert selection["audit"]["hindsight_outcome_used_for_selection"] is False
    assert selection["audit"]["higher_utility_branch_was_not_allowed_to_reroute_selection"] is True

    committed = commit_selected_branch(harness, selection["selected"])
    assert harness.environment.calls == ["measure_a"]
    assert committed["deterministic_replay_match"] is True


def test_duplicate_proposals_do_not_fake_tool_choice_coverage() -> None:
    selection = select_evidence_branch(
        harness=FakeHarness(),
        actions=[_action("measure_a", "A1"), _action("measure_a", "A2")],
        visible_semantic_tools=["measure_a", "measure_b"],
    )
    assert selection["audit"]["proposed_choice_count"] == 1
    assert selection["audit"]["proposal_coverage_passed"] is False


def test_negative_measurement_gets_information_credit_without_target_credit() -> None:
    harness = FakeHarness()
    branch = harness.clone_state(environment=copy.deepcopy(harness.environment))
    result = branch.execute_tool("measure_a", {}, "CALL-A")
    utility = label_blind_branch_score(
        before_state=harness.render_state(),
        result=result,
        target_fields=harness.target_fields,
    )
    assert utility["negative_measurement_is_information"] is True
    assert utility["components"]["target_field_coverage"] == 0.0
    assert utility["score"] > 0.5
    assert utility["oracle_or_verifier_used"] is False


def test_search_teacher_samples_full_surface_and_records_committed_replay() -> None:
    harness = FakeHarness()
    teacher = EvidenceChoiceSearchTeacher(
        CyclingDelegate(), branch_proposals_per_turn=4
    )
    teacher.bind_harness(harness)
    tools = [
        {"name": "measure_a", "parameters": {"type": "object"}},
        {"name": "measure_b", "parameters": {"type": "object"}},
    ]
    action = teacher.next_action([{"role": "user", "content": "{}"}], tools)
    assert harness.environment.calls == []
    assert action.tool_name == "measure_b"
    assert action.usage["total_tokens"] == 48
    search = teacher.audit_log[-1]["evidence_choice_search"]
    assert search["proposal_coverage_passed"] is True
    assert search["preview_observations_student_visible"] is False
    assert search["selection_frozen_before_branch_execution"] is True
    assert search["selected_branch_execution_admissible"] is True

    result = harness.execute_tool(action.tool_name, action.params, action.call_id)
    teacher.record_committed_result(action, result)
    search = teacher.audit_log[-1]["evidence_choice_search"]
    assert search["committed_replay_checked"] is True
    assert search["deterministic_replay_match"] is True


def test_equal_outcomes_require_preaction_reasoning_to_break_tie() -> None:
    actions = [_action("measure_b", "B"), _action("measure_c", "C")]
    ambiguous = select_evidence_branch(
        harness=FakeHarness(),
        actions=actions,
        visible_semantic_tools=["measure_b", "measure_c"],
    )
    assert ambiguous["audit"]["selection_ambiguous"] is True

    resolved = select_evidence_branch(
        harness=FakeHarness(),
        actions=actions,
        visible_semantic_tools=["measure_b", "measure_c"],
        proposal_scores_by_call_id={"CALL-B": 0.8, "CALL-C": 0.9},
    )
    assert resolved["audit"]["selection_ambiguous"] is False
    assert resolved["selected"].action.tool_name == "measure_c"


def test_method_fit_precedes_generic_reasoning_when_execution_utility_ties() -> None:
    actions = [_action("measure_b", "B"), _action("measure_c", "C")]
    resolved = select_evidence_branch(
        harness=FakeHarness(),
        actions=actions,
        visible_semantic_tools=["measure_b", "measure_c"],
        proposal_method_fit_by_call_id={"CALL-B": 1.0, "CALL-C": 0.4},
        proposal_scores_by_call_id={"CALL-B": 0.7, "CALL-C": 1.0},
    )
    assert resolved["selected"].action.tool_name == "measure_b"
    assert resolved["audit"]["selected_preaction_method_fit_score"] == 1.0
    assert resolved["audit"]["selection_ambiguous"] is False


def test_checkpoint_resampling_recovers_from_preaction_collapse_without_outcome_feedback() -> None:
    delegate = ScriptedProposalDelegate(
        ["measure_a", "measure_a", "measure_a", "measure_b"]
    )
    teacher = EvidenceChoiceSearchTeacher(
        delegate,
        branch_proposals_per_turn=2,
        proposal_rounds_per_turn=2,
    )
    harness = FakeHarness()
    teacher.bind_harness(harness)
    action = teacher.next_action(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "{}"}],
        [
            {"name": "measure_a", "parameters": {"type": "object"}},
            {"name": "measure_b", "parameters": {"type": "object"}},
        ],
    )
    assert action.tool_name == "measure_b"
    assert action.usage["total_tokens"] == 48
    search = teacher.audit_log[-1]["evidence_choice_search"]
    assert search["proposal_rounds_attempted"] == 2
    assert search["prior_rejected_proposal_rounds"][0][
        "preaction_rejection_reasons"
    ] == ["proposal_coverage_failed"]
    assert "no prior preview outcome is available" in delegate.seen_inputs[2][0]["content"]
    assert harness.environment.calls == []


def test_checkpoint_resampling_exhaustion_is_structured_and_never_committed() -> None:
    teacher = EvidenceChoiceSearchTeacher(
        ScriptedProposalDelegate(["measure_a"] * 4),
        branch_proposals_per_turn=2,
        proposal_rounds_per_turn=2,
    )
    harness = FakeHarness()
    teacher.bind_harness(harness)
    with pytest.raises(EvidenceChoiceProposalRejected) as raised:
        teacher.next_action(
            [{"role": "system", "content": "base"}, {"role": "user", "content": "{}"}],
            [
                {"name": "measure_a", "parameters": {"type": "object"}},
                {"name": "measure_b", "parameters": {"type": "object"}},
            ],
        )
    artifact = raised.value.failure_artifact
    assert artifact["reason_counts"] == {"proposal_coverage_failed": 2}
    assert artifact["preview_outcome_feedback_sent_to_teacher"] is False
    assert harness.environment.calls == []


def test_state_action_reasoning_gets_bounded_same_checkpoint_recovery() -> None:
    delegate = ScriptedStateDelegate([0.5, 0.6, 0.95, 0.9])
    teacher = EvidenceChoiceSearchTeacher(
        delegate,
        state_proposals_per_turn=2,
        proposal_rounds_per_turn=2,
    )
    teacher.bind_harness(FakeHarness())
    action = teacher.next_action(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "{}"}],
        [{"name": "stop", "parameters": {"type": "object"}}],
    )
    assert action.tool_name == "stop"
    assert action.usage["total_tokens"] == 48
    search = teacher.audit_log[-1]["evidence_choice_search"]
    assert search["state_reasoning_contract_passed"] is True
    assert search["proposal_rounds_attempted"] == 2
    assert search["prior_rejected_proposal_rounds"][0]["rejection_reasons"] == [
        "state_consequence_missing",
        "state_reasoning_score_below_minimum",
    ]
    assert "no tool outcome was executed or revealed" in delegate.seen_inputs[2][0][
        "content"
    ]
