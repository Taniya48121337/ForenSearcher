"""Local fault-injection audit for bounded checkpoint proposal recovery."""

from __future__ import annotations

import copy
import json
from typing import Any

from evidence_choice_search import (
    EvidenceChoiceProposalRejected,
    EvidenceChoiceSearchTeacher,
)
from generation_admission import generation_failure_metadata
from llm_teacher_rollout import TeacherAction
from pipeline import ROOT, write_json


class AuditEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        return {
            "status": "executed",
            "result": {"tested_method": name, "material_difference_detected": True},
            "evidence_atoms": [
                {
                    "evidence_id": f"EA-{name}",
                    "kind": "measurement",
                    "supports_fields": ["answer"],
                    "observation": {"tested_method": name},
                    "provenance": {"method": name},
                }
            ],
        }


class AuditHarness:
    def __init__(self, environment: AuditEnvironment | None = None) -> None:
        self.environment = environment or AuditEnvironment()
        self.target_fields = ["answer"]

    def render_state(self) -> dict[str, Any]:
        return {"open_high_priority_obligations": ["G-DIFFERENCE"]}

    def clone_state(self, *, environment=None):
        return AuditHarness(environment=environment)

    def record_policy_call(self, *_args) -> None:
        return None

    def execute_tool(self, name, arguments, _call_id):
        result = self.environment.execute(name, arguments)
        result["harness_state"] = {"open_high_priority_obligations": []}
        return result


def _action(name: str, index: int) -> TeacherAction:
    return TeacherAction(
        reasoning=f"Visible diagnostic structure makes {name} a prospective discriminating measurement.",
        tool_name=name,
        params={"media_ref": "current"},
        raw_arguments={},
        call_id=f"CALL-{index}",
        response_id=f"RESP-{index}",
        replay_items=[],
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


class AuditDelegate:
    model = "local-audit"
    source = "deterministic-fault-injection"

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.index = 0
        self.audit_log: list[dict[str, Any]] = []
        self.inputs: list[list[dict[str, Any]]] = []

    def next_action(self, input_items, _tools):
        self.inputs.append(copy.deepcopy(input_items))
        name = self.names[self.index]
        self.index += 1
        rank = 0.95 if name == "measure_b" else 0.9
        action = _action(name, self.index)
        self.audit_log.append(
            {
                "selected_score": 0.9 if name == "measure_b" else 0.8,
                "selected_method_fit_score": 1.0,
                "selected_method_fit_ranking_score": rank,
                "selected_method_fit": {
                    "counterfactual_tool_swap_passed": True,
                    "directional_contrast": {"alternative_tool": "other"},
                },
                "selected_failures": [],
                "selected_action": {"tool_name": name},
                "control_plane_exposure": {
                    "semantic_measurement_tool_names": ["measure_a", "measure_b"],
                    "semantic_measurement_choice_count": 2,
                },
            }
        )
        return action


TOOLS = [
    {"name": "measure_a", "parameters": {"type": "object"}},
    {"name": "measure_b", "parameters": {"type": "object"}},
]
INPUT = [{"role": "system", "content": "base"}, {"role": "user", "content": "{}"}]


def audit() -> dict[str, Any]:
    recovered_delegate = AuditDelegate(
        ["measure_a", "measure_a", "measure_a", "measure_b"]
    )
    recovered_harness = AuditHarness()
    recovered_teacher = EvidenceChoiceSearchTeacher(
        recovered_delegate,
        branch_proposals_per_turn=2,
        proposal_rounds_per_turn=2,
    )
    recovered_teacher.bind_harness(recovered_harness)
    selected = recovered_teacher.next_action(copy.deepcopy(INPUT), copy.deepcopy(TOOLS))
    recovered_search = recovered_teacher.audit_log[-1]["evidence_choice_search"]

    exhausted_harness = AuditHarness()
    exhausted_teacher = EvidenceChoiceSearchTeacher(
        AuditDelegate(["measure_a"] * 4),
        branch_proposals_per_turn=2,
        proposal_rounds_per_turn=2,
    )
    exhausted_teacher.bind_harness(exhausted_harness)
    exhausted: dict[str, Any]
    try:
        exhausted_teacher.next_action(copy.deepcopy(INPUT), copy.deepcopy(TOOLS))
        exhausted = {"raised": False}
    except EvidenceChoiceProposalRejected as error:
        exhausted = {
            "raised": True,
            **generation_failure_metadata(error),
            "preview_outcome_feedback_sent_to_teacher": error.failure_artifact[
                "preview_outcome_feedback_sent_to_teacher"
            ],
        }

    report = {
        "artifact": "bounded_checkpoint_recovery_contract_v1",
        "scope": "deterministic_local_fault_injection_no_teacher_service_no_oracle",
        "recovered": {
            "selected_tool": selected.tool_name,
            "proposal_rounds_attempted": recovered_search[
                "proposal_rounds_attempted"
            ],
            "first_round_reasons": recovered_search[
                "prior_rejected_proposal_rounds"
            ][0]["preaction_rejection_reasons"],
            "proposal_retry_outcome_feedback_used": recovered_search[
                "proposal_retry_outcome_feedback_used"
            ],
            "parent_environment_calls_before_commit": recovered_harness.environment.calls,
        },
        "exhausted": exhausted,
    }
    report["all_passed"] = bool(
        selected.tool_name == "measure_b"
        and report["recovered"]["proposal_rounds_attempted"] == 2
        and report["recovered"]["first_round_reasons"]
        == ["proposal_coverage_failed"]
        and report["recovered"]["proposal_retry_outcome_feedback_used"] is False
        and not report["recovered"]["parent_environment_calls_before_commit"]
        and exhausted.get("failure_category")
        == "preaction_proposal_quality_exhausted"
        and exhausted.get("failure_reason_counts")
        == {"proposal_coverage_failed": 2}
        and exhausted.get("preview_outcome_feedback_sent_to_teacher") is False
    )
    return report


def main() -> None:
    report = audit()
    output = (
        ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1"
        / "generation_recovery_contract.json"
    )
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
