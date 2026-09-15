"""Compile stateful candidate-assessment -> adjudication warm-up trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from interactive_harness import EVIDENCE_ADJUDICATION_TOOL, EVIDENCE_ASSESSMENT_TOOL
from pipeline import write_json, write_jsonl
from prepare_evidence_adjudication_sft import (
    DEFECT_TEXT,
    _emit_row,
    _student_visible_privacy_problems,
)
from run_student_closed_loop_batch import (
    DEFAULT_SFT_ROOT,
    _all_viddiff_items,
    _case_id,
    _environment,
    _first_rows,
    _mode,
)
from run_student_closed_loop_canary import (
    PROGRESSIVE_COMPRESSION_CONTRACT,
    PROGRESSIVE_EVIDENCE_SYSTEM,
    _compile_visible_decision_actions,
    _evidence_assessment_reasoning_frontier,
    _execution_frontier,
    _payload,
    _post_meta_policy_view,
    _post_meta_task_view,
    _preexecution_reasoning_problems,
    _result_summary,
)
from shadow_execute_generated_actions import _restore_harness
from student_tool_contract import build_student_tools


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "outputs/student_external_contract_v1/progressive_evidence_assessment_warmup_v1"
)

def _parallel_conflict_context(
    seed_row: dict[str, Any],
    viddiff_items: dict[str, dict[str, Any]],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]] | tuple[None, dict[str, Any]]:
    case_id = _case_id(seed_row)
    sample_key = case_id.removeprefix("viddiff-")
    item = viddiff_items[sample_key]
    environment, registry = _environment(seed_row, "clean", viddiff_items)
    harness = _restore_harness(seed_row)
    harness.environment = environment
    harness.tool_registry = [
        *copy.deepcopy(registry),
        copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL),
        copy.deepcopy(EVIDENCE_ASSESSMENT_TOOL),
    ]
    claims = list(harness.claims.values())
    if len(claims) != 1:
        return None, {"case_id": case_id, "status": "claim_count_not_one"}
    claim = claims[0]
    selected_method = str(claim.fields["measurement_method"].value)
    candidate_names = {str(entry["name"]) for entry in item["candidate_tools"]}
    if selected_method not in candidate_names:
        return None, {
            "case_id": case_id,
            "status": "selected_method_not_in_parallel_candidate_set",
            "selected_method": selected_method,
        }

    environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
    measurement_atoms = []
    history: list[dict[str, Any]] = []
    for step, candidate in enumerate(item["candidate_tools"]):
        name = str(candidate["name"])
        result = environment.execute(name, copy.deepcopy(candidate["arguments"]))
        atoms = list(result.get("evidence_atoms") or [])
        measurement_atoms.extend(atoms)
        history.append(
            {
                "step_index": step,
                "parallel_group_id": f"PARALLEL-VIDDIFF-{sample_key}",
                "past_action": {
                    "executed_tool": name,
                    "past_input": copy.deepcopy(candidate["arguments"]),
                },
                "environment_outcome": {
                    "status": "executed",
                    "authority": "external_tool_environment",
                    "created_evidence_ids": [atom.evidence_id for atom in atoms],
                },
            }
        )

    identity_atoms = {
        evidence_id: copy.deepcopy(atom)
        for evidence_id, atom in harness.evidence_store.items()
        if atom.active and atom.kind == "comparison_task_identity"
    }
    harness.evidence_store = {
        **identity_atoms,
        **{atom.evidence_id: copy.deepcopy(atom) for atom in measurement_atoms},
    }
    harness.evidence_assessments.clear()
    harness.evidence_adjudications.clear()
    harness.dropped_evidence_ids = []
    harness.terminal = False
    harness.stop_decision = None
    claim_id = claim.claim_id
    for field_name in harness.target_fields:
        field_state = claim.fields[field_name]
        field_state.evidence_status = "unsupported"
        field_state.decision_status = "open"
        field_state.support_ids = []
        field_state.attack_ids = []
        harness.bindings.setdefault(claim_id, {}).pop(field_name, None)
    harness.goals["G-CONTEXT"].status = "satisfied"
    harness.goals["G-DIFFERENCE"].status = "satisfied"
    harness.goals["G-DIFFERENCE"].evidence_ids = [
        atom.evidence_id for atom in measurement_atoms
    ]
    harness.goals["G-REPORT"].status = "open"
    harness.goals["G-REPORT"].evidence_ids = []
    state = harness.render_state()
    if state.get("eligible_tools") != ["assess_evidence_candidate"]:
        return None, {
            "case_id": case_id,
            "status": "parallel_state_not_progressively_assessable",
            "eligible_tools": state.get("eligible_tools"),
        }

    created_ids = [atom.evidence_id for atom in measurement_atoms]
    latest = {
        "status": "executed",
        "authority": "parallel_external_tool_environment",
        "parallel_group_id": f"PARALLEL-VIDDIFF-{sample_key}",
        "created_evidence_ids": created_ids,
    }
    task = copy.deepcopy(_payload(seed_row)["task"])
    task["task"] = (
        "Assess every candidate in a completed label-blind parallel measurement batch, "
        "adjudicate the visible ledger, and preserve the existing claim values."
    )
    task["mutation"] = {
        "operator": "parallel_evidence_return_with_progressive_assessment",
        "added_evidence_ids": created_ids,
        "media_changed": False,
        "claim_values_changed": False,
    }
    return harness, state, history, latest, task


def _assessment_reasoning(
    state: dict[str, Any], arguments: dict[str, Any]
) -> str:
    evidence_id = str(arguments["evidence_id"])
    observation = (state["evidence_store"][evidence_id].get("observation") or {})
    preferred = observation.get("preferred_video")
    method = str(observation.get("measurement_method") or "")
    claim_fields = (state.get("candidate_claims") or [{}])[0].get("fields") or {}
    statement = (claim_fields.get("comparison_statement") or {}).get("value")
    sample_key = (claim_fields.get("sample_key") or {}).get("value")
    order = observation.get("order_consistent")
    order_text = (
        "`order_consistent` is unobserved"
        if order is None
        else f"`order_consistent={str(order).lower()}`"
    )
    disposition = str(arguments["disposition"])
    if disposition == "retain":
        justification = (
            "retain it because its visible direction and intervals match the claim while "
            "role-swap stability remains explicitly uncertain"
        )
    else:
        defect = DEFECT_TEXT[str(arguments["defect_codes"][0])]
        if (
            preferred == "c"
            and order is False
            and observation.get("decision_rule")
            in {"order_disagreement_abstain", "targeted_disagreement_abstain"}
        ):
            justification = (
                "reject it: it abstains because the directions disagree across A/B "
                f"presentation orders, and {defect}"
            )
        else:
            justification = f"reject it because {defect}"
    return (
        f"I select `assess_evidence_candidate` because the remaining gap is the pending ledger "
        f"entry for the unique method `{method}` on sample `{sample_key}` and "
        f"{json.dumps(statement, ensure_ascii=False)}. "
        f"It reports `preferred_video={preferred}`, and {order_text}; I "
        f"{justification}. This records one candidate disposition without creating visual "
        "evidence or changing claim values."
    )


def _emit_assessment_row(
    *,
    seed_row: dict[str, Any],
    harness: Any,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    latest: dict[str, Any],
    task: dict[str, Any],
    action: dict[str, Any],
    assessment_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = list(state.get("eligible_tools") or [])
    if eligible != ["assess_evidence_candidate"]:
        raise ValueError(f"expected sole assessment tool, got {eligible}")
    _compile_visible_decision_actions(state)
    arguments = copy.deepcopy(action["arguments"])
    policy_state, policy_history = _post_meta_policy_view(state, history)
    policy_task = _post_meta_task_view(copy.deepcopy(task), policy_state)
    actions = list(
        (state.get("state_action_affordances") or {}).get(
            "executable_evidence_assessment_actions"
        )
        or []
    )
    tools = build_student_tools(
        harness.claim_schema,
        harness.tool_registry,
        eligible,
        executable_actions=actions,
    )
    payload = {
        "task": policy_task,
        "current_harness_state": policy_state,
        "past_execution_history": policy_history,
        "latest_environment_observation": latest,
        "compression_contract": PROGRESSIVE_COMPRESSION_CONTRACT,
        "execution_frontier": _execution_frontier(eligible, tools, actions),
        "evidence_assessment_frontier": _evidence_assessment_reasoning_frontier(
            state, actions
        ),
    }
    privacy_problems = _student_visible_privacy_problems(payload)
    if privacy_problems:
        raise ValueError(f"student-visible privacy failure: {privacy_problems}")
    reasoning = _assessment_reasoning(state, arguments)
    problems = _preexecution_reasoning_problems(
        reasoning,
        "assess_evidence_candidate",
        arguments,
        meta_visible=False,
        visible_state={**state, "latest_environment_observation": latest},
    )
    if problems:
        raise ValueError(f"compiled assessment reasoning failed: {problems}")
    result = harness.execute_tool(
        "assess_evidence_candidate",
        arguments,
        f"compiled-progressive-assessment-{assessment_index:03d}",
    )
    if result.get("status") != "accepted":
        raise ValueError(f"compiled assessment was not accepted: {result}")
    post_state = harness.render_state()
    pending_before = len(
        (state.get("evidence_assessment") or {}).get("pending_evidence_ids") or []
    )
    pending_after = len(
        (post_state.get("evidence_assessment") or {}).get("pending_evidence_ids") or []
    )
    if pending_after != pending_before - 1:
        raise ValueError("assessment did not reduce the pending ledger by exactly one")

    system = {"role": "system", "content": PROGRESSIVE_EVIDENCE_SYSTEM}
    case_id = _case_id(seed_row)
    evidence_id = str(arguments["evidence_id"])
    row = {
        "messages": [
            system,
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": reasoning,
                "tool_calls": [
                    {
                        "id": f"progressive-{case_id}-assess-{assessment_index:03d}",
                        "type": "function",
                        "function": {
                            "name": "assess_evidence_candidate",
                            "arguments": arguments,
                        },
                    }
                ],
            },
        ],
        "tools": tools,
        "metadata": {
            "row_id": (
                f"VidDiffBench:{case_id}:progressive_assessment_{assessment_index:03d}:repair:"
                f"{len(policy_history) + 1}"
            ),
            "dataset": "VidDiffBench",
            "case_id": case_id,
            "split": "train",
            "target_tool": "assess_evidence_candidate",
            "trajectory_mode": "repair",
            "turn_index": len(policy_history) + 1,
            "target_execution_status": "accepted",
            "target_includes_observation": False,
            "loss_roles": ["assistant"],
            "loss_scope": "last_assistant_action_only",
            "oracle_or_verifier_used": False,
            "teacher_action_used": False,
            "strict_closed_loop_accepted": True,
            "generation_source": "student_visible_progressive_assessment_compiler",
        },
    }
    audit = {
        "case_id": case_id,
        "assessment_index": assessment_index,
        "evidence_id": evidence_id,
        "disposition": arguments["disposition"],
        "defect_codes": arguments["defect_codes"],
        "reasoning_contract_problems": problems,
        "student_visible_privacy_problems": privacy_problems,
        "execution_status": result["status"],
        "pending_candidates_before": pending_before,
        "pending_candidates_after": pending_after,
    }
    return row, audit


def _compile_case(
    seed_row: dict[str, Any], viddiff_items: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = _parallel_conflict_context(seed_row, viddiff_items)
    if context[0] is None:
        return [], context[1]
    harness, state, history, latest, task = context
    rows: list[dict[str, Any]] = []
    assessment_audits: list[dict[str, Any]] = []
    assessed_ids: list[str] = []
    for assessment_index in range(1, 5):
        state = harness.render_state()
        if state.get("eligible_tools") == ["adjudicate_evidence"]:
            break
        actions = list(
            state["state_action_affordances"][
                "executable_evidence_assessment_actions"
            ]
        )
        if not actions:
            raise ValueError("assessment state exposed no executable candidate action")
        action = actions[0]
        row, audit = _emit_assessment_row(
            seed_row=seed_row,
            harness=harness,
            state=state,
            history=history,
            latest=latest,
            task=task,
            action=action,
            assessment_index=assessment_index,
        )
        rows.append(row)
        assessment_audits.append(audit)
        arguments = action["arguments"]
        assessed_ids.append(str(arguments["evidence_id"]))
        latest = {
            "status": "accepted",
            "authority": "visible_candidate_assessment_checker",
            "created_evidence_ids": [],
            "assessment_evidence_id": arguments["evidence_id"],
            "assessment_disposition": arguments["disposition"],
        }
        history.append(
            {
                "step_index": len(history),
                "past_action": {
                    "executed_tool": "assess_evidence_candidate",
                    "past_input": copy.deepcopy(arguments),
                },
                "environment_outcome": copy.deepcopy(latest),
            }
        )
    state = harness.render_state()
    if state.get("eligible_tools") != ["adjudicate_evidence"]:
        raise ValueError("complete assessment ledger did not unlock adjudication")
    final_row, final_audit = _emit_row(
        seed_row=seed_row,
        harness=harness,
        state=state,
        history=history,
        latest=latest,
        generation_source="student_visible_progressive_assessment_compiler",
        row_suffix="progressive_final_adjudication",
        task_override=task,
        compression_contract=PROGRESSIVE_COMPRESSION_CONTRACT,
        system_content_override=PROGRESSIVE_EVIDENCE_SYSTEM,
    )
    rows.append(final_row)
    return rows, {
        "case_id": _case_id(seed_row),
        "status": "compiled",
        "assessment_count": len(assessment_audits),
        "assessed_evidence_ids": assessed_ids,
        "assessment_audits": assessment_audits,
        "final_adjudication_audit": final_audit,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = _all_viddiff_items()
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    for seed_row in _first_rows(args.sft_root, "train"):
        if seed_row["metadata"]["dataset"] != "VidDiffBench" or _mode(seed_row) != "repair":
            continue
        case_id = _case_id(seed_row)
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        try:
            case_rows, audit = _compile_case(seed_row, items)
        except (KeyError, StopIteration, ValueError) as error:
            audits.append(
                {
                    "case_id": case_id,
                    "status": "skipped",
                    "error": f"{type(error).__name__}:{error}",
                }
            )
            continue
        audits.append(audit)
        rows.extend(case_rows)

    target = (
        args.output
        / "decision_sufficient_state_v6_executable_affordances"
        / "train"
    )
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "sft_all.jsonl", rows)
    compiled = [audit for audit in audits if audit.get("status") == "compiled"]
    assessment_audits = [
        assessment
        for audit in compiled
        for assessment in audit["assessment_audits"]
    ]
    final_audits = [audit["final_adjudication_audit"] for audit in compiled]
    reasoning_texts = [str(row["messages"][-1]["content"]) for row in rows]
    hard_checks = {
        "nonempty_expected_trajectory_set": bool(compiled)
        and len(rows) == 4 * len(compiled),
        "three_atomic_assessments_per_trajectory": all(
            audit["assessment_count"] == 3 for audit in compiled
        ),
        "every_assessment_executes": all(
            audit["execution_status"] == "accepted" for audit in assessment_audits
        ),
        "every_assessment_has_real_state_effect": all(
            audit["pending_candidates_after"]
            == audit["pending_candidates_before"] - 1
            for audit in assessment_audits
        ),
        "one_retain_two_reject_per_trajectory": all(
            sum(
                item["disposition"] == "retain"
                for item in audit["assessment_audits"]
            )
            == 1
            and sum(
                item["disposition"] == "reject"
                for item in audit["assessment_audits"]
            )
            == 2
            for audit in compiled
        ),
        "final_adjudication_executes": all(
            audit["execution_status"] == "accepted" for audit in final_audits
        ),
        "selected_support_binds": all(
            audit["post_adjudication_binding_status"] == "accepted"
            for audit in final_audits
        ),
        "rejected_support_counterfactual_is_blocked": all(
            audit["rejected_bind_counterfactual_status"] == "rejected"
            and audit["rejected_bind_counterfactual_reason"]
            == "evidence_not_selected_by_latest_adjudication"
            for audit in final_audits
        ),
        "reasoning_contract_clean": all(
            not audit["reasoning_contract_problems"] for audit in assessment_audits
        )
        and all(not audit["reasoning_contract_problems"] for audit in final_audits),
        "student_visible_privacy_clean": all(
            not audit["student_visible_privacy_problems"] for audit in assessment_audits
        )
        and all(not audit["student_visible_privacy_problems"] for audit in final_audits),
        "exact_reasoning_not_duplicated": len(reasoning_texts)
        == len(set(reasoning_texts)),
    }
    quality_report = {
        "artifact_class": "progressive_candidate_assessment_warmup_trajectory",
        "direct_production_sft": False,
        "recommended_use": (
            "auxiliary multi-turn warm-up mixed with complete accepted closed-loop "
            "episodes; protected full-episode evaluation remains required"
        ),
        "rows": len(rows),
        "trajectories": len(compiled),
        "hard_checks": hard_checks,
        "all_applicable_checks_passed": bool(hard_checks) and all(hard_checks.values()),
        "utqr5_applicability": {
            "measured": ["H2", "H3", "H4", "Q3", "Q5", "Q7", "Q8", "Q9"],
            "not_applicable_until_full_episode": ["Q1", "Q2", "Q4", "Q6"],
        },
    }
    report = {
        "split": "train",
        "rows": len(rows),
        "trajectories": len(compiled),
        "case_ids": [audit["case_id"] for audit in compiled],
        "target_tool_counts": {
            "assess_evidence_candidate": sum(
                row["metadata"]["target_tool"] == "assess_evidence_candidate"
                for row in rows
            ),
            "adjudicate_evidence": sum(
                row["metadata"]["target_tool"] == "adjudicate_evidence"
                for row in rows
            ),
        },
        "label_blind": True,
        "source": "student_visible_training_states_only",
        "teacher_actions_used": False,
        "quality_report": quality_report,
        "audits": audits,
    }
    write_json(args.output / "compile_report.json", report)
    write_json(args.output / "progressive_assessment_quality_report.json", quality_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
