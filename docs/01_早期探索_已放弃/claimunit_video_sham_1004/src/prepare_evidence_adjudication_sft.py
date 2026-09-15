"""Compile label-blind evidence-adjudication warm-up rows from training states.

These rows teach the native state transition, not a dataset answer: every target is
derived from the harness's student-visible compatible-evidence relation, and source
cases come only from the requested training split.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from interactive_harness import EVIDENCE_ADJUDICATION_TOOL
from pipeline import write_json, write_jsonl
from run_student_closed_loop_batch import (
    DEFAULT_SFT_ROOT,
    _all_viddiff_items,
    _case_id,
    _environment,
    _first_rows,
    _mode,
    _repair_profile,
)
from run_student_closed_loop_canary import (
    COMPRESSION_CONTRACT,
    DERIVED_META_POLICY_CONTRACT,
    _adjudication_reasoning_frontier,
    _compile_visible_decision_actions,
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
    / "outputs/student_external_contract_v1/evidence_adjudication_warmup_v1"
)


DEFECT_TEXT = {
    "claim_value_conflict": "it conflicts with the claim",
    "cross_method_direction_conflict": (
        "its A/B/c direction conflicts with the label-blind cross-method result"
    ),
    "order_inconsistent": "it is order-inconsistent",
    "lower_confidence": "it has lower confidence",
    "abstention_conflicts_with_claim": "it reports an abstention",
    "redundant_same_conclusion": "it is redundant",
}

PARALLEL_CONFLICT_POLICY = (
    " The current task may restore a completed label-blind parallel measurement batch. "
    "Those tool results are already active student-visible evidence, so do not remeasure. "
    "When adjudicate_evidence is the sole eligible action, compare every returned method, "
    "record the complete partition, and only then bind the selected support."
)


def _reasoning(state: dict[str, Any], arguments: dict[str, Any]) -> str:
    store = state.get("evidence_store") or {}
    claim_fields = (state.get("candidate_claims") or [{}])[0].get("fields") or {}
    statement = (claim_fields.get("comparison_statement") or {}).get("value")
    sample_key = (claim_fields.get("sample_key") or {}).get("value")
    selected_parts: list[str] = []
    for evidence_id in arguments["selected_evidence_ids"]:
        observation = (store[evidence_id].get("observation") or {})
        selected_parts.append(
            f"`{evidence_id}` from `{observation.get('measurement_method')}` "
            f"(`preferred_video={observation.get('preferred_video')}`)"
        )
    rejected_parts: list[str] = []
    for item in arguments["rejected_evidence"]:
        observation = store[item["evidence_id"]].get("observation") or {}
        defects = [DEFECT_TEXT[code] for code in item["defect_codes"] if code in DEFECT_TEXT][:1]
        rejected_parts.append(
            f"`{observation.get('measurement_method')}` because "
            f"{' and '.join(defects)}"
        )
    selected = "; ".join(selected_parts)
    rejected = "; ".join(rejected_parts)
    return (
        f"I select `adjudicate_evidence` because the remaining gap after the completed "
        f"per-candidate assessment ledger "
        f"is an exact partition for sample `{sample_key}` and "
        f"{json.dumps(statement, ensure_ascii=False)} before binding. "
        f"I retain {selected} because its visible values match the "
        f"claim without inferring unobserved order consistency, and reject {rejected}. This "
        f"partition agrees with every recorded disposition and avoids newest, majority, and "
        f"confidence-only shortcuts."
    )


def _native_bind_arguments(action: dict[str, Any]) -> dict[str, Any]:
    arguments = copy.deepcopy(action["arguments"])
    return {
        "claim_id": arguments["claim_id"],
        "bindings": {
            item["field"]: list(item["support_ids"])
            for item in arguments.get("binding_items") or []
        },
    }


def _post_adjudication_binding_audit(
    harness: Any,
    arguments: dict[str, Any],
    *,
    row_suffix: str,
) -> dict[str, Any]:
    """Measure both positive binding and rejected-evidence counterfactuals."""

    selected_ids = set(arguments["selected_evidence_ids"])
    rejected_ids = {
        item["evidence_id"] for item in arguments["rejected_evidence"]
    }
    post_state = harness.render_state()
    binding_actions = list(
        (post_state.get("state_action_affordances") or {}).get(
            "executable_binding_actions"
        )
        or []
    )
    if not binding_actions:
        raise ValueError("accepted adjudication did not expose a binding action")

    target_fields = set(
        (post_state.get("evidence_adjudications") or {})
        [arguments["claim_id"]]["target_fields"]
    )
    action = binding_actions[0]
    native_bind = _native_bind_arguments(action)
    target_bound_ids = {
        evidence_id
        for field_name, evidence_ids in native_bind["bindings"].items()
        if field_name in target_fields
        for evidence_id in evidence_ids
    }
    if not target_bound_ids or not target_bound_ids.issubset(selected_ids):
        raise ValueError(
            "post-adjudication binding action is not restricted to selected evidence"
        )

    rejected_id = sorted(rejected_ids)[0]
    counterfactual = harness.clone_state()
    counterfactual_field = sorted(target_fields)[0]
    counterfactual_result = counterfactual.execute_tool(
        "bind_evidence",
        {
            "claim_id": arguments["claim_id"],
            "bindings": {counterfactual_field: [rejected_id]},
        },
        f"compiled-{row_suffix}-rejected-bind-counterfactual-001",
    )
    if counterfactual_result.get("status") != "rejected":
        raise ValueError("rejected evidence remained bindable after adjudication")
    if (
        counterfactual_result.get("rejected_fields", {}).get(counterfactual_field)
        != "evidence_not_selected_by_latest_adjudication"
    ):
        raise ValueError("rejected-evidence counterfactual failed for the wrong reason")

    bind_result = harness.execute_tool(
        "bind_evidence",
        native_bind,
        f"compiled-{row_suffix}-selected-bind-001",
    )
    if bind_result.get("status") != "accepted":
        raise ValueError(f"selected evidence did not bind cleanly: {bind_result}")
    return {
        "post_adjudication_binding_status": bind_result["status"],
        "post_adjudication_target_bound_ids": sorted(target_bound_ids),
        "rejected_bind_counterfactual_status": counterfactual_result["status"],
        "rejected_bind_counterfactual_reason": (
            counterfactual_result["rejected_fields"][counterfactual_field]
        ),
    }


def _student_visible_privacy_problems(payload: dict[str, Any]) -> list[str]:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    forbidden_fragments = {
        "oracle_direction": "oracle_direction",
        "differences_gt": "differences_gt",
        "verifier_result": "verifier_result",
        "video_path": "video_path",
        "local_path": "local_path",
        "http://": "url",
        "https://": "url",
        "c:\\\\": "windows_local_path",
    }
    return sorted(
        label for fragment, label in forbidden_fragments.items() if fragment in serialized
    )


def _quality_report(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]]
) -> dict[str, Any]:
    included_audits = [
        audit
        for audit in audits
        if audit.get("execution_status") == "accepted"
        and audit.get("status") is None
    ]
    selected_method_counts: dict[str, int] = {}
    defect_code_counts: dict[str, int] = {}
    reasoning_texts: list[str] = []
    for row in rows:
        payload = json.loads(row["messages"][1]["content"])
        state = payload["current_harness_state"]
        function = row["messages"][-1]["tool_calls"][0]["function"]
        arguments = function["arguments"]
        selected_id = arguments["selected_evidence_ids"][0]
        method = str(
            ((state.get("evidence_store") or {}).get(selected_id, {}).get("observation") or {}).get(
                "measurement_method", "unknown"
            )
        )
        selected_method_counts[method] = selected_method_counts.get(method, 0) + 1
        for rejected in arguments["rejected_evidence"]:
            for code in rejected["defect_codes"]:
                defect_code_counts[code] = defect_code_counts.get(code, 0) + 1
        reasoning_texts.append(str(row["messages"][-1]["content"]))

    hard_checks = {
        "native_adjudication_executes": all(
            audit.get("execution_status") == "accepted" for audit in included_audits
        ),
        "complete_three_atom_partition": all(
            audit.get("competing_evidence_count") == 3 for audit in included_audits
        ),
        "selected_support_binds": all(
            audit.get("post_adjudication_binding_status") == "accepted"
            for audit in included_audits
        ),
        "rejected_support_counterfactual_is_blocked": all(
            audit.get("rejected_bind_counterfactual_status") == "rejected"
            and audit.get("rejected_bind_counterfactual_reason")
            == "evidence_not_selected_by_latest_adjudication"
            for audit in included_audits
        ),
        "student_visible_privacy_clean": all(
            not audit.get("student_visible_privacy_problems")
            for audit in included_audits
        ),
        "preexecution_reasoning_contract_clean": all(
            not audit.get("reasoning_contract_problems") for audit in included_audits
        ),
        "exact_reasoning_not_duplicated": len(reasoning_texts)
        == len(set(reasoning_texts)),
        "selected_method_diversity_at_least_three": len(selected_method_counts) >= 3,
    }
    return {
        "artifact_class": "single_turn_evidence_adjudication_warmup",
        "direct_production_sft": False,
        "recommended_use": (
            "auxiliary warm-up mixed with full accepted closed-loop trajectories; "
            "do not count these rows as complete episodes"
        ),
        "rows": len(rows),
        "unique_case_ids": len({row["metadata"]["case_id"] for row in rows}),
        "hard_checks": hard_checks,
        "all_applicable_checks_passed": bool(hard_checks) and all(hard_checks.values()),
        "coverage": {
            "selected_method_counts": selected_method_counts,
            "rejected_defect_code_counts": defect_code_counts,
        },
        "utqr5_applicability": {
            "measured": ["H2", "H3", "H4", "Q3", "Q5", "Q7", "Q8", "Q9"],
            "not_applicable_to_single_turn_warmup": ["Q1", "Q2", "Q4", "Q6"],
            "note": (
                "Outcome correctness, efficiency, stopping, and full causal coherence "
                "must be measured on protected full closed-loop episodes."
            ),
        },
    }


def _emit_row(
    *,
    seed_row: dict[str, Any],
    harness: Any,
    state: dict[str, Any],
    history: list[dict[str, Any]],
    latest: dict[str, Any],
    generation_source: str,
    row_suffix: str,
    task_override: dict[str, Any] | None = None,
    system_suffix: str = "",
    compression_contract: str = COMPRESSION_CONTRACT,
    system_content_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = list(state.get("eligible_tools") or [])
    if eligible != ["adjudicate_evidence"]:
        raise ValueError(f"expected sole adjudication action, got {eligible}")
    _compile_visible_decision_actions(state)
    action = state["state_action_affordances"]["executable_adjudication_actions"][0]
    arguments = copy.deepcopy(action["arguments"])
    policy_state, policy_history = _post_meta_policy_view(state, history)
    task = _post_meta_task_view(
        copy.deepcopy(task_override or _payload(seed_row)["task"]), policy_state
    )
    tools = build_student_tools(
        harness.claim_schema,
        harness.tool_registry,
        eligible,
        executable_actions=[action],
    )
    user_payload = {
        "task": task,
        "current_harness_state": policy_state,
        "past_execution_history": policy_history,
        "latest_environment_observation": latest,
        "compression_contract": compression_contract,
        "execution_frontier": _execution_frontier(eligible, tools, [action]),
        "evidence_adjudication_frontier": _adjudication_reasoning_frontier(
            state, [action]
        ),
    }
    privacy_problems = _student_visible_privacy_problems(user_payload)
    if privacy_problems:
        raise ValueError(f"student-visible privacy failure: {privacy_problems}")
    reasoning = _reasoning(state, arguments)
    problems = _preexecution_reasoning_problems(
        reasoning,
        "adjudicate_evidence",
        arguments,
        meta_visible=False,
        visible_state={**state, "latest_environment_observation": latest},
    )
    if problems:
        raise ValueError(f"compiled adjudication reasoning failed: {problems}")
    result = harness.execute_tool(
        "adjudicate_evidence", arguments, f"compiled-{row_suffix}-adjudication-001"
    )
    if result.get("status") != "accepted":
        raise ValueError(f"compiled adjudication was not accepted: {result}")
    binding_audit = _post_adjudication_binding_audit(
        harness, arguments, row_suffix=row_suffix
    )
    if system_content_override is not None:
        system = {"role": "system", "content": system_content_override}
    else:
        system = copy.deepcopy(seed_row["messages"][0])
        if DERIVED_META_POLICY_CONTRACT not in str(system.get("content", "")):
            system["content"] = str(system.get("content", "")) + DERIVED_META_POLICY_CONTRACT
        system["content"] = str(system.get("content", "")) + system_suffix
    case_id = _case_id(seed_row)
    row = {
        "messages": [
            system,
            {
                "role": "user",
                "content": json.dumps(
                    user_payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
            {
                "role": "assistant",
                "content": reasoning,
                "tool_calls": [
                    {
                        "id": f"warmup-{case_id}-{row_suffix}-adjudicate",
                        "type": "function",
                        "function": {
                            "name": "adjudicate_evidence",
                            "arguments": arguments,
                        },
                    }
                ],
            },
        ],
        "tools": tools,
        "metadata": {
            "row_id": f"VidDiffBench:{case_id}:{row_suffix}:repair:1",
            "dataset": "VidDiffBench",
            "case_id": case_id,
            "split": "train",
            "target_tool": "adjudicate_evidence",
            "trajectory_mode": "repair",
            "turn_index": len(policy_history) + 1,
            "target_execution_status": "accepted",
            "target_includes_observation": False,
            "loss_roles": ["assistant"],
            "loss_scope": "last_assistant_action_only",
            "oracle_or_verifier_used": False,
            "teacher_action_used": False,
            "strict_closed_loop_accepted": True,
            "generation_source": generation_source,
        },
    }
    audit = {
        "case_id": case_id,
        "row_suffix": row_suffix,
        "competing_evidence_count": len(
            state["evidence_adjudication"]["requirement"]["competing_evidence_ids"]
        ),
        "selected_evidence_ids": arguments["selected_evidence_ids"],
        "rejected_evidence_ids": [
            item["evidence_id"] for item in arguments["rejected_evidence"]
        ],
        "reasoning_contract_problems": problems,
        "student_visible_privacy_problems": privacy_problems,
        "execution_status": result["status"],
        **binding_audit,
    }
    return row, audit


def _compile_row(
    seed_row: dict[str, Any],
    viddiff_items: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    environment, registry = _environment(seed_row, "repair", viddiff_items)
    harness = _restore_harness(seed_row)
    harness.environment = environment
    harness.tool_registry = [
        *copy.deepcopy(registry),
        copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL),
    ]
    history: list[dict[str, Any]] = []
    latest: dict[str, Any] = {}
    for step in range(10):
        state = harness.render_state()
        eligible = list(state.get("eligible_tools") or [])
        if eligible == ["adjudicate_evidence"]:
            return _emit_row(
                seed_row=seed_row,
                harness=harness,
                state=state,
                history=history,
                latest=latest,
                generation_source="student_visible_state_compiled_warmup",
                row_suffix="adjudication_warmup",
            )

        reopen = (state.get("action_constraints") or {}).get(
            "must_reopen_before_rebinding"
        ) or []
        if "set_field_decision" in eligible and reopen:
            native = {
                "claim_id": reopen[0]["claim_id"],
                "fields": reopen[0]["fields"],
                "decision_status": "open",
            }
            name = "set_field_decision"
        else:
            external = [name for name in eligible if name.startswith("compare_")]
            if not external:
                return None, {"case_id": _case_id(seed_row), "status": "no_conflict"}
            name = external[0]
            candidate = next(
                item for item in environment.item["candidate_tools"] if item["name"] == name
            )
            native = copy.deepcopy(candidate["arguments"])
        result = harness.execute_tool(name, native, f"compiled-prefix-{step + 1:03d}")
        latest = _result_summary(result)
        history.append(
            {
                "step_index": step,
                "past_action": {"executed_tool": name, "past_input": native},
                "environment_outcome": copy.deepcopy(latest),
            }
        )
    return None, {"case_id": _case_id(seed_row), "status": "step_limit"}


def _compile_parallel_conflict_row(
    seed_row: dict[str, Any],
    viddiff_items: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Restore a claim after a completed label-blind parallel tool fan-out."""

    case_id = _case_id(seed_row)
    sample_key = case_id.removeprefix("viddiff-")
    item = viddiff_items[sample_key]
    environment, registry = _environment(seed_row, "clean", viddiff_items)
    harness = _restore_harness(seed_row)
    harness.environment = environment
    harness.tool_registry = [
        *copy.deepcopy(registry),
        copy.deepcopy(EVIDENCE_ADJUDICATION_TOOL),
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
    if state.get("eligible_tools") != ["adjudicate_evidence"]:
        return None, {
            "case_id": case_id,
            "status": "parallel_state_not_adjudicable",
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
        "Resolve a completed label-blind parallel measurement batch, bind only the "
        "visibly compatible support, and preserve the existing claim values."
    )
    task["mutation"] = {
        "operator": "parallel_evidence_return",
        "added_evidence_ids": created_ids,
        "media_changed": False,
        "claim_values_changed": False,
    }
    return _emit_row(
        seed_row=seed_row,
        harness=harness,
        state=state,
        history=history,
        latest=latest,
        generation_source="student_visible_parallel_conflict_compiler",
        row_suffix="parallel_adjudication_warmup",
        task_override=task,
        system_suffix=PARALLEL_CONFLICT_POLICY,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = _all_viddiff_items()
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seen_action_signatures: set[str] = set()
    for seed_row in _first_rows(args.sft_root, args.split):
        if seed_row["metadata"]["dataset"] != "VidDiffBench" or _mode(seed_row) != "repair":
            continue
        state = _payload(seed_row)["current_harness_state"]
        compilers = []
        if _repair_profile(state)[0] == "evidence_dropout":
            compilers.append(("natural_repair", _compile_row))
        if args.include_parallel_conflicts:
            compilers.append(("parallel_return", _compile_parallel_conflict_row))
        for compiler_name, compiler in compilers:
            try:
                row, audit = compiler(seed_row, items)
            except (KeyError, StopIteration, ValueError) as error:
                audits.append(
                    {
                        "case_id": _case_id(seed_row),
                        "compiler": compiler_name,
                        "status": "skipped",
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
                continue
            audit["compiler"] = compiler_name
            audits.append(audit)
            if (
                row is not None
                and args.max_competing_evidence is not None
                and int(audit.get("competing_evidence_count", 0))
                > args.max_competing_evidence
            ):
                audit["status"] = "excluded_by_competitor_limit"
                continue
            if row is not None:
                function = row["messages"][-1]["tool_calls"][0]["function"]
                signature = json.dumps(
                    {
                        "case_id": row["metadata"]["case_id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    },
                    sort_keys=True,
                )
                if signature in seen_action_signatures:
                    audit["status"] = "excluded_duplicate_case_action"
                    continue
                seen_action_signatures.add(signature)
                rows.append(row)
    target = args.output / "decision_sufficient_state_v6_executable_affordances" / args.split
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "sft_all.jsonl", rows)
    quality_report = _quality_report(rows, audits)
    report = {
        "split": args.split,
        "rows": len(rows),
        "case_ids": [row["metadata"]["case_id"] for row in rows],
        "target_tool_counts": {"adjudicate_evidence": len(rows)},
        "label_blind": True,
        "source": "student_visible_training_states_only",
        "parallel_conflict_compilation_enabled": args.include_parallel_conflicts,
        "quality_report": quality_report,
        "audits": audits,
    }
    write_json(args.output / "compile_report.json", report)
    write_json(args.output / "adjudication_warmup_quality_report.json", quality_report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-competing-evidence", type=int)
    parser.add_argument("--include-parallel-conflicts", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
