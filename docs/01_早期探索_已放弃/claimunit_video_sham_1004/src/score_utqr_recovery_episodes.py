"""Produce evidence-linked UTQR scorecards without impersonating human review."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_recovery_reasoning_quality import audit_rows
from audit_recovery_step_value import audit_episode_step_value
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BATCH = BASE / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio"
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = DEFAULT_BATCH / "utqr_episode_scorecards.json"

CORE_STATE_KEYS = (
    "candidate_claims",
    "field_bindings",
    "evidence_store",
    "evidence_assessments",
    "evidence_adjudications",
    "evidence_challenges",
    "open_evidence_challenges",
    "open_high_priority_obligations",
    "terminal",
    "stop_decision",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _messages_state(message: dict[str, Any], *, initial: bool = False) -> dict[str, Any]:
    content = message.get("content")
    payload = json.loads(content) if isinstance(content, str) else content
    if not isinstance(payload, dict):
        return {}
    key = "current_harness_state" if initial else "harness_state"
    state = payload.get(key)
    return state if isinstance(state, dict) else {}


def _projection(state: dict[str, Any]) -> str:
    core = {key: state.get(key) for key in CORE_STATE_KEYS}
    return json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stage_delta(
    stage: str,
    before: dict[str, Any],
    after: dict[str, Any],
    result: dict[str, Any],
) -> bool:
    if _projection(before) == _projection(after):
        return False
    if stage == "challenge":
        return len(after.get("evidence_challenges") or []) > len(
            before.get("evidence_challenges") or []
        )
    if stage == "fresh_measurement":
        created = result.get("created_evidence_ids") or []
        return bool(created) and all(
            evidence_id in (after.get("evidence_store") or {})
            for evidence_id in created
        )
    if stage == "local_reopen":
        return before.get("candidate_claims") != after.get("candidate_claims")
    if stage == "claim_revision":
        return before.get("candidate_claims") != after.get("candidate_claims")
    if stage == "evidence_assessment":
        return before.get("evidence_assessments") != after.get("evidence_assessments")
    if stage == "evidence_adjudication":
        return before.get("evidence_adjudications") != after.get(
            "evidence_adjudications"
        )
    if stage == "fresh_binding":
        return before.get("field_bindings") != after.get("field_bindings")
    if stage == "recommit":
        return before.get("candidate_claims") != after.get("candidate_claims")
    if stage == "terminal_verification":
        return before.get("terminal") is not True and after.get("terminal") is True
    return True


def _final_binding_integrity(state: dict[str, Any]) -> bool:
    claims = state.get("candidate_claims") or []
    if len(claims) != 1 or state.get("terminal") is not True:
        return False
    evidence = state.get("evidence_store") or {}
    fields = (claims[0].get("fields") or {}).values()
    for field in fields:
        if field.get("evidence_status") != "supported":
            return False
        if field.get("decision_status") != "committed":
            return False
        support_ids = field.get("support_ids") or []
        if not support_ids:
            return False
        if any(
            evidence_id not in evidence
            or (evidence[evidence_id] or {}).get("active") is not True
            for evidence_id in support_ids
        ):
            return False
    return not state.get("open_high_priority_obligations")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _tool_events(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover the student-visible state around each executed tool call."""

    messages = episode["messages"]
    before = _messages_state(messages[1], initial=True)
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or len(message.get("tool_calls") or []) != 1:
            continue
        if index + 1 >= len(messages):
            continue
        call = message["tool_calls"][0]["function"]
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result_message = messages[index + 1]
        content = result_message.get("content")
        result = json.loads(content) if isinstance(content, str) else content
        result = result if isinstance(result, dict) else {}
        after = _messages_state(result_message)
        events.append(
            {
                "message_index": index,
                "tool": str(call["name"]),
                "arguments": arguments,
                "result": result,
                "before": before,
                "after": after,
            }
        )
        before = after
    return events


def _same_visible_value(left: Any, right: Any) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
        right, ensure_ascii=False, sort_keys=True
    )


def _expected_evidence_first_selection(
    evidence_store: dict[str, Any], requirement: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Recompute the winner from visible atoms, never from the current claim."""

    candidates = [
        evidence_id
        for evidence_id in requirement.get("competing_evidence_ids") or []
        if evidence_id in evidence_store
        and (evidence_store[evidence_id] or {}).get("active") is True
    ]
    target_fields = list(requirement.get("target_fields") or [])
    complete = [
        evidence_id
        for evidence_id in candidates
        if all(
            field in ((evidence_store[evidence_id] or {}).get("supports_fields") or [])
            and field in ((evidence_store[evidence_id] or {}).get("observation") or {})
            for field in target_fields
        )
        and ((evidence_store[evidence_id] or {}).get("provenance") or {}).get(
            "role_assignment_valid_for_current_contract", True
        )
        is not False
    ]
    if not complete:
        raise ValueError("no complete active candidate")

    def confidence(evidence_id: str) -> float:
        value = ((evidence_store[evidence_id] or {}).get("observation") or {}).get(
            "confidence"
        )
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else -1.0
        )

    preferred = [
        ((evidence_store[evidence_id] or {}).get("observation") or {}).get(
            "preferred_video"
        )
        for evidence_id in complete
    ]
    diagnostics: dict[str, Any] = {
        "candidate_count": len(candidates),
        "complete_candidate_count": len(complete),
        "method_representatives": {},
        "method_votes": {},
    }
    if all(value in {"a", "b", "c"} for value in preferred):
        representatives: dict[str, str] = {}
        for evidence_id in complete:
            atom = evidence_store[evidence_id] or {}
            observation = atom.get("observation") or {}
            method = str(
                observation.get("measurement_method")
                or (atom.get("provenance") or {}).get("method")
                or evidence_id
            )
            incumbent = representatives.get(method)
            if incumbent is None or (confidence(evidence_id), evidence_id) > (
                confidence(incumbent),
                incumbent,
            ):
                representatives[method] = evidence_id
        votes = Counter(
            str(
                ((evidence_store[evidence_id] or {}).get("observation") or {}).get(
                    "preferred_video"
                )
            )
            for evidence_id in representatives.values()
        )
        directional = [value for value in ("a", "b") if votes[value] >= 2]
        if len(directional) == 1 and votes["b" if directional[0] == "a" else "a"] == 0:
            selected_value = directional[0]
        elif votes["c"]:
            selected_value = "c"
        elif votes["a"] != votes["b"]:
            selected_value = "a" if votes["a"] > votes["b"] else "b"
        else:
            raise ValueError("cross-method evidence is tied without abstention")
        matching = [
            evidence_id
            for evidence_id in complete
            if ((evidence_store[evidence_id] or {}).get("observation") or {}).get(
                "preferred_video"
            )
            == selected_value
        ]
        stable = [
            evidence_id
            for evidence_id in matching
            if selected_value == "c"
            or ((evidence_store[evidence_id] or {}).get("observation") or {}).get(
                "order_consistent"
            )
            is not False
        ]
        pool = stable or matching
        best_confidence = max(confidence(item) for item in pool)
        acceptable = [
            item for item in pool if confidence(item) == best_confidence
        ]
        selected = [max(acceptable)]
        diagnostics["method_representatives"] = representatives
        diagnostics["method_votes"] = dict(votes)
        diagnostics["selected_value"] = selected_value
        diagnostics["acceptable_selected_evidence_ids"] = sorted(acceptable)
        return selected, diagnostics

    best_confidence = max(confidence(item) for item in complete)
    acceptable = [item for item in complete if confidence(item) == best_confidence]
    diagnostics["acceptable_selected_evidence_ids"] = sorted(acceptable)
    return [max(acceptable)], diagnostics


def _adjudication_integrity(episode: dict[str, Any]) -> dict[str, Any]:
    """Audit evidence ordering, independent voting, revision, and rejected re-entry."""

    events = _tool_events(episode)
    required_seen = any(
        ((event["before"].get("evidence_adjudication") or {}).get("required"))
        for event in events
    )
    adjudication_indices = [
        index for index, event in enumerate(events) if event["tool"] == "adjudicate_evidence"
    ]
    audits: list[dict[str, Any]] = []
    for position, event_index in enumerate(adjudication_indices):
        event = events[event_index]
        requirement = (
            (event["before"].get("evidence_adjudication") or {}).get("requirement")
            or {}
        )
        evidence_store = event["before"].get("evidence_store") or {}
        arguments = event["arguments"]
        actual_selected = list(arguments.get("selected_evidence_ids") or [])
        rejected_ids = [
            str(item.get("evidence_id"))
            for item in arguments.get("rejected_evidence") or []
        ]
        competitors = list(requirement.get("competing_evidence_ids") or [])
        target_fields = list(requirement.get("target_fields") or [])
        basis = str(requirement.get("selection_basis") or "").lower()
        try:
            expected_selected, vote_diagnostics = _expected_evidence_first_selection(
                evidence_store, requirement
            )
            recompute_error = None
        except ValueError as exc:
            expected_selected, vote_diagnostics = [], {}
            recompute_error = str(exc)

        next_index = (
            adjudication_indices[position + 1]
            if position + 1 < len(adjudication_indices)
            else len(events)
        )
        following = events[event_index + 1 : next_index]
        binding_events = [item for item in following if item["tool"] == "bind_evidence"]
        binding = binding_events[0] if binding_events else None
        binding_items = (
            list(binding["arguments"].get("binding_items") or []) if binding else []
        )
        relevant_binding = {
            str(item.get("field")): list(item.get("support_ids") or [])
            for item in binding_items
            if str(item.get("field")) in target_fields
        }
        binding_selected_only = bool(target_fields) and all(
            field in relevant_binding
            and bool(relevant_binding[field])
            and set(relevant_binding[field]).issubset(set(actual_selected))
            for field in target_fields
        )

        final_state = events[-1]["after"] if events else {}
        claim_id = str(arguments.get("claim_id") or requirement.get("claim_id") or "")
        final_bindings = (final_state.get("field_bindings") or {}).get(claim_id) or {}
        final_selected_only = all(
            field in final_bindings
            and bool((final_bindings[field] or {}).get("support_ids"))
            and set((final_bindings[field] or {}).get("support_ids") or []).issubset(
                set(actual_selected)
            )
            for field in target_fields
        )
        rejected_reentered = any(
            rejected_id
            in {
                evidence_id
                for field in target_fields
                for evidence_id in (
                    (final_bindings.get(field) or {}).get("support_ids") or []
                )
            }
            for rejected_id in rejected_ids
        )

        claims_before = {
            str(item.get("claim_id")): item
            for item in event["before"].get("candidate_claims") or []
        }
        claim_before = claims_before.get(claim_id) or {}
        selected_observation = (
            (evidence_store.get(actual_selected[0]) or {}).get("observation") or {}
            if len(actual_selected) == 1
            else {}
        )
        changed_fields = [
            field
            for field in target_fields
            if field in selected_observation
            and not _same_visible_value(
                (((claim_before.get("fields") or {}).get(field) or {}).get("value")),
                selected_observation[field],
            )
        ]
        revision_events = [
            item for item in following if item["tool"] == "revise_claim_fields"
        ]
        revision_updates = {
            str(item.get("field")): item.get("value")
            for revision in revision_events
            if str(revision["arguments"].get("claim_id")) == claim_id
            for item in revision["arguments"].get("field_updates") or []
        }
        exact_revision = all(
            field in revision_updates
            and _same_visible_value(revision_updates[field], selected_observation[field])
            for field in changed_fields
        )

        partition_exact = (
            len(actual_selected) == 1
            and len(set(actual_selected + rejected_ids))
            == len(actual_selected) + len(rejected_ids)
            and set(actual_selected + rejected_ids) == set(competitors)
        )
        checks = {
            "requirement_present": bool(requirement),
            "claim_independent_rule_declared": (
                "one vote per measurement method" in basis
                and "current claim values do not select" in basis
            ),
            "selected_matches_visible_recompute": (
                recompute_error is None
                and len(actual_selected) == 1
                and actual_selected[0]
                in vote_diagnostics.get(
                    "acceptable_selected_evidence_ids", expected_selected
                )
            ),
            "candidate_partition_exact": partition_exact,
            "claim_revision_exact_when_needed": exact_revision,
            "binding_uses_selected_only": binding_selected_only,
            "final_binding_uses_selected_only": final_selected_only,
            "rejected_evidence_does_not_reenter": not rejected_reentered,
        }
        audits.append(
            {
                "claim_id": claim_id,
                "target_fields": target_fields,
                "expected_selected_evidence_ids": expected_selected,
                "actual_selected_evidence_ids": actual_selected,
                "changed_fields_requiring_revision": changed_fields,
                "recompute_error": recompute_error,
                "vote_diagnostics": vote_diagnostics,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    applicable = required_seen or bool(adjudication_indices)
    return {
        "applicable": applicable,
        "status": "not_applicable" if not applicable else "audited",
        "required_state_seen": required_seen,
        "adjudications": len(audits),
        "passed": (not applicable) or (bool(audits) and all(item["passed"] for item in audits)),
        "details": audits,
    }


def _action_audit(
    episode: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    messages = episode["messages"]
    actions: list[dict[str, Any]] = []
    before = _messages_state(messages[1], initial=True)
    row_index = 0
    signatures: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or index + 1 >= len(messages):
            actions.append(
                {"tool": None, "eligible": False, "material_delta": False, "status": None}
            )
            continue
        call = calls[0]["function"]
        tool = str(call["name"])
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result_message = messages[index + 1]
        content = result_message.get("content")
        result = json.loads(content) if isinstance(content, str) else content
        result = result if isinstance(result, dict) else {}
        after = _messages_state(result_message)
        stage = str(rows[row_index]["metadata"]["causal_recovery_stage"])
        signature = json.dumps(
            {"tool": tool, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        signatures.append(signature)
        actions.append(
            {
                "turn": row_index + 1,
                "stage": stage,
                "tool": tool,
                "eligible": tool in (before.get("eligible_tools") or []),
                "status": result.get("status"),
                "executed_or_accepted": result.get("status") in {"accepted", "executed"},
                "material_delta": _stage_delta(stage, before, after, result),
            }
        )
        before = after
        row_index += 1
    count = len(actions)
    return {
        "actions": actions,
        "action_count": count,
        "all_single_tool": count == len(rows),
        "eligible_rate": round(sum(item["eligible"] for item in actions) / count, 4),
        "executed_rate": round(
            sum(item["executed_or_accepted"] for item in actions) / count, 4
        ),
        "material_state_delta_rate": round(
            sum(item["material_delta"] for item in actions) / count, 4
        ),
        "exact_duplicate_action_count": len(signatures) - len(set(signatures)),
        "all_actions_useful": all(
            item["eligible"]
            and item["executed_or_accepted"]
            and item["material_delta"]
            for item in actions
        ),
        "final_state": before,
    }


def score_batch(
    episodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    compile_report: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    rows_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_episode[str(row["metadata"]["episode_id"])].append(row)
    compile_by_episode = {
        str(item["episode_id"]): item for item in compile_report.get("results") or []
    }
    weights = {item["id"]: int(item["weight"]) for item in rubric["criteria"]}
    scorecards: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_rows = rows_by_episode[episode_id]
        result = compile_by_episode[episode_id]
        action = _action_audit(episode, episode_rows)
        step_value = audit_episode_step_value(episode, episode_rows)
        stages = [str(row["metadata"]["causal_recovery_stage"]) for row in episode_rows]
        language = audit_rows(episode_rows)["metrics"]
        language_rows_pass = language["row_passed"] == len(episode_rows)
        final_integrity = _final_binding_integrity(action.pop("final_state"))
        target_hygiene = all(
            row["metadata"].get("target_was_executed") is True
            and row["metadata"].get("compiled_actions_visible") is False
            and row["metadata"].get("private_intervention_visible") is False
            and row["metadata"].get("loss_scope") == "last_assistant_action_only"
            and float(row["metadata"].get("learning_value_weight") or 0) > 0
            for row in episode_rows
        )
        stage_order = [
            "challenge",
            "fresh_measurement",
            "local_reopen",
            "fresh_binding",
            "recommit",
            "terminal_verification",
        ]
        ordered = all(stage in stages for stage in stage_order) and all(
            stages.index(left) < stages.index(right)
            for left, right in zip(stage_order, stage_order[1:])
        )
        revision_rows = [
            row for row in episode_rows if row["metadata"]["causal_recovery_stage"] == "claim_revision"
        ]
        revision_separate = all(
            row["metadata"]["target_tool"] == "revise_claim_fields"
            for row in revision_rows
        )
        strict_step_value = str(rubric.get("rubric_id")) == "UTQR-12"
        legacy_c5 = round(
            (
                action["eligible_rate"]
                + action["material_state_delta_rate"]
                + float(action["exact_duplicate_action_count"] == 0)
            )
            / 3,
            4,
        )
        criteria = {
            "C1": float(
                action["executed_rate"] == 1.0
                and result.get("terminal_success") is True
                and final_integrity
            ),
            "C2": float(
                not episode.get("student_visible_leakage_hits") and final_integrity
            ),
            "C3": float(
                episode.get("clean_control_suppresses_challenge") is True
                and episode.get("defect_target_moves_with_atom") is True
                and episode.get("fresh_removal_blocks_repair") is True
            ),
            "C4": float(episode.get("independent_fields_preserved") is True),
            "C5": (
                float(
                    action["eligible_rate"] == 1.0
                    and action["material_state_delta_rate"] == 1.0
                    and action["exact_duplicate_action_count"] == 0
                    and step_value["passed"]
                )
                if strict_step_value
                else legacy_c5
            ),
            "C6": float(ordered and stages[0] == "challenge"),
            "C7": float(
                episode.get("fresh_evidence_consumed") is True
                and ordered
                and final_integrity
            ),
            "C8": float(revision_separate and final_integrity),
            "C9": None,
            "C10": float(target_hygiene and language_rows_pass),
        }
        adjudication_integrity = _adjudication_integrity(episode)
        if "C11" in weights:
            criteria["C11"] = float(adjudication_integrity["passed"])
        auto_score = round(
            sum(weights[key] * value for key, value in criteria.items() if value is not None),
            3,
        )
        language_proxy = round(language["row_passed"] / len(episode_rows), 4)
        blind_proxy = {
            "prospective_reasoning": round(
                1 + 3.5 * min(action["eligible_rate"], action["material_state_delta_rate"]), 3
            ),
            "visible_evidence_grounding": round(1 + 3.5 * language_proxy, 3),
            "belief_update": round(1 + 3.5 * min(criteria["C6"], criteria["C7"]), 3),
            "method_fit": round(1 + 3.5 * criteria["C5"], 3),
            "locality": round(1 + 3.5 * criteria["C4"], 3),
            "naturalness": round(1 + 3.0 * language_proxy, 3),
        }
        scorecards.append(
            {
                "episode_id": episode_id,
                "dataset": episode["dataset"],
                "criteria": criteria,
                "automated_core_score": auto_score,
                "automated_core_maximum": sum(
                    weight for key, weight in weights.items() if key != "C9"
                ),
                "machine_language_proxy": language_proxy,
                "blind_dimension_proxy": blind_proxy,
                "blind_dimension_proxy_mean": round(
                    sum(blind_proxy.values()) / len(blind_proxy), 3
                ),
                "human_blind_score": None,
                "human_review_status": "pending",
                "hard_gates": {
                    "H1_to_H7_automated": all(
                        value == 1.0 for key, value in criteria.items() if key != "C9"
                    ),
                    "H8_independent_blind_review": "pending",
                },
                "action_utility": action,
                "step_value": step_value,
                "evidence_first_adjudication": adjudication_integrity,
            }
        )

    dataset_summary: dict[str, Any] = {}
    for dataset in sorted({item["dataset"] for item in scorecards}):
        items = [item for item in scorecards if item["dataset"] == dataset]
        scores = [float(item["automated_core_score"]) for item in items]
        proxies = [float(item["blind_dimension_proxy_mean"]) for item in items]
        dataset_summary[dataset] = {
            "episodes": len(items),
            "automated_core_mean": round(sum(scores) / len(scores), 3),
            "automated_core_p10": _percentile(scores, 0.10),
            "automated_core_min": min(scores),
            "blind_proxy_mean": round(sum(proxies) / len(proxies), 3),
            "all_actions_useful": all(
                item["action_utility"]["all_actions_useful"] for item in items
            ),
            "all_steps_have_downstream_decision_role": all(
                item["step_value"]["passed"] for item in items
            ),
            "human_review_completed": False,
        }
    return {
        "artifact": "utqr_episode_scorecards_v1",
        "rubric_id": rubric["rubric_id"],
        "score_semantics": {
            "automated_core": (
                "All non-C9 criteria only; maximum 92/100. C9 and H8 remain pending until "
                "independent human review."
            ),
            "blind_dimension_proxy": (
                "A diagnostic ceiling of 4.5/5 for machine-checkable dimensions and "
                "4.0/5 for naturalness; never counted as human review."
            ),
        },
        "episodes": len(scorecards),
        "dataset_summary": dataset_summary,
        "batch_automated_core_p10": _percentile(
            [float(item["automated_core_score"]) for item in scorecards], 0.10
        ),
        "all_H1_to_H7_automated": all(
            item["hard_gates"]["H1_to_H7_automated"] for item in scorecards
        ),
        "H8_independent_blind_review": "pending",
        "scorecards": scorecards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = score_batch(
        _jsonl(args.batch_root / "recovery_episodes.jsonl"),
        _jsonl(args.batch_root / "recovery_sft_candidate.jsonl"),
        _load(args.batch_root / "compile_report.json"),
        _load(args.rubric),
    )
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "episodes": report["episodes"],
        "dataset_summary": report["dataset_summary"],
        "batch_automated_core_p10": report["batch_automated_core_p10"],
        "H8": report["H8_independent_blind_review"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
