"""Counterfactual step-utility audit for executable teacher trajectories.

The audit replays recorded environment outputs through a fresh harness.  For
each action it removes that action and replays the fixed suffix.  A training
target is eligible only when it was accepted, its rationale matches visible
state, and removing it prevents the same valid terminal result.  External tool
steps additionally undergo observation dropout to detect answer-pattern
shortcuts that ignore tool evidence.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from interactive_harness import EvidenceAtom, TaskContract, VideoForensicHarness
from sft_learning_value import learning_value_profile


ACCEPTED_STATUSES = {"accepted", "executed"}
# Dataset adapters may preserve lowercase tool names and underscores in an
# evidence identifier (for example ``EA-VIDDIFF-compare_global-001``).  Match
# the complete identifier so grounding checks do not mistake a valid ID for
# the truncated, nonexistent prefix ``EA-VIDDIFF-``.
EVIDENCE_ID_PATTERN = re.compile(r"\bEA-[A-Za-z0-9][A-Za-z0-9_-]*\b")
OBSERVATION_TOKEN_PATTERN = re.compile(r"[A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff]+")
GENERIC_OBSERVATION_TOKENS = {
    "arguments",
    "evidence",
    "observation",
    "reference",
    "result",
    "source",
    "suspect",
    "value",
}


@dataclass
class RecordedToolEnvironment:
    payloads: dict[str, list[dict[str, Any]]]

    def __post_init__(self) -> None:
        self.indices = {name: 0 for name in self.payloads}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        queue = self.payloads.get(name, [])
        index = self.indices.get(name, 0)
        if index >= len(queue):
            raise ValueError(f"no recorded observation remains for {name}")
        payload = copy.deepcopy(queue[index])
        self.indices[name] = index + 1
        recorded_arguments = payload.get("arguments")
        if recorded_arguments != arguments:
            raise ValueError(
                f"replay arguments differ for {name}; recorded={recorded_arguments}, replay={arguments}"
            )
        return payload


def _recorded_payloads(bundle: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    payloads: dict[str, list[dict[str, Any]]] = {}
    for episode_name in ("clean_episode", "repair_episode"):
        for event in bundle[episode_name]["events"]:
            if event["action_type"] != "external_tool_execution":
                continue
            result = event["result"]
            observation = copy.deepcopy(result["observation"])
            observation["evidence_atoms"] = [EvidenceAtom(**atom) for atom in result["evidence_atoms"]]
            payloads.setdefault(event["action"], []).append(observation)
    return payloads


def _new_harness(bundle: dict[str, Any]) -> tuple[VideoForensicHarness, RecordedToolEnvironment]:
    contract = TaskContract(**bundle["clean_episode"]["final_state"]["task_contract"])
    environment = RecordedToolEnvironment(_recorded_payloads(bundle))
    config = bundle.get("harness_config", {})
    return VideoForensicHarness(
        contract,
        environment,
        claim_schema=config.get("claim_schema"),
        tool_registry=config.get("tool_registry"),
        target_fields=config.get("target_fields"),
        invariant_fields=config.get("invariant_fields"),
    ), environment


def _execute(harness: VideoForensicHarness, turn: dict[str, Any], suffix: str) -> dict[str, Any]:
    try:
        return harness.execute_tool(
            turn["tool_name"],
            copy.deepcopy(turn["params"]),
            f"audit_{suffix}_{turn['turn_idx']:03d}",
        )
    except (ValueError, RuntimeError) as error:
        return {"status": "rejected", "error": str(error), "harness_state": harness.render_state()}


def _prepare_repair_harness(bundle: dict[str, Any]) -> VideoForensicHarness:
    harness, _environment = _new_harness(bundle)
    for turn in bundle["clean_episode"]["turn_history"]:
        _execute(harness, turn, "repair_base")
    repair_harness = harness.clone_state()
    summary = bundle["execution_summary"]
    operator = summary.get("repair_operator", "evidence_dropout")
    evidence_id = summary.get("mutation_evidence_id", summary.get("dropped_evidence_id"))
    atom = repair_harness.evidence_store.get(evidence_id)
    if operator in {"evidence_dropout", "binding_invalidation"} and (
        atom is None or not atom.active
    ):
        claim_id = next(iter(repair_harness.claims), None)
        equivalent_ids = []
        if claim_id is not None:
            for candidate_id, candidate in repair_harness.evidence_store.items():
                if not candidate.active:
                    continue
                supports_all_targets = all(
                    candidate_id in repair_harness.bindings[claim_id][field_name].support_ids
                    for field_name in repair_harness.target_fields
                )
                if supports_all_targets:
                    equivalent_ids.append(candidate_id)
        if len(equivalent_ids) != 1:
            raise ValueError(
                "recorded mutation evidence is unavailable and replay does not contain "
                f"exactly one equivalent bound atom: requested={evidence_id!r}, "
                f"equivalent={equivalent_ids!r}"
            )
        evidence_id = equivalent_ids[0]
    if operator == "evidence_dropout":
        repair_harness.inject_evidence_dropout(evidence_id)
    elif operator == "binding_invalidation":
        repair_harness.inject_binding_invalidation(evidence_id)
    elif operator == "decision_reopen":
        repair_harness.inject_decision_reopen()
    else:
        raise ValueError(f"unsupported repair operator in audit: {operator}")
    return repair_harness


def _state_projection(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "claims": [
            {
                "claim_id": claim["claim_id"],
                "fields": {
                    name: {
                        "value": item["value"],
                        "evidence_status": item["evidence_status"],
                        "decision_status": item["decision_status"],
                        "support_ids": item["support_ids"],
                    }
                    for name, item in claim["fields"].items()
                },
            }
            for claim in state["candidate_claims"]
        ],
        "open_high_priority_obligations": state["open_high_priority_obligations"],
        "terminal": state["terminal"],
        "stop_allowed": bool((state.get("stop_decision") or {}).get("allowed")),
    }


def _replay_counterfactual(
    bundle: dict[str, Any],
    *,
    episode_name: str,
    removed_turn: int | None = None,
    perturb_observation_turn: int | None = None,
) -> dict[str, Any]:
    if episode_name == "clean_episode":
        harness, _environment = _new_harness(bundle)
    else:
        harness = _prepare_repair_harness(bundle)
    turns = bundle[episode_name]["turn_history"]
    failures: list[dict[str, Any]] = []
    for turn in turns:
        if turn["turn_idx"] == removed_turn:
            continue
        result = _execute(harness, turn, f"{episode_name}_{removed_turn or 'full'}")
        if result.get("status") not in ACCEPTED_STATUSES:
            failures.append(
                {"turn_idx": turn["turn_idx"], "tool_name": turn["tool_name"], "status": result.get("status")}
            )
        if turn["turn_idx"] == perturb_observation_turn and result.get("status") == "executed":
            for evidence_id in result.get("created_evidence_ids", []):
                if evidence_id in harness.evidence_store and harness.evidence_store[evidence_id].active:
                    harness.inject_evidence_dropout(evidence_id)
    final_state = harness.render_state()
    expected = bundle[episode_name]["final_state"]
    same_final_result = _state_projection(final_state) == _state_projection(expected)
    return {
        "same_final_result": same_final_result,
        "terminal": final_state["terminal"],
        "open_high_priority_obligations": final_state["open_high_priority_obligations"],
        "replay_failures": failures,
    }


def _field_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not state["candidate_claims"]:
        return {}
    return state["candidate_claims"][0]["fields"]


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_obligations = set(before["open_high_priority_obligations"])
    after_obligations = set(after["open_high_priority_obligations"])
    before_evidence = set(before["available_evidence_ids"])
    after_evidence = set(after["available_evidence_ids"])
    before_fields = _field_map(before)
    after_fields = _field_map(after)
    transitions = []
    for field_name in sorted(set(before_fields).intersection(after_fields)):
        old = before_fields[field_name]
        new = after_fields[field_name]
        if (
            old["evidence_status"] != new["evidence_status"]
            or old["decision_status"] != new["decision_status"]
            or old["support_ids"] != new["support_ids"]
        ):
            transitions.append(
                {
                    "field": field_name,
                    "evidence_status": [old["evidence_status"], new["evidence_status"]],
                    "decision_status": [old["decision_status"], new["decision_status"]],
                    "support_ids": [old["support_ids"], new["support_ids"]],
                }
            )
    return {
        "evidence_added": sorted(after_evidence - before_evidence),
        "obligations_closed": sorted(before_obligations - after_obligations),
        "obligations_opened": sorted(after_obligations - before_obligations),
        "field_transitions": transitions,
        "claim_created": not before["candidate_claims"] and bool(after["candidate_claims"]),
        "terminal_changed": before["terminal"] != after["terminal"],
    }


def _reasoning_grounded(
    turn: dict[str, Any],
    before: dict[str, Any],
    tool_registry: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    reasoning = turn.get("reasoning", "")
    if not isinstance(reasoning, str) or len(reasoning.strip()) < 48:
        reasons.append("reasoning_too_short")
    referenced = set(EVIDENCE_ID_PATTERN.findall(reasoning))
    visible_ids = set(before["available_evidence_ids"]).union(before["dropped_evidence_ids"])
    if not referenced.issubset(visible_ids):
        reasons.append(f"references_nonvisible_evidence:{sorted(referenced - visible_ids)}")

    tool_name = turn["tool_name"]
    goals = {item["goal_id"]: item["status"] for item in before["case_goals"]}
    fields = _field_map(before)
    params = turn["params"]
    registry_entry = next((item for item in tool_registry if item.get("name") == tool_name), {})
    goal_effect = registry_entry.get(
        "satisfies_goal",
        "G-CONTEXT" if tool_name == "inspect_media" else "G-DIFFERENCE"
        if registry_entry.get("domain") == "external_media"
        else None,
    )
    if goal_effect == "G-CONTEXT" and goals.get("G-CONTEXT") == "satisfied":
        reasons.append("context_already_satisfied")
    elif goal_effect == "G-DIFFERENCE" and goals.get("G-DIFFERENCE") == "satisfied":
        reasons.append("difference_goal_already_satisfied")
    elif tool_name == "create_candidate_claim" and before["candidate_claims"]:
        reasons.append("claim_already_exists")
    elif tool_name == "bind_evidence":
        requested = set(params.get("bindings", {}))
        if not requested or any(name not in fields for name in requested):
            reasons.append("binding_has_no_open_known_field")
    elif tool_name == "set_field_decision":
        requested = params.get("fields", [])
        status = params.get("decision_status")
        if status == "open" and not any(
            name in fields
            and fields[name]["evidence_status"] != "supported"
            and fields[name]["decision_status"] == "committed"
            for name in requested
        ):
            reasons.append("reopen_does_not_target_invalidated_committed_field")
        if status == "committed" and any(
            name not in fields or fields[name]["evidence_status"] != "supported" for name in requested
        ):
            reasons.append("commit_targets_unsupported_field")
    elif tool_name == "stop" and before["open_high_priority_obligations"]:
        reasons.append("stop_with_open_obligations")
    return not reasons, reasons


def _initial_state(episode: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(episode["messages"][1]["content"])
    return payload["harness_state"]


def _downstream_consumed(
    turn: dict[str, Any],
    later_turns: list[dict[str, Any]],
    delta: dict[str, Any],
    final_state: dict[str, Any],
) -> bool:
    produced_ids = set(turn["observation"].get("created_evidence_ids", []))
    later_payload = json.dumps([item["params"] for item in later_turns], ensure_ascii=False)
    final_payload = json.dumps(final_state, ensure_ascii=False)
    if produced_ids and any(evidence_id in later_payload or evidence_id in final_payload for evidence_id in produced_ids):
        return True
    tool_name = turn.get("tool_name")
    params = turn.get("params", {})
    if tool_name == "assess_evidence_candidate":
        claim_id = params.get("claim_id")
        evidence_id = params.get("evidence_id")
        for later in later_turns:
            if later.get("tool_name") != "adjudicate_evidence":
                continue
            adjudication = later.get("params", {})
            if adjudication.get("claim_id") != claim_id:
                continue
            rejected_ids = {
                item.get("evidence_id")
                for item in adjudication.get("rejected_evidence", [])
                if isinstance(item, dict)
            }
            selected_ids = set(adjudication.get("selected_evidence_ids", []))
            if evidence_id in selected_ids or evidence_id in rejected_ids:
                return True
    if tool_name == "adjudicate_evidence":
        claim_id = params.get("claim_id")
        selected_ids = set(params.get("selected_evidence_ids", []))
        target_fields = set(params.get("target_fields", []))
        for later in later_turns:
            if later.get("tool_name") != "bind_evidence":
                continue
            binding = later.get("params", {})
            if binding.get("claim_id") != claim_id:
                continue
            bindings = binding.get("bindings", {})
            bound_target_ids = {
                evidence_id
                for field, evidence_ids in bindings.items()
                if field in target_fields
                for evidence_id in evidence_ids
            }
            if bound_target_ids and bound_target_ids.issubset(selected_ids):
                return True
    return bool(
        delta["obligations_closed"]
        or delta["field_transitions"]
        or delta["claim_created"]
        or delta["terminal_changed"]
    )


def _branch_enabling_measurement(
    turn: dict[str, Any],
    later_turns: list[dict[str, Any]],
    external_tools: set[str],
) -> bool:
    """Detect a real inconclusive observation that justifies another tool.

    Fixed-suffix deletion alone undervalues exploratory measurements: after
    deleting the observation it still replays the already-recorded next action,
    even though the policy would no longer have a visible reason to choose it.
    Adaptive environments therefore expose an explicit goal_satisfied=false
    routing state, and the audit requires a later external action to consume it.
    """
    observation = turn.get("observation", {}).get("observation", {})
    if observation.get("goal_satisfied") is not False:
        return False
    return any(item.get("tool_name") in external_tools for item in later_turns)


def _external_observation_uptake(
    turn: dict[str, Any],
    next_turn: dict[str, Any] | None,
    external_tools: set[str],
) -> dict[str, Any]:
    """Check that the next policy note visibly interprets a new EvidenceAtom."""

    created_ids = {
        str(value)
        for value in turn.get("observation", {}).get("created_evidence_ids", [])
    }
    if not created_ids:
        created_ids = {
            str(atom.get("evidence_id"))
            for atom in turn.get("observation", {}).get("evidence_atoms", [])
            if isinstance(atom, dict) and atom.get("evidence_id")
        }
    if not created_ids:
        return {
            "expected": False,
            "passed": True,
            "created_evidence_ids": [],
            "failures": [],
        }
    reasoning = str((next_turn or {}).get("reasoning", ""))
    lowered = reasoning.lower()
    named_new_atom = bool(created_ids.intersection(EVIDENCE_ID_PATTERN.findall(reasoning)))
    consequence_markers = (
        "show",
        "report",
        "indicat",
        "support",
        "observ",
        "measur",
        "detect",
        "localiz",
        "conflict",
        "mismatch",
        "negative",
        "inconclusive",
        "no material",
        "未",
        "显示",
        "表明",
        "支持",
        "冲突",
    )
    consequence_stated = any(marker in lowered for marker in consequence_markers)
    detail_tokens: set[str] = set()
    for atom in turn.get("observation", {}).get("evidence_atoms", []):
        if not isinstance(atom, dict):
            continue
        for key in (atom.get("observation") or {}):
            detail_tokens.update(
                token.lower()
                for token in OBSERVATION_TOKEN_PATTERN.findall(str(key).replace("_", " "))
                if len(token) >= 4 and token.lower() not in GENERIC_OBSERVATION_TOKENS
            )
    matched_detail_tokens = sorted(token for token in detail_tokens if token in lowered)
    semantic_detail_stated = bool(matched_detail_tokens)
    observation = turn.get("observation", {}).get("observation", {})
    negative_branch = observation.get("goal_satisfied") is False
    negative_update_passed = not negative_branch or bool(
        next_turn
        and next_turn.get("tool_name") in external_tools
        and any(
            marker in lowered
            for marker in (
                "remain",
                "unresolved",
                "negative",
                "inconclusive",
                "no material",
                "not support",
                "still need",
                "尚",
                "未",
                "不足",
            )
        )
    )
    components = {
        "next_turn_exists": next_turn is not None,
        "new_atom_named": named_new_atom,
        "semantic_consequence_stated": consequence_stated,
        "observation_detail_stated": semantic_detail_stated,
        "negative_measurement_changes_route": negative_update_passed,
    }
    return {
        "expected": True,
        "passed": all(components.values()),
        "created_evidence_ids": sorted(created_ids),
        "components": components,
        "matched_observation_detail_tokens": matched_detail_tokens,
        "negative_measurement": negative_branch,
        "failures": [name for name, passed in components.items() if not passed],
    }


def audit_step_utility(bundle: dict[str, Any]) -> dict[str, Any]:
    tool_registry = bundle.get("harness_config", {}).get("tool_registry", [])
    episode_reports: dict[str, list[dict[str, Any]]] = {}
    for short_name, episode_name in (("clean", "clean_episode"), ("repair", "repair_episode")):
        episode = bundle[episode_name]
        before = _initial_state(episode)
        reports = []
        for index, turn in enumerate(episode["turn_history"]):
            after = turn["observation"].get("harness_state", before)
            delta = _state_delta(before, after)
            grounded, grounding_failures = _reasoning_grounded(turn, before, tool_registry)
            removal = _replay_counterfactual(
                bundle,
                episode_name=episode_name,
                removed_turn=turn["turn_idx"],
            )
            observation_test = None
            external_tools = {
                item["name"]
                for item in bundle.get("harness_config", {}).get("tool_registry", [])
                if item.get("domain") == "external_media"
            } or {"inspect_media", "align_videos_dense"}
            if turn["tool_name"] in external_tools:
                observation_test = _replay_counterfactual(
                    bundle,
                    episode_name=episode_name,
                    perturb_observation_turn=turn["turn_idx"],
                )
            accepted = turn["observation"].get("status") in ACCEPTED_STATUSES
            later_turns = episode["turn_history"][index + 1 :]
            branch_enabling = _branch_enabling_measurement(turn, later_turns, external_tools)
            observation_uptake = (
                _external_observation_uptake(
                    turn,
                    later_turns[0] if later_turns else None,
                    external_tools,
                )
                if turn["tool_name"] in external_tools
                else {"expected": False, "passed": True, "failures": []}
            )
            learning_value = learning_value_profile(
                turn["tool_name"],
                params=turn.get("params", {}),
                external_measurement=turn["tool_name"] in external_tools,
                evidence_ids_in_reasoning=len(
                    EVIDENCE_ID_PATTERN.findall(str(turn.get("reasoning", "")))
                ),
                branch_enabling_measurement=branch_enabling,
            )
            consumed = _downstream_consumed(
                turn,
                later_turns,
                delta,
                episode["final_state"],
            ) or branch_enabling
            causally_necessary = not removal["same_final_result"] or branch_enabling
            observation_sensitive = (
                observation_test is None
                or not observation_test["same_final_result"]
                or branch_enabling
            )
            eligible = (
                accepted
                and grounded
                and consumed
                and causally_necessary
                and observation_sensitive
            )
            verdict = (
                "indispensable"
                if eligible
                else "invalid"
                if not accepted or not grounded
                else "redundant"
                if removal["same_final_result"]
                else "unused"
                if not consumed
                else "observation_insensitive"
            )
            reports.append(
                {
                    "turn_idx": turn["turn_idx"],
                    "tool_name": turn["tool_name"],
                    "execution_status": turn["observation"].get("status"),
                    "reasoning_grounded": grounded,
                    "reasoning_failures": grounding_failures,
                    "state_delta": delta,
                    "downstream_consumed": consumed,
                    "branch_enabling_measurement": branch_enabling,
                    "learning_value": learning_value,
                    "observation_to_belief_update": observation_uptake,
                    "decision_role": (
                        "branch_enabling_measurement"
                        if branch_enabling
                        else "terminal_evidence_or_state_transition"
                    ),
                    "removal_counterfactual": removal,
                    "observation_counterfactual": observation_test,
                    "causally_necessary": causally_necessary,
                    "observation_sensitive": observation_sensitive,
                    "verdict": verdict,
                    "accepted_for_sft": eligible,
                }
            )
            before = after
        episode_reports[short_name] = reports

    all_steps = episode_reports["clean"] + episode_reports["repair"]
    exported_steps = [item for item in all_steps if item["execution_status"] in ACCEPTED_STATUSES]
    reopen_steps = [
        item
        for item in episode_reports["repair"]
        if item["tool_name"] == "set_field_decision"
        and bundle["repair_episode"]["turn_history"][item["turn_idx"] - 1]["params"].get("decision_status") == "open"
    ]
    repair_operator = bundle.get("execution_summary", {}).get("repair_operator", "evidence_dropout")
    checks = {
        "rejected_actions_are_context_only": all(
            item["execution_status"] in ACCEPTED_STATUSES
            for short_name, key in (("clean", "clean_turn_sft"), ("repair", "repair_turn_sft"))
            for datum in bundle[key]
            for item in [episode_reports[short_name][datum["metadata"]["source_turn_index"] - 1]]
        ),
        "all_exported_actions_executed_or_accepted": all(
            item["execution_status"] in ACCEPTED_STATUSES for item in exported_steps
        ),
        "all_reasoning_grounded_in_visible_state": all(item["reasoning_grounded"] for item in exported_steps),
        "all_steps_consumed_or_change_required_state": all(item["downstream_consumed"] for item in exported_steps),
        "all_steps_fail_fixed_suffix_when_removed": all(item["causally_necessary"] for item in exported_steps),
        "external_actions_are_observation_sensitive": all(
            item["observation_sensitive"] for item in exported_steps
        ),
        "repair_reopen_is_causally_required": (
            repair_operator == "decision_reopen"
            or (bool(reopen_steps) and all(item["causally_necessary"] for item in reopen_steps))
        ),
        "all_exported_steps_eligible_for_sft": all(item["accepted_for_sft"] for item in exported_steps),
    }
    return {
        "audit_id": "STEP-UTILITY-CAUSAL-REPLAY-V1",
        "method": (
            "recorded-observation fixed-suffix deletion replay plus external-observation dropout "
            "and explicit adaptive-routing dependency"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "external_observations_receive_visible_belief_updates": all(
                item["observation_to_belief_update"]["passed"]
                for item in exported_steps
                if item["tool_name"] in {
                    entry["name"]
                    for entry in tool_registry
                    if entry.get("domain") == "external_media"
                }
            ),
            "belief_update_failures": sum(
                not item["observation_to_belief_update"]["passed"]
                for item in exported_steps
                if item["tool_name"] in {
                    entry["name"]
                    for entry in tool_registry
                    if entry.get("domain") == "external_media"
                }
            ),
            "legacy_admission_unchanged": True,
        },
        "episodes": episode_reports,
        "summary": {
            "total_steps": len(all_steps),
            "context_only_rejected_steps": len(all_steps) - len(exported_steps),
            "exported_candidate_steps": len(exported_steps),
            "accepted_for_sft": sum(item["accepted_for_sft"] for item in all_steps),
            "verdict_counts": {
                verdict: sum(item["verdict"] == verdict for item in all_steps)
                for verdict in sorted({item["verdict"] for item in all_steps})
            },
        },
    }


def apply_step_utility_filter(bundle: dict[str, Any], report: dict[str, Any]) -> None:
    bundle["step_utility_report"] = report
    for short_name, key in (("clean", "clean_turn_sft"), ("repair", "repair_turn_sft")):
        by_turn = {item["turn_idx"]: item for item in report["episodes"][short_name]}
        kept = []
        for datum in bundle[key]:
            source_turn = datum["metadata"]["source_turn_index"]
            audit = by_turn[source_turn]
            datum["metadata"]["step_utility_verdict"] = audit["verdict"]
            datum["metadata"]["counterfactual_removal_changes_outcome"] = audit["causally_necessary"]
            datum["metadata"]["reasoning_grounded"] = audit["reasoning_grounded"]
            datum["metadata"]["decision_role"] = audit["decision_role"]
            datum["metadata"]["branch_enabling_measurement"] = audit["branch_enabling_measurement"]
            datum["metadata"].update(audit["learning_value"])
            datum["metadata"]["observation_to_belief_update"] = copy.deepcopy(
                audit["observation_to_belief_update"]
            )
            if audit["accepted_for_sft"]:
                kept.append(datum)
        bundle[key] = kept
