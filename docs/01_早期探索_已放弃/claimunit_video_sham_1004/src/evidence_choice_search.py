"""Label-blind branch execution for evidence-choice trajectory generation.

The selector previews independently proposed semantic tool actions in isolated
copies of the same pre-action checkpoint.  Preview observations never enter the
student transcript and never choose the action.  The action is frozen from
pre-action method fit and reasoning; previews only audit execution and evidence
integrity.  No oracle, verifier label, target answer, or favorable future
outcome participates in routing.

This module is deliberately separate from the rollout loop: a batch must record
whether proposals covered at least two distinct tools and whether replaying the
selected action on the committed branch reproduced the preview.  Missing
coverage is reported as missing evidence of autonomy, not silently upgraded to
a successful tree search.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from causal_trajectory_generation import (
    CausalPolicyViewTeacher,
    control_plane_exposure_audit,
    is_semantic_measurement_tool,
)
from llm_teacher_rollout import TeacherAction, _normalize_harness_params


IDENTIFIER_KEYS = {
    "observation_id",
    "execution_id",
    "evidence_id",
    "created_evidence_ids",
    "call_id",
    "tool_call_id",
    "generated_by",
    "harness_state",
}


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def _without_identifiers(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return {
            str(key): _without_identifiers(child)
            for key, child in value.items()
            if key not in IDENTIFIER_KEYS
        }
    if isinstance(value, list):
        return [_without_identifiers(child) for child in value]
    return value


def _nonempty_semantic_leaf_count(value: Any) -> int:
    value = _plain(value)
    if isinstance(value, dict):
        return sum(_nonempty_semantic_leaf_count(child) for child in value.values())
    if isinstance(value, list):
        return sum(_nonempty_semantic_leaf_count(child) for child in value)
    return int(value not in (None, "", "[SEMANTIC CONTENT HIDDEN]"))


def _atoms(result: dict[str, Any]) -> list[dict[str, Any]]:
    atoms = result.get("evidence_atoms") or []
    return [_plain(atom) for atom in atoms if isinstance(_plain(atom), dict)]


def _open_obligations(state: dict[str, Any]) -> set[str]:
    return {str(item) for item in state.get("open_high_priority_obligations") or []}


def label_blind_branch_score(
    *,
    before_state: dict[str, Any],
    result: dict[str, Any],
    target_fields: Iterable[str],
) -> dict[str, Any]:
    """Score evidence utility without consulting an answer or verifier."""

    executed = result.get("status") != "rejected"
    atoms = _atoms(result)
    target = set(target_fields)
    supported = {
        str(field)
        for atom in atoms
        for field in atom.get("supports_fields") or []
        if str(field) in target
    }
    atom_integrity = bool(atoms) and all(
        atom.get("evidence_id")
        and isinstance(atom.get("supports_fields"), list)
        and isinstance(atom.get("observation"), dict)
        and atom.get("observation")
        and isinstance(atom.get("provenance"), dict)
        and atom.get("provenance", {}).get("method")
        for atom in atoms
    )
    semantic_payload = result.get("result", result.get("observation", {}))
    semantic_leaves = _nonempty_semantic_leaf_count(semantic_payload)
    after_state = result.get("harness_state") or {}
    obligations_closed = len(
        _open_obligations(before_state) - _open_obligations(after_state)
    )
    negative_information = bool(atoms) and not supported and any(
        str(atom.get("kind") or "").startswith("negative_")
        or atom.get("observation", {}).get("material_difference_detected") is False
        for atom in atoms
    )
    target_coverage = len(supported) / max(1, len(target))
    components = {
        "executed": 1.0 if executed else 0.0,
        "evidence_created": 1.0 if atoms else 0.0,
        "semantic_observation": min(1.0, semantic_leaves / 3.0),
        "atom_integrity": 1.0 if atom_integrity else 0.0,
        "target_field_coverage": target_coverage,
        "obligation_progress": min(1.0, float(obligations_closed)),
        # A well-formed negative measurement is useful evidence rather than an
        # interface failure; it receives explicit credit without pretending to
        # satisfy the target fields.
        "negative_information": 1.0 if negative_information else 0.0,
    }
    weights = {
        "executed": 0.15,
        "evidence_created": 0.15,
        "semantic_observation": 0.15,
        "atom_integrity": 0.15,
        "target_field_coverage": 0.20,
        "obligation_progress": 0.10,
        "negative_information": 0.10,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    return {
        "score": round(score, 6),
        "components": {name: round(value, 6) for name, value in components.items()},
        "weights": weights,
        "supported_target_fields": sorted(supported),
        "evidence_atom_count": len(atoms),
        "semantic_leaf_count": semantic_leaves,
        "negative_measurement_is_information": negative_information,
        "oracle_or_verifier_used": False,
    }


@dataclass
class ExecutedBranch:
    action: TeacherAction
    normalized_params: dict[str, Any]
    result: dict[str, Any]
    utility: dict[str, Any]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "action": asdict(self.action),
            "normalized_params": copy.deepcopy(self.normalized_params),
            "result": copy.deepcopy(self.result),
            "utility": copy.deepcopy(self.utility),
        }


def execute_isolated_branches(
    harness: Any, actions: list[TeacherAction]
) -> list[ExecutedBranch]:
    """Execute unique semantic proposals without mutating the committed branch."""

    branches: list[ExecutedBranch] = []
    seen: set[tuple[str, str]] = set()
    before_state = harness.render_state()
    for action in actions:
        if not is_semantic_measurement_tool(action.tool_name):
            continue
        params = _normalize_harness_params(action.tool_name, action.params)
        signature = (
            action.tool_name,
            json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        branch_environment = copy.deepcopy(harness.environment)
        branch = harness.clone_state(environment=branch_environment)
        branch.record_policy_call(
            action.tool_name, params, action.call_id, action.reasoning
        )
        try:
            result = branch.execute_tool(action.tool_name, params, action.call_id)
        except (ValueError, RuntimeError) as error:
            result = {
                "status": "rejected",
                "error_type": type(error).__name__,
                "error": str(error),
                "harness_state": branch.render_state(),
            }
        branches.append(
            ExecutedBranch(
                action=copy.deepcopy(action),
                normalized_params=params,
                result=copy.deepcopy(result),
                utility=label_blind_branch_score(
                    before_state=before_state,
                    result=result,
                    target_fields=harness.target_fields,
                ),
            )
        )
    return branches


def select_evidence_branch(
    *,
    harness: Any,
    actions: list[TeacherAction],
    visible_semantic_tools: list[str],
    proposal_method_fit_by_call_id: dict[str, float] | None = None,
    proposal_method_fit_ranking_by_call_id: dict[str, float] | None = None,
    proposal_scores_by_call_id: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Freeze a choice from pre-action evidence, then verify branch execution.

    Preview outcomes are intentionally unavailable to the ranking function.
    Their only role is to establish whether the already selected action can
    produce a well-formed executable observation and EvidenceAtom.
    """

    unique_actions: list[TeacherAction] = []
    seen: set[tuple[str, str]] = set()
    for action in actions:
        if not is_semantic_measurement_tool(action.tool_name):
            continue
        params = _normalize_harness_params(action.tool_name, action.params)
        signature = (
            action.tool_name,
            json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique_actions.append(action)

    proposed_tools = sorted({action.tool_name for action in unique_actions})
    visible = sorted(set(visible_semantic_tools))
    coverage_passed = len(visible) >= 2 and len(proposed_tools) >= 2
    method_fit_scores = proposal_method_fit_by_call_id or {}
    method_fit_ranking = proposal_method_fit_ranking_by_call_id or method_fit_scores
    proposal_scores = proposal_scores_by_call_id or {}
    preaction_ranked = sorted(
        enumerate(unique_actions),
        key=lambda item: (
            -float(method_fit_ranking.get(item[1].call_id, 0.0)),
            -float(proposal_scores.get(item[1].call_id, 0.0)),
            item[0],
        ),
    )
    planned_action = preaction_ranked[0][1] if preaction_ranked else None
    top_pair = (
        (
            float(method_fit_ranking.get(planned_action.call_id, 0.0)),
            float(proposal_scores.get(planned_action.call_id, 0.0)),
        )
        if planned_action is not None
        else None
    )
    top_pair_count = sum(
        (
            float(method_fit_ranking.get(action.call_id, 0.0)),
            float(proposal_scores.get(action.call_id, 0.0)),
        )
        == top_pair
        for action in unique_actions
    )
    # The choice is now frozen.  Only after this point may previews execute.
    branches = execute_isolated_branches(harness, unique_actions)
    selected = next(
        (
            branch
            for branch in branches
            if planned_action is not None
            and branch.action.call_id == planned_action.call_id
        ),
        None,
    )
    selected_components = (selected.utility.get("components") or {}) if selected else {}
    selected_admissible = bool(
        selected is not None
        and selected_components.get("executed") == 1.0
        and selected_components.get("evidence_created") == 1.0
        and float(selected_components.get("semantic_observation", 0.0)) > 0.0
        and selected_components.get("atom_integrity") == 1.0
    )
    selected_outcome_utility = (
        float(selected.utility["score"]) if selected is not None else None
    )
    best_outcome_utility = max(
        (float(branch.utility["score"]) for branch in branches), default=None
    )
    return {
        "selected": selected,
        "audit": {
            "search_method": "preaction_frozen_choice_with_isolated_execution_audit_v2",
            "visible_semantic_tools": visible,
            "proposed_semantic_tools": proposed_tools,
            "visible_choice_count": len(visible),
            "proposed_choice_count": len(proposed_tools),
            "proposal_coverage_passed": coverage_passed,
            "proposal_preaction_reasoning_scores": {
                branch.action.call_id: float(
                    proposal_scores.get(branch.action.call_id, 0.0)
                )
                for branch in branches
            },
            "proposal_preaction_method_fit_scores": {
                branch.action.call_id: float(
                    method_fit_scores.get(branch.action.call_id, 0.0)
                )
                for branch in branches
            },
            "proposal_preaction_method_fit_ranking_scores": {
                branch.action.call_id: float(
                    method_fit_ranking.get(branch.action.call_id, 0.0)
                )
                for branch in branches
            },
            "selected_preaction_method_fit_score": (
                float(method_fit_scores.get(selected.action.call_id, 0.0))
                if selected is not None
                else None
            ),
            "selected_preaction_method_fit_ranking_score": (
                float(method_fit_ranking.get(selected.action.call_id, 0.0))
                if selected is not None
                else None
            ),
            "selection_ambiguous": top_pair_count > 1,
            "top_rank_tie_count": top_pair_count,
            "selection_frozen_before_branch_execution": True,
            "hindsight_outcome_used_for_selection": False,
            "selected_branch_execution_admissible": selected_admissible,
            "selected_outcome_utility": selected_outcome_utility,
            "best_preview_outcome_utility": best_outcome_utility,
            "higher_utility_branch_was_not_allowed_to_reroute_selection": bool(
                selected_outcome_utility is not None
                and best_outcome_utility is not None
                and best_outcome_utility > selected_outcome_utility
            ),
            "selection_basis": "preaction_method_fit_then_reasoning_preview_is_audit_only",
            "oracle_or_verifier_used": False,
            "preview_observations_student_visible": False,
            "branches": [branch.audit_dict() for branch in branches],
        },
    }


def commit_selected_branch(harness: Any, branch: ExecutedBranch) -> dict[str, Any]:
    """Replay only the chosen action on the committed branch and compare semantics."""

    action = branch.action
    harness.record_policy_call(
        action.tool_name,
        branch.normalized_params,
        action.call_id,
        action.reasoning,
    )
    result = harness.execute_tool(
        action.tool_name, branch.normalized_params, action.call_id
    )
    preview = _without_identifiers(branch.result)
    committed = _without_identifiers(result)
    return {
        "result": result,
        "deterministic_replay_match": preview == committed,
        "preview_semantic_payload": preview,
        "committed_semantic_payload": committed,
    }


def _usage_total(actions: list[TeacherAction]) -> dict[str, int]:
    keys = {key for action in actions for key in (action.usage or {})}
    return {
        key: sum(
            int(action.usage.get(key, 0))
            for action in actions
            if isinstance(action.usage.get(key, 0), int)
        )
        for key in keys
    }


class EvidenceChoiceProposalRejected(RuntimeError):
    """All bounded pre-action proposal rounds failed the strict contract."""

    def __init__(self, message: str, *, failure_artifact: dict[str, Any]) -> None:
        super().__init__(message)
        self.failure_artifact = failure_artifact


class EvidenceChoiceSearchTeacher:
    """Branch-execute-select wrapper for a reduced-control teacher.

    Every proposal is sampled from the same full pre-action tool surface.  The
    wrapper never forces a tool by presenting a singleton schema.  When the
    delegate fails to propose two distinct tools, the turn remains executable
    but its audit explicitly fails proposal coverage.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        branch_proposals_per_turn: int = 4,
        state_proposals_per_turn: int = 3,
        proposal_rounds_per_turn: int = 2,
        state_min_reasoning_score: float = 0.875,
    ) -> None:
        if branch_proposals_per_turn < 2:
            raise ValueError("branch_proposals_per_turn must be at least two")
        if state_proposals_per_turn < 1:
            raise ValueError("state_proposals_per_turn must be positive")
        if proposal_rounds_per_turn < 1:
            raise ValueError("proposal_rounds_per_turn must be positive")
        if not 0.0 <= state_min_reasoning_score <= 1.0:
            raise ValueError("state_min_reasoning_score must be between zero and one")
        self.delegate = delegate
        self.branch_proposals_per_turn = branch_proposals_per_turn
        self.state_proposals_per_turn = state_proposals_per_turn
        self.proposal_rounds_per_turn = proposal_rounds_per_turn
        self.state_min_reasoning_score = state_min_reasoning_score
        self.model = getattr(delegate, "model", "unknown_teacher")
        self.source = f"{getattr(delegate, 'source', 'unknown_teacher')}+evidence_choice_search"
        self.policy_view = "causal_reduced_control_evidence_choice_tree_v1"
        self.audit_log: list[dict[str, Any]] = []
        self.harness: Any | None = None
        self._pending_branch: ExecutedBranch | None = None

    def bind_harness(self, harness: Any) -> None:
        self.harness = harness
        self._pending_branch = None

    def _delegate_audit(self, before: int) -> dict[str, Any]:
        rows = getattr(self.delegate, "audit_log", [])[before:]
        return copy.deepcopy(rows[-1]) if rows else {}

    def next_action(
        self, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> TeacherAction:
        if self.harness is None:
            raise RuntimeError("EvidenceChoiceSearchTeacher must be bound to a harness")
        semantic_tools = sorted(
            str(tool.get("name"))
            for tool in tools
            if is_semantic_measurement_tool(str(tool.get("name") or ""))
        )
        if len(semantic_tools) < 2:
            visible_names = {str(tool.get("name")) for tool in tools}
            all_actions: list[TeacherAction] = []
            round_records: list[dict[str, Any]] = []
            prior_failures: list[str] = []
            for round_index in range(1, self.proposal_rounds_per_turn + 1):
                proposals: list[tuple[TeacherAction, dict[str, Any]]] = []
                for _ in range(self.state_proposals_per_turn):
                    before = len(getattr(self.delegate, "audit_log", []))
                    proposal_input = copy.deepcopy(input_items)
                    if (
                        round_index > 1
                        and proposal_input
                        and proposal_input[0].get("role") == "system"
                    ):
                        proposal_input[0]["content"] = (
                            str(proposal_input[0].get("content") or "")
                            + " A prior state-action proposal set failed pre-action reasoning checks ("
                            + ", ".join(sorted(set(prior_failures)))
                            + "). Re-decide from the unchanged visible checkpoint; no tool outcome was executed or revealed."
                        )
                    candidate = self.delegate.next_action(
                        proposal_input, copy.deepcopy(tools)
                    )
                    all_actions.append(candidate)
                    if candidate.tool_name in visible_names:
                        proposals.append((candidate, self._delegate_audit(before)))
                if not proposals:
                    failures = ["no_legal_state_action_proposal"]
                    round_records.append(
                        {"proposal_round": round_index, "rejection_reasons": failures}
                    )
                    prior_failures = failures
                    continue
                action, audit = max(
                    proposals,
                    key=lambda item: float(item[1].get("selected_score", 0.0)),
                )
                exposure = audit.get("control_plane_exposure") or control_plane_exposure_audit(
                    input_items, tools
                )
                failures = list(audit.get("selected_failures") or [])
                if float(audit.get("selected_score", 0.0)) < self.state_min_reasoning_score:
                    failures.append("state_reasoning_score_below_minimum")
                if any(
                    exposure.get(key) is True
                    for key in (
                        "compiled_semantic_actions_visible",
                        "repair_recipe_visible",
                        "full_target_call_materializable",
                        "categorical_router_answer_visible",
                    )
                ):
                    failures.append("answer_like_control_visible")
                failures = sorted(set(failures))
                round_records.append(
                    {
                        "proposal_round": round_index,
                        "rejection_reasons": failures,
                        "candidate_count": len(proposals),
                        "selected_score": float(audit.get("selected_score", 0.0)),
                    }
                )
                if failures:
                    prior_failures = failures
                    continue
                action = copy.deepcopy(action)
                action.usage = _usage_total(all_actions)
                audit.setdefault("control_plane_exposure", exposure)
                audit.setdefault("selected_action", asdict(action))
                audit["evidence_choice_search"] = {
                    "applicable": False,
                    "reason": "fewer_than_two_visible_semantic_tools",
                    "state_reasoning_candidates": len(proposals),
                    "state_reasoning_contract_passed": True,
                    "proposal_retry_outcome_feedback_used": False,
                    "proposal_rounds_attempted": round_index,
                    "prior_rejected_proposal_rounds": copy.deepcopy(
                        round_records[:-1]
                    ),
                }
                self.audit_log.append(audit)
                self._pending_branch = None
                return action
            reason_counts: dict[str, int] = {}
            for record in round_records:
                for reason in record["rejection_reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            raise EvidenceChoiceProposalRejected(
                "state-action proposal rounds exhausted without a high-quality action",
                failure_artifact={
                    "schema_version": "evidence_choice_proposal_failure_v1",
                    "failure_kind": "bounded_state_proposal_exhaustion",
                    "proposal_round_budget": self.proposal_rounds_per_turn,
                    "proposals_per_round": self.state_proposals_per_turn,
                    "visible_tools": sorted(visible_names),
                    "reason_counts": reason_counts,
                    "rounds": round_records,
                    "oracle_or_verifier_used": False,
                    "tool_outcome_feedback_sent_to_teacher": False,
                },
            )

        visible_names = {str(tool.get("name")) for tool in tools}
        round_records: list[dict[str, Any]] = []
        all_actions: list[TeacherAction] = []
        prior_preaction_reasons: list[str] = []
        for round_index in range(1, self.proposal_rounds_per_turn + 1):
            actions: list[TeacherAction] = []
            action_audits: dict[str, dict[str, Any]] = {}
            previously_proposed: list[str] = []
            for proposal_index in range(self.branch_proposals_per_turn):
                before = len(getattr(self.delegate, "audit_log", []))
                proposal_input = copy.deepcopy(input_items)
                if proposal_input and proposal_input[0].get("role") == "system":
                    reminder = ""
                    if round_index > 1:
                        reminder += (
                            " A prior proposal set failed only pre-action quality checks ("
                            + ", ".join(sorted(set(prior_preaction_reasons)))
                            + "). Re-evaluate from the unchanged visible checkpoint; no prior preview outcome is available."
                        )
                    if proposal_index:
                        reminder += (
                            " Generate an independent evidence-action proposal from the full visible tool set. "
                            "Previously proposed tools in this round were: "
                            + ", ".join(sorted(set(previously_proposed)))
                            + ". Consider a different measurement only when the original pre-action evidence justifies it."
                        )
                    proposal_input[0]["content"] = (
                        str(proposal_input[0].get("content") or "") + reminder
                    )
                action = self.delegate.next_action(
                    proposal_input, copy.deepcopy(tools)
                )
                if action.tool_name not in visible_names:
                    continue
                actions.append(action)
                all_actions.append(action)
                previously_proposed.append(action.tool_name)
                action_audits[action.call_id] = self._delegate_audit(before)
            selection = select_evidence_branch(
                harness=self.harness,
                actions=actions,
                visible_semantic_tools=semantic_tools,
                proposal_method_fit_by_call_id={
                    call_id: float(audit.get("selected_method_fit_score", 0.0))
                    for call_id, audit in action_audits.items()
                },
                proposal_method_fit_ranking_by_call_id={
                    call_id: float(
                        audit.get(
                            "selected_method_fit_ranking_score",
                            audit.get("selected_method_fit_score", 0.0),
                        )
                    )
                    for call_id, audit in action_audits.items()
                },
                proposal_scores_by_call_id={
                    call_id: float(audit.get("selected_score", 0.0))
                    for call_id, audit in action_audits.items()
                },
            )
            selected = selection["selected"]
            selection_audit = selection["audit"]
            selected_delegate_audit = (
                action_audits.get(selected.action.call_id, {})
                if selected is not None
                else {}
            )
            selected_method_fit_audit = (
                selected_delegate_audit.get("selected_method_fit") or {}
            )
            selection_audit["selected_counterfactual_tool_swap_passed"] = (
                selected_method_fit_audit.get("counterfactual_tool_swap_passed")
            )
            selection_audit["selected_directional_contrast"] = copy.deepcopy(
                selected_method_fit_audit.get("directional_contrast")
            )
            preaction_reasons = []
            if selection_audit.get("proposal_coverage_passed") is not True:
                preaction_reasons.append("proposal_coverage_failed")
            if selection_audit.get("selection_ambiguous") is not False:
                preaction_reasons.append("preaction_selection_ambiguous")
            if float(selection_audit.get("selected_preaction_method_fit_score", 0.0)) < 1.0:
                preaction_reasons.append("prospective_method_fit_incomplete")
            if selection_audit.get("selected_counterfactual_tool_swap_passed") is not True:
                preaction_reasons.append("counterfactual_tool_swap_specificity_failed")
            if (
                selected_delegate_audit.get("latest_observation_uptake_expected") is True
                and selected_delegate_audit.get("latest_observation_uptake_passed") is not True
            ):
                preaction_reasons.append("latest_observation_uptake_failed")
            execution_reasons = []
            if not preaction_reasons and (
                selected is None
                or selection_audit.get("selected_branch_execution_admissible") is not True
            ):
                execution_reasons.append("selected_branch_execution_inadmissible")
            round_records.append(
                {
                    "proposal_round": round_index,
                    "preaction_rejection_reasons": preaction_reasons,
                    "execution_rejection_reasons": execution_reasons,
                    "selection_audit": copy.deepcopy(selection_audit),
                }
            )
            if preaction_reasons:
                prior_preaction_reasons = preaction_reasons
                continue
            if execution_reasons:
                break
            assert selected is not None
            self._pending_branch = selected
            action = copy.deepcopy(selected.action)
            action.usage = _usage_total(all_actions)
            audit = action_audits.get(selected.action.call_id, {})
            audit.setdefault(
                "control_plane_exposure",
                control_plane_exposure_audit(input_items, tools),
            )
            audit["selected_action"] = asdict(action)
            audit["evidence_choice_search"] = {
                "applicable": True,
                **copy.deepcopy(selection_audit),
                "selected_tool": action.tool_name,
                "selected_preview_utility": copy.deepcopy(selected.utility),
                "proposal_rounds_attempted": round_index,
                "proposal_retry_outcome_feedback_used": False,
                "prior_rejected_proposal_rounds": copy.deepcopy(round_records[:-1]),
                "committed_replay_checked": False,
            }
            self.audit_log.append(audit)
            return action

        reason_counts: dict[str, int] = {}
        for record in round_records:
            for reason in (
                record["preaction_rejection_reasons"]
                + record["execution_rejection_reasons"]
            ):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        raise EvidenceChoiceProposalRejected(
            "evidence-choice proposal rounds exhausted without a commit-eligible action",
            failure_artifact={
                "schema_version": "evidence_choice_proposal_failure_v1",
                "failure_kind": "bounded_checkpoint_proposal_exhaustion",
                "proposal_round_budget": self.proposal_rounds_per_turn,
                "proposals_per_round": self.branch_proposals_per_turn,
                "visible_semantic_tools": semantic_tools,
                "reason_counts": reason_counts,
                "rounds": round_records,
                "oracle_or_verifier_used": False,
                "preview_outcome_feedback_sent_to_teacher": False,
            },
        )

    def record_committed_result(
        self, action: TeacherAction, result: dict[str, Any]
    ) -> None:
        if self._pending_branch is None:
            return
        if action.call_id != self._pending_branch.action.call_id:
            raise RuntimeError("committed action does not match selected evidence branch")
        preview = _without_identifiers(self._pending_branch.result)
        committed = _without_identifiers(result)
        search = self.audit_log[-1]["evidence_choice_search"]
        search["committed_replay_checked"] = True
        search["deterministic_replay_match"] = preview == committed
        search["committed_result_status"] = result.get("status", "executed")
        self._pending_branch = None


def build_evidence_choice_teacher(
    base_teacher: Any,
    *,
    candidates_per_turn: int = 3,
    proposal_rounds_per_turn: int = 2,
) -> EvidenceChoiceSearchTeacher:
    """Construct the default generator with bounded strict pre-action recovery."""

    if candidates_per_turn < 2:
        raise ValueError("evidence-choice generation requires at least two candidates per turn")
    reduced = CausalPolicyViewTeacher(base_teacher, candidates_per_turn=1)
    return EvidenceChoiceSearchTeacher(
        reduced,
        branch_proposals_per_turn=candidates_per_turn,
        state_proposals_per_turn=candidates_per_turn,
        proposal_rounds_per_turn=proposal_rounds_per_turn,
    )
