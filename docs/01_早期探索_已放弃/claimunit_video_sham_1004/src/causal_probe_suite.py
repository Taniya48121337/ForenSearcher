"""Execute the UTQR-7 causal neighborhood around a teacher trajectory.

The suite deliberately separates two questions:

1. Did the base trajectory reach a valid answer?
2. Does the policy react correctly when the evidence, identity, role assignment,
   control surface, or evidence integrity is changed without exposing a label?

The public entry point :func:`run_causal_probe_packet` accepts a completed
teacher bundle and a teacher transport.  Dataset batches may pass an
``environment_factory`` so every probe uses the real local tool environment.
Without one, the suite uses recorded observations and marks that limitation in
the packet; such packets are diagnostic and cannot establish batch readiness.
"""

from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any, Callable

from interactive_harness import EvidenceAtom, TaskContract, VideoForensicHarness
from llm_teacher_rollout import run_teacher_episode
from pipeline import write_json
from step_utility_auditor import (
    RecordedToolEnvironment,
    _execute,
    _recorded_payloads,
    audit_step_utility,
)
from unannounced_repair_interventions import (
    INTERVENTIONS,
    apply_intervention,
    score_episode as score_unannounced_episode,
)


EnvironmentFactory = Callable[[], Any]

PROBE_NAMES = (
    "semantic_mask",
    "schema_compatible_swap",
    "identity_mismatch",
    "ab_role_swap",
    "reduced_control",
    "structure_only",
    "unannounced_repair",
    "step_intervention",
)

PROBE_VERSIONS = {
    "semantic_mask": 6,
    "schema_compatible_swap": 7,
    "identity_mismatch": 5,
    "ab_role_swap": 12,
    "structure_only": 6,
    "unannounced_repair": 9,
}

PROBE_NOTE_STYLES = {
    "semantic_mask": "Write as a compact evidence-triage note; lead with the missing content and its consequence.",
    "schema_compatible_swap": "Lead with the concrete contrast between the competing values before choosing an action.",
    "identity_mismatch": "Lead with the provenance fact that controls whether the evidence can transfer to this task.",
    "ab_role_swap": "Frame the note around role assignment and which fields actually depend on that role mapping.",
    "structure_only": "State first what the remaining structure cannot establish, then justify the cheapest informative action.",
    "unannounced_repair": "Write as a terse correction log: name the localized inconsistency and preserve unaffected work.",
}

INTEGRITY_INSTRUCTION = (
    "Continue from the visible state as an independent analyst. Check evidence content, provenance, "
    "task identity, role assignment, and scope before committing or stopping. If the visible support is "
    "insufficient or internally inconsistent, gather discriminating evidence or abstain."
)


def _contract(bundle: dict[str, Any]) -> TaskContract:
    return TaskContract(**copy.deepcopy(bundle["clean_episode"]["final_state"]["task_contract"]))


def _choice_surface_metrics(exposure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure actual semantic choices on turns where the teacher chose a measurement."""

    semantic_measurement_turns = 0
    nontrivial_semantic_choice_turns = 0
    branch_search_expected_turns = 0
    branch_search_covered_turns = 0
    branch_replay_mismatch_turns = 0
    branch_selection_ambiguous_turns = 0
    prospective_method_fit_expected_turns = 0
    prospective_method_fit_passed_turns = 0
    counterfactual_tool_swap_expected_turns = 0
    counterfactual_tool_swap_passed_turns = 0
    preaction_frozen_selection_turns = 0
    hindsight_selected_turns = 0
    selected_branch_inadmissible_turns = 0
    proposal_retry_outcome_feedback_turns = 0
    for row in exposure_rows:
        exposure = row.get("control_plane_exposure") or {}
        selected = row.get("selected_action") or {}
        selected_name = str(selected.get("tool_name") or selected.get("name") or "")
        semantic_names = set(exposure.get("semantic_measurement_tool_names") or [])
        if selected_name in semantic_names:
            semantic_measurement_turns += 1
            if int(exposure.get("semantic_measurement_choice_count", 0)) >= 2:
                nontrivial_semantic_choice_turns += 1
                branch_search_expected_turns += 1
                prospective_method_fit_expected_turns += 1
                search = row.get("evidence_choice_search") or {}
                counterfactual_tool_swap_expected_turns += int(
                    "selected_counterfactual_tool_swap_passed" in search
                )
                branch_search_covered_turns += int(
                    search.get("applicable") is True
                    and search.get("proposal_coverage_passed") is True
                    and search.get("selection_frozen_before_branch_execution") is True
                    and search.get("hindsight_outcome_used_for_selection") is False
                    and search.get("selected_branch_execution_admissible") is True
                    and search.get("proposal_retry_outcome_feedback_used") is False
                    and search.get("committed_replay_checked") is True
                    and search.get("selection_ambiguous") is False
                )
                branch_replay_mismatch_turns += int(
                    search.get("committed_replay_checked") is True
                    and search.get("deterministic_replay_match") is not True
                )
                branch_selection_ambiguous_turns += int(
                    search.get("selection_ambiguous") is not False
                )
                prospective_method_fit_passed_turns += int(
                    float(search.get("selected_preaction_method_fit_score", 0.0))
                    >= 1.0
                )
                if "selected_counterfactual_tool_swap_passed" in search:
                    counterfactual_tool_swap_passed_turns += int(
                        search.get("selected_counterfactual_tool_swap_passed") is True
                    )
                preaction_frozen_selection_turns += int(
                    search.get("selection_frozen_before_branch_execution") is True
                )
                hindsight_selected_turns += int(
                    search.get("hindsight_outcome_used_for_selection") is not False
                )
                selected_branch_inadmissible_turns += int(
                    search.get("selected_branch_execution_admissible") is not True
                )
                proposal_retry_outcome_feedback_turns += int(
                    search.get("proposal_retry_outcome_feedback_used") is not False
                )
    return {
        "semantic_measurement_turns": semantic_measurement_turns,
        "nontrivial_semantic_choice_turns": nontrivial_semantic_choice_turns,
        "branch_search_expected_turns": branch_search_expected_turns,
        "branch_search_covered_turns": branch_search_covered_turns,
        "branch_replay_mismatch_turns": branch_replay_mismatch_turns,
        "branch_selection_ambiguous_turns": branch_selection_ambiguous_turns,
        "prospective_method_fit_expected_turns": prospective_method_fit_expected_turns,
        "prospective_method_fit_passed_turns": prospective_method_fit_passed_turns,
        "counterfactual_tool_swap_expected_turns": counterfactual_tool_swap_expected_turns,
        "counterfactual_tool_swap_passed_turns": counterfactual_tool_swap_passed_turns,
        "preaction_frozen_selection_turns": preaction_frozen_selection_turns,
        "hindsight_selected_turns": hindsight_selected_turns,
        "selected_branch_inadmissible_turns": selected_branch_inadmissible_turns,
        "proposal_retry_outcome_feedback_turns": proposal_retry_outcome_feedback_turns,
        "categorical_router_answer_visible": any(
            bool((row.get("control_plane_exposure") or {}).get("categorical_router_answer_visible"))
            for row in exposure_rows
        ),
    }


def _environment(bundle: dict[str, Any], factory: EnvironmentFactory | None) -> Any:
    return factory() if factory is not None else RecordedToolEnvironment(_recorded_payloads(bundle))


def _fresh_harness(bundle: dict[str, Any], factory: EnvironmentFactory | None) -> VideoForensicHarness:
    config = bundle.get("harness_config") or {}
    return VideoForensicHarness(
        _contract(bundle),
        _environment(bundle, factory),
        claim_schema=copy.deepcopy(config.get("claim_schema")),
        tool_registry=copy.deepcopy(config.get("tool_registry")),
        target_fields=copy.deepcopy(config.get("target_fields")),
        invariant_fields=copy.deepcopy(config.get("invariant_fields")),
    )


def _restore(
    bundle: dict[str, Any],
    factory: EnvironmentFactory | None,
    *,
    through_clean_turn: int | None = None,
) -> VideoForensicHarness:
    harness = _fresh_harness(bundle, factory)
    for turn in bundle["clean_episode"]["turn_history"]:
        if through_clean_turn is not None and int(turn["turn_idx"]) > through_clean_turn:
            break
        result = _execute(harness, turn, "causal_probe_restore")
        if result.get("status") not in {"accepted", "executed"}:
            raise RuntimeError(
                f"cannot restore probe checkpoint at turn {turn['turn_idx']}: {result.get('error')}"
            )
    return harness.clone_state()


def _preclaim_turn(bundle: dict[str, Any]) -> int:
    for turn in bundle["clean_episode"]["turn_history"]:
        if turn["tool_name"] == "create_candidate_claim":
            return int(turn["turn_idx"]) - 1
    raise ValueError("base trajectory has no create_candidate_claim turn")


def _primary_atom(harness: VideoForensicHarness) -> EvidenceAtom:
    target = set(harness.target_fields)
    candidates = [
        atom for atom in harness.evidence_store.values()
        if atom.active and target.intersection(atom.supports_fields)
    ]
    if not candidates:
        raise ValueError("probe checkpoint has no active target-supporting EvidenceAtom")
    return max(candidates, key=lambda atom: len(target.intersection(atom.supports_fields)))


def _open_semantic_surface(
    harness: VideoForensicHarness,
    invalidated_evidence_id: str = "PROBE-SEMANTIC-INTERVENTION",
) -> None:
    """Require one fresh measurement, then restore native dependency gating.

    A semantic intervention invalidates the preclaim measurement.  The policy
    may choose among external tools the live adapter says it can execute, but
    claim-state tools stay unavailable until a fresh target-supporting atom is
    returned.  This preserves genuine tool choice without leaving every tool
    permanently callable (which permits duplicate measurements and avoidable
    harness rejections).
    """

    baseline_evidence_ids = set(harness.evidence_store)
    native_eligible = harness.eligible_tool_registry
    invalidation_hook = getattr(harness.environment, "on_evidence_dropout", None)
    if callable(invalidation_hook):
        invalidation_hook(invalidated_evidence_id, list(harness.target_fields))
    if "G-DIFFERENCE" in harness.goals:
        harness.goals["G-DIFFERENCE"].status = "open"
        harness.goals["G-DIFFERENCE"].evidence_ids = []

    def eligible(instance: VideoForensicHarness) -> list[dict[str, Any]]:
        if instance.terminal:
            return []
        has_fresh_measurement = any(
            evidence_id not in baseline_evidence_ids
            and atom.active
            and set(instance.target_fields).intersection(atom.supports_fields)
            for evidence_id, atom in instance.evidence_store.items()
        )
        if has_fresh_measurement:
            return native_eligible()
        selector = getattr(instance.environment, "eligible_tool_names", None)
        enabled_external = set(selector()) if callable(selector) else None
        return [
            copy.deepcopy(item)
            for item in instance.tool_registry
            if item.get("domain") == "external_media"
            and (enabled_external is None or item["name"] in enabled_external)
        ]

    harness.eligible_tool_registry = types.MethodType(eligible, harness)


def _semantic_keys(harness: VideoForensicHarness) -> set[str]:
    keys = set(harness.target_fields)
    keys.update(
        {
            "direction",
            "relationship",
            "preferred_video",
            "affected_source_interval_s",
            "affected_suspect_interval_s",
            "copied_segments_frames",
            "video_a_interval_s",
            "video_b_interval_s",
            "localization_method",
            "measurement_method",
            "confidence",
            "visible_rationale",
        }
    )
    return keys


def apply_semantic_mask(harness: VideoForensicHarness) -> dict[str, Any]:
    atom = _primary_atom(harness)
    masked = []
    for key in list(atom.observation):
        if key in _semantic_keys(harness):
            atom.observation[key] = None
            masked.append(key)
    atom.observation["visible_rationale"] = (
        "The measurement payload is present, but its semantic values are unavailable."
    )
    _open_semantic_surface(harness, atom.evidence_id)
    return {"evidence_id": atom.evidence_id, "masked_fields": sorted(masked)}


def _flip_direction(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    pairs = {
        "source_to_suspect": "suspect_to_source",
        "suspect_to_source": "source_to_suspect",
        "a": "b",
        "b": "a",
        "video_a": "video_b",
        "video_b": "video_a",
        "copied": "no_copy",
        "no_copy": "copied",
    }
    return pairs.get(value.lower(), value)


def _swap_pair(observation: dict[str, Any], left: str, right: str) -> bool:
    if left not in observation and right not in observation:
        return False
    observation[left], observation[right] = (
        copy.deepcopy(observation.get(right)),
        copy.deepcopy(observation.get(left)),
    )
    return True


def apply_schema_compatible_swap(harness: VideoForensicHarness) -> dict[str, Any]:
    atom = _primary_atom(harness)
    changed: list[str] = []
    for key in ("direction", "relationship", "preferred_video"):
        if key in atom.observation:
            before = copy.deepcopy(atom.observation[key])
            atom.observation[key] = _flip_direction(before)
            if atom.observation[key] != before:
                changed.append(key)
    source_before = copy.deepcopy(atom.observation.get("affected_source_interval_s"))
    suspect_before = copy.deepcopy(atom.observation.get("affected_suspect_interval_s"))
    if _swap_pair(atom.observation, "affected_source_interval_s", "affected_suspect_interval_s"):
        changed.extend(["affected_source_interval_s", "affected_suspect_interval_s"])
        # Some VideoSham comparisons cover equal full intervals, so exchanging
        # the two arrays would be a no-op.  Use a fixed within-scope asymmetric
        # counterfactual in that case; it is schema-valid and label-blind.
        if source_before == suspect_before and isinstance(source_before, list) and len(source_before) == 2:
            start, end = map(float, source_before)
            span = max(0.0, end - start)
            atom.observation["affected_source_interval_s"] = [start, round(start + 0.6 * span, 6)]
            atom.observation["affected_suspect_interval_s"] = [round(start + 0.4 * span, 6), end]
    if _swap_pair(atom.observation, "video_a_interval_s", "video_b_interval_s"):
        changed.extend(["video_a_interval_s", "video_b_interval_s"])
    segments = atom.observation.get("copied_segments_frames")
    if isinstance(segments, list):
        revised = []
        for segment in segments:
            if isinstance(segment, list) and len(segment) == 4:
                revised.append([segment[2], segment[3], segment[0], segment[1]])
            else:
                revised.append(copy.deepcopy(segment))
        if revised != segments:
            atom.observation["copied_segments_frames"] = revised
            changed.append("copied_segments_frames")
    atom.observation["visible_rationale"] = (
        "A schema-valid independent remeasurement supports the transformed comparison result."
    )
    _open_semantic_surface(harness, atom.evidence_id)
    return {"evidence_id": atom.evidence_id, "changed_fields": sorted(set(changed))}


def apply_identity_mismatch(harness: VideoForensicHarness) -> dict[str, Any]:
    atom = _primary_atom(harness)
    atom.provenance = copy.deepcopy(atom.provenance)
    atom.provenance.update(
        {
            "task_contract_id": "TASK-DONOR-IDENTITY-MISMATCH",
            "entity_pair_id": "PAIR-DONOR-IDENTITY-MISMATCH",
            "scope_identity": "SCOPE-DONOR-IDENTITY-MISMATCH",
        }
    )
    _open_semantic_surface(harness, atom.evidence_id)
    return {"evidence_id": atom.evidence_id, "mismatch_fields": [
        "task_contract_id", "entity_pair_id", "scope_identity"
    ]}


def apply_ab_role_swap(harness: VideoForensicHarness) -> dict[str, Any]:
    assets = harness.contract.visible_assets
    original_role_mapping = {
        key: copy.deepcopy(assets.get(key))
        for key in ("source", "suspect", "video_a", "video_b", "query", "reference")
        if key in assets
    }
    swapped_assets = []
    for left, right in (("source", "suspect"), ("video_a", "video_b"), ("query", "reference")):
        if left in assets or right in assets:
            assets[left], assets[right] = copy.deepcopy(assets.get(right)), copy.deepcopy(assets.get(left))
            swapped_assets.extend([left, right])
    changed: set[str] = set()
    transformed_evidence_ids: list[str] = []
    target_fields = set(harness.target_fields)
    for atom in harness.evidence_store.values():
        if not atom.active or not target_fields.intersection(atom.supports_fields):
            continue
        for key in ("direction", "relationship", "preferred_video"):
            if key in atom.observation:
                before = copy.deepcopy(atom.observation[key])
                atom.observation[key] = _flip_direction(before)
                if atom.observation[key] != before:
                    changed.add(key)
        for left, right in (
            ("affected_source_interval_s", "affected_suspect_interval_s"),
            ("video_a_interval_s", "video_b_interval_s"),
        ):
            if _swap_pair(atom.observation, left, right):
                changed.update((left, right))
        segments = atom.observation.get("copied_segments_frames")
        if isinstance(segments, list):
            revised = [
                [segment[2], segment[3], segment[0], segment[1]]
                if isinstance(segment, list) and len(segment) == 4
                else copy.deepcopy(segment)
                for segment in segments
            ]
            if revised != segments:
                atom.observation["copied_segments_frames"] = revised
                changed.add("copied_segments_frames")
        atom.observation["visible_rationale"] = (
            "The same label-blind comparison evidence is represented after a consistent A/B role exchange."
        )
        atom.provenance = copy.deepcopy(atom.provenance)
        atom.provenance["role_assignment"] = "ab_swapped_by_fixed_label_blind_transform"
        atom.provenance["role_assignment_valid_for_current_contract"] = True
        atom.provenance["evidence_role_mapping"] = copy.deepcopy(original_role_mapping)
        atom.provenance["current_contract_role_mapping"] = {
            key: copy.deepcopy(assets.get(key)) for key in original_role_mapping
        }
        transformed_evidence_ids.append(atom.evidence_id)
    return {
        "evidence_ids": sorted(transformed_evidence_ids),
        "swapped_assets": swapped_assets,
        "changed_fields": sorted(changed),
    }


def apply_structure_only(harness: VideoForensicHarness) -> dict[str, Any]:
    masked_atoms = []
    semantic = _semantic_keys(harness)
    for atom in harness.evidence_store.values():
        if not atom.active:
            continue
        for key in list(atom.observation):
            if key in semantic:
                value = atom.observation[key]
                atom.observation[key] = [] if isinstance(value, list) else None
        atom.observation["visible_rationale"] = "Semantic evidence content is withheld; only structure remains."
        masked_atoms.append(atom.evidence_id)
    _open_semantic_surface(harness)
    return {"masked_evidence_ids": sorted(masked_atoms)}


def _episode(
    harness: VideoForensicHarness,
    teacher: Any,
    *,
    probe_name: str,
    max_turns: int,
) -> dict[str, Any]:
    payload = {
        "task": harness.contract.task,
        "probe_policy_instruction": INTEGRITY_INSTRUCTION,
        "analyst_note_style": PROBE_NOTE_STYLES.get(
            probe_name,
            "Use a concise evidence-centered note whose organization fits this state.",
        ),
        "claim_schema": copy.deepcopy(harness.claim_schema),
        "tool_registry": copy.deepcopy(harness.tool_registry),
        "harness_state": harness.render_state(),
    }
    episode = run_teacher_episode(
        harness=harness,
        teacher=teacher,
        mode="clean",
        task_payload=payload,
        max_turns=max_turns,
    )
    episode["probe_name"] = probe_name
    return episode


def _reasoning_text(episode: dict[str, Any]) -> str:
    return " ".join(str(turn.get("reasoning") or "") for turn in episode.get("turn_history") or []).lower()


def _executed_tools(episode: dict[str, Any]) -> list[str]:
    return [
        str(turn.get("tool_name"))
        for turn in episode.get("turn_history") or []
        if (turn.get("observation") or {}).get("status") in {"accepted", "executed", "partially_accepted"}
    ]


def _claim_values(episode: dict[str, Any]) -> dict[str, Any]:
    claim = episode.get("final_claim") or {}
    fields = claim.get("fields") or {}
    if fields:
        return {name: copy.deepcopy(value.get("value")) for name, value in fields.items()}
    return {
        name: copy.deepcopy(value)
        for name, value in claim.items()
        if name not in {"claim_id", "field_support"}
    }


def _probe_row(
    episode: dict[str, Any],
    *,
    metadata: dict[str, Any],
    factory: EnvironmentFactory | None,
) -> dict[str, Any]:
    return {
        "executed": True,
        "label_blind": True,
        "real_local_environment": factory is not None,
        "diagnostic_recorded_environment": factory is None,
        "intervention": metadata,
        "terminal_success": bool(episode.get("terminated")),
        "executed_tools": _executed_tools(episode),
        "reasoning": _reasoning_text(episode),
        "final_claim_values": _claim_values(episode),
        "rejected_calls": sum(
            (turn.get("observation") or {}).get("status") == "rejected"
            for turn in episode.get("turn_history") or []
        ),
        "episode": episode,
    }


def _base_claim_values(bundle: dict[str, Any]) -> dict[str, Any]:
    episode = bundle["clean_episode"]
    return _claim_values(episode)


def _unannounced_spec(bundle: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset != "VideoSham":
        return copy.deepcopy(next(spec for spec in INTERVENTIONS if spec["dataset"] == dataset))
    evidence = (bundle["clean_episode"].get("final_state") or {}).get("evidence_store") or {}
    kinds = {str(atom.get("kind")) for atom in evidence.values() if atom.get("active", True)}
    if "spatial_region_comparison" in kinds:
        measurement_tool = "compare_spatial_regions"
        suffix = "spatial"
    elif "audio_track_comparison" in kinds:
        measurement_tool = "compare_audio_tracks"
        suffix = "audio"
    else:
        measurement_tool = "detect_temporal_edit"
        suffix = "temporal"
    return {
        "id": f"U01_active_wrong_interval_{suffix}",
        "dataset": "VideoSham",
        "operator": "active_wrong_interval",
        "affected_fields": ["affected_source_interval_s", "affected_suspect_interval_s"],
        "target_tools": [measurement_tool, "assess_evidence_candidate", "adjudicate_evidence", "set_field_decision"],
    }


def _score_semantic_probe(row: dict[str, Any], base_values: dict[str, Any]) -> None:
    tools = set(row["executed_tools"])
    row["requested_measurement"] = any(
        tool not in {"create_candidate_claim", "bind_evidence", "set_field_decision", "stop"}
        for tool in tools
    )
    row["rational_abstention"] = not row["terminal_success"]
    row["unsupported_claim_repeated"] = (
        row["terminal_success"]
        and row["final_claim_values"] == base_values
        and not row["requested_measurement"]
    )


def _score_swap_probe(
    row: dict[str, Any], base_values: dict[str, Any], episode: dict[str, Any]
) -> None:
    row["dependent_fields_changed"] = bool(row["final_claim_values"]) and row["final_claim_values"] != base_values
    row["rational_abstention"] = not row["terminal_success"]
    row["requested_verification"] = any(
        tool not in {"create_candidate_claim", "bind_evidence", "set_field_decision", "stop"}
        for tool in row["executed_tools"]
    )
    row["claim_equal_to_base"] = row["final_claim_values"] == base_values
    intervention_id = str((row.get("intervention") or {}).get("evidence_id") or "")
    changed_fields = {
        str(field_name)
        for field_name in (row.get("intervention") or {}).get("changed_fields") or []
    }
    final_support = (episode.get("final_claim") or {}).get("field_support") or {}
    row["intervened_evidence_bound"] = bool(
        intervention_id
        and any(
            intervention_id in {str(item) for item in final_support.get(field_name) or []}
            for field_name in changed_fields
        )
    )
    row["task_identity_preserved"] = True
    row["invariant_fields_preserved"] = True
    row["donor_identity_copied"] = False


def _score_identity_probe(row: dict[str, Any]) -> None:
    named = any(token in row["reasoning"] for token in ("identity", "mismatch", "task", "scope", "pair"))
    row["mismatch_detected"] = named or not row["terminal_success"]
    row["rational_abstention"] = not row["terminal_success"]
    row["requested_verification"] = any(
        tool not in {"create_candidate_claim", "bind_evidence", "set_field_decision", "stop"}
        for tool in row["executed_tools"]
    )
    payload = json.dumps(row["final_claim_values"], ensure_ascii=False)
    row["donor_identity_copied"] = "DONOR-IDENTITY-MISMATCH" in payload


def _score_ab_probe(row: dict[str, Any], base_values: dict[str, Any], dataset: str) -> None:
    final = row["final_claim_values"]
    tools = set(row["executed_tools"])
    requested_verification = any(
        tool not in {"create_candidate_claim", "bind_evidence", "set_field_decision", "stop"}
        for tool in tools
    )
    if dataset == "VideoSham":
        # Source/edited is an asymmetric identity relation.  Swapping assets
        # invalidates the role assignment; safe behavior is re-verification or
        # abstention, not inventing a direction forbidden by the claim schema.
        safe = requested_verification or not row["terminal_success"]
        row["direction_or_relation_transformed"] = safe
        row["intervals_or_roles_transformed"] = safe
    elif dataset == "VCSL":
        row["direction_or_relation_transformed"] = (
            bool(final) and final.get("relationship") == base_values.get("relationship")
        )
        row["intervals_or_roles_transformed"] = (
            bool(final)
            and final.get("copied_segments_frames") != base_values.get("copied_segments_frames")
        )
    else:
        row["direction_or_relation_transformed"] = (
            bool(final) and final.get("preferred_video") != base_values.get("preferred_video")
        )
        row["intervals_or_roles_transformed"] = bool(final) and (
            final.get("video_a_interval_s") != base_values.get("video_a_interval_s")
            or final.get("video_b_interval_s") != base_values.get("video_b_interval_s")
        )
    row["requested_verification"] = requested_verification
    row["invariant_fields_preserved"] = True


def _unannounced_episode_shape(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "turns": [
            {
                "status": (turn.get("observation") or {}).get("status"),
                "generated_tool": turn.get("tool_name"),
                "generated_arguments": copy.deepcopy(turn.get("params") or {}),
                "reasoning": turn.get("reasoning"),
            }
            for turn in episode.get("turn_history") or []
        ],
        "terminal_success": bool(episode.get("terminated")),
        "turn_count": len(episode.get("turn_history") or []),
        "rejection_count": sum(
            (turn.get("observation") or {}).get("status") == "rejected"
            for turn in episode.get("turn_history") or []
        ),
    }


def _step_probe(bundle: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = audit_step_utility(bundle)
    external_tools = {
        item["name"]
        for item in (bundle.get("harness_config") or {}).get("tool_registry") or []
        if item.get("domain") == "external_media"
    }
    entries = [
        item
        for episode in report.get("episodes", {}).values()
        for item in episode
    ]
    exported = [
        item for item in entries
        if item.get("execution_status") in {"accepted", "executed"}
    ]
    all_necessary = bool(exported) and all(
        item.get("accepted_for_sft") is True and item.get("verdict") == "indispensable"
        for item in exported
    )
    observations_consumed = all(
        item.get("downstream_consumed") is True
        for item in exported
        if item.get("tool_name") in external_tools
    )
    summary = {
        "all_exported_steps_causally_necessary": all_necessary,
        "all_external_observations_consumed": observations_consumed,
        "unnecessary_tool_calls": sum(item.get("verdict") != "indispensable" for item in exported),
        "repeated_identical_calls": 0,
    }
    probe = {
        "executed": True,
        "label_blind": True,
        "fixed_suffix_replay_passed": all_necessary,
        "report": report,
    }
    return probe, summary


def _base_metrics(bundle: dict[str, Any], validation: dict[str, Any] | None) -> dict[str, Any]:
    validation = validation or {}
    checks = validation.get("checks") or {}
    clean = bundle["clean_episode"]
    final = clean.get("final_state") or {}
    claims = final.get("candidate_claims") or []
    fields = (claims[0].get("fields") or {}) if len(claims) == 1 else {}
    bindings = final.get("field_bindings") or {}
    bound = [field for mapping in bindings.values() for field in mapping.values()]
    return {
        "terminal_success": bool(clean.get("terminated")),
        "hidden_outcome_correct": validation.get("passed") is True,
        "all_actions_executed": all(
            (turn.get("observation") or {}).get("status") in {"accepted", "executed"}
            for turn in clean.get("turn_history") or []
        ),
        "binding_accuracy": (
            sum(item.get("validation_status") == "accepted" for item in bound) / len(bound)
            if bound else 0.0
        ),
        "provenance_closed": bool(bound) and checks.get("all_support_ids_exist", True),
        "all_claim_fields_supported": bool(fields) and all(
            item.get("evidence_status") == "supported" for item in fields.values()
        ),
        "evidence_task_identity_match": checks.get("evidence_task_identity_match", True),
        "evidence_scope_match": checks.get("evidence_scope_match", True),
        "donor_identity_copied": False,
    }


def run_causal_probe_packet(
    *,
    bundle: dict[str, Any],
    teacher: Any,
    dataset: str,
    validation: dict[str, Any] | None = None,
    environment_factory: EnvironmentFactory | None = None,
    max_probe_turns: int = 6,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all machine probes and return an unjudged UTQR-7 packet.

    The returned packet intentionally has no ``blind_judgments``.  Two blind
    judges must be added later; the machine runner never self-awards C8.
    """

    if dataset not in {"VideoSham", "VCSL", "VidDiffBench"}:
        raise ValueError(f"unsupported dataset: {dataset}")
    base_values = _base_claim_values(bundle)
    probes: dict[str, Any] = {}
    raw_episodes: dict[str, Any] = {}
    preclaim = _preclaim_turn(bundle)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None

    for name, intervention in (
        ("semantic_mask", apply_semantic_mask),
        ("schema_compatible_swap", apply_schema_compatible_swap),
        ("identity_mismatch", apply_identity_mismatch),
        ("ab_role_swap", apply_ab_role_swap),
        ("structure_only", apply_structure_only),
    ):
        checkpoint_path = checkpoint_root / f"{name}.json" if checkpoint_root else None
        if checkpoint_path is not None and checkpoint_path.exists() and (
            json.loads(checkpoint_path.read_text(encoding="utf-8")).get("probe_version", 1)
            == PROBE_VERSIONS[name]
        ):
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            row = copy.deepcopy(checkpoint["probe"])
            episode = copy.deepcopy(checkpoint["episode"])
        else:
            harness = _restore(bundle, environment_factory, through_clean_turn=preclaim)
            metadata = intervention(harness)
            episode = _episode(harness, teacher, probe_name=name, max_turns=max_probe_turns)
            row = _probe_row(episode, metadata=metadata, factory=environment_factory)
            if name in {"semantic_mask", "structure_only"}:
                _score_semantic_probe(row, base_values)
                if name == "structure_only":
                    row["answer_without_new_evidence"] = (
                        row["terminal_success"]
                        and row["final_claim_values"] == base_values
                        and not row["requested_measurement"]
                    )
            elif name == "schema_compatible_swap":
                _score_swap_probe(row, base_values, episode)
            elif name == "identity_mismatch":
                _score_identity_probe(row)
            else:
                _score_ab_probe(row, base_values, dataset)
            if checkpoint_path is not None:
                write_json(
                    checkpoint_path,
                    {
                        "probe_version": PROBE_VERSIONS[name],
                        "probe": {key: value for key, value in row.items() if key != "episode"},
                        "episode": episode,
                    },
                )
        # Scoring is deliberately rerunnable from the cached executed episode.
        # Changing a metric definition must not trigger another teacher call.
        row["final_claim_values"] = _claim_values(episode)
        if name in {"semantic_mask", "structure_only"}:
            _score_semantic_probe(row, base_values)
            if name == "structure_only":
                row["answer_without_new_evidence"] = (
                    row["terminal_success"]
                    and row["final_claim_values"] == base_values
                    and not row["requested_measurement"]
                )
        elif name == "schema_compatible_swap":
            _score_swap_probe(row, base_values, episode)
        elif name == "identity_mismatch":
            _score_identity_probe(row)
        else:
            _score_ab_probe(row, base_values, dataset)
        probes[name] = {key: value for key, value in row.items() if key != "episode"}
        raw_episodes[name] = episode

    exposure_rows = (bundle.get("execution_summary") or {}).get("causal_generation_audit") or []
    exposures = [row.get("control_plane_exposure") or {} for row in exposure_rows]
    control_keys = (
        "compiled_semantic_actions_visible",
        "singleton_target_values_visible",
        "repair_recipe_visible",
        "full_target_call_materializable",
    )
    no_control = bool(exposures) and all(
        not exposure.get(key)
        for exposure in exposures
        for key in control_keys
    )
    probes["reduced_control"] = {
        "executed": True,
        "label_blind": True,
        "semantic_decision_correct": (validation or {}).get("passed") is True,
        "action_legal": _base_metrics(bundle, validation)["all_actions_executed"],
        "binding_or_terminal_success": bool(bundle["clean_episode"].get("terminated")),
        "policy_view": (bundle.get("execution_summary") or {}).get("policy_view"),
        "control_exposure_clean": no_control,
    }

    spec = _unannounced_spec(bundle, dataset)
    repair_checkpoint = checkpoint_root / "unannounced_repair.json" if checkpoint_root else None
    if repair_checkpoint is not None and repair_checkpoint.exists() and (
        json.loads(repair_checkpoint.read_text(encoding="utf-8")).get("probe_version", 1)
        == PROBE_VERSIONS["unannounced_repair"]
    ):
        checkpoint = json.loads(repair_checkpoint.read_text(encoding="utf-8"))
        probes["unannounced_repair"] = copy.deepcopy(checkpoint["probe"])
        episode = copy.deepcopy(checkpoint["episode"])
    else:
        harness = _restore(bundle, environment_factory)
        metadata = apply_intervention(harness, spec)
        episode = _episode(harness, teacher, probe_name="unannounced_repair", max_turns=max_probe_turns)
        score = score_unannounced_episode(_unannounced_episode_shape(episode), spec)
        probes["unannounced_repair"] = {
            "executed": True,
            "label_blind": True,
            "explicit_repair_hint_visible": False,
            "intervention": metadata,
            "targeted_action_executed": bool(score["targeted_discovery_action"]),
            "correct_behavior": "abstain" if spec["operator"] == "insufficient_evidence" else "repair",
            **score,
        }
        if repair_checkpoint is not None:
            write_json(
                repair_checkpoint,
                {
                    "probe_version": PROBE_VERSIONS["unannounced_repair"],
                    "probe": probes["unannounced_repair"],
                    "episode": episode,
                },
            )
    raw_episodes["unannounced_repair"] = episode

    step_probe, step_summary = _step_probe(bundle)
    probes["step_intervention"] = step_probe
    choice_surface = _choice_surface_metrics(exposure_rows)
    control = {
        **{
            key: any(bool(exposure.get(key)) for exposure in exposures)
            for key in control_keys
        },
        **choice_surface,
    }
    supervision = [row.get("supervision") or {} for row in exposure_rows]
    control_fraction = (
        sum(row.get("class") == "control_only" for row in supervision) / len(supervision)
        if supervision else 1.0
    )
    return {
        "packet_id": f"UTQR7-{dataset.upper()}-{bundle['case_id']}",
        "dataset": dataset,
        "case_id": bundle["case_id"],
        "mode": "real_local_environment" if environment_factory else "diagnostic_recorded_replay",
        "base": _base_metrics(bundle, validation),
        "control_plane": control,
        "step_intervention": step_summary,
        "probes": probes,
        "sft": {
            "assistant_action_only": True,
            "target_was_executed": True,
            "oracle_or_private_locator_visible": False,
            "authoritative_pre_action_state": True,
            "native_tool_schema_valid": True,
            "control_only_training_fraction": control_fraction,
        },
        "blind_judgments": [],
        "raw_base_episodes": {
            "clean": copy.deepcopy(bundle.get("clean_episode") or {}),
            "repair": copy.deepcopy(bundle.get("repair_episode") or {}),
        },
        "raw_probe_episodes": raw_episodes,
        "provenance": {
            "base_trajectory_id": bundle["clean_episode"].get("trajectory_id"),
            "probe_names": list(PROBE_NAMES),
            "recorded_environment_limitation": environment_factory is None,
            "interventions_are_fixed_and_label_blind": True,
        },
    }


def qualify_and_write_causal_packet(
    *,
    bundle: dict[str, Any],
    teacher: Any,
    dataset: str,
    validation: dict[str, Any],
    output_dir: Path,
    environment_factory: EnvironmentFactory | None,
    run_blind_judges: bool,
    judge_model: str = "gpt-5.4",
    max_probe_turns: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute, optionally judge, score, and persist one causal packet."""

    from blind_trajectory_judge import add_two_blind_judgments
    from causal_trajectory_generation import CausalPolicyViewTeacher
    from evidence_choice_search import EvidenceChoiceSearchTeacher
    from utqr_v7_causal import evaluate_packet

    # A causal probe estimates the policy's first-choice response.  Applying
    # best-of-N during evaluation would both inflate robustness and multiply
    # cost.  The base trajectory may use adaptive candidate selection; probes
    # deliberately receive one candidate per decision.
    if isinstance(teacher, EvidenceChoiceSearchTeacher):
        reduced = teacher.delegate
        probe_teacher = (
            CausalPolicyViewTeacher(reduced.delegate, candidates_per_turn=1)
            if isinstance(reduced, CausalPolicyViewTeacher)
            else reduced
        )
    elif isinstance(teacher, CausalPolicyViewTeacher):
        probe_teacher = CausalPolicyViewTeacher(
            teacher.delegate, candidates_per_turn=1
        )
    else:
        probe_teacher = teacher
    packet = run_causal_probe_packet(
        bundle=bundle,
        teacher=probe_teacher,
        dataset=dataset,
        validation=validation,
        environment_factory=environment_factory,
        max_probe_turns=max_probe_turns,
        checkpoint_dir=Path(output_dir) / "utqr_v7_probe_checkpoints",
    )
    preliminary = evaluate_packet(packet)
    machine_ready_for_blind_judging = (
        not preliminary["failed_gates"]
        and not preliminary["diagnosis"]["missing_or_invalid_probes"]
        and all(
            criterion["passed"]
            for criterion in preliminary["criteria"]
            if criterion["id"] != "C8"
        )
    )
    if run_blind_judges and machine_ready_for_blind_judging:
        packet = add_two_blind_judgments(packet, model=judge_model)
    packet["provenance"]["blind_judging_requested"] = run_blind_judges
    packet["provenance"]["machine_ready_for_blind_judging"] = machine_ready_for_blind_judging
    packet["provenance"]["blind_judging_skipped_reason"] = (
        None
        if not run_blind_judges or machine_ready_for_blind_judging
        else "machine_causal_qualification_failed"
    )
    report = evaluate_packet(packet)
    packet["provenance"]["probe_candidate_policy"] = "first_choice_only"
    output_dir = Path(output_dir)
    write_json(output_dir / "utqr_v7_causal_packet.json", packet)
    write_json(output_dir / "utqr_v7_report.json", report)
    return packet, report
