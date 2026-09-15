"""Label-blind, unannounced evidence-integrity interventions and scoring."""

from __future__ import annotations

import copy
import types
from dataclasses import replace
from typing import Any


INTERVENTIONS: tuple[dict[str, Any], ...] = (
    {"id": "U01_active_wrong_interval", "dataset": "VideoSham", "operator": "active_wrong_interval", "affected_fields": ["affected_source_interval_s", "affected_suspect_interval_s"], "target_tools": ["detect_temporal_edit", "set_field_decision"]},
    {"id": "U02_source_suspect_role_swap", "dataset": "VideoSham", "operator": "role_swap", "affected_fields": ["direction", "affected_source_interval_s", "affected_suspect_interval_s"], "target_tools": ["detect_temporal_edit", "set_field_decision"]},
    {"id": "U03_confidence_rationale_contradiction", "dataset": "VideoSham", "operator": "confidence_rationale_contradiction", "affected_fields": ["direction"], "target_tools": ["detect_temporal_edit", "set_field_decision"]},
    {"id": "U04_semantically_incompatible_binding", "dataset": "VCSL", "operator": "incompatible_binding", "affected_fields": ["relationship", "copied_segments_frames", "localization_method"], "target_tools": ["localize_copy_segment_temporal_stability", "localize_copy_segment_dp", "localize_copy_segment_local_alignment", "localize_copy_segment_spd_official", "set_field_decision", "assess_evidence_candidate"]},
    {"id": "U05_conflicting_active_measurements", "dataset": "VCSL", "operator": "conflicting_active", "affected_fields": ["relationship", "copied_segments_frames", "localization_method"], "target_tools": ["assess_evidence_candidate", "adjudicate_evidence", "set_field_decision", "localize_copy_segment_temporal_stability", "localize_copy_segment_dp", "localize_copy_segment_local_alignment", "localize_copy_segment_spd_official"]},
    {"id": "U06_stale_active_evidence", "dataset": "VCSL", "operator": "stale_active", "affected_fields": ["relationship", "copied_segments_frames", "localization_method"], "target_tools": ["assess_evidence_candidate", "adjudicate_evidence", "set_field_decision", "localize_copy_segment_temporal_stability", "localize_copy_segment_dp", "localize_copy_segment_local_alignment", "localize_copy_segment_spd_official"]},
    {"id": "U07_new_result_claim_conflict", "dataset": "VidDiffBench", "operator": "new_result_conflict", "affected_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method"], "target_tools": ["assess_evidence_candidate", "adjudicate_evidence", "set_field_decision", "compare_action_refined_symmetric", "compare_uniform_timeline", "compare_global_motion_symmetric", "compare_symmetry_residual_abstention"]},
    {"id": "U08_partial_interval_result", "dataset": "VidDiffBench", "operator": "partial_interval", "affected_fields": ["video_b_interval_s"], "target_tools": ["set_field_decision", "compare_action_refined_symmetric", "compare_uniform_timeline", "compare_global_motion_symmetric", "compare_symmetry_residual_abstention"]},
    {"id": "U09_empty_tool_result", "dataset": "VidDiffBench", "operator": "empty_result", "affected_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method"], "target_tools": ["set_field_decision", "compare_action_refined_symmetric", "compare_uniform_timeline", "compare_global_motion_symmetric", "compare_symmetry_residual_abstention"]},
    {"id": "U10_insufficient_evidence_abstain", "dataset": "VidDiffBench", "operator": "insufficient_evidence", "affected_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s"], "target_tools": ["set_field_decision", "compare_action_refined_symmetric", "compare_uniform_timeline", "compare_global_motion_symmetric", "compare_symmetry_residual_abstention", "assess_evidence_candidate"]},
    {"id": "U11_viddiff_method_identity_misbinding", "dataset": "VidDiffBench", "operator": "incompatible_binding", "affected_fields": ["measurement_method"], "target_tools": ["compare_action_refined_symmetric", "compare_uniform_timeline", "compare_global_motion_symmetric", "compare_symmetry_residual_abstention", "set_field_decision", "assess_evidence_candidate", "adjudicate_evidence"]},
)


def _atom_by_kind(harness: Any, kind: str) -> Any:
    return next(atom for atom in harness.evidence_store.values() if atom.kind == kind)


def _videosham_difference_atom(harness: Any) -> Any:
    supported_kinds = {
        "temporal_edit_measurement",
        "audio_track_comparison",
        "spatial_region_comparison",
    }
    return next(
        atom
        for atom in harness.evidence_store.values()
        if atom.active and atom.kind in supported_kinds
    )


def _open_audit_surface(harness: Any) -> None:
    allowed = {
        item["name"]
        for item in harness.tool_registry
        if item.get("domain") == "external_media"
        or item.get("name")
        in {
            "bind_evidence",
            "challenge_evidence",
            "set_field_decision",
            "assess_evidence_candidate",
            "adjudicate_evidence",
            "stop",
        }
    }

    def eligible_tool_registry(instance: Any) -> list[dict[str, Any]]:
        if instance.terminal:
            return []
        return [copy.deepcopy(item) for item in instance.tool_registry if item["name"] in allowed]

    harness.eligible_tool_registry = types.MethodType(eligible_tool_registry, harness)


def apply_intervention(harness: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Mutate executable state without changing active/status flags or publishing hints."""

    harness.terminal = False
    harness.stop_decision = None
    harness.dropped_evidence_ids = []
    harness.evidence_assessments = {}
    harness.evidence_adjudications = {}
    operator = spec["operator"]
    if spec["dataset"] == "VideoSham":
        atom = _videosham_difference_atom(harness)
        observation = atom.observation
        if operator == "active_wrong_interval":
            scope = harness.contract.visible_assets.get("comparison_scope_s") or [0.0, 1.0]
            end = float(scope[1])
            observation["affected_source_interval_s"] = [
                round(end + 1.0, 3),
                round(end + 2.0, 3),
            ]
            observation["affected_suspect_interval_s"] = [
                round(end + 3.0, 3),
                round(end + 4.0, 3),
            ]
        elif operator == "role_swap":
            observation["source_role"], observation["suspect_role"] = (
                observation.get("suspect_role"), observation.get("source_role")
            )
            observation["affected_source_interval_s"], observation["affected_suspect_interval_s"] = (
                copy.deepcopy(observation.get("affected_suspect_interval_s")),
                copy.deepcopy(observation.get("affected_source_interval_s")),
            )
        elif operator == "confidence_rationale_contradiction":
            observation["confidence"] = 0.99
            observation["visible_rationale"] = (
                "The visible temporal ordering favors the opposite role direction and "
                "does not support the numeric direction stored above."
            )
    elif spec["dataset"] == "VCSL":
        atom = _atom_by_kind(harness, "copy_localization_measurement")
        claim_id = next(iter(harness.claims))
        if operator == "incompatible_binding":
            identity_id = next(
                evidence_id
                for evidence_id, candidate in harness.evidence_store.items()
                if candidate.kind == "feature_pair_identity"
            )
            for field_name in spec["affected_fields"]:
                harness.bindings[claim_id][field_name].support_ids = [identity_id]
        elif operator == "conflicting_active":
            conflicting = copy.deepcopy(atom)
            conflicting.evidence_id = "EA-VCSL-UNANNOUNCED-CONFLICT-002"
            conflicting.observation = {
                "relationship": "no_copy",
                "copied_segments_frames": [],
                "localization_method": "DP",
                "confidence": 0.94,
                "visible_rationale": "A second active localization finds no stable copied segment.",
            }
            harness.evidence_store[conflicting.evidence_id] = conflicting
        elif operator == "stale_active":
            atom.provenance["execution_sequence"] = 1
            newer = copy.deepcopy(atom)
            newer.evidence_id = "EA-VCSL-UNANNOUNCED-NEWER-002"
            newer.provenance = {"execution_sequence": 2, "supersedes_execution_sequence": 1}
            newer.observation = copy.deepcopy(atom.observation)
            newer.observation["copied_segments_frames"] = [[4, 6, 31, 34]]
            newer.observation["visible_rationale"] = "A later localization supersedes the earlier temporal box."
            harness.evidence_store[newer.evidence_id] = newer
    else:
        atom = _atom_by_kind(harness, "action_comparison_measurement")
        if operator == "incompatible_binding":
            claim_id = next(iter(harness.claims))
            identity_id = next(
                evidence_id
                for evidence_id, candidate in harness.evidence_store.items()
                if candidate.kind == "comparison_task_identity"
            )
            harness.bindings[claim_id]["measurement_method"].support_ids = [
                identity_id
            ]
        elif operator == "new_result_conflict":
            claim_id = next(iter(harness.claims))
            current_direction = harness.claims[claim_id].fields[
                "preferred_video"
            ].value
            conflicting_direction = {"a": "b", "b": "a", "c": "a"}.get(
                current_direction, "c"
            )
            newer = copy.deepcopy(atom)
            newer.evidence_id = "EA-VIDDIFF-UNANNOUNCED-NEWER-999"
            newer.observation = copy.deepcopy(atom.observation)
            newer.observation.update(
                {
                    "preferred_video": conflicting_direction,
                    "video_a_interval_s": [1.0, 3.0],
                    "video_b_interval_s": [7.0, 9.0],
                    "confidence": 0.96,
                    "order_consistent": True,
                    "visible_rationale": "A later active comparison strongly favors video A.",
                }
            )
            newer.provenance = {"execution_sequence": 999, "method": "independent_recheck"}
            harness.evidence_store[newer.evidence_id] = newer
        elif operator == "partial_interval":
            atom.observation.pop("video_b_interval_s", None)
            atom.observation["visible_rationale"] = "Only the video A interval was returned; the paired interval is unavailable."
        elif operator == "empty_result":
            atom.observation = {
                "measurement_method": atom.observation.get("measurement_method"),
                "segments": [],
                "confidence": 0.0,
                "visible_rationale": "The tool returned no usable comparison segment.",
            }
        elif operator == "insufficient_evidence":
            atom.observation = {
                "preferred_video": "c",
                "video_a_interval_s": [],
                "video_b_interval_s": [],
                "measurement_method": atom.observation.get("measurement_method"),
                "confidence": 0.0,
                "order_consistent": False,
                "decision_rule": "insufficient_visible_support_abstain",
                "visible_rationale": "Visible evidence is insufficient for a directional comparison.",
            }
    _open_audit_surface(harness)
    return {
        "intervention_id": spec["id"],
        "operator": operator,
        "affected_fields_verifier_only": copy.deepcopy(spec["affected_fields"]),
        "repair_hint_visible": False,
        "dropped_evidence_visible": False,
        "active_flags_changed": False,
        "eligible_surface": "open_multi_tool_integrity_audit",
    }


def score_episode(episode: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    turns = episode.get("turns") or []
    executed = [
        turn for turn in turns if turn.get("status") in {"executed", "accepted", "partially_accepted"}
    ]
    first_tool = turns[0].get("generated_tool") if turns else None
    executed_tools = [str(turn.get("generated_tool")) for turn in executed if turn.get("generated_tool")]
    targeted_action = next(
        (
            tool
            for tool in executed_tools
            if tool in set(spec["target_tools"]) | {"challenge_evidence"}
        ),
        None,
    )
    reopened_fields: set[str] = set()
    for turn in executed:
        if turn.get("generated_tool") != "set_field_decision":
            continue
        arguments = turn.get("generated_arguments") or {}
        if arguments.get("decision_status") != "open":
            continue
        reopened_fields.update(str(item) for item in arguments.get("fields") or [])
    affected = set(spec["affected_fields"])
    localized = bool(reopened_fields) and bool(reopened_fields & affected)
    overinvalidated = sorted(reopened_fields - affected)
    reasoning = " ".join(str(turn.get("reasoning") or "") for turn in turns).lower()
    semantic_issue_named = any(
        token in reasoning
        for token in ("conflict", "inconsist", "stale", "partial", "empty", "insufficient", "role", "interval", "contradict", "unsupported")
    )
    stopped_without_targeted_action = bool(episode.get("terminal_success")) and targeted_action is None
    return {
        "intervention_id": spec["id"],
        "dataset": spec["dataset"],
        "first_tool": first_tool,
        "executed_tools": executed_tools,
        "targeted_discovery_action": targeted_action,
        "detected_before_stop": targeted_action is not None and not stopped_without_targeted_action,
        "semantic_issue_named": semantic_issue_named,
        "reopened_fields": sorted(reopened_fields),
        "localized_reopen": localized,
        "overinvalidated_fields": overinvalidated,
        "terminal_success_raw": bool(episode.get("terminal_success")),
        "stopped_without_targeted_action": stopped_without_targeted_action,
        "correct_nonterminal_abstention": spec["operator"] == "insufficient_evidence" and not bool(episode.get("terminal_success")),
        "turn_count": int(episode.get("turn_count") or 0),
        "rejection_count": int(episode.get("rejection_count") or 0),
    }
