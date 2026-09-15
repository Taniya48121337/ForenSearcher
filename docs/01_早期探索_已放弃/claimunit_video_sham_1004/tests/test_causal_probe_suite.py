from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from causal_probe_suite import (  # noqa: E402
    PROBE_NAMES,
    _choice_surface_metrics,
    _claim_values,
    _unannounced_spec,
    apply_ab_role_swap,
    apply_identity_mismatch,
    apply_schema_compatible_swap,
    apply_semantic_mask,
    apply_structure_only,
)
from interactive_harness import EvidenceAtom, TaskContract, VideoForensicHarness  # noqa: E402


class EmptyEnvironment:
    def execute(self, name, arguments):  # pragma: no cover - transforms do not execute tools
        raise AssertionError((name, arguments))


def _harness() -> VideoForensicHarness:
    schema = {
        "schema_id": "TEST-COMPARISON-V1",
        "required_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s", "statement"],
        "target_fields": ["preferred_video", "video_a_interval_s", "video_b_interval_s"],
        "invariant_fields": ["statement"],
        "properties": {
            "preferred_video": {"type": "string", "enum": ["a", "b", "c"]},
            "video_a_interval_s": {"type": "array"},
            "video_b_interval_s": {"type": "array"},
            "statement": {"type": "string"},
        },
    }
    registry = [
        {"name": "compare_uniform_timeline", "domain": "external_media"},
        {"name": "create_candidate_claim", "domain": "harness"},
        {"name": "bind_evidence", "domain": "harness"},
        {"name": "set_field_decision", "domain": "harness"},
        {"name": "stop", "domain": "harness"},
    ]
    harness = VideoForensicHarness(
        TaskContract(
            contract_id="TASK-TEST",
            episode_mode="paired_compare",
            task="Compare A and B.",
            visible_assets={"video_a": "A", "video_b": "B"},
            claim_schema_id=schema["schema_id"],
        ),
        EmptyEnvironment(),
        claim_schema=schema,
        tool_registry=registry,
        target_fields=schema["target_fields"],
        invariant_fields=schema["invariant_fields"],
    )
    atom = EvidenceAtom(
        evidence_id="EA-TEST-001",
        kind="action_comparison_measurement",
        supports_fields=list(schema["target_fields"]),
        observation={
            "preferred_video": "a",
            "video_a_interval_s": [1.0, 2.0],
            "video_b_interval_s": [3.0, 4.0],
            "measurement_method": "uniform_timeline",
            "confidence": 0.8,
            "visible_rationale": "A completes the action first.",
        },
        provenance={"task_contract_id": "TASK-TEST", "entity_pair_id": "PAIR-TEST"},
    )
    harness.evidence_store[atom.evidence_id] = atom
    return harness


def test_probe_family_is_complete_and_fixed() -> None:
    assert set(PROBE_NAMES) == {
        "semantic_mask",
        "schema_compatible_swap",
        "identity_mismatch",
        "ab_role_swap",
        "reduced_control",
        "structure_only",
        "unannounced_repair",
        "step_intervention",
    }


def test_choice_surface_counts_only_selected_measurement_turns() -> None:
    rows = [
        {
            "selected_action": {"tool_name": "inspect_media"},
            "control_plane_exposure": {
                "semantic_measurement_tool_names": [],
                "semantic_measurement_choice_count": 0,
            },
        },
        {
            "selected_action": {"tool_name": "compare_audio_tracks"},
            "control_plane_exposure": {
                "semantic_measurement_tool_names": [
                    "compare_audio_tracks",
                    "compare_spatial_regions",
                ],
                "semantic_measurement_choice_count": 2,
            },
            "evidence_choice_search": {
                "applicable": True,
                "proposal_coverage_passed": True,
                "committed_replay_checked": True,
                "deterministic_replay_match": True,
                "selection_ambiguous": False,
                "selected_preaction_method_fit_score": 1.0,
                "selected_counterfactual_tool_swap_passed": True,
                "selection_frozen_before_branch_execution": True,
                "hindsight_outcome_used_for_selection": False,
                "selected_branch_execution_admissible": True,
                "proposal_retry_outcome_feedback_used": False,
            },
        },
        {
            "selected_action": {"tool_name": "detect_temporal_edit"},
            "control_plane_exposure": {
                "semantic_measurement_tool_names": ["detect_temporal_edit"],
                "semantic_measurement_choice_count": 1,
                "categorical_router_answer_visible": True,
            },
        },
    ]
    assert _choice_surface_metrics(rows) == {
        "semantic_measurement_turns": 2,
        "nontrivial_semantic_choice_turns": 1,
        "branch_search_expected_turns": 1,
        "branch_search_covered_turns": 1,
        "branch_replay_mismatch_turns": 0,
        "branch_selection_ambiguous_turns": 0,
        "prospective_method_fit_expected_turns": 1,
        "prospective_method_fit_passed_turns": 1,
        "counterfactual_tool_swap_expected_turns": 1,
        "counterfactual_tool_swap_passed_turns": 1,
        "preaction_frozen_selection_turns": 1,
        "hindsight_selected_turns": 0,
        "selected_branch_inadmissible_turns": 0,
        "proposal_retry_outcome_feedback_turns": 0,
        "categorical_router_answer_visible": True,
    }


def test_semantic_mask_keeps_evidence_identity_but_removes_answer() -> None:
    harness = _harness()
    result = apply_semantic_mask(harness)
    atom = harness.evidence_store[result["evidence_id"]]
    assert atom.evidence_id == "EA-TEST-001"
    assert atom.observation["preferred_video"] is None
    assert atom.observation["video_a_interval_s"] is None
    assert "unavailable" in atom.observation["visible_rationale"]
    assert [item["name"] for item in harness.eligible_tool_registry()] == [
        "compare_uniform_timeline"
    ]


def test_schema_swap_changes_only_dependent_semantics() -> None:
    harness = _harness()
    result = apply_schema_compatible_swap(harness)
    observation = harness.evidence_store["EA-TEST-001"].observation
    assert observation["preferred_video"] == "b"
    assert observation["video_a_interval_s"] == [3.0, 4.0]
    assert observation["video_b_interval_s"] == [1.0, 2.0]
    assert "preferred_video" in result["changed_fields"]
    assert harness.contract.contract_id == "TASK-TEST"


def test_schema_swap_makes_equal_intervals_causally_distinct() -> None:
    harness = _harness()
    atom = harness.evidence_store["EA-TEST-001"]
    atom.observation["video_b_interval_s"] = [1.0, 2.0]
    atom.observation["affected_source_interval_s"] = [0.0, 10.0]
    atom.observation["affected_suspect_interval_s"] = [0.0, 10.0]
    apply_schema_compatible_swap(harness)
    assert atom.observation["affected_source_interval_s"] == [0.0, 6.0]
    assert atom.observation["affected_suspect_interval_s"] == [4.0, 10.0]


def test_claim_value_extractor_accepts_flat_harness_final_claim() -> None:
    episode = {
        "final_claim": {
            "claim_id": "CL-1",
            "preferred_video": "b",
            "video_a_interval_s": [1, 2],
            "field_support": {"preferred_video": ["EA-1"]},
        }
    }
    assert _claim_values(episode) == {
        "preferred_video": "b",
        "video_a_interval_s": [1, 2],
    }


def test_identity_mismatch_is_visible_in_provenance_not_claim_content() -> None:
    harness = _harness()
    apply_identity_mismatch(harness)
    atom = harness.evidence_store["EA-TEST-001"]
    assert atom.observation["preferred_video"] == "a"
    assert atom.provenance["task_contract_id"] == "TASK-DONOR-IDENTITY-MISMATCH"
    assert atom.provenance["entity_pair_id"] == "PAIR-DONOR-IDENTITY-MISMATCH"


def test_ab_swap_is_equivariant_and_structure_only_cannot_leak_answer() -> None:
    harness = _harness()
    second = EvidenceAtom(
        evidence_id="EA-TEST-002",
        kind="action_comparison_measurement",
        supports_fields=list(harness.target_fields),
        observation={
            "preferred_video": "b",
            "video_a_interval_s": [5.0, 6.0],
            "video_b_interval_s": [7.0, 8.0],
            "measurement_method": "action_refined",
        },
        provenance={"task_contract_id": "TASK-TEST", "entity_pair_id": "PAIR-TEST"},
    )
    harness.evidence_store[second.evidence_id] = second
    result = apply_ab_role_swap(harness)
    assert harness.contract.visible_assets == {"video_a": "B", "video_b": "A"}
    assert harness.evidence_store["EA-TEST-001"].observation["preferred_video"] == "b"
    assert harness.evidence_store["EA-TEST-002"].observation["preferred_video"] == "a"
    assert harness.evidence_store["EA-TEST-002"].observation["video_a_interval_s"] == [7.0, 8.0]
    assert result["evidence_ids"] == ["EA-TEST-001", "EA-TEST-002"]
    assert all(
        atom.provenance["role_assignment_valid_for_current_contract"]
        for atom in harness.evidence_store.values()
    )

    masked = _harness()
    apply_structure_only(masked)
    observation = masked.evidence_store["EA-TEST-001"].observation
    assert observation["preferred_video"] is None
    assert observation["video_a_interval_s"] == []
    assert "structure remains" in observation["visible_rationale"]


def test_videosham_unannounced_probe_routes_to_observed_modality() -> None:
    bundle = {
        "clean_episode": {
            "final_state": {
                "evidence_store": {
                    "EA-SPATIAL-001": {"kind": "spatial_region_comparison", "active": True}
                }
            }
        }
    }
    spec = _unannounced_spec(bundle, "VideoSham")
    assert spec["operator"] == "active_wrong_interval"
    assert "compare_spatial_regions" in spec["target_tools"]
    assert "detect_temporal_edit" not in spec["target_tools"]
