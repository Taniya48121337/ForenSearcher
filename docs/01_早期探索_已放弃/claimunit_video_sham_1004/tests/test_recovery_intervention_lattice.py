from __future__ import annotations

import copy

from build_balanced_recovery_sft_mix import build_mix
from build_recovery_causal_frontier_mix import select_frontier
from build_recovery_intervention_lattice import build_lattice
from build_recovery_semantic_contrast_view import build as build_contrast
from audit_recovery_semantic_contrast_integrity import audit as audit_contrast


def _source_manifest() -> dict:
    return {
        "artifact": "fixture",
        "jobs": [
            {
                "episode_id": "source-1",
                "dataset": "VideoSham",
                "case_id": "case-a",
                "checkpoint": {"file": "unused", "trajectory_id": "t", "turn_index": 1},
                "intervention": {"id": "old"},
            },
            {
                "episode_id": "source-2",
                "dataset": "VidDiffBench",
                "case_id": "case-b",
                "checkpoint": {"file": "unused", "trajectory_id": "t", "turn_index": 1},
                "intervention": {"id": "old"},
            },
        ],
    }


def test_lattice_crosses_each_case_with_dataset_interventions() -> None:
    source = _source_manifest()
    lattice = build_lattice(source)
    assert len(lattice["jobs"]) == 8  # VideoSham 3 + VidDiffBench 5
    assert lattice["lattice"]["source_cases"] == 2
    assert len({item["episode_id"] for item in lattice["jobs"]}) == 8
    assert source["jobs"][0]["intervention"] == {"id": "old"}
    assert all(
        item["intervention"]["dataset"] == item["dataset"]
        for item in lattice["jobs"]
    )
    assert all(
        item["lattice"]["case_mass_policy"] == "normalize_after_compile"
        for item in lattice["jobs"]
    )


def _row(episode: str, case: str, stage: str = "challenge") -> dict:
    return {
        "messages": [{"role": "assistant", "content": episode}],
        "metadata": {
            "episode_id": episode,
            "case_id": case,
            "dataset": "VideoSham",
            "causal_recovery_stage": stage,
            "learning_value_weight": 1.0,
        },
    }


def test_case_normalization_prevents_variant_rich_case_dominance() -> None:
    rows = [
        _row("a-1", "case-a"),
        _row("a-2", "case-a", "fresh_measurement"),
        _row("b-1", "case-b"),
    ]
    report = {
        "results": [
            {"episode_id": "a-1", "status": "compiled", "selected_intervention_id": "U01"},
            {"episode_id": "a-2", "status": "compiled", "selected_intervention_id": "U02"},
            {"episode_id": "b-1", "status": "compiled", "selected_intervention_id": "U03"},
        ]
    }
    original = copy.deepcopy(rows)
    weighted, audit = build_mix(rows, report, normalize_case_mass=True)
    assert rows == original
    assert [item["messages"] for item in weighted] == [item["messages"] for item in rows]
    ranges = audit["weighted_case_mass_range_by_dataset"]["VideoSham"]
    assert ranges["minimum"] == ranges["maximum"]
    assert all(
        item["metadata"]["sampling_policy_id"].endswith("_v2")
        for item in weighted
    )


def test_balancer_ignores_compiled_episodes_removed_by_frontier_selection() -> None:
    rows = [_row("kept", "case-a")]
    report = {
        "results": [
            {"episode_id": "kept", "status": "compiled", "selected_intervention_id": "U01"},
            {"episode_id": "removed", "status": "compiled", "selected_intervention_id": "U02"},
        ]
    }
    weighted, audit = build_mix(rows, report, normalize_case_mass=True)
    assert len(weighted) == 1
    assert audit["output_rows"] == 1


def test_within_case_tempering_reduces_weight_concentration() -> None:
    rows = [
        _row("a", "case-a", "challenge"),
        _row("b", "case-a", "fresh_measurement"),
        _row("c", "case-b", "challenge"),
        _row("d", "case-b", "challenge"),
    ]
    rows[0]["metadata"]["learning_value_weight"] = 9.0
    report = {
        "results": [
            {"episode_id": episode, "status": "compiled", "selected_intervention_id": "U01"}
            for episode in ("a", "b", "c", "d")
        ]
    }
    _plain, plain_audit = build_mix(
        rows, report, normalize_case_mass=True, case_tempering_power=1.0
    )
    _tempered, tempered_audit = build_mix(
        rows, report, normalize_case_mass=True, case_tempering_power=0.5
    )
    assert (
        tempered_audit["weight_distribution"]["maximum"]
        < plain_audit["weight_distribution"]["maximum"]
    )
    assert (
        tempered_audit["weight_distribution"]["effective_sample_fraction"]
        > plain_audit["weight_distribution"]["effective_sample_fraction"]
    )


def test_causal_frontier_keeps_backbone_and_caps_repeated_templates() -> None:
    base = [_row("base", "case-a")]
    base[0]["messages"][-1]["content"] = "Inspect evidence 1 before binding."
    lattice = []
    for index, text in enumerate(
        [
            "Inspect evidence 1 before binding.",
            "Compare evidence 2 before binding.",
            "Compare evidence 3 before binding.",
            "Compare evidence 4 before binding.",
        ],
        start=1,
    ):
        row = _row(f"lattice-{index}", f"case-{index}")
        row["metadata"]["row_id"] = f"row-{index}"
        row["messages"][-1]["content"] = text
        lattice.append(row)

    selected, report = select_frontier(
        base, lattice, maximum_contexts_per_dataset_template=2
    )
    # The base is always retained; the exact duplicate is removed.  Number
    # normalization makes the other three one template, so only two are added.
    assert len(selected) == 3
    assert selected[0]["metadata"]["augmentation_role"] == "complete_episode_backbone"
    assert all(
        item["metadata"]["augmentation_role"] == "novel_causal_frontier"
        for item in selected[1:]
    )
    assert report["excluded_lattice_rows"] == {
        "exact_text_already_represented": 1,
        "template_context_budget_reached": 1,
    }
    assert report["messages_unchanged"] is True


def test_partial_frontier_is_not_misreported_as_incomplete_episode_failure() -> None:
    row = _row("partial", "case-a", "challenge")
    row["metadata"].update(
        {
            "row_id": "partial:01:challenge",
            "augmentation_role": "novel_causal_frontier",
        }
    )
    row["tools"] = []
    row["messages"] = [
        {
            "role": "user",
            "content": '{"current_harness_state":{"eligible_tools":["challenge_evidence","stop"]}}',
        },
        {
        "role": "assistant",
        "content": "The visible evidence is inconsistent, so I will challenge it before binding.",
        "tool_calls": [
            {
                "function": {
                    "name": "challenge_evidence",
                    "arguments": {
                        "evidence_ids": ["EA-1"],
                        "defect_codes": ["scope_mismatch"],
                        "target_fields": ["direction"],
                        "requested_check": "remeasure_evidence",
                    },
                }
            }
        ],
        },
    ]
    output, result = build_contrast([row])
    assert len(output) == 1
    assert result["report"]["construction_failures"] == []
    assert (
        result["report"][
            "partial_frontier_episodes_skipped_for_episode_level_pairs"
        ]
        == 1
    )


def test_same_case_partial_variants_form_stop_challenge_contrast() -> None:
    challenge = _row("partial-a", "case-a", "challenge")
    challenge["metadata"].update(
        {"row_id": "partial-a:01", "augmentation_role": "novel_causal_frontier"}
    )
    challenge["messages"] = [
        {
            "role": "user",
            "content": '{"current_harness_state":{"eligible_tools":["challenge_evidence","stop"]}}',
        },
        {
            "role": "assistant",
            "content": "The interval is outside the current scope, so I will challenge this evidence.",
            "tool_calls": [
                {
                    "function": {
                        "name": "challenge_evidence",
                        "arguments": {
                            "evidence_ids": ["EA-1"],
                            "defect_codes": ["scope_mismatch"],
                            "target_fields": ["direction"],
                            "requested_check": "remeasure_evidence",
                        },
                    }
                }
            ],
        },
    ]
    challenge["tools"] = []
    stop = copy.deepcopy(challenge)
    stop["metadata"].update(
        {
            "episode_id": "partial-b",
            "row_id": "partial-b:01",
            "causal_recovery_stage": "terminal_verification",
        }
    )
    stop["messages"][-1] = {
        "role": "assistant",
        "content": "All visible evidence is consistent, so the episode can now stop.",
        "tool_calls": [{"function": {"name": "stop", "arguments": {}}}],
    }
    output, result = build_contrast([challenge, stop])
    assert len(output) == 2
    assert result["report"]["construction_failures"] == []
    assert result["report"]["same_case_cross_variant_stop_vs_challenge_groups"] == 1
    assert all(row["metadata"].get("semantic_contrast_groups") for row in output)
    integrity = audit_contrast(output, result["groups"])
    assert integrity["passed"] is True
