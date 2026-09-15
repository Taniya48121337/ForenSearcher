from __future__ import annotations

from aggregate_blind_recovery_reviews import aggregate_reviews
from export_blind_recovery_delta_review_packets import (
    apply_state_delta,
    select_stratified,
    state_delta,
    wilson_lower_bound,
)


BASE_DIMENSIONS = [
    "prospective_reasoning",
    "visible_evidence_grounding",
    "belief_update",
    "method_fit",
    "locality",
    "naturalness",
]
SUPPLEMENTAL = [
    "anti_shortcut_semantic_identifiability",
    "per_step_sft_pedagogical_value",
]


def _review_row(score: str = "4.5") -> dict[str, str]:
    return {
        **{dimension: score for dimension in [*BASE_DIMENSIONS, *SUPPLEMENTAL]},
        "visible_hard_failure": "no",
        "accept_for_sft": "yes",
        "strongest_turn": "A2",
        "weakest_turn": "A1",
        "required_revision": "",
        "reviewer_notes": "A2 uses the fresh visible atom; A1 identifies the defect precisely.",
    }


def test_delta_records_only_changed_paths_and_is_reconstructable() -> None:
    before = {"claim": {"x": 1, "stable": 7}, "tools": ["measure"]}
    after = {"claim": {"x": 2, "stable": 7, "support": "EA-2"}, "tools": ["stop"]}
    changes = state_delta(before, after)
    assert changes == [
        {"change": "added", "path": "/claim/support", "after": "EA-2"},
        {"change": "replaced", "path": "/claim/x", "before": 1, "after": 2},
        {
            "change": "replaced",
            "path": "/tools",
            "before": ["measure"],
            "after": ["stop"],
        },
    ]
    assert apply_state_delta(before, changes) == after


def test_stratified_selection_covers_every_family_without_using_scores() -> None:
    episodes = []
    sidecars = []
    for dataset in ("A", "B", "C"):
        for family in ("U1", "U2", "U3"):
            for case in range(4):
                episode_id = f"{dataset}-{family}-{case}"
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "dataset": dataset,
                        "case_id": f"{dataset}-case-{case}",
                    }
                )
                sidecars.append(
                    {"episode_id": episode_id, "intervention_id": family}
                )
    selected, report = select_stratified(episodes, sidecars, 6, 19)
    assert len(selected) == 18
    assert all(item["selected_episodes"] == 6 for item in report.values())
    assert all(set(item["intervention_quotas"]) == {"U1", "U2", "U3"} for item in report.values())
    assert all(sum(item["intervention_quotas"].values()) == 6 for item in report.values())


def test_wilson_rule_makes_one_rejection_fail_a_35_item_stratum() -> None:
    ids = [f"review-{index:03d}" for index in range(105)]
    datasets = ["VideoSham"] * 35 + ["VCSL"] * 35 + ["VidDiffBench"] * 35
    key = {
        review_id: {"episode_id": f"episode-{index}", "dataset": dataset}
        for index, (review_id, dataset) in enumerate(zip(ids, datasets))
    }
    automated = {review_id: 92.0 for review_id in ids}
    left = {review_id: _review_row() for review_id in ids}
    right = {review_id: _review_row() for review_id in ids}

    def run() -> dict:
        return aggregate_reviews(
            ids,
            left,
            right,
            {},
            key,
            automated,
            BASE_DIMENSIONS,
            1.0,
            4.0,
            0.9,
            90.0,
            {review_id: 2 for review_id in ids},
            ["VideoSham", "VCSL", "VidDiffBench"],
            None,
            {review_id: True for review_id in ids},
            90.0,
            supplemental_dimensions=SUPPLEMENTAL,
            minimum_supplemental_dimension_mean=4.0,
            minimum_acceptance_wilson_lower_bound=0.9,
        )

    perfect = run()
    assert wilson_lower_bound(35, 35) > 0.9
    assert perfect["blind_review_passed"] is True

    left["review-000"][SUPPLEMENTAL[0]] = "3"
    right["review-000"][SUPPLEMENTAL[0]] = "3"
    rejected = run()
    summary = rejected["dataset_summary"]["VideoSham"]
    assert summary["accepted"] == 34
    assert summary["acceptance_rate_passed"] is True
    assert summary["acceptance_wilson95_lower_bound_passed"] is False
    assert rejected["blind_review_passed"] is False
