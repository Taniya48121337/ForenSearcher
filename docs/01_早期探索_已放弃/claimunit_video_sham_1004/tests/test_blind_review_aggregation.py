from __future__ import annotations

from aggregate_blind_recovery_reviews import aggregate_reviews


DIMENSIONS = [
    "prospective_reasoning",
    "visible_evidence_grounding",
    "belief_update",
    "method_fit",
    "locality",
    "naturalness",
]
IDS = ["review-001", "review-002", "review-003"]
KEY = {
    review_id: {
        "episode_id": f"episode-{index}",
        "dataset": dataset,
    }
    for index, (review_id, dataset) in enumerate(
        zip(IDS, ["VideoSham", "VCSL", "VidDiffBench"]), start=1
    )
}
AUTOMATED = {review_id: 92.0 for review_id in IDS}


def row(score: str = "4.5", *, hard: str = "no", accept: str = "yes") -> dict[str, str]:
    return {
        **{dimension: score for dimension in DIMENSIONS},
        "visible_hard_failure": hard,
        "accept_for_sft": accept,
        "strongest_turn": "A2",
        "weakest_turn": "A1",
        "required_revision": "",
        "reviewer_notes": "A2 uses the new atom precisely; A1 is less specific but still grounded.",
    }


def aggregate(
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
    adjudication: dict[str, dict[str, str]] | None = None,
) -> dict:
    return aggregate_reviews(
        IDS,
        left,
        right,
        adjudication or {},
        KEY,
        AUTOMATED,
        DIMENSIONS,
        1.0,
        4.0,
        0.9,
        90.0,
        {review_id: 3 for review_id in IDS},
        ["VideoSham", "VCSL", "VidDiffBench"],
        30,
        {review_id: True for review_id in IDS},
        90.0,
    )


def test_blank_forms_remain_awaiting_scores() -> None:
    report = aggregate(
        {review_id: row("") for review_id in IDS},
        {review_id: row("") for review_id in IDS},
    )
    assert report["status"] == "awaiting_scores"
    assert report["review_completed"] is False
    assert report["blind_review_passed"] is False
    assert report["missing_or_invalid_review_ids"] == IDS


def test_complete_high_scores_pass_blind_review_but_not_training_qualification() -> None:
    report = aggregate(
        {review_id: row("4.5") for review_id in IDS},
        {review_id: row("4.5") for review_id in IDS},
    )
    assert report["status"] == "complete"
    assert report["review_completed"] is True
    assert report["blind_review_passed"] is True
    assert report["training_qualified"] is False
    assert all(
        not item["passed"]
        for item in report["dataset_packet_count_readiness"].values()
    )
    assert all(item["accepted_for_sft"] for item in report["items"])
    assert all(
        summary["acceptance_rate"] == 1.0 and summary["score_p10"] == 99.2
        for summary in report["dataset_summary"].values()
    )


def test_low_dimension_or_hard_failure_rejects_episode() -> None:
    left = {review_id: row("4.5") for review_id in IDS}
    right = {review_id: row("4.5") for review_id in IDS}
    left["review-001"]["naturalness"] = "3"
    right["review-001"]["naturalness"] = "3"
    left["review-002"]["visible_hard_failure"] = "yes"
    report = aggregate(left, right)
    rejected = {
        item["review_id"] for item in report["items"] if not item["accepted_for_sft"]
    }
    assert report["status"] == "complete"
    assert report["blind_review_passed"] is False
    assert rejected == {"review-001", "review-002"}


def test_large_disagreement_requires_then_accepts_explicit_adjudication() -> None:
    left = {review_id: row("4.5") for review_id in IDS}
    right = {review_id: row("4.5") for review_id in IDS}
    left["review-003"]["naturalness"] = "1"
    right["review-003"]["naturalness"] = "5"
    pending = aggregate(left, right)
    assert pending["status"] == "awaiting_adjudication"
    assert pending["review_completed"] is False
    assert pending["unresolved_disagreements"] == [
        {"review_id": "review-003", "dimension": "naturalness"}
    ]

    adjudicated = aggregate(
        left,
        right,
        {
            "review-003:naturalness": {
                "adjudicated_score": "4",
                "adjudication_reason": "Transcript evidence supports the upper anchor.",
            }
        },
    )
    assert adjudicated["status"] == "complete"
    assert adjudicated["review_completed"] is True
    assert adjudicated["blind_review_passed"] is True


def test_numeric_rubber_stamp_without_auditable_turn_rationale_is_incomplete() -> None:
    left = {review_id: row("5") for review_id in IDS}
    right = {review_id: row("5") for review_id in IDS}
    left["review-001"]["strongest_turn"] = "turn 99"
    right["review-002"]["weakest_turn"] = "A2"
    right["review-003"]["reviewer_notes"] = "great"
    report = aggregate(left, right)
    assert report["status"] == "awaiting_scores"
    assert report["review_completed"] is False
    assert report["missing_or_invalid_review_ids"] == IDS
    assert report["items"][0]["review_evidence"]["reviewer_A"]["errors"] == [
        "invalid_or_missing_strongest_turn"
    ]


def test_automated_hard_gate_failure_cannot_be_overridden_by_human_scores() -> None:
    left = {review_id: row("5") for review_id in IDS}
    right = {review_id: row("5") for review_id in IDS}
    report = aggregate_reviews(
        IDS,
        left,
        right,
        {},
        KEY,
        AUTOMATED,
        DIMENSIONS,
        1.0,
        4.0,
        0.9,
        90.0,
        {review_id: 3 for review_id in IDS},
        ["VideoSham", "VCSL", "VidDiffBench"],
        30,
        {"review-001": False, "review-002": True, "review-003": True},
        90.0,
    )
    first = report["items"][0]
    assert first["H8_episode_passed"] is True
    assert first["automated_H1_to_H7_passed"] is False
    assert first["accepted_for_sft"] is False
    assert report["blind_review_passed"] is False


def test_duplicate_or_unneeded_adjudication_is_not_silently_accepted() -> None:
    report = aggregate_reviews(
        IDS,
        {review_id: row("4.5") for review_id in IDS},
        {review_id: row("4.5") for review_id in IDS},
        {
            "review-001:naturalness": {
                "adjudicated_score": "5",
                "adjudication_reason": "No disagreement exists for this row.",
            }
        },
        KEY,
        AUTOMATED,
        DIMENSIONS,
        1.0,
        4.0,
        0.9,
        90.0,
        {review_id: 3 for review_id in IDS},
        ["VideoSham", "VCSL", "VidDiffBench"],
        30,
        {review_id: True for review_id in IDS},
        90.0,
        [],
        [],
        ["review-001:naturalness"],
    )
    assert report["status"] == "awaiting_scores"
    assert report["review_completed"] is False
    assert report["structural_errors"]["adjudication_duplicate_keys"] == [
        "review-001:naturalness"
    ]
    assert report["structural_errors"]["adjudication_unknown_or_unneeded_keys"] == [
        "review-001:naturalness"
    ]
