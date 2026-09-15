from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from audit_local_trajectory_inventory import build_inventory
from audit_recovery_production_readiness import build_readiness_report
from audit_recovery_production_readiness_v2 import build_readiness_report_v2
from build_expanded_recovery_manifest import build_expanded_manifest, current6_case_ids
from export_blind_recovery_review_packets import export_packets


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/student_external_contract_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_inventory_never_relabels_accepted_cases_as_sealed() -> None:
    report = build_inventory(
        ROOT / "outputs",
        BASE / "recovery_batch_manifest_v1/manifest.json",
        BASE / "cross_dataset_blind_canary_6episodes_v3/semantic_audit.json",
    )
    assert report["unique_cases"] >= 61
    assert report["sealed_holdout_available"] is False
    assert all(item["sealed_holdout_eligible"] is False for item in report["cases"])
    assert sum(
        "recovery_v6_development" in item["exposure_tags"]
        for item in report["cases"]
    ) == 28
    assert (
        report["dataset_summary"]["VCSL"][
            "maximum_entity_disjoint_exposed_cases"
        ]
        < report["dataset_summary"]["VCSL"]["accepted_exposed_cases"]
    )


def test_blind_packet_removes_identity_and_separates_private_key(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    manifest = export_packets(
        BASE / "recovery_batch_candidate_v6/recovery_episodes.jsonl",
        ROOT / "rubric/unified_trajectory_quality_rubric_v9.json",
        output,
    )
    assert manifest["episodes"] == 28
    assert manifest["human_review_completed"] is False
    public = (output / "public_share_only/review_packets.jsonl").read_text(
        encoding="utf-8"
    )
    for pattern in (
        r"VideoSham",
        r"VidDiffBench",
        r"VCSL",
        r"videosham-\d+",
        r"viddiff-surgery_\d+",
        r"vcsl-[0-9a-f]{32}",
        r"[A-Z]:\\",
        r"private_intervention",
        r"verifier_result",
    ):
        assert re.search(pattern, public, flags=re.I) is None
    assert (output / "private_do_not_share_with_judges/review_key.json").exists()

    def review_ids(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [row["review_id"] for row in csv.DictReader(handle)]

    order_a = review_ids(output / "public_share_only/reviewer_A.csv")
    order_b = review_ids(output / "public_share_only/reviewer_B.csv")
    assert set(order_a) == set(order_b)
    assert len(order_a) == 28
    assert order_a != order_b


def test_readiness_distinguishes_clean_development_from_training_qualified() -> None:
    report = build_readiness_report(
        _load(BASE / "production_readiness_audit_v1/local_trajectory_inventory.json"),
        _load(BASE / "recovery_batch_candidate_v6/compile_report.json"),
        _load(BASE / "recovery_blind_review_packet_v2/manifest.json"),
        _load(ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"),
    )
    assert report["development_batch"]["structurally_clean"] is True
    assert report["training_qualified"] is False
    assert report["dataset_checks"]["VidDiffBench"][
        "recovery_case_minimum_passed"
    ] is False
    assert "no_sealed_entity_disjoint_holdout" in report["qualification_blockers"]


def test_expanded_manifest_is_entity_disjoint_and_excludes_current6() -> None:
    current6_path = BASE / "cross_dataset_blind_canary_6episodes_v3/semantic_audit.json"
    manifest = build_expanded_manifest(current6_path)
    assert manifest["datasets"] == {
        "VideoSham": 16,
        "VCSL": 18,
        "VidDiffBench": 21,
    }
    assert len(manifest["jobs"]) == 55
    assert manifest["current6_reused"] is False
    assert {
        job["case_id"].lower() for job in manifest["jobs"]
    }.isdisjoint(current6_case_ids(current6_path))
    vcsl_entities: set[str] = set()
    for job in manifest["jobs"]:
        if job["dataset"] != "VCSL":
            continue
        entities = set(job["case_id"].removeprefix("vcsl-").split("-"))
        assert entities.isdisjoint(vcsl_entities)
        vcsl_entities.update(entities)
    verified = "outputs/batch/viddiff_verified_seed_20260818_accepted_8/"
    assert all(
        job["checkpoint"]["file"].startswith(verified)
        for job in manifest["jobs"]
        if job["case_id"]
        in {
            "viddiff-surgery_116",
            "viddiff-surgery_131",
            "viddiff-surgery_155",
            "viddiff-surgery_158",
            "viddiff-surgery_28",
            "viddiff-surgery_57",
            "viddiff-surgery_80",
            "viddiff-surgery_87",
        }
    )
    assert manifest["availability_excluded"] == []


def test_expanded_checkpoint_portfolio_batch_meets_local_recovery_floor() -> None:
    report = _load(
        BASE
        / "recovery_batch_candidate_v9_expanded_checkpoint_portfolio/compile_report.json"
    )
    assert report["compiled_episodes"] == report["jobs_selected"] == 46
    assert report["datasets_selected"] == {
        "VideoSham": 16,
        "VCSL": 18,
        "VidDiffBench": 12,
    }
    assert report["terminal_success"] == 46
    assert report["fresh_evidence_consumed"] == 46
    assert report["independent_fields_preserved"] == 46
    assert report["counterfactuals_passed"] == 46
    assert report["student_visible_leakage_episodes"] == 0
    assert report["reasoning_quality_audit"]["passed"] is True
    assert report["portfolio_recovered_episodes"] == 1
    assert set(report["selected_intervention_counts"]) == {
        f"U{index:02d}_{suffix}"
        for index, suffix in (
            (1, "active_wrong_interval"),
            (2, "source_suspect_role_swap"),
            (3, "confidence_rationale_contradiction"),
            (4, "semantically_incompatible_binding"),
            (5, "conflicting_active_measurements"),
            (6, "stale_active_evidence"),
            (7, "new_result_claim_conflict"),
            (8, "partial_interval_result"),
            (9, "empty_tool_result"),
            (10, "insufficient_evidence_abstain"),
        )
    }


def test_expanded55_readiness_keeps_automated_success_distinct_from_production() -> None:
    batch = BASE / "recovery_batch_candidate_v13_expanded_catalog_u11_compact"
    packet = BASE / "recovery_blind_review_packet_v4_expanded55"
    portable = BASE / "recovery_sft_portable_v4_expanded55"
    report = build_readiness_report_v2(
        _load(BASE / "production_readiness_audit_v1/local_trajectory_inventory.json"),
        _load(batch / "compile_report.json"),
        _load(batch / "utqr_episode_scorecards.json"),
        _load(batch / "utqr_falsification_report.json"),
        _load(packet / "manifest.json"),
        _load(packet / "private_do_not_share_with_judges/aggregation_status.json"),
        _load(portable / "portability_audit.json"),
        _load(portable / "reasoning_quality_audit.json"),
        _load(ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"),
    )
    assert report["development_batch"]["structurally_clean"] is True
    assert report["development_batch"]["dataset_counts"] == {
        "VideoSham": 16,
        "VCSL": 18,
        "VidDiffBench": 21,
    }
    assert all(
        item["automated_score_floor_passed"]
        for item in report["dataset_checks"].values()
    )
    assert all(
        not item["entity_disjoint_count_passed"]
        for item in report["dataset_checks"].values()
    )
    assert report["blind_review"]["aggregation_status"] == "awaiting_scores"
    assert report["training_qualified"] is False
