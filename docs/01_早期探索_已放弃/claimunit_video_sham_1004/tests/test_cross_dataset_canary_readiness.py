from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_cross_dataset_canary_readiness import audit  # noqa: E402


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_canary_readiness_keeps_human_and_population_claims_pending(tmp_path: Path) -> None:
    package = _write(
        tmp_path / "package.json",
        {
            "ready_for_training": True,
            "rows": 2,
            "privacy_leaks": [],
            "semantic_audit": {"coverage_missing": []},
            "validation": {
                "all_passed": True,
                "duplicate_row_ids": [],
                "exact_reasoning_duplicates": 0,
                "dataset_rows": {"VidDiffBench": 1, "VideoSham": 1, "VCSL": 1},
                "mode_rows": {"clean": 1, "repair": 1},
                "tool_rows": {"inspect_media": 1},
            },
        },
    )
    semantic = _write(
        tmp_path / "semantic.json", {"all_passed": True, "episodes": 1, "turns": 2}
    )
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps(
            {
                "blind_id": "BLIND-001",
                "turns": [],
                "review_form": {
                    "dimension_scores": {"grounding": None},
                    "hard_fail": None,
                    "keep_for_sft": None,
                },
            }
        ),
        encoding="utf-8",
    )
    ablation = _write(
        tmp_path / "ablation.json",
        {
            "decisions_frozen_before_oracle_load": True,
            "external_service_calls": 0,
            "generic_routes_only": {"decision": "a", "posthoc_correct": False},
            "typed_proof_route": {"decision": "c", "posthoc_correct": True},
            "interpretation_limit": "Mechanistic only, not a population estimate.",
        },
    )
    report = audit(package, semantic, packet, ablation)
    assert report["machine_canary_ready"] is True
    assert report["summary"]["human_reviews_pending"] == 1
    assert report["ready_to_claim_population_improvement"] is False


def test_canary_readiness_rejects_forbidden_blind_packet_content(tmp_path: Path) -> None:
    package = _write(
        tmp_path / "package.json",
        {
            "ready_for_training": True,
            "rows": 1,
            "privacy_leaks": [],
            "semantic_audit": {"coverage_missing": []},
            "validation": {
                "all_passed": True,
                "duplicate_row_ids": [],
                "exact_reasoning_duplicates": 0,
                "dataset_rows": {"VidDiffBench": 1, "VideoSham": 1, "VCSL": 1},
                "mode_rows": {"clean": 1, "repair": 1},
                "tool_rows": {"stop": 1},
            },
        },
    )
    semantic = _write(
        tmp_path / "semantic.json", {"all_passed": True, "episodes": 1, "turns": 1}
    )
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps({"blind_id": "BLIND-001", "turns": [{"reasoning": "use verifier"}]}),
        encoding="utf-8",
    )
    ablation = _write(
        tmp_path / "ablation.json",
        {
            "decisions_frozen_before_oracle_load": True,
            "external_service_calls": 0,
            "generic_routes_only": {"decision": "a", "posthoc_correct": False},
            "typed_proof_route": {"decision": "c", "posthoc_correct": True},
            "interpretation_limit": "Not a population estimate.",
        },
    )
    report = audit(package, semantic, packet, ablation)
    assert report["machine_canary_ready"] is False
    assert "verifier" in report["review_packet_checks"]["forbidden_term_hits"]
