from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_recovery_reasoning_quality_v2 import audit_rows  # noqa: E402
from audit_recovery_semantic_transparency import compare  # noqa: E402


BASE = ROOT / "outputs/student_external_contract_v1"
BEFORE = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike"
AFTER = BASE / "recovery_batch_candidate_v20_abstention_semantic_transparency"
PORTABLE = BASE / "recovery_sft_semantic_contrast_portable_v2_abstention55"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v20_makes_abstention_and_adjudication_semantics_explicit() -> None:
    report = compare(
        _jsonl(BEFORE / "recovery_sft_candidate.jsonl"),
        _jsonl(AFTER / "recovery_sft_candidate.jsonl"),
    )
    assert report["executed_action_signatures_unchanged"] is True
    assert report["passed"] is True
    assert report["before"]["preferred_c_semantics_explicit"] == 0
    assert report["after"]["preferred_c_rows"] == 110
    assert report["after"]["preferred_c_semantics_explicit"] == 110
    assert report["after"]["retained_c_order_inconsistency_explained"] == 8
    assert report["after"]["redundancy_explained_without_false_contradiction"] == 2
    assert report["after"]["claim_conflict_value_and_binding_consequence_explicit"] == 29
    assert report["after"]["adjudication_partition_semantically_rendered"] == 12


def test_v20_and_portable_view_pass_strict_reasoning_checks() -> None:
    canonical = audit_rows(_jsonl(AFTER / "recovery_sft_candidate.jsonl"))
    portable = audit_rows(_jsonl(PORTABLE / "recovery_sft_portable_weighted.jsonl"))
    assert canonical["passed"] is True
    assert canonical["metrics"]["row_passed"] == 415
    assert portable["passed"] is True
    assert portable["metrics"]["row_passed"] == 415


def test_v20_readiness_stays_conservative() -> None:
    readiness = _load(
        BASE / "production_readiness_audit_v5_abstention55/readiness_report.json"
    )
    contrast = _load(PORTABLE / "semantic_contrast_report.json")
    portability = _load(PORTABLE / "portability_audit.json")
    assert readiness["semantic_transparency"]["passed"] is True
    assert readiness["semantic_contrast_view"]["contrast_groups"] == 67
    assert readiness["blind_review"]["aggregation_status"] == "awaiting_scores"
    assert readiness["training_qualified"] is False
    assert contrast["semantic_transparency_audit_passed"] is True
    assert portability["all_semantic_equivalence_checks_passed"] is True
    assert portability["all_forbidden_identifier_hits_zero"] is True

