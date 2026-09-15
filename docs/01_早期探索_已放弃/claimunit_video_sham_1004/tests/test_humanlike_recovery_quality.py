from __future__ import annotations

import json
from pathlib import Path

from audit_recovery_reasoning_quality_v2 import audit_rows
from compare_recovery_humanlike_quality import compare
from shadow_execute_generated_actions import _jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/student_external_contract_v1"
BEFORE = BASE / "recovery_batch_candidate_v13_expanded_catalog_u11_compact"
AFTER = BASE / "recovery_batch_candidate_v16_edge_aware_humanlike"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_strict_h7_detects_old_text_failures_and_accepts_v16() -> None:
    before = audit_rows(_jsonl(BEFORE / "recovery_sft_candidate.jsonl"))
    after = audit_rows(_jsonl(AFTER / "recovery_sft_candidate.jsonl"))
    assert before["passed"] is False
    assert before["metrics"]["strict_additional_check_passed_rows"] == 296
    assert before["metrics"]["meta_process_narration_hits"] == 75
    assert before["metrics"]["unbalanced_inline_serialization_hits"] == 44
    assert after["passed"] is True
    assert after["metrics"]["row_passed"] == 415
    assert after["metrics"]["strict_additional_check_passed_rows"] == 415


def test_reasoning_fix_does_not_change_executed_action_targets() -> None:
    report = compare(
        _jsonl(BEFORE / "recovery_sft_candidate.jsonl"),
        _jsonl(AFTER / "recovery_sft_candidate.jsonl"),
    )
    assert report["executed_action_signatures_unchanged"] is True
    assert report["improvement_established"] is True
    assert report["edge_aware_binding_diagnoses"] == {
        "episodes": 10,
        "by_dataset": {"VCSL": 6, "VidDiffBench": 4},
        "meaning": (
            "The trajectory identifies an invalid evidence-field support edge while "
            "preserving the EvidenceAtom's validity for its advertised fields."
        ),
    }


def test_edge_misbinding_explanation_preserves_valid_atom_semantics() -> None:
    rows = _jsonl(AFTER / "recovery_sft_candidate.jsonl")
    challenges = {
        row["metadata"]["episode_id"]: row["messages"][-1]["content"]
        for row in rows
        if row["metadata"]["causal_recovery_stage"] == "challenge"
    }
    for episode_id in ("expanded-recovery-017", "expanded-recovery-039"):
        text = challenges[episode_id]
        assert "support edge is invalid" in text
        assert "may remain valid for its own fields" in text


def test_v16_portable_and_readiness_reports_remain_conservative() -> None:
    portable = _load(
        BASE / "recovery_sft_portable_v6_edgeaware55/reasoning_quality_audit_v2.json"
    )
    readiness = _load(
        BASE / "production_readiness_audit_v3_edgeaware55/readiness_report.json"
    )
    assert portable["passed"] is True
    assert readiness["development_batch"]["structurally_clean"] is True
    assert readiness["humanlike_quality_improvement"][
        "established_without_action_target_changes"
    ] is True
    assert readiness["blind_review"]["aggregation_status"] == "awaiting_scores"
    assert readiness["training_qualified"] is False
