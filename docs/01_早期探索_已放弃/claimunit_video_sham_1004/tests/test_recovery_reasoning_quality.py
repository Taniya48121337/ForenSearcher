from __future__ import annotations

from pathlib import Path

from audit_recovery_reasoning_quality import audit_rows
from shadow_execute_generated_actions import _jsonl


ROOT = Path(__file__).resolve().parents[1]
FINAL_BATCH = ROOT / "outputs/student_external_contract_v1/recovery_batch_candidate_v6"


def test_executed_batch_reasoning_passes_high_bar_local_audit() -> None:
    rows = _jsonl(FINAL_BATCH / "recovery_sft_candidate.jsonl")
    report = audit_rows(rows)
    assert len(rows) == 195
    assert report["passed"] is True
    assert report["metrics"]["row_passed"] == 195
    assert report["metrics"]["exact_duplicate_fraction"] <= 0.05
    assert report["metrics"]["normalized_template_duplicate_fraction"] <= 0.35


def test_claim_revision_is_sparse_executed_and_separate_from_binding() -> None:
    rows = _jsonl(FINAL_BATCH / "recovery_sft_candidate.jsonl")
    revisions = [
        row
        for row in rows
        if row["metadata"]["causal_recovery_stage"] == "claim_revision"
    ]
    assert len(revisions) == 2
    assert all(row["metadata"]["target_tool"] == "revise_claim_fields" for row in revisions)
    for row in revisions:
        assistant = row["messages"][-1]
        arguments = assistant["tool_calls"][0]["function"]["arguments"]
        assert [item["field"] for item in arguments["field_updates"]] == [
            "affected_suspect_interval_s"
        ]
        assert len(arguments["field_updates"][0]["value"]) == 2
        assert "binding" in assistant["content"].lower()
        assert "commit" in assistant["content"].lower()
