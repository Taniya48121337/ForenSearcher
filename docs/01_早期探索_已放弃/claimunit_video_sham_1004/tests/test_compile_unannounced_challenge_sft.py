from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compile_unannounced_challenge_sft import (  # noqa: E402
    DEFAULT_SFT,
    compile_candidates,
)
from shadow_execute_generated_actions import _jsonl  # noqa: E402


def test_compile_candidates_executes_all_targets_without_visible_intervention_leakage() -> None:
    source_rows = {
        str(row["metadata"]["row_id"]): row for row in _jsonl(DEFAULT_SFT)
    }
    rows, sidecars, summary = compile_candidates(source_rows)
    assert len(rows) == len(sidecars) == 11
    assert summary["datasets"] == {
        "VideoSham": 3,
        "VCSL": 3,
        "VidDiffBench": 5,
    }
    assert summary["target_schema_valid"] == 11
    assert summary["target_executed"] == 11
    assert summary["student_visible_leakage_rows"] == 0
    for row, sidecar in zip(rows, sidecars):
        assert row["metadata"]["target_was_executed"] is True
        assert row["metadata"]["structural_candidate_only"] is True
        assert row["messages"][-1]["tool_calls"][0]["function"]["name"] == (
            "challenge_evidence"
        )
        assert sidecar["postaction_open_challenges"]
        assert sidecar["student_visible_leakage_hits"] == []
