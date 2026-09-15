from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compile_causal_recovery_sft import (  # noqa: E402
    DEFAULT_SFT,
    compile_recovery_episodes,
)
from shadow_execute_generated_actions import _jsonl  # noqa: E402


def test_complete_recovery_compiler_executes_and_consumes_fresh_evidence() -> None:
    source_rows = {
        str(row["metadata"]["row_id"]): row for row in _jsonl(DEFAULT_SFT)
    }
    rows, episodes, sidecars, report = compile_recovery_episodes(source_rows)
    assert len(episodes) == len(sidecars) == 11
    assert len(rows) == 102
    assert report["datasets"] == {
        "VideoSham": 3,
        "VCSL": 3,
        "VidDiffBench": 5,
    }
    assert report["terminal_success"] == 11
    assert report["fresh_evidence_consumed"] == 11
    assert report["independent_fields_preserved"] == 11
    assert report["challenged_evidence_rebound"] == 0
    assert report["clean_control_suppresses_challenge"] == 11
    assert report["defect_target_moves_with_atom"] == 11
    assert report["fresh_removal_blocks_repair"] == 11
    assert report["student_visible_leakage_episodes"] == 0
    for episode in episodes:
        assert episode["terminal"] is True
        assert episode["fresh_evidence_consumed"] is True
        assert episode["independent_fields_preserved"] is True
        assert episode["challenged_evidence_rebound"] is False
    assert all(row["metadata"]["target_was_executed"] for row in rows)
    assert all(row["metadata"]["structural_candidate_only"] for row in rows)
