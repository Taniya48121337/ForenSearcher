from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_viddiff30_four_arm import score_episode  # noqa: E402


def test_empty_episode_scores_zero_without_oracle_success() -> None:
    item = {
        "sample_key": "surgery_test",
        "task": {"comparison_statement": "visible statement"},
    }
    score = score_episode(
        {"policy_view_arm": "full", "terminal_success": False, "final_state": {}},
        item,
        "a",
    )
    assert score["terminal_and_oracle_correct"] is False
    assert score["oracle_direction_correct"] is False
    assert score["binding_accuracy"] == 0.0
    assert score["claim_check_rate"] == 0.0
