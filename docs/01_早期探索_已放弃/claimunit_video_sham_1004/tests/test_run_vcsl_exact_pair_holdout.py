from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_vcsl_exact_pair_holdout import build_seed_row, score_episode  # noqa: E402
from shadow_execute_generated_actions import _jsonl  # noqa: E402


SFT = (
    ROOT
    / "outputs/student_external_contract_v1/utqr_v6_sft_gain_experiment_v4_bind_compact"
    / "decision_sufficient_state_v6_executable_affordances/validation/sft_all.jsonl"
)
MANIFEST = (
    ROOT
    / "outputs/student_external_contract_v1/control_plane_falsification_v1/manifests/sealed_cases.json"
)


def _template() -> dict:
    return next(
        row
        for row in _jsonl(SFT)
        if row["metadata"]["dataset"] == "VCSL"
        and row["metadata"]["row_id"].endswith(":clean:1")
    )


def test_seed_contains_no_oracle_or_assistant_target() -> None:
    candidate = json.loads(MANIFEST.read_text(encoding="utf-8"))["selected_cases"]["VCSL"][0]
    row = build_seed_row(_template(), candidate)
    serialized = json.dumps(row, ensure_ascii=False).lower()
    assert "oracle_boxes" not in serialized
    assert '"role": "assistant"' not in serialized
    assert row["metadata"]["case_id"] == candidate["case_id"]


def test_score_empty_episode_is_zero_and_nonterminal() -> None:
    candidate = {
        "case_id": "vcsl-a-b",
        "query_id": "a",
        "reference_id": "b",
    }
    score = score_episode(
        {"policy_view_arm": "full", "terminal_success": False, "final_state": {}},
        candidate,
        [[0, 0, 1, 1]],
    )
    assert score["terminal_success"] is False
    assert score["hidden_oracle_field_accuracy"] == 0.0
    assert score["binding_accuracy"] == 0.0
