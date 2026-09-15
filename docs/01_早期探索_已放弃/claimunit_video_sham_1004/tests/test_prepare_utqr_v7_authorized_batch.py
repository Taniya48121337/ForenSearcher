from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_utqr_v7_authorized_batch import build_manifest  # noqa: E402


def test_authorized_batch_is_fixed_label_blind_and_preserves_failures() -> None:
    manifest = build_manifest()
    policy = manifest["selection_policy"]
    assert policy["case_selection_uses_oracle"] is False
    assert policy["oracle_reachability_prescreen"] is False
    assert policy["retain_all_execution_failures"] is True
    assert policy["teacher_receives_oracle"] is False

    datasets = manifest["datasets"]
    assert datasets["VideoSham"]["requested_cases"] == 7
    assert datasets["VideoSham"]["shortfall_from_ten"] == 3
    assert datasets["VCSL"]["case_indices"] == [*range(11, 20), 21]
    assert datasets["VCSL"]["entity_disjoint"] is True
    assert datasets["VCSL"]["selection_path"] is None
    assert datasets["VidDiffBench"]["requested_cases"] == 10
    assert datasets["VidDiffBench"]["candidate_tool_counts"] == [3] * 10
    assert datasets["VidDiffBench"]["skip_oracle_unreachable"] is False

    serialized = json.dumps(manifest, ensure_ascii=False).lower()
    assert "oracle_direction" not in serialized
    assert "oracle_boxes" not in serialized
    assert "verifier_result" not in serialized
