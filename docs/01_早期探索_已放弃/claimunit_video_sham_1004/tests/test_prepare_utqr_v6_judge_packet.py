from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_utqr_v6_judge_packet import build  # noqa: E402


def test_judge_packet_uses_strict_anchors_and_blank_auditable_form(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps({"blind_id": "BLIND-001", "turns": [{"turn": 1}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = build(packet)
    assert len(rows) == 1
    assert len(rows[0]["dimensions"]) == 8
    assert rows[0]["dimensions"][0]["score_5"]
    assert rows[0]["judgment_form"]["judge_id"] is None
    assert set(rows[0]["judgment_form"]["dimension_evidence"]) == {
        f"D{index}" for index in range(1, 9)
    }
