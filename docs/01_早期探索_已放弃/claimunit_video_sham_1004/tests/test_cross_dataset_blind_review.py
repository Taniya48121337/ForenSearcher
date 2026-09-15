from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_cross_dataset_blind_review import build  # noqa: E402


def test_blind_packet_separates_identity_mapping_and_has_empty_review_form(tmp_path: Path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "HiddenDataset",
                "case_id": "hidden-case",
                "mode": "clean",
                "episode": {
                    "terminal_success": True,
                    "tool_sequence": ["inspect_media"],
                    "turns": [
                        {
                            "turn": 1,
                            "generated_tool": "inspect_media",
                            "generated_arguments": {},
                            "reasoning": "Visible context is missing, so inspect the media.",
                            "status": "executed",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    packet, key = build([path], seed=7)
    assert "dataset" not in packet[0]
    assert "case_id" not in packet[0]
    assert packet[0]["review_form"]["keep_for_sft"] is None
    assert key["mapping"][0]["dataset"] == "HiddenDataset"
    assert "oracle" not in json.dumps(packet).lower()
