from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_saved_episode_semantics import audit_episode  # noqa: E402


def test_saved_episode_semantic_audit_rejects_prompt_scaffolding(tmp_path: Path) -> None:
    state = {"evidence_store": {}, "eligible_tools": ["inspect_media"]}
    artifact = {
        "dataset": "VideoSham",
        "case_id": "case-1",
        "mode": "clean",
        "quality": {"score": 100, "accepted_for_high_quality_sft": True},
        "episode": {
            "terminal_success": True,
            "turns": [
                {
                    "turn": 1,
                    "status": "executed",
                    "generated_tool": "inspect_media",
                    "generated_arguments": {},
                    "reasoning": (
                        "The required compact grounding clause is visible in the prompt. "
                        "I inspect the media to close the initial evidence gap."
                    ),
                    "sft_datum": {
                        "messages": [
                            {"role": "user", "content": json.dumps({"current_harness_state": state})}
                        ]
                    },
                }
            ],
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    report = audit_episode(path)
    assert report["semantic_passed"] is False
    assert "reasoning_mentions_prompt_scaffolding" in report["turn_reports"][0][
        "reasoning_problems"
    ]
