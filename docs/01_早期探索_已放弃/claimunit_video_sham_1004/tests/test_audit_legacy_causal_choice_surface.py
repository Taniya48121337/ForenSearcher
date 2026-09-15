from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_legacy_causal_choice_surface import (  # noqa: E402
    audit_episode,
    audit_exposure_jsonl,
)


def _message(role: str, content: dict) -> dict:
    return {"role": role, "content": json.dumps(content)}


def test_reconstructs_choice_from_preaction_visible_state() -> None:
    episode = {
        "trajectory_id": "T-1",
        "messages": [
            _message(
                "user",
                {
                    "harness_state": {
                        "eligible_tools": ["compare_audio_tracks", "compare_spatial_regions"]
                    }
                },
            ),
            {
                "role": "assistant",
                "content": "measure",
                "tool_calls": [
                    {"function": {"name": "compare_audio_tracks", "arguments": "{}"}}
                ],
            },
        ],
    }
    result = audit_episode(episode, episode_name="clean")
    assert result["semantic_measurement_turns"] == 1
    assert result["nontrivial_semantic_choice_turns"] == 1


def test_detects_singleton_measurement_and_visible_router_answer() -> None:
    episode = {
        "messages": [
            _message(
                "user",
                {
                    "primary_modality": "spatial",
                    "harness_state": {"eligible_tools": ["compare_spatial_regions"]},
                },
            ),
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "compare_spatial_regions", "arguments": "{}"}}
                ],
            },
        ]
    }
    result = audit_episode(episode, episode_name="clean")
    assert result["semantic_measurement_turns"] == 1
    assert result["nontrivial_semantic_choice_turns"] == 0
    assert result["measurement_turns_with_router_answer"] == 1


def test_exposure_jsonl_counts_only_semantic_measurement_choices(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"row_id": "1", "target_tool": "inspect_media", "eligible_tools": ["inspect_media"]},
        {
            "row_id": "2",
            "dataset": "VideoSham",
            "target_tool": "compare_audio_tracks",
            "eligible_tools": ["compare_audio_tracks", "compare_spatial_regions"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = audit_exposure_jsonl(path)
    assert result["semantic_measurement_turns"] == 1
    assert result["nontrivial_semantic_choice_turns"] == 1
