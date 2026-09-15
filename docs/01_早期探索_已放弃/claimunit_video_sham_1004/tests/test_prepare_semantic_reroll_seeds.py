from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_semantic_reroll_seeds import prepare  # noqa: E402


def _row(dataset: str, case_id: str, mode: str) -> dict:
    return {
        "metadata": {
            "dataset": dataset,
            "case_id": case_id,
            "row_id": f"{dataset}:{case_id}:trajectory:{mode}:1",
        }
    }


def test_prepare_resolves_exact_dataset_case_mode_only() -> None:
    manifest = {
        "items": [
            {
                "dataset": "VidDiffBench",
                "case_id": "viddiff-surgery_26",
                "mode": "repair",
                "failure_codes": ["x"],
            }
        ]
    }
    rows, report = prepare(
        manifest,
        [
            _row("VidDiffBench", "viddiff-surgery_26", "clean"),
            _row("VidDiffBench", "viddiff-surgery_26", "repair"),
        ],
    )
    assert rows[0]["metadata"]["row_id"].endswith(":repair:1")
    assert report["exact_match"] is True
    assert report["teacher_or_external_service_called"] is False


def test_prepare_rejects_missing_or_ambiguous_seeds() -> None:
    manifest = {
        "items": [
            {"dataset": "VCSL", "case_id": "vcsl-x-y", "mode": "clean"}
        ]
    }
    with pytest.raises(ValueError, match="missing"):
        prepare(manifest, [])
    with pytest.raises(ValueError, match="ambiguous"):
        prepare(
            manifest,
            [
                _row("VCSL", "vcsl-x-y", "clean"),
                _row("VCSL", "vcsl-x-y", "clean"),
            ],
        )


def test_prepare_can_recover_exact_student_visible_turn1(tmp_path: Path) -> None:
    case_id = "viddiff-surgery_148"
    source_id = (
        "VidDiffBench:viddiff-surgery_148:STUDENT-HOLDOUT-SURGERY_148-CLEAN-001:clean:1"
    )
    episode_path = tmp_path / "clean.json"
    episode_path.write_text(
        json.dumps(
            {
                "episode": {
                    "trajectory_type": "clean",
                    "turns": [
                        {
                            "sft_datum": {
                                "messages": [
                                    {"role": "system", "content": "policy"},
                                    {
                                        "role": "user",
                                        "content": json.dumps(
                                            {
                                                "task": {},
                                                "current_harness_state": {},
                                            }
                                        ),
                                    },
                                    {"role": "assistant", "content": "old action"},
                                ],
                                "metadata": {"source_seed_row_id": source_id},
                            }
                        },
                        {
                            "visible_eligible_tools": [
                                "compare_uniform_timeline",
                                "compare_global_motion_symmetric",
                            ]
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "items": [
            {
                "dataset": "VidDiffBench",
                "case_id": case_id,
                "mode": "clean",
                "episode_path": str(episode_path),
            }
        ]
    }
    rows, report = prepare(manifest, [])
    assert rows[0]["metadata"]["row_id"] == source_id
    assert rows[0]["metadata"]["routing_variant"] == "balanced"
    assert report["selection"][0]["resolution_source"] == (
        "student_visible_turn1_fallback"
    )
