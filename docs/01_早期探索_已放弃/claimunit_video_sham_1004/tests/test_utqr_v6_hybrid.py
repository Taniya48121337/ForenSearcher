from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utqr_v6_hybrid import evaluate  # noqa: E402


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, reasoning: str = "The current evidence leaves one measured gap, so inspect_media will establish visible context."):
    package = _write_json(tmp_path / "package.json", {"ready_for_training": True})
    semantic = _write_json(
        tmp_path / "semantic.json",
        {
            "all_passed": True,
            "episode_reports": [
                {
                    "dataset": "VideoSham",
                    "case_id": "case-1",
                    "mode": "clean",
                    "semantic_passed": True,
                    "terminal_success": True,
                }
            ],
        },
    )
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps(
            {
                "blind_id": "BLIND-001",
                "turns": [{"reasoning": reasoning, "tool": "inspect_media"}],
            }
        ),
        encoding="utf-8",
    )
    key = _write_json(
        tmp_path / "key.json",
        {
            "mapping": [
                {
                    "blind_id": "BLIND-001",
                    "dataset": "VideoSham",
                    "case_id": "case-1",
                    "mode": "clean",
                }
            ]
        },
    )
    return package, semantic, packet, key


def _judgment(judge_id: str, score: int = 5) -> dict:
    ids = [f"D{index}" for index in range(1, 9)]
    return {
        "blind_id": "BLIND-001",
        "judge_id": judge_id,
        "dimension_scores": {item: score for item in ids},
        "dimension_evidence": {
            item: "The cited turn provides concrete episode evidence." for item in ids
        },
        "hard_fail": False,
        "keep_for_sft": True,
        "overall_comment": "The complete trace is grounded, necessary, and suitable for training.",
    }


def test_machine_pass_does_not_become_gold_without_two_blind_judges(tmp_path: Path) -> None:
    package, semantic, packet, key = _fixture(tmp_path)
    report = evaluate(package, semantic, packet, key, [])
    assert report["machine_qualified"] is True
    assert report["blind_judging_complete"] is False
    assert report["current_batch_gold"] is False
    assert report["mass_production_ready"] is False


def test_two_strict_blind_judges_can_qualify_episode_but_not_tiny_batch(tmp_path: Path) -> None:
    package, semantic, packet, key = _fixture(tmp_path)
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        "\n".join(json.dumps(_judgment(judge)) for judge in ("judge-a", "judge-b")),
        encoding="utf-8",
    )
    report = evaluate(package, semantic, packet, key, [judgments])
    assert report["current_batch_gold"] is True
    assert report["mass_production_ready"] is False
    assert report["episodes"][0]["weighted_mean"] == 5.0


def test_known_process_artifact_blocks_machine_qualification(tmp_path: Path) -> None:
    package, semantic, packet, key = _fixture(
        tmp_path, "The required compact grounding clause is present, so inspect_media is next."
    )
    report = evaluate(package, semantic, packet, key, [])
    assert report["machine_qualified"] is False
    assert report["episodes"][0]["machine_process_artifact_hits"] == [
        "required compact grounding clause"
    ]
