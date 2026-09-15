from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assemble_current_semantic_panel import assemble  # noqa: E402


def test_replacement_overlays_same_dataset_case_and_mode(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    for run in (run_a, run_b):
        episode = run / "cases" / "videosham" / "case-1" / "clean.json"
        episode.parent.mkdir(parents=True)
        episode.write_text("{}", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "dataset": "VideoSham",
                        "case_id": "case-1",
                        "mode": "clean",
                        "accepted_for_high_quality_sft": True,
                        "source_report": str(run_a / "batch_report.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "dataset": "VideoSham",
                        "case_id": "case-1",
                        "mode": "clean",
                        "accepted_for_high_quality_sft": True,
                        "source_report": str(run_b / "batch_report.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = assemble(baseline, [replacement])
    assert report["episodes"] == 1
    assert "run_b" in report["items"][0]["episode_path"]
    assert report["all_episode_files_resolved"] is True
