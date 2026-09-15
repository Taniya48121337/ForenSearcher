from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_replacement_lineage_curriculum_sft import (  # noqa: E402
    ARM,
    ROUTE_ROW_IDS,
    compile_curriculum,
)


def _row(row_id: str, case_id: str, tool: str) -> dict:
    new_id = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-002"
    dropped_id = "EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"
    return {
        "messages": [
            {"role": "system", "content": "visible only"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_harness_state": {
                            "eligible_tools": [tool, "compare_uniform_timeline"],
                            "dropped_evidence_ids": [dropped_id],
                            "evidence_store": {
                                new_id: {
                                    "active": True,
                                    "observation": {
                                        "measurement_method": "compare_global_motion_symmetric",
                                        "preferred_video": "a",
                                        "order_consistent": True,
                                    },
                                },
                                "EA-VIDDIFF-COMPARE_UNIFORM_TIMELINE-001": {
                                    "active": True,
                                    "observation": {
                                        "measurement_method": "compare_uniform_timeline",
                                        "preferred_video": "b",
                                        "order_consistent": None,
                                    },
                                },
                            },
                        },
                        "latest_environment_observation": {
                            "created_evidence_ids": [new_id]
                        },
                    }
                ),
            },
            {
                "role": "assistant",
                "content": "The only eligible measurement tool closes the gap.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": {
                                "video_a_ref": "video_a",
                                "video_b_ref": "video_b",
                                "comparison_statement": "Visible comparison",
                            },
                        },
                    }
                ],
            },
        ],
        "metadata": {"row_id": row_id, "case_id": case_id, "target_tool": tool},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_curriculum_mixes_queue_and_balanced_replacement_routes(tmp_path: Path) -> None:
    queued_root = tmp_path / "queued"
    route_root = tmp_path / "route"
    output = tmp_path / "out"
    _write_rows(
        queued_root / ARM / "train" / "sft_all.jsonl",
        [_row("queue-1", "viddiff-surgery_1", "assess_evidence_candidate")],
    )
    route_rows = [
        _row(row_id, row_id.split(":")[1], "compare_action_refined_symmetric")
        for row_id in sorted(ROUTE_ROW_IDS)
    ]
    _write_rows(route_root / ARM / "train" / "sft_all.jsonl", route_rows)

    report = compile_curriculum(queued_root, route_root, output)
    assert report["rows"] == 5
    assert report["replacement_route_rows"] == 4
    assert report["route_identity_contrast_rows"] == 4
    assert report["route_reasoning_all_clean"]
    assert report["protected_cases_excluded"] == [
        "viddiff-surgery_16",
        "viddiff-surgery_26",
    ]
    compiled = [
        json.loads(line)
        for line in (output / ARM / "train" / "sft_all.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["metadata"]["protected_case_disjoint"] for row in compiled)
    route_rows = [
        row for row in compiled if row["metadata"]["curriculum_source"] == "replacement_route"
    ]
    assert all(row["metadata"]["identity_contrast_reasoning"] for row in route_rows)
    assert all("ID-distinct" in row["messages"][-1]["content"] for row in route_rows)
    assert all("best next eligible action" in row["messages"][-1]["content"] for row in route_rows)


def test_curriculum_rejects_protected_case_leak(tmp_path: Path) -> None:
    queued_root = tmp_path / "queued"
    route_root = tmp_path / "route"
    _write_rows(
        queued_root / ARM / "train" / "sft_all.jsonl",
        [_row("protected", "viddiff-surgery_16", "assess_evidence_candidate")],
    )
    _write_rows(
        route_root / ARM / "train" / "sft_all.jsonl",
        [
            _row(row_id, row_id.split(":")[1], "compare_uniform_timeline")
            for row_id in sorted(ROUTE_ROW_IDS)
        ],
    )
    with pytest.raises(ValueError, match="protected case leaked"):
        compile_curriculum(queued_root, route_root, tmp_path / "out")


def test_curriculum_rejects_empty_progressive_queue(tmp_path: Path) -> None:
    queued_root = tmp_path / "queued"
    route_root = tmp_path / "route"
    _write_rows(queued_root / ARM / "train" / "sft_all.jsonl", [])
    _write_rows(
        route_root / ARM / "train" / "sft_all.jsonl",
        [
            _row(row_id, row_id.split(":")[1], "compare_uniform_timeline")
            for row_id in sorted(ROUTE_ROW_IDS)
        ],
    )
    with pytest.raises(ValueError, match="progressive queue is empty"):
        compile_curriculum(queued_root, route_root, tmp_path / "out")


def test_curriculum_rejects_queue_with_failed_quality_report(tmp_path: Path) -> None:
    queued_root = tmp_path / "queued"
    route_root = tmp_path / "route"
    _write_rows(
        queued_root / ARM / "train" / "sft_all.jsonl",
        [_row("queue-1", "viddiff-surgery_1", "assess_evidence_candidate")],
    )
    (queued_root / "compile_report.json").write_text(
        json.dumps({"quality_report": {"all_applicable_checks_passed": False}}),
        encoding="utf-8",
    )
    _write_rows(
        route_root / ARM / "train" / "sft_all.jsonl",
        [
            _row(row_id, row_id.split(":")[1], "compare_uniform_timeline")
            for row_id in sorted(ROUTE_ROW_IDS)
        ],
    )
    with pytest.raises(ValueError, match="quality checks did not pass"):
        compile_curriculum(queued_root, route_root, tmp_path / "out")
