from __future__ import annotations

import json
from pathlib import Path

from src.audit_student_policy_diversity import audit_batches


def _write_case(
    root: Path,
    case_id: str,
    statement: str,
    measurement_route: list[str],
    *,
    dataset: str = "VidDiffBench",
) -> None:
    path = root / "cases" / dataset.lower() / case_id / "clean.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence = ["inspect_comparison_task", *measurement_route, "create_candidate_claim", "stop"]
    payload = {
        "quality": {"score": 100, "accepted_for_high_quality_sft": True, "tool_sequence": sequence},
        "episode": {
            "dataset": dataset,
            "source_case_id": case_id,
            "tool_sequence": sequence,
            "turns": [
                {
                    "generated_tool": "create_candidate_claim",
                    "generated_arguments": {
                        "field_values": {
                            "comparison_statement": statement,
                            "measurement_method": measurement_route[-1],
                            "preferred_video": "c" if "abstention" in measurement_route[-1] else "a",
                        }
                    },
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reports_route_diversity_without_making_it_a_gate(tmp_path: Path) -> None:
    for index in range(5):
        statement = "The needle is more perpendicular." if index < 3 else "The tube moves more often."
        _write_case(tmp_path, f"case-{index}", statement, ["compare_uniform_timeline"])

    report = audit_batches([tmp_path])

    assert report["is_acceptance_gate"] is False
    assert report["episode_count"] == 5
    assert report["unique_measurement_routes"] == 1
    assert report["dominant_measurement_route_share"] == 1.0
    assert report["warnings"][0]["code"] == "dominant_macro_route_across_predicate_families"


def test_recognizes_deep_search_and_meta_abstention(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "case-meta",
        "The number of stops is larger.",
        [
            "compare_global_motion_symmetric",
            "compare_action_refined_symmetric",
            "compare_symmetry_residual_abstention",
        ],
    )

    report = audit_batches([tmp_path])

    assert report["deep_search_episode_count"] == 1
    assert report["meta_abstention_episode_count"] == 1
    assert report["selected_direction_counts"] == {"c": 1}
    assert report["warnings"] == []


def test_reports_dataset_specific_non_compare_measurement_tools(tmp_path: Path) -> None:
    _write_case(
        tmp_path,
        "videosham-1",
        "Locate the temporal edit.",
        ["inspect_media", "detect_temporal_edit"],
        dataset="VideoSham",
    )
    _write_case(
        tmp_path,
        "vcsl-1",
        "Locate the copied segment.",
        ["localize_copy_segment_spd_official"],
        dataset="VCSL",
    )

    report = audit_batches([tmp_path])

    assert report["by_dataset"]["VideoSham"]["tools_used"] == {
        "inspect_media": 1,
        "detect_temporal_edit": 1,
    }
    assert report["by_dataset"]["VCSL"]["tools_used"] == {
        "localize_copy_segment_spd_official": 1
    }
