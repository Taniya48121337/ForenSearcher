from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from videosham_batch import (  # noqa: E402
    MultiFamilyVideoEnvironment,
    _suspect_interval_covered_by_comparison,
    _case_number,
    _rows,
    _time_window,
    compile_case,
    localize_comparison_scope,
    preflight_case,
    select_rows,
)
from videosham_consolidate import (  # noqa: E402
    DEFAULT_RUNS as CONSOLIDATION_RUNS,
    consolidate,
)


class VideoShamBatchTest(unittest.TestCase):
    def test_consolidated_seventeen_case_export_is_deduplicated_and_tool_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            report = consolidate(CONSOLIDATION_RUNS, output_root)

            self.assertEqual(report["summary"]["cases"], 17)
            self.assertEqual(report["summary"]["passed"], 17)
            self.assertEqual(report["tool_telemetry"]["unused_implemented_tools"], [])
            self.assertEqual(report["tool_telemetry"]["selected_but_not_indispensable"], {})
            self.assertEqual(
                sum(bool(line.strip()) for line in (output_root / "sft_clean_accepted.jsonl").read_text(encoding="utf-8").splitlines()),
                102,
            )

    def test_seeded_sample_is_reproducible(self) -> None:
        selected = random.Random(20260818).sample(_rows(), 10)
        self.assertEqual(
            [_case_number(row) for row in selected],
            ["4054", "1007", "4197", "4112", "2010", "2004", "4189", "4142", "4224", "4039"],
        )

    def test_explicit_case_selection_preserves_requested_coverage_order(self) -> None:
        selected = select_rows(
            _rows(),
            seed=0,
            sample_size=1,
            case_ids=["videosham-2004", "1007"],
        )
        self.assertEqual([_case_number(row) for row in selected], ["2004", "1007"])

    def test_multifamily_adapter_admits_downloaded_temporal_cases(self) -> None:
        rows = {_case_number(row): row for row in _rows()}
        self.assertTrue(preflight_case(rows["1007"])["cli_eligible"])
        self.assertTrue(preflight_case(rows["4189"])["cli_eligible"])
        self.assertEqual(preflight_case(rows["4189"])["missing_capabilities"], [])

    def test_first_seconds_window_is_parsed(self) -> None:
        self.assertEqual(_time_window("Extract the first 5 seconds. Mute it."), [0.0, 5.0])

    def test_preclipped_original_window_maps_to_local_timeline(self) -> None:
        self.assertEqual(localize_comparison_scope([7.0, 12.0], 5.105104), [0.0, 5.105])
        self.assertEqual(localize_comparison_scope([89.0, 94.0], 4.338), [0.0, 4.338])
        self.assertEqual(localize_comparison_scope([0.0, 15.0], 20.073333), [0.0, 15.0])

    def test_modality_screen_preserves_a_real_policy_choice_after_inspection(self) -> None:
        environment = object.__new__(MultiFamilyVideoEnvironment)
        environment.call_counts = {
            "inspect_media": 0,
            "detect_temporal_edit": 0,
            "compare_audio_tracks": 0,
            "compare_spatial_regions": 0,
        }
        environment._triage = {"primary_modality": "spatial"}
        self.assertEqual(environment.eligible_tool_names(), ["inspect_media"])
        environment.call_counts["inspect_media"] = 1
        self.assertEqual(
            environment.eligible_tool_names(),
            [
                "detect_temporal_edit",
                "compare_audio_tracks",
                "compare_spatial_regions",
            ],
        )

    def test_student_visible_modality_screen_omits_router_answer(self) -> None:
        environment = object.__new__(MultiFamilyVideoEnvironment)
        environment._triage = {
            "candidate_modalities": ["spatial"],
            "primary_modality": "spatial",
            "duration_ratio": 1.0,
            "visual_p05_similarity": 0.92,
            "rgb_tile_p95_mae": 0.04,
            "rgb_tile_max_mae": 0.08,
            "rgb_changed_tile_fraction": 0.12,
            "audio_similarity": 0.99,
        }
        visible = environment._student_visible_triage_metrics()
        self.assertNotIn("primary_modality", visible)
        self.assertNotIn("candidate_modalities", visible)
        self.assertEqual(visible["rgb_tile_p95_mae"], 0.04)

    def test_wrong_modality_is_an_executed_negative_measurement_not_a_rejection(self) -> None:
        environment = object.__new__(MultiFamilyVideoEnvironment)
        environment.manifest = {"comparison_scope_s": [0.0, 5.0]}
        environment.call_counts = {
            "inspect_media": 1,
            "detect_temporal_edit": 0,
            "compare_audio_tracks": 0,
            "compare_spatial_regions": 0,
        }
        environment._triage = {"primary_modality": "spatial"}
        environment._repair_due = False
        environment._last_successful_tool = None
        environment._visual_metrics = lambda: {
            "temporal_edit_detected": False,
            "spatial_edit_detected": True,
        }
        result = environment.execute(
            "detect_temporal_edit",
            {
                "source_ref": "source",
                "suspect_ref": "suspect",
                "comparison_scope_s": [0.0, 5.0],
            },
        )
        self.assertFalse(result["goal_satisfied"])
        self.assertFalse(result["result"]["material_difference_detected"])
        self.assertEqual(result["evidence_atoms"][0].supports_fields, [])
        self.assertNotIn("detect_temporal_edit", environment.eligible_tool_names())

    def test_suspect_interval_reports_the_requested_window_when_timeline_is_available(self) -> None:
        self.assertEqual(
            _suspect_interval_covered_by_comparison([1.0, 11.0], 11.96),
            [1.0, 11.0],
        )
        self.assertEqual(
            _suspect_interval_covered_by_comparison([0.0, 15.0], 13.32),
            [0.0, 13.32],
        )

    def test_teacher_manifest_uses_only_logical_media_refs(self) -> None:
        row = next(row for row in _rows() if _case_number(row) == "1004")
        preflight = preflight_case(row)
        with tempfile.TemporaryDirectory() as directory:
            manifest, _oracle = compile_case(row, preflight, Path(directory))
        self.assertEqual(manifest["source_media"], {"ref": "source"})
        self.assertEqual(manifest["edited_media"], {"ref": "suspect"})
        self.assertNotIn(row["Task"], str(manifest))
        self.assertNotIn(str(ROOT), str(manifest))


if __name__ == "__main__":
    unittest.main()
