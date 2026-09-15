from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_student_batch_sft import validate_rows
from run_student_closed_loop_batch import _all_viddiff_items, _episode_style_diagnostics


OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1/student_batch_test_v10_native"
)
EXTERNAL_VIDEO = (
    ROOT
    / "outputs/student_external_contract_v1/student_batch_videosham_external_holdout_v1"
)
EXTERNAL_SEEDS = (
    ROOT
    / "outputs/student_external_contract_v1/videosham_external_holdout_v1"
)
EXTERNAL_VCSL_SPD = (
    ROOT
    / "outputs/student_external_contract_v1/student_batch_vcsl_spd_external_holdout_v2"
)
EXTERNAL_VCSL_SPD_SEEDS = (
    ROOT / "outputs/student_external_contract_v1/vcsl_spd_external_holdout_v2"
)
REPEAT_PANEL = (
    ROOT / "outputs/student_external_contract_v1/student_passk_repeat_panel_v1"
)
REPAIR_REPEAT_PANEL = (
    ROOT / "outputs/student_external_contract_v1/student_repair_passk_repeat_panel_v1"
)
EXTERNAL_VIDEO_REPEAT_PANEL = (
    ROOT
    / "outputs/student_external_contract_v1/student_external_videosham_passk_repeat_panel_v1"
)
FINAL_VIDEO_REPEAT_PANEL = (
    ROOT
    / "outputs/student_external_contract_v1/student_external_videosham_final_repeat_v2"
)
FINAL_VIDEO_SEED_1203 = (
    ROOT
    / "outputs/student_external_contract_v1/student_external_videosham_final_seed1203_v1"
)
FINAL_VCSL_REPEAT_PANEL = (
    ROOT
    / "outputs/student_external_contract_v1/student_external_vcsl_spd_final_repeat_v2"
)
FROZEN_VIDDIFF_PILOT_CANDIDATES = (
    ROOT
    / "outputs/selection/viddiff_authorized_20260823_pilot5_frozen"
    / "viddiff_best_of_tree_seed_20260821_n5"
    / "candidate_measurements.student.jsonl"
)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class StudentClosedLoopBatchTest(unittest.TestCase):
    def test_style_diagnostics_reports_but_does_not_reject_repeated_scaffolds(self) -> None:
        episode = {
            "turns": [
                {"turn": 1, "generated_tool": "inspect", "reasoning": (
                    "The latest visible state has identity. The only remaining gap is evidence."
                )},
                {"turn": 2, "generated_tool": "measure", "reasoning": (
                    "The latest visible state has evidence. The only remaining gap is binding."
                )},
            ]
        }
        report = _episode_style_diagnostics(episode)
        self.assertEqual(report["repeated_scaffold_turns"], 1)
        self.assertEqual(report["repeated_scaffold_problem_count"], 2)

    def test_viddiff_runtime_loader_scrubs_legacy_private_locators(self) -> None:
        items = _all_viddiff_items()
        self.assertTrue(items)
        serialized = json.dumps(items)
        self.assertNotIn('"video_path"', serialized)
        self.assertNotIn('"candidate_report"', serialized)
        self.assertNotIn('"oracle_direction"', serialized)

    def test_viddiff_runtime_loader_can_pin_frozen_candidate_file(self) -> None:
        items = _all_viddiff_items([FROZEN_VIDDIFF_PILOT_CANDIDATES])
        self.assertEqual(
            set(items),
            {"surgery_20", "surgery_117", "surgery_73", "surgery_166", "surgery_78"},
        )

    def test_all_held_out_batch_episodes_and_exported_rows_pass(self) -> None:
        report = json.loads((OUTPUT / "batch_report.json").read_text(encoding="utf-8"))
        validation = json.loads(
            (OUTPUT / "sft_validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["split"], "test")
        self.assertEqual(report["summary"]["episodes"], 12)
        self.assertEqual(report["summary"]["accepted_episodes"], 12)
        self.assertEqual(report["summary"]["minimum_score"], 100)
        self.assertEqual(report["summary"]["episodes_with_rejection"], 0)
        self.assertEqual(report["summary"]["accepted_sft_rows"], 68)
        self.assertTrue(validation["all_passed"])
        self.assertEqual(validation["failed_rows"], 0)
        self.assertEqual(
            validation["dataset_rows"],
            {"VCSL": 22, "VidDiffBench": 26, "VideoSham": 20},
        )

    def test_preflight_reconstructs_all_held_out_first_states(self) -> None:
        report = json.loads((OUTPUT / "preflight.json").read_text(encoding="utf-8"))
        self.assertEqual(report["episodes"], 12)
        self.assertEqual(report["environment_builds_succeeded"], 12)
        self.assertEqual(report["eligible_tool_matches"], 12)
        self.assertTrue(report["all_passed"])

    def test_validator_rejects_post_action_history_leakage(self) -> None:
        datum = copy.deepcopy(
            _jsonl(OUTPUT / "accepted_student_sft.jsonl")[0]
        )
        payload = json.loads(datum["messages"][1]["content"])
        payload["past_execution_history"].append(
            {
                "step_index": 0,
                "past_action": {
                    "executed_tool": datum["metadata"]["target_tool"],
                    "past_input": {},
                },
                "environment_outcome": {"status": "executed"},
            }
        )
        datum["messages"][1]["content"] = json.dumps(payload)
        validation = validate_rows([datum])
        self.assertFalse(validation["all_passed"])
        self.assertIn(
            "pre_action_history_length_mismatch",
            validation["failures"][0]["failures"],
        )

    def test_teacher_free_videosham_external_holdout_batch_passes(self) -> None:
        manifest = json.loads(
            (EXTERNAL_SEEDS / "manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (EXTERNAL_VIDEO / "batch_report.json").read_text(encoding="utf-8")
        )
        balanced = json.loads(
            (EXTERNAL_VIDEO / "diversity_balanced_sft_validation.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["cases"], 13)
        self.assertFalse(manifest["teacher_actions_used"])
        self.assertFalse(manifest["oracle_or_verifier_used"])
        self.assertFalse(manifest["original_edit_tasks_exported"])
        self.assertFalse(manifest["raw_urls_exported"])
        self.assertEqual(report["split"], "external_holdout")
        self.assertEqual(report["summary"]["accepted_episodes"], 13)
        self.assertEqual(report["summary"]["actions"], 78)
        self.assertEqual(report["summary"]["accepted_sft_rows"], 78)
        self.assertEqual(report["summary"]["diversity_balanced_sft_rows"], 66)
        self.assertTrue(balanced["validation"]["all_passed"])

    def test_teacher_free_vcsl_spd_external_holdout_batch_passes(self) -> None:
        manifest = json.loads(
            (EXTERNAL_VCSL_SPD_SEEDS / "manifest.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (EXTERNAL_VCSL_SPD / "batch_report.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (EXTERNAL_VCSL_SPD / "sft_validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["cases"], 2)
        self.assertFalse(manifest["teacher_actions_used"])
        self.assertFalse(manifest["oracle_or_verifier_in_student_seed"])
        self.assertTrue(manifest["verifier_used_only_for_candidate_admission"])
        self.assertEqual(report["summary"]["accepted_episodes"], 2)
        self.assertEqual(report["summary"]["actions"], 12)
        self.assertEqual(report["summary"]["minimum_score"], 100)
        self.assertEqual(report["summary"]["episodes_with_rejection"], 0)
        self.assertEqual(
            validation["tool_rows"]["localize_copy_segment_spd_official"], 2
        )
        self.assertTrue(validation["all_passed"])

    def test_stochastic_repeat_panel_is_rekeyed_and_valid(self) -> None:
        report = json.loads(
            (REPEAT_PANEL / "repeat_panel_audit.json").read_text(encoding="utf-8")
        )
        rows = _jsonl(REPEAT_PANEL / "rekeyed_repeat_sft.jsonl")
        self.assertEqual(report["episodes"], 9)
        self.assertEqual(report["accepted_episodes"], 9)
        self.assertEqual(report["original_row_id_collision_occurrences"], 40)
        self.assertEqual(report["rekeyed_unique_row_ids"], 60)
        self.assertTrue(report["distinct_generation_seeds"])
        self.assertTrue(report["tool_sequence_stable_for_every_case"])
        self.assertTrue(report["rekeyed_validation"]["all_passed"])
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["metadata"]["row_id"] for row in rows}), 60)
        self.assertTrue(all(row["metadata"].get("rollout_id") for row in rows))

    def test_stochastic_repair_repeat_panel_is_natively_unique_and_valid(self) -> None:
        report = json.loads(
            (REPAIR_REPEAT_PANEL / "repeat_panel_audit.json").read_text(encoding="utf-8")
        )
        rows = _jsonl(REPAIR_REPEAT_PANEL / "rekeyed_repeat_sft.jsonl")
        self.assertEqual(report["episodes"], 9)
        self.assertEqual(report["accepted_episodes"], 9)
        self.assertEqual(report["original_row_id_collision_occurrences"], 0)
        self.assertEqual(report["rekeyed_unique_row_ids"], 39)
        self.assertTrue(report["tool_sequence_stable_for_every_case"])
        self.assertTrue(report["rekeyed_validation"]["all_passed"])
        self.assertEqual(len(rows), 39)
        self.assertTrue(all(row["metadata"].get("rollout_id") for row in rows))

    def test_external_videosham_repeat_panel_has_temporal_and_spatial_pass2(self) -> None:
        report = json.loads(
            (EXTERNAL_VIDEO_REPEAT_PANEL / "repeat_panel_audit.json").read_text(
                encoding="utf-8"
            )
        )
        rows = _jsonl(EXTERNAL_VIDEO_REPEAT_PANEL / "rekeyed_repeat_sft.jsonl")
        tools = {row["metadata"]["target_tool"] for row in rows}
        self.assertEqual(report["episodes"], 10)
        self.assertEqual(report["accepted_episodes"], 10)
        self.assertEqual(report["source_sft_rows"], 60)
        self.assertEqual(report["original_row_id_collision_occurrences"], 0)
        self.assertEqual(report["reasoning_variant_count_distribution"], {"2": 30})
        self.assertTrue(report["tool_sequence_stable_for_every_case"])
        self.assertTrue(report["rekeyed_validation"]["all_passed"])
        self.assertIn("detect_temporal_edit", tools)
        self.assertIn("compare_spatial_regions", tools)

    def test_final_external_repeat_panels_are_independent_and_valid(self) -> None:
        video = json.loads(
            (FINAL_VIDEO_REPEAT_PANEL / "repeat_panel_audit.json").read_text(
                encoding="utf-8"
            )
        )
        vcsl = json.loads(
            (FINAL_VCSL_REPEAT_PANEL / "repeat_panel_audit.json").read_text(
                encoding="utf-8"
            )
        )
        telemetry = json.loads(
            (FINAL_VIDEO_SEED_1203 / "batch_report.json").read_text(encoding="utf-8")
        )["summary"]
        self.assertEqual(video["episodes"], 6)
        self.assertEqual(video["reasoning_variant_count_distribution"], {"2": 18})
        self.assertEqual(vcsl["episodes"], 4)
        self.assertEqual(vcsl["reasoning_variant_count_distribution"], {"2": 12})
        self.assertTrue(video["rekeyed_validation"]["all_passed"])
        self.assertTrue(vcsl["rekeyed_validation"]["all_passed"])
        self.assertGreater(telemetry["episode_seconds_p95"], 0)
        self.assertGreater(telemetry["generation_seconds_max_per_action"], 0)
        self.assertGreater(telemetry["max_generated_tokens_per_action"], 0)


if __name__ == "__main__":
    unittest.main()
