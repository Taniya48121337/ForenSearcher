from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_student_closed_loop_quality import (
    _candidate_episode_selected,
    _hydrate_episode_from_sft,
    _measurement_tools,
    _seed_row_for_episode,
)

OUTPUT = ROOT / "outputs/student_external_contract_v1"


class StudentClosedLoopQualityTest(unittest.TestCase):
    def test_measurement_tool_audit_ignores_generation_failure_without_tool_name(self) -> None:
        turns = [
            {"status": "generation_contract_failure"},
            {"generated_tool": "inspect_media"},
            {"generated_tool": "detect_temporal_edit"},
        ]
        self.assertEqual(_measurement_tools(turns), ["detect_temporal_edit"])

    def test_candidate_only_rescore_filters_old_rejections_before_independent_audit(self) -> None:
        accepted = {"quality": {"accepted_for_high_quality_sft": True}}
        rejected = {"quality": {"accepted_for_high_quality_sft": False}}
        self.assertTrue(_candidate_episode_selected(accepted, True))
        self.assertFalse(_candidate_episode_selected(rejected, True))
        self.assertTrue(_candidate_episode_selected(rejected, False))

    def test_batch_rescore_can_restore_seed_from_first_turn(self) -> None:
        embedded = {
            "messages": [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": "{}"},
            ]
        }
        episode = {"turns": [{"sft_datum": embedded}]}
        self.assertIs(
            _seed_row_for_episode(episode, {}, "missing-source-seed"), embedded
        )

    def test_batch_rescore_prefers_original_seed_when_present(self) -> None:
        original = {"messages": [{"role": "user", "content": "original"}]}
        embedded = {"messages": [{"role": "user", "content": "embedded"}]}
        episode = {"turns": [{"sft_datum": embedded}]}
        self.assertIs(
            _seed_row_for_episode(episode, {"seed-1": original}, "seed-1"),
            original,
        )

    def test_compact_batch_episode_is_hydrated_from_native_sft_rows(self) -> None:
        def row(turn: int, tool: str, evidence_ids: list[str]) -> dict:
            state = {"evidence_store": {item: {} for item in evidence_ids}}
            return {
                "metadata": {"turn_index": turn},
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"current_harness_state": state}, separators=(",", ":")
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "The blocking gap is visible. This action closes it.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": tool, "arguments": {}},
                            }
                        ],
                    },
                ],
            }

        compact = {
            "quality": {"dimension_scores": {"terminal_integrity": 15}}
        }
        hydrated = _hydrate_episode_from_sft(
            compact,
            [row(1, "inspect_media", []), row(2, "stop", ["EA-001"])],
        )
        self.assertEqual(len(hydrated["turns"]), 2)
        self.assertEqual(
            hydrated["turns"][0]["environment_outcome"]["created_evidence_ids"],
            ["EA-001"],
        )
        self.assertTrue(hydrated["terminal_success"])

    def test_all_clean_and_repair_episodes_meet_strict_unified_rubric(self) -> None:
        report = json.loads(
            (OUTPUT / "student_closed_loop_unified_quality_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["summary"]["episodes"], 6)
        self.assertEqual(report["summary"]["accepted"], 6)
        self.assertEqual(report["summary"]["minimum_score"], 100)
        self.assertTrue(report["summary"]["all_accepted"])
        self.assertFalse(report["teacher_target_used"])

    def test_all_closed_loop_hard_negative_families_are_rejected(self) -> None:
        report = json.loads(
            (
                OUTPUT
                / "student_closed_loop_rubric_hard_negative_calibration.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["families"], 10)
        self.assertEqual(report["rejected"], 10)
        self.assertEqual(report["missed"], 0)
        self.assertTrue(report["all_hard_negatives_rejected"])

    def test_expanded_rubric_distinguishes_decision_reopen_and_bad_double_bind(self) -> None:
        report = json.loads(
            (
                OUTPUT
                / "student_closed_loop_expanded_quality_audit_before_v8.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["summary"]["episodes"], 6)
        self.assertEqual(report["summary"]["accepted"], 5)
        decision_repairs = [
            item
            for item in report["episodes"]
            if item.get("repair_subtype") == "decision_reopen"
        ]
        self.assertEqual(len(decision_repairs), 2)
        self.assertTrue(all(item["turns"] == 2 for item in decision_repairs))
        self.assertTrue(
            all(item["accepted_for_high_quality_sft"] for item in decision_repairs)
        )
        rejected = [
            item
            for item in report["episodes"]
            if not item["accepted_for_high_quality_sft"]
        ]
        self.assertEqual(len(rejected), 1)
        self.assertIn("execution_integrity", rejected[0]["failures"])
        self.assertIn("dependency_ordering", rejected[0]["failures"])
        self.assertIn("route_efficiency", rejected[0]["failures"])

    def test_v10_expanded_panel_meets_strict_unified_rubric(self) -> None:
        report = json.loads(
            (
                OUTPUT
                / "student_closed_loop_v10_native_constrained_expanded_quality_audit.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["summary"]["episodes"], 6)
        self.assertEqual(report["summary"]["accepted"], 6)
        self.assertEqual(report["summary"]["minimum_score"], 100)
        self.assertTrue(report["summary"]["all_accepted"])
        self.assertFalse(report["teacher_target_used"])
        self.assertEqual(
            sorted(
                item["turns"]
                for item in report["episodes"]
                if item.get("repair_subtype") == "decision_reopen"
            ),
            [2, 2],
        )


if __name__ == "__main__":
    unittest.main()
