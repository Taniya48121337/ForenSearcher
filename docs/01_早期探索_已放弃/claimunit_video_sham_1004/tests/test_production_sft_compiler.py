import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from production_sft_compiler import (
    DEFAULT_DATASET_ROOTS,
    _component_map,
    _evidence_choice_eligible_trajectory_keys,
    compile_production_sft,
)
from sft_context_compaction import CONTEXT_VIEW


class ProductionSFTCompilerTest(unittest.TestCase):
    def test_current_gold_pool_compiles_without_cross_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = compile_production_sft(DEFAULT_DATASET_ROOTS, output)
            self.assertEqual(report["summary"]["gold_cases"], 59)
            self.assertEqual(report["summary"]["sft_rows"], 600)
            self.assertEqual(report["summary"]["transition_rows"], 600)
            self.assertEqual(report["summary"]["sft_transition_pairs_verified"], 600)
            self.assertEqual(report["summary"]["route_preference_pairs"], 13)
            self.assertEqual(report["summary"]["strict_causal_core_rows"], 38)
            self.assertEqual(
                report["summary"]["strict_evidence_choice_core_rows"], 0
            )
            self.assertEqual(report["summary"]["training_qualified_sft_rows"], 0)
            self.assertFalse(report["training_qualified"])
            self.assertTrue(report["leakage_audit"]["passed"])
            self.assertEqual(
                report["leakage_audit"]["semantic_target_preexposure_after_compile"],
                0,
            )
            self.assertLessEqual(report["sampling"]["train_weight_max"], 2.0)
            self.assertGreaterEqual(report["sampling"]["train_weight_min"], 0.1)
            self.assertLess(
                report["sampling"]["effective_minimal_control_loss_mass_fraction"],
                0.20,
            )
            self.assertLess(
                report["sampling"]["weighted_trajectory_mass_dispersion_by_dataset"]
                ["VidDiffBench"]["max_min_ratio"],
                3.1,
            )
            self.assertTrue(
                all(
                    item["max_min_ratio"] < 1.7
                    for item in report["sampling"][
                        "weighted_entity_mass_dispersion_by_dataset"
                    ].values()
                )
            )

            case_splits = defaultdict(set)
            for split in ("train", "validation", "test"):
                rows = [
                    json.loads(line)
                    for line in (
                        output / split / "sft_candidate_all_not_for_training.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertFalse((output / split / "sft_all.jsonl").exists())
                self.assertTrue(
                    (
                        output
                        / split
                        / "sft_strict_evidence_choice_core_not_production_ready.jsonl"
                    ).exists()
                )
                for row in rows:
                    metadata = row["metadata"]
                    case_splits[metadata["case_id"]].add(split)
                    self.assertEqual(metadata["data_split"], split)
                    self.assertEqual(metadata["context_view"], CONTEXT_VIEW)
                    self.assertEqual(
                        metadata["policy_view"], "evidence_first_reduced_control_v1"
                    )
                    self.assertTrue(metadata["teacher_action_used"])
                    self.assertFalse(metadata["oracle_or_verifier_used_during_generation"])
                    self.assertEqual(len(row["messages"]), 3)
                    self.assertEqual([message["role"] for message in row["messages"]], ["system", "user", "assistant"])
                    compact_payload = json.loads(row["messages"][1]["content"])
                    self.assertIn("current_harness_state", compact_payload)
                    self.assertIn("past_execution_history", compact_payload)
                    history_text = json.dumps(compact_payload["past_execution_history"])
                    self.assertNotIn('"tool_name"', history_text)
                    self.assertNotIn('"arguments"', history_text)
                    self.assertTrue(metadata["private_locator_normalized"])
                    self.assertEqual(metadata["step_utility_verdict"], "indispensable")
                    self.assertTrue(metadata["counterfactual_removal_changes_outcome"])
                    self.assertIn(
                        metadata["learning_value_tier"],
                        {"core_reasoning", "supporting", "minimal_control"},
                    )
            self.assertTrue(case_splits)
            self.assertTrue(all(len(splits) == 1 for splits in case_splits.values()))
            self.assertEqual(report["context_compaction"]["rows_verified"], 600)
            self.assertEqual(report["context_compaction"]["rows_failed"], 0)
            self.assertEqual(report["context_compaction"]["private_locator_or_oracle_hits"], 0)
            self.assertGreater(report["context_compaction"]["visible_char_reduction_rate"], 0.50)

    def test_shared_vcsl_video_connects_cases_into_one_entity_group(self):
        cases = [
            {"case_id": "vcsl-a-b", "query_id": "a", "reference_id": "b"},
            {"case_id": "vcsl-b-c", "query_id": "b", "reference_id": "c"},
            {"case_id": "vcsl-x-y", "query_id": "x", "reference_id": "y"},
        ]
        mapping, entities = _component_map("VCSL", cases)
        self.assertEqual(mapping["vcsl-a-b"], mapping["vcsl-b-c"])
        self.assertNotEqual(mapping["vcsl-a-b"], mapping["vcsl-x-y"])
        self.assertIn("VCSL:b", entities[mapping["vcsl-a-b"]])

    def test_choice_eligibility_is_trajectory_level_not_row_level(self):
        def row(turn, group, eligible):
            metadata = {
                    "dataset": "VideoSham",
                    "case_id": "case-1",
                    "trajectory_id": "trajectory-1",
                    "source_kind": "clean",
                    "turn_index": turn,
                    "action_group": group,
                    "policy_choice_training_eligible": eligible,
                }
            if group == "external_measurement" and eligible:
                metadata.update(
                    {
                        "semantic_measurement_turn": True,
                        "visible_semantic_choice_count": 3,
                        "evidence_choice_search": {
                            "applicable": True,
                            "proposal_coverage_passed": True,
                            "selection_ambiguous": False,
                            "selected_preaction_method_fit_score": 1.0,
                            "selected_counterfactual_tool_swap_passed": True,
                            "selection_frozen_before_branch_execution": True,
                            "hindsight_outcome_used_for_selection": False,
                            "selected_branch_execution_admissible": True,
                            "proposal_retry_outcome_feedback_used": False,
                            "committed_replay_checked": True,
                            "deterministic_replay_match": True,
                        },
                    }
                )
            return {
                "metadata": metadata
            }

        good = [
            row(1, "context_inspection", True),
            row(2, "external_measurement", True),
            row(3, "harness_state", True),
        ]
        self.assertEqual(len(_evidence_choice_eligible_trajectory_keys(good)), 1)
        bad = good + [row(4, "external_measurement", False)]
        self.assertEqual(_evidence_choice_eligible_trajectory_keys(bad), set())
        forged = [row(1, "context_inspection", True), row(2, "external_measurement", True)]
        forged[1]["metadata"]["evidence_choice_search"][
            "hindsight_outcome_used_for_selection"
        ] = True
        self.assertEqual(_evidence_choice_eligible_trajectory_keys(forged), set())
        swappable = [row(1, "context_inspection", True), row(2, "external_measurement", True)]
        swappable[1]["metadata"]["evidence_choice_search"][
            "selected_counterfactual_tool_swap_passed"
        ] = False
        self.assertEqual(
            _evidence_choice_eligible_trajectory_keys(swappable), set()
        )


if __name__ == "__main__":
    unittest.main()
