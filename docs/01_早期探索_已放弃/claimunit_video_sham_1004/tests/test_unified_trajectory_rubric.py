import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from unified_trajectory_rubric import (  # noqa: E402
    _adjudication_quality,
    evaluate_bundle,
    evaluate_run,
    load_saved_bundle,
)


VIDEOSHAM_CASE = (
    ROOT / "outputs" / "batch" / "videosham_seed_20260818_n10"
    / "cases" / "videosham-1007"
)
VCSL_LOW_PRECISION_CASE = (
    ROOT / "outputs" / "batch" / "vcsl_seed_20260818_n10" / "cases"
    / "cb4b162ca3554c17b21af9cabd801014-2d1b2c073ce0466bb5d92e7659e536bb"
)
VCSL_SUB_GOLD_RECALL_CASE = (
    ROOT
    / "outputs"
    / "utqr_v2_vcsl_10"
    / "vcsl_seed_20260818_n10_bon1"
    / "cases"
    / "004_010af0a8_4bc2ef10"
    / "attempts"
    / "01_auto"
)
VIDDIFF_DECISION_REPAIR_CASE = (
    ROOT / "outputs" / "sft_v2_best_of_n_remaining" / "viddiff_rejected_set"
    / "viddiff_adaptive_seed_20260818_n6_bon3" / "cases" / "surgery_57"
)
VIDDIFF_TRUE_CONFLICT_CASE = (
    ROOT / "outputs" / "sft_v2_batch_10" / "viddiff_adaptive_v8_final2"
    / "viddiff_adaptive_seed_20260818_n2" / "cases" / "surgery_158"
)
VIDDIFF_LOW_CONFIDENCE_AGREEMENT_CASE = (
    ROOT / "outputs" / "sft_v2_batch_10" / "viddiff_adaptive_v8_final2"
    / "viddiff_adaptive_seed_20260818_n2" / "cases" / "surgery_80"
)
VIDDIFF_NATURAL_ABSTENTION_CASE = (
    ROOT
    / "outputs"
    / "utqr_v2_viddiff_reachable_remaining"
    / "viddiff_adaptive_seed_20260818_n5_bon4"
    / "cases"
    / "surgery_57"
    / "attempts"
    / "02_global_first"
)
VIDDIFF_SINGLE_VERIFIED_MEASUREMENT_CASE = (
    ROOT
    / "outputs"
    / "scale30_viddiff_teacher_25_29_repair121"
    / "viddiff_verified_seed_20260818_accepted_1"
    / "cases"
    / "surgery_121"
    / "attempts"
    / "01_legacy"
)


class UnifiedTrajectoryRubricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.video_bundle, cls.video_validation = load_saved_bundle(VIDEOSHAM_CASE)
        cls.viddiff_bundle, cls.viddiff_validation = load_saved_bundle(VIDDIFF_DECISION_REPAIR_CASE)
        cls.viddiff_conflict_bundle, cls.viddiff_conflict_validation = load_saved_bundle(
            VIDDIFF_TRUE_CONFLICT_CASE
        )
        cls.viddiff_agreement_bundle, cls.viddiff_agreement_validation = load_saved_bundle(
            VIDDIFF_LOW_CONFIDENCE_AGREEMENT_CASE
        )
        cls.viddiff_abstention_bundle, cls.viddiff_abstention_validation = load_saved_bundle(
            VIDDIFF_NATURAL_ABSTENTION_CASE
        )

    def test_verified_grounded_bundle_is_gold(self):
        report = evaluate_bundle(copy.deepcopy(self.video_bundle), copy.deepcopy(self.video_validation))
        self.assertTrue(report["accepted_for_sft"])
        self.assertGreaterEqual(report["total_score"], 92)

    def test_recall_only_oversized_localization_is_rejected(self):
        bundle, validation = load_saved_bundle(VCSL_LOW_PRECISION_CASE)
        report = evaluate_bundle(bundle, validation)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("H1", report["diagnosis"]["failed_gates"])
        self.assertLess(validation["localization_metrics"]["precision"], 0.1)

    def test_vcsl_gold_requires_each_localization_metric_at_least_point_85(self):
        bundle, validation = load_saved_bundle(VCSL_SUB_GOLD_RECALL_CASE)
        report = evaluate_bundle(bundle, validation)
        q1 = next(item for item in report["criteria"] if item["criterion_id"] == "Q1")

        self.assertEqual(validation["localization_metrics"]["recall"], 0.75)
        self.assertFalse(report["accepted_for_sft"])
        self.assertEqual(report["grade"], "B_repair")
        self.assertEqual(q1["normalized_score"], 0.75)
        self.assertIn("Q1", report["diagnosis"]["failed_criteria"])

    def test_oracle_leak_is_hard_rejected(self):
        bundle = copy.deepcopy(self.video_bundle)
        bundle["clean_turn_sft"][0]["messages"].append(
            {"role": "user", "content": "oracle_direction=a"}
        )
        report = evaluate_bundle(bundle, copy.deepcopy(self.video_validation))
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("H4", report["diagnosis"]["failed_gates"])

    def test_missing_field_support_is_hard_rejected(self):
        bundle = copy.deepcopy(self.video_bundle)
        for episode_name in ("clean_episode", "repair_episode"):
            fields = bundle[episode_name]["final_state"]["candidate_claims"][0]["fields"]
            for field_state in fields.values():
                field_state["support_ids"] = ["EA-FABRICATED-999"]
        report = evaluate_bundle(bundle, copy.deepcopy(self.video_validation))
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("H3", report["diagnosis"]["failed_gates"])

    def test_redundant_or_noncausal_positive_steps_are_hard_rejected(self):
        bundle = copy.deepcopy(self.video_bundle)
        bundle["step_utility_report"]["passed"] = False
        for steps in bundle["step_utility_report"]["episodes"].values():
            steps[0]["causally_necessary"] = False
            steps[0]["downstream_consumed"] = False
            steps[0]["verdict"] = "redundant"
        report = evaluate_bundle(bundle, copy.deepcopy(self.video_validation))
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("H5", report["diagnosis"]["failed_gates"])

    def test_copy_pasted_generic_reasoning_fails_semantic_minimum(self):
        bundle = copy.deepcopy(self.video_bundle)
        generic = (
            "The latest observation established something useful, and a remaining gap still exists. "
            "The next action should make progress because it is generally appropriate for the task. "
            "This generic statement deliberately omits concrete visible evidence, field values, and tool-specific justification."
        )
        for episode_name in ("clean_episode", "repair_episode"):
            for turn in bundle[episode_name]["turn_history"]:
                turn["reasoning"] = generic
        report = evaluate_bundle(bundle, copy.deepcopy(self.video_validation))
        q5 = next(item for item in report["criteria"] if item["criterion_id"] == "Q5")
        self.assertFalse(report["accepted_for_sft"])
        self.assertLess(q5["normalized_score"], 0.8)

    def test_decision_reopen_with_remeasurement_is_hard_rejected(self):
        bundle = copy.deepcopy(self.viddiff_bundle)
        bundle["repair_episode"]["turn_history"][0]["observation"]["authority"] = "external_tool_environment"
        report = evaluate_bundle(bundle, copy.deepcopy(self.viddiff_validation))
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("H6", report["diagnosis"]["failed_gates"])

    def test_batch_baseline_detects_hidden_vcsl_quality_gap(self):
        report = evaluate_run(ROOT / "outputs" / "batch" / "vcsl_seed_20260818_n10")
        self.assertEqual(report["summary"]["scored_cases"], 10)
        self.assertLess(report["summary"]["gold_rate"], 1.0)
        self.assertGreater(report["summary"]["hard_gate_failure_cases"], 0)

    def test_natural_semantic_conflict_adjudication_is_gold_without_literal_tool_names(self):
        report = evaluate_bundle(
            copy.deepcopy(self.viddiff_conflict_bundle),
            copy.deepcopy(self.viddiff_conflict_validation),
        )
        q7 = next(item for item in report["criteria"] if item["criterion_id"] == "Q7")
        self.assertTrue(report["accepted_for_sft"])
        self.assertEqual(q7["normalized_score"], 1.0)

    def test_low_confidence_measurement_can_be_resolved_by_grounded_cross_method_agreement(self):
        report = evaluate_bundle(
            copy.deepcopy(self.viddiff_agreement_bundle),
            copy.deepcopy(self.viddiff_agreement_validation),
        )
        q7 = next(item for item in report["criteria"] if item["criterion_id"] == "Q7")
        self.assertTrue(report["accepted_for_sft"])
        self.assertEqual(q7["normalized_score"], 1.0)

    def test_natural_both_support_abstention_language_is_semantically_calibrated(self):
        report = evaluate_bundle(
            copy.deepcopy(self.viddiff_abstention_bundle),
            copy.deepcopy(self.viddiff_abstention_validation),
        )
        q7 = next(item for item in report["criteria"] if item["criterion_id"] == "Q7")

        self.assertTrue(report["accepted_for_sft"])
        self.assertEqual(q7["normalized_score"], 1.0)

    def test_tool_name_and_confidence_only_conflict_story_is_rejected(self):
        bundle = copy.deepcopy(self.viddiff_conflict_bundle)
        shallow = (
            "compare_action_refined_symmetric, compare_uniform_timeline, and "
            "compare_global_motion_symmetric conflict. I select EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001 "
            "because it has the highest confidence; therefore it is strongest."
        )
        for turn in bundle["clean_episode"]["turn_history"]:
            if turn["tool_name"] in {"create_candidate_claim", "bind_evidence"}:
                turn["reasoning"] = shallow
        report = evaluate_bundle(bundle, copy.deepcopy(self.viddiff_conflict_validation))
        q7 = next(item for item in report["criteria"] if item["criterion_id"] == "Q7")
        self.assertFalse(report["accepted_for_sft"])
        self.assertLess(q7["normalized_score"], 0.8)
        self.assertIn(
            "viddiff_adjudication_relies_on_confidence_or_vote_only",
            q7["failure_codes"],
        )

    def test_single_verified_measurement_uses_calibration_plus_visible_basis(self):
        bundle, validation = load_saved_bundle(VIDDIFF_SINGLE_VERIFIED_MEASUREMENT_CASE)
        report = evaluate_bundle(bundle, validation)
        q7 = next(item for item in report["criteria"] if item["criterion_id"] == "Q7")

        self.assertTrue(report["accepted_for_sft"])
        self.assertEqual(q7["normalized_score"], 1.0)

    def test_explicit_adjudication_requires_exact_partition_and_matching_bind(self):
        adjudicate = {
            "tool_name": "adjudicate_evidence",
            "arguments": {
                "claim_id": "CC-001",
                "target_fields": ["preferred_video"],
                "selected_evidence_ids": ["EA-SELECTED-001"],
                "rejected_evidence": [
                    {
                        "evidence_id": "EA-REJECTED-001",
                        "defect_codes": ["order_inconsistent"],
                    }
                ],
            },
            "reasoning": (
                "I select EA-SELECTED-001 because it fits the visible claim, while "
                "EA-REJECTED-001 is order-inconsistent and therefore weaker."
            ),
            "observation": {
                "adjudication": {
                    "considered_evidence_ids": ["EA-SELECTED-001", "EA-REJECTED-001"]
                }
            },
        }
        bind = {
            "tool_name": "bind_evidence",
            "arguments": {
                "claim_id": "CC-001",
                "binding_items": [
                    {"field": "preferred_video", "support_ids": ["EA-SELECTED-001"]}
                ],
            },
        }
        bundle = {
            "clean_episode": {"turn_history": [adjudicate, bind]},
            "repair_episode": {"turn_history": []},
        }
        score, structural, reasoning, stats = _adjudication_quality(bundle)
        self.assertEqual(score, 1.0)
        self.assertEqual(structural, [])
        self.assertEqual(reasoning, [])
        self.assertEqual(stats["explicit_adjudication_records"], 1)

        broken = copy.deepcopy(bundle)
        broken["clean_episode"]["turn_history"][0]["arguments"]["rejected_evidence"] = []
        broken["clean_episode"]["turn_history"][1]["arguments"]["binding_items"][0][
            "support_ids"
        ] = ["EA-REJECTED-001"]
        score, structural, reasoning, _stats = _adjudication_quality(broken)
        self.assertLess(score, 1.0)
        self.assertTrue(any("omits_or_invents_competitor" in item for item in structural))
        self.assertTrue(any("bypasses_selected_evidence" in item for item in structural))


if __name__ == "__main__":
    unittest.main()
