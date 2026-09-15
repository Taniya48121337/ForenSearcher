import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from production_readiness_audit import DEFAULT_RUNS, audit_readiness


class ProductionReadinessAuditTest(unittest.TestCase):
    def test_measurement_reachability_is_separate_from_teacher_quality(self):
        report = audit_readiness(DEFAULT_RUNS)
        datasets = {item["dataset"]: item for item in report["datasets"]}
        self.assertEqual(datasets["VideoSham"]["case_count"], 30)
        self.assertEqual(datasets["VideoSham"]["measurement_reachability_rate"], 1.0)
        self.assertEqual(datasets["VCSL"]["case_count"], 30)
        self.assertEqual(datasets["VCSL"]["measurement_reachable_case_count"], 23)
        self.assertEqual(datasets["VCSL"]["measurement_reachability_rate"], 0.7667)
        self.assertEqual(len(datasets["VCSL"]["measurement_pool_supplement_sources"]), 1)
        self.assertEqual(datasets["VCSL"]["capability_family_counts"]["spd_official"], 2)
        viddiff = datasets["VidDiffBench"]
        self.assertEqual(viddiff["case_count"], 30)
        self.assertEqual(viddiff["teacher_started_case_count"], 21)
        self.assertEqual(viddiff["measurement_unreachable_case_count"], 17)
        self.assertEqual(viddiff["measurement_reachability_rate"], 0.4333)
        self.assertEqual(viddiff["teacher_started_gold_rate"], 1.0)
        self.assertTrue(viddiff["quality_canary_passed"])
        self.assertFalse(viddiff["scale_validation_passed"])
        self.assertEqual(report["status"], "quality_canaries_passed_scale_validation_pending")
        self.assertEqual(report["rubric_id"], "UTQR-5")
        self.assertEqual(datasets["VideoSham"]["quality_rescored_case_count"], 17)
        self.assertEqual(datasets["VCSL"]["quality_rescored_case_count"], 21)
        self.assertEqual(datasets["VidDiffBench"]["quality_rescored_case_count"], 21)
        self.assertEqual(report["hard_negative_calibration"]["count"], 862)
        self.assertEqual(report["hard_negative_calibration"]["detection_rate"], 1.0)
        student = report["student_closed_loop_evidence"]
        self.assertEqual(
            student["held_out_cases_by_dataset"],
            {"VCSL": 4, "VidDiffBench": 15, "VideoSham": 15},
        )
        self.assertEqual(student["held_out_target_per_dataset"], 30)
        self.assertEqual(
            student["held_out_shortfall_by_dataset"],
            {"VideoSham": 15, "VCSL": 26, "VidDiffBench": 15},
        )
        self.assertEqual(student["total_additional_held_out_cases_required"], 56)
        self.assertEqual(student["episodes"], 40)
        self.assertEqual(student["accepted_episodes"], 40)
        self.assertEqual(student["actions"], 264)
        self.assertEqual(student["accepted_sft_rows"], 264)
        self.assertEqual(student["diversity_balanced_sft_rows"], 245)
        self.assertTrue(student["quality_canary_passed"])
        self.assertFalse(student["scale_validation_passed"])
        self.assertEqual(student["higher_order_pass_k"]["measured_cases"], 25)
        self.assertEqual(
            student["higher_order_pass_k"]["evaluated_case_coverage"], 0.7353
        )
        self.assertFalse(
            student["scale_requirements"][
                "independent_rollouts_per_case_at_least_2"
            ]
        )
        self.assertFalse(
            student["scale_requirements"]["held_out_cases_per_dataset_at_least_30"]
        )


if __name__ == "__main__":
    unittest.main()
