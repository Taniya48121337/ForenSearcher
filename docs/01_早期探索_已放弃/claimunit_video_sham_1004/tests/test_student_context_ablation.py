import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "outputs" / "student_context_ablation_v2"


class StudentContextAblationTest(unittest.TestCase):
    def test_three_arms_share_rows_splits_and_targets(self):
        manifest = json.loads((ABLATION / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_counts"], {"train": 472, "validation": 64, "test": 64})
        self.assertEqual(manifest["invariants"]["row_count"], 600)
        self.assertTrue(manifest["invariants"]["all_targets_identical"])
        self.assertTrue(manifest["invariants"]["all_splits_identical"])
        self.assertEqual(manifest["naive_tail_budget"]["max_budget_excess_tokens"], 0)
        self.assertGreater(manifest["naive_tail_budget"]["rows_history_truncated"], 0)
        self.assertGreater(manifest["naive_tail_budget"]["mean_utilization_when_truncated"], 0.99)
        for arm in manifest["arms"].values():
            self.assertEqual(arm["rows"], 600)
            self.assertEqual(arm["private_locator_or_oracle_hits"], {})

    def test_local_qlora_smoke_completed_with_target_only_loss(self):
        report = json.loads(
            (ABLATION / "smoke_compact_1step" / "run_report.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["assistant_target_only_loss"])
        self.assertEqual(report["max_steps"], 1)
        self.assertGreater(report["train_metrics"]["train_loss"], 0)
        self.assertLess(report["peak_cuda_memory_gib"], 32)


if __name__ == "__main__":
    unittest.main()
