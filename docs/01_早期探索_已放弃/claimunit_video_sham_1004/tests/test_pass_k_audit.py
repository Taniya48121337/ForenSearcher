import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pass_k_audit import audit_pass_k  # noqa: E402


class PassKAuditTest(unittest.TestCase):
    def test_strict_combinatorial_pass_k(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for index, passed in enumerate((True, True, False), start=1):
                root = Path(directory) / str(index)
                root.mkdir()
                case = {
                    "case_id": "case-1",
                    "status": "passed" if passed else "rubric_rejected",
                    "rubric_score": 98 if passed else 91,
                    "rubric_failed_gates": [],
                    "rubric_failed_criteria": [] if passed else ["Q7"],
                }
                (root / "batch_report.json").write_text(
                    json.dumps({"cases": [case]}), encoding="utf-8"
                )
                roots.append(root)
            report = audit_pass_k(roots)
            self.assertEqual(report["pass_k"], {"pass^1": 0.666667, "pass^2": 0.333333, "pass^3": 0.0})
            self.assertEqual(report["pass_at_k"], {"pass@1": 0.666667, "pass@2": 1.0, "pass@3": 1.0})
            self.assertFalse(report["route_portfolio_attempts_count_as_independent"])

    def test_student_batch_reports_and_distinct_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for seed, accepted in ((101, True), (202, True), (303, False)):
                root = Path(directory) / str(seed)
                root.mkdir()
                report = {
                    "generation_mode": "stochastic_local_model",
                    "generation_parameters": {
                        "do_sample": True,
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "base_seed": seed,
                    },
                    "episodes": [
                        {
                            "case_id": "student-case-1",
                            "mode": "clean",
                            "quality": {
                                "accepted_for_high_quality_sft": accepted,
                                "score": 100 if accepted else 80,
                                "failures": {} if accepted else {"terminal": "failed"},
                            },
                            "tool_sequence": ["inspect_media", "stop"],
                        }
                    ],
                }
                (root / "batch_report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                roots.append(root)
            audited = audit_pass_k(roots)
            self.assertEqual(audited["gold_rollouts"], 2)
            self.assertEqual(audited["pass_k"]["pass^2"], 0.333333)
            self.assertTrue(audited["distinct_recorded_seeds"])
            self.assertEqual(audited["rollout_seeds"], [101, 202, 303])


if __name__ == "__main__":
    unittest.main()
