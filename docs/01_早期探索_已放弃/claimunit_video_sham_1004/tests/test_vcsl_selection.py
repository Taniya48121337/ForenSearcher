import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vcsl_batch import LOCALIZER_SPECS, VCSLFeatureEnvironment, _verify_selected_localization
from vcsl_probe import DEFAULT_FEATURE_ROOT, select_cases
from vcsl_rejection_sampler import DEFAULT_REPORTS, run


class VCSLSelectionTest(unittest.TestCase):
    def test_best_of_localizer_tree_accepts_21_without_student_oracle_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            report = run(
                argparse.Namespace(
                    reports=DEFAULT_REPORTS,
                    output_root=output_root,
                )
            )
            self.assertEqual(report["summary"]["accepted"], 21)
            self.assertEqual(report["summary"]["rejected"], 9)
            self.assertEqual(report["summary"]["student_export_oracle_leak_tokens"], [])
            student_path = output_root / report["run_id"] / "accepted_localizations.student.jsonl"
            serialized = student_path.read_text(encoding="utf-8")
            self.assertNotIn("oracle_boxes", serialized)
            self.assertNotIn('"metrics"', serialized)

    def test_new_formal_localizer_tools_execute_without_oracle_fields(self):
        case = select_cases(20260818, 1)[0]
        for variant in ("hv_segment", "local_alignment", "temporal_network"):
            environment = VCSLFeatureEnvironment(
                case=case,
                feature_root=DEFAULT_FEATURE_ROOT,
                routing_variant=variant,
            )
            environment.execute(
                "inspect_pair_features",
                {"query_ref": "query", "reference_ref": "reference"},
            )
            spec = LOCALIZER_SPECS[variant]
            try:
                observation = environment.execute(
                    spec["tool_name"],
                    {
                        "query_ref": "query",
                        "reference_ref": "reference",
                        "method": spec["method"],
                    },
                )
            except ValueError as error:
                self.assertIn("found no supported copied segment", str(error))
                continue
            self.assertNotIn("oracle", json.dumps(observation, default=str).lower())

    def test_selected_localization_is_exactly_replayable(self):
        selection_path = (
            ROOT
            / "outputs"
            / "selection"
            / "vcsl_best_of_tree_seed_20260818_n30"
            / "accepted_localizations.student.jsonl"
        )
        item = json.loads(selection_path.read_text(encoding="utf-8").splitlines()[0])
        case = next(case for case in select_cases(20260818, 30) if case["pair_id"] == item["pair_id"])
        result = _verify_selected_localization(case, item, DEFAULT_FEATURE_ROOT)
        self.assertTrue(result["exact_replay"])


if __name__ == "__main__":
    unittest.main()
