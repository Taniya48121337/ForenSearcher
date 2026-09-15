import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_consolidate import _prefer_new_case  # noqa: E402


class VidDiffConsolidationPreferenceTest(unittest.TestCase):
    def test_failed_retry_does_not_replace_verified_gold(self) -> None:
        gold = {"status": "passed", "rubric_score": 98.0}
        failed = {"status": "rubric_rejected", "rubric_score": 99.0}
        self.assertFalse(_prefer_new_case(gold, failed))

    def test_higher_scoring_verified_gold_replaces_older_gold(self) -> None:
        old = {"status": "passed", "rubric_score": 97.0}
        new = {"status": "passed", "rubric_score": 99.0}
        self.assertTrue(_prefer_new_case(old, new))

    def test_latest_diagnosis_replaces_older_nonpassing_row(self) -> None:
        old = {"status": "execution_failed"}
        new = {"status": "oracle_unreachable_candidate_set"}
        self.assertTrue(_prefer_new_case(old, new))


if __name__ == "__main__":
    unittest.main()
