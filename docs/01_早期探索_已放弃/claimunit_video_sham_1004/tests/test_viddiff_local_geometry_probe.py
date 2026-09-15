import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_local_geometry_probe import _metrics, _predict, _split  # noqa: E402


class VidDiffLocalGeometryProbeTest(unittest.TestCase):
    def test_prediction_is_swap_consistent(self) -> None:
        self.assertEqual(_predict(0.8, 0.2, 0.1), "a")
        self.assertEqual(_predict(0.2, 0.8, 0.1), "b")
        self.assertEqual(_predict(0.8, 0.76, 0.1), "c")

    def test_split_is_deterministic_and_disjoint(self) -> None:
        self.assertEqual(_split("surgery_56"), "development")
        self.assertEqual(_split("surgery_57"), "holdout")

    def test_metrics_penalize_majority_only_predictor(self) -> None:
        rows = [
            {"oracle_direction": "a", "score_a": 1.0, "score_b": 1.0},
            {"oracle_direction": "b", "score_a": 1.0, "score_b": 1.0},
            {"oracle_direction": "c", "score_a": 1.0, "score_b": 1.0},
        ]
        report = _metrics(rows, margin=0.1)
        self.assertEqual(report["accuracy"], 0.3333)
        self.assertEqual(report["balanced_accuracy"], 0.3333)
        self.assertEqual(report["non_c_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
