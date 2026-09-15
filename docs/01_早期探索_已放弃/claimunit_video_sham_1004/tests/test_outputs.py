from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class OutputContractTest(unittest.TestCase):
    def test_validation_passes(self) -> None:
        report = load_json("outputs/validation_report.json")
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))

    def test_real_case_is_boundary_omission(self) -> None:
        clean = load_json("outputs/clean_trajectory.json")
        claim = clean["final_state"]["claim_ledger"][0]["claim"]
        self.assertEqual(claim["claim_family"], "selective_temporal_omission")
        self.assertEqual(len(claim["omitted_source_intervals_s"]), 2)
        self.assertGreater(claim["estimated_removed_time_s"], 2.9)

    def test_dropout_then_repair(self) -> None:
        mutated = load_json("outputs/mutated_state.json")
        self.assertNotIn("EA-OMIT-001", set(mutated["available_evidence_ids"]))
        repaired = load_json("outputs/repair_trajectory.json")
        self.assertEqual(repaired["repair_evidence"]["evidence_id"], "EA-REPAIR-001")
        claim = repaired["final_state"]["claim_ledger"][0]["claim"]
        self.assertIn("EA-REPAIR-001", claim["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
