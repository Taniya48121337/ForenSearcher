from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_rubric import evaluate  # noqa: E402


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class RubricMetaEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = {
            "clean": load("outputs/clean_trajectory.json"),
            "repair": load("outputs/repair_trajectory.json"),
            "graph": load("outputs/provenance_graph.json"),
            "oracle": load("oracle/claim_oracle.json"),
            "manifest": load("student_view/case_manifest.json"),
            "clean_sft": (ROOT / "outputs/sft_clean.jsonl").read_text(encoding="utf-8"),
            "repair_sft": (ROOT / "outputs/sft_repair.jsonl").read_text(encoding="utf-8"),
        }

    def score(self, **changes):
        fixture = copy.deepcopy(self.base)
        fixture.update(changes)
        return evaluate(fixture, write_report=False, raise_on_reject=False)

    def test_gold_bundle_is_accepted(self) -> None:
        report = self.score()
        self.assertTrue(report["accepted_for_sft"])
        self.assertEqual(report["grade"], "A_gold")

    def test_oracle_leak_is_hard_reject(self) -> None:
        leaked = self.base["clean_sft"] + self.base["oracle"]["annotation_text"]
        report = self.score(clean_sft=leaked)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G1", report["diagnosis"]["failed_gates"])

    def test_fabricated_evidence_id_is_hard_reject(self) -> None:
        clean = copy.deepcopy(self.base["clean"])
        clean["final_state"]["claim_ledger"][0]["field_bindings"][0]["evidence_ids"] = ["EA-FABRICATED-999"]
        report = self.score(clean=clean)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G2", report["diagnosis"]["failed_gates"])

    def test_real_but_incompatible_evidence_is_hard_reject(self) -> None:
        clean = copy.deepcopy(self.base["clean"])
        bindings = clean["final_state"]["claim_ledger"][0]["field_bindings"]
        manipulation = next(item for item in bindings if item["field"] == "manipulation_type")
        manipulation["evidence_ids"] = ["EA-META-001"]
        report = self.score(clean=clean)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G2", report["diagnosis"]["failed_gates"])
        self.assertIn("E1", report["diagnosis"]["failed_criteria"])

    def test_extra_distractor_evidence_is_not_rewarded(self) -> None:
        clean = copy.deepcopy(self.base["clean"])
        bindings = clean["final_state"]["claim_ledger"][0]["field_bindings"]
        manipulation = next(item for item in bindings if item["field"] == "manipulation_type")
        manipulation["evidence_ids"].append("EA-META-001")
        report = self.score(clean=clean)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G2", report["diagnosis"]["failed_gates"])
        self.assertIn("E1", report["diagnosis"]["failed_criteria"])

    def test_broken_provenance_edge_fails_closure(self) -> None:
        graph = copy.deepcopy(self.base["graph"])
        graph["edges"] = [
            edge
            for edge in graph["edges"]
            if not (
                edge["from"] == "EA-REPAIR-001"
                and edge["to"] == "FIELD-manipulation_type"
                and edge["relation"] == "repairs_support"
            )
        ]
        report = self.score(graph=graph)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("E3", report["diagnosis"]["failed_criteria"])

    def test_open_obligation_cannot_stop(self) -> None:
        clean = copy.deepcopy(self.base["clean"])
        clean["final_state"]["stop_decision"]["open_high_priority_obligations"] = ["OBL-1004-001"]
        report = self.score(clean=clean)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G3", report["diagnosis"]["failed_gates"])

    def test_text_side_mutation_is_hard_reject(self) -> None:
        repair = copy.deepcopy(self.base["repair"])
        repair["mutation"]["claim_text_changed_at_injection"] = True
        report = self.score(repair=repair)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G5", report["diagnosis"]["failed_gates"])

    def test_removed_evidence_cannot_be_reused(self) -> None:
        repair = copy.deepcopy(self.base["repair"])
        binding = repair["final_state"]["claim_ledger"][0]["field_bindings"][1]
        binding["evidence_ids"] = ["EA-OMIT-001"]
        repair["final_state"]["available_evidence_ids"].append("EA-OMIT-001")
        report = self.score(repair=repair)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("G6", report["diagnosis"]["failed_gates"])

    def test_global_reset_fails_off_target_invariance(self) -> None:
        repair = copy.deepcopy(self.base["repair"])
        for binding in repair["mutated_state"]["field_bindings"]:
            binding["status"] = "unsupported"
            binding["evidence_ids"] = []
        report = self.score(repair=repair)
        self.assertFalse(report["accepted_for_sft"])
        self.assertIn("M2", report["diagnosis"]["failed_criteria"])

    def test_skipping_verification_fails_process_rubric(self) -> None:
        clean = copy.deepcopy(self.base["clean"])
        clean["events"] = [event for event in clean["events"] if event["action"] != "verify_claim"]
        report = self.score(clean=clean)
        self.assertFalse(report["accepted_for_sft"])
        self.assertTrue({"P1", "P2", "S1"}.intersection(report["diagnosis"]["failed_criteria"]))


if __name__ == "__main__":
    unittest.main()
