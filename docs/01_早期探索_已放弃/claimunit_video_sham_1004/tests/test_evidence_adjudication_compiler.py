from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_evidence_adjudication_sft import (  # noqa: E402
    _compile_parallel_conflict_row,
    _student_visible_privacy_problems,
)
from prepare_progressive_evidence_assessment_sft import _compile_case  # noqa: E402
from run_student_closed_loop_batch import (  # noqa: E402
    DEFAULT_SFT_ROOT,
    _all_viddiff_items,
    _case_id,
    _first_rows,
    _mode,
)


class EvidenceAdjudicationCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed_row = next(
            row
            for row in _first_rows(DEFAULT_SFT_ROOT, "train")
            if _case_id(row) == "viddiff-surgery_80"
            and _mode(row) == "repair"
        )
        cls.seed_row = seed_row
        cls.viddiff_items = _all_viddiff_items()
        cls.row, cls.audit = _compile_parallel_conflict_row(
            seed_row, cls.viddiff_items
        )
        cls.progressive_rows, cls.progressive_audit = _compile_case(
            seed_row, cls.viddiff_items
        )

    def test_compiles_one_label_blind_executable_adjudication(self) -> None:
        self.assertIsNotNone(self.row)
        function = self.row["messages"][-1]["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "adjudicate_evidence")
        arguments = function["arguments"]
        selected = set(arguments["selected_evidence_ids"])
        rejected = {
            item["evidence_id"] for item in arguments["rejected_evidence"]
        }
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(rejected), 2)
        self.assertFalse(selected.intersection(rejected))

    def test_selected_support_binds_and_rejected_support_does_not(self) -> None:
        self.assertEqual(self.audit["execution_status"], "accepted")
        self.assertEqual(
            self.audit["post_adjudication_binding_status"], "accepted"
        )
        self.assertEqual(
            self.audit["rejected_bind_counterfactual_status"], "rejected"
        )
        self.assertEqual(
            self.audit["rejected_bind_counterfactual_reason"],
            "evidence_not_selected_by_latest_adjudication",
        )

    def test_student_payload_contains_no_disallowed_external_fields(self) -> None:
        payload = json.loads(self.row["messages"][1]["content"])
        self.assertEqual(_student_visible_privacy_problems(payload), [])
        mutation = payload["task"]["mutation"]
        self.assertNotIn("oracle_or_verifier_used", mutation)
        self.assertFalse(mutation["media_changed"])
        self.assertFalse(mutation["claim_values_changed"])

    def test_progressive_compiler_emits_three_assessments_then_adjudication(self) -> None:
        tools = [
            row["messages"][-1]["tool_calls"][0]["function"]["name"]
            for row in self.progressive_rows
        ]
        self.assertEqual(
            tools,
            [
                "assess_evidence_candidate",
                "assess_evidence_candidate",
                "assess_evidence_candidate",
                "adjudicate_evidence",
            ],
        )
        self.assertEqual(self.progressive_audit["assessment_count"], 3)
        self.assertEqual(
            [
                (
                    audit["pending_candidates_before"],
                    audit["pending_candidates_after"],
                )
                for audit in self.progressive_audit["assessment_audits"]
            ],
            [(3, 2), (2, 1), (1, 0)],
        )
        self.assertEqual(
            self.progressive_audit["final_adjudication_audit"][
                "post_adjudication_binding_status"
            ],
            "accepted",
        )


if __name__ == "__main__":
    unittest.main()
