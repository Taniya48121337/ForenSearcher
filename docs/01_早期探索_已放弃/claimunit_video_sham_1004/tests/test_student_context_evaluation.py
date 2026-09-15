import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_student_context_arm import (  # noqa: E402
    _parse_generated_call,
    _arguments_semantically_equal,
    _generated_sections,
    _same_argument_shape,
    _spread_subset_by_dataset,
    _stratified_subset,
)


class StudentContextEvaluationTest(unittest.TestCase):
    def test_generated_sections_separate_reasoning_from_payload(self):
        reasoning, payload = _generated_sections(
            'Inspect the gap.\n<tool_call>{"name":"x","arguments":{}}</tool_call>'
        )
        self.assertEqual(reasoning, "Inspect the gap.")
        self.assertTrue(payload.startswith("<tool_call>"))

    def test_create_claim_ids_are_alpha_equivalent(self):
        reference = {"claim_id": "hidden-reference-id", "field_values": {"x": 1}}
        generated = {"claim_id": "student-chosen-id", "field_values": {"x": 1}}
        self.assertTrue(
            _arguments_semantically_equal("create_candidate_claim", generated, reference)
        )
        generated["field_values"]["x"] = 2
        self.assertFalse(
            _arguments_semantically_equal("create_candidate_claim", generated, reference)
        )

    def test_tool_call_parser_and_shape_check(self):
        text = '<tool_call>{"name":"inspect_media","arguments":{"media_refs":["a","b"]}}</tool_call>'
        name, arguments = _parse_generated_call(text)
        self.assertEqual(name, "inspect_media")
        self.assertEqual(arguments, {"media_refs": ["a", "b"]})
        self.assertTrue(_same_argument_shape(arguments, {"media_refs": ["source", "suspect"]}))
        self.assertFalse(_same_argument_shape({"media_refs": "a"}, {"media_refs": ["a"]}))

    def test_bind_evidence_semantics_ignore_only_list_order(self):
        first = {
            "claim_id": "C1",
            "binding_items": [
                {"field": "a", "support_ids": ["E2", "E1"]},
                {"field": "b", "support_ids": ["E3"]},
            ],
        }
        reordered = {
            "claim_id": "C1",
            "binding_items": [
                {"field": "b", "support_ids": ["E3"]},
                {"field": "a", "support_ids": ["E1", "E2"]},
            ],
        }
        self.assertTrue(_arguments_semantically_equal("bind_evidence", first, reordered))
        reordered["binding_items"][1]["support_ids"] = ["E9"]
        self.assertFalse(_arguments_semantically_equal("bind_evidence", first, reordered))

    def test_validation_subset_is_dataset_balanced(self):
        path = (
            ROOT
            / "outputs"
            / "student_context_ablation_v2"
            / "decision_sufficient_state_v1"
            / "validation"
            / "sft_all.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        selected = _stratified_subset(rows, 18)
        self.assertEqual(len(selected), 18)
        counts = Counter(row["metadata"]["dataset"] for row in selected)
        self.assertEqual(counts, {"VideoSham": 6, "VCSL": 6, "VidDiffBench": 6})
        self.assertGreaterEqual(len({row["metadata"]["target_tool"] for row in selected}), 10)
        generated = _spread_subset_by_dataset(selected, 9)
        generated_counts = Counter(row["metadata"]["dataset"] for row in generated)
        self.assertEqual(generated_counts, {"VideoSham": 3, "VCSL": 3, "VidDiffBench": 3})
        self.assertGreaterEqual(len({row["metadata"]["target_tool"] for row in generated}), 7)


if __name__ == "__main__":
    unittest.main()
