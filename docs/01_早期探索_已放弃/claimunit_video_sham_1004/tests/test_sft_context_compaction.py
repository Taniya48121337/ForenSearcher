import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sft_context_compaction import audit_jsonl, compact_sft_row, validate_compact_sft_row  # noqa: E402
from student_tool_contract import student_tool_names  # noqa: E402


FULL_PRODUCTION = ROOT / "outputs" / "production_sft_v3_scale"


class SFTContextCompactionTest(unittest.TestCase):
    def test_all_current_gold_rows_have_lossless_private_clean_compact_views(self):
        report = audit_jsonl([
            FULL_PRODUCTION / split / "sft_all.jsonl"
            for split in ("train", "validation", "test")
        ])
        self.assertEqual(report["rows"], 600)
        self.assertEqual(report["passed"], 600)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["failure_counts"], {})
        self.assertEqual(report["privacy_hit_counts"], {})
        self.assertLess(report["context_char_ratio"]["mean"], 0.50)
        self.assertLess(report["compact_visible_chars"], report["source_visible_chars"])

    def test_compact_row_exposes_exactly_eligible_executable_tool_schemas(self):
        source_path = FULL_PRODUCTION / "validation" / "sft_all.jsonl"
        import json

        source = json.loads(source_path.read_text(encoding="utf-8").splitlines()[0])
        compact = compact_sft_row(source)
        audit = validate_compact_sft_row(source, compact)
        payload = json.loads(compact["messages"][1]["content"])
        eligible = payload["current_harness_state"]["eligible_tools"]
        self.assertEqual(student_tool_names(compact["tools"]), eligible)
        self.assertTrue(audit["checks"]["tool_schemas_match_executable_teacher_contract"])
        for tool in compact["tools"]:
            parameters = tool["function"]["parameters"]
            self.assertNotIn("reasoning", parameters.get("properties", {}))
            self.assertNotIn("reasoning", parameters.get("required", []))

    def test_missing_schema_and_internal_target_arguments_are_rejected(self):
        import copy
        import json

        source_path = FULL_PRODUCTION / "validation" / "sft_all.jsonl"
        sources = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
        bind_source = next(
            row for row in sources if row["metadata"]["target_tool"] == "bind_evidence"
        )
        compact = compact_sft_row(bind_source)
        missing = copy.deepcopy(compact)
        missing["tools"] = []
        missing_audit = validate_compact_sft_row(bind_source, missing)
        self.assertFalse(missing_audit["checks"]["target_tool_has_student_schema"])

        internal = copy.deepcopy(compact)
        function = internal["messages"][-1]["tool_calls"][0]["function"]
        items = function["arguments"].pop("binding_items")
        function["arguments"]["bindings"] = {
            item["field"]: item["support_ids"] for item in items
        }
        internal_audit = validate_compact_sft_row(bind_source, internal)
        self.assertFalse(internal_audit["checks"]["target_arguments_validate_student_schema"])


if __name__ == "__main__":
    unittest.main()
