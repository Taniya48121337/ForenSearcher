from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "outputs/student_external_contract_v1/decision_sufficient_state_v5_action_affordances"
EXECUTABLE_ARM = (
    ROOT
    / "outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances"
)


def _rows(split: str) -> list[dict]:
    path = ARM / split / "sft_all.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class StudentActionAffordanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation = _rows("validation")

    def test_conflicting_visible_evidence_is_filtered_by_claim_value(self) -> None:
        row_id = (
            "VidDiffBench:viddiff-surgery_155:LLM-TRJ-VIDDIFF-SURGERY_155-"
            "CLEAN-001:clean:6"
        )
        row = next(item for item in self.validation if item["metadata"]["row_id"] == row_id)
        payload = json.loads(row["messages"][1]["content"])
        candidates = payload["current_harness_state"]["state_action_affordances"][
            "binding_candidates"
        ]["claim_surgery_155_1"]
        expected = ["EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-001"]
        for field in (
            "preferred_video",
            "video_a_interval_s",
            "video_b_interval_s",
            "measurement_method",
        ):
            self.assertEqual(candidates[field], expected)

    def test_supported_open_fields_expose_only_meaningful_commit_transition(self) -> None:
        row = next(
            item
            for item in self.validation
            if item["metadata"]["row_id"].endswith("VIDDIFF-SURGERY_155-CLEAN-001:clean:7")
        )
        payload = json.loads(row["messages"][1]["content"])
        transitions = payload["current_harness_state"]["state_action_affordances"][
            "meaningful_decision_transitions"
        ]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["decision_status"], "committed")
        self.assertEqual(len(transitions[0]["fields"]), 6)

    def test_viddiff_dropout_exposes_same_method_restoration_candidate(self) -> None:
        row_id = (
            "VidDiffBench:viddiff-surgery_155:LLM-TRJ-VIDDIFF-SURGERY_155-"
            "REPAIR-001:repair:2"
        )
        row = next(item for item in self.validation if item["metadata"]["row_id"] == row_id)
        payload = json.loads(row["messages"][1]["content"])
        candidates = payload["current_harness_state"]["state_action_affordances"][
            "repair_restoration_candidates"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["claim_id"], "claim_surgery_155_1")
        self.assertEqual(candidates[0]["tool_name"], "compare_global_motion_symmetric")
        self.assertEqual(
            candidates[0]["evidence_gap_fields"],
            [
                "preferred_video",
                "video_a_interval_s",
                "video_b_interval_s",
                "measurement_method",
            ],
        )

    def test_conflict_repair_compiles_one_maximal_compatible_binding_call(self) -> None:
        row_id = (
            "VidDiffBench:viddiff-surgery_70:LLM-TRJ-VIDDIFF-SURGERY_70-"
            "REPAIR-001:repair:4"
        )
        row = next(item for item in self.validation if item["metadata"]["row_id"] == row_id)
        payload = json.loads(row["messages"][1]["content"])
        actions = payload["current_harness_state"]["state_action_affordances"][
            "executable_binding_actions"
        ]
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action["tool_name"], "bind_evidence")
        self.assertEqual(action["coverage"]["field_count"], 4)
        self.assertEqual(action["coverage"]["distinct_evidence_count"], 1)
        for item in action["arguments"]["binding_items"]:
            self.assertEqual(
                item["support_ids"],
                ["EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-002"],
            )

    def test_binding_call_is_not_advertised_before_bind_is_eligible(self) -> None:
        row_id = (
            "VidDiffBench:viddiff-surgery_70:LLM-TRJ-VIDDIFF-SURGERY_70-"
            "REPAIR-001:repair:3"
        )
        row = next(item for item in self.validation if item["metadata"]["row_id"] == row_id)
        payload = json.loads(row["messages"][1]["content"])
        state = payload["current_harness_state"]
        self.assertNotIn("bind_evidence", state["eligible_tools"])
        self.assertEqual(
            state["state_action_affordances"]["binding_candidates"],
            {},
        )
        self.assertEqual(
            state["state_action_affordances"]["executable_binding_actions"],
            [],
        )

    def test_every_teacher_state_action_aligns_with_visible_affordances(self) -> None:
        manifest = json.loads((ARM / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["rows"], 600)
        self.assertEqual(manifest["target_affordance_alignment_failures"], 0)
        self.assertFalse(manifest["oracle_inputs_used"])

    def test_v6_compacts_redundant_candidates_and_constrains_native_schemas(self) -> None:
        manifest = json.loads((EXECUTABLE_ARM / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["rows_with_raw_binding_candidates_compacted"], 103)
        self.assertEqual(manifest["rows_with_dynamic_native_argument_constraints"], 265)
        rows = [
            json.loads(line)
            for line in (EXECUTABLE_ARM / "validation" / "sft_all.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        row = next(
            item
            for item in rows
            if item["metadata"]["row_id"].endswith(
                "VIDDIFF-SURGERY_70-REPAIR-001:repair:4"
            )
        )
        payload = json.loads(row["messages"][1]["content"])
        affordances = payload["current_harness_state"]["state_action_affordances"]
        self.assertEqual(affordances["binding_candidates"], {})
        bind_tool = next(
            tool for tool in row["tools"] if tool["function"]["name"] == "bind_evidence"
        )
        parameters = bind_tool["function"]["parameters"]
        self.assertEqual(
            parameters["properties"]["binding_items"]["const"],
            affordances["executable_binding_actions"][0]["arguments"]["binding_items"],
        )


if __name__ == "__main__":
    unittest.main()
