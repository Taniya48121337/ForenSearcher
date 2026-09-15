from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_FIELDS = {
    "claim_family",
    "manipulation_type",
    "direction",
    "omitted_source_intervals_s",
    "estimated_removed_time_s",
}
INVARIANT_FIELDS = {"source_id", "comparison_scope_s", "retained_source_interval_s"}


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line]


class InteractiveHarnessOutputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean = load_json("outputs/interactive_clean_episode.json")
        cls.repair = load_json("outputs/interactive_repair_episode.json")
        cls.summary = load_json("outputs/interactive_execution_summary.json")
        cls.report = load_json("outputs/interactive_validation_report.json")
        cls.clean_datums = load_jsonl("outputs/interactive_sft_clean.jsonl")
        cls.repair_datums = load_jsonl("outputs/interactive_sft_repair.jsonl")

    def test_real_external_alignment_runs_clean_and_repair(self) -> None:
        self.assertEqual(self.summary["external_tool_call_counts"]["align_videos_dense"], 2)
        alignment_ids = []
        for episode in (self.clean, self.repair):
            event = next(
                item
                for item in episode["events"]
                if item["action"] == "align_videos_dense"
                and item["action_type"] == "external_tool_execution"
            )
            alignment_ids.append(event["result"]["observation"]["execution_id"])
        self.assertNotEqual(
            self.summary.get("clean_alignment_execution_id", alignment_ids[0]),
            self.summary.get("repair_alignment_execution_id", alignment_ids[1]),
        )
        repair_alignment = next(
            event
            for event in self.repair["events"]
            if event["action"] == "align_videos_dense"
            and event["action_type"] == "external_tool_execution"
        )
        self.assertEqual(repair_alignment["actor"], "environment")
        self.assertEqual(repair_alignment["result"]["status"], "executed")

    def test_sft_uses_standard_tool_call_rounds(self) -> None:
        messages = self.clean["messages"]
        assistant_calls = [
            message for message in messages if message["role"] == "assistant" and message.get("tool_calls")
        ]
        tool_results = [message for message in messages if message["role"] == "tool"]
        self.assertEqual(len(assistant_calls), 6)
        self.assertEqual(len(tool_results), 6)
        for call, result in zip(assistant_calls, tool_results):
            self.assertEqual(call["tool_calls"][0]["id"], result["tool_call_id"])

    def test_every_action_has_grounded_natural_language_reasoning(self) -> None:
        calls = [
            message
            for episode in (self.clean, self.repair)
            for message in episode["messages"]
            if message["role"] == "assistant" and message.get("tool_calls")
        ]
        self.assertEqual(len(calls), 11)
        for message in calls:
            self.assertIsInstance(message["content"], str)
            self.assertGreaterEqual(len(message["content"].split()), 18)
            self.assertFalse(message["content"].lstrip().startswith("{"))

    def test_harness1_style_one_datum_per_turn(self) -> None:
        self.assertEqual(len(self.clean_datums), 6)
        self.assertEqual(len(self.repair_datums), 5)
        for datum in self.clean_datums + self.repair_datums:
            target = datum["messages"][-1]
            self.assertEqual(target["role"], "assistant")
            self.assertTrue(target["tool_calls"])
            self.assertFalse(datum["metadata"]["target_includes_observation"])
            self.assertEqual(datum["metadata"]["loss_scope"], "last_assistant_action_only")

    def test_policy_and_environment_authorities_are_separate(self) -> None:
        proposals = [event for event in self.clean["events"] if event["actor"] == "policy"]
        external = [
            event for event in self.clean["events"] if event["action_type"] == "external_tool_execution"
        ]
        state_edits = [
            event
            for event in self.clean["events"]
            if event["action_type"] in {"policy_state_edit_validation", "binding_validation"}
        ]
        self.assertEqual(len(proposals), 6)
        self.assertTrue(external)
        self.assertTrue(state_edits)
        self.assertTrue(all(event["result"]["state_changed"] is False for event in proposals))
        self.assertTrue(all(event["actor"] == "environment" for event in external))
        self.assertTrue(all(event["actor"] == "harness" for event in state_edits))
        created_ids = {
            evidence_id
            for event in external
            for evidence_id in event["result"]["created_evidence_ids"]
        }
        self.assertIn(self.summary["dropped_evidence_id"], created_ids)

    def test_dropout_and_policy_reopen_only_target_fields(self) -> None:
        clean_fields = self.clean["final_state"]["candidate_claims"][0]["fields"]
        repair_user = json.loads(self.repair["messages"][1]["content"])
        mutated_state = self.repair.get("mutated_state", repair_user["harness_state"])
        mutated_fields = mutated_state["candidate_claims"][0]["fields"]
        constrained = mutated_state["action_constraints"]["must_reopen_before_rebinding"]
        self.assertEqual(len(constrained), 1)
        self.assertEqual(set(constrained[0]["fields"]), TARGET_FIELDS)
        post_reopen_state = (
            self.repair["post_reopen_state"]
            if "post_reopen_state" in self.repair
            else self.repair["turn_history"][0]["observation"]["harness_state"]
        )
        reopened_fields = post_reopen_state["candidate_claims"][0]["fields"]
        self.assertEqual(
            post_reopen_state["action_constraints"]["must_reopen_before_rebinding"], []
        )
        for name in TARGET_FIELDS:
            self.assertEqual(mutated_fields[name]["evidence_status"], "unsupported")
            self.assertEqual(reopened_fields[name]["decision_status"], "open")
        for name in INVARIANT_FIELDS:
            self.assertEqual(mutated_fields[name], clean_fields[name])
            self.assertEqual(reopened_fields[name], clean_fields[name])

        self.assertNotIn("affected_fields", repair_user["mutation"])
        self.assertNotIn("target_fields", repair_user["mutation"])
        self.assertNotIn("invariant_fields", repair_user["mutation"])

    def test_local_repair_preserves_invariants(self) -> None:
        final_fields = self.repair["final_state"]["candidate_claims"][0]["fields"]
        repair_evidence_id = (
            self.summary["repair_evidence_id"]
            if "repair_evidence_id" in self.summary
            else self.summary["repair_active_gap_ids"][0]
        )
        for name in TARGET_FIELDS:
            self.assertEqual(final_fields[name]["evidence_status"], "supported")
            self.assertEqual(final_fields[name]["decision_status"], "committed")
            self.assertIn(repair_evidence_id, final_fields[name]["support_ids"])
        self.assertEqual(
            self.repair["invariant_fields_before"],
            self.repair["invariant_fields_after"],
        )
        self.assertNotIn(
            self.summary["dropped_evidence_id"],
            self.repair["final_state"]["available_evidence_ids"],
        )

    def test_terminal_and_no_leak_validation_passes(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(all(self.report["checks"].values()))
        self.assertTrue(self.clean["final_state"]["stop_decision"]["allowed"])
        self.assertTrue(self.repair["final_state"]["stop_decision"]["allowed"])


if __name__ == "__main__":
    unittest.main()
