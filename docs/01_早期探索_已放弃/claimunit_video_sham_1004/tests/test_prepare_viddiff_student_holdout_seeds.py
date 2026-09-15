import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_viddiff_student_holdout_seeds import DEFAULT_V6, build
from run_student_closed_loop_batch import _preflight


def _candidate() -> dict:
    statement = "The right tool moves more radially."
    tools = []
    for name, direction in (
        ("compare_uniform_timeline", "a"),
        ("compare_global_motion_symmetric", "a"),
        ("compare_action_refined_symmetric", "b"),
    ):
        tools.append(
            {
                "name": name,
                "arguments": {
                    "video_a_ref": "video_a",
                    "video_b_ref": "video_b",
                    "comparison_statement": statement,
                },
                "tool_observation": {
                    "preferred_video": direction,
                    "video_a_interval_s": [0.0, 1.0],
                    "video_b_interval_s": [0.0, 1.0],
                    "confidence": 0.8,
                    "order_consistent": True,
                    "visual_rationale": "Adjacent visible frames provide a concrete comparative motion pattern.",
                },
                "provenance": {
                    "candidate_run_id": f"fixture-{name}",
                    "label_visible_to_tool": False,
                },
            }
        )
    return {
        "case_id": "viddiff-surgery_external_fixture",
        "sample_key": "surgery_external_fixture",
        "task": {
            "action_name": "Suturing",
            "action_description": "Perform the shared suturing action.",
            "comparison_statement": statement,
            "visible_assets": [
                {"role": "A", "local_duration_s": 2.0},
                {"role": "B", "local_duration_s": 3.0},
            ],
        },
        "candidate_tools": tools,
        "routing_contract": {
            "policy": "visible_observation_adaptive_branching_v2",
            "hidden_label_may_select_next_action": False,
        },
    }


class PrepareVidDiffStudentHoldoutSeedsTest(unittest.TestCase):
    def _write_inputs(self, root: Path, candidate: dict) -> tuple[Path, Path]:
        candidates = root / "candidate_measurements.student.jsonl"
        candidates.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        audit = root / "selection_audit.json"
        audit.write_text(
            json.dumps(
                {
                    "cases": [
                        {"sample_key": candidate["sample_key"], "accepted": True}
                    ]
                }
            ),
            encoding="utf-8",
        )
        return candidates, audit

    def test_builds_neutral_executable_seed_without_teacher_target(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = _candidate()
            candidates, audit = self._write_inputs(Path(directory), candidate)
            rows, manifest = build(DEFAULT_V6, candidates, audit)
        self.assertEqual(manifest["cases"], 1)
        self.assertFalse(manifest["teacher_actions_used"])
        self.assertFalse(manifest["oracle_or_verifier_in_student_seed"])
        row = rows[0]
        self.assertEqual(len(row["messages"]), 2)
        self.assertEqual([message["role"] for message in row["messages"]], ["system", "user"])
        payload = json.loads(row["messages"][1]["content"])
        self.assertEqual(
            payload["current_harness_state"]["eligible_tools"],
            ["inspect_comparison_task"],
        )
        self.assertEqual(
            payload["task"]["task_contract"]["visible_assets"]["comparison_scope_s"],
            [0.0, 3.0],
        )
        serialized = json.dumps(row)
        for token in ("oracle_direction", "video_path", "candidate_report"):
            self.assertNotIn(token, serialized)
        preflight = _preflight(rows, {candidate["sample_key"]: candidate})
        self.assertTrue(preflight["all_passed"])

    def test_rejects_private_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = _candidate()
            private = copy.deepcopy(candidate)
            private["task"]["visible_assets"][0]["video_path"] = r"C:\\private\\a.mp4"
            candidates, audit = self._write_inputs(Path(directory), private)
            with self.assertRaisesRegex(ValueError, "forbidden fields"):
                build(DEFAULT_V6, candidates, audit)


if __name__ == "__main__":
    unittest.main()
