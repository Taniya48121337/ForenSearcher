import unittest

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_batch import VidDiffAdaptiveMeasurementEnvironment
from viddiff_symmetry_abstention import derive_candidate, predicate_family


def _item(statement: str, global_direction: str = "c", action_direction: str = "c") -> dict:
    tools = []
    for name, direction in (
        ("compare_global_motion_symmetric", global_direction),
        ("compare_action_refined_symmetric", action_direction),
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
                    "video_a_interval_s": [1.0, 2.0],
                    "video_b_interval_s": [2.0, 3.0],
                    "confidence": 0.6,
                    "order_consistent": False,
                    "measurement_id": f"measure-{name}",
                },
                "provenance": {
                    "candidate_run_id": f"run-{name}",
                    "label_visible_to_tool": False,
                },
            }
        )
    return {
        "sample_key": "surgery_fixture",
        "task": {"comparison_statement": statement},
        "candidate_tools": tools,
    }


class VidDiffSymmetryAbstentionTest(unittest.TestCase):
    def test_eligible_families_are_visible_statement_only(self) -> None:
        self.assertEqual(predicate_family("The needle is more perpendicular."), "geometry")
        self.assertEqual(predicate_family("The number of movements is lower."), "count_frequency")
        self.assertEqual(predicate_family("The tube moves more."), "object_motion")
        self.assertIsNone(predicate_family("The force on the target is lower."))

    def test_derives_role_invariant_abstention_from_two_unstable_routes(self) -> None:
        candidate = derive_candidate(_item("The needle is more perpendicular."))
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["name"], "compare_symmetry_residual_abstention")
        self.assertEqual(candidate["tool_observation"]["preferred_video"], "c")
        self.assertTrue(candidate["tool_observation"]["order_consistent"])
        self.assertFalse(candidate["provenance"]["label_visible_to_tool"])

    def test_rejects_force_predicate_or_stable_directional_source(self) -> None:
        self.assertIsNone(derive_candidate(_item("The force on the target is lower.")))
        self.assertIsNone(
            derive_candidate(
                _item("The needle is more perpendicular.", global_direction="a")
            )
        )

    def test_meta_tool_is_eligible_only_after_both_source_routes(self) -> None:
        item = _item("The needle is more perpendicular.")
        item["candidate_tools"].insert(
            0,
            {
                "name": "compare_uniform_timeline",
                "arguments": {
                    "video_a_ref": "video_a",
                    "video_b_ref": "video_b",
                    "comparison_statement": "The needle is more perpendicular.",
                },
                "tool_observation": {
                    "preferred_video": "a",
                    "video_a_interval_s": [0.0, 1.0],
                    "video_b_interval_s": [0.0, 1.0],
                    "confidence": 0.5,
                },
                "provenance": {"candidate_run_id": "uniform", "label_visible_to_tool": False},
            },
        )
        item["candidate_tools"].append(derive_candidate(item))
        environment = VidDiffAdaptiveMeasurementEnvironment(item)
        environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        self.assertNotIn(
            "compare_symmetry_residual_abstention", environment.eligible_tool_names()
        )
        by_name = {entry["name"]: entry for entry in item["candidate_tools"]}
        environment.execute(
            "compare_global_motion_symmetric",
            by_name["compare_global_motion_symmetric"]["arguments"],
        )
        self.assertNotIn(
            "compare_symmetry_residual_abstention", environment.eligible_tool_names()
        )
        environment.execute(
            "compare_action_refined_symmetric",
            by_name["compare_action_refined_symmetric"]["arguments"],
        )
        self.assertIn(
            "compare_symmetry_residual_abstention", environment.eligible_tool_names()
        )


if __name__ == "__main__":
    unittest.main()
