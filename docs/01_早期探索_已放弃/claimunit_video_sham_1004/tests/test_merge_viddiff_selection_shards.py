import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from merge_viddiff_selection_shards import merge


def _write_shard(root: Path, key: str, *, accepted: bool = True, private: bool = False) -> Path:
    root.mkdir(parents=True)
    candidate = {
        "case_id": f"viddiff-{key}",
        "sample_key": key,
        "task": {"comparison_statement": "A neutral comparison."},
        "candidate_tools": [],
        "routing_contract": {},
    }
    if private:
        candidate["video_path"] = "C:/private/video.mp4"
    (root / "candidate_measurements.student.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    (root / "selection_audit.json").write_text(
        json.dumps(
            {
                "dataset": "VidDiffBench",
                "seed": 20260821,
                "cases": [
                    {
                        "sample_key": key,
                        "accepted": accepted,
                        "selected_tool": "compare_uniform_timeline" if accepted else None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class MergeVidDiffSelectionShardsTest(unittest.TestCase):
    def test_merges_disjoint_safe_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = _write_shard(root / "first", "surgery_1")
            second = _write_shard(root / "second", "surgery_2", accepted=False)
            candidates, audit = merge([first, second], expected_cases=2)
            self.assertEqual([item["sample_key"] for item in candidates], ["surgery_1", "surgery_2"])
            self.assertEqual(audit["summary"]["accepted"], 1)
            self.assertEqual(audit["summary"]["rejected"], 1)

    def test_rejects_duplicate_case(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = _write_shard(root / "first", "surgery_1")
            second = _write_shard(root / "second", "surgery_1")
            with self.assertRaisesRegex(ValueError, "duplicate VidDiff sample_key"):
                merge([first, second])

    def test_rejects_private_student_field(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            shard = _write_shard(Path(raw) / "private", "surgery_1", private=True)
            with self.assertRaisesRegex(ValueError, "private fields"):
                merge([shard])


if __name__ == "__main__":
    unittest.main()
