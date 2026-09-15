import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fetch_videosham_subset import (
    TARGET_FINAL_COUNTS,
    _member_case_number,
    select_balanced_missing_rows,
    validate_video_file,
)
from videosham_batch import EDITED_ROOT, SOURCE_ROOT, _case_number, _rows, capability_family


class FetchVideoShamSubsetTest(unittest.TestCase):
    def test_balanced_selection_fills_only_the_remaining_family_deficits(self):
        selected = select_balanced_missing_rows(20260820)
        selected_numbers = {_case_number(row) for row in selected}
        self.assertEqual(len(selected_numbers), len(selected))
        self.assertTrue(all(number is not None for number in selected_numbers))

        existing_counts = Counter(
            capability_family(row)
            for row in _rows()
            if (number := _case_number(row))
            and (SOURCE_ROOT / f"{number}_unedited.mp4").exists()
            and (EDITED_ROOT / f"{number}_processed.mp4").exists()
        )
        expected_missing = sum(
            max(0, target - existing_counts.get(family, 0))
            for family, target in TARGET_FINAL_COUNTS.items()
        )
        self.assertEqual(len(selected), expected_missing)
        final_counts = existing_counts + Counter(capability_family(row) for row in selected)
        self.assertEqual({family: final_counts[family] for family in TARGET_FINAL_COUNTS}, TARGET_FINAL_COUNTS)

    def test_release_member_parser_accepts_only_expected_video_names(self):
        self.assertEqual(_member_case_number("Unedited/0061_unedited.mp4"), "0061")
        self.assertEqual(_member_case_number("Processed/4189_processed.mp4"), "4189")
        self.assertIsNone(_member_case_number("README.txt"))
        self.assertIsNone(_member_case_number("0061.mp4"))

    def test_existing_real_video_passes_container_validation(self):
        report = validate_video_file(SOURCE_ROOT / "1007_unedited.mp4")
        self.assertTrue(report["valid"])
        self.assertTrue(report["first_frame_decoded"])
        self.assertGreater(report["duration_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
