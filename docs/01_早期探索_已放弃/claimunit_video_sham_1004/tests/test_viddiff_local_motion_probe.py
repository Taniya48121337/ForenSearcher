import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_local_motion_probe import green_thread_mask, motion_measurement  # noqa: E402


class VidDiffLocalMotionProbeTest(unittest.TestCase):
    def test_green_thread_mask_and_motion_respond_to_translation(self) -> None:
        frames = []
        for offset in (2, 5, 8, 11):
            image = np.zeros((20, 30, 3), dtype=np.uint8) + 25
            image[8:10, offset : offset + 8] = [45, 90, 40]
            frames.append(Image.fromarray(image))
        static = [frames[0]] * 4
        self.assertGreater(green_thread_mask(frames[0]).sum(), 0)
        self.assertGreater(
            motion_measurement(frames)["motion_score"],
            motion_measurement(static)["motion_score"],
        )


if __name__ == "__main__":
    unittest.main()
