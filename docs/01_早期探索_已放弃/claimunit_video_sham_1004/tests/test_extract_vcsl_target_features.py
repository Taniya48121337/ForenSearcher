import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_vcsl_target_features import extract  # noqa: E402


class ExtractVCSLTargetFeaturesTest(unittest.TestCase):
    def test_extracts_only_requested_valid_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "features.tar.gz"
            payload = io.BytesIO()
            np.save(payload, np.ones((3, 256), dtype=np.float32), allow_pickle=False)
            data = payload.getvalue()
            with tarfile.open(archive, "w:gz") as handle:
                member = tarfile.TarInfo("./eff256d/target.npy")
                member.size = len(data)
                handle.addfile(member, io.BytesIO(data))
                extra = tarfile.TarInfo("./eff256d/extra.npy")
                extra.size = len(data)
                handle.addfile(extra, io.BytesIO(data))

            output = root / "out"
            report = extract(archive, output, {"target"})
            self.assertTrue(report["all_target_features_valid"])
            self.assertTrue((output / "target.npy").exists())
            self.assertFalse((output / "extra.npy").exists())


if __name__ == "__main__":
    unittest.main()
