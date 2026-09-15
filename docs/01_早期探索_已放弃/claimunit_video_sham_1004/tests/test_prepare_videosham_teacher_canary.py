import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_videosham_teacher_canary import (  # noqa: E402
    DEFAULT_ACQUISITION,
    DEFAULT_MEASUREMENTS,
    DEFAULT_VERIFIER,
    prepare,
)
from llm_teacher_rollout import _contract_and_environment  # noqa: E402


class PrepareVideoShamTeacherCanaryTest(unittest.TestCase):
    def test_teacher_contract_uses_logical_media_refs_not_local_paths(self):
        manifest = {
            "case_id": "videosham-test",
            "task": "compare",
            "source_media": {"ref": "source"},
            "edited_media": {"ref": "suspect"},
            "comparison_scope_s": [0.0, 1.0],
        }
        environment = object()
        contract, returned_environment = _contract_and_environment(
            manifest=manifest,
            source_frames=object(),
            suspect_frames=object(),
            external_environment=environment,
        )
        self.assertEqual(contract.visible_assets["source"], "source")
        self.assertEqual(contract.visible_assets["suspect"], "suspect")
        self.assertIs(returned_environment, environment)

    def test_canary_is_stratified_and_student_payload_is_privacy_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare(DEFAULT_MEASUREMENTS, DEFAULT_VERIFIER, DEFAULT_ACQUISITION, output)

            self.assertEqual(report["selected_cases"], 7)
            self.assertEqual(report["families_covered"], 7)
            self.assertEqual(sorted(report["repair_distribution"].values()), [2, 2, 3])
            self.assertTrue(report["privacy_scan_passed"])
            self.assertFalse(any(report["forbidden_pattern_hits"].values()))
            self.assertEqual(report["teacher_calls_made"], 0)

            previews = [
                json.loads(line)
                for line in (output / "teacher_payload_preview.student.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            audits = [
                json.loads(line)
                for line in (output / "selection_audit.local_verifier.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(previews), 7)
            self.assertEqual(len({row["dataset_family"] for row in audits}), 7)
            self.assertEqual({row["case_id"] for row in previews}, {row["case_id"] for row in audits})
            self.assertTrue(all(row["teacher_transport"] == "text_json_only" for row in previews))
            self.assertTrue(all(not row["media_bytes_sent"] for row in previews))

            outbound = "\n".join(json.dumps(row, ensure_ascii=False) for row in previews).lower()
            for forbidden in ("dataset_family", "annotation_text", "oracle", "verifier", ".mp4", ".npy", "http://", "https://"):
                self.assertNotIn(forbidden, outbound)


if __name__ == "__main__":
    unittest.main()
