from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from package_verified_student_sft import package


def test_strict_two_case_package_is_valid_and_leak_free() -> None:
    inputs = [
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_surgery117_clean_full_v1/sft/sft_all.jsonl",
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_surgery166_clean_full_v1/sft/sft_all.jsonl",
    ]
    rows, report = package(inputs)

    assert len(rows) == 26
    assert report["ready_for_training"] is True
    assert report["privacy_leaks"] == []
    assert report["validation"]["exact_reasoning_duplicates"] == 0


def test_semantic_audit_must_cover_every_packaged_identity(tmp_path: Path) -> None:
    inputs = [
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_surgery117_clean_full_v1/sft/sft_all.jsonl"
    ]
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "all_passed": True,
                "episode_reports": [
                    {
                        "dataset": "VidDiffBench",
                        "case_id": "different-case",
                        "mode": "clean",
                        "semantic_passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows, report = package(inputs, semantic_audit_path=audit)
    assert rows == []
    assert report["ready_for_training"] is False
    assert report["semantic_audit"]["coverage_missing"]
