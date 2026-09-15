from __future__ import annotations

import json
import copy
from pathlib import Path

from audit_recovery_reasoning_quality import audit_rows
from build_portable_recovery_sft import (
    GUID,
    HTTP_URL,
    VCSL_CASE,
    VIDDIFF_CASE,
    VIDDIFF_URI,
    VIDEOSHAM_CASE,
    WINDOWS_PATH,
    WORKSPACE_PATH,
    build_portable_rows,
)
from shadow_execute_generated_actions import _jsonl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/student_external_contract_v1"
SOURCE = BASE / "recovery_sft_training_mix_v1/recovery_sft_weighted.jsonl"


def test_portable_serialization_removes_identity_without_semantic_loss() -> None:
    source = _jsonl(SOURCE)
    portable, report = build_portable_rows(source)
    assert len(portable) == len(source) == 319
    assert report["all_semantic_equivalence_checks_passed"] is True
    assert report["all_forbidden_identifier_hits_zero"] is True
    assert all(value == 0 for value in report["forbidden_identifier_hits"].values())
    assert all(item["passed"] for item in report["rows_detail"])
    serialized = "\n".join(json.dumps(item, ensure_ascii=False) for item in portable)
    for pattern in (
        WINDOWS_PATH,
        WORKSPACE_PATH,
        HTTP_URL,
        VIDDIFF_URI,
        VIDEOSHAM_CASE,
        VIDDIFF_CASE,
        VCSL_CASE,
        GUID,
    ):
        assert pattern.search(serialized) is None


def test_portable_reasoning_still_passes_unchanged_quality_thresholds() -> None:
    rows = _jsonl(
        BASE / "recovery_sft_portable_v3/recovery_sft_portable_weighted.jsonl"
    )
    report = audit_rows(rows)
    assert report["passed"] is True
    assert report["metrics"]["row_passed"] == 319
    assert report["metrics"]["exact_duplicate_fraction"] <= 0.05
    assert report["metrics"]["normalized_template_duplicate_fraction"] <= 0.35


def test_portable_case_ids_preserve_cross_variant_equivalence_without_raw_id() -> None:
    source = _jsonl(SOURCE)[:1]
    variant = copy.deepcopy(source[0])
    variant["metadata"]["episode_id"] = "different-episode"
    variant["metadata"]["row_id"] = "different-episode:01"
    portable, report = build_portable_rows([source[0], variant])
    assert portable[0]["metadata"]["case_id"] == portable[1]["metadata"]["case_id"]
    assert portable[0]["metadata"]["case_id"].startswith("portable://")
    assert source[0]["metadata"]["case_id"] not in portable[0]["metadata"]["case_id"]
    assert report["case_equivalence_classes_preserved"] is True
