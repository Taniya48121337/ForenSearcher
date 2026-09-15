from __future__ import annotations

import pytest

from run_student_closed_loop_canary import (
    VIDDIFF_DEVELOPMENT_CANDIDATE_PATHS,
    _viddiff_item,
)
from build_expanded_recovery_manifest import build_expanded_manifest
from compile_recovery_batch import compile_batch
from unannounced_repair_interventions import INTERVENTIONS


def test_explicit_development_catalog_covers_all_thirty_seed_20260818_cases() -> None:
    sample_keys: set[str] = set()
    for path in VIDDIFF_DEVELOPMENT_CANDIDATE_PATHS:
        assert path.is_file()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                import json

                sample_keys.add(str(json.loads(line)["sample_key"]))
    assert len(sample_keys) == 30
    assert _viddiff_item("surgery_128")["sample_key"] == "surgery_128"
    assert _viddiff_item("surgery_121")["sample_key"] == "surgery_121"


def test_development_catalog_does_not_fall_through_to_external_holdout() -> None:
    with pytest.raises(ValueError, match="measurements unavailable"):
        _viddiff_item("surgery_20")


def test_method_identity_misbinding_gives_low_consensus_case_a_local_repair() -> None:
    manifest = build_expanded_manifest()
    job = next(
        item for item in manifest["jobs"]
        if item["case_id"] == "viddiff-surgery_61"
    )
    job["intervention"] = next(
        item for item in INTERVENTIONS
        if item["id"] == "U11_viddiff_method_identity_misbinding"
    )
    rows, episodes, _, report = compile_batch(
        {**manifest, "jobs": [job]}, intervention_portfolio=False
    )
    assert report["compiled_episodes"] == 1
    assert report["terminal_success"] == 1
    assert report["fresh_evidence_consumed"] == 1
    assert report["independent_fields_preserved"] == 1
    assert report["counterfactuals_passed"] == 1
    assert report["reasoning_quality_audit"]["passed"] is True
    assert rows and episodes
