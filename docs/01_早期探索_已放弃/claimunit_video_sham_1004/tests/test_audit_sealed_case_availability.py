from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_sealed_case_availability import (  # noqa: E402
    _is_exposed,
    _references,
    _references_in_files,
)


def test_references_are_case_insensitive() -> None:
    records = [("a.json", "prefix VIDSHAм-0000"), ("b.json", "VIDEOSHAM-0064")]
    assert _references(records, ["videosham-0064"])["videosham-0064"] == ["b.json"]


def test_vcsl_reverse_pair_counts_as_exposed() -> None:
    candidate = {
        "case_id": "vcsl-query-reference",
        "query_id": "query",
        "reference_id": "reference",
    }
    references = {
        "vcsl-query-reference": [],
        "vcsl-reference-query": ["prior.json"],
    }
    exposed, paths = _is_exposed("VCSL", candidate, references)
    assert exposed is True
    assert paths == ["prior.json"]


def test_non_vcsl_has_no_reverse_alias() -> None:
    candidate = {"case_id": "viddiff-surgery_999"}
    exposed, paths = _is_exposed("VidDiffBench", candidate, {})
    assert exposed is False
    assert paths == []


def test_streaming_reference_normalizes_bare_viddiff_and_vcsl(tmp_path: Path) -> None:
    history = tmp_path / "history.json"
    history.write_text(
        '"sample_key":"surgery_42" '
        '"pair_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"',
        encoding="utf-8",
    )
    tokens = [
        "viddiff-surgery_42",
        "vcsl-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    ]
    result = _references_in_files([history], tokens)
    assert result[tokens[0]] == [history.as_posix()]
    assert result[tokens[1]] == [history.as_posix()]
