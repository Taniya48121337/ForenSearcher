from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_student_context_evaluations import BOOLEAN_METRICS, compare  # noqa: E402


def _row(row_id: str, dataset: str, passed: bool, *, limit: bool = False) -> dict:
    row = {
        "row_id": row_id,
        "dataset": dataset,
        "generated": True,
        "reference_tool": "inspect",
        "generated_tool": "inspect" if passed else None,
        "generation_hit_token_limit": limit,
    }
    row.update({metric: passed for metric in BOOLEAN_METRICS})
    return row


def _report(rows: list[dict], nll: float) -> dict:
    return {
        "summary": {
            "target_nll": nll,
            "target_perplexity": 1.5,
            "generation_token_limit_hit_rate": 0.0,
        },
        "rows": rows,
    }


def test_compare_counts_repairs_and_passes_strict_pilot() -> None:
    baseline = _report([_row("a", "D1", False, limit=True), _row("b", "D2", True)], 0.5)
    candidate = _report([_row("a", "D1", True), _row("b", "D2", True)], 0.4)
    result = compare(baseline, candidate)
    assert result["metrics"]["tool_name_exact"]["fixed"] == 1
    assert result["metrics"]["tool_name_exact"]["regressed"] == 0
    assert result["token_limit"]["fixed"] == 1
    assert result["pilot_verdict"]["passed"] is True


def test_compare_rejects_offsetting_regression() -> None:
    baseline = _report([_row("a", "D", False), _row("b", "D", True)], 0.5)
    candidate = _report([_row("a", "D", True), _row("b", "D", False)], 0.4)
    result = compare(baseline, candidate)
    assert result["metrics"]["tool_call_parseable"]["fixed"] == 1
    assert result["metrics"]["tool_call_parseable"]["regressed"] == 1
    assert result["pilot_verdict"]["passed"] is False


def test_compare_requires_identical_generation_rows() -> None:
    with pytest.raises(ValueError, match="generation row mismatch"):
        compare(_report([_row("a", "D", True)], 0.5), _report([_row("b", "D", True)], 0.4))
