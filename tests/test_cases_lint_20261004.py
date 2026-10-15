"""案卷结构检查：must_cite 要求必须有证据项支撑，否则所有 run 的答案都会被判 premature_finish。"""

from __future__ import annotations

from pathlib import Path

import pytest

from foren_searcher.datagen.cases import Case

CASES = Path(__file__).resolve().parents[1] / "experiments" / "sft_data" / "cases"


def test_orphan_requirement_detected() -> None:
    case = Case.from_dict({
        "case_id": "true:1", "dataset": "true", "source_id": "1", "split": "dev_train", "task_kind": "claim_check",
        "task_text": "t", "family_key": "true:1",
        "requirements": [{"id": "REQ1", "text": "a", "must_cite": True}, {"id": "REQ2", "text": "b", "must_cite": True},
                         {"id": "REQ3", "text": "c", "must_cite": False}],
        "required_evidence": [{"key": "src", "description": "d", "url_pattern": "example.org/*", "supports": ["REQ1"]}],
    })
    assert case.orphan_requirements() == ("REQ2",)


@pytest.mark.skipif(not CASES.is_dir(), reason="没有 experiments/sft_data/cases")
def test_repo_cases_have_no_orphan_requirements() -> None:
    bad = {p.name: Case.load(p).orphan_requirements() for p in sorted(CASES.glob("*.json"))}
    assert not {k: v for k, v in bad.items() if v}


def test_group_missing_evidence_counts_once_0_3_5() -> None:
    """any_one_of 组没取到：第一层记一条（extra.keys 整组），第三层对组内其余 key 的逐项缺项不再重复计 major。"""

    from foren_searcher.datagen.errors import ErrorType, Severity, error
    from foren_searcher.datagen.verify.verdict import decide, merge_sufficiency_errors

    group = error(ErrorType.OMITTED_REQUIRED_FACT, "必需证据 a / b / c 未被任何 Finding 引用", method="exec",
                  extra={"check": "required_evidence_cited", "key": "a", "keys": ["a", "b", "c"], "mode": "any_one_of"})
    per_key = [error(ErrorType.OMITTED_REQUIRED_FACT, f"任务要求 {k!r} 未被回答", method="model", extra={"requirement": k, "from_case": True})
               for k in ("b", "c")]
    other = error(ErrorType.OMITTED_REQUIRED_FACT, "任务要求 'd' 未被回答", method="model", extra={"requirement": "d", "from_case": True})
    merged = merge_sufficiency_errors([group], [*per_key, other])
    assert [e.extra.get("key") or e.extra.get("requirement") for e in merged] == ["a", "d"]
    assert decide(merge_sufficiency_errors([group], per_key), []) == "positive"
    assert decide(merged, []) == "negative"
