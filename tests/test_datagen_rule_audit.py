"""校准规则 R1/R2/R4 反例注入（calibrate_rule_audit）：verdict 模式在真实报告上算假接受 / 真接受；report 模式用假核验模型跑通流程。

断言里的比率是**现行规则的逻辑后果**（模拟核验器如实标出了错误）：反例全被豁免 = 规则可钻；正例里门槛 / 长度 / 子串
匹配造成的假拒绝。将来改规则时这些断言会失败，提醒同步更新 docs/核验器量化工具.md。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.errors import ErrorType, error
from foren_searcher.datagen.verify.anchored import COVERAGE_SYSTEM_PROMPTS, PAIR_SYSTEM_PROMPTS
from foren_searcher.datagen.verify.verdict import Verdict

from test_datagen_judged_units import RUN, VERDICTS, FakeJudge, needs_run


def _base() -> list[Verdict]:
    return [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]


@needs_run
def test_rule_audit_verdict_mode_on_real_reports():
    from foren_searcher.datagen.verify.calibrate_rule_audit import BUILDERS, audit

    base = _base()
    snapshot = [v.to_dict() for v in base]
    summary = asyncio.run(audit([RUN], base, kinds=list(BUILDERS), per_kind=1, seed=3, real_verdicts=base, say=lambda _: None))
    assert [v.to_dict() for v in base] == snapshot                       # 模拟判定是副本，不动真实判定
    by_kind = summary["by_kind"]
    for kind in BUILDERS:
        assert by_kind[kind]["n"] >= 1 and by_kind[kind]["not_flagged"] == 0, kind
    # 反例：全部被豁免（规则可钻）；R2 / R4 的豁免能把报告处置翻成 positive，R1 豁免的是 minor 不翻
    for rule in ("R1", "R2", "R4"):
        assert summary["by_rule"][rule]["false_accept"]["rate"] == 1.0, rule
    assert summary["by_rule"]["R1"]["false_accept"]["disposition_flipped"] == 0
    assert summary["by_rule"]["R2"]["false_accept"]["disposition_flipped"] >= 1
    assert summary["by_rule"]["R4"]["false_accept"]["disposition_flipped"] >= 1
    # 正例：规则照顾到的情况全豁免；门槛（2 条 E）、长度（>20 字）、转述（子串对不上）全假拒绝
    for kind, rate in (("r1_joint_support", 1.0), ("r2_one_sided_relation", 1.0), ("r4_earlier_state", 1.0),
                       ("r1_joint_support_two", 0.0), ("r2_one_sided_relation_long", 0.0), ("r4_paraphrased_sibling", 0.0)):
        assert by_kind[kind]["rate"] == rate, (kind, by_kind[kind])
    assert 0 < summary["by_rule"]["R1"]["true_accept"]["rate"] < 1
    rows = summary["rows"]
    r2 = next(r for r in rows if r["kind"] == "r2_false_relation")
    assert r2["rules_applied"] == ["R2_relational_clause_citation"] and r2["should_exempt"] is False and "关系" in r2["condition"]
    assert summary["observed_real_triggers"]["applied"].get("R1_multi_evidence_scope", 0) >= 1


def test_classify_outcomes_on_toy_verdicts():
    from foren_searcher.datagen.verify.calibrate import recalibrate
    from foren_searcher.datagen.verify.calibrate_rule_audit import RuleCase, classify

    e = error(ErrorType.UNSUPPORTED_ASSERTION, "x", span="与通报一致", evidence_ids=["E3"], method="model", extra={"finding": "F2"})
    j_other = {"finding_id": "F1", "evidence_id": "E9", "verdict": "supported", "is_image": False}
    case = RuleCase("r2_false_relation", "R2", False, "c", "r", "W1-1", "F2", (ErrorType.UNSUPPORTED_ASSERTION,), "与通报一致", "d", {}, Verdict(subject={}))
    before = Verdict(subject={}, errors=[e], judgments=[j_other], disposition="negative")
    got = classify(case, before, recalibrate(before))
    assert got["outcome"] == "exempted" and got["disposition_flipped"] is True and got["rules_applied"] == ["R2_relational_clause_citation"]
    # 同报告没有别的 supported 文本 E：不降级 → kept
    lonely = Verdict(subject={}, errors=[e], disposition="negative")
    assert classify(case, lonely, recalibrate(lonely))["outcome"] == "kept"
    # 目标错误根本没标出 → not_flagged
    clean = Verdict(subject={}, errors=[], disposition="positive")
    assert classify(case, clean, recalibrate(clean))["outcome"] == "not_flagged"


class RuleJudge(FakeJudge):
    """report 模式用：文本来源一律支持句首；句中有全称判断 → 逐对 scope_overreach；覆盖层把注入的关系句报为无支持。"""

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        from foren_searcher.datagen.verify import calibrate_rule_audit as audit_mod

        out = await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)
        statement = str(payload.get("statement") or "")
        if system_prompt in PAIR_SYSTEM_PROMPTS:
            out = {**out, "contradicted_parts": [], "scope_overreach": audit_mod.UNIVERSAL_CLAUSE in statement and bool(out.get("supported_parts"))}
        elif system_prompt in COVERAGE_SYSTEM_PROMPTS:
            bad = [c for c in (audit_mod.R2_FALSE_RELATION, audit_mod.R2_WRAPPED_FACT) if c in statement]
            out = {"unsupported_parts": bad, "grounded": not bad, "notes": "fake"}
        return out


@needs_run
def test_rule_audit_report_mode_runs_full_pipeline_with_fake_judge(tmp_path: Path):
    from foren_searcher.datagen.verify.calibrate_rule_audit import audit, main

    base = _base()
    kinds = ["r1_universal_overreach", "r2_false_relation", "r2_fact_in_relational_wrapper", "r4_substring_sibling"]
    summary = asyncio.run(audit([RUN], base, kinds=kinds, per_kind=1, seed=3, mode="report", judge=RuleJudge(), judge_name="fake",
                                say=lambda _: None))
    assert summary["mode"] == "report" and summary["n_cases"] >= 4
    got = {k: v for k, v in summary["by_kind"].items()}
    assert got["r1_universal_overreach"]["flagged"] >= 1 and got["r1_universal_overreach"]["rate"] == 1.0
    assert got["r2_false_relation"]["flagged"] >= 1 and got["r2_false_relation"]["rate"] == 1.0
    assert got["r2_fact_in_relational_wrapper"]["rate"] == 1.0
    assert got["r4_substring_sibling"]["not_flagged"] == got["r4_substring_sibling"]["n"]      # 假核验模型不报反驳：规则没被考到，不进分母
    with pytest.raises(ValueError):
        asyncio.run(audit([RUN], base, kinds=kinds, per_kind=1, seed=3, mode="report", judge=None, say=lambda _: None))
    # CLI（verdict 模式，不调模型）
    out = tmp_path / "rule_audit.json"
    assert main(["--runs", str(RUN), "--verdicts", str(VERDICTS), "--units", str(tmp_path / "none.jsonl"), "--per-kind", "1", "--seed", "3",
                 "--kinds", "r2_false_relation", "r2_one_sided_relation_long", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["by_rule"]["R2"]["false_accept"]["rate"] == 1.0 and data["by_rule"]["R2"]["true_accept"]["rate"] == 0.0
    assert data["observed_real_triggers"]["units"] >= 1
