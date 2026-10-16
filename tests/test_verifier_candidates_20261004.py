"""核验器候选（默认关闭）：推理核对、答案语义覆盖、程序规则不碰案卷断言。片段取自 r0c / r0cn 的真实判定。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from foren_searcher.datagen.cases import Case
from foren_searcher.datagen.errors import ErrorType, Severity, error
from foren_searcher.datagen.verify import anchored as A
from foren_searcher.datagen.verify import inference as I
from foren_searcher.datagen.verify import semantic_coverage as S
from foren_searcher.datagen.verify.verdict import Verdict


class FakeJudge:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply, self.calls = reply, []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list, schema: dict | None = None) -> dict[str, Any]:
        self.calls.append(payload)
        return self.reply


@pytest.fixture(autouse=True)
def _reset():
    saved = (dict(A.PROGRAM_RULES), dict(I.INFERENCE_CHECK), dict(S.SEMANTIC_COVERAGE))
    yield
    A.PROGRAM_RULES.update(saved[0]); I.INFERENCE_CHECK.update(saved[1]); S.SEMANTIC_COVERAGE.update(saved[2])


def _err(sid: str, span: str, **extra: Any):
    return error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=span, method="model",
                 severity=Severity.CRITICAL, extra={"finding": sid, "sentence": sid, **extra})


STATEMENTS_699 = {"A1": "结论：该声明错误，混淆了两根铁柱；",
                  "A2": "德里铁柱的铭文研究支持笈多时期的建造归属，而非莫卧儿人建造。",
                  "A3": "附图对应的是拉贾斯坦邦 Bharatpur 的 Lohagarh Fort 内 Jawahar Burj 铁柱。"}
JUDGMENTS_699 = [
    {"finding_id": "A2", "evidence_id": "E27", "verdict": "supported",
     "supported": [{"part": "笈多时期的建造归属", "quote": "constructed during the rule of the imperial Guptas", "verified": True}]},
    {"finding_id": "A3", "evidence_id": "E36", "verdict": "supported", "is_image": True,
     "supported": [{"part": "Jawahar Burj 铁柱", "quote": "石碑写有“जवाहर बुर्ज”", "verified": None}]},
    {"finding_id": "A1", "evidence_id": "E27", "verdict": "not_found", "supported": []},
]


def test_helpers() -> None:
    s = "2025年的报道仍记载它的生活，反驳了“Shenkin IV在2022年该活动中被献祭”的解释。"
    assert I.quoted_mention("Shenkin IV在2022年该活动中被献祭", s)
    assert not I.quoted_mention("Shenkin IV在2022年该活动中被献祭", "ITV 称“Shenkin IV在2022年该活动中被献祭”。")   # 没有反驳提示词
    assert I.new_tokens("后段约从2.333秒延伸至结束", "B 在 2.5 秒出现") == ["2.333"]
    assert I.new_tokens("混淆了两根铁柱", "德里铁柱 Bharatpur") == []
    assert I.new_tokens("位于 Agra 的铁柱", "Bharatpur 铁柱") == ["Agra"]


def test_inference_disabled_is_noop() -> None:
    I.INFERENCE_CHECK["enabled"] = False
    judge = FakeJudge({"status": "derivable", "premises": ["A2"], "missing": "", "note": ""})
    errs = [_err("A1", "混淆了两根铁柱")]
    assert asyncio.run(I.apply_inference_check(errs, STATEMENTS_699, JUDGMENTS_699, judge, key="sentence")) == errs
    assert not judge.calls


def test_inference_derivable_downgrades_and_keeps_trace() -> None:
    I.INFERENCE_CHECK["enabled"] = True
    judge = FakeJudge({"status": "derivable", "premises": ["A2", "A3"], "missing": "", "note": "组合两句"})
    out = asyncio.run(I.apply_inference_check([_err("A1", "混淆了两根铁柱")], STATEMENTS_699, JUDGMENTS_699, judge, key="sentence"))
    assert out[0].severity is Severity.MINOR and out[0].extra["program_rule"] == "inference_derivable"
    assert out[0].extra["original_severity"] == "critical" and out[0].extra["premises"] == ["A2", "A3"]
    ids = [p["id"] for p in judge.calls[0]["premises"]]
    assert ids == ["A2", "A3"]                        # 本句与没有支持片段的句子不当前提


def test_inference_keeps_new_fact_and_guards() -> None:
    I.INFERENCE_CHECK["enabled"] = True
    judge = FakeJudge({"status": "new_fact", "premises": [], "missing": "缺建造年代", "note": ""})
    out = asyncio.run(I.apply_inference_check([_err("A1", "混淆了两根铁柱")], STATEMENTS_699, JUDGMENTS_699, judge, key="sentence"))
    assert out[0].severity is Severity.CRITICAL and out[0].extra["inference"]["status"] == "new_fact"
    # 片段带了前提里没有的拉丁专名：不问模型，保留原判
    judge2 = FakeJudge({"status": "derivable", "premises": ["A2"], "missing": "", "note": ""})
    out = asyncio.run(I.apply_inference_check([_err("A1", "铁柱实际在 Agra")], STATEMENTS_699, JUDGMENTS_699, judge2, key="sentence"))
    assert out[0].severity is Severity.CRITICAL and not judge2.calls
    # 说可推出却没指出用了哪句前提：不算
    judge3 = FakeJudge({"status": "derivable", "premises": [], "missing": "", "note": ""})
    out = asyncio.run(I.apply_inference_check([_err("A1", "混淆了两根铁柱")], STATEMENTS_699, JUDGMENTS_699, judge3, key="sentence"))
    assert out[0].severity is Severity.CRITICAL
    # 前提句自己有实锤错误：不当前提
    judge4 = FakeJudge({"status": "derivable", "premises": ["A2"], "missing": "", "note": ""})
    errs = [_err("A1", "混淆了两根铁柱"), _err("A3", "Lohagarh Fort")]
    asyncio.run(I.apply_inference_check(errs, STATEMENTS_699, JUDGMENTS_699, judge4, key="sentence"))
    assert all("A3" not in [p["id"] for p in c["premises"]] for c in judge4.calls)


def test_case_checks_are_never_touched() -> None:
    """17838107：标签判定（extra.check）是案卷断言，规则与推理核对都不能降它。"""

    A.PROGRAM_RULES.update({k: True for k in A.PROGRAM_RULES})
    I.INFERENCE_CHECK["enabled"] = True
    label = _err("A1", "该声明有证据支持，但应限定为某些设备可能出现短暂读数", check="label_expected")
    assert A.apply_program_rules([label], {"A1": label.span}, None, answer=True, key="sentence")[0].severity is Severity.CRITICAL
    judge = FakeJudge({"status": "derivable", "premises": ["A2"], "missing": "", "note": ""})
    out = asyncio.run(I.apply_inference_check([label], STATEMENTS_699, JUDGMENTS_699, judge, key="sentence"))
    assert out[0].severity is Severity.CRITICAL and not judge.calls


def _case() -> Case:
    return Case.from_dict({
        "case_id": "averimatec:699", "dataset": "averimatec", "source_id": "699", "split": "dev_train", "task_kind": "claim_check",
        "task_text": "核查铁柱声明", "family_key": "averimatec:699",
        "requirements": [{"id": "REQ1", "text": "图中铁柱的实际身份与所在地", "must_cite": True},
                         {"id": "REQ2", "text": "顾特卜塔建筑群内铁柱铭文的实际内容", "must_cite": True}],
        "required_evidence": [{"key": "altnews", "description": "Alt News", "must_cite": True, "supports": ["REQ1"],
                               "url_pattern": "altnews.in/*"},
                              {"key": "wiki", "description": "Wikipedia", "must_cite": True, "supports": ["REQ2"],
                               "url_pattern": "en.wikipedia.org/wiki/iron_pillar_of_delhi*"}],
    })


def _answer_verdict() -> Verdict:
    v = Verdict(subject={"run": "averimatec_699_r0cn_1", "request": "request_0039"}, unit="answer",
                errors=[error(ErrorType.PREMATURE_FINISH, "must_cite 要求 REQ1, REQ2 没有任何核实的 Finding 覆盖却 FINISH")])
    v.extra = {"sufficiency": {"must_cite": ["REQ1", "REQ2"], "covered": [], "uncovered": ["REQ1", "REQ2"], "premature_finish": True}}
    v.disposition = "negative"
    return v


REPORT_ROWS = [{"unit": "report", "subject": {"workspace_id": "W1-2"}, "tainted_findings": ["F9"], "judgments": [
    {"finding_id": "F3", "verdict": "supported", "source": "eprints.nmlindia.org",
     "supported": [{"part": "笈多时期", "quote": "constructed during the rule of the imperial Guptas", "verified": True}]},
    {"finding_id": "F9", "verdict": "supported", "source": "x", "supported": [{"part": "污染", "quote": "q", "verified": True}]},
    {"finding_id": "F4", "verdict": "supported", "source": "y", "supported": [{"part": "未回查", "quote": "q", "verified": False}]},
]}]


def test_verified_fragments_skip_tainted_and_unverified() -> None:
    frags = S.verified_fragments(REPORT_ROWS)
    assert [f["finding"] for f in frags] == ["W1-2/F3"]


def test_semantic_coverage() -> None:
    S.SEMANTIC_COVERAGE["enabled"] = True
    both = FakeJudge({"requirements": [{"id": "REQ1", "covered": True, "by": ["W1-2/F3"], "note": ""},
                                       {"id": "REQ2", "covered": True, "by": ["W1-2/F3"], "note": ""}]})
    v = asyncio.run(S.apply_semantic_coverage(_answer_verdict(), _case(), REPORT_ROWS, both))
    assert not any(e.type is ErrorType.PREMATURE_FINISH for e in v.errors) and v.disposition == "positive"
    assert v.extra["sufficiency"]["uncovered_by_url"] == ["REQ1", "REQ2"] and v.extra["sufficiency"]["uncovered"] == []
    one = FakeJudge({"requirements": [{"id": "REQ1", "covered": True, "by": [], "note": ""}, {"id": "REQ2", "covered": False, "by": [], "note": ""}]})
    v = asyncio.run(S.apply_semantic_coverage(_answer_verdict(), _case(), REPORT_ROWS, one))
    pf = [e for e in v.errors if e.type is ErrorType.PREMATURE_FINISH]
    assert len(pf) == 1 and "REQ2" in pf[0].detail and "REQ1" not in pf[0].detail and v.disposition == "negative"
    S.SEMANTIC_COVERAGE["enabled"] = False
    judge = FakeJudge({})
    v = asyncio.run(S.apply_semantic_coverage(_answer_verdict(), _case(), REPORT_ROWS, judge))
    assert v.disposition == "negative" and not judge.calls


def test_feature_version_labels() -> None:
    A.PROGRAM_RULES.update({k: True for k in A.PROGRAM_RULES})
    I.INFERENCE_CHECK["enabled"] = True
    S.SEMANTIC_COVERAGE["enabled"] = True
    assert A.feature_version() == "verifier@0.4.0"
    A.PROGRAM_RULES.update({k: False for k in A.PROGRAM_RULES})
    assert A.feature_version() == "verifier@0.4.0-partial:infer+semcov"
    I.INFERENCE_CHECK["enabled"] = False
    S.SEMANTIC_COVERAGE["enabled"] = False
    assert A.feature_version() == "verifier@0.3.5"


def test_defaults_on_and_env_switch(monkeypatch) -> None:
    monkeypatch.delenv("FOREN_VERIFIER_CANDIDATES", raising=False)
    assert A.candidate_enabled("rules") and A.candidate_enabled("infer") and A.candidate_enabled("semcov")
    monkeypatch.setenv("FOREN_VERIFIER_CANDIDATES", "none")
    assert not any(A.candidate_enabled(x) for x in ("rules", "infer", "semcov"))
    monkeypatch.setenv("FOREN_VERIFIER_CANDIDATES", "rules,semcov")
    assert A.candidate_enabled("rules") and A.candidate_enabled("semcov") and not A.candidate_enabled("infer")


def _report_verdict(errors: list, judgments: list, scope: list[str] | None = None) -> Verdict:
    v = Verdict(subject={"run": "true_20361105_r0c_1", "workspace_id": "W1-1"}, unit="report", errors=errors, judgments=judgments)
    v.extra = {"scope": scope} if scope else {}
    v.disposition = "negative"
    return v


def _case_wilson() -> Case:
    return Case.from_dict({
        "case_id": "true:20361105", "dataset": "true", "source_id": "20361105", "split": "dev_train", "task_kind": "claim_check",
        "task_text": "核查 Wilson 的说法", "family_key": "true:20361105",
        "requirements": [{"id": "REQ1", "text": "Wilson 是否说过这句话", "must_cite": True},
                         {"id": "REQ2", "text": "记者提问的语境", "must_cite": True}],
        "required_evidence": [{"key": "theblaze", "description": "The Blaze", "must_cite": True, "supports": ["REQ1"], "url_pattern": "theblaze.com/*"},
                              {"key": "freebeacon", "description": "Free Beacon", "must_cite": True, "supports": ["REQ1", "REQ2"],
                               "url_pattern": "freebeacon.com/*"}],
        "required_evidence_rule": "any_one_of:theblaze,freebeacon",
    })


FOX = [{"finding_id": "F1", "verdict": "supported", "source": "foxnews.com",
        "supported": [{"part": "Wilson 说应起诉", "quote": "they should be prosecuted", "verified": True}]}]


def test_report_semantic_coverage_downgrades_only_resolved_keys() -> None:
    S.SEMANTIC_COVERAGE["enabled"] = True
    exec_err = error(ErrorType.OMITTED_REQUIRED_FACT, "必需证据 theblaze / freebeacon 未被任何 Finding 引用", method="exec",
                     extra={"confidence": "hard", "check": "required_evidence_cited", "key": "theblaze", "keys": ["theblaze", "freebeacon"]})
    l3 = error(ErrorType.OMITTED_REQUIRED_FACT, "任务要求 'freebeacon' 未被回答", method="model", extra={"requirement": "freebeacon", "from_case": True})
    cwf = error(ErrorType.COMPLETED_WITHOUT_FINISHING, "报 COMPLETED 但任务要求未全部覆盖：freebeacon", method="model", extra={})
    other = error(ErrorType.COMPLETED_WITHOUT_FINISHING, "报 COMPLETED 但审核认为应为 INCOMPLETE", method="model", extra={})
    # 范围只有 REQ1：freebeacon 的 REQ2 不在本工作区，不必问
    judge = FakeJudge({"requirements": [{"id": "REQ1", "covered": True, "by": ["W1-1/F1"], "note": "Fox 引语"}]})
    v = asyncio.run(S.apply_report_semantic_coverage(_report_verdict([exec_err, l3, cwf], FOX, ["REQ1"]), _case_wilson(), judge))
    assert [r["id"] for r in judge.calls[0]["requirements"]] == ["REQ1"]
    assert all(e.severity is Severity.MINOR and e.extra["program_rule"] == "semantic_coverage" for e in v.errors)
    assert v.disposition == "positive"
    # 不收窄范围时 freebeacon 还撑着 REQ2；REQ2 没建立 → freebeacon 相关错误不降；"应为 INCOMPLETE"这类没列缺项的不碰
    judge = FakeJudge({"requirements": [{"id": "REQ1", "covered": True, "by": [], "note": ""}, {"id": "REQ2", "covered": False, "by": [], "note": ""}]})
    v = asyncio.run(S.apply_report_semantic_coverage(_report_verdict([exec_err, l3, other], FOX), _case_wilson(), judge))
    assert [e.severity for e in v.errors] == [exec_err.severity, l3.severity, other.severity] and v.disposition == "negative"
    # 关闭时不调用
    S.SEMANTIC_COVERAGE["enabled"] = False
    judge = FakeJudge({})
    asyncio.run(S.apply_report_semantic_coverage(_report_verdict([exec_err], FOX, ["REQ1"]), _case_wilson(), judge))
    assert not judge.calls


def test_sufficiency_merges_any_one_of_groups_0_3_6() -> None:
    from foren_searcher.datagen.verify import sufficiency as SU

    # verifier@0.4.0：按网址指定的参考来源不进充分性层（是答案对照，第一层照常核）；合组逻辑改用按取得方式匹配的同构案卷测
    wilson = _case_wilson()
    assert SU._required_entries(wilson, wilson.scoped_required_evidence({"REQ1"})) == []
    raw = wilson.to_dict()
    for item in raw["required_evidence"]:
        item["acquired_by"], item["url_pattern"] = f"SEARCH/open/{item['key']}", None
    case = Case.from_dict(raw)
    entries = SU._required_entries(case, case.scoped_required_evidence({"REQ1"}))
    assert [e["key"] for e in entries] == ["theblaze|freebeacon"] and entries[0]["must_cite"]
    look = SU._entry_lookup(entries)
    assert look[SU._norm_key("freebeacon(Free Beacon 报道)")] == "theblaze|freebeacon"
    assert look[SU._norm_key("theblaze|freebeacon")] == "theblaze|freebeacon"

    class J:
        def __init__(self, reply):
            self.reply = reply

        async def judge(self, **kw):
            return self.reply

    report = {"status": "COMPLETED", "findings": [{"finding_id": "F1", "statement": "x", "evidence_ids": ["E1"]}]}
    # 模型只写组里一个 key 且加了括号：归一到组；covered 就没有缺项
    ok = asyncio.run(SU.verify_sufficiency(report, "任务", J({"requirements": [{"key": "freebeacon（报道）", "covered": True, "by": "F1", "note": ""}],
                                                              "answerable_from_report": True, "status_should_be": "COMPLETED", "missing": []}),
                                            case=case, requirement_ids={"REQ1"}))
    assert ok.errors == []
    bad = asyncio.run(SU.verify_sufficiency(report, "任务", J({"requirements": [{"key": "theblaze|freebeacon", "covered": False, "by": None, "note": "n"}],
                                                               "answerable_from_report": False, "status_should_be": "INCOMPLETE", "missing": []}),
                                             case=case, requirement_ids={"REQ1"}))
    omitted = [e for e in bad.errors if e.type is ErrorType.OMITTED_REQUIRED_FACT]
    assert len(omitted) == 1 and omitted[0].extra["keys"] == ["theblaze", "freebeacon"] and omitted[0].extra["requirement"] == "theblaze"
