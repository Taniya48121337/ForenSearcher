"""verifier@0.3.1：组内全部 must_cite=False 的必需证据组 = 可选证据。

没引用只记 severity=minor 的 omitted（extra.optional=True），不计入处置、不转 uncertain、不污染 Finding；
must_cite=True 的组照旧记 hard MAJOR；any_one_of 组里只要有一项 must_cite=True，整组仍是必需。
"""

import asyncio

from foren_searcher.datagen import cases as cases_mod
from foren_searcher.datagen.errors import ErrorType, Severity
from foren_searcher.datagen.verify import exec_checks, verify_report
from foren_searcher.datagen.verify.sufficiency import verify_sufficiency
from foren_searcher.datagen.verify.verdict import VERIFIER_VERSION, decide

from test_datagen_verify import FakeJudge, lookup, report


def _case(*evidence: cases_mod.RequiredEvidence, rule: str = "all") -> cases_mod.Case:
    return cases_mod.Case(case_id="averimatec:441", dataset="averimatec", source_id="441", split="dev", task_kind="claim_check",
                          task_text="核查照片中的坟墓是否在巴基斯坦", family_key="averimatec:441",
                          required_evidence=tuple(evidence), required_evidence_rule=rule)


AFP = cases_mod.RequiredEvidence("afp", "AFP 实地报道", url_pattern="*afp.com/*")
ALT = cases_mod.RequiredEvidence("alt", "Alt News 实地报道", url_pattern="*altnews.in/*")
ALT_OPT = cases_mod.RequiredEvidence("alt", "Alt News 实地报道（旁证，可选）", url_pattern="*altnews.in/*", must_cite=False)
PROFILE_OPT = cases_mod.RequiredEvidence("media_profile", "媒体档案（可选）", acquired_by="media_profile", must_cite=False)


def test_version_bumped():
    assert VERIFIER_VERSION.startswith("verifier@0.4.")


def test_optional_group_is_minor_note_not_disposition():
    errors = exec_checks(report(), lookup(), case=_case(AFP, ALT_OPT, PROFILE_OPT))   # report 只引了 E3（AFP）
    omitted = [e for e in errors if e.type is ErrorType.OMITTED_REQUIRED_FACT]
    assert {e.extra["key"] for e in omitted} == {"alt", "media_profile"}
    assert all(e.severity is Severity.MINOR and e.extra["optional"] and e.extra["confidence"] == "hard" for e in omitted)
    assert ErrorType.COMPLETED_WITHOUT_FINISHING not in [e.type for e in errors]
    assert decide(errors, []) == "positive"


def test_required_group_still_hard_major():
    errors = exec_checks(report(), lookup(), case=_case(AFP, ALT))
    [omitted] = [e for e in errors if e.type is ErrorType.OMITTED_REQUIRED_FACT]
    assert omitted.severity is Severity.MAJOR and omitted.extra["confidence"] == "hard" and not omitted.extra.get("optional")
    assert ErrorType.COMPLETED_WITHOUT_FINISHING in [e.type for e in errors]


def test_any_one_of_with_one_must_item_stays_required():
    alt_must = cases_mod.RequiredEvidence("alt", "Alt News", url_pattern="*altnews.in/*")
    boom_opt = cases_mod.RequiredEvidence("boom", "BOOM", url_pattern="*boomlive.in/*", must_cite=False)
    errors = exec_checks(report(), lookup(), case=_case(AFP, alt_must, boom_opt, rule="all:afp; any_one_of:alt,boom"))
    [omitted] = [e for e in errors if e.type is ErrorType.OMITTED_REQUIRED_FACT]
    assert omitted.severity is Severity.MAJOR and omitted.extra["keys"] == ["alt", "boom"]


def test_unmatchable_optional_item_does_not_flag():
    """没有程序特征的可选项以前记 flag（→ uncertain，被 held）；现在只记 minor 备注。"""
    loose = cases_mod.RequiredEvidence("context", "背景材料", must_cite=False)
    errors = exec_checks(report(), lookup(), case=_case(AFP, loose))
    assert not [e for e in errors if e.extra.get("confidence") == "flag"]
    assert decide(errors, []) == "positive"


def test_sufficiency_optional_uncovered_is_minor():
    judge = FakeJudge({}, sufficiency={"requirements": [{"key": "afp", "covered": True}, {"key": "alt", "covered": False, "note": "没提 Alt News"}],
                                       "status_should_be": "COMPLETED"})
    result = asyncio.run(verify_sufficiency(report(), "t", judge, case=_case(AFP, ALT_OPT)))
    [e] = result.errors
    assert e.type is ErrorType.OMITTED_REQUIRED_FACT and e.severity is Severity.MINOR and e.extra["optional"]
    judge_req = FakeJudge({}, sufficiency={"requirements": [{"key": "afp", "covered": True}, {"key": "alt", "covered": False}],
                                           "status_should_be": "COMPLETED"})
    result = asyncio.run(verify_sufficiency(report(), "t", judge_req, case=_case(AFP, ALT)))
    assert [e.severity for e in result.errors if e.type is ErrorType.OMITTED_REQUIRED_FACT] == [Severity.MAJOR]


def test_full_verdict_positive_with_optional_gaps():
    judge = FakeJudge({}, sufficiency={"requirements": [{"key": "afp", "covered": True}, {"key": "alt", "covered": False},
                                                         {"key": "media_profile", "covered": False}], "status_should_be": "COMPLETED"},
                      coverage={"坟墓": {"unsupported_parts": [], "grounded": True}})
    judge.table = {("坟墓", "E3"): {"supported_parts": [{"part": "坟墓位于印度海得拉巴，AFP 记者于 2023-05-02 到访拍摄。",
                                                         "quote": "AFP reporters visited the cemetery on May 2, 2023"}],
                                   "contradicted_parts": []}}
    verdict = asyncio.run(verify_report(report(), lookup(), subject={"case_id": "averimatec:441"},
                                        case=_case(AFP, ALT_OPT, PROFILE_OPT), judge=judge))
    minors = [e for e in verdict.errors if e.type is ErrorType.OMITTED_REQUIRED_FACT]
    assert minors and all(e.severity is Severity.MINOR for e in minors)
    assert not verdict.tainted_findings
    assert verdict.verifier == VERIFIER_VERSION and verdict.disposition == "positive"
