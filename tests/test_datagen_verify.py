"""三层核验器：第一层纯程序、第二层引语回查、第三层充分性、编排与处置。第二三层用假的核验模型。"""

import asyncio
import json
from pathlib import Path

import pytest

from foren_searcher.datagen import cases as cases_mod
from foren_searcher.datagen.errors import ErrorType, Severity
from foren_searcher.datagen.verify import DictLookup, StoreLookup, exec_checks, verify_report
from foren_searcher.datagen.verify.anchored import quote_in_source, verify_anchored
from foren_searcher.datagen.verify.context import EvidenceText, normalize_source_key
from foren_searcher.datagen.verify.exec_checks import dates_in, numbers_in
from foren_searcher.datagen.verify.verdict import Verdict, decide, read_verdicts, write_verdicts
from foren_searcher.store import GlobalStore

RUN = Path(__file__).resolve().parents[1] / "experiments" / "sft_data" / "runs" / "averimatec_441_q0_r1"

PAGE = ("Fact check: the grave in the viral photo is in Hyderabad, India. AFP reporters visited the cemetery on May 2, 2023 "
        "and photographed the grave. Locals told AFP the grille was installed to stop stray dogs. The post claimed it was in Pakistan.")


def lookup() -> DictLookup:
    return DictLookup({
        "E1": EvidenceText("E1", "", modalities=("image",), label="材料 image_0"),
        "E3": EvidenceText("E3", PAGE, source_url="https://www.afp.com/fact-check/grave-hyderabad", source_ref="R2"),
        "E6": EvidenceText("E6", "Alt News visited Darab Jung Colony, Madannapet, Hyderabad on 2023-05-01.",
                           source_url="https://www.altnews.in/grave-hyderabad", source_ref="R8"),
        "E9": EvidenceText("E9", "Unrelated page about cricket scores 2023.", source_url="https://www.boomlive.in/fact-check/x"),
    })


def report(**overrides) -> dict:
    base = {"task_id": "W1-1", "status": "COMPLETED",
            "findings": [{"finding_id": "F1", "statement": "坟墓位于印度海得拉巴，AFP 记者于 2023-05-02 到访拍摄。",
                          "evidence_ids": ["E3"], "limitations": []}],
            "selected_evidence_ids": ["E3"]}
    base.update(overrides)
    return base


def kinds(errors):
    return [e.type for e in errors]


# ------------------------------------------------------------------ 工具函数

def test_number_and_date_extraction():
    assert numbers_in("共 1,234 人，占 12.5%，2023 年，第 3 次") == {"1234", "12.5", "2023"}
    assert dates_in("2023年5月2日 与 2023-05-01") == ["2023-05-02", "2023-05-01"]
    assert normalize_source_key("https://www.BoomLive.in/fact-check/x/?utm=1") == "boomlive.in/fact-check/x"
    assert quote_in_source("visited the  cemetery on May 2, 2023", PAGE) and not quote_in_source("Pakistan grave", PAGE)


# ------------------------------------------------------------------ 第一层

def test_exec_hard_errors():
    errors = exec_checks(report(findings=[{"statement": "x", "evidence_ids": ["E42"]}, {"statement": "y", "evidence_ids": []}]), lookup())
    assert kinds(errors) == [ErrorType.FABRICATED_EVIDENCE_ID, ErrorType.UNSUPPORTED_ASSERTION]
    assert all(e.extra["confidence"] == "hard" and e.method == "exec" for e in errors)
    assert kinds(exec_checks(report(findings=[]), lookup())) == [ErrorType.EMPTY_REPORT]
    hidden = exec_checks(report(), lookup(), visible_ids={"E1"})
    assert hidden[0].type is ErrorType.FABRICATED_EVIDENCE_ID and hidden[0].extra["reason"] == "not_visible"


def test_exec_numeric_flag_only_when_number_absent():
    assert exec_checks(report(), lookup()) == []       # 2023 在原文里
    flagged = exec_checks(report(findings=[{"statement": "现场有 350 名居民围观。", "evidence_ids": ["E3"]}]), lookup())
    [flag] = flagged
    assert flag.type is ErrorType.UNSUPPORTED_ASSERTION and flag.extra["confidence"] == "flag" and flag.extra["values"] == ["350"]
    assert flag.severity is Severity.MAJOR
    with_image = exec_checks(report(findings=[{"statement": "门上有 350 号。", "evidence_ids": ["E1", "E3"]}]), lookup())
    assert with_image[0].extra["needs_pixels"] is True


def make_case(**overrides) -> cases_mod.Case:
    base = dict(case_id="averimatec:441", dataset="averimatec", source_id="441", split="dev", task_kind="claim_check",
                task_text="核查照片中的坟墓是否在巴基斯坦", family_key="averimatec:441",
                source_policy=cases_mod.SourcePolicy(mode="exclude_original", excluded_keys=("boomlive.in/fact-check",)),
                required_evidence=(cases_mod.RequiredEvidence("location_report", "实地核查报道", source_hint="altnews"),),
                checks=(cases_mod.Check("entity_in_evidence", {"text": "Hyderabad"}),
                        cases_mod.Check("status_expected", {"status": "COMPLETED"}),
                        cases_mod.Check("evidence_key_required", {"key": "location_report"}),
                        cases_mod.Check("date_in_range", {"min": "2023-04-25", "max": "2023-05-10"})))
    base.update(overrides)
    return cases_mod.Case(**base)


def test_exec_case_checks():
    good = report(findings=[{"statement": "坟墓位于 Hyderabad，Alt News 于 2023-05-01 实地核查。", "evidence_ids": ["E3", "E6"]}])
    assert exec_checks(good, lookup(), case=make_case()) == []
    bad = report(status="INCOMPLETE", findings=[{"statement": "坟墓位于 Karachi，记者 2024-01-01 到访。", "evidence_ids": ["E9"]}])
    errors = exec_checks(bad, lookup(), case=make_case())
    assert set(kinds(errors)) == {ErrorType.SOURCE_RESTRICTION_MISS, ErrorType.OMITTED_REQUIRED_FACT,
                                  ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, ErrorType.UNSUPPORTED_ASSERTION}
    completed_without = report(findings=[{"statement": "坟墓位于 Hyderabad，2023-05-02。", "evidence_ids": ["E3"]}])
    errors = exec_checks(completed_without, lookup(), case=make_case())
    assert ErrorType.COMPLETED_WITHOUT_FINISHING in kinds(errors) and ErrorType.OMITTED_REQUIRED_FACT in kinds(errors)


def test_exec_interval_and_label_checks():
    case = make_case(task_kind="video_diff", checks=(cases_mod.Check("interval_iou", {"start": 12.0, "end": 15.5, "min_iou": 0.5}),
                                                     cases_mod.Check("label_expected", {"label": "已编辑"})),
                     required_evidence=(), source_policy=cases_mod.SourcePolicy())
    ok = report(findings=[{"statement": "视频在 12.0–15.0 秒被编辑，判定为已编辑。", "evidence_ids": ["E3"]}])
    hard = [e for e in exec_checks(ok, lookup(), case=case) if e.extra["confidence"] == "hard"]
    assert hard == []                      # 12.0/15.0 不在文本里只会是 flag（数字来自视频）
    off = report(findings=[{"statement": "视频在 40–45 秒有差异。", "evidence_ids": ["E3"]}])
    hard = [e for e in exec_checks(off, lookup(), case=case) if e.extra["confidence"] == "hard"]
    assert set(kinds(hard)) == {ErrorType.UNSUPPORTED_ASSERTION, ErrorType.OMITTED_REQUIRED_FACT}
    assert [e.extra.get("best_iou") for e in hard if e.type is ErrorType.UNSUPPORTED_ASSERTION] == [0.0]


# ------------------------------------------------------------------ 第二层（假模型）

class FakeJudge:
    """按 (statement 片段, evidence_id) 查表返回逐对判定；覆盖汇总按 coverage 表；充分性按 sufficiency。记录调用。"""

    def __init__(self, table: dict, sufficiency: dict | None = None, coverage: dict | None = None, fail_times: int = 0):
        self.table, self.sufficiency, self.coverage, self.calls = table, sufficiency or {}, coverage or {}, []
        self.fail_times = fail_times

    async def judge(self, *, system_prompt, payload, images):
        self.calls.append(payload)
        if self.fail_times:
            self.fail_times -= 1
            raise ValueError("bad json")
        if "report" in payload:
            return self.sufficiency
        if "supported_parts" in payload:                       # 覆盖汇总
            for needle, answer in self.coverage.items():
                if needle in payload["statement"]:
                    return answer
            return {"unsupported_parts": [], "grounded": True}
        eid = payload["source"]["evidence_id"]
        for (needle, wanted), answer in self.table.items():
            if needle in payload["statement"] and wanted == eid:
                return answer
        return {"supported_parts": [], "contradicted_parts": []}


def sup(part, quote, **extra):
    return {"supported_parts": [{"part": part, "quote": quote}], "contradicted_parts": [], **extra}


def con(part, quote, hint=None):
    return {"supported_parts": [], "contradicted_parts": [{"part": part, "quote": quote, "hint": hint}]}


def run(coro):
    return asyncio.run(coro)


def test_anchored_quote_is_checked_against_source():
    fake = FakeJudge({("海得拉巴", "E3"): sup("坟墓位于印度海得拉巴", "the grave in the viral photo is in Hyderabad, India")})
    result = run(verify_anchored(report(), lookup(), fake))
    assert result.errors == [] and result.judgments[0].supported[0].verified is True and result.coverage[0]["grounded"]
    liar = FakeJudge({("海得拉巴", "E3"): sup("坟墓位于印度海得拉巴", "这段话原文里根本没有")},
                     coverage={"海得拉巴": {"unsupported_parts": ["坟墓位于印度海得拉巴"], "grounded": False}})
    result = run(verify_anchored(report(), lookup(), liar))
    assert result.judgments[0].verdict == "not_found" and result.judgments[0].supported == []     # 引语对不上，整条不采信
    [err] = result.errors
    assert err.type is ErrorType.UNSUPPORTED_ASSERTION and err.span == "坟墓位于印度海得拉巴"


def test_anchored_partial_support_is_not_an_error():
    """复合句：每条 E 只支持一部分，合起来全覆盖 → 不报错（v1 会把每对都判 not_found）。"""

    findings = [{"finding_id": "F1", "statement": "坟墓在海得拉巴；Alt News 于 2023-05-01 实地核查。", "evidence_ids": ["E3", "E6"]}]
    fake = FakeJudge({("海得拉巴", "E3"): sup("坟墓在海得拉巴", "is in Hyderabad, India"),
                      ("海得拉巴", "E6"): sup("Alt News 于 2023-05-01 实地核查", "Alt News visited Darab Jung Colony, Madannapet, Hyderabad on 2023-05-01")})
    result = run(verify_anchored(report(findings=findings), lookup(), fake))
    assert result.errors == [] and [j.verdict for j in result.judgments] == ["supported", "supported"]
    assert len(fake.calls) == 3      # 两对 + 一次覆盖汇总


def test_anchored_error_mapping():
    findings = [{"finding_id": "F1", "statement": "坟墓在巴基斯坦。", "evidence_ids": ["E3"]},
                {"finding_id": "F2", "statement": "坟墓在海得拉巴，共 350 人围观。", "evidence_ids": ["E3", "E9"]},
                {"finding_id": "F3", "statement": "栅栏是为了防野狗。", "evidence_ids": ["E3"]},
                {"finding_id": "F4", "statement": "记者 2023-05-03 到访。", "evidence_ids": ["E3"]}]
    fake = FakeJudge({
        ("巴基斯坦", "E3"): con("坟墓在巴基斯坦", "is in Hyderabad, India"),
        ("海得拉巴", "E3"): sup("坟墓在海得拉巴", "is in Hyderabad, India"),
        ("防野狗", "E3"): sup("栅栏是为了防野狗", "Locals told AFP the grille was installed to stop stray dogs",
                             source_reports_claim=True, statement_asserts_fact=True),
        ("2023-05-03", "E3"): con("2023-05-03", "May 2, 2023", hint="sloppy_numeric"),
    }, coverage={"350": {"unsupported_parts": ["共 350 人围观"], "grounded": False}})
    result = run(verify_anchored(report(findings=findings), lookup(), fake))
    by = {}
    for e in result.errors:
        by.setdefault(e.extra["finding"], []).append(e)
    assert by["F1"][0].type is ErrorType.WRONG_CITATION and by["F1"][0].severity is Severity.CRITICAL and by["F1"][0].span == "坟墓在巴基斯坦"
    assert by["F2"][0].type is ErrorType.UNSUPPORTED_ASSERTION and by["F2"][0].span == "共 350 人围观"
    assert by["F3"][0].type is ErrorType.ATTRIBUTION_DRIFT
    assert by["F4"][0].type is ErrorType.SLOPPY_NUMERIC


def test_anchored_bad_judge_output_becomes_pending_not_crash():
    fake = FakeJudge({}, fail_times=2)      # 第一对连坏两次 → failed → pending；后续正常
    result = run(verify_anchored(report(), lookup(), fake))
    assert result.judgments[0].verdict == "failed" and result.pending[0]["reason"] == "judge_output_invalid"
    assert result.errors == []


def test_anchored_image_without_pixels_is_pending():
    result = run(verify_anchored(report(findings=[{"statement": "图中有锁。", "evidence_ids": ["E1"]}]), lookup(), FakeJudge({})))
    assert result.errors == [] and result.pending[0]["reason"] == "image_without_pixels"


# ------------------------------------------------------------------ 编排 + 处置

def test_verify_report_without_judge_is_uncertain_and_keeps_flags():
    verdict = run(verify_report(report(findings=[{"statement": "现场 350 人。", "evidence_ids": ["E3"]}]), lookup(), subject={"case_id": "averimatec:441"}))
    assert verdict.layers_run == ["exec"] and verdict.disposition == "uncertain"
    assert verdict.pending[0]["reason"] == "no_judge" and verdict.errors[0].extra["confidence"] == "flag"


def test_sufficiency_only_reports_case_requirements():
    from foren_searcher.datagen.verify.sufficiency import verify_sufficiency
    fake = FakeJudge({}, sufficiency={"requirements": [{"key": "说明来源与适用范围", "covered": False, "note": "套话"},
                                                       {"key": "location_report", "covered": False, "note": "未引用"}],
                                      "answerable_from_report": False, "status_should_be": "INCOMPLETE"})
    result = run(verify_sufficiency(report(), "任务", fake, case=make_case()))
    assert [e.extra.get("requirement") for e in result.errors if e.type is ErrorType.OMITTED_REQUIRED_FACT] == ["location_report"]
    assert ErrorType.COMPLETED_WITHOUT_FINISHING in kinds(result.errors)
    no_case = run(verify_sufficiency(report(), "任务", fake))
    assert kinds(no_case.errors) == [ErrorType.COMPLETED_WITHOUT_FINISHING]     # 模型归纳的要求只留 raw


def test_verify_report_full_pipeline_and_dedup(tmp_path: Path):
    findings = [{"finding_id": "F1", "statement": "坟墓位于 Hyderabad，AFP 于 2023-05-02 到访。", "evidence_ids": ["E3"]}]
    fake = FakeJudge({("Hyderabad", "E3"): sup("AFP 于 2023-05-02 到访", "AFP reporters visited the cemetery on May 2, 2023")},
                     sufficiency={"requirements": [{"key": "location_report", "covered": False, "by": None, "note": "未引用 Alt News"}],
                                  "answerable_from_report": True, "status_should_be": "INCOMPLETE", "missing": ["location_report"]})
    verdict = run(verify_report(report(findings=findings), lookup(), subject={"case_id": "averimatec:441", "workspace_id": "W1-1"},
                                case=make_case(), judge=fake, judge_model="fake", spec_hash="sha256:abc", state_renderer="compact_16k"))
    assert verdict.layers_run == ["exec", "anchored", "sufficiency"]
    # 第一层已判 location_report 未引用（hard）+ COMPLETED_WITHOUT_FINISHING；第三层同 key 不重复
    assert verdict.counts["omitted_required_fact"] == 1 and verdict.counts["completed_without_finishing"] == 1
    assert verdict.disposition == "negative" and verdict.tainted_findings == []
    path = write_verdicts(tmp_path / "verdicts.jsonl", [verdict])
    [loaded] = read_verdicts(path)
    assert loaded.to_dict() == verdict.to_dict() and loaded.spec_hash == "sha256:abc"


def test_decide_rules():
    from foren_searcher.datagen.errors import error
    major = lambda: error(ErrorType.SCOPE_OVERREACH, "x")
    assert decide([], []) == "positive"
    assert decide([major()], []) == "positive"
    assert decide([major(), major()], []) == "negative"
    assert decide([error(ErrorType.WRONG_CITATION, "x")], []) == "negative"
    assert decide([], [{"reason": "image_without_pixels"}]) == "uncertain"
    assert decide([error(ErrorType.UNSUPPORTED_ASSERTION, "x", extra={"confidence": "flag"})], []) == "uncertain"


def test_tainted_findings_follow_propagation():
    findings = [{"finding_id": "F1", "statement": "在巴基斯坦。", "evidence_ids": ["E3"]},
                {"finding_id": "F2", "statement": "记者 2023-05-02 到访。", "evidence_ids": ["E3"]}]
    fake = FakeJudge({("巴基斯坦", "E3"): con("在巴基斯坦", "is in Hyderabad, India"),
                      ("到访", "E3"): sup("记者 2023-05-02 到访", "visited the cemetery on May 2, 2023")},
                     sufficiency={"requirements": [], "answerable_from_report": True, "status_should_be": "COMPLETED"})
    verdict = run(verify_report(report(findings=findings), lookup(), subject={}, task_text="核查地点", judge=fake))
    assert verdict.tainted_findings == ["F1"] and verdict.disposition == "negative"


# ------------------------------------------------------------------ 真实 Store 冒烟

@pytest.mark.skipif(not (RUN / "store.sqlite3").is_file(), reason="试点 run 未随仓库入库")
def test_store_lookup_and_exec_layer_on_pilot_run():
    store = GlobalStore(RUN / "store.sqlite3", RUN / "artifacts")
    look = StoreLookup(store)
    assert look.has("E6") and not look.has("E999") and not look.has("R1")
    e6 = look.get("E6")
    assert e6.is_text and e6.source_url and "altnews" in e6.source_url and e6.source_ref == "R8"
    assert look.get("E1").is_image
    published = store.workspace_report("W1-1").to_dict()
    errors = exec_checks(published, look)
    assert all(e.extra["confidence"] == "flag" for e in errors)     # 试点报告：无硬错，数字可能被标记


# ------------------------------------------------------------------ 接法与重核 CLI

def test_callable_and_openrouter_judges_parse_json():
    from foren_searcher.datagen.verify.judges import CallableJudge, OpenRouterJudge, build_judge

    async def gen(system_prompt, payload, images):
        return '```json\n{"verdict": "supported", "quote": "x"}\n```'
    assert run(CallableJudge(gen).judge(system_prompt="s", payload={}, images=[]))["verdict"] == "supported"

    class FakeClient:
        model = "teacher-model"
        def __init__(self): self.calls, self.preloads = [], 0
        async def preload(self): self.preloads += 1
        async def complete(self, messages, *, role, json_output=False, **kw):
            if not self.preloads:
                raise RuntimeError("OpenRouter client must be preloaded")
            self.calls.append((messages, role, json_output))
            return {"text": '{"verdict":"not_found","quote":""}'}
    client = FakeClient()
    judge = OpenRouterJudge(client, model="verifier-model")
    assert judge.name == "openrouter:verifier-model" and client.model == "verifier-model"
    out = run(judge.judge(system_prompt="s", payload={"statement": "x"}, images=[]))
    run(judge.judge(system_prompt="s", payload={"statement": "y"}, images=[]))
    assert out["verdict"] == "not_found" and client.preloads == 1        # preload 只做一次
    messages, role, json_output = client.calls[0]
    assert role == "VERIFIER" and json_output and messages[0]["role"] == "system" and "statement" in messages[1]["content"]
    assert build_judge("none") is None
    with pytest.raises(ValueError):
        build_judge("magic")


@pytest.mark.skipif(not (RUN / "store.sqlite3").is_file(), reason="试点 run 未随仓库入库")
def test_reverify_pilot_run_exec_only(tmp_path: Path):
    from foren_searcher.datagen.verify import reverify

    verdicts = run(reverify.reverify_run(RUN, None, None, check_visibility=True))
    assert len(verdicts) == 1 and verdicts[0].subject["workspace_id"] == "W1-1" and verdicts[0].subject["case_id"] == "averimatec:441"
    assert verdicts[0].layers_run == ["exec"] and verdicts[0].disposition == "uncertain"
    summary = reverify.summarize(verdicts)
    assert summary["reports"] == 1 and summary["pending"] == {"no_judge": 1}
    assert reverify.main(["--runs", str(RUN.parent), "--out", str(tmp_path / "v.jsonl"), "--limit", "1"]) == 0
    assert (tmp_path / "v.jsonl.summary.json").is_file() and len(read_verdicts(tmp_path / "v.jsonl")) >= 1


def test_numbers_missing_tolerates_formatting():
    from foren_searcher.datagen.verify.exec_checks import numbers_missing
    assert numbers_missing("支出 R942,827,946，帧时刻 1.033333 秒", "total 942 827 946 ... 视频E1在1.033秒的帧") == []
    assert numbers_missing("合计 3,735,245", "row 2 179 767 635 223 735 245 Total") == []      # 分组数字串匹配
    assert numbers_missing("约 493,100,000 兰特", "R493.1 million") == ["493100000"]           # 换算值留给模型层


@pytest.mark.skipif(not (RUN.parent / "videosham_4126_split_intervals_r7" / "trajectory.jsonl").is_file(), reason="需要试点 run")
def test_visible_ids_is_union_over_all_turns():
    from foren_searcher.datagen.verify.reverify import _visible_ids
    ids = _visible_ids(RUN.parent / "videosham_4126_split_intervals_r7", "W3-3")
    assert ids is not None and "E165" in ids      # 只看最后一次输入会漏掉它（卡片区只留 12 张）
    assert _visible_ids(RUN.parent / "videosham_4126_split_intervals_r7", "W9-9") is None


def test_cached_judge_reuses_answers(tmp_path: Path):
    from foren_searcher.datagen.verify.judges import CachedJudge
    inner = FakeJudge({("海得拉巴", "E3"): sup("坟墓位于印度海得拉巴", "the grave in the viral photo is in Hyderabad, India")})
    cached = CachedJudge(inner, tmp_path / "cache.jsonl")
    run(verify_anchored(report(), lookup(), cached))
    run(verify_anchored(report(), lookup(), cached))
    assert cached.misses == 2 and cached.hits == 2 and len(inner.calls) == 2      # 第二遍全部命中缓存
    reloaded = CachedJudge(inner, tmp_path / "cache.jsonl")
    run(verify_anchored(report(), lookup(), reloaded))
    assert reloaded.hits == 2 and len(inner.calls) == 2                            # 缓存文件能读回
