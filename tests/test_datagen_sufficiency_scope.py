"""verifier@0.3.0 充分性按工作区收窄：Case.scoped_required_evidence、scope.requirements_for_workspace，
以及真实 run 上的召回校准（scoped 对照 whole）：删必需 Finding 仍能抓到，其它注入不再多出假的 omitted_required_fact。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.cases import Case, Requirement, RequiredEvidence, load_cases
from foren_searcher.datagen.verify.scope import FALLBACK_THRESHOLD, judge_targets_from_units, overlap, requirements_for_workspace
from foren_searcher.datagen.verify.sufficiency import SUFFICIENCY_SYSTEM_PROMPT
from foren_searcher.datagen.verify.verdict import Verdict

from test_datagen_judged_units import CASES, RUN, VERDICTS, FakeJudge, _replay_from_base, needs_run

UNITS = RUN.parents[1] / "verdicts_units_judged.jsonl"
needs_units = pytest.mark.skipif(not UNITS.is_file(), reason="plan 单元模型评判文件未随仓库入库")
CASE_ID = "true:4124903"


def _read(path: Path) -> list[Verdict]:
    return [Verdict.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _toy_case(**kw: Any) -> Case:
    base: dict[str, Any] = dict(case_id="true:1", dataset="true", source_id="1", split="dev", task_kind="claim_check", task_text="t",
                                family_key="true:1")
    base.update(kw)
    return Case(**base)


def _toy_with_evidence() -> Case:
    return _toy_case(
        requirements=(Requirement(id="REQ1", text="官方通报火情原因"), Requirement(id="REQ2", text="视频画面显示了什么")),
        required_evidence=(RequiredEvidence(key="notice", description="官方通报", supports=("REQ1",)),
                           RequiredEvidence(key="frames", description="视频帧", supports=("REQ2",)),
                           RequiredEvidence(key="both", description="同时支持两项", supports=("REQ1", "REQ2")),
                           RequiredEvidence(key="general", description="不分工作区的通用项")))


# ------------------------------------------------------------------ (a) 过滤与范围规则

def test_scoped_required_evidence_filters_by_supports_and_keeps_general_items():
    case = _toy_with_evidence()
    keys = lambda ids: [r.key for r in case.scoped_required_evidence(ids)]      # noqa: E731
    assert keys(None) == ["notice", "frames", "both", "general"]               # None → 整案
    assert keys({"REQ1"}) == ["notice", "both", "general"]
    assert keys({"REQ2"}) == ["frames", "both", "general"]
    assert keys(set()) == ["general"]                                           # 空集合：只剩 supports 为空的通用项
    assert keys({"REQ9"}) == ["general"]


def test_requirements_for_workspace_prefers_judge_targets_then_text_overlap():
    case = _toy_with_evidence()
    plan = Verdict(subject={"run": "r1", "round": 1}, unit="plan", extra={"gap": {"judge_targets": {"T1": ["REQ2"], "T2": ["REQ1"]}}})
    other_run = Verdict(subject={"run": "r2", "round": 1}, unit="plan", extra={"gap": {"judge_targets": {"T1": ["REQ1"]}}})
    units = [other_run, plan]
    # judge_targets 优先，即使任务文字和 REQ1 几乎一样；W{r}-{i} ↔ T{i}
    ids, note = requirements_for_workspace(case, "官方通报火情原因", run="r1", workspace_id="W1-1", units=units)
    assert ids == {"REQ2"} and note["method"] == "judge_targets"
    assert requirements_for_workspace(case, "x", run="r1", workspace_id="W1-2", units=units)[0] == {"REQ1"}
    assert judge_targets_from_units(units, "r1", "W2-1") is None                # 轮次对不上
    assert judge_targets_from_units(units, "r1", "W1-3") is None                # 没有 T3
    assert judge_targets_from_units(units, "r1", "final") is None               # 不是 W{r}-{i}
    # 取不到 judge_targets → 文字重合兜底，分数记进说明
    ids, note = requirements_for_workspace(case, "请核实官方通报里的火情原因", run="r1", workspace_id="W2-1", units=units)
    assert note["method"] == "text_overlap" and ids == {"REQ1"} and set(note["scores"]) == {"REQ1", "REQ2"}
    assert note["scores"]["REQ1"] >= FALLBACK_THRESHOLD > note["scores"]["REQ2"]
    assert overlap("视频画面", "视频画面") == 1.0 and overlap("", "x") == 0.0
    # 无 units 也走兜底
    assert requirements_for_workspace(case, "视频画面显示了什么", run="r1", workspace_id="W1-1")[1]["method"] == "text_overlap"
    # 案卷没有 requirements / 没有案卷 → None（不收窄）
    for c in (_toy_case(), None):
        ids, note = requirements_for_workspace(c, "任意", run="r1", workspace_id="W1-1", units=units)
        assert ids is None and note["method"] == "none"


@needs_units
def test_real_case_scopes_from_units_judged():
    case = load_cases(CASES)[CASE_ID]
    units = _read(UNITS)
    w1, _ = requirements_for_workspace(case, "", run=RUN.name, workspace_id="W1-1", units=units)
    w2, _ = requirements_for_workspace(case, "", run=RUN.name, workspace_id="W1-2", units=units)
    # 期望值跟 verdicts_units_judged.jsonl 走：09-25 重生成后核验模型把 T2（外部来源查火情原因）只对到 REQ1
    # （之前是 REQ1+REQ2）。这里断言的是"收窄按 judge_targets 执行"，不是某次模型判定的具体内容。
    assert w1 == {"REQ2", "REQ3"} and w2 == {"REQ1"}
    assert [r.key for r in case.scoped_required_evidence(w1)] == ["frames", "youtube_upload"]
    assert [r.key for r in case.scoped_required_evidence(w2)] == ["prefpolice_tweets", "snopes"]


# ------------------------------------------------------------------ (b) 真实 run 上的召回校准：scoped 对照 whole

class ScopeJudge(FakeJudge):
    """在 FakeJudge 上换掉充分性分支：payload 里每个 required_evidence key，只要报告仍有 Finding 引用了程序匹配到
    该 key 的 E（url_pattern / acquired_by，与第一层同一口径）就算 covered，否则 uncovered；按案卷分组规则定 status_should_be。
    删掉唯一引用某项必需证据的 Finding → 报 uncovered，这正是 drop_required_finding 要抓的。"""

    def __init__(self, replay: dict[tuple[str, str], dict[str, Any]], case: Case, matches: dict[str, set[str]]) -> None:
        super().__init__(replay)
        self.case, self.matches = case, matches
        self.sufficiency_keys: list[list[str]] = []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        if system_prompt != SUFFICIENCY_SYSTEM_PROMPT:
            return await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)
        report = payload["report"]
        cited = {e for f in report["findings"] for e in f["evidence_ids"]}
        given = {r["key"]: r for r in payload.get("required_evidence") or []}
        self.sufficiency_keys.append(sorted(given))
        covered = {k for k in given if self.matches.get(k, set()) & cited}
        short = False
        for mode, keys in self.case.evidence_groups():
            keys = tuple(k for k in keys if k in given)
            if keys and not (all if mode == "all" else any)(k in covered for k in keys) and any(given[k]["must_cite"] for k in keys):
                short = True
        return {"requirements": [{"key": k, "covered": k in covered, "by": None, "note": "fake"} for k in given],
                "answerable_from_report": bool(report["findings"]), "status_should_be": "INCOMPLETE" if short else "COMPLETED", "missing": []}


def _matches(case: Case) -> dict[str, set[str]]:
    from foren_searcher.datagen.runlog import load_run

    log = load_run(RUN)
    try:
        ids = {e for rec in log.store.workspace_report_records() for f in rec.to_dict()["findings"] for e in f.get("evidence_ids") or []}
        out: dict[str, set[str]] = {r.key: set() for r in case.required_evidence}
        for eid in ids:
            if not log.lookup.has(eid):
                continue
            t = log.lookup.get(eid)
            for r in case.required_evidence:
                if r.matches(source_url=t.source_url, acquired_by=t.acquired_by, label=t.label):
                    out[r.key].add(eid)
        return out
    finally:
        log.close()


# 不动引用的注入：它们不该改变"必需证据拿齐没有"。swap_citation 会换掉引用，可能真把唯一的必需证据换没了，不在此列。
_CITATION_NEUTRAL = ("fabricate_id", "append_clause", "alter_number", "drop_attribution")


def _omitted(row: dict[str, Any]) -> int:
    return sum(1 for e in row["errors"] if e["type"] == "omitted_required_fact")


async def _clean_omitted(case: Case, judge: ScopeJudge, units: list[Verdict], mode: str) -> dict[str, int]:
    """未注入的原报告在同一模式下的 omitted_required_fact 条数（扣掉与范围无关的案卷断言，例如结论标签检查）。"""

    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.verify.reverify import _visible_ids
    from foren_searcher.datagen.verify.verdict import verify_report

    log = load_run(RUN)
    try:
        out: dict[str, int] = {}
        for rec in log.store.workspace_report_records():
            task = str(log.store.workspace_task_record(rec.task_id).get("task") or "")
            ids = requirements_for_workspace(case, task, run=RUN.name, workspace_id=rec.task_id, units=units)[0] if mode == "scoped" else None
            v = await verify_report(rec.to_dict(), log.lookup, subject={}, case=case, task_text=task or None, judge=judge,
                                    requirement_ids=ids, visible_ids=_visible_ids(RUN, rec.task_id))
            out[rec.task_id] = sum(1 for e in v.errors if e.type.value == "omitted_required_fact")
        return out
    finally:
        log.close()


@needs_run
@needs_units
def test_scoped_calibration_catches_dropped_finding_without_false_omissions():
    from foren_searcher.datagen.verify.calibrate_recall import MUTATORS, run_calibration

    base = _read(VERDICTS)
    units = _read(UNITS)
    cases = load_cases(CASES)
    case = cases[CASE_ID]
    replay, matches = _replay_from_base(base), _matches(case)
    assert matches["frames"] and matches["youtube_upload"] and matches["prefpolice_tweets"]

    def calibrate(mode: str) -> tuple[dict[str, Any], ScopeJudge]:
        judge = ScopeJudge(replay, case, matches)
        s = asyncio.run(run_calibration([RUN], cases, base, judge, judge_name="fake", kinds=list(MUTATORS), per_kind=1, seed=3,
                                        case_mode=mode, units=units, say=lambda _: None))
        return s, judge

    scoped, sj = calibrate("scoped")
    whole, wj = calibrate("whole")
    clean_scoped = asyncio.run(_clean_omitted(case, ScopeJudge(replay, case, matches), units, "scoped"))
    clean_whole = asyncio.run(_clean_omitted(case, ScopeJudge(replay, case, matches), units, "whole"))

    # 充分性层确实只拿到本工作区的清单
    assert scoped["case_mode"] == "scoped" and whole["case_mode"] == "whole"
    assert {tuple(k) for k in sj.sufficiency_keys} == {("frames", "youtube_upload"), ("prefpolice_tweets", "snopes")}
    assert {tuple(k) for k in wj.sufficiency_keys} == {("frames", "prefpolice_tweets", "snopes", "youtube_upload")}
    # drop_required_finding 按工作区范围挑（09-29）：units 重判后 W1-2 只对 REQ1，那里 prefpolice / snopes 由 F5、F6 两条互为备份，
    # W1-1 的帧与 youtube 也各有别的 Finding 兜着 → scoped 下这个 run 没有"真会缺必需证据"的 Finding，不注入（不制造假漏报）
    drops = [r for r in scoped["rows"] if r["kind"] == "drop_required_finding"]
    assert all(r["scope"]["method"] == "judge_targets" for r in drops)
    assert all("omitted_required_fact" in r["hits"] for r in drops)
    # whole 模式按整案清单挑：W1-2 删唯一引帧的 F7 会缺 frames 组，核验器应抓到
    whole_drops = [r for r in whole["rows"] if r["kind"] == "drop_required_finding"]
    assert whole_drops and all("omitted_required_fact" in r["hits"] for r in whole_drops), whole_drops

    # scoped：W1-2 原报告已无缺项；不动引用的注入不多出 omitted_required_fact（与原报告同模式核验相比）
    assert clean_scoped["W1-2"] == 0
    neutral = [r for r in scoped["rows"] if r["kind"] in _CITATION_NEUTRAL]
    assert neutral
    for r in neutral:
        assert _omitted(r) == clean_scoped[r["workspace_id"]], r
    assert not any("omitted_required_fact" in f["types"] for f in scoped["false_new_errors"] if f["workspace_id"] == "W1-2")

    # whole 对照：整案清单压给 W1-2（没领 REQ3 却要它拿 youtube_upload），原报告本身就多出假的 omitted_required_fact；
    # 09-25 起命中 / 新错都相对同条件对照（未改原报告同 case_mode 核验）比，这条假缺项进 control_error_types，不再记成 false_new
    assert clean_whole["W1-2"] > clean_scoped["W1-2"]
    assert "omitted_required_fact" in whole["controls"]["true_4124903_r1/W1-2"]["control_error_types"]
    assert "omitted_required_fact" not in scoped["controls"]["true_4124903_r1/W1-2"]["control_error_types"]
    n_false = lambda s: sum(1 for f in s["false_new_errors"] if "omitted_required_fact" in f["types"])   # noqa: E731
    assert n_false(scoped) == 0 and n_false(whole) == 0


@needs_run
def test_exec_checks_skips_label_expected_on_workspace_reports():
    """09-24 收窄测试发现：整案标签只在 answer 单元核，工作区报告（W1-1）的第一层不再报 label_expected。"""

    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.verify.exec_checks import exec_checks

    case = load_cases(CASES)[CASE_ID]
    assert any(c.kind == "label_expected" for c in case.checks)          # 案卷确实带标签断言，跳过才有意义
    log = load_run(RUN)
    try:
        recs = [r for r in log.store.workspace_report_records() if r.task_id == "W1-1"]
        assert recs
        errors = exec_checks(recs[0].to_dict(), log.lookup, case=case)
    finally:
        log.close()
    assert not [e for e in errors if e.extra.get("check") == "label_expected"]


# ------------------------------------------------------------------ 召回校准：核验调用失败 → inconclusive，不算漏报（09-25）

def test_recall_summary_separates_inconclusive_from_misses():
    from foren_searcher.datagen.verify.calibrate_recall import pending_reasons, summarize

    reasons, last = pending_reasons([{"reason": "judge_output_invalid", "error": {"RuntimeError": "429"}, "attempts": 4,
                                      "last_error": "RuntimeError: 429"},
                                     {"reason": "judge_output_invalid", "error": {"RuntimeError": "503"}},
                                     {"reason": "image_without_pixels"}])
    assert reasons == {"judge_output_invalid": 2, "image_without_pixels": 1} and last == "RuntimeError: 503"
    base = {"run": "r", "workspace_id": "W", "finding_id": "F1", "detail": "", "expected": ["unsupported_assertion"], "new_error_types": []}
    rows = [{**base, "kind": "append_clause", "detected": True, "hits": ["unsupported_assertion"], "pending": 0},
            {**base, "kind": "append_clause", "detected": False, "hits": [], "pending": 3},
            {**base, "kind": "append_clause", "detected": False, "hits": [], "pending": 0}]
    s = summarize(rows, judge_name="fake", kinds=["append_clause"])
    assert (s["recall_overall"], s["inconclusive"], s["recall_conclusive"]) == (0.333, 1, 0.5)
    row = s["by_kind"]["append_clause"]
    assert (row["recall"], row["inconclusive"], row["recall_conclusive"]) == (0.333, 1, 0.5)
    assert [m["inconclusive"] for m in row["misses"]] == [True, False]


@needs_run
def test_calibration_marks_failed_pairs_inconclusive(monkeypatch):
    from foren_searcher.datagen.verify import anchored
    from foren_searcher.datagen.verify.anchored import PAIR_SYSTEM_PROMPTS
    from foren_searcher.datagen.verify.calibrate_recall import run_calibration
    from test_datagen_judged_units import _FAKE_MARKERS

    async def no_wait(delay, result=None):
        return result

    monkeypatch.setattr(anchored.asyncio, "sleep", no_wait)

    class FailingPairs(FakeJudge):
        """注入了编造句的 Finding：所有 pair 调用都抛（模拟并发限流）；对照（原报告）照常。"""

        async def judge(self, *, system_prompt, payload, images, schema=None):
            if system_prompt in PAIR_SYSTEM_PROMPTS and any(m in str(payload.get("statement")) for m in _FAKE_MARKERS):
                raise RuntimeError("429 rate limited")
            return await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)

    base = _read(VERDICTS)
    s = asyncio.run(run_calibration([RUN], load_cases(CASES), base, FailingPairs(_replay_from_base(base)), judge_name="fake",
                                    kinds=["append_clause"], per_kind=1, seed=3, say=lambda _: None))
    row = s["by_kind"]["append_clause"]
    assert row["n"] >= 1 and row["detected"] == 0
    assert row["inconclusive"] == row["n"] and row["recall_conclusive"] is None and s["inconclusive"] == s["n_injections"]
    for r in s["rows"]:
        assert r["pending"] > 0 and r["pending_reasons"].get("judge_output_invalid", 0) >= 1
        assert r["pending_last_error"] == "RuntimeError: 429 rate limited"
    # 同一模型不失败时这些注入都能抓到（对照：不是真漏报）
    ok = asyncio.run(run_calibration([RUN], load_cases(CASES), base, FakeJudge(_replay_from_base(base)), judge_name="fake",
                                     kinds=["append_clause"], per_kind=1, seed=3, say=lambda _: None))
    assert ok["by_kind"]["append_clause"]["recall"] == 1.0 and ok["inconclusive"] == 0
