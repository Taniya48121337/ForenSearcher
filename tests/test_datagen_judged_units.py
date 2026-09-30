"""answer 逐句锚定、plan 模型评判、召回率校准、export --units 叠加（假核验模型；对真实 run 的端到端断言）。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.cases import load_cases
from foren_searcher.datagen.errors import ErrorType
from foren_searcher.datagen.verify.answer_anchored import split_answer
from foren_searcher.datagen.verify.anchored import COVERAGE_SYSTEM_PROMPTS, PAIR_SYSTEM_PROMPTS
from foren_searcher.datagen.verify.judges import CallableJudge
from foren_searcher.datagen.verify.plan_judged import PLAN_SYSTEM_PROMPT
from foren_searcher.datagen.verify.sufficiency import SUFFICIENCY_SYSTEM_PROMPT
from foren_searcher.datagen.verify.verdict import Verdict

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r1"
VERDICTS = ROOT / "experiments" / "sft_data" / "verdicts_r0_judged.jsonl"
CASES = ROOT / "experiments" / "sft_data" / "cases"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not VERDICTS.is_file(), reason="试点 run / 判定文件未随仓库入库")

_FAKE_MARKERS = ("37 人受伤", "凌晨 2 点 40 分", "4471", "Laurent Mercier", "路透社")


def _replay_from_base(base: list[Verdict]) -> dict[tuple[str, str], dict[str, Any]]:
    """用真实核验模型对本 run 报告的 (Finding, E) 判定做回放表。"""

    from foren_searcher.datagen.runlog import load_run

    log = load_run(RUN)
    try:
        statements = {str(f["finding_id"]): str(f["statement"]) for rec in log.store.workspace_report_records()
                      for f in rec.to_dict().get("findings") or []}
    finally:
        log.close()
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for v in base:
        if v.subject.get("run") != RUN.name:
            continue
        for j in v.judgments:
            st = statements.get(str(j.get("finding_id")))
            if st is None or j.get("verdict") in ("failed", "needs_pixels"):
                continue
            table[(st, str(j.get("evidence_id")))] = {
                "supported_parts": [{"part": p["part"], "quote": p["quote"]} for p in j.get("supported") or []],
                "contradicted_parts": [{"part": p["part"], "quote": p["quote"], "hint": p.get("hint")} for p in j.get("contradicted") or []],
                "source_reports_claim": bool(j.get("source_reports_claim")), "statement_asserts_fact": bool(j.get("statement_asserts_fact", True)),
                "scope_overreach": bool(j.get("scope_overreach")), "notes": "replay"}
    return table


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


class FakeJudge:
    """规则版"核验模型"：文本来源里能找到待核句前 10 个字 → supported（引语取自来源）；含编造标记 → 覆盖层报无支持；
    plan：任务文字含"来源/语境/视频" → REQ2/REQ3，含"恐怖袭击/火情原因" → REQ1；"耶拿桥"当既定事实写 → unverified 前提。"""

    def __init__(self, replay: dict[tuple[str, str], dict[str, Any]] | None = None) -> None:
        self.calls: list[str] = []
        self.replay = replay or {}          # (Finding 原句, E) → 真实核验模型当时的判定（回放；配错的 E 就查不到 → not_found）
        self.known_statements = {k[0] for k in self.replay}

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append(system_prompt[:12])
        if system_prompt in PAIR_SYSTEM_PROMPTS:
            statement, source = str(payload["statement"]), payload["source"]
            content = str(source.get("content") or "")
            if images:
                return {"supported_parts": [{"part": statement[:20], "quote": "画面：铁塔附近有烟雾与火光"}], "contradicted_parts": [],
                        "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": "img"}
            if (statement, source["evidence_id"]) in self.replay:
                return self.replay[(statement, source["evidence_id"])]
            if statement in self.known_statements:
                return {"supported_parts": [], "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                        "scope_overreach": False, "notes": "replay: 该 E 不曾支持这句"}
            # 数字改坏（calibrate_recall.alter_number 的 +13）：句中数字来源里没有、但它减 13 有 → 反驳
            for num in re.findall(r"(?<![\dEF])\d{2,4}(?!\d)", statement):
                if num not in content and str(int(num) - 13) in content:
                    return {"supported_parts": [], "contradicted_parts": [{"part": num, "quote": content[:40], "hint": "sloppy_numeric"}],
                            "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": "num"}
            if content.strip():
                # 文本来源一律算支持句首（引语逐字取自来源，能过程序回查）；编造标记交给覆盖层去报
                return {"supported_parts": [{"part": statement[:30], "quote": content[:30]}], "contradicted_parts": [],
                        "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": "text"}
            return {"supported_parts": [], "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                    "scope_overreach": False, "notes": "none"}
        if system_prompt in COVERAGE_SYSTEM_PROMPTS:
            statement = str(payload.get("statement") or "")
            bad = [m for m in _FAKE_MARKERS if m in statement]
            if not payload.get("supported_parts"):
                bad = [statement[:40]]                      # 没有任何支持片段 → 整句无据
            return {"unsupported_parts": bad, "grounded": not bad, "notes": "fake"}
        if system_prompt == SUFFICIENCY_SYSTEM_PROMPT:
            report = payload["report"]
            covered = [{"key": r["key"], "covered": True, "by": "F1", "note": ""} for r in payload.get("required_evidence") or []]
            return {"requirements": covered, "answerable_from_report": bool(report["findings"]),
                    "status_should_be": "COMPLETED" if report["findings"] else "INCOMPLETE", "missing": []}
        if system_prompt == PLAN_SYSTEM_PROMPT:
            covered = set(payload["covered_before_this_round"])
            tasks = []
            for t in payload["tasks"]:
                text = t["task"]
                targets = []
                if re.search(r"来源|语境|视频", text):
                    targets += ["REQ2", "REQ3"]
                if re.search(r"恐怖袭击|火情|原因", text):
                    targets.append("REQ1")
                premises = []
                if "耶拿桥" in text and "核查" not in text[:4]:
                    premises.append({"phrase": "耶拿桥卡车起火", "status": "unverified", "why": "案卷未给"})
                tasks.append({"index": t["index"], "targets": sorted(set(targets)), "gap_driven": bool(set(targets) - covered), "premises": premises})
            return {"tasks": tasks, "notes": "fake plan"}
        raise AssertionError(f"未知提示 {system_prompt[:30]}")


# ------------------------------------------------------------------ 拆句

def test_split_answer_carries_citations_back_and_maps_findings():
    answer = "结论：该声明不实。警方通报卡车起火；随后确认是意外。（E24、E21）影像被误传。（E11）\n\n画面有烟雾（F1）。无引用的一句。"
    rows = split_answer(answer, {"F1": ["E3", "E7"]})
    assert [r.evidence_ids for r in rows] == [["E24", "E21"], ["E24", "E21"], ["E24", "E21"], ["E11"], ["E3", "E7"], []]
    assert [r.scope for r in rows] == ["carried_back", "carried_back", "own", "own", "own", "uncited"]
    assert rows[4].finding_ids == ["F1"] and rows[0].sentence_id == "A1" and "（" not in rows[2].text


def test_split_answer_accepts_square_and_lenticular_brackets():
    answer = "警方通报卡车起火。[E29]\n\n约23秒显示“.000”；约50秒显示“.018”。[E21、E17] 另一句【E5】。无引用的尾句。"
    rows = split_answer(answer)
    assert [r.evidence_ids for r in rows] == [["E29"], ["E21", "E17"], ["E21", "E17"], ["E5"], []]
    assert [r.scope for r in rows] == ["own", "carried_back", "own", "own", "uncited"]
    assert all("[" not in r.text and "【" not in r.text for r in rows)


def test_split_answer_merges_adjacent_citation_groups():
    rows = split_answer("A 的 1.36 秒出现划船者。[E3][E6] [E10] 画面有烟（F1）(E2)。\n句首引用之后。[E7]【E8】第二句。（E9）", {"F1": ["E4"]})
    assert rows[0].evidence_ids == ["E3", "E6", "E10"] and rows[0].scope == "own"
    assert rows[1].evidence_ids == ["E2", "E4"] and rows[1].finding_ids == ["F1"]
    # 句首的相邻引用组整体结算给上一句（同段），不会漏一半给下一句
    assert rows[2].evidence_ids == ["E7", "E8"] and rows[3].evidence_ids == ["E9"]
    assert [r.scope for r in rows].count("uncited") == 0


# ------------------------------------------------------------------ 真实 run：answer / plan 模型判定

@needs_run
def test_complete_units_on_real_run_replaces_proxy_and_judges_plan():
    from foren_searcher.datagen.verify.complete_units import complete_run

    base = [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    case = load_cases(CASES).get("true:4124903")
    judge = FakeJudge()
    out = asyncio.run(complete_run(RUN, case, base, judge, judge_name="fake", say=lambda _: None))
    by_unit = {v.unit: v for v in out}
    assert set(by_unit) == {"plan", "answer"}
    answer = by_unit["answer"]
    a = answer.extra["anchored"]
    assert a["method"] == "judge_sentences" and "anchored_sentences" in answer.layers_run and "proxy_anchored" not in answer.layers_run
    assert len(a["sentences"]) == 7 and a["uncited_sentences"] == []
    assert not any(e.extra.get("confidence") == "flag" for e in answer.errors)
    assert answer.judgments and all(j["finding_id"].startswith("A") for j in answer.judgments)
    assert answer.disposition == "positive" and answer.extra["recovered_from_tainted"] is True
    assert answer.judge_model == "fake" and not answer.extra["pending_judge"]
    plan = by_unit["plan"]
    g = plan.extra["gap"]
    assert "judge_plan" in plan.layers_run and g["judge_targets"]["T1"] and g["agreement"]["T1"]["hindsight"] == ["REQ1", "REQ2", "REQ3"]
    assert isinstance(plan.extra["checks"]["premises"], dict) and plan.extra["checks"]["premises"]["T1"] == []
    assert not any(e.type is ErrorType.PREMISE_SMUGGLING for e in plan.errors)
    assert plan.disposition == "positive"


def test_apply_plan_judgment_flags_not_gap_driven_and_premise_smuggling():
    from foren_searcher.datagen.cases import Case, Requirement
    from foren_searcher.datagen.verify.plan_judged import apply_plan_judgment

    case = Case(case_id="true:1", dataset="true", source_id="1", split="dev", task_kind="claim_check", task_text="t", family_key="true:1",
                requirements=(Requirement(id="REQ1", text="a"), Requirement(id="REQ2", text="b")))
    v = Verdict(subject={"run": "r", "request": "request_0009"}, unit="plan", disposition="positive",
                extra={"tasks": 2, "checks": {"premises": "pending_judge"}, "gap": {"covered_before": ["REQ1"], "task_targets": {"T1": ["REQ1"], "T2": []}}})
    raw = {"tasks": [{"index": 1, "targets": ["REQ1"], "gap_driven": True, "premises": []},
                     {"index": 2, "targets": ["REQ2"], "gap_driven": True,
                      "premises": [{"phrase": "该桥当晚已封锁", "status": "unverified", "why": "无来源"}]}], "notes": "n"}
    apply_plan_judgment(v, raw, case=case, judge_model="fake")
    types = sorted(e.type.value for e in v.errors)
    assert types == ["premise_smuggling", "task_not_gap_driven"]      # T1 目标已覆盖（judge 的 gap_driven 被 covered_before 否决）；T2 前提走私
    assert v.disposition == "negative" and v.extra["gap"]["agreement"]["T2"]["same"] is False
    apply_plan_judgment(v, None, case=case, judge_model="fake")
    assert v.pending and v.extra["checks"]["premises"] == "pending_judge" and v.disposition == "negative"


# ------------------------------------------------------------------ 召回率校准

@needs_run
def test_recall_calibration_injections_are_caught_by_fake_judge():
    from foren_searcher.datagen.verify.calibrate_recall import MUTATORS, run_calibration

    base = [Verdict.from_dict(l and json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = load_cases(CASES)
    kinds = list(MUTATORS)
    judge = FakeJudge(_replay_from_base(base))
    assert judge.replay
    summary = asyncio.run(run_calibration([RUN], cases, base, judge, judge_name="fake", kinds=kinds, per_kind=1, seed=3, say=lambda _: None))
    assert summary["n_injections"] >= 8 and set(summary["by_kind"]) == set(kinds)
    for kind in ("fabricate_id", "append_clause", "swap_citation", "alter_number"):
        row = summary["by_kind"][kind]
        assert row["n"] >= 1 and row["recall"] == 1.0, (kind, row)
    assert summary["by_expected_type"]["fabricated_evidence_id"]["hit"] >= 1
    # exec-only（无核验模型）只能抓程序层的注入
    summary0 = asyncio.run(run_calibration([RUN], cases, base, None, judge_name=None, kinds=["fabricate_id", "append_clause"], per_kind=1, seed=3,
                                           say=lambda _: None))
    assert summary0["by_kind"]["fabricate_id"]["recall"] == 1.0 and summary0["by_kind"]["append_clause"]["recall"] == 0.0


# ------------------------------------------------------------------ export --units 叠加

@needs_run
def test_export_overlays_judged_units(tmp_path: Path):
    from foren_searcher.datagen.export import main as export_main
    from foren_searcher.datagen.verify.complete_units import complete_run

    base = [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    case = load_cases(CASES).get("true:4124903")
    judged = asyncio.run(complete_run(RUN, case, base, FakeJudge(), judge_name="fake", say=lambda _: None))
    units_path = tmp_path / "units.jsonl"
    units_path.write_text("".join(json.dumps(v.to_dict(), ensure_ascii=False) + "\n" for v in judged), encoding="utf-8")
    out = tmp_path / "export"
    assert export_main(["--runs", str(RUN), "--cases", str(CASES), "--verdicts", str(VERDICTS), "--specs", str(ROOT / "specs"),
                        "--hi-spec", "hi@0.2.0", "--units", str(units_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["units_source"] == str(units_path)
    assert manifest["runtime_system_matches_spec"].get("hi:True", 0) == 20 and "hi:False" not in manifest["runtime_system_matches_spec"]
    gia = [json.loads(l) for l in (out / "gia_accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    metas = {m["metadata"]["unit"]: m["metadata"] for m in gia}
    assert metas["answer"]["anchored_method"] == "judge_sentences" and metas["answer"]["unsupported_sentences"] == []
    assert metas["plan"]["plan_judged"] is True and isinstance(metas["plan"]["premises"], dict)
    assert metas["answer"]["judge_model"] == manifest["judge_model"] if "judge_model" in manifest else True


# ------------------------------------------------------------------ R4：同句另一条 E 已支持的部分不算反驳（真实 answer 判定的第一个误报）

def test_recalibrate_r4_dismisses_contradiction_covered_by_sibling():
    from foren_searcher.datagen.errors import error
    from foren_searcher.datagen.verify.calibrate import recalibrate

    e = error(ErrorType.WRONG_CITATION, "E24 与该部分相反：Intervention des pompiers", span="已经扑灭", evidence_ids=["E24"], method="model",
              extra={"finding": "A3", "sentence": "A3"})
    v = Verdict(subject={"run": "r", "request": "request_0022"}, unit="answer", errors=[e], disposition="negative",
                layers_run=["exec", "anchored_sentences"],
                judgments=[{"finding_id": "A3", "evidence_id": "E24", "verdict": "contradicted", "supported": [], "contradicted": [{"part": "已经扑灭"}]},
                           {"finding_id": "A3", "evidence_id": "E21", "verdict": "supported",
                            "supported": [{"part": "已经扑灭", "quote": "Le feu est éteint", "verified": True}], "contradicted": []}],
                extra={"anchored": {"method": "judge_sentences", "contradicted_sentences": ["A3"], "unsupported_sentences": []}})
    n = recalibrate(v)
    assert n.disposition == "positive" and n.errors == [] and n.extra["anchored"]["contradicted_sentences"] == []
    assert n.extra["calibration"]["applied"][0]["rule"] == "R4_contradiction_covered_by_sibling"
    assert v.errors == [e]                                    # 原判不动
    # 没做逐句锚定（还是代理）的 answer 单元不碰
    proxy = Verdict(subject={}, unit="answer", errors=[e], extra={"anchored": {"method": "proxy_report_judgments"}})
    assert recalibrate(proxy) is proxy


# ------------------------------------------------------------------ plan：既有报告取 store 里本轮之前的全部工作区（09-25）

RUN2 = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r2"


@pytest.mark.skipif(not (RUN2 / "store.sqlite3").is_file(), reason="多轮 run 未随仓库入库")
def test_judge_plan_payload_carries_prior_round_reports():
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.verify.plan_judged import judge_plan

    case = load_cases(CASES).get("true:4124903")
    judge = FakeJudge()
    seen: list[dict[str, Any]] = []
    inner = judge.judge

    async def capture(**kw: Any) -> dict[str, Any]:
        seen.append(kw["payload"])
        return await inner(**kw)

    judge.judge = capture  # type: ignore[method-assign]
    log = load_run(RUN2)
    try:
        plans = sorted((d for d in log.decisions if d.kind == "plan"), key=lambda d: d.round_index or 0)
        assert [d.round_index for d in plans][:2] == [1, 2]
        r1 = [rec.to_dict() for rec in log.store.workspace_report_records() if rec.task_id.startswith("W1-")]
        r1_statements = {str(f["statement"])[:600] for rep in r1 for f in rep.get("findings") or []}
        for d in plans[:2]:
            assert asyncio.run(judge_plan(d, case, set(), judge, log)) is not None
    finally:
        log.close()
    first, second = seen
    assert first["existing_reports"] == []
    existing = second["existing_reports"]
    assert existing and all(r["workspace_id"].startswith("W1-") for r in existing)
    got = {f["statement"] for r in existing for f in r["findings"]}
    assert r1_statements and r1_statements <= got
    assert all(f["finding_id"] and isinstance(f["evidence_ids"], list) for r in existing for f in r["findings"])


def test_plan_reports_summary_reads_nested_view_report():
    from foren_searcher.datagen.verify.plan_judged import _reports_summary

    view = {"latest_reports": [{"task": "t", "run_status": "reported",
                                "report": {"task_id": "W1-1", "status": "INCOMPLETE",
                                           "findings": [{"finding_id": "F1", "statement": "A 1.160s/B 1.635s 对应", "evidence_ids": ["E16"]}]}}]}
    rows = _reports_summary(view)
    assert rows == [{"workspace_id": "W1-1", "status": "INCOMPLETE",
                     "findings": [{"finding_id": "F1", "statement": "A 1.160s/B 1.635s 对应", "evidence_ids": ["E16"]}]}]


# ------------------------------------------------------------------ 召回校准：相对同条件对照（09-25）

@needs_run
def test_recall_calibration_compares_against_same_condition_control():
    import dataclasses

    from foren_searcher.datagen.cases import RequiredEvidence
    from foren_searcher.datagen.verify.calibrate_recall import run_calibration

    base = [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    case = load_cases(CASES)["true:4124903"]
    # 一份谁也没引过的必需证据（supports 为空 → 每个工作区都在范围内）：原报告在 scoped 条件下本来就带 omitted_required_fact
    phantom = RequiredEvidence(key="phantom_notice", description="一份从未取得的官方通报", url_pattern="example.invalid/never/*")
    case = dataclasses.replace(case, required_evidence=(*case.required_evidence, phantom),
                               required_evidence_rule=(case.required_evidence_rule or "all") + "; all:phantom_notice")
    plain = load_cases(CASES)["true:4124903"]
    kinds = ["fabricate_id", "append_clause", "drop_required_finding"]

    def run(c: Any) -> dict[str, Any]:
        return asyncio.run(run_calibration([RUN], {c.case_id: c}, base, FakeJudge(_replay_from_base(base)), judge_name="fake", kinds=kinds,
                                           per_kind=1, seed=3, case_mode="scoped", say=lambda _: None))

    summary, reference = run(case), run(plain)
    assert summary["controls"] and all("omitted_required_fact" in c["control_error_types"] for c in summary["controls"].values())
    rows = summary["rows"]
    assert rows
    for r in rows:
        assert any(e["type"] == "omitted_required_fact" and "phantom_notice" in e["detail"] for e in r["errors"])   # 注入后仍在
        assert all(len(e["detail"]) <= 160 for e in r["errors"])
        if r["kind"] != "drop_required_finding":
            assert "omitted_required_fact" not in r["new_error_types"] and "omitted_required_fact" not in r["hits"]
    assert not any("omitted_required_fact" in f["types"] for f in summary["false_new_errors"])
    # 原报告就有的缺项被对照抵消：命中与新错与不加这份必需证据时完全一致
    assert [(r["detected"], r["hits"], r["new_error_types"]) for r in rows] == \
        [(r["detected"], r["hits"], r["new_error_types"]) for r in reference["rows"]]
    assert summary["by_kind"]["fabricate_id"]["recall"] == 1.0
