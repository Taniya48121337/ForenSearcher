"""plan / answer 单元的召回注入（calibrate_units_recall）：假核验模型在真实 run 上跑通；程序层能抓的类型在 judge=none 下命中；
结构性漏报（核验器根本没有对应检查的注入）也钉住，将来补上检查时这里会失败，提醒同步更新 docs/核验器量化工具.md。"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from foren_searcher.datagen.cases import load_cases
from foren_searcher.datagen.errors import ErrorType
from foren_searcher.datagen.verify.anchored import COVERAGE_SYSTEM_PROMPTS, PAIR_SYSTEM_PROMPTS
from foren_searcher.datagen.verify.answer_anchored import split_answer
from foren_searcher.datagen.verify.plan_judged import PLAN_SYSTEM_PROMPT
from foren_searcher.datagen.verify.scope import overlap
from foren_searcher.datagen.verify.verdict import Verdict

from test_datagen_judged_units import CASES, RUN, VERDICTS, FakeJudge, _replay_from_base, needs_run

CASE_ID = "true:4124903"


def _base() -> list[Verdict]:
    return [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]


class UnitsJudge(FakeJudge):
    """在 FakeJudge 上加两件事（只为验证流程，不代表真实核验模型的判断力）：
    - answer 句子：用原答案拆出的 (句, E) 作"真值"——句子原样且 E 本来就引着 → 支持；句子是原句加尾巴 → 原句部分支持、尾巴无据；
      其它（换了 E、句子被改写、被回溯到别的引用组）→ 不支持。
    - plan：targets 按任务文字与 REQ 文字的二元组重合（≥0.25）；任务文字里出现 FAKE_PREMISES → unverified 前提。"""

    def __init__(self, replay: dict[tuple[str, str], dict[str, Any]], truth: dict[str, set[str]]) -> None:
        super().__init__(replay)
        self.truth = truth

    def _match(self, statement: str) -> tuple[str | None, bool]:
        if statement in self.truth:
            return statement, True
        for t in self.truth:
            stem = t.rstrip("。；;.")
            if len(stem) >= 6 and statement.startswith(stem) and statement != t:
                return t, False
        return None, False

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        from foren_searcher.datagen.verify.calibrate_units_recall import FAKE_PREMISES

        if system_prompt in PAIR_SYSTEM_PROMPTS and str(payload["statement"]) not in self.known_statements:
            statement, source = str(payload["statement"]), payload["source"]
            t, _exact = self._match(statement)
            if t is not None and source["evidence_id"] in self.truth[t]:
                quote = "画面：铁塔附近有烟雾与火光" if images else str(source.get("content") or "")[:30]
                return {"supported_parts": [{"part": t.rstrip("。；;."), "quote": quote}], "contradicted_parts": [], "source_reports_claim": False,
                        "statement_asserts_fact": True, "scope_overreach": False, "notes": "truth"}
            return {"supported_parts": [], "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                    "scope_overreach": False, "notes": "truth: 不支持"}
        if system_prompt in COVERAGE_SYSTEM_PROMPTS and str(payload.get("statement")) not in self.known_statements:
            statement = str(payload.get("statement") or "")
            t, exact = self._match(statement)
            if not payload.get("supported_parts"):
                return {"unsupported_parts": [statement[:40]], "grounded": False, "notes": "fake"}
            if t is not None and not exact:
                return {"unsupported_parts": [statement[len(t.rstrip("。；;.")):].strip("；;。")], "grounded": False, "notes": "fake tail"}
            return {"unsupported_parts": [], "grounded": True, "notes": "fake"}
        if system_prompt == PLAN_SYSTEM_PROMPT:
            covered = set(payload["covered_before_this_round"])
            tasks = []
            for t in payload["tasks"]:
                text = t["task"]
                targets = sorted(r["id"] for r in payload["requirements"] if overlap(r["text"], text) >= 0.25)
                premises = [{"phrase": p, "status": "unverified", "why": "案卷未给"} for p in FAKE_PREMISES if p in text]
                tasks.append({"index": t["index"], "targets": targets, "gap_driven": bool(set(targets) - covered), "premises": premises})
            return {"tasks": tasks, "notes": "fake plan"}
        return await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)


def _truth(case: Any, base: list[Verdict]) -> dict[str, set[str]]:
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.verify.calibrate_units_recall import _finding_evidence, prepare_run

    log = load_run(RUN)
    try:
        ru = prepare_run(log, case, base)
        answer = next(d for d in log.decisions if d.kind == "answer")
        rows = split_answer(str(answer.output.get("answer") or ""), _finding_evidence(log, ru.useful))
    finally:
        log.close()
    return {r.text: set(r.evidence_ids) for r in rows}


def units_judge(case: Any, base: list[Verdict]) -> UnitsJudge:
    return UnitsJudge(_replay_from_base(base), _truth(case, base))


# ------------------------------------------------------------------ 切块

def test_answer_blocks_follow_split_answer_scope():
    from foren_searcher.datagen.verify.calibrate_units_recall import answer_blocks

    answer = "结论：不实。警方通报起火。（E24、E21）另一句。（E11）\n\n画面有烟（F1）。无引用的尾句。"
    blocks = answer_blocks(answer)
    assert [b.ids for b in blocks] == [["E24", "E21"], ["E11"], ["F1"]]
    assert blocks[0].text == "结论：不实。警方通报起火。" and blocks[1].text == "另一句。" and blocks[2].text == "画面有烟"
    assert [b.last_in_paragraph for b in blocks] == [False, True, True]
    # 与 split_answer 同一作用域：每块的句子正好是 split_answer 里挂同一组引用的句子
    rows = split_answer(answer, {"F1": ["E3"]})
    assert [r.evidence_ids for r in rows] == [["E24", "E21"], ["E24", "E21"], ["E11"], ["E3"], []]


# ------------------------------------------------------------------ 真实 run：假核验模型

@needs_run
def test_units_recall_with_fake_judge_every_mutator_injects():
    from foren_searcher.datagen.verify.calibrate_units_recall import ANSWER_MUTATORS, KINDS, PLAN_MUTATORS, run_units_calibration

    base = _base()
    cases = load_cases(CASES)
    judge = units_judge(cases[CASE_ID], base)
    summary = asyncio.run(run_units_calibration([RUN], cases, base, judge, judge_name="fake", kinds=KINDS, per_kind=1, seed=3,
                                                say=lambda _: None))
    assert len(ANSWER_MUTATORS) >= 4 and len(PLAN_MUTATORS) >= 3
    assert set(summary["by_kind"]) == set(KINDS)
    for kind in KINDS:
        assert summary["by_kind"][kind]["n"] >= 1, kind
    assert summary["by_unit"]["answer"]["n"] >= 7 and summary["by_unit"]["plan"]["n"] >= 6
    assert summary["controls"] and all(c["unit"] in ("answer", "plan") for c in summary["controls"].values())
    rows = {r["kind"]: r for r in summary["rows"]}
    caught = ("add_unsupported_clause", "swap_citation", "fabricate_citation", "flip_label", "negate_label",
              "smuggle_premise", "retarget_covered", "sibling_dependency", "fabricate_material")
    for kind in caught:
        assert summary["by_kind"][kind]["recall"] == 1.0, (kind, rows[kind])
    assert rows["smuggle_premise"]["hits"] == ["premise_smuggling"] and rows["retarget_covered"]["hits"] == ["task_not_gap_driven"]
    assert rows["retarget_covered"]["condition"]["round_index"] == 2
    # 检出 ≠ 拦下：plan 单元单条 major（前提走私 / 不对缺口 / 兄弟依赖）或 minor（illegal_action）不改处置，export 仍收进 accepted
    for kind in ("smuggle_premise", "retarget_covered", "sibling_dependency", "fabricate_material"):
        assert summary["by_kind"][kind]["rejected"] == 0, kind
    for kind in ("add_unsupported_clause", "fabricate_citation", "flip_label"):
        assert summary["by_kind"][kind]["rejected"] == 1, kind
    # 结构性漏报：核验器没有对应检查（见模块 docstring），假核验模型再"聪明"也抓不到
    for kind in ("drop_required_block", "duplicate_sibling", "sibling_dependency_paraphrase"):
        assert summary["by_kind"][kind]["recall"] == 0.0 and summary["by_kind"][kind]["inconclusive"] == 0, (kind, rows[kind])
    # 行格式与 calibrate_recall 对齐
    for key in ("kind", "expected", "detected", "hits", "new_error_types", "pending", "control_error_types", "errors", "run"):
        assert key in summary["rows"][0]


@needs_run
def test_units_recall_drop_citation_depends_on_citation_scope():
    """删段中那组引用：句子被回溯到下一组引用上 → 锚定层判无支持；删段末那组：句子变 uncited，按设计不报错。"""

    from foren_searcher.datagen.verify.calibrate_units_recall import run_units_calibration

    base = _base()
    cases = load_cases(CASES)
    judge = units_judge(cases[CASE_ID], base)
    summary = asyncio.run(run_units_calibration([RUN], cases, base, judge, judge_name="fake", kinds=["drop_citation"], per_kind=4, seed=1,
                                                say=lambda _: None))
    rows = summary["rows"]
    assert {r["becomes_uncited"] for r in rows} == {True, False}
    for r in rows:
        assert r["detected"] is (not r["becomes_uncited"]), r


@needs_run
def test_units_recall_exec_only_catches_program_checkable_types(tmp_path: Path):
    from foren_searcher.datagen.verify.calibrate_units_recall import KINDS, main, run_units_calibration

    base = _base()
    cases = load_cases(CASES)
    summary = asyncio.run(run_units_calibration([RUN], cases, base, None, judge_name=None, kinds=KINDS, per_kind=1, seed=3, say=lambda _: None))
    for kind in ("fabricate_citation", "flip_label", "sibling_dependency", "fabricate_material"):
        assert summary["by_kind"][kind]["recall"] == 1.0, kind
    for kind in ("add_unsupported_clause", "swap_citation", "negate_label", "smuggle_premise", "retarget_covered", "sibling_dependency_paraphrase"):
        row = summary["by_kind"][kind]
        assert row["recall"] == 0.0 and row["inconclusive"] == row["n"], kind      # 模型层没跑：记 inconclusive，不当漏报
    assert summary["layers"] == "exec+hindsight only"
    hit = {r["kind"]: r["hits"] for r in summary["rows"] if r["detected"]}
    assert hit["fabricate_citation"] == ["fabricated_evidence_id"] and hit["flip_label"] == ["omitted_required_fact"]
    assert hit["sibling_dependency"] == ["dependent_sibling_tasks"] and hit["fabricate_material"] == ["illegal_action"]
    # CLI
    out = tmp_path / "units_recall.json"
    assert main(["--runs", str(RUN), "--cases", str(CASES), "--verdicts", str(VERDICTS), "--judge", "none", "--per-kind", "1", "--seed", "3",
                 "--kinds", "flip_label", "sibling_dependency", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_injections"] == 2 and data["recall_overall"] == 1.0 and set(data["by_kind"]) == {"flip_label", "sibling_dependency"}


def test_unit_error_key_ignores_shifted_sentence_ids():
    from foren_searcher.datagen.errors import error
    from foren_searcher.datagen.verify.calibrate_units_recall import unit_error_key, unit_new_errors

    a = error(ErrorType.UNSUPPORTED_ASSERTION, "x", span="现场 37 人受伤", evidence_ids=["E3"], method="model", extra={"finding": "A3", "sentence": "A3"})
    b = error(ErrorType.UNSUPPORTED_ASSERTION, "x", span="现场 37 人受伤", evidence_ids=["E3"], method="model", extra={"finding": "A2", "sentence": "A2"})
    assert unit_error_key(a) == unit_error_key(b)
    control = Verdict(subject={}, unit="answer", errors=[a])
    assert unit_new_errors(Verdict(subject={}, unit="answer", errors=[b]), control) == []
    flag = error(ErrorType.UNSUPPORTED_ASSERTION, "flag", method="exec", extra={"confidence": "flag"})
    assert unit_new_errors(Verdict(subject={}, unit="answer", errors=[flag]), control) == []
    assert re.match(r"^\(", str(unit_error_key(flag)))
