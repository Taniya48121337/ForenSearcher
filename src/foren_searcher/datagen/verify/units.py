"""step / plan / answer 三个核验单元（v2 §2.2–2.5）。report 单元仍由 verify_report() 三层负责。

大白话：以前只核 HI 交上来的报告；这里补上"HI 每一步选得对不对"（step）、"GIA 派得对不对"（plan）、
"GIA 最终答案的证据链对不对"（answer）。三个单元都先做**纯程序**部分；需要核验模型的部分（步级抽样评判、
前提走私抽取、答案逐句锚定）留了接口，没有 judge 时记为 pending / proxy，并在 verdict.extra 里写明方法。

归因规则（§2.5）：HI 的错只相对它收到的任务和可见证据判；GIA 的错只相对它收到的报告判。GIA 引用了被污染的
Finding 不算 GIA 的错，但该行不进 GIA 正例（tainted_input）；GIA 没继承 HI 的错（比如 HI 报 COMPLETED 而答案
如实写"未核实"）记 recovered_from_tainted。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity, error
from .exec_checks import dates_in, fact_check_errors
from .hindsight import UsefulEvidence, build_useful_evidence, clue_gain, produced_evidence
from .lang import HEDGE, LABEL_NEGATORS_ZH, SIBLING_DEP, content_words, is_cjk
from .lang import grams as _lang_grams
from .lang import label_hit as _lang_label_hit
from .verdict import Verdict, decide

if TYPE_CHECKING:  # runlog 引用 verify.context，这里只作类型引用，避免循环导入
    from ..runlog import Decision, RunLog

_E = re.compile(r"\bE\d+\b")
_F = re.compile(r"\bF\d+\b")
_SIBLING_DEP = SIBLING_DEP       # 10-07：中英双语（lang.py）
_HEDGE = HEDGE
_EXEC_OK = {"executed", "report_committed", "gia_decision_accepted"}


def _err(kind: ErrorType, detail: str, **kw: Any) -> ErrorRecord:
    return error(kind, detail, method="exec", **kw)


def _same_request(a: Decision, b: Decision) -> bool:
    return a.output.get("action") == b.output.get("action") and a.output.get("arguments") == b.output.get("arguments")


def _observation_failed(decision: Decision) -> bool:
    """整步失败才算环境错误：工具报 error / failed，或搜索一条可用页面都没打开。个别候选打不开是常态，不算。"""

    obs = decision.observation or {}
    details = obs.get("details") or {}
    if details.get("error") or details.get("stop_reason") == "error" or details.get("status") == "failed":
        return True
    if obs.get("action") == "SEARCH":
        results = details.get("results") or []
        if results and not produced_evidence(decision):
            return True
    return any("ACCESS_RESTRICTED" in str(i) for i in obs.get("items") or [])


# --------------------------------------------------------------------------- step

def check_step(decision: Decision, earlier: list[Decision], useful: UsefulEvidence, first_seen: dict[str, tuple[str, int]],
               *, subject: dict[str, Any]) -> Verdict:
    errors: list[ErrorRecord] = []
    env: list[dict[str, Any]] = []
    args = decision.output.get("arguments") or {}
    action = decision.output.get("action")
    # 合法性：Harness 是否接受
    if decision.status not in _EXEC_OK:
        errors.append(_err(ErrorType.ILLEGAL_ACTION, f"Harness 未执行该动作（status={decision.status}）"))
    # 目标可见性
    targets: list[str] = []
    if action == "INSPECT":
        targets = [t for t in args.get("targets") or [] if isinstance(t, str)]
    elif action == "READ_RECORD" and isinstance(args.get("record_id"), str):
        targets = [args["record_id"]]
    elif action == "SEARCH" and isinstance(args.get("query"), dict) and isinstance(args["query"].get("evidence_id"), str):
        targets = [args["query"]["evidence_id"]]
    invisible = [t for t in targets if t.startswith("E") and t not in decision.visible_ids]
    if invisible:
        errors.append(_err(ErrorType.ILLEGAL_ACTION, f"动作目标 {', '.join(invisible)} 在当时的桌面上不可见", evidence_ids=invisible))
    # 重复 / 无视反馈
    duplicate_of = None
    for prev in reversed(earlier):
        if prev.kind == "step" and _same_request(prev, decision):
            duplicate_of = prev
            break
    if duplicate_of is not None:
        new_ids = [e for e in produced_evidence(decision) if first_seen.get(e) == (decision.actor_id, decision.turn)]
        if earlier and _same_request(earlier[-1], decision) and _observation_failed(earlier[-1]):
            errors.append(_err(ErrorType.IGNORED_FEEDBACK, f"上一步（turn {earlier[-1].turn}）工具失败后原样重发", extra={"duplicate_of_turn": earlier[-1].turn}))
        elif not new_ids:
            errors.append(_err(ErrorType.REDUNDANT_ACTION, f"与 turn {duplicate_of.turn} 的动作参数完全相同且未产生新 E", extra={"duplicate_of_turn": duplicate_of.turn}))
    # 环境错误
    if _observation_failed(decision):
        obs = decision.observation or {}
        detail = (obs.get("details") or {}).get("error") or (obs.get("items") or ["failed"])[0]
        env.append({"type": ErrorType.ENV_TOOL_FAILURE.value, "detail": str(detail)[:200]})
    gain = clue_gain(decision, useful, first_seen)
    verdict = Verdict(subject={**subject, "workspace_id": decision.actor_id, "turn": decision.turn, "request": decision.request_id},
                      unit="step", errors=errors, layers_run=["exec", "hindsight"])
    verdict.disposition = decide(errors, [])
    verdict.extra = {"action": action, "checks": {"legal": decision.status in _EXEC_OK, "targets": targets, "invisible": invisible,
                                                  "duplicate_of_turn": duplicate_of.turn if duplicate_of else None},
                     "env_errors": env, "clue_gain": gain, "sampled_judge": None}
    return verdict


# --------------------------------------------------------------------------- plan

def check_plan(decision: Decision, log: RunLog, case: Case | None, useful: UsefulEvidence, *,
               covered_before: set[str], subject: dict[str, Any]) -> Verdict:
    errors: list[ErrorRecord] = []
    raw_tasks = decision.output.get("tasks")
    parsable = isinstance(raw_tasks, list) and all(isinstance(t, dict) and isinstance(t.get("task"), str) for t in raw_tasks or [])
    tasks: list[dict[str, Any]] = [dict(t) for t in raw_tasks] if parsable and isinstance(raw_tasks, list) else []
    if not parsable:
        errors.append(_err(ErrorType.ILLEGAL_ACTION, "INVESTIGATE 的 tasks 不可解析"))
    view = decision.view or {}
    known = set(_E.findall(json.dumps(view, ensure_ascii=False))) | {m.get("reference_id") for m in view.get("original_materials") or [] if isinstance(m, dict)}
    missing_materials: list[str] = []
    dependent: list[int] = []
    for i, task in enumerate(tasks):
        for m in task.get("materials") or []:
            if isinstance(m, str) and m not in known:
                missing_materials.append(m)
        if _SIBLING_DEP.search(str(task.get("task"))):
            dependent.append(i)
    if missing_materials:
        errors.append(_err(ErrorType.ILLEGAL_ACTION, f"任务引用了视图里不存在的材料 {', '.join(missing_materials)}"))
    if dependent:
        errors.append(_err(ErrorType.DEPENDENT_SIBLING_TASKS, f"任务 {dependent} 的文字依赖同轮兄弟任务的结果"))
    # hindsight：这一轮各工作区（W{r}-{i}）经核验的 Finding 覆盖了哪些要求
    round_index = int(decision.round_index or 1)
    task_targets: dict[str, list[str]] = {}
    covered_after: set[str] = set(covered_before)
    by_workspace = useful.coverage_by_workspace(case) if case else {}
    overlap: dict[str, list[str]] = {}
    for i, _task in enumerate(tasks):
        ws = f"W{round_index}-{i + 1}"
        reqs = sorted(by_workspace.get(ws, set()))
        task_targets[f"T{i + 1}"] = reqs
        covered_after.update(reqs)
        for r in reqs:
            overlap.setdefault(r, []).append(f"T{i + 1}")
    overlapping = {r: ts for r, ts in overlap.items() if len(ts) > 1}
    if overlapping:
        errors.append(_err(ErrorType.OVERLAPPING_SIBLING_TASKS, "同轮任务的工作区用同一批必需证据覆盖了同一要求：" +
                           "; ".join(f"{r}←{'+'.join(ts)}" for r, ts in overlapping.items()), extra={"overlap": overlapping}))
    must = set(case.must_cite_requirements) if case else set()
    if case and must and must <= covered_before:
        errors.append(_err(ErrorType.UNNECESSARY_ROUND, "must_cite 要求在本轮前已全部被核实的 Finding 覆盖，仍 INVESTIGATE"))
    verdict = Verdict(subject={**subject, "round": round_index, "request": decision.request_id}, unit="plan", errors=errors,
                      layers_run=["exec", "hindsight"])
    verdict.disposition = decide(errors, [])
    verdict.extra = {"tasks": len(tasks), "checks": {"parsable": parsable, "missing_materials": missing_materials, "dependent": dependent,
                                                   "premises": "pending_judge"},
                     "gap": {"method": "hindsight_required_evidence", "covered_before": sorted(covered_before), "task_targets": task_targets,
                             "must_cite_uncovered_after_round": sorted(must - covered_after), "overlap": overlapping},
                     "round_yield": sorted(covered_after - covered_before)}
    return verdict


# --------------------------------------------------------------------------- answer

def _overlap_ratio(a: str, b: str) -> float:
    """两段短文本的字符二元组重合率（中文友好）。"""

    # 10-07：纯英文用实词（英文字母二元组几乎处处重合，会把无关答案判成复述了污染片段）
    english = not (is_cjk(a) or is_cjk(b))
    ga, gb = (content_words(a), content_words(b)) if english else (_lang_grams(a), _lang_grams(b))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


# verifier@0.3.3：标签子串匹配排除否定 / 限定前缀。"属实"不能算进"不属实""部分属实"里，"不实"不能算进"并非不实"里。
# 只挡明显的假阳性；等价说法（"现有证据支持该声明"）和反向结论由核验模型的语义判定处理（label_judged.py）。
_LABEL_NEGATORS = LABEL_NEGATORS_ZH
label_hit = _lang_label_hit     # 10-07：英文另看前一个词（not / partly / mostly …）与词边界（untrue 不算 true），见 lang.label_hit


def check_answer(decision: Decision, log: RunLog, case: Case | None, useful: UsefulEvidence, report_verdicts: list[Verdict], *,
                 subject: dict[str, Any]) -> Verdict:
    errors: list[ErrorRecord] = []
    answer = str(decision.output.get("answer") or "")
    cited_e = sorted(set(_E.findall(answer)), key=lambda x: int(x[1:]))
    cited_f = sorted(set(_F.findall(answer)), key=lambda x: int(x[1:]))
    if not answer.strip():
        errors.append(_err(ErrorType.EMPTY_REPORT, "FINISH 没有答案"))
    if not cited_e and not cited_f:
        errors.append(_err(ErrorType.UNSUPPORTED_ASSERTION, "答案没有引用任何 E / F", span=answer[:80]))
    invisible = [e for e in cited_e if e not in decision.visible_ids and not log.lookup.has(e)]
    not_in_view = [e for e in cited_e if e not in decision.visible_ids and log.lookup.has(e)]
    if invisible:
        errors.append(_err(ErrorType.FABRICATED_EVIDENCE_ID, f"答案引用了不存在的 {', '.join(invisible)}", evidence_ids=invisible))
    if not_in_view:
        errors.append(_err(ErrorType.FABRICATED_EVIDENCE_ID, f"答案引用的 {', '.join(not_in_view)} 不在 GIA 本轮视图里", evidence_ids=not_in_view,
                           extra={"reason": "not_visible"}))
    # 案卷断言（answer 单元：label / date / entity / numeric；答案是整案结论，不分 check.requirement 全查）
    label_match = None
    if case:
        # 答案所引 E 的文本 = 直接引的 E + 所引干净 F 的 E
        pool_ids = [e for e in cited_e if log.lookup.has(e)]
        for fid in cited_f:
            row = useful.clean_findings.get(fid)
            if row:
                pool_ids.extend(e for e in row["evidence_ids"] if e not in pool_ids and log.lookup.has(e))
        texts = [log.lookup.get(e) for e in pool_ids]
        answer_pool = "\n".join(t.content for t in texts if t.is_text)
        for check in case.checks:
            if check.kind in ("entity_in_evidence", "numeric_in_evidence"):
                errors.extend(fact_check_errors(check, answer, answer_pool))
                continue
            if check.kind == "label_expected":
                # params.aliases（可选）：答案里可接受的等价措辞（数据集标签 True → 答案可能写"属实""支持该声明"）
                forms = [str(check.params["label"]), *(str(a) for a in check.params.get("aliases") or [])]
                label_match = any(label_hit(f, answer) for f in forms if f)
                if not label_match:
                    errors.append(_err(ErrorType.OMITTED_REQUIRED_FACT, f"答案没有出现应有的标签 {check.params['label']!r}（含等价措辞）",
                                       extra={"check": "label_expected", "forms": forms}))
            elif check.kind == "date_in_range":
                found = dates_in(answer)
                lo, hi = check.params.get("min"), check.params.get("max")
                bad = [d for d in found if (lo and d < lo) or (hi and d > hi)]
                if found and bad and len(bad) == len(found):
                    errors.append(_err(ErrorType.UNSUPPORTED_ASSERTION, f"答案日期 {', '.join(bad)} 不在案卷范围内", extra={"check": "date_in_range"}))
    # 污染输入：答案是否复述了被污染 Finding 的出错片段
    tainted_hits: list[dict[str, Any]] = []
    for verdict in report_verdicts:
        for e in verdict.errors:
            fid = e.extra.get("finding") if isinstance(e.extra, dict) else None
            if fid in verdict.tainted_findings and isinstance(e.span, str) and e.span.strip():
                ratio = _overlap_ratio(e.span, answer)
                if ratio >= 0.6 or (len(e.span) >= 8 and e.span in answer):
                    tainted_hits.append({"finding": fid, "error": e.type.value, "span": e.span[:60], "overlap": round(ratio, 2)})
    tainted_input = bool(tainted_hits)
    # 继承关系：答案引用的 E 各被哪些 Finding 引用、干净还是污染
    inherits: dict[str, dict[str, Any]] = {}
    for fid, row in useful.clean_findings.items():
        if set(row["evidence_ids"]) & set(cited_e):
            inherits[fid] = {"status": "clean", "workspace": row["workspace"]}
    for fid, kinds in useful.tainted_findings.items():
        inherits[fid] = {"status": "tainted", "errors": kinds}
    # 锚定（代理）：没有 judge 时，用 report 单元已核实的 (Finding, E) 对做代理——答案引用的每条 E 至少有一个 supported 片段
    supported_e: set[str] = set()
    for verdict in report_verdicts:
        for j in verdict.judgments:
            if j.get("verdict") == "supported" and j.get("finding_id") not in verdict.tainted_findings and isinstance(j.get("evidence_id"), str):
                supported_e.add(j["evidence_id"])
    unsupported_cites = [e for e in cited_e if e not in supported_e]
    anchored = {"method": "proxy_report_judgments", "cited": cited_e, "supported_via_report": sorted(set(cited_e) & supported_e),
                "unsupported_cites": unsupported_cites, "pending_judge": True}
    if unsupported_cites:
        errors.append(error(ErrorType.UNSUPPORTED_ASSERTION, f"答案引用的 {', '.join(unsupported_cites)} 在报告级核验里没有任何 supported 片段（代理判定）",
                            method="exec", severity=Severity.MAJOR, evidence_ids=unsupported_cites, extra={"confidence": "flag"}))
    # 充分性：must_cite 要求是否都被干净 Finding 覆盖
    covered: set[str] = set()
    if case:
        for fid in useful.clean_findings:
            covered |= useful.requirements_covered_by_finding(case, fid)
    must = set(case.must_cite_requirements) if case else set()
    uncovered = sorted(must - covered)
    if uncovered:
        errors.append(_err(ErrorType.PREMATURE_FINISH, f"must_cite 要求 {', '.join(uncovered)} 没有任何核实的 Finding 覆盖却 FINISH"))
    # 纠错：某工作区被判应为 INCOMPLETE（报了 COMPLETED），而答案对相应缺口写了"未核实"
    recovered = False
    overclaimed_ws = [v.subject.get("workspace_id") for v in report_verdicts
                      if any(e.type is ErrorType.COMPLETED_WITHOUT_FINISHING for e in v.errors)]
    if overclaimed_ws and _HEDGE.search(answer):
        recovered = True
    pending = [{"reason": "anchored_pending_judge", "note": "答案逐句锚定需核验模型；当前为报告级代理"}]
    verdict = Verdict(subject={**subject, "round": decision.round_index, "request": decision.request_id}, unit="answer", errors=errors,
                      layers_run=["exec", "proxy_anchored", "sufficiency_hindsight"], pending=[])
    firm = [e for e in errors if e.extra.get("confidence") != "flag"]
    verdict.disposition = "negative" if any(e.severity is Severity.CRITICAL for e in firm) or sum(1 for e in firm if e.severity is Severity.MAJOR) >= 2 \
        else ("uncertain" if any(e.extra.get("confidence") == "flag" for e in errors) else "positive")
    verdict.extra = {"tainted_input": tainted_input, "tainted_hits": tainted_hits, "recovered_from_tainted": recovered,
                     "overclaimed_workspaces": overclaimed_ws, "inherits_from": inherits, "anchored": anchored,
                     "exec": {"cited_e": cited_e, "cited_f": cited_f, "label_match": label_match},
                     "sufficiency": {"must_cite": sorted(must), "covered": sorted(covered), "uncovered": uncovered, "premature_finish": bool(uncovered)},
                     "pending_judge": pending}
    return verdict


# --------------------------------------------------------------------------- 一个 run 全部单元

def verify_units(log: RunLog, case: Case | None, report_verdicts: list[Verdict]) -> tuple[list[Verdict], UsefulEvidence]:
    """对一个 run 出全部 step / plan / answer 单元的判定。report_verdicts 是（校准后的）三层判定。"""

    useful = build_useful_evidence(log, case, report_verdicts)
    first_seen = log.evidence_first_seen()
    base = {"run": log.name, "case_id": log.case_id, "teacher": log.teacher_model}
    out: list[Verdict] = []
    covered_so_far: set[str] = set()
    by_workspace = useful.coverage_by_workspace(case) if case else {}
    for decision in log.decisions:
        if decision.role == "HI" and decision.kind == "step":
            earlier = [d for d in log.steps(decision.actor_id) if d.turn < decision.turn]
            out.append(check_step(decision, earlier, useful, first_seen, subject=base))
        elif decision.role == "GIA" and decision.kind == "plan":
            v = check_plan(decision, log, case, useful, covered_before=set(covered_so_far), subject=base)
            out.append(v)
            for ws in log.round_workspaces(int(decision.round_index or 1)):
                covered_so_far |= by_workspace.get(ws, set())
        elif decision.role == "GIA" and decision.kind == "answer":
            out.append(check_answer(decision, log, case, useful, report_verdicts, subject=base))
    return out, useful


__all__ = ["check_answer", "check_plan", "check_step", "verify_units"]
