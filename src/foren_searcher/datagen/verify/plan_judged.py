"""plan 单元的模型评判（接核验模型后补 units.check_plan 的 hindsight 缺口）。

大白话：check_plan 里"任务对准哪项要求"是用该任务工作区最终核实的 Finding 倒推的（hindsight），
派得对但 HI 没办成会被算成"没对准"；"前提走私"（任务文字把没核实的说法当既定事实写给 HI）
更是只有读任务文字才能判。这里让核验模型**只读任务文字 + 案卷要求 + 本轮前已覆盖的要求 +
本轮之前全部工作区的报告**，回答三件事：每个任务对准哪些 REQ；是否对准了本轮前仍未覆盖的要求
（gap_driven）；任务文字里有哪些前提，各自是案卷给的、此前核实的还是没根据的。

写回 plan 单元 Verdict：
- gap.judge_targets / gap.agreement（与 hindsight 对照，不互相覆盖）
- 全部目标都在 covered_before 内 → task_not_gap_driven（major）
- 有 unverified 前提且写成既定事实 → premise_smuggling（major，传播）
- checks.premises 从 pending_judge 变成判定结果

既有报告（09-25）：先前只从 GIA 视图 `latest_reports` 取 findings，但视图里每条是 {"task", "report": {..., "findings"}}，
findings 嵌在 report 下，读顶层恒为空 → 核验模型看到"既有报告 findings 为空"，第 2 轮起把上一轮已核实的事实全判成
premise_smuggling。现在改从 store 取本轮之前（W{round}-{i} 的 round < 当前轮）所有工作区的报告全文，视图摘要只补 store 里没有的工作区。
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, error
from .anchored import JsonJudge, _call, _obj
from .verdict import Verdict, decide

if TYPE_CHECKING:
    from ..runlog import Decision, RunLog

PLAN_SYSTEM_PROMPT = """你是调查规划审核员。给你：案卷里的调查要求清单（REQ）、本轮之前已被核实证据覆盖的要求、
本轮之前各调查员已交回的既有报告（existing_reports，每条 Finding 带 statement 与所引证据），以及总指挥（GIA）
这一轮派给各调查员的任务文字。只读这些，不用外部知识。
对每个任务判断：
1. targets：这条任务文字对准哪些 REQ（按任务要查的内容判，不按调查员后来查到了什么判）。可以为空。
2. gap_driven：它对准的 REQ 里是否至少有一项在"本轮前已覆盖"之外；如果 targets 全在已覆盖集合里、或为空，填 false。
3. premises：任务文字里当作既定事实写出来的说法（时间、地点、人物、事件性质、来源归属等）。每条给出原话片段和来源判断：
   given_in_case = 案卷/原始声明本身就这么说（转述待核的声明不算走私）；verified_earlier = 既有报告（existing_reports）里任何一条 Finding 写明了这个内容（不论该报告状态是 COMPLETED 还是 INCOMPLETE；
   那些 Finding 本身是否站得住由报告核验另行负责，这里不重判，只要 Finding 里写了就算）；
   unverified = 案卷没给、既有报告的 Finding 里也没有，却被写成事实。只有 unverified 才算问题；"核查声明 X 是否成立"这种把 X 当待核对象的写法不是前提。
输出 JSON（字符串里的双引号用“”）：
{"tasks":[{"index":1,"targets":["REQ1"],"gap_driven":true,
  "premises":[{"phrase":"任务原话片段","status":"given_in_case|verified_earlier|unverified","why":"一句话"}]}],
 "notes":"一句话"}"""

_PREMISE = _obj({"phrase": {"type": "string"}, "status": {"type": "string", "enum": ["given_in_case", "verified_earlier", "unverified"]},
                 "why": {"type": "string"}})
_TASK = _obj({"index": {"type": "integer"}, "targets": {"type": "array", "items": {"type": "string"}},
              "gap_driven": {"type": "boolean"}, "premises": {"type": "array", "items": _PREMISE}})
PLAN_SCHEMA = _obj({"tasks": {"type": "array", "items": _TASK}, "notes": {"type": "string"}})


_WORKSPACE_ROUND = re.compile(r"^W(\d+)-\d+$")
_STATEMENT_LIMIT = 600


def _finding_rows(findings: Any, limit: int) -> list[dict[str, Any]]:
    return [{"finding_id": f.get("finding_id"), "statement": str(f.get("statement") or "")[:_STATEMENT_LIMIT],
             "evidence_ids": f.get("evidence_ids")} for f in (findings or []) if isinstance(f, dict)][:limit]


def _reports_summary(view: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """GIA 视图里的 latest_reports：每条是 {"task", "run_status", "report": {task_id, status, findings…}}（旧格式 findings 在顶层也认）。"""

    out: list[dict[str, Any]] = []
    for rep in view.get("latest_reports") or []:
        if not isinstance(rep, dict):
            continue
        nested = rep.get("report")
        inner: dict[str, Any] = nested if isinstance(nested, dict) else rep
        out.append({"workspace_id": inner.get("task_id") or inner.get("workspace_id") or rep.get("workspace_id") or rep.get("task_id"),
                    "status": inner.get("status") or rep.get("status"),
                    "findings": _finding_rows(inner.get("findings") or rep.get("findings"), limit)})
    return out


def _round_of(workspace_id: str) -> int | None:
    m = _WORKSPACE_ROUND.match(workspace_id)
    return int(m.group(1)) if m else None


def prior_reports(log: RunLog, round_index: int | None, limit: int = 20) -> list[dict[str, Any]]:
    """store 里本轮之前（W{round}-{i} 的 round < round_index）全部工作区的报告全文。round_index 为空时返回空。"""

    if not round_index:
        return []
    out: list[dict[str, Any]] = []
    for rec in log.store.workspace_report_records():
        r = _round_of(rec.task_id)
        if r is None or r >= round_index:
            continue
        report = rec.to_dict()
        try:
            task = str(log.store.workspace_task_record(rec.task_id).get("task") or "")
        except KeyError:
            task = ""
        out.append({"workspace_id": rec.task_id, "round": r, "task": task[:300], "status": report.get("status"),
                    "findings": _finding_rows(report.get("findings"), limit)})
    return out


def plan_payload(decision: Decision, case: Case | None, covered_before: set[str], log: RunLog | None = None) -> dict[str, Any]:
    raw_tasks = decision.output.get("tasks")
    tasks: list[Any] = list(raw_tasks) if isinstance(raw_tasks, list) else []
    view = decision.view or {}
    view_task = view.get("task")
    case_task = view_task if isinstance(view_task, str) and view_task.strip() else (case.task_text if case else "")
    existing = prior_reports(log, decision.round_index) if log is not None else []
    seen = {str(r["workspace_id"]) for r in existing}
    # 视图摘要只作补充：store 里没有的工作区（或没给 log 时）才用
    existing += [r for r in _reports_summary(view) if str(r.get("workspace_id")) not in seen]
    return {
        "case_task": case_task,
        "requirements": [{"id": r.id, "text": r.text, "must_cite": r.must_cite} for r in (case.requirements if case else ())],
        "covered_before_this_round": sorted(covered_before),
        "existing_reports": existing,
        "round_index": decision.round_index,
        "tasks": [{"index": i + 1, "task": str(t.get("task") or ""), "materials": t.get("materials") or []}
                  for i, t in enumerate(tasks) if isinstance(t, dict)],
    }


async def judge_plan(decision: Decision, case: Case | None, covered_before: set[str], judge: JsonJudge,
                     log: RunLog | None = None) -> dict[str, Any] | None:
    """log：给了就从 store 取本轮之前的报告全文作 existing_reports（推荐）；不给只用视图摘要。"""

    payload = plan_payload(decision, case, covered_before, log)
    if not payload["tasks"]:
        return {"tasks": [], "notes": "no tasks"}
    raw = await _call(judge, PLAN_SYSTEM_PROMPT, payload, [], PLAN_SCHEMA)
    if raw is None or not isinstance(raw.get("tasks"), list):
        return None
    return raw


def apply_plan_judgment(verdict: Verdict, raw: dict[str, Any] | None, *, case: Case | None, judge_model: str | None) -> Verdict:
    """把模型评判写回 plan 单元（原地修改并返回）。raw 为 None 记 pending。"""

    if verdict.unit != "plan":
        raise ValueError("apply_plan_judgment 只用于 plan 单元")
    gap = verdict.extra.setdefault("gap", {})
    checks = verdict.extra.setdefault("checks", {})
    verdict.judge_model = judge_model
    if raw is None:
        verdict.pending.append({"reason": "plan_judge_output_invalid"})
        checks["premises"] = "pending_judge"
        verdict.disposition = decide(verdict.errors, verdict.pending)
        return verdict
    known = {r.id for r in (case.requirements if case else ())}
    covered_before = set(gap.get("covered_before") or [])
    n_tasks = int(verdict.extra.get("tasks") or 0)
    judge_targets: dict[str, list[str]] = {}
    premises: dict[str, list[dict[str, Any]]] = {}
    new_errors: list[ErrorRecord] = []
    for item in raw.get("tasks") or []:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index") or 0)
        if not 1 <= idx <= max(n_tasks, 1):
            continue
        key = f"T{idx}"
        targets = sorted({str(t) for t in item.get("targets") or [] if str(t) in known or not known})
        judge_targets[key] = targets
        prem = [p for p in item.get("premises") or [] if isinstance(p, dict)]
        premises[key] = [{"phrase": str(p.get("phrase") or "")[:200], "status": p.get("status"), "why": str(p.get("why") or "")[:200]} for p in prem]
        gap_driven = bool(item.get("gap_driven")) and bool(set(targets) - covered_before)
        if known and targets and not gap_driven:
            new_errors.append(error(ErrorType.TASK_NOT_GAP_DRIVEN, f"{key} 对准的 {', '.join(targets)} 在本轮前已全部覆盖", span=key,
                                    method="model", extra={"task": key, "targets": targets}))
        elif known and not targets:
            # 对不上任何案卷要求：只记 flag，让人看（可能是案卷要求写漏了）
            new_errors.append(error(ErrorType.TASK_NOT_GAP_DRIVEN, f"{key} 对不上案卷的任何要求", span=key, method="model",
                                    extra={"task": key, "targets": [], "confidence": "flag"}))
        smuggled = [p for p in premises[key] if p["status"] == "unverified" and p["phrase"]]
        if smuggled:
            new_errors.append(error(ErrorType.PREMISE_SMUGGLING, f"{key} 把未核实的说法写成既定事实：" + "；".join(p["phrase"][:60] for p in smuggled),
                                    span=smuggled[0]["phrase"], method="model", extra={"task": key, "premises": smuggled}))
    verdict.errors = [e for e in verdict.errors if e.type not in (ErrorType.TASK_NOT_GAP_DRIVEN, ErrorType.PREMISE_SMUGGLING)] + new_errors
    hind = gap.get("task_targets") or {}
    agreement = {k: {"judge": judge_targets.get(k, []), "hindsight": hind.get(k, []),
                     "same": sorted(judge_targets.get(k, [])) == sorted(hind.get(k, []))}
                 for k in sorted(set(judge_targets) | set(hind), key=lambda x: int(x[1:]))}
    gap.update({"judge_targets": judge_targets, "agreement": agreement, "judge_notes": str(raw.get("notes") or "")[:300],
                "method": "hindsight_required_evidence+judge_task_text"})
    checks["premises"] = premises
    verdict.pending = [p for p in verdict.pending if p.get("reason") != "plan_judge_output_invalid"]
    verdict.layers_run = [x for x in verdict.layers_run if x != "judge_plan"] + ["judge_plan"]
    verdict.disposition = decide(verdict.errors, verdict.pending)
    return verdict


def summarize_plan(verdict: Verdict) -> dict[str, Any]:
    gap = verdict.extra.get("gap") or {}
    return {"disposition": verdict.disposition, "judge_targets": gap.get("judge_targets"), "hindsight_targets": gap.get("task_targets"),
            "agreement": {k: v.get("same") for k, v in (gap.get("agreement") or {}).items()},
            "premises": verdict.extra.get("checks", {}).get("premises"), "errors": [e.type.value for e in verdict.errors]}


__all__ = ["PLAN_SCHEMA", "PLAN_SYSTEM_PROMPT", "apply_plan_judgment", "judge_plan", "plan_payload", "prior_reports", "summarize_plan"]
