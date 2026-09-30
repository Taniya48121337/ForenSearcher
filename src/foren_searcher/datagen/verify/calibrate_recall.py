"""核验器召回率校准：往已核验的报告副本里注入已知错误，再跑三层核验，看能不能抓回来。

大白话：calibrate.py 的 R1/R2 修的是"误报"（本来没错却被判错）。这里反过来测"漏报"：
拿一份判定干净的 Finding，故意把它改坏——换成别的 E、加一句编的、把数字改掉、去掉"据……称"、
引一个不存在的 E、删掉一条必需 Finding 却仍报 COMPLETED——每种改法我们**事先知道**该报什么错，
所以不用人标。改坏的报告用与正式核验完全相同的 verify_report 走一遍（同一核验模型、同一缓存），
抓到算命中。产出每种注入 / 每种错误类型的召回率，附漏报清单。

对照（09-25）：命中与"误报新错"都相对**同条件对照**比——每个工作区先用同 case_mode / 同 requirement_ids / 同核验模型
把**未改动的原报告**跑一遍 verify_report 得到 control 判定（未改的 (Finding, E) 全命中缓存）。先前拿 verdicts_r0_judged
（不带案卷的 reverify）当基线，而重核带 scoped 案卷，原报告本来就有的 omitted_required_fact 等全被记成 false_new（202 处里 78 条）。

只改报告 dict，不动证据库；未改的 (Finding, E) 对命中 CachedJudge 缓存，所以一次注入只多花
被改那条 Finding 的核验调用。

用法（本机；默认连本地核验服务 --judge local，不花钱。--judge openrouter 走付费 API）：
    python -m foren_searcher.datagen.verify.calibrate_recall --runs experiments/sft_data/runs/true_4124903_r1 \
        --verdicts experiments/sft_data/verdicts_r0_judged.jsonl --judge local \
        --out experiments/sft_data/calibration/recall_r0.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..cases import Case, load_cases
from ..errors import ErrorRecord, ErrorType
from ..runlog import RunLog, load_run
from .anchored import BUDGET_EXIT_CODE, JsonJudge, JudgeBudgetExhausted, _squash, budget_message
from .calibrate import recalibrate
from .judges import add_judge_arguments, judge_from_args, judge_summary
from .reverify import _visible_ids
from .scope import requirements_for_workspace
from .verdict import Verdict, verify_report

_NUM = re.compile(r"(?<!\d)(\d{2,4})(?!\d)")     # 只改两位以上的数（改一位数会出"20 月"这种一眼假的东西）
_ATTRIB = re.compile(r"(据|根据)[^，,。；;]{1,30}?(称|报道|表示|说|通报|指出)[，,]?")
_FAKE_CLAUSES = ("现场共有 37 人受伤", "事发时间为当地时间凌晨 2 点 40 分", "涉事车辆车牌尾号为 4471",
                 "现场负责人姓名为 Laurent Mercier", "该说法此前已被路透社在 2014 年核实")


@dataclass
class Injection:
    kind: str
    workspace_id: str
    finding_id: str | None
    expected: tuple[ErrorType, ...]          # 命中任一即算召回
    detail: str
    report: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "workspace_id": self.workspace_id, "finding_id": self.finding_id,
                "expected": [e.value for e in self.expected], "detail": self.detail, **self.extra}


Mutator = Callable[[dict[str, Any], dict[str, Any], "RunLog", Case | None, random.Random], Injection | None]


def _finding(report: dict[str, Any], fid: str) -> dict[str, Any]:
    for f in report["findings"]:
        if f.get("finding_id") == fid:
            return f
    raise KeyError(fid)


def _text_evidence(log: RunLog, ids: list[str]) -> list[str]:
    return [e for e in ids if log.lookup.has(e) and log.lookup.get(e).is_text]


def _cited_text(log: RunLog, ids: list[str]) -> str:
    return "\n".join(log.lookup.get(e).content or "" for e in _text_evidence(log, ids))


# --------------------------------------------------------------------------- 六种注入

def swap_citation(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """把这条 Finding 的引用整体换成报告里其它 Finding 引用、且与本条不重叠的文本 E。期望 wrong_citation / unsupported_assertion。"""

    own = set(finding.get("evidence_ids") or [])
    pool = sorted({e for f in report["findings"] for e in f.get("evidence_ids") or [] if e not in own})
    pool = _text_evidence(log, pool)
    # 内容里恰好也提到本句关键词的 E 不算"错引"，粗筛：本句前 12 个字不出现在候选 E 正文里
    head = _squash(str(finding.get("statement") or ""))[:12]
    pool = [e for e in pool if head and head not in _squash(log.lookup.get(e).content or "")]
    if not pool:
        return None
    new_ids = rng.sample(pool, k=min(2, len(pool)))
    mutated = copy.deepcopy(report)
    _finding(mutated, finding["finding_id"])["evidence_ids"] = new_ids
    return Injection("swap_citation", "", finding["finding_id"], (ErrorType.WRONG_CITATION, ErrorType.UNSUPPORTED_ASSERTION),
                     f"引用 {sorted(own)} → {new_ids}", mutated, {"before": sorted(own), "after": new_ids})


def fabricate_id(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    fake = "E9%03d" % rng.randint(0, 999)
    while log.lookup.has(fake):
        fake = "E9%03d" % rng.randint(0, 999)
    mutated = copy.deepcopy(report)
    _finding(mutated, finding["finding_id"])["evidence_ids"] = list(finding.get("evidence_ids") or []) + [fake]
    return Injection("fabricate_id", "", finding["finding_id"], (ErrorType.FABRICATED_EVIDENCE_ID,), f"追加不存在的 {fake}", mutated, {"fake": fake})


def append_clause(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """句尾加一句编造的可核事实（数字 / 人名 / 时间），来源里没有。期望 unsupported_assertion（或 wrong_citation：
    核验模型有时把编造句判成"与来源相反"，09-25 起也算命中）。"""

    cited = _squash(_cited_text(log, finding.get("evidence_ids") or []))
    candidates = [c for c in _FAKE_CLAUSES if _squash(c)[-6:] not in cited]
    if not candidates:
        return None
    clause = rng.choice(candidates)
    statement = str(finding.get("statement") or "").rstrip()
    statement = (statement[:-1] if statement.endswith(("。", ".", "；")) else statement) + "；" + clause + "。"
    mutated = copy.deepcopy(report)
    _finding(mutated, finding["finding_id"])["statement"] = statement
    return Injection("append_clause", "", finding["finding_id"], (ErrorType.UNSUPPORTED_ASSERTION, ErrorType.WRONG_CITATION),
                     f"追加 “{clause}”", mutated,
                     {"clause": clause})


def alter_number(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """把句中一个在来源里出现过的数字加 13（年份也一样，2016 → 2029，改法明显但可核）。期望 wrong_citation 一族。"""

    statement = str(finding.get("statement") or "")
    cited = _cited_text(log, finding.get("evidence_ids") or [])
    hits = [m for m in _NUM.finditer(statement) if m.group(1) in cited and not statement[max(0, m.start() - 8):m.start()].startswith("http")]
    hits = [m for m in hits if "E" != statement[m.start() - 1:m.start()] and "F" != statement[m.start() - 1:m.start()]]
    if not hits:
        return None
    m = rng.choice(hits)
    old = m.group(1)
    new = str(int(old) + 13)
    mutated = copy.deepcopy(report)
    _finding(mutated, finding["finding_id"])["statement"] = statement[:m.start(1)] + new + statement[m.end(1):]
    return Injection("alter_number", "", finding["finding_id"],
                     (ErrorType.WRONG_CITATION, ErrorType.SLOPPY_NUMERIC, ErrorType.MISREAD_FINE_TEXT, ErrorType.UNSUPPORTED_ASSERTION),
                     f"{old} → {new}", mutated, {"before": old, "after": new})


def drop_attribution(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """去掉"据 X 称/通报"，把转述写成事实。期望 attribution_drift（或 unsupported_assertion）。"""

    statement = str(finding.get("statement") or "")
    m = _ATTRIB.search(statement)
    if not m:
        return None
    mutated = copy.deepcopy(report)
    _finding(mutated, finding["finding_id"])["statement"] = statement[:m.start()] + statement[m.end():]
    return Injection("drop_attribution", "", finding["finding_id"], (ErrorType.ATTRIBUTION_DRIFT, ErrorType.UNSUPPORTED_ASSERTION),
                     f"删去 “{m.group(0)}”", mutated, {"removed": m.group(0)})


def _matched_required_keys(findings: list[dict[str, Any]], log: RunLog, case: Case) -> set[str]:
    """这些 Finding 所引的 E 里，按 RequiredEvidence.matches 能对上的必需证据 key（整案范围）。"""

    texts = [log.lookup.get(str(eid)) for f in findings for eid in f.get("evidence_ids") or [] if log.lookup.has(str(eid))]
    return {req.key for req in case.scoped_required_evidence(None)
            if any(req.matches(source_url=t.source_url, acquired_by=t.acquired_by, label=t.label) for t in texts)}


def _unsatisfied_groups(matched: set[str], case: Case) -> set[tuple[str, tuple[str, ...]]]:
    return {(mode, keys) for mode, keys in case.evidence_groups()
            if not (all(k in matched for k in keys) if mode == "all" else any(k in matched for k in keys))}


def drop_required_finding(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """删掉这条 Finding 仍报 COMPLETED。期望 completed_without_finishing / omitted_required_fact。

    只挑"删掉后某组必需证据（all 组的单项 / any_one_of 组的全部候选）在报告里再无任何 Finding 引用"的 Finding（09-24）：
    别的 Finding 还引着同一批 E（如 W1-1 删 F1 而 F3 仍引那几帧）就不构成缺项，注入它只会制造假漏报。
    """

    if case is None or not case.must_cite_requirements or report.get("status") != "COMPLETED" or len(report["findings"]) < 2:
        return None
    mutated = copy.deepcopy(report)
    mutated["findings"] = [f for f in mutated["findings"] if f.get("finding_id") != finding["finding_id"]]
    before = _unsatisfied_groups(_matched_required_keys(report["findings"], log, case), case)
    lost = _unsatisfied_groups(_matched_required_keys(mutated["findings"], log, case), case) - before
    if not lost:
        return None
    return Injection("drop_required_finding", "", finding["finding_id"],
                     (ErrorType.COMPLETED_WITHOUT_FINISHING, ErrorType.OMITTED_REQUIRED_FACT), "删除该 Finding，状态保持 COMPLETED", mutated,
                     {"lost_required": sorted(k for _, keys in lost for k in keys)})


def flip_to_incomplete(report: dict[str, Any], finding: dict[str, Any], log: RunLog, case: Case | None, rng: random.Random) -> Injection | None:
    """把一份无错误、案卷判"可回答"的 COMPLETED 报告改成 INCOMPLETE（Finding 原样保留，加一句空泛的 status_reason）。
    期望 incomplete_though_answerable。测的是核验器对"诚实放弃"有多宽松（09-28 数据质量评估：accepted 报告 INCOMPLETE 11 : COMPLETED 9）。
    每份报告只注入一次（finding 参数只用来去重）。"""

    if report.get("status") != "COMPLETED" or not report.get("findings"):
        return None
    if finding is not report["findings"][0]:          # 只在第一条 Finding 上触发一次
        return None
    mutated = copy.deepcopy(report)
    mutated["status"] = "INCOMPLETE"
    mutated["status_reason"] = "已有材料还不足以完全确认，可能还需要更多来源。"
    return Injection("flip_to_incomplete", "", finding["finding_id"], (ErrorType.INCOMPLETE_THOUGH_ANSWERABLE,),
                     "COMPLETED → INCOMPLETE，Finding 不变，status_reason 空泛", mutated, {})


MUTATORS: dict[str, Mutator] = {
    "swap_citation": swap_citation, "fabricate_id": fabricate_id, "append_clause": append_clause,
    "alter_number": alter_number, "drop_attribution": drop_attribution, "drop_required_finding": drop_required_finding,
    "flip_to_incomplete": flip_to_incomplete,
}


def plan_injections(log: RunLog, case: Case | None, base: dict[str, Verdict], *, kinds: list[str], per_kind: int, seed: int,
                    scopes: dict[str, set[str] | None] | None = None) -> list[Injection]:
    """scopes：工作区 → 充分性收窄到的 requirement_ids（scoped 模式）。给了就按范围挑 drop_required_finding——
    删掉只影响范围外证据组的 Finding 不算"缺必需证据"（09-29：units 重判后 W1-2 只对 REQ1，删唯一引帧的 F7 不再构成缺项）。"""

    rng = random.Random(seed)
    out: list[Injection] = []
    for rec in log.store.workspace_report_records():
        report = rec.to_dict()
        scoped_case = case
        if case is not None and scopes is not None and scopes.get(rec.task_id) is not None:
            scoped_case = dataclasses.replace(case, required_evidence=case.scoped_required_evidence(scopes[rec.task_id]))
        verdict = base.get(rec.task_id)
        tainted = set(verdict.tainted_findings) if verdict else set()
        clean = [f for f in report.get("findings") or [] if isinstance(f, dict) and f.get("finding_id") not in tainted and f.get("evidence_ids")]
        for kind in kinds:
            made = 0
            for finding in rng.sample(clean, k=len(clean)):
                if made >= per_kind:
                    break
                inj = MUTATORS[kind](report, finding, log, scoped_case if kind == "drop_required_finding" else case, rng)
                if inj is None:
                    continue
                inj.workspace_id = rec.task_id
                out.append(inj)
                made += 1
    return out


# --------------------------------------------------------------------------- 跑与打分

def _error_key(e: ErrorRecord) -> tuple[ErrorType, Any, Any]:
    """错误的比对键：(类型, Finding, 必需证据 key / 案卷要求 key)。status 级错误 finding 为 None。"""

    return e.type, e.extra.get("finding"), e.extra.get("key") or e.extra.get("requirement")


def new_errors(verdict: Verdict, control: Verdict | None) -> list[ErrorRecord]:
    """注入后判定里、同条件对照判定没有的非 flag 错误。"""

    known = {_error_key(e) for e in control.errors} if control else set()
    return [e for e in verdict.errors if e.extra.get("confidence") != "flag" and _error_key(e) not in known]


def detected(injection: Injection, verdict: Verdict, control: Verdict | None = None) -> tuple[bool, list[str]]:
    """命中 = 注入后出现了期望类型的错误，且该错误**不在**同工作区同条件对照判定里（按 _error_key 比）。
    control 应是同 case_mode / requirement_ids / 核验模型对未改原报告的判定（09-25；09-24 用的 r0 基线条件不同）。"""

    hits: list[str] = []
    for e in new_errors(verdict, control):
        if e.type not in injection.expected:
            continue
        finding = e.extra.get("finding")
        if injection.kind in ("drop_required_finding", "flip_to_incomplete") or finding in (None, injection.finding_id):
            hits.append(e.type.value)
    return bool(hits), sorted(set(hits))


async def run_calibration(run_dirs: list[Path], cases: dict[str, Case], base_verdicts: list[Verdict], judge: JsonJudge | None, *,
                          judge_name: str | None, kinds: list[str], per_kind: int, seed: int, case_mode: str = "scoped",
                          units: list[Verdict] | None = None, say: Callable[[str], None] = print) -> dict[str, Any]:
    """case_mode：scoped（0.3.0 默认，按工作区范围给案卷）| none（不带案卷，09-23 基线条件）| whole（整案清单，09-21 第一次的条件）。"""

    rows: list[dict[str, Any]] = []
    controls: dict[str, dict[str, Any]] = {}
    aborted: str | None = None
    abort_message = ""
    try:
        await _calibrate_runs(run_dirs, cases, base_verdicts, judge, rows, controls, judge_name=judge_name, kinds=kinds,
                              per_kind=per_kind, seed=seed, case_mode=case_mode, units=units, say=say)
    except JudgeBudgetExhausted as exc:      # 没钱了：已完成的 rows 照常汇总，标 aborted
        aborted = "budget_exhausted"
        say(budget_message(exc))
        abort_message = str(exc)
    summary = summarize(rows, judge_name=judge_name, kinds=kinds)
    summary["case_mode"] = case_mode
    summary["baseline"] = "control: 同 case_mode / requirement_ids / judge 下未改原报告的判定"
    summary["controls"] = controls
    if aborted:
        summary["aborted"] = aborted
        summary["aborted_message"] = abort_message
    return summary


async def _calibrate_runs(run_dirs: list[Path], cases: dict[str, Case], base_verdicts: list[Verdict], judge: JsonJudge | None,
                          rows: list[dict[str, Any]], controls: dict[str, dict[str, Any]], *, judge_name: str | None, kinds: list[str],
                          per_kind: int, seed: int, case_mode: str, units: list[Verdict] | None, say: Callable[[str], None]) -> None:
    """逐 run 逐注入核验，结果即时追加进 rows / controls（中途抛 JudgeBudgetExhausted 时已完成的仍保留）。"""

    for run_dir in run_dirs:
        log = load_run(run_dir)
        try:
            if log.origin == "D":        # 离线换题复制了源 run 的证据库，报告是源 run 的，召回已在源 run 上算过，不重复计
                say(f"[{log.name}] 离线换题（来源 D），跳过召回注入")
                continue
            case = cases.get(log.case_id or "")
            base = {str(v.subject.get("workspace_id")): recalibrate(v) for v in base_verdicts if v.subject.get("run") == log.name and v.unit == "report"}
            scopes = None
            if case_mode == "scoped" and case is not None:
                scopes = {}
                for rec in log.store.workspace_report_records():
                    try:
                        text = str(log.store.workspace_task_record(rec.task_id).get("task") or "")
                    except KeyError:
                        text = ""
                    scopes[rec.task_id] = requirements_for_workspace(case, text, run=log.name, workspace_id=rec.task_id, units=units)[0]
            injections = plan_injections(log, case, base, kinds=kinds, per_kind=per_kind, seed=seed, scopes=scopes)
            say(f"[{log.name}] {len(injections)} 处注入")
            # 同条件对照：每个工作区一次，未改原报告 + 同 case_mode / requirement_ids / judge
            conditions: dict[str, tuple[str, set[str] | None, dict[str, Any], Verdict]] = {}
            for inj in injections:
                if inj.workspace_id not in conditions:
                    conditions[inj.workspace_id] = await _control(log, run_dir, case, inj.workspace_id, judge, judge_name=judge_name,
                                                                  case_mode=case_mode, units=units)
                    control = conditions[inj.workspace_id][3]
                    controls[f"{log.name}/{inj.workspace_id}"] = {
                        "run": log.name, "workspace_id": inj.workspace_id, "disposition": control.disposition,
                        "control_error_types": sorted({e.type.value for e in control.errors if e.extra.get("confidence") != "flag"}),
                        "scope": conditions[inj.workspace_id][2]}
                task_text, req_ids, scope_note, control = conditions[inj.workspace_id]
                subject = {"run": log.name, "case_id": log.case_id, "workspace_id": inj.workspace_id, "injection": inj.kind, "finding": inj.finding_id}
                verdict = await verify_report(inj.report, log.lookup, subject=subject, case=None if case_mode == "none" else case,
                                              task_text=task_text or None, judge=judge, requirement_ids=req_ids,
                                              visible_ids=_visible_ids(run_dir, inj.workspace_id), judge_model=judge_name)
                verdict = recalibrate(verdict)
                ok, hits = detected(inj, verdict, control)
                new_types = sorted({e.type.value for e in new_errors(verdict, control)})
                reasons, last_error = pending_reasons(verdict.pending)
                rows.append({**inj.to_dict(), "run": log.name, "detected": ok, "hits": hits, "new_error_types": new_types, "scope": scope_note,
                             "disposition": verdict.disposition, "pending": len(verdict.pending),
                             "pending_reasons": reasons, "pending_last_error": last_error,
                             "control_error_types": controls[f"{log.name}/{inj.workspace_id}"]["control_error_types"],
                             "errors": [{"type": e.type.value, "severity": e.severity.value, "finding": e.extra.get("finding"), "span": (e.span or "")[:80],
                                         "detail": e.detail[:160]}
                                        for e in verdict.errors]})
                mark = "命中" if ok else "未定" if verdict.pending else "漏报"
                say(f"  {inj.kind:22s} {inj.workspace_id} {inj.finding_id or '-':4s} {mark} {hits or new_types} — {inj.detail}"
                    + (f"（pending {reasons}：{last_error[:80]}）" if verdict.pending and not ok else ""))
        finally:
            log.close()


async def _control(log: RunLog, run_dir: Path, case: Case | None, workspace_id: str, judge: JsonJudge | None, *, judge_name: str | None,
                   case_mode: str, units: list[Verdict] | None) -> tuple[str, set[str] | None, dict[str, Any], Verdict]:
    """一个工作区的注入条件（任务文字、案卷范围）与对照判定：未改原报告按同条件跑 verify_report。"""

    try:
        task_text = str(log.store.workspace_task_record(workspace_id).get("task") or "")
    except KeyError:
        task_text = ""
    # 0.3.0：案卷按工作区范围给（scope.py），既不是整案清单（09-21 每处多一条假 omitted），也不是不带（09-23 删 Finding 抓不到）
    req_ids: set[str] | None = None
    scope_note: dict[str, Any] = {"method": case_mode}
    if case_mode == "scoped":
        req_ids, scope_note = requirements_for_workspace(case, task_text, run=log.name, workspace_id=workspace_id, units=units)
    original = log.store.workspace_report(workspace_id).to_dict()
    subject = {"run": log.name, "case_id": log.case_id, "workspace_id": workspace_id, "injection": "control", "finding": None}
    control = await verify_report(original, log.lookup, subject=subject, case=None if case_mode == "none" else case,
                                  task_text=task_text or None, judge=judge, requirement_ids=req_ids,
                                  visible_ids=_visible_ids(run_dir, workspace_id), judge_model=judge_name)
    return task_text, req_ids, scope_note, recalibrate(control)


def pending_reasons(pending: list[dict[str, Any]]) -> tuple[dict[str, int], str]:
    """pending 按 reason 计数，外加最后一条带错误的 pending 的错误文本（没有就空串）。"""

    reasons: dict[str, int] = {}
    last_error = ""
    for item in pending:
        reason = str(item.get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        text = item.get("last_error") or ("; ".join(f"{k}: {v}" for k, v in item["error"].items()) if isinstance(item.get("error"), dict) else "")
        if text:
            last_error = str(text)[:300]
    return reasons, last_error


def is_inconclusive(row: dict[str, Any]) -> bool:
    """未命中但核验有 pending（模型调用失败、缺像素等）：说不清是漏报还是没判成，不算进 conclusive 召回的分母。"""

    return not row["detected"] and int(row.get("pending") or 0) > 0


def _recall(sub: list[dict[str, Any]]) -> tuple[float | None, int, float | None]:
    """(召回, inconclusive 数, conclusive 召回 = 命中 / (n − inconclusive))。"""

    hit = sum(1 for r in sub if r["detected"])
    inconclusive = sum(1 for r in sub if is_inconclusive(r))
    conclusive = len(sub) - inconclusive
    return (round(hit / len(sub), 3) if sub else None), inconclusive, (round(hit / conclusive, 3) if conclusive else None)


def summarize(rows: list[dict[str, Any]], *, judge_name: str | None, kinds: list[str]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        sub = [r for r in rows if r["kind"] == kind]
        recall, inconclusive, recall_conclusive = _recall(sub)
        by_kind[kind] = {"n": len(sub), "detected": sum(1 for r in sub if r["detected"]),
                         "recall": recall, "inconclusive": inconclusive, "recall_conclusive": recall_conclusive,
                         "misses": [{"run": r["run"], "workspace_id": r["workspace_id"], "finding_id": r["finding_id"], "detail": r["detail"],
                                     "got": r["new_error_types"], "inconclusive": is_inconclusive(r)} for r in sub if not r["detected"]]}
    by_type: dict[str, dict[str, int]] = {}
    for r in rows:
        for t in r["expected"]:
            slot = by_type.setdefault(t, {"expected_in": 0, "hit": 0})
            slot["expected_in"] += 1
            if t in r["hits"]:
                slot["hit"] += 1
    n = len(rows)
    recall, inconclusive, recall_conclusive = _recall(rows)
    return {"judge_model": judge_name, "layers": "exec+anchored+sufficiency" if judge_name else "exec only",
            "n_injections": n, "recall_overall": recall,
            "inconclusive": inconclusive, "recall_conclusive": recall_conclusive,
            "note_inconclusive": "inconclusive = 未命中且有 pending（核验调用失败等）；recall_conclusive 的分母去掉这些",
            "by_kind": by_kind, "by_expected_type": by_type,
            "false_new_errors": [{"run": r["run"], "workspace_id": r["workspace_id"], "kind": r["kind"], "types": r["new_error_types"]}
                                 for r in rows if [t for t in r["new_error_types"] if t not in r["expected"]]],
            "rows": rows}


def _load_units(path: str | None) -> list[Verdict]:
    if not path or not Path(path).is_file():
        return []
    return [Verdict.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核验器召回率校准（注入已知错误再核验）")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cases", default="experiments/sft_data/cases")
    parser.add_argument("--verdicts", default="experiments/sft_data/verdicts_r0_judged.jsonl")
    add_judge_arguments(parser, default="none")
    parser.add_argument("--cache", default="experiments/sft_data/verify_cache.jsonl")
    parser.add_argument("--kinds", nargs="+", default=list(MUTATORS))
    parser.add_argument("--per-kind", type=int, default=2, help="每份报告每种注入最多改几条 Finding")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-mode", default="scoped", choices=["scoped", "none", "whole"],
                        help="scoped=按工作区范围给案卷（0.3.0 默认）；none=不带案卷（09-23 基线）；whole=整案清单（09-21 第一次）")
    parser.add_argument("--units", default="experiments/sft_data/verdicts_units_judged.jsonl",
                        help="plan 单元的模型评判（scoped 模式用 judge_targets 定范围；没有就按任务文字兜底）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    judge = judge_from_args(args, cache_path=args.cache or None)
    judge_name = getattr(judge, "name", None) if judge is not None else None
    cases = load_cases(args.cases) if Path(args.cases).is_dir() else {}
    base = [Verdict.from_dict(json.loads(line)) for line in Path(args.verdicts).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = asyncio.run(run_calibration([Path(r) for r in args.runs], cases, base, judge, judge_name=judge_name, kinds=args.kinds,
                                          per_kind=args.per_kind, seed=args.seed, case_mode=args.case_mode, units=_load_units(args.units)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"召回 {summary['recall_overall']}（{summary['n_injections']} 处注入）；inconclusive {summary['inconclusive']}，"
          f"conclusive 召回 {summary['recall_conclusive']} → {out}")
    for kind, row in summary["by_kind"].items():
        print(f"  {kind:22s} {row['detected']}/{row['n']}  inconclusive {row['inconclusive']}  conclusive {row['recall_conclusive']}")
    line = judge_summary(judge)
    if line:
        print(line, flush=True)
    if summary.get("aborted"):
        print(f"（预算用尽，中途停止：summary 只含已完成的 {summary['n_injections']} 处注入，已标 aborted）", flush=True)
        print(budget_message(summary.get("aborted_message") or "budget exhausted"), flush=True)
        return BUDGET_EXIT_CODE
    return 0


__all__ = ["Injection", "MUTATORS", "detected", "is_inconclusive", "main", "new_errors", "pending_reasons", "plan_injections", "run_calibration",
           "summarize"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
