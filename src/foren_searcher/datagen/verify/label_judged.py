"""answer 单元的标签语义判定（verifier@0.3.3）。

大白话：案卷的 label_expected 写着这道题应有的结论（"不实""属实""重复帧"…）和几个等价说法。程序层只做子串匹配，
r0c 里有两类错：
- 漏：答案写"现有证据支持该声明""与声明相符"，意思就是"属实"，但一个字都对不上 → 记了 omitted_required_fact；
- 误：标签"属实"是"不属实""部分属实"的子串 → 反向结论也算命中（程序层已加否定前缀排除，见 units.label_hit）。
接核验模型时，这里让模型读答案、任务和标签，判三种情况：
- matches：答案的最终结论与标签同义（要能从答案逐字摘出表达结论的片段，程序回查）；
- contradicts：答案给出了相反或明显不同的结论；
- absent：答案没有给出关于这一点的明确结论（只说"尚不能确定"也算）。
写回：替掉程序层的 label_expected 错误——matches 不记错；absent 记 omitted_required_fact（major，与原程序层一致）；
contradicts 记 unsupported_assertion（critical：最终结论与案卷标签相反）。模型调用失败或摘录回查不上时保留程序层结果。
"""

from __future__ import annotations

from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity, error
from .anchored import JsonJudge, _call_checked, _obj, _squash, quote_in_source
from .verdict import Verdict, decide

LABEL_SYSTEM_PROMPT = """你是答案审核员。给你一道调查题（task）、调查的最终答案（answer），以及这道题期望的结论标签（label）和它的等价说法（aliases）。
判断答案的最终结论与标签的关系，输出 JSON（字符串里的双引号用“”）：
{"status":"matches"|"contradicts"|"absent",
 "quote":"答案中表达最终结论的原文片段，逐字连续摘录；status=absent 时为空串",
 "notes":"一句话"}
规则：
- 只比较结论，不评价证据是否充分，也不用你自己的知识判断标签对不对。
- matches：答案的结论与标签意思相同，措辞可以不同（如标签“属实”，答案写“现有证据支持该声明”“与声明相符”）。
- contradicts：答案给出了相反或明显不同的结论（如标签“属实”，答案写“不实”“部分属实”“无法证实且倾向虚假”）。
- absent：答案没有对这一点给出明确结论，只罗列事实或写“尚不能确定”。
- 标签不是真假判断时（如“重复帧”“卡通”），看答案是否明确给出了这一要点：明确给出 = matches，给出不同判断 = contradicts，没提 = absent。
- 否定要看清：“并非属实”“不属实”不是“属实”。"""

LABEL_SCHEMA = _obj({"status": {"type": "string", "enum": ["matches", "contradicts", "absent"]},
                     "quote": {"type": "string"}, "notes": {"type": "string"}})
MAX_ANSWER_CHARS = 12_000


def label_checks(case: Case | None) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for check in case.checks if case else ():
        if check.kind == "label_expected":
            out.append((str(check.params["label"]), [str(a) for a in check.params.get("aliases") or [] if str(a)]))
    return out


async def judge_label(judge: JsonJudge, answer: str, task: str, label: str, aliases: list[str]) -> tuple[dict[str, Any] | None, str]:
    """返回 (判定, 失败说明)。判定：{"status", "quote", "quote_verified", "notes"}；调用失败时 (None, 错误)。"""

    payload = {"task": task[:4000], "answer": answer[:MAX_ANSWER_CHARS], "label": label, "aliases": aliases}
    raw, failure = await _call_checked(judge, LABEL_SYSTEM_PROMPT, payload, [], LABEL_SCHEMA)
    if raw is None:
        return None, failure.last_error
    status = raw.get("status")
    if status not in ("matches", "contradicts", "absent"):
        return None, f"bad status {status!r}"
    quote = str(raw.get("quote") or "").strip()
    # 结论片段常很短（"属实""不实"），quote_in_source 要求 ≥6 字符；这里短片段按去空白子串回查
    verified = bool(quote) and (_squash(quote) in _squash(answer) or quote_in_source(quote, answer))
    return {"status": status, "quote": quote, "quote_verified": verified, "notes": str(raw.get("notes") or "")}, ""


def apply_label_judgment(verdict: Verdict, label: str, aliases: list[str], judged: dict[str, Any] | None, *,
                         failure: str = "") -> Verdict:
    """把一个 label_expected 的语义判定写回 answer 单元（原地修改并返回）。

    采信条件：调用成功，且 status=absent 或摘录能在答案里逐字回查到（matches / contradicts 必须指得出原文）。
    不采信时保留程序层结果，只在 extra.label_check 里记原因。
    """

    forms = [label, *aliases]
    program = [e for e in verdict.errors if e.extra.get("check") == "label_expected" and e.extra.get("forms", forms)[:1] == [label]]
    record: dict[str, Any] = {"label": label, "program_hit": not program, "method": "substring"}
    usable = judged is not None and (judged["status"] == "absent" or judged["quote_verified"])
    if judged is None:
        record["judge_failed"] = failure or "unknown"
    elif not usable:
        record["judge_unverified"] = judged
    else:
        record.update(method="judge", **judged)
        verdict.errors = [e for e in verdict.errors if e not in program]
        new: ErrorRecord | None = None
        if judged["status"] == "absent":
            new = error(ErrorType.OMITTED_REQUIRED_FACT, f"答案没有给出应有的结论 {label!r}（核验模型语义判定）", method="model",
                        extra={"check": "label_expected", "forms": forms, "label_judgment": "absent", "notes": judged["notes"]})
        elif judged["status"] == "contradicts":
            new = error(ErrorType.UNSUPPORTED_ASSERTION, f"答案的最终结论与案卷标签 {label!r} 相反或不同：{judged['quote'][:120]}",
                        span=judged["quote"][:200], method="model", severity=Severity.CRITICAL,
                        extra={"check": "label_expected", "forms": forms, "label_judgment": "contradicts", "notes": judged["notes"]})
        if new is not None:
            verdict.errors.append(new)
    verdict.extra.setdefault("label_checks", []).append(record)
    exec_info = verdict.extra.get("exec")
    if isinstance(exec_info, dict) and record["method"] == "judge":
        exec_info["label_match_judged"] = record.get("status") == "matches"
    verdict.disposition = decide(verdict.errors, verdict.pending)
    return verdict


async def check_labels(verdict: Verdict, answer: str, case: Case | None, judge: JsonJudge) -> Verdict:
    for label, aliases in label_checks(case):
        judged, failure = await judge_label(judge, answer, case.task_text if case else "", label, aliases)
        apply_label_judgment(verdict, label, aliases, judged, failure=failure)
    return verdict


__all__ = ["LABEL_SCHEMA", "LABEL_SYSTEM_PROMPT", "apply_label_judgment", "check_labels", "judge_label", "label_checks"]
