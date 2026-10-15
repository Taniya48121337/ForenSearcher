"""答案单元 premature_finish 的语义覆盖（候选，默认关闭；2026-10-04）。

大白话：现在判"must_cite 要求有没有被干净 Finding 覆盖"，只看 Finding 引的 E 能不能按网址 / 获取方式对上案卷的
required_evidence。required_evidence 的网址来自试点 run 当时打开过的页面：换一个来源讲同一件事就不算，
被检索护栏拦掉的网址更是永远取不到。r0cn 的 660、699 都是答案已经把两条要求讲清楚了，仍被判 premature_finish。

这里只对网址没覆盖到的 must_cite 要求再问一次核验模型：把要求原文和本 run 干净 Finding 里**已核验的支持片段**
（文字来源：引语已程序回查；图片来源：单图判定给的描述）交给它，判该要求是否已经建立。建立了就从 uncovered 里去掉；
全部建立则去掉 premature_finish。留痕在 extra.sufficiency.semantic。
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity
from .anchored import JsonJudge, _call_checked, _obj, candidate_enabled
from .verdict import Verdict, decide, tainted

SEMANTIC_COVERAGE = {"enabled": candidate_enabled("semcov")}   # verifier@0.3.6 起默认打开

SEMANTIC_SYSTEM_PROMPT = """你是调查审核员。给你案卷里的一组调查要求（requirements），以及调查报告中已经核验过的 Finding 片段：每条是"Finding 写的一部分 + 来源原文引语（或图片内容描述）+ 来源标题/网址"。文字来源的引语已由程序在来源正文中找到；图片来源的描述来自逐图核验。
只依据这些已核验片段判断每条要求是否已被建立：片段里的来源内容足以回答该要求才算 covered；只沾边、只回答了一半、或只说明"没找到"都不算。不评价来源是不是某个特定网站，可靠来源都可以。
保守：拿不准就 covered=false。输出 JSON：{"requirements":[{"id":"REQ1","covered":true|false,"by":["W1-1/F1"],"note":"一句话"}]}"""

SEMANTIC_SCHEMA = _obj({"requirements": {"type": "array", "items": _obj({
    "id": {"type": "string"}, "covered": {"type": "boolean"}, "by": {"type": "array", "items": {"type": "string"}}, "note": {"type": "string"}})}})

MAX_FRAGMENTS = 120


def verified_fragments(report_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """同一 run 的报告判定行 → 干净 Finding 的已核验支持片段。"""

    out: list[dict[str, str]] = []
    for row in report_rows:
        if row.get("unit") != "report":
            continue
        ws = str((row.get("subject") or {}).get("workspace_id"))
        tainted = set(row.get("tainted_findings") or [])
        for j in row.get("judgments") or []:
            if j.get("finding_id") in tainted or j.get("verdict") != "supported":
                continue
            for p in j.get("supported") or []:
                if p.get("verified") is True or (j.get("is_image") and p.get("verified") is None):
                    out.append({"finding": f"{ws}/{j.get('finding_id')}", "part": str(p.get("part") or "")[:200],
                                "quote": str(p.get("quote") or "")[:400], "source": str(j.get("source") or "")[:200],
                                **({"kind": "image"} if j.get("is_image") else {})})
    return out


async def apply_semantic_coverage(verdict: Verdict, case: Case | None, report_rows: list[dict[str, Any]], judge: JsonJudge) -> Verdict:
    """answer 单元：网址没覆盖的 must_cite 要求按语义再判；原地修改并返回。"""

    if not SEMANTIC_COVERAGE["enabled"] or verdict.unit != "answer" or case is None:
        return verdict
    suff = verdict.extra.get("sufficiency") or {}
    uncovered = [r for r in suff.get("uncovered") or []]
    if not uncovered or not any(e.type is ErrorType.PREMATURE_FINISH for e in verdict.errors):
        return verdict
    texts = {r.id: r.text for r in case.requirements}
    frags = verified_fragments(report_rows)
    if not frags:
        return verdict
    payload = {"requirements": [{"id": r, "text": texts.get(r, "")} for r in uncovered], "verified_fragments": frags[:MAX_FRAGMENTS]}
    raw, _failure = await _call_checked(judge, SEMANTIC_SYSTEM_PROMPT, payload, [], SEMANTIC_SCHEMA)
    if not isinstance(raw, dict):
        return verdict                                    # 调用失败：保留原判
    sem = {str(x.get("id")): x for x in raw.get("requirements") or [] if isinstance(x, dict)}
    still = [r for r in uncovered if not (sem.get(r) or {}).get("covered")]
    verdict.extra["sufficiency"] = {**suff, "uncovered_by_url": uncovered, "uncovered": still, "premature_finish": bool(still),
                                    "semantic": {r: {"covered": bool((sem.get(r) or {}).get("covered")),
                                                     "by": list((sem.get(r) or {}).get("by") or [])[:6],
                                                     "note": str((sem.get(r) or {}).get("note") or "")[:200]} for r in uncovered}}
    kept = []
    for e in verdict.errors:
        if e.type is ErrorType.PREMATURE_FINISH:
            if not still:
                continue
            e = replace(e, detail=f"must_cite 要求 {', '.join(still)} 没有任何核实的 Finding 覆盖却 FINISH（网址与语义都未覆盖）")
        kept.append(e)
    verdict.errors = kept
    verdict.disposition = decide(verdict.errors, verdict.pending)
    return verdict


# ---------------------------------------------------------------- 报告级（第一层 required_evidence_cited + 第三层"任务要求未被回答"）

_FIRM_SEV = (Severity.MAJOR, Severity.CRITICAL)
_UNCOVERED = re.compile(r"任务要求未全部覆盖[:：]\s*(.+)$")


def _missing_keys(e: ErrorRecord) -> list[str] | None:
    """这条错误是"某必需证据没取到"类的话，返回涉及的 required_evidence key；否则 None。"""

    x = e.extra or {}
    if e.severity not in _FIRM_SEV or x.get("confidence") == "flag" or x.get("optional"):
        return None
    if e.type is ErrorType.OMITTED_REQUIRED_FACT:
        if x.get("check") == "required_evidence_cited":
            return [str(k) for k in (x.get("keys") or [x.get("key")]) if k]
        if x.get("from_case") and x.get("requirement"):
            return [str(k) for k in x.get("keys") or [x["requirement"]]]
    if e.type is ErrorType.COMPLETED_WITHOUT_FINISHING:
        if x.get("check") == "required_evidence_cited" and x.get("key"):
            return [str(x["key"])]
        m = _UNCOVERED.search(e.detail or "")
        if m and e.method == "model":
            return [k2 for k in m.group(1).split(",") for k2 in re.split(r"[（(]", k, maxsplit=1)[0].strip().split("|") if k2]
    return None


async def apply_report_semantic_coverage(verdict: Verdict, case: Case | None, judge: JsonJudge) -> Verdict:
    """报告单元：因"必需证据没取到"而记的 major / critical，若该证据支撑的（本工作区范围内的）要求已由本报告的已核验片段建立，
    就降为 minor（留痕 program_rule=semantic_coverage）。要求没建立、或错误另有原因（如审核认为应报 INCOMPLETE 而没列缺项）的不动。"""

    if not SEMANTIC_COVERAGE["enabled"] or verdict.unit != "report" or case is None:
        return verdict
    targets = [(i, keys) for i, e in enumerate(verdict.errors) if (keys := _missing_keys(e))]
    if not targets:
        return verdict
    evidence = {r.key: r for r in case.required_evidence}
    scope = set(verdict.extra.get("scope") or []) or None
    supports: dict[str, set[str]] = {}
    for _, keys in targets:
        for k in keys:
            if k in evidence:
                reqs = set(evidence[k].supports)
                supports[k] = (reqs & scope) if scope else reqs
    asked = sorted(set().union(*supports.values())) if supports else []
    frags = verified_fragments([verdict.to_dict()])
    if not asked or not frags:
        return verdict
    texts = {r.id: r.text for r in case.requirements}
    payload = {"requirements": [{"id": r, "text": texts.get(r, "")} for r in asked], "verified_fragments": frags[:MAX_FRAGMENTS]}
    raw, _failure = await _call_checked(judge, SEMANTIC_SYSTEM_PROMPT, payload, [], SEMANTIC_SCHEMA)
    if not isinstance(raw, dict):
        return verdict
    sem = {str(x.get("id")): x for x in raw.get("requirements") or [] if isinstance(x, dict)}
    covered = {r for r in asked if (sem.get(r) or {}).get("covered")}
    resolved = {k for k, reqs in supports.items() if reqs and reqs <= covered}
    out = list(verdict.errors)
    for i, keys in targets:
        if keys and all(k in resolved for k in keys):
            e = out[i]
            reqs = sorted(set().union(*(supports[k] for k in keys)))
            out[i] = replace(e, severity=Severity.MINOR,
                             detail=f"（语义覆盖：{', '.join(reqs)} 已由本报告的已核验片段建立，仅备注）" + e.detail,
                             extra={**e.extra, "program_rule": "semantic_coverage", "original_severity": e.severity.value,
                                    "requirements": reqs})
    verdict.errors = out
    verdict.extra["semantic_coverage"] = {"asked": asked, "covered": sorted(covered), "resolved_keys": sorted(resolved),
                                          "notes": {r: str((sem.get(r) or {}).get("note") or "")[:200] for r in asked},
                                          "by": {r: list((sem.get(r) or {}).get("by") or [])[:6] for r in asked}}
    verdict.tainted_findings = tainted(verdict.errors)
    verdict.disposition = decide(verdict.errors, verdict.pending)
    return verdict


__all__ = ["SEMANTIC_COVERAGE", "SEMANTIC_SCHEMA", "SEMANTIC_SYSTEM_PROMPT", "apply_report_semantic_coverage", "apply_semantic_coverage",
           "verified_fragments"]
