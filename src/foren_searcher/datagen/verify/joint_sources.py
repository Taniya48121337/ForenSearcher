"""第二层的"多来源合并复核"（候选，默认关闭；2026-10-05）。

大白话：第二层逐个 (句, E) 单独判，再汇总"还有哪些部分没有任何一条来源支持"。比较 / 拼合类的说法在这里吃亏：
"两侧全段自动转写及分段一致"（r0e videosham:1028 A10）要把 E13（A 的转写）和 E14（B 的转写）放在一起才看得出，
单看哪一条都只讲了一侧，于是两条都判"未找到"，汇总记成"所引 E 均未支持这一部分"（critical）。
汇总那一步只拿到各来源已核实的支持片段，也看不到原文；附图规则（0.3.3）只解决了多张图片的跨帧比较，文本来源没有对应办法。

这里对仍是实锤"所引 E 均未支持这一部分"、且这句话引了两条以上来源（至少一条是文本）的片段，
把这几条来源的原文放在一起再问一次：能否**对照或拼合至少两条来源**直接得出该片段。
- 是：降为 minor，留痕 extra.program_rule = "joint_sources"、extra.joint（用到的来源与引语）、original_severity；
- 否：原判不动，extra.joint 记模型理由。

防放水：
- 模型必须给出至少两条、来自不同 E 的原文引语；文本来源的引语程序回查（quote_in_source），至少要有一条文本引语查得到，
  任何一条文本引语查不到就不算；
- 只用一条来源就能得出的不算（那是第二层单对判定的事，不在这里翻案）；
- 来源之间矛盾、需要外部知识、拿不准一律 false。
每句话最多一次调用（同句的几个片段一起问），进缓存。
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..errors import ErrorRecord, ErrorType, Severity
from .anchored import COVERAGE_MAX_IMAGES, JsonJudge, _call_checked, _obj, _source_payload, quote_in_source
from .context import EvidenceLookup, EvidenceText

JOINT_SOURCES = {"enabled": os.environ.get("FOREN_VERIFIER_JOINT", "0").strip().lower() in ("1", "on", "true")}   # 候选，默认关闭

JOINT_MAX_CHARS_PER_SOURCE = 30_000
JOINT_MAX_CHARS_TOTAL = 100_000
_FIRM = (Severity.MAJOR, Severity.CRITICAL)
_UNSUPPORTED_DETAIL = "所引 E 均未支持这一部分"

JOINT_SYSTEM_PROMPT = """你是证据核对员。一句调查结论（statement）同时引用了几条来源；逐条单独核对时，下列片段（spans）没有被任何一条来源单独支持。
你的任务：判断每个片段能否由**这几条来源合在一起**直接确立。典型情况是对照或拼合：两份材料的转写、画面、数字逐项对得上或对不上（"两侧一致""B 比 A 多一段"），或几条来源分别给出的事实拼成该片段。
给你：statement、spans、sources（每条带 evidence_id、来源标注与全文；图片来源按 attached_images 的顺序附图）。
规则：
- 只依据来源内容，不使用外部知识。
- 只有把至少两条来源对照或拼合后能直接得出该片段时，才填 supported=true，并在 quotes 里给出用到的每条来源的原文：逐字摘录，图片来源写看到的内容；至少两条，来自不同 evidence_id。
- 只用一条来源就能得出的，填 false（那不属于本次复核）。
- 需要来源之外的推测或背景知识、来源之间相互矛盾、某条来源与片段相反、或拿不准，一律填 false，并在 note 说明。
输出 JSON（字符串里的双引号用“”）：
{"spans":[{"span":"原样照抄","supported":true|false,"quotes":[{"evidence_id":"E13","quote":"原文"}],"note":"一句话"}]}"""

JOINT_SCHEMA = _obj({"spans": {"type": "array", "items": _obj({
    "span": {"type": "string"}, "supported": {"type": "boolean"},
    "quotes": {"type": "array", "items": _obj({"evidence_id": {"type": "string"}, "quote": {"type": "string"}})},
    "note": {"type": "string"}})}})


def is_candidate(e: ErrorRecord) -> bool:
    """实锤"所引 E 均未支持这一部分"、引了两条以上不同的 E、还没被别的规则处理过。"""

    return (e.type is ErrorType.UNSUPPORTED_ASSERTION and e.severity in _FIRM and e.method == "model"
            and (e.detail or "").startswith(_UNSUPPORTED_DETAIL) and isinstance(e.span, str) and bool(e.span.strip())
            and len(set(e.evidence_ids)) >= 2 and not e.extra.get("program_rule") and not e.extra.get("check"))


def joint_texts(evidence_ids: list[str] | tuple[str, ...], lookup: EvidenceLookup) -> list[EvidenceText]:
    """可合并判断的来源：有正文的文本、或有像素的图片；按引用顺序去重。"""

    out: list[EvidenceText] = []
    for eid in dict.fromkeys(evidence_ids):
        if not lookup.has(eid):
            continue
        text = lookup.get(eid)
        if text.is_text or (text.is_image and text.image_path is not None):
            out.append(text)
    return out


def joint_payload(statement: str, spans: list[str], texts: list[EvidenceText]) -> tuple[dict[str, Any], list[Path]]:
    sources: list[dict[str, Any]] = []
    images: list[Path] = []
    budget = JOINT_MAX_CHARS_TOTAL
    for text in texts:
        src = _source_payload(text)
        if text.is_text:
            cap = max(2_000, min(JOINT_MAX_CHARS_PER_SOURCE, budget))
            if len(src["content"]) > cap:
                src = {**src, "content": src["content"][:cap], "content_truncated": True}
            budget -= len(src["content"])
        elif text.image_path is not None and len(images) < COVERAGE_MAX_IMAGES:
            images.append(text.image_path)
            src["attached_image_index"] = len(images)
        sources.append(src)
    return {"statement": statement, "spans": spans, "sources": sources}, images


def check_quotes(quotes: Any, texts: dict[str, EvidenceText]) -> dict[str, Any]:
    """程序回查：返回 {"ok": bool, "used": [E...], "quotes": [...], "why": str}。"""

    rows = []
    for q in quotes if isinstance(quotes, list) else []:
        if not isinstance(q, dict):
            continue
        eid, quote = str(q.get("evidence_id") or "").strip(), str(q.get("quote") or "").strip()
        if eid not in texts or not quote:
            continue
        text = texts[eid]
        verified = quote_in_source(quote, text.content) if text.is_text else None
        rows.append({"evidence_id": eid, "quote": quote[:300], "verified": verified})
    used = sorted({r["evidence_id"] for r in rows})
    if len(used) < 2:
        return {"ok": False, "used": used, "quotes": rows, "why": "引语不足两条不同来源"}
    if any(r["verified"] is False for r in rows):
        return {"ok": False, "used": used, "quotes": rows, "why": "有文本引语在来源里查不到"}
    if not any(r["verified"] is True for r in rows):
        return {"ok": False, "used": used, "quotes": rows, "why": "没有一条查得到的文本引语"}
    return {"ok": True, "used": used, "quotes": rows, "why": ""}


async def apply_joint_check(errors: list[ErrorRecord], statements: dict[str, str], lookup: EvidenceLookup, judge: JsonJudge,
                            *, key: str = "finding") -> list[ErrorRecord]:
    """对候选片段做多来源合并复核；命中的降为 minor。statements 是同一单元全部 Finding / 答案句（按 extra[key] 取）。"""

    if not JOINT_SOURCES["enabled"]:
        return errors
    groups: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for i, e in enumerate(errors):
        if is_candidate(e):
            sid = str(e.extra.get(key) or e.extra.get("finding"))
            groups.setdefault((sid, tuple(dict.fromkeys(e.evidence_ids))), []).append(i)
    if not groups:
        return errors
    out = list(errors)

    async def one(sid: str, eids: tuple[str, ...], idx: list[int]) -> None:
        texts = joint_texts(eids, lookup)
        statement = statements.get(sid, "")
        if len(texts) < 2 or not any(t.is_text for t in texts) or not statement:
            return
        spans = list(dict.fromkeys(str(out[i].span) for i in idx))
        payload, images = joint_payload(statement, spans, texts)
        raw, _failure = await _call_checked(judge, JOINT_SYSTEM_PROMPT, payload, images, JOINT_SCHEMA)
        if not isinstance(raw, dict):
            return                                         # 调用失败：保留原判
        by_span = {str(x.get("span") or "").strip(): x for x in raw.get("spans") or [] if isinstance(x, dict)}
        lookup_texts = {t.evidence_id: t for t in texts}
        for i in idx:
            e = out[i]
            item = by_span.get(str(e.span).strip())
            if item is None and len(spans) == 1 and len(by_span) == 1:
                item = next(iter(by_span.values()))       # 模型改了片段写法、但只问了一段
            if item is None:
                continue
            note = str(item.get("note") or "")[:200]
            checked = check_quotes(item.get("quotes"), lookup_texts)
            if item.get("supported") is True and checked["ok"]:
                out[i] = replace(e, severity=Severity.MINOR,
                                 detail=f"（多来源合并复核：{'+'.join(checked['used'])} 对照后可得出，仅备注）" + e.detail,
                                 extra={**e.extra, "program_rule": "joint_sources", "original_severity": e.severity.value,
                                        "joint": {"status": "supported", "used": checked["used"], "quotes": checked["quotes"],
                                                  "note": note}})
            else:
                status = "rejected_quotes" if item.get("supported") is True else "not_joint"
                out[i] = replace(e, extra={**e.extra, "joint": {"status": status, "why": checked["why"] if status == "rejected_quotes" else "",
                                                                "note": note}})

    await asyncio.gather(*(one(sid, eids, idx) for (sid, eids), idx in groups.items()))
    return out


__all__ = ["JOINT_SCHEMA", "JOINT_SOURCES", "JOINT_SYSTEM_PROMPT", "apply_joint_check", "check_quotes", "is_candidate", "joint_payload",
           "joint_texts"]
