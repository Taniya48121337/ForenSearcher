"""verifier@0.4.0 第 5 处：sloppy_numeric 的日期容差（r0g2 金标 F001 / F013）。

报告照抄 dates 卡（SEARCH/dates："published date declared by the page: 2024-01-26"）写"该页声明的发布日期是 2024-01-26"，
核验模型拿页面正文里另一个日期（"Posted on 08/Feb/2023"、"Petition created on 31 January 2021"）判"与该部分相反"、记
sloppy_numeric。句子说的是页面自己声明的那个日期，数没有抄错：正文差一天多是时区；差得多是页面本身有两个日期，该写进
limitations，不是这句话的数字错误。

规则 declared_date（只作用于第二层文本来源的反驳类错误，不调模型）：被反驳的部分（或它所在的句子）写明是页面声明的日期，
部分里的日期与同句所引 dates 卡的支持片段一致 → 降为 minor；与正文日期差 ≤ 1 天记 timezone，其余记 page_has_other_date。
B 族的时间戳（秒）不是日历日期，不受影响。
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from typing import Any

from ..errors import ErrorRecord, ErrorType, Severity
from .lang import dates_in
from .pixel_rules import sentence_of

_CONTRA = (ErrorType.SLOPPY_NUMERIC, ErrorType.WRONG_CITATION)
_DECLARED = re.compile(r"\bdeclar\w*|\bpage\s+(?:says|states|lists|gives|shows)\s+(?:it\s+was\s+)?(?:published|posted)"
                       r"|\bself-reported\b|\bstated\s+publication|页面声明|页面自报|自报|声明的发布", re.I)
_DATES_CARD = re.compile(r"declared by the page|^dates of\b|页面声明", re.I)
_SLASH_DATE = re.compile(r"\b(\d{1,2})/([A-Za-z]{3,9})/(\d{4})\b")
_MON = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _dates(text: str) -> set[str]:
    out = set(dates_in(text or ""))
    for d, mon, y in _SLASH_DATE.findall(text or ""):
        if mon[:3].lower() in _MON:
            out.add(f"{int(y):04d}-{_MON[mon[:3].lower()]:02d}-{int(d):02d}")
    return out


def _gap_days(a: set[str], b: set[str]) -> int | None:
    best = None
    for x in a:
        for y in b:
            try:
                gap = abs((date.fromisoformat(x) - date.fromisoformat(y)).days)
            except ValueError:
                continue
            best = gap if best is None else min(best, gap)
    return best


def _card_dates(judgments: list[Any], exclude: str) -> set[str]:
    """同句所引 dates 卡（标题 "dates of …" 或引语 "declared by the page"）已核实的支持引语里的日期。"""

    out: set[str] = set()
    for j in judgments:
        if j.evidence_id == exclude or j.is_image:
            continue
        card = bool(_DATES_CARD.search(str(getattr(j, "source_label", "") or "")))
        for p in j.supported:
            if p.verified is False:
                continue
            if card or _DATES_CARD.search(p.quote or ""):
                out |= _dates(p.quote)               # 只认来源原文（已回查的引语），不认报告自己的措辞
    return out


def apply_date_rule(errors: list[ErrorRecord], statement: str, judgments: list[Any]) -> list[ErrorRecord]:
    by_id = {j.evidence_id: j for j in judgments}
    out: list[ErrorRecord] = []
    for e in errors:
        j = by_id.get(e.evidence_ids[0]) if len(e.evidence_ids) == 1 else None
        if (j is None or j.is_image or e.method != "model" or e.type not in _CONTRA or e.severity is Severity.MINOR
                or not isinstance(e.span, str)):
            out.append(e)
            continue
        part = e.span
        sentence = sentence_of(statement, part)
        stated = _dates(part) or _dates(sentence)
        if not (stated and (_DECLARED.search(part) or _DECLARED.search(sentence))):
            out.append(e)
            continue
        card = _card_dates(judgments, j.evidence_id)
        if not stated & card:
            out.append(e)
            continue
        gap = _gap_days(stated & card, _dates(str(e.extra.get("quote") or e.detail)))
        why = "timezone" if gap is not None and gap <= 1 else "page_has_other_date"
        note = ("写明是页面声明的日期、与 dates 卡一致，正文日期只差一天（时区）" if why == "timezone"
                else "写明是页面声明的日期、与 dates 卡一致；页面正文另有日期，应写进 limitations")
        out.append(replace(e, severity=Severity.MINOR, detail=f"（程序规则：{note}，仅备注）" + e.detail,
                           extra={**e.extra, "program_rule": "declared_date", "date_rule": why, "gap_days": gap,
                                  "original_severity": e.severity.value}))
    return out


__all__ = ["apply_date_rule"]
