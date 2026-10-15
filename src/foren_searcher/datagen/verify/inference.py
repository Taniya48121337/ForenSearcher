"""第二层的"推理核对"（候选，默认关闭；2026-10-04）。

大白话：第二层只拿一句话对它自己引的来源。综合判断句的依据常在同一份报告 / 答案的其他句子里：
"该声明混淆了两根铁柱"靠的是前面"德里铁柱是笈多时期建的"和"照片里是 Bharatpur 的铁柱"两句；
"即位宣告与加冕不能混为一谈"靠的是前面两个不同日期。这类句子被判"无支持（critical）"是结构性误判。

这里对第二层仍判为实锤"无支持"的片段再问一次：
- 它能不能**只由**同一单元里已核实的句子（附来源引语）直接推出、不引入新事实 → derivable；
- 它是不是句子引述后加以反驳 / 否定的说法，而不是句子自己的断言 → mentioned；
- 否则 new_fact，原判不动。
命中前两种就把该错误降为 minor（留痕：extra.program_rule、premises、original_severity）。

防放水：
- 前提只取同单元**没有实锤错误**、且至少有一条来源支持片段的句子（加上本句已被支持的部分）；
- 片段里出现前提和本句已支持部分都没有的数字 / 拉丁字母专名，不问模型，直接保留原判；
- 模型拿不准时按提示词规定回 new_fact。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from typing import Any

from ..errors import ErrorRecord, ErrorType, Severity
from .anchored import JsonJudge, _call_checked, _obj, candidate_enabled
from .lang import REFUTE, is_cjk, proper_nouns

INFERENCE_CHECK = {"enabled": candidate_enabled("infer")}      # verifier@0.3.6 起默认打开
_FIRM = (Severity.MAJOR, Severity.CRITICAL)

INFERENCE_SYSTEM_PROMPT = """你是事实核查审核员。一份调查报告（或最终答案）里有一句话，其中某个片段没有被它自己引用的来源直接支持。
给你：
- statement：这句话全文；span：没被支持的片段；
- own_supported：这句话里已被它自己的来源支持的部分（附来源原文引语）；
- premises：同一份报告 / 答案里其他已经核实的句子，每句附来源原文引语。

判断 span 属于哪种情况：
1. derivable：span 只是由 own_supported 和 premises 直接推出的结论——组合两个已核实事实、比较日期先后、排除、归纳成总判断（如"该声明错误""混淆了两件事""两者不能混为一谈"）——没有引入任何前提里没有的新事实（新的人物、地点、机构、数字、日期、事件、因果关系、引语）。
2. mentioned：span 是这句话**引述后加以反驳、否定或排除**的说法（例如"反驳了'X 被献祭'的解释"里的"X 被献祭"），这句话并没有断言它为真。
3. new_fact：推出它还需要前提里没有的事实；或推理跳跃、需要背景知识；或拿不准。

只依据给出的材料，不用你自己的背景知识补全。拿不准一律 new_fact。
输出 JSON：{"status": "derivable" | "mentioned" | "new_fact", "premises": ["用到的句子编号"], "missing": "若 new_fact，缺的是什么事实；否则空串", "note": "一句话理由"}"""

INFERENCE_SCHEMA = _obj({"status": {"type": "string", "enum": ["derivable", "mentioned", "new_fact"]},
                         "premises": {"type": "array", "items": {"type": "string"}},
                         "missing": {"type": "string"}, "note": {"type": "string"}})

_NUM = re.compile(r"\d+(?:\.\d+)?")
_LATIN = re.compile(r"[A-Za-z][A-Za-z.'\-]{1,}")
_QUOTED = re.compile(r"[“\"「『]([^”\"」』]{2,})[”\"」』]")
_REFUTE = REFUTE                 # 10-07：中英双语（lang.py）


def _parts_by_statement(judgments: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for j in judgments:
        if j.get("verdict") != "supported":
            continue
        for p in j.get("supported") or []:
            if p.get("verified") is False:
                continue                      # 引语回查失败的不当前提
            out.setdefault(str(j.get("finding_id")), []).append(
                {"evidence_id": str(j.get("evidence_id")), "part": str(p.get("part") or "")[:200], "quote": str(p.get("quote") or "")[:300]})
    return out


def quoted_mention(span: str, statement: str) -> bool:
    """程序先判：片段整段落在句中引号里，且句子带反驳 / 否定提示词 → 被引述的说法，不是断言。"""

    if not span or not statement or not _REFUTE.search(statement):
        return False
    core = span.strip("“”\"「」『』 ")
    return any(core and core in q for q in _QUOTED.findall(statement))


def new_tokens(span: str, context: str) -> list[str]:
    """片段里出现、而前提与本句已支持部分里都没有的数字 / 拉丁字母专名（大小写不敏感）。"""

    low = context.lower()
    out = [n for n in _NUM.findall(span) if n not in context]
    # 中文片段里的拉丁字母词就是专名；纯英文片段（10-07）只看句中大写词，否则每个普通英文词都成了"新专名"
    latin = _LATIN.findall(span) if is_cjk(span) else proper_nouns(span)
    out += [w for w in latin if w.lower().strip(".'-") not in low]
    return out


async def apply_inference_check(errors: list[ErrorRecord], statements: dict[str, str], judgments: list[dict[str, Any]],
                                judge: JsonJudge, *, key: str = "finding") -> list[ErrorRecord]:
    """对实锤"无支持"片段做推理核对；命中 derivable / mentioned 的降为 minor。statements 是同一单元全部 Finding / 答案句。"""

    if not INFERENCE_CHECK["enabled"]:
        return errors
    firm_ids = {str(e.extra.get(key) or e.extra.get("finding")) for e in errors if e.severity in _FIRM}
    parts = _parts_by_statement(judgments)

    def candidate(e: ErrorRecord) -> bool:
        return (e.type is ErrorType.UNSUPPORTED_ASSERTION and e.severity in _FIRM and e.method == "model"
                and isinstance(e.span, str) and bool(e.span.strip()) and not e.extra.get("program_rule") and not e.extra.get("check"))

    async def check(e: ErrorRecord) -> ErrorRecord:
        sid = str(e.extra.get(key) or e.extra.get("finding"))
        statement = statements.get(sid, "")
        if quoted_mention(e.span, statement):
            return _downgrade(e, "mentioned", [], "片段在句中引号内，且句子是在反驳 / 否定它")
        premises = [{"id": f, "statement": t, "sources": parts[f][:3]} for f, t in statements.items()
                    if f != sid and f not in firm_ids and parts.get(f)]
        own = parts.get(sid, [])[:4]
        context = " ".join([statement.replace(e.span, " "), *(p["statement"] for p in premises),
                            *(x["part"] + " " + x["quote"] for p in premises for x in p["sources"]),
                            *(x["part"] + " " + x["quote"] for x in own)])
        extra_tokens = new_tokens(e.span, context)
        if not premises and not own:
            return e
        if extra_tokens:
            return replace(e, extra={**e.extra, "inference": {"status": "new_tokens", "tokens": extra_tokens[:6]}})
        payload = {"statement": statement, "span": e.span, "own_supported": own, "premises": premises[:24]}
        raw, _failure = await _call_checked(judge, INFERENCE_SYSTEM_PROMPT, payload, [], INFERENCE_SCHEMA)
        if not isinstance(raw, dict):
            return e                                    # 调用失败：保留原判
        status = raw.get("status")
        used = [str(x) for x in raw.get("premises") or []]
        if status == "derivable" and not ({*used} & ({p["id"] for p in premises} | {sid})):
            status = "new_fact"                         # 说可推出却指不出用了哪句：不算
        if status in ("derivable", "mentioned"):
            return _downgrade(e, status, used, str(raw.get("note") or ""))
        return replace(e, extra={**e.extra, "inference": {"status": "new_fact", "missing": str(raw.get("missing") or "")[:200]}})

    targets = [i for i, e in enumerate(errors) if candidate(e)]
    if not targets:
        return errors
    checked = await asyncio.gather(*(check(errors[i]) for i in targets))
    out = list(errors)
    for i, e in zip(targets, checked):
        out[i] = e
    return out


def _downgrade(e: ErrorRecord, status: str, premises: list[str], note: str) -> ErrorRecord:
    label = {"derivable": "由同单元已核实的句子直接推出", "mentioned": "是句子引述后加以反驳的说法，不是断言"}[status]
    return replace(e, severity=Severity.MINOR, detail=f"（推理核对：{label}，仅备注）" + e.detail,
                   extra={**e.extra, "program_rule": "inference_" + status, "original_severity": e.severity.value,
                          "premises": premises[:8], "inference": {"status": status, "note": note[:200]}})


__all__ = ["INFERENCE_CHECK", "INFERENCE_SCHEMA", "INFERENCE_SYSTEM_PROMPT", "apply_inference_check", "new_tokens", "quoted_mention"]
