"""回溯线索评分（ABSeeker 2608.05102 的思路，v2 §2.2）：从核实过的结论倒推"哪几步真正拿到了有用的证据"。

大白话：办完案子、报告核验完，我们知道哪些证据最后撑住了结论（U 集合）。再回头看每一步：这一步有没有新
拿到 U 里的证据（acquired）、有没有第一次读到 U 里证据的正文（observed）、翻的这一页里有没有落着核验器摘出的
引语（quote_page）。三样加起来就是这一步的 clue_gain。它不做入选门槛（探索步天然为 0），只做权重和 metadata。

U = 案卷 required_evidence 匹配到的 E ∪ 未被污染的 Finding 所引用的 E。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..cases import Case
from .verdict import Verdict

if TYPE_CHECKING:  # runlog 引用 verify.context，这里只作类型引用，避免循环导入
    from ..runlog import Decision, RunLog

_WS = re.compile(r"\s+")


@dataclass
class UsefulEvidence:
    ids: set[str] = field(default_factory=set)
    reasons: dict[str, list[str]] = field(default_factory=dict)          # E → ["required:prefpolice_tweets", "finding:F5"]
    required_keys: dict[str, set[str]] = field(default_factory=dict)     # required_evidence.key → {E}
    clean_findings: dict[str, dict[str, Any]] = field(default_factory=dict)   # F → {workspace, evidence_ids, statement}
    tainted_findings: dict[str, list[str]] = field(default_factory=dict)      # F → error types
    quotes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)     # E → [{finding_id, quote, index}]

    def add(self, eid: str, reason: str) -> None:
        self.ids.add(eid)
        self.reasons.setdefault(eid, [])
        if reason not in self.reasons[eid]:
            self.reasons[eid].append(reason)

    def requirements_of_evidence(self, case: Case, eid: str) -> set[str]:
        out: set[str] = set()
        for req in case.required_evidence:
            if eid in self.required_keys.get(req.key, set()):
                out.update(req.supports)
        return out

    def requirements_covered_by_finding(self, case: Case, finding_id: str) -> set[str]:
        row = self.clean_findings.get(finding_id)
        if not row:
            return set()
        out: set[str] = set()
        for eid in row["evidence_ids"]:
            out |= self.requirements_of_evidence(case, eid)
        return out

    def coverage_by_workspace(self, case: Case) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for fid, row in self.clean_findings.items():
            out.setdefault(row["workspace"], set()).update(self.requirements_covered_by_finding(case, fid))
        return out


def _norm(text: str) -> str:
    return _WS.sub("", text).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").lower()


def locate_quote(content: str, quote: str) -> int | None:
    """引语在原文里的字符位置（去空白、统一引号后匹配；回到原文坐标是近似值——按去空白前的比例换算）。"""

    if not content or not quote:
        return None
    n_content, n_quote = _norm(content), _norm(quote)
    if len(n_quote) < 6:
        return None
    idx = n_content.find(n_quote)
    if idx < 0:
        # 引语可能被核验模型轻微改写：退到前 24 个字符
        idx = n_content.find(n_quote[:24])
        if idx < 0:
            return None
    ratio = len(content) / max(1, len(n_content))
    return int(idx * ratio)


def build_useful_evidence(log: RunLog, case: Case | None, report_verdicts: list[Verdict]) -> UsefulEvidence:
    useful = UsefulEvidence()
    # 1) 案卷必需证据：扫全部 E
    if case is not None:
        for eid in _all_evidence_ids(log):
            text = log.lookup.get(eid)
            acquired_by = getattr(log.store.evidence(eid), "acquired_by", None)
            for req in case.required_evidence:
                if req.matches(source_url=text.source_url, acquired_by=acquired_by, label=text.label):
                    useful.required_keys.setdefault(req.key, set()).add(eid)
                    useful.add(eid, f"required:{req.key}")
    # 2) 未被污染的 Finding 所引的 E；同时收集引语
    for verdict in report_verdicts:
        workspace = str(verdict.subject.get("workspace_id"))
        tainted = set(verdict.tainted_findings)
        errors_by_finding: dict[str, list[str]] = {}
        for e in verdict.errors:
            flagged = e.extra.get("finding") if isinstance(e.extra, dict) else None
            if isinstance(flagged, str):
                errors_by_finding.setdefault(flagged, []).append(e.type.value)
        for rec in log.store.workspace_report_records():
            if rec.task_id != workspace:
                continue
            for finding in rec.to_dict().get("findings") or []:
                fid = str(finding.get("finding_id"))
                ids = [x for x in finding.get("evidence_ids") or [] if isinstance(x, str)]
                if fid in tainted:
                    useful.tainted_findings[fid] = errors_by_finding.get(fid, [])
                    continue
                useful.clean_findings[fid] = {"workspace": workspace, "evidence_ids": ids, "statement": finding.get("statement", "")}
                for eid in ids:
                    useful.add(eid, f"finding:{fid}")
        for j in verdict.judgments:
            jeid: Any = j.get("evidence_id")
            jfid = j.get("finding_id")
            if not isinstance(jeid, str) or jfid in tainted:
                continue
            for part in j.get("supported") or []:
                quote = part.get("quote") if isinstance(part, dict) else None
                if not isinstance(quote, str) or j.get("is_image"):
                    continue
                try:
                    content = log.lookup.get(jeid).content
                except KeyError:
                    continue
                index = locate_quote(content, quote)
                if index is not None:
                    useful.quotes.setdefault(jeid, []).append({"finding_id": jfid, "quote": quote, "index": index})
    return useful


def _all_evidence_ids(log: RunLog) -> list[str]:
    ids: set[str] = set()
    for e in log.events:
        ids.update(re.findall(r"\bE\d+\b", str(e.get("payload"))))
    return sorted((i for i in ids if log.lookup.has(i)), key=lambda x: int(x[1:]))


# --------------------------------------------------------------------------- 每一步的收益

def produced_evidence(decision: Decision) -> list[str]:
    """这步动作的观察里出现的 E（INSPECT 派生 / SEARCH 打开的页面）。"""

    obs = decision.observation or {}
    details = obs.get("details") or {}
    out: list[str] = []
    if obs.get("action") == "INSPECT":
        result = details.get("result") or {}
        if isinstance(result, dict):
            for frame in result.get("frames") or []:
                if isinstance(frame, dict) and isinstance(frame.get("evidence_id"), str):
                    out.append(frame["evidence_id"])
            for key in ("result_evidence_id", "evidence_id"):
                if isinstance(result.get(key), str):
                    out.append(result[key])
        if isinstance(details.get("result_evidence_id"), str):
            out.append(details["result_evidence_id"])
    elif obs.get("action") == "SEARCH":
        for result in details.get("results") or []:
            for material in (result or {}).get("materials") or []:
                for card in (material or {}).get("evidence") or []:
                    if isinstance(card, dict) and isinstance(card.get("evidence_id"), str):
                        out.append(card["evidence_id"])
    seen: list[str] = []
    for eid in out:
        if eid not in seen:
            seen.append(eid)
    return seen


def read_page(decision: Decision) -> tuple[str | None, int, int | None]:
    """READ_RECORD 这步读的是哪条 E 的哪一页：(E, offset, next_offset)。"""

    obs = decision.observation or {}
    details = obs.get("details") or {}
    eid: Any = details.get("evidence_id") or (decision.output.get("arguments") or {}).get("record_id")
    offset = int((decision.output.get("arguments") or {}).get("offset") or details.get("content_offset") or 0)
    nxt = details.get("next_offset")
    if nxt is None:
        m = re.search(r"next_offset=(\d+|None)", " ".join(str(i) for i in obs.get("items") or []))
        if m and m.group(1) != "None":
            nxt = int(m.group(1))
    return (eid if isinstance(eid, str) else None), offset, (int(nxt) if isinstance(nxt, (int, float)) else None)


def clue_gain(decision: Decision, useful: UsefulEvidence, first_seen: dict[str, tuple[str, int]]) -> dict[str, Any]:
    acquired = [e for e in produced_evidence(decision)
                if e in useful.ids and first_seen.get(e) == (decision.actor_id, decision.turn)]
    resurfaced = [e for e in produced_evidence(decision) if e in useful.ids and e not in acquired]
    observed: list[str] = []
    quote_pages: list[dict[str, Any]] = []
    if decision.action == "READ_RECORD":
        eid, offset, nxt = read_page(decision)
        if eid and eid in useful.ids and eid not in decision.content_observed_before:
            observed.append(eid)
        if eid:
            hits = [q for q in useful.quotes.get(eid, []) if q["index"] >= offset and (nxt is None or q["index"] < nxt)]
            if hits:
                quote_pages.append({"evidence_id": eid, "offset": offset, "next_offset": nxt,
                                    "quotes": [{"finding_id": q["finding_id"], "quote": q["quote"][:80]} for q in hits]})
    total = len(acquired) + len(observed) + len(quote_pages)
    return {"acquired": len(acquired), "acquired_ids": acquired, "resurfaced_ids": resurfaced,
            "observed": len(observed), "observed_ids": observed,
            "quote_page": len(quote_pages), "quote_pages": quote_pages, "total": total}


__all__ = ["UsefulEvidence", "build_useful_evidence", "clue_gain", "locate_quote", "produced_evidence", "read_page"]
