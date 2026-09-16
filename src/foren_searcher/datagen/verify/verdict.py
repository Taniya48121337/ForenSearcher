"""核验结论 Verdict 与三层编排 verify_report()。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..cases import Case
from ..errors import SPECS, ErrorRecord, Severity, tally
from .anchored import JsonJudge, verify_anchored
from .context import EvidenceLookup
from .exec_checks import exec_checks
from .sufficiency import verify_sufficiency

Disposition = Literal["positive", "negative", "uncertain"]
VERIFIER_VERSION = "verifier@0.1.0"


@dataclass
class Verdict:
    subject: dict[str, Any]                       # {case_id, run, workspace_id, request, ...} 指向被核的那份报告
    errors: list[ErrorRecord] = field(default_factory=list)
    disposition: Disposition = "uncertain"
    tainted_findings: list[str] = field(default_factory=list)   # 含传播型错误的 Finding
    pending: list[dict[str, Any]] = field(default_factory=list)  # 未能判定的项（缺模型、缺像素）
    layers_run: list[str] = field(default_factory=list)
    judgments: list[dict[str, Any]] = field(default_factory=list)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    verifier: str = VERIFIER_VERSION
    judge_model: str | None = None
    spec_hash: str | None = None
    state_renderer: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        return tally(self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "disposition": self.disposition, "errors": [e.to_dict() for e in self.errors],
            "tally": {k: v for k, v in self.counts.items() if v}, "tainted_findings": self.tainted_findings,
            "pending": self.pending, "layers_run": self.layers_run, "judgments": self.judgments,
            "sufficiency": self.sufficiency, "verifier": self.verifier, "judge_model": self.judge_model,
            "spec_hash": self.spec_hash, "state_renderer": self.state_renderer,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Verdict":
        return cls(subject=dict(value["subject"]), errors=[ErrorRecord.from_dict(e) for e in value.get("errors", [])],
                   disposition=value.get("disposition", "uncertain"), tainted_findings=list(value.get("tainted_findings", [])),
                   pending=list(value.get("pending", [])), layers_run=list(value.get("layers_run", [])),
                   judgments=list(value.get("judgments", [])), sufficiency=dict(value.get("sufficiency", {})),
                   verifier=value.get("verifier", VERIFIER_VERSION), judge_model=value.get("judge_model"),
                   spec_hash=value.get("spec_hash"), state_renderer=value.get("state_renderer"))


def decide(errors: list[ErrorRecord], pending: list[dict[str, Any]]) -> Disposition:
    """处置规则：任一 critical 定错 → negative；≥2 个 major 定错 → negative；有未决项或仅有 flag → uncertain；否则 positive。"""

    firm = [e for e in errors if e.extra.get("confidence") != "flag"]
    if any(e.severity is Severity.CRITICAL for e in firm):
        return "negative"
    if sum(1 for e in firm if e.severity is Severity.MAJOR) >= 2:
        return "negative"
    if pending or any(e.extra.get("confidence") == "flag" for e in errors):
        return "uncertain"
    return "positive"


def tainted(errors: list[ErrorRecord]) -> list[str]:
    out: list[str] = []
    for e in errors:
        if e.extra.get("confidence") == "flag" or not SPECS[e.type].propagates:
            continue
        finding = e.extra.get("finding")
        if isinstance(finding, str) and finding not in out:
            out.append(finding)
    return out


async def verify_report(report: dict[str, Any], lookup: EvidenceLookup, *, subject: dict[str, Any], case: Case | None = None,
                        task_text: str | None = None, judge: JsonJudge | None = None, visible_ids: set[str] | None = None,
                        judge_model: str | None = None, spec_hash: str | None = None, state_renderer: str | None = None) -> Verdict:
    """三层顺序执行：exec → anchored → sufficiency。没有 judge 就只跑第一层，flag 留着，处置为 uncertain。"""

    verdict = Verdict(subject=dict(subject), judge_model=judge_model, spec_hash=spec_hash, state_renderer=state_renderer)
    first = exec_checks(report, lookup, case=case, visible_ids=visible_ids)
    verdict.layers_run.append("exec")
    errors: list[ErrorRecord] = list(first)
    if judge is not None:
        skip = {eid for e in first for eid in e.evidence_ids if e.type.value == "fabricated_evidence_id"}
        anchored = await verify_anchored(report, lookup, judge, skip_evidence_ids=skip)
        verdict.layers_run.append("anchored")
        judged = {j.finding_id for j in anchored.judgments}
        # 第二层看过的 Finding，第一层的 flag 由模型判定取代
        errors = [e for e in errors if not (e.extra.get("confidence") == "flag" and e.extra.get("finding") in judged)]
        errors.extend(anchored.errors)
        verdict.pending.extend(anchored.pending)
        verdict.judgments = [j.to_dict() for j in anchored.judgments]
        text = task_text if task_text is not None else (case.task_text if case else None)
        if text:
            sufficiency = await verify_sufficiency(report, text, judge, case=case)
            verdict.layers_run.append("sufficiency")
            # 第三层与第一层可能重复：状态级错误按类型去重，缺项按 (类型, key) 去重
            def dedup_key(e: ErrorRecord) -> tuple[Any, ...]:
                if SPECS[e.type].level.value == "status":
                    return (e.type,)
                return (e.type, e.extra.get("key") or e.extra.get("requirement"))

            seen = {dedup_key(e) for e in errors}
            for e in sufficiency.errors:
                if dedup_key(e) not in seen:
                    errors.append(e)
                    seen.add(dedup_key(e))
            verdict.sufficiency = sufficiency.raw
    else:
        verdict.pending.append({"reason": "no_judge", "note": "第二、三层未运行"})
    verdict.errors = errors
    verdict.tainted_findings = tainted(errors)
    verdict.disposition = decide(errors, verdict.pending)
    return verdict


def write_verdicts(path: str | Path, verdicts: list[Verdict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
    return target


def read_verdicts(path: str | Path) -> list[Verdict]:
    return [Verdict.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["VERIFIER_VERSION", "Disposition", "JsonJudge", "Verdict", "decide", "read_verdicts", "tainted", "verify_report", "write_verdicts"]
