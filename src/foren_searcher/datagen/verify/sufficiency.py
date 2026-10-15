"""第三层：充分性。整份报告对照任务文字和必需证据清单：该拿的拿齐没有、状态报得对不对。

这一层看的是"缺了什么"，所以不能只看 Finding 本身；输入是任务全文 + 案卷的必需证据清单 + 报告。
它的错误都是非传播型（omitted_required_fact / completed_without_finishing / incomplete_though_answerable）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity, error
from .anchored import JsonJudge, _obj

SUFFICIENCY_SCHEMA = _obj({
    "requirements": {"type": "array", "items": _obj({"key": {"type": "string"}, "covered": {"type": "boolean"},
                                                     "by": {"type": ["string", "null"]}, "note": {"type": "string"}})},
    "answerable_from_report": {"type": "boolean"},
    "status_should_be": {"type": "string", "enum": ["COMPLETED", "INCOMPLETE"]},
    "missing": {"type": "array", "items": {"type": "string"}},
})

SUFFICIENCY_SYSTEM_PROMPT = """你是报告审核员。给你一个局部调查任务、任务要求取得的证据清单（可能为空）和调查员提交的报告。
只依据报告里写出来的内容判断，不补充外部知识，也不评价文笔。输出 JSON：
{"requirements":[{"key":"清单里的 key 或你从任务文字里归纳的要求（短语）","covered":true|false,"by":"支持它的 finding 编号或 null","note":"一句话"}],
 "answerable_from_report":true|false,   // 仅凭报告中的 Finding 是否已能回答该局部任务
 "status_should_be":"COMPLETED|INCOMPLETE",
 "missing":["报告没有回答、而任务明确要求的点"]}
规则：
- 清单里 must_cite=true 的项，报告必须有 Finding 引用了对应证据才算 covered。
- key 里用 | 分隔的一项是可互相替代的几份证据，报告取得其中任一份就算这一项 covered，不要把其余几份当作缺项。
- requirements 里的 key 请原样照抄清单里的 key，不要加括号说明。
- 任务文字里明确要求的每一点（例如"说明来源""给出时间区间""核验原站"）都要列成一条 requirement。
- 报告写了 limitations 说明为何取不到，不等于 covered，但可以让 status_should_be 是 INCOMPLETE。
- 保守：拿不准就 covered=false，并在 note 里说明。"""


@dataclass
class SufficiencyResult:
    errors: list[ErrorRecord] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors], "raw": self.raw}


def _report_payload(report: dict[str, Any]) -> dict[str, Any]:
    findings = []
    for index, finding in enumerate(report.get("findings") or []):
        if isinstance(finding, dict):
            findings.append({"finding_id": finding.get("finding_id") or f"finding[{index}]",
                             "statement": finding.get("statement"), "evidence_ids": list(finding.get("evidence_ids") or []),
                             "limitations": list(finding.get("limitations") or [])})
    return {"status": report.get("status"), "status_reason": report.get("status_reason"), "findings": findings}


async def verify_sufficiency(report: dict[str, Any], task_text: str, judge: JsonJudge, *, case: Case | None = None,
                             requirement_ids: set[str] | None = None) -> SufficiencyResult:
    """requirement_ids：本工作区对准的要求；给了就只把对准这些要求的必需证据交给模型（verifier@0.3.0，整案清单不再压给单个工作区）。"""

    scoped = case.scoped_required_evidence(requirement_ids) if case else ()
    required = _required_entries(case, scoped)
    payload = {"task": task_text, "required_evidence": required, "report": _report_payload(report)}
    try:
        raw = await judge.judge(system_prompt=SUFFICIENCY_SYSTEM_PROMPT, payload=payload, images=[], schema=SUFFICIENCY_SCHEMA)
    except Exception:  # noqa: BLE001 —— 坏输出：这一层记空，不影响前两层
        raw = {"_failed": True}
    result = SufficiencyResult(raw={**raw, "scope": sorted(requirement_ids) if requirement_ids is not None else None,
                                    "required_keys": [r["key"] for r in required]})
    status = report.get("status")
    entry_of = _entry_lookup(required)
    requirements = []
    for r in raw.get("requirements") or []:
        if isinstance(r, dict):
            r = dict(r)
            r["key"] = entry_of.get(_norm_key(r.get("key")), str(r.get("key") or "?"))   # 模型加了括号说明 / 只写了组里一个 key 时归一到清单项
            requirements.append(r)
    must_keys = {r["key"] for r in required if r["must_cite"]}
    case_keys = {r["key"] for r in required}
    # verifier@0.3.6：any_one_of 组在清单里合成一项（key 用 | 连接），组内任一取得即 covered；同一项被模型列多次时任一 covered 即算
    covered_keys = {str(r.get("key")) for r in requirements if r.get("covered")}
    optional_keys = {r["key"] for r in required if not r["must_cite"]}   # verifier@0.3.1：整组 must_cite=False 只记 minor 备注
    uncovered, listed = [], set()
    for r in requirements:
        k = str(r.get("key"))
        if not r.get("covered") and k not in covered_keys and k not in listed:
            uncovered.append(r)
            listed.add(k)
    for item in uncovered:
        key = str(item.get("key") or "?")
        if key not in case_keys:
            continue      # 模型自己从任务文字归纳的要求只留在 raw 里作参考；试点上它把“说明来源与适用范围”这类套话也算成缺项，噪音太大
        members = key.split("|")
        extra = {"requirement": members[0], "from_case": True, **({"keys": members} if len(members) > 1 else {})}
        if key in optional_keys:
            result.errors.append(error(ErrorType.OMITTED_REQUIRED_FACT, f"可选证据 {key!r} 未覆盖（仅备注）：{item.get('note') or ''}".strip("："),
                                       method="model", severity=Severity.MINOR, extra={**extra, "optional": True}))
            continue
        result.errors.append(error(ErrorType.OMITTED_REQUIRED_FACT, f"任务要求 {key!r} 未被回答：{item.get('note') or ''}".strip("："),
                                   method="model", extra=extra))
    missing_must = [r for r in uncovered if str(r.get("key")) in must_keys]
    should = raw.get("status_should_be")
    if status == "COMPLETED" and (missing_must or should == "INCOMPLETE"):
        keys = ", ".join(str(r.get("key")) for r in (missing_must or uncovered))
        detail = f"报 COMPLETED 但任务要求未全部覆盖：{keys}" if keys else "报 COMPLETED 但审核认为应为 INCOMPLETE"
        result.errors.append(error(ErrorType.COMPLETED_WITHOUT_FINISHING, detail, method="model", extra={"status_should_be": should}))
    if status == "INCOMPLETE" and bool(raw.get("answerable_from_report")) and not uncovered and should == "COMPLETED":
        result.errors.append(error(ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, "报告内容已足以回答任务，却报 INCOMPLETE",
                                   method="model", extra={"status_should_be": should}))
    return result


def _required_entries(case: Case | None, scoped: Any) -> list[dict[str, Any]]:
    """必需证据清单：any_one_of 组合成一项（key 用 | 连接、描述写明任一即可），其余逐项（verifier@0.3.6）。

    此前组内每个 key 单列、各自标 must_cite，模型会把没取到的替代来源也当缺项（true_20361105 的 freebeacon、
    VideoSham 加了 diff_images 之后 4126 W1-2 的"未覆盖 diff_images"）。"""

    by_key = {r.key: r for r in scoped}
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    groups = case.evidence_groups() if case is not None else [("all", (k,)) for k in by_key]
    for mode, group in groups:
        keys = [k for k in group if k in by_key and k not in used]
        if not keys:
            continue
        used.update(keys)
        if mode == "any_one_of" and len(keys) > 1:
            out.append({"key": "|".join(keys), "description": "以下任一即可：" + "；".join(f"{k}（{by_key[k].description}）" for k in keys),
                        "must_cite": any(by_key[k].must_cite for k in keys)})
        else:
            out += [{"key": k, "description": by_key[k].description, "must_cite": by_key[k].must_cite} for k in keys]
    out += [{"key": k, "description": r.description, "must_cite": r.must_cite} for k, r in by_key.items() if k not in used]
    return out


def _norm_key(key: Any) -> str:
    return re.split(r"[（(]", str(key or ""), maxsplit=1)[0].strip()


def _entry_lookup(required: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in required:
        out[r["key"]] = r["key"]
        for member in r["key"].split("|"):
            out.setdefault(member, r["key"])
    return out


__all__ = ["SUFFICIENCY_SYSTEM_PROMPT", "SufficiencyResult", "verify_sufficiency"]
