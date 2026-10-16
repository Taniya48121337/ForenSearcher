"""第三层：充分性。整份报告对照任务文字和案卷要求：任务提出的问题有没有被证据回答、状态报得对不对。

这一层看的是"缺了什么"，所以不能只看 Finding 本身；输入是任务全文 + 案卷要求 + 报告所引证据的索引 + 报告。
它的错误都是非传播型（omitted_required_fact / completed_without_finishing / incomplete_though_answerable）。

verifier@0.4.0（r1）：这一层不再拿 GT。以前给模型的"必需证据清单"里有"标注的最早来源：gettyimages…（2013-12-12）"这类
条目，r0g 的 46 份改写报告里 18 份的 missing 写的是"GT 里的来源没引"（e17 Getty 2013、e4 Getty 2018-02-01）——那是答案
对照，不是报告本身的缺口。现在给的是案卷要求（REQ + must_cite；VideoSham 的 REQ 原文带官方编辑描述，换成不含 GT 的说法）、
GT 无关的必需证据（标注来源 / 网址指定的参考来源不给）和证据索引（报告所引 E 的标题、取得方式、网址）。GT 只留给答案层的
wrong_label 等判定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..cases import Case
from ..errors import ErrorRecord, ErrorType, Severity, error
from .anchored import JsonJudge, _obj
from .context import EvidenceLookup

SUFFICIENCY_SCHEMA = _obj({
    "requirements": {"type": "array", "items": _obj({"key": {"type": "string"}, "covered": {"type": "boolean"},
                                                     "by": {"type": ["string", "null"]}, "note": {"type": "string"}})},
    "answerable_from_report": {"type": "boolean"},
    "status_should_be": {"type": "string", "enum": ["COMPLETED", "INCOMPLETE"]},
    "missing": {"type": "array", "items": {"type": "string"}},
    "limitations_not_gaps": {"type": "array", "items": {"type": "string"}},
})

SUFFICIENCY_SYSTEM_PROMPT = """你是报告审核员。给你一个局部调查任务（task）、案卷对这个任务的要求清单（requirements，可能为空）、
任务要取得的证据类型（required_evidence，可能为空）、报告所引证据的索引（evidence_index：编号、标题、取得方式、网址）和调查员提交的报告。
你只判断一件事：任务提出的问题，报告有没有用它引用的证据回答。你看不到"标准答案"（最早来源是哪一个、日期、地点、真伪），
也不要猜测；不要因为报告没找到某个你以为存在的来源而判缺口。
只依据报告里写出来的内容判断，不补充外部知识，也不评价文笔。输出 JSON：
{"requirements":[{"key":"清单里的 key（REQ1、dates…）或你从任务文字里归纳的要求（短语）","covered":true|false,"by":"回答它的 finding 编号或 null","note":"一句话"}],
 "answerable_from_report":true|false,   // 仅凭报告中的 Finding 是否已能回答该局部任务
 "status_should_be":"COMPLETED|INCOMPLETE",
 "missing":["任务明确提出、报告却没有用证据回答的问题（用英文写，一条一句）"],
 "limitations_not_gaps":["报告的局限，但不构成缺口的点（用英文写），见下面的规则"]}
规则：
- requirements 清单里 must_cite=true 的要求，要有 Finding 引用证据回答了它才算 covered；required_evidence 里 must_cite=true 的项，
  要有 Finding 引用了这类证据才算 covered。
- key 里用 | 分隔的一项是可互相替代的几份证据，报告取得其中任一份就算这一项 covered，不要把其余几份当作缺项。
- requirements 里的 key 请原样照抄清单里的 key，不要加括号说明。
- 任务文字里明确要求的每一点（例如"说明来源""给出时间区间""核验原站"）都要列成一条 requirement。
- covered 只看问题有没有被证据回答，不看结论是否与你预期的一致。
- 报告写了 limitations 说明为何取不到，不等于 covered，但可以让 status_should_be 是 INCOMPLETE。
- missing 用英文写，只写任务问了而报告没回答的问题；不要写清单、任务文字和报告里都没有出现的来源名、网址、日期、地名。
- 候选页打不开、没读到（failed / not examined）本身不是缺口：只要已读的页面里有证据回答了这条要求，"N 个候选页没检查"
  只写进 limitations_not_gaps，不进 missing，也不因此判 INCOMPLETE；只有没有任何已读页面回答这条要求时，才是这条要求没被回答
  （missing 写没回答的问题本身，不写"某某页打不开"）。
- 精度以任务文字和清单为准：任务没有明确要求的精度——每个对应段帧精确的起止边界、每一段都配成对的抽帧、差异图、媒体属性
  （时长 / 帧率 / 音轨数）——报告给出了时间区间或采样对应点、并在 limitations 里说明了精度的，写进 limitations_not_gaps，不算缺口。
  任务或清单明确要求了的（例如"给出两侧的时间区间"），没给仍是缺口。
- 反过来，任务问的核心问题没有回答仍是缺口。下面三类出现时，对应的那条要求记 covered=false、写进 missing，status_should_be
  =INCOMPLETE——报告在 limitations 里承认了也一样，承认不等于回答：
  (1) 声明说的是视频画面里发生的事（某个动作、事件、物体是否出现：助推器被机械臂夹住、有人被抢、有人在抽大麻），报告却没有看
      画面帧、没说画面里有没有这件事——只有屏幕文字（OCR）或语音转写不算回答了"画面里有什么"；任务要求说明视频说了什么，
      报告既没有转写也没有屏幕文字，同样算。声明说的是视频之外的事实或视频的说法（选举结果、政策、某人宣布了什么、观众反响），
      屏幕文字 / 转写已说明视频说了什么的，不要因为没看画面帧判缺口；
  (2) 报告依据某个网页上的图下结论（最早副本、原始语境、拍摄地），却没有把那张图与给定图片比对、确认是同一张；
  (3) 任务问声明说的地点对不对，报告没有把给定图片与该地点的参考照片比对。
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


def sufficiency_payload(report: dict[str, Any], task_text: str, *, case: Case | None = None,
                        requirement_ids: set[str] | None = None, lookup: EvidenceLookup | None = None) -> dict[str, Any]:
    """交给充分性模型的 payload（verifier@0.4.0）：不含 GT。"""

    scoped = case.scoped_required_evidence(requirement_ids) if case else ()
    return {"task": task_text, "requirements": _requirement_entries(case, requirement_ids),
            "required_evidence": _required_entries(case, scoped), "evidence_index": _evidence_index(report, lookup),
            "report": _report_payload(report)}


async def verify_sufficiency(report: dict[str, Any], task_text: str, judge: JsonJudge, *, case: Case | None = None,
                             requirement_ids: set[str] | None = None, lookup: EvidenceLookup | None = None) -> SufficiencyResult:
    """requirement_ids：本工作区对准的要求；给了就只把对准这些要求的案卷要求与必需证据交给模型（verifier@0.3.0 起按工作区收窄）。
    lookup：给了就在 payload 里附报告所引 E 的索引（标题 / 取得方式 / 网址）；没给（walker 循环内）只列编号。"""

    payload = sufficiency_payload(report, task_text, case=case, requirement_ids=requirement_ids, lookup=lookup)
    required = list(payload["required_evidence"]) + list(payload["requirements"])
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
    req_keys = {r["key"] for r in payload["requirements"]}
    for item in uncovered:
        key = str(item.get("key") or "?")
        if key not in case_keys:
            continue      # 模型自己从任务文字归纳的要求只留在 raw 里作参考；试点上它把“说明来源与适用范围”这类套话也算成缺项，噪音太大
        if key in req_keys:
            continue      # verifier@0.4.0：案卷要求（REQ）只用来判状态；缺项错误仍按必需证据记，免得同一缺口按 REQ 和证据各记一条 major
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
    only_failed_pages = _drop_failed_pages(raw, result)
    if status == "COMPLETED" and (missing_must or (should == "INCOMPLETE" and not only_failed_pages)):
        keys = ", ".join(str(r.get("key")) for r in (missing_must or uncovered))
        detail = f"报 COMPLETED 但任务要求未全部覆盖：{keys}" if keys else "报 COMPLETED 但审核认为应为 INCOMPLETE"
        result.errors.append(error(ErrorType.COMPLETED_WITHOUT_FINISHING, detail, method="model", extra={"status_should_be": should}))
    if status == "INCOMPLETE" and bool(raw.get("answerable_from_report")) and not uncovered and should == "COMPLETED":
        result.errors.append(error(ErrorType.INCOMPLETE_THOUGH_ANSWERABLE, "报告内容已足以回答任务，却报 INCOMPLETE",
                                   method="model", extra={"status_should_be": should}))
    return result


# verifier@0.4.0：只说"N 个候选页打不开 / 没检查（所以可能还有更早的副本）"的 missing 条目是 limitation，不是缺口。
# r0g 的例子："2 of 6 candidate pages failed to load and were not examined"、"Examination of the 8 candidate pages that failed
# to load"、"3 of 8 candidate pages were not examined, so an earlier copy may exist."；和别的缺口写在同一句的不算（"Only one
# external source was used. 3 of 5 candidate pages failed …"）。
_FAILED_PAGES = re.compile(
    r"(?:the\s+)?(?:examination\s+of\s+(?:the\s+)?)?(?:\d+\s*(?:of|/)\s*\d+\s+|\d+\s+|all\s+|several\s+|some\s+|other\s+|remaining\s+)*"
    r"(?:other\s+|remaining\s+)?candidate\s+(?:external\s+|web\s+)?(?:pages?|sources?|links?|urls?)"
    r"(?:\s+that)?(?:\s+(?:failed\s+to\s+(?:load|open)|could\s+not\s+be\s+(?:opened|loaded|read|retrieved)|did\s+not\s+(?:load|open)"
    r"|(?:were|was)\s+(?:never\s+|not\s+)?(?:examined|opened|read|checked|retried|loaded|inspected|reviewed)))?"
    r"(?:,?\s+and\s+(?:were|was)\s+(?:not|never)\s+(?:examined|checked|retried|read|opened|inspected))?", re.I)
_FAILED_PAGES_ZH = re.compile(r"(?:\d+\s*(?:/|个中的?|中的?)\s*)?\d*\s*个?候选(?:网)?页(?:面)?[^；;。]*?(?:打不开|未能打开|加载失败|未检查|未查看|没有检查|未读)")
_CONSEQUENCE = re.compile(r",?\s*(?:so|and|therefore)\s+(?:an?\s+|the\s+)?(?:earlier|other|older|further|additional)\b[^.;]*?"
                          r"(?:may|might|could)\s+(?:exist|be\s+missed|remain)[^.;]*", re.I)
_FILLER = re.compile(r"[\W\d_]+|\b(?:and|the|were|was|not|never|examined|retried|of|so|thus|therefore|pages?|candidates?)\b", re.I)


def failed_pages_only(item: str) -> bool:
    """missing 条目是否只是在说"候选页打不开 / 没检查"（后面最多跟"所以可能还有更早的副本"）。"""

    text = str(item or "")
    if not (_FAILED_PAGES.search(text) and re.search(r"fail|not\s+(?:be\s+)?(?:examined|opened|loaded|checked|read|retried)|never|did\s+not", text, re.I)) \
            and not _FAILED_PAGES_ZH.search(text):
        return False
    rest = _FAILED_PAGES_ZH.sub(" ", _CONSEQUENCE.sub(" ", _FAILED_PAGES.sub(" ", text)))
    rest = re.sub(r"[，。；、（）]|所以|因此|可能|还有|更早|副本|存在|未检查|未查看|没有检查|未读|打不开", "", rest)
    return len(_FILLER.sub("", rest)) <= 2


def _drop_failed_pages(raw: dict[str, Any], result: "SufficiencyResult") -> bool:
    """只讲 failed 页的 missing 条目移到 raw["missing_failed_pages"]（不进 status_reason）。返回 True = missing 原本非空且全是这种条目，
    此时模型的 status_should_be=INCOMPLETE 不单独构成 completed_without_finishing（必需要求没覆盖的照样判）。"""

    missing = [str(m) for m in raw.get("missing") or [] if str(m).strip()]
    dropped = [m for m in missing if failed_pages_only(m)]
    if dropped:
        result.raw["missing"] = [m for m in missing if m not in dropped]
        result.raw["missing_failed_pages"] = dropped
    return bool(missing) and len(dropped) == len(missing)


def _required_entries(case: Case | None, scoped: Any) -> list[dict[str, Any]]:
    """必需证据清单：any_one_of 组合成一项（key 用 | 连接、描述写明任一即可），其余逐项（verifier@0.3.6）。

    此前组内每个 key 单列、各自标 must_cite，模型会把没取到的替代来源也当缺项（true_20361105 的 freebeacon、
    VideoSham 加了 diff_images 之后 4126 W1-2 的"未覆盖 diff_images"）。"""

    by_key = {r.key: r for r in scoped if not is_reference_entry(r)}
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


def is_reference_entry(entry: Any) -> bool:
    """按网址指定、不按取得方式匹配的必需证据 = 标注 / 指定的参考来源（"标注的最早来源：gettyimages…（2013-12-12）"、
    "NOAA/climate.gov 关于全球海平面上升的说明"）。它们是答案对照，不进充分性层（verifier@0.4.0）；第一层照常按 url_pattern 核。"""

    return bool(getattr(entry, "url_pattern", None)) and not getattr(entry, "acquired_by", None)


_OFFICIAL = re.compile(r"official|官方", re.I)
GT_FREE_EDIT_REQUIREMENT = ("Whether A contains an edit relative to B that the footage can confirm: what the difference is, and where it "
                            "is on both sides (time position, and screen region if visual)")


def _requirement_entries(case: Case | None, requirement_ids: set[str] | None) -> list[dict[str, Any]]:
    """案卷要求（REQ）：英文用 text_en。VideoSham 的 REQ1 原文带官方编辑描述（"Replace the sound with an accordion sound
    00:00-00:10"、"replaced with lorem ipsum"），那是 GT，换成不含描述的同义要求。"""

    out: list[dict[str, Any]] = []
    for r in (case.requirements if case is not None else ()):
        if requirement_ids is not None and r.id not in requirement_ids:
            continue
        text = str(r.text_en or r.text)
        if _OFFICIAL.search(text) or _OFFICIAL.search(r.text):
            text = GT_FREE_EDIT_REQUIREMENT
        out.append({"key": r.id, "requirement": text, "must_cite": r.must_cite})
    return out


def _evidence_index(report: dict[str, Any], lookup: EvidenceLookup | None) -> list[dict[str, Any]]:
    """报告所引 E 的索引（按首次出现顺序）：学生看得到的标题、取得方式、网址；没有 lookup 时只列编号。"""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in report.get("findings") or []:
        for eid in (finding.get("evidence_ids") or []) if isinstance(finding, dict) else []:
            if not isinstance(eid, str) or eid in seen:
                continue
            seen.add(eid)
            item: dict[str, Any] = {"evidence_id": eid}
            if lookup is not None and lookup.has(eid):
                text = lookup.get(eid)
                item.update({"label": (text.label or "")[:200], "acquired_by": text.acquired_by, "source_url": text.source_url,
                             "modalities": list(text.modalities)})
            out.append(item)
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


__all__ = ["GT_FREE_EDIT_REQUIREMENT", "SUFFICIENCY_SYSTEM_PROMPT", "SufficiencyResult", "is_reference_entry", "failed_pages_only", "sufficiency_payload", "verify_sufficiency"]
