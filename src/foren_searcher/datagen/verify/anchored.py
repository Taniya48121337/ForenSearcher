"""第二层：证据锚定的模型核验（v2：按"部分"判定，再按 Finding 汇总覆盖）。

第一版对每条 (Finding, E) 只给一个三态，规则"只支持一部分算 not_found"在真实报告上失灵：报告里的句子几乎
都是复合句（"41 秒可见餐盒；48 秒女子举盒；51.48 秒食物不可见"），每条 E（一帧、一页）天然只支持其中一部分，
结果 85% 的判定是 not_found、92% 的报告被判负。这不是报告的问题，是核验器的问题。

现在分两步：
1. 逐对 (Finding, E)：核验模型列出这条 E **支持的部分**和**反驳的部分**，每一部分都带原文引语（文本来源
   程序回查引语；图片来源引语是它看到的画面描述）。句子里讲别的材料的部分，这条 E 管不着，不算反驳。
2. 逐 Finding 汇总：把所有 E 支持的部分（引语已核实）交给模型，问"句子里还有哪些实质部分没有任何来源支持"。
   有 → unsupported_assertion，span 是那一部分而不是整句；有被反驳的部分 → wrong_citation（或按提示改记
   misread_fine_text / sloppy_numeric）；转述写成事实 → attribution_drift；以偏概全提示 → scope_overreach。

核验模型只看"这句话"和"这条 E 的全文/像素"，不看任务、不看演员桌面（CaRR / Rulers 的做法）。
模型调用失败（限流、瞬时错误、坏 JSON）指数退避重试（见 CALL_ATTEMPTS），仍失败就记 pending（带尝试次数与最后错误），不让整个 run 报废。

并发：同一 Finding 的各对 (Finding, E) 与不同 Finding 之间都用 asyncio.gather 并发，所有核验模型调用经 `_call`
共用一个并发上限（默认 4，环境变量 FOREN_JUDGE_CONCURRENCY 覆盖）；judge_coverage 等该 Finding 的所有 pair 完成后才跑。
结果（judgments / pending / coverage / errors）按 Finding 顺序、evidence_ids 顺序拼回，与串行时逐字一致。
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import random
import re
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from ..errors import ErrorRecord, ErrorType, Severity, error
from .context import EvidenceLookup, EvidenceText
from .lang import (BACKREF, CONCRETE_ZH, DATE_NUMERIC, LIMITATION, TASK_WORDS, VERDICT, dates_in, grams, has_concrete,
                   strip_dates)
from .date_rules import apply_date_rule
from .pixel_rules import apply_pixel_rules

MAX_SOURCE_CHARS = 120_000   # 第一次全量 20K 把 85K 字符的 PDF 章节截掉了大半，表格数字全判成无支持；27b 有 262K 上下文，放宽

PAIR_SYSTEM_PROMPT = """你是证据核对员。只依据给出的"来源"（全文或图片）核对"待核句"，不使用任何外部知识。
待核句通常是复合句，这条来源可能只覆盖其中一部分；你的任务是说清楚它覆盖了哪些部分、怎么覆盖的。
输出一个 JSON 对象（所有字符串里的双引号用中文引号“”代替，避免破坏 JSON）：
{"supported_parts":[{"part":"待核句中被这条来源支持的那一部分（尽量逐字摘自待核句）","quote":"来源中支持它的原文片段（文本来源必须逐字连续摘录，不改写不翻译；图片来源写出你实际看到的具体内容）"}],
 "contradicted_parts":[{"part":"待核句中被这条来源反驳的那一部分","quote":"来源中与之相反的原文/画面","hint":null|"misread_fine_text"|"sloppy_numeric"}],
 "source_reports_claim":true|false,   // 来源只是转述别人的说法（“据称”“某人表示”），不是自身断言
 "statement_asserts_fact":true|false, // 待核句把这些内容当作事实断言，而不是写成“来源称”
 "scope_overreach":true|false,        // 待核句用这条来源（少数帧/局部片段）推断了整体或连续区间
 "notes":"一句话"}
规则：
- 只判断这条来源能覆盖的内容。待核句里讲别的材料、别的时刻、别的页面的部分，不写进 supported 也不写进 contradicted。
- contradicted 只在来源与待核句关于同一对象的说法相反（不同数字、不同日期、不同主体、相反的画面）时使用；
  来源“没提到”不是反驳。来源是视频某一帧时，只能反驳关于这一帧/这个时刻的说法。
- 图片来源：只报告能直接看到的内容；看不清的细小文字不要猜，不放进 supported。
- 数字口径/单位不一致填 hint=sloppy_numeric；图片细字读错填 misread_fine_text。
- 把这条来源能支持的**每一个**部分都列出来（时间、数字、地点、人物、谁说了什么、画面内容各算一部分），不要只列最明显的一个；
  漏列的部分后面会被当成"没有支持"。
- 两个列表都可以为空。"""

# 来源归属规则。只在 payload 带 provenance 时用带这条规则的提示词（PAIR_SYSTEM_PROMPT）；
# 外部网页等不带 provenance 的对仍用旧提示词（_PAIR_SYSTEM_PROMPT_LEGACY，与改动前逐字相同）——
# CachedJudge 的 key 含 system_prompt，统一换提示词会让全部文本对的缓存失效。
_PROVENANCE_RULE = ("- 来源标注了属于哪份材料（A/B）时，待核句里关于**另一份材料**的部分既不写进 supported 也不写进 contradicted；"
                    "只核关于本材料的部分。\n")
_PAIR_SYSTEM_PROMPT_LEGACY = PAIR_SYSTEM_PROMPT
PAIR_SYSTEM_PROMPT = PAIR_SYSTEM_PROMPT.replace(
    "- 只判断这条来源能覆盖的内容。待核句里讲别的材料、别的时刻、别的页面的部分，不写进 supported 也不写进 contradicted。\n",
    "- 只判断这条来源能覆盖的内容。待核句里讲别的材料、别的时刻、别的页面的部分，不写进 supported 也不写进 contradicted。\n"
    + _PROVENANCE_RULE)
# verifier@0.3.3（r0c 发现）：图片对带两样新东西时才换提示词，其余对照旧（缓存继续命中）——
# - media_note：帧的取帧时间、单帧时长、视频总长（"结尾帧""视频共 6 秒"以前因核验模型不知道总长被判无支持）；
# - task：本报告所接的调查任务，只用来理解"材料A""给定视频""该人物"指什么（4126 里核验模型把两段视频的指代弄反）。
_MEDIA_RULE = ("- 图片来源带 media_note（程序记录的取帧时间、单帧时长、所属视频总长）时，待核句里关于这一帧在第几秒、"
               "是不是第一帧 / 最后一帧、视频总长的部分可以据此判断：写进 supported，quote 写 media_note 里的相应内容。\n")
_TASK_RULE = ("- payload 带 task（这份报告所接的调查任务）时，只用它理解待核句里的指代（“材料A”“给定视频”“该人物”分别指什么）；"
              "task 里的说法不是证据，不能据此支持或反驳任何部分。\n")
# verifier@0.4.0（r0g2 金标 F006 / F009 / F012 / F007）：图片只反驳对画面内容的直接描述；一句引了几张图时只核说这张图的部分
_IMAGE_RULE = ("- 来源是图片时：待核句里归因给工具 / 比对结果的测量值（“the comparison reports 100% …”“audio matches in 9 of 9 windows”"
               "“similarity 0.97”）不是对画面的描述，图片不能反驳它，不写进 contradicted；图片只能反驳待核句对画面内容的直接描述。\n")
_OTHERS_RULE = ("- payload 带 other_cited_images（同一句还引用了的其他图：编号、标题、取帧时间）时：先按待核句里的时间、区域、图类型"
                "（整帧对照面板 / 差异图 / 裁剪图 / 单帧 / 给定图片 / 外部网页上的参考图）判断哪些部分说的是这张图；说的是另一张图的部分"
                "既不写进 supported 也不写进 contradicted。\n")


def pair_system_prompt(*, provenance: bool, media_note: bool = False, task: bool = False, image: bool = False,
                       others: bool = False) -> str:
    """按 payload 实际带的字段选提示词：没有新字段时与旧版逐字相同（保住缓存）。image / others 是 verifier@0.4.0 的图片规则。"""

    prompt = PAIR_SYSTEM_PROMPT if provenance else _PAIR_SYSTEM_PROMPT_LEGACY
    extra = ((_MEDIA_RULE if media_note else "") + (_TASK_RULE if task else "") + (_IMAGE_RULE if image else "")
             + (_OTHERS_RULE if others else ""))
    if not extra:
        return prompt
    return prompt.replace("- 两个列表都可以为空。", extra + "- 两个列表都可以为空。")


PAIR_SYSTEM_PROMPTS = tuple(dict.fromkeys(
    pair_system_prompt(provenance=p, media_note=m, task=t, image=i, others=o) for i in (False, True) for o in (False, True)
    for m in (False, True) for t in (False, True) for p in (True, False)))
# 识别"这是一次 pair 调用"时用（假核验模型、统计）；前两项仍是 PAIR_SYSTEM_PROMPT / 旧提示词

COVERAGE_SYSTEM_PROMPT = """你是报告审核员。给你一句调查结论（待核句）和它引用的各条来源已经核实的支持片段。
判断待核句里还有哪些**实质性的可核部分**没有任何来源支持。输出 JSON（字符串里的双引号用“”）：
{"unsupported_parts":["待核句中没有任何支持片段覆盖的实质部分，尽量逐字摘自待核句"],
 "grounded":true|false,   // 待核句的全部实质内容是否都被支持片段覆盖
 "notes":"一句话"}
规则：
- 实质部分 = 可以被证据证实或证伪的事实断言（时间、数字、地点、谁说了什么、画面里有什么）。
- 以下不算实质部分，不要列：限定语和保守措辞（“约”“尚不能确定”“未核验”）、对已支持内容的合理归纳、
  由已支持的数字直接算出的结果、对来源类型的描述（“报道”“帧”）。
- 支持片段已经由程序核实确实出现在来源里，请信任它们；不要因为片段短就认为不支持。
- 拿不准的部分列出来，但 grounded 只在确实缺少支持时才为 false。"""
# 同理：只在支持片段带来源材料标注时用加了这句的提示词，其余沿用旧提示词保住缓存。
_COVERAGE_SYSTEM_PROMPT_LEGACY = COVERAGE_SYSTEM_PROMPT
COVERAGE_SYSTEM_PROMPT = COVERAGE_SYSTEM_PROMPT + (
    "\n- 支持片段旁标注了来源材料；一句话同时讲 A 和 B 时，A 的部分由 A 的来源支持、B 的部分由 B 的来源支持即算覆盖。")
# verifier@0.3.3：一句话引了两张以上图片时，汇总也附上这些图片——"前后两帧画面相同""B 比 A 多出一段"这类跨帧比较
# 不在任何单帧的支持片段里，只看片段必然判无支持。只有附图时才换提示词。
_COVERAGE_IMAGE_RULE = ("\n- 附了图片时（attached_images 按顺序对应附图，带来源材料与取帧时间）：待核句里的跨帧比较"
                        "（前后变化、两段画面是否相同、是否重复、先后顺序）可以直接看附图判断；附图能看出的部分不算没有支持，看不出的照常列出。")


# verifier@0.3.4（2026-10-02，四层精确率实测发现）：
#   1. 限制句：答案 / 报告里交代调查自身范围的句子（"画面直接核查仅覆盖上述四个时刻""未核验作者"）被汇总判成无据断言、
#      记 critical；Opus 整条重判不认、Sonnet 重采样也不稳。只在待核句命中 _LIMITATION 时加 _COVERAGE_LIMITATION_RULE
#      （其余句子提示词不变、缓存照常命中）；规则要求句中夹带的具体事实照常核对——召回守门实验专门测"限制语包着编造事实"。
#   2. 实锤复核：第一遍产生 major/critical 的模型层错误时，同一核验模型独立再判一遍（payload 加 resample，缓存键随之不同）；
#      第二遍没有复现同组错误的，降为 minor 并记 extra.unconfirmed（不计入处置）。r0c 上 23 个实锤单元原样重采样只复现 18 个。
FEATURES: dict[str, bool] = {"limitation_rule": True, "confirm_firm": True}
# 候选程序规则（2026-10-04，金标 v1p2_hard 复核前默认关闭；人工标注确认后再开，届时版本记 verifier@0.3.6）。
# 都只作用于模型层"所引 E 均未支持这一部分"的 critical/major，命中就降为 minor 并在 extra 里记规则名；不调模型。
#   task_given       片段只是复述任务给出的信息（声明原文、声明日期）——392 W1-2、660 A8；
#   back_reference   片段回指本报告/本答案已由其他干净 Finding 建立的内容（"上述差异""已确认的局部对应"）——0069、2010、H020/H050/H051；
#   verdict_sentence 答案里对声明下的总结论（"该声明错误""该声明有证据支持……"），依据在同一答案的其他干净句子里——699、4124903。
def candidate_enabled(name: str) -> bool:
    """verifier@0.3.6 起三项默认打开（10-05 金标采纳后）。环境变量 FOREN_VERIFIER_CANDIDATES 可只开其中几项：
    逗号分隔的 rules / infer / semcov；设成 none 全关（= 0.3.5 行为）。"""

    raw = os.environ.get("FOREN_VERIFIER_CANDIDATES")
    return True if raw is None else name in {x.strip() for x in raw.split(",")}


PROGRAM_RULES: dict[str, bool] = {k: candidate_enabled("rules") for k in ("task_given", "back_reference", "verdict_sentence")}

_LIMITATION = LIMITATION          # 10-07：中英双语（lang.py），中文部分与以前逐字相同
# 程序兜底（0.3.4）：覆盖汇总列出的"无支持部分"本身就是限制语、且不夹带具体事实时，降为 minor（extra.limitation_part）。
# "具体事实"按保守口径：含数字、拉丁字母词、或转折连词（"尚不能确定，但警方已逮捕嫌疑人"这类包装）一律不降。
# 英文（10-07）：拉丁字母词不能再一律算"具体事实"，改看数字、数词、转折连词与句中大写专名（lang.has_concrete）
_CONCRETE = CONCRETE_ZH


def is_limitation_part(part: str) -> bool:
    return bool(_LIMITATION.search(part or "")) and not has_concrete(part or "")
_COVERAGE_LIMITATION_RULE = (
    "\n- 待核句里交代调查自身范围或局限的部分（“仅核查了……”“未能取得……”“不能据此断言……”“尚不能确定……”）不是关于来源内容的事实断言，"
    "不列入 unsupported_parts；但这些部分里夹带的具体事实（数字、时间、地点、谁说了什么、画面里有什么）照常核对，没有支持就列出。")


def is_limitation_statement(statement: str) -> bool:
    return bool(_LIMITATION.search(statement or ""))


# verifier@0.4.0（r0g2 金标 F002 / F003）："matches the title's assertion of a Trump victory""the title's stated February 18 posting
# date" 复述的是任务原文（标题、posted 日期），没有哪条 E 会载有它，汇总一律判"所引 E 均未支持"。现在汇总带 task，复述任务原文的部分以它为据。
_COVERAGE_TASK_RULE = ("\n- payload 带 task（学生拿到的调查任务原文：声明 / 标题原文、声明或发布日期、给定材料的说明）时，待核句里复述任务原文"
                       "所给信息的部分（“标题说……”“标题所称的 2 月 18 日发布日期”“声明说……”）以 task 为依据，算有支持，不列入 unsupported_parts；"
                       "但对这些说法真伪的判断、拿它与证据比较得出的结论、以及任务里没有的事实，照常要有支持片段。")


def coverage_system_prompt(*, provenance: bool, images: bool = False, limitation: bool = False, task: bool = False) -> str:
    prompt = COVERAGE_SYSTEM_PROMPT if provenance else _COVERAGE_SYSTEM_PROMPT_LEGACY
    prompt = prompt + _COVERAGE_IMAGE_RULE if images else prompt
    prompt = prompt + _COVERAGE_LIMITATION_RULE if limitation else prompt
    return prompt + _COVERAGE_TASK_RULE if task else prompt


COVERAGE_SYSTEM_PROMPTS = tuple(coverage_system_prompt(provenance=p, images=i, limitation=l, task=k)
                                for k in (False, True) for l in (False, True) for i in (False, True) for p in (True, False))
COVERAGE_MAX_IMAGES = 8
MAX_TASK_CHARS = 1_500


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    """OpenRouter strict json_schema：所有字段必填、不允许多余键。"""

    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_PART = _obj({"part": {"type": "string"}, "quote": {"type": "string"}})
_CPART = _obj({"part": {"type": "string"}, "quote": {"type": "string"},
               "hint": {"type": ["string", "null"], "enum": ["misread_fine_text", "sloppy_numeric", None]}})
PAIR_SCHEMA = _obj({
    "supported_parts": {"type": "array", "items": _PART},
    "contradicted_parts": {"type": "array", "items": _CPART},
    "source_reports_claim": {"type": "boolean"}, "statement_asserts_fact": {"type": "boolean"},
    "scope_overreach": {"type": "boolean"}, "notes": {"type": "string"},
})
COVERAGE_SCHEMA = _obj({"unsupported_parts": {"type": "array", "items": {"type": "string"}},
                        "grounded": {"type": "boolean"}, "notes": {"type": "string"}})


DEFAULT_CONCURRENCY = 4


def judge_concurrency() -> int:
    """并发上限：FOREN_JUDGE_CONCURRENCY（≥1），缺省或写错时用 4。"""

    try:
        return max(1, int(os.environ.get("FOREN_JUDGE_CONCURRENCY", DEFAULT_CONCURRENCY)))
    except ValueError:
        return DEFAULT_CONCURRENCY


# 模块级信号量。asyncio.Semaphore 一旦在某个事件循环里发生等待就绑定到那个循环，而 CLI 每个 run 各调一次
# asyncio.run（新循环），所以按事件循环各建一个：同一循环内所有调用共用同一个上限，循环结束后自动回收。
_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = weakref.WeakKeyDictionary()


def judge_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = judge_concurrency()
    entry = _SEMAPHORES.get(loop)
    if entry is None or entry[0] != limit:
        entry = (limit, asyncio.Semaphore(limit))
        _SEMAPHORES[loop] = entry
    return entry[1]


class JsonJudge(Protocol):
    """核验模型接口：给 system + payload（+ 可选图片、+ 可选输出 schema），返回 JSON 对象。与教师模型分离。"""

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path],
                    schema: dict[str, Any] | None = None) -> dict[str, Any]: ...


@dataclass
class Part:
    part: str
    quote: str
    verified: bool | None = None      # None = 图片来源，无法程序回查
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"part": self.part, "quote": self.quote, "verified": self.verified, "hint": self.hint}


@dataclass
class Judgment:
    """一对 (Finding, E) 的判定。verdict 是汇总：supported（有核实的支持部分）/ contradicted / not_found / needs_pixels / failed。"""

    finding_id: str
    evidence_id: str
    verdict: str
    supported: list[Part] = field(default_factory=list)
    contradicted: list[Part] = field(default_factory=list)
    rejected: list[Part] = field(default_factory=list)      # 模型给了但引语对不上原文、没采信的部分（排查用）
    source_reports_claim: bool = False
    statement_asserts_fact: bool = True
    scope_overreach: bool = False
    is_image: bool = False
    notes: str = ""
    source_label: str = ""            # 来源标签 + URL，汇总时让模型知道"E14 就是 WFAA 那篇报道"
    provenance: str = ""              # 来源归属（材料 A/B 或外部网页），仅当它进了 pair payload 时才有值；汇总时带给模型
    media_note: str = ""              # 图片帧的时间说明（verifier@0.3.3）；汇总附图时带给模型
    image_path: Path | None = None    # 图片来源的像素文件；汇总附图用，不进 to_dict
    raw: dict[str, Any] = field(default_factory=dict)
    call_error: dict[str, str] = field(default_factory=dict)   # verdict=failed 时的异常（并发下不能靠全局 LAST_CALL_ERROR）；不进 to_dict
    call_failure: CallFailure | None = None                    # verdict=failed 时的完整失败记录（尝试次数、最后错误）；不进 to_dict

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, "evidence_id": self.evidence_id, "verdict": self.verdict, "source": self.source_label,
                "supported": [p.to_dict() for p in self.supported], "contradicted": [p.to_dict() for p in self.contradicted],
                "rejected": [p.to_dict() for p in self.rejected], "source_reports_claim": self.source_reports_claim, "statement_asserts_fact": self.statement_asserts_fact,
                "scope_overreach": self.scope_overreach, "is_image": self.is_image, "notes": self.notes}


@dataclass
class AnchoredResult:
    errors: list[ErrorRecord] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)     # 每条 Finding 的汇总结果
    pending: list[dict[str, Any]] = field(default_factory=list)      # 需要看像素 / 模型输出坏掉 的项

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors], "judgments": [j.to_dict() for j in self.judgments],
                "coverage": list(self.coverage), "pending": list(self.pending)}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").lower()


_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…+|\[\s*(?:\.{3}|…)\s*\])\s*")
_QUOTE_MARKS = "\"'“”‘’「」『』"


def _token_in(token: str, content: str) -> bool:
    """短引语（OCR 的 "ISSF" "2024" "34%"）：在原文里作为独立的词出现（前后不是字母数字），不分大小写。"""

    token = token.strip().strip(_QUOTE_MARKS).strip()
    if len(token) < 2:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", content, re.I) is not None


def quote_in_source(quote: str, content: str) -> bool:
    """引语回查：去掉所有空白、统一引号后做子串匹配（引语 ≥ 6 字符）；不是子串时允许按顺序的公共片段合计 ≥ 85%（容忍省略号、标点小改）。

    verifier@0.4.0（r0g2 金标 F008 / F011）：核验模型摘 OCR 结果常写成 "KAMALA ... HARRIS ... TRUMP"、或只摘一个短词 "ISSF"，
    以前前者按顺序公共片段不到 85%、后者不足 6 字符，一律不采信，汇总时这些部分就成了"所引 E 均未支持"。现在：
    引语里有省略号时按省略号分段，每段都在原文里（短段按独立的词找）就算；整条引语不足 6 字符时按独立的词找。"""

    pieces = [x for x in _ELLIPSIS.split(quote or "") if x.strip().strip(_QUOTE_MARKS).strip(" ,;:")]
    if len(pieces) >= 2:
        c = _squash(content)
        return all((len(_squash(x)) >= 6 and _squash(x) in c) or _token_in(x.strip(" ,;:"), content) for x in pieces)
    q = _squash(quote)
    if len(q) < 6:
        return _token_in(quote or "", content)
    c = _squash(content)
    if q in c:
        return True
    if len(q) > 400:
        q = q[:400]
    matcher = difflib.SequenceMatcher(None, c, q, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks() if block.size >= 4)   # 只算 ≥4 字符的连续片段，防碎片凑数
    return matched >= 0.85 * len(q)


def _payload_provenance(text: EvidenceText) -> str:
    """这条来源进 payload 的 provenance（空串 = 不进 payload）。

    成本权衡：CachedJudge 的 key 含 payload，多一个字段就让该对缓存失效。
    - 图片来源：一律带（媒体对照里 A/B 帧交叉引用被误判反驳的正是它们；图片对缓存本来就因 content 句改动失效）。
    - 文本来源：只有派生证据（derived_from 非空，如媒体属性、align_media 结果）才带；外部网页等非派生文本不带，
      它们的 payload 与改动前逐字相同，缓存继续命中（配合 _PAIR_SYSTEM_PROMPT_LEGACY）。
    """

    if not text.provenance:
        return ""
    if text.is_text and not text.derived_from:
        return ""
    return text.provenance


_NEEDLE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,60})[\"“”'‘’]|\b[A-Z][A-Za-z0-9%.&-]{2,}\b|\b\d[\d.,:%-]*\b")
FOCUS_WINDOW = 600


def focused_content(content: str, statement: str, limit: int = MAX_SOURCE_CHARS) -> tuple[str, bool]:
    """超长来源（r0g 有 12 条 OCR 记录 120K–633K 字符）不再只截前 limit 字：前一半照给开头（OCR 的汇总、网页正文开头），
    后一半给待核句里的词（引号里的串、大写词、数字）在其余部分出现处前后各 FOCUS_WINDOW 字，中间用"（中略）"隔开。
    返回 (内容, 是否有删节)。引语回查仍对全文做。"""

    if len(content) <= limit:
        return content, False
    head = content[: limit // 2]
    rest = content[limit // 2:]
    needles = []
    for m in _NEEDLE.finditer(statement or ""):
        word = (m.group(1) or m.group(0)).strip()
        if len(word) >= 2 and word not in needles:
            needles.append(word)
    spans: list[list[int]] = []
    for word in needles:
        for m in re.finditer(re.escape(word), rest, re.I):
            spans.append([max(0, m.start() - FOCUS_WINDOW), min(len(rest), m.end() + FOCUS_WINDOW)])
            if len(spans) > 400:
                break
    spans.sort()
    merged: list[list[int]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append(s)
    budget = limit - len(head)
    pieces = []
    for start, end in merged:
        if budget <= 0:
            break
        piece = rest[start:min(end, start + budget)]
        pieces.append(piece)
        budget -= len(piece)
    gap = "\n…（中略）…\n"
    return head + (gap + gap.join(pieces) if pieces else "") + gap, True


def _source_payload(text: EvidenceText, statement: str = "") -> dict[str, Any]:
    content, truncated = focused_content(text.content, statement)
    provenance = _payload_provenance(text)
    if text.is_text:
        shown = content
    elif provenance:
        shown = f"（图片来源：请看附图；{provenance}）"
    else:
        shown = "（图片来源：请看附图）"
    payload = {"evidence_id": text.evidence_id, "modalities": list(text.modalities), "label": text.label,
               "derived_from": list(text.derived_from), "source_url": text.source_url,
               "content": shown, "content_truncated": truncated}
    if provenance:
        payload["provenance"] = provenance
    if text.is_image and text.media_note:
        payload["media_note"] = text.media_note
    return payload


LAST_CALL_ERROR: dict[str, str] = {}      # 最近一次失败的异常（类型: 消息）；并发时只作兼容，本模块用 _call_checked 的返回值

# 重试（09-25）：本机 4 路并发跑召回校准时，大量 pair 调用失败（限流 / 瞬时错误）→ 整条 Finding 跳过 coverage、只记 pending，
# append_clause 召回从 60/60 掉到 19/60。现在对任何异常都指数退避重试，最多 CALL_ATTEMPTS 次。
# 与 OpenRouterClient 的分工：client.complete() 自己接住 429（按 Retry-After / x-ratelimit-reset 等 _retry_at，最多 2 次重发、
# 累计等待 ≤180s），用尽才抛 OpenRouterRateLimitError——这里不再叠第二层等待，直接记失败；BudgetExceeded、HTTP 400/401/402/403/404
# 这类确定性拒绝也不重试。client 没接住的（5xx、ReadTimeout 等网络错误、HTTP 200 里的 error、截断、坏 JSON）由这里退避重试。
# 坏输出（坏 JSON / 非 dict / 模型输出不合规）也算一次尝试，但同一次调用里最多坏 BAD_OUTPUT_ATTEMPTS 次（同一问题连坏两次，再问多半还坏），
# 且坏输出后立即重问、不退避（不是限流问题；也保持旧行为：坏 JSON 连坏两次即 failed，tests/test_datagen_verify.py 依赖这一点）。
CALL_ATTEMPTS = 4
BAD_OUTPUT_ATTEMPTS = 2
BACKOFF_SECONDS = (1.0, 3.0, 9.0)          # 第 n 次失败后等 BACKOFF_SECONDS[n-1] + U(0, 1) 秒
_NO_RETRY = ("OpenRouterRateLimitError", "BudgetExceeded")
_NO_RETRY_STATUS = {400, 401, 402, 403, 404}
_BAD_OUTPUT = ("ValueError", "PolicyOutputError", "OpenRouterOutputError")


class JudgeBudgetExhausted(RuntimeError):
    """OpenRouter 预算用尽（client 抛 BudgetExceeded）：不重试、不记 pending，直接向上抛，由 CLI 落盘已完成产物后以退出码 3 结束。"""


_BUDGET_EXHAUSTED = "BudgetExceeded"
BUDGET_EXIT_CODE = 3


def budget_message(exc: object) -> str:
    """CLI 预算用尽时打在 stdout 的醒目一行（scripts/complete_verify_local.py 按退出码 3 给同一句提示）。"""

    return (f"!!! OpenRouter 预算用尽：{exc}。请调高 .local\\openrouter.json 的 budget_usd 后重跑本步，"
            "已判过的对会命中缓存。")


def _names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _retryable(exc: BaseException) -> bool:
    if _names(exc) & set(_NO_RETRY):
        return False
    return getattr(exc, "status_code", None) not in _NO_RETRY_STATUS


def _is_bad_output(exc: BaseException) -> bool:
    return bool(_names(exc) & set(_BAD_OUTPUT))


def _backoff(failures: int) -> float:
    base = BACKOFF_SECONDS[min(failures, len(BACKOFF_SECONDS)) - 1]
    return base + random.random()


@dataclass
class CallFailure:
    """一次核验调用（含重试）的失败记录：各类异常的最后消息、实际尝试次数、最后一次的错误。"""

    errors: dict[str, str] = field(default_factory=dict)
    attempts: int = 0
    last_error: str = ""

    def pending_fields(self) -> dict[str, Any]:
        return {"error": dict(self.errors), "attempts": self.attempts, "last_error": self.last_error}


async def _call_checked(judge: JsonJudge, system_prompt: str, payload: dict[str, Any], images: list[Path],
                        schema: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, CallFailure]:
    """调核验模型（每次尝试受并发上限约束，退避等待时不占并发名额）；异常退避重试（坏输出立即重问），最多 CALL_ATTEMPTS 次。
    成功返回 (dict, 空 CallFailure)；最终失败返回 (None, CallFailure)。预算用尽（BudgetExceeded）抛 JudgeBudgetExhausted。"""

    failure = CallFailure()
    bad_outputs = 0
    while failure.attempts < CALL_ATTEMPTS:
        failure.attempts += 1
        retry = True
        async with judge_semaphore():
            try:
                raw = await judge.judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)
            except Exception as exc:  # noqa: BLE001 —— 模型输出坏掉 / 限流不应让整个 run 报废
                if _BUDGET_EXHAUSTED in _names(exc):      # 按类名匹配，不硬依赖 openrouter 模块
                    raise JudgeBudgetExhausted(str(exc)) from exc
                name, message = type(exc).__name__, str(exc)[:300]
                retry = _retryable(exc)
                bad_output = _is_bad_output(exc)
            else:
                if isinstance(raw, dict):
                    return raw, CallFailure()
                name, message = "not_a_dict", str(raw)[:300]
                bad_output = True
        bad_outputs += bad_output
        failure.errors[name] = message
        failure.last_error = f"{name}: {message}"
        if not retry or bad_outputs >= BAD_OUTPUT_ATTEMPTS or failure.attempts >= CALL_ATTEMPTS:
            break
        if not bad_output:
            await asyncio.sleep(_backoff(failure.attempts))
    return None, failure


async def _call(judge: JsonJudge, system_prompt: str, payload: dict[str, Any], images: list[Path],
                schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """兼容旧接口（plan_judged 用）：失败时把异常写进 LAST_CALL_ERROR。"""

    raw, failure = await _call_checked(judge, system_prompt, payload, images, schema)
    LAST_CALL_ERROR.clear()
    LAST_CALL_ERROR.update(failure.errors)
    return raw


def _parts(items: Any, text: EvidenceText) -> list[Part]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not part:
            continue
        verified: bool | None = quote_in_source(quote, text.content) if text.is_text else None
        hint = item.get("hint") if item.get("hint") in ("misread_fine_text", "sloppy_numeric") else None
        out.append(Part(part=part, quote=quote, verified=verified, hint=hint))
    return out


def _wants_task(text: EvidenceText) -> bool:
    """只有来自输入材料的图片（视频帧、页面图）带任务上下文：它们的待核句最常用"材料A""给定视频"指代。"""

    return text.is_image and text.provenance.startswith("来源材料")


def _other_images(text: EvidenceText, others: list[EvidenceText] | None) -> list[dict[str, Any]]:
    """同一句引用的其他图（verifier@0.4.0）：编号、标题、取帧时间，让核验模型分清句子哪部分说的是这张图。"""

    out = []
    for other in others or []:
        if other.evidence_id == text.evidence_id or not other.is_image:
            continue
        item: dict[str, Any] = {"evidence_id": other.evidence_id, "label": (other.label or "")[:160]}
        if other.media_note:
            item["media_note"] = other.media_note
        out.append(item)
    return out


async def judge_pair(judge: JsonJudge, finding_id: str, statement: str, text: EvidenceText, *, task: str | None = None,
                     sample: int = 0, others: list[EvidenceText] | None = None) -> Judgment:
    images = [text.image_path] if text.is_image and text.image_path else []
    if text.is_image and not images:
        return Judgment(finding_id, text.evidence_id, "needs_pixels", is_image=True)
    source = _source_payload(text, statement)
    provenance = str(source.get("provenance") or "")
    payload = {"statement": statement, "source": source, "image_attached": bool(images)}
    other_images = _other_images(text, others) if text.is_image else []
    if other_images:
        payload["other_cited_images"] = other_images
    if sample:
        payload["resample"] = sample          # 0.3.4 实锤复核：独立重采样（缓存键不同）；sample=0 时 payload 与旧版逐字相同
    with_task = bool(task and task.strip()) and _wants_task(text)
    if with_task:
        payload["task"] = task.strip()[:MAX_TASK_CHARS]  # type: ignore[union-attr]
    system_prompt = pair_system_prompt(provenance=bool(provenance), media_note="media_note" in source, task=with_task,
                                       image=text.is_image, others=bool(other_images))
    raw, failure = await _call_checked(judge, system_prompt, payload, images, PAIR_SCHEMA)
    if raw is None:
        return Judgment(finding_id, text.evidence_id, "failed", is_image=text.is_image, call_error=dict(failure.errors),
                        call_failure=failure)
    all_supported = _parts(raw.get("supported_parts"), text)
    all_contradicted = _parts(raw.get("contradicted_parts"), text)
    supported = [p for p in all_supported if p.verified is not False]        # 文本来源引语对不上：不采信
    contradicted = [p for p in all_contradicted if p.verified is not False]
    rejected = [p for p in all_supported + all_contradicted if p.verified is False]
    verdict = "contradicted" if contradicted else "supported" if supported else "not_found"
    label = " ".join(x for x in (text.label, text.source_url) if x)
    return Judgment(finding_id, text.evidence_id, verdict, supported=supported, contradicted=contradicted, rejected=rejected,
                    source_label=label, provenance=provenance, media_note=text.media_note if text.is_image else "",
                    image_path=images[0] if images else None,
                    source_reports_claim=bool(raw.get("source_reports_claim")),
                    statement_asserts_fact=bool(raw.get("statement_asserts_fact", True)),
                    scope_overreach=bool(raw.get("scope_overreach")), is_image=text.is_image,
                    notes=str(raw.get("notes") or ""), raw=dict(raw))


async def judge_coverage(judge: JsonJudge, finding_id: str, statement: str, judgments: list[Judgment]) -> dict[str, Any] | None:
    coverage, _errors = await _judge_coverage_checked(judge, finding_id, statement, judgments)
    return coverage


def _coverage_images(judgments: list[Judgment]) -> list[Judgment]:
    """引了两张以上有像素的图片时，汇总附上它们（按引用顺序，最多 COVERAGE_MAX_IMAGES 张）；只有一张时单帧判定已经看过，不附。"""

    with_pixels = [j for j in judgments if j.is_image and j.image_path is not None]
    return with_pixels[:COVERAGE_MAX_IMAGES] if len(with_pixels) >= 2 else []


async def _judge_coverage_checked(judge: JsonJudge, finding_id: str, statement: str,
                                  judgments: list[Judgment], *, sample: int = 0,
                                  task: str | None = None) -> tuple[dict[str, Any] | None, CallFailure]:
    support: list[dict[str, Any]] = []
    for j in judgments:
        for p in j.supported:
            item: dict[str, Any] = {"evidence_id": j.evidence_id, "source": j.source_label, "part": p.part, "quote": p.quote}
            if j.provenance:          # 没有归属的（外部网页）不加键，保持旧 payload / 旧缓存
                item["provenance"] = j.provenance
            support.append(item)
    payload: dict[str, Any] = {"statement": statement, "supported_parts": support}
    attached = _coverage_images(judgments)
    if attached:
        payload["attached_images"] = [{"evidence_id": j.evidence_id, **({"provenance": j.provenance} if j.provenance else {}),
                                       **({"media_note": j.media_note} if j.media_note else {})} for j in attached]
    if sample:
        payload["resample"] = sample
    with_task = bool(task and task.strip())
    if with_task:
        payload["task"] = task.strip()[:MAX_TASK_CHARS]  # type: ignore[union-attr]
    system_prompt = coverage_system_prompt(provenance=any("provenance" in item for item in support), images=bool(attached),
                                           limitation=FEATURES["limitation_rule"] and is_limitation_statement(statement), task=with_task)
    raw, failure = await _call_checked(judge, system_prompt, payload, [j.image_path for j in attached if j.image_path], COVERAGE_SCHEMA)
    if raw is None:
        return None, failure
    parts = [str(p).strip() for p in (raw.get("unsupported_parts") or []) if isinstance(p, (str, int, float)) and str(p).strip()]
    # 程序兜底：模型列出的"无支持部分"若已逐字出现在某个已核实的支持片段（或其引语）里，不算
    covered_text = _squash(" ".join(f"{p.part} {p.quote}" for j in judgments for p in j.supported))
    parts = [p for p in parts if _squash(p) not in covered_text]
    # model_grounded（verifier@0.3.2）：提示词让模型"拿不准的部分也列出来，但 grounded 只在确实缺少支持时才为 false"。
    # 模型列了部分却仍判 grounded=true，说明它认为这些只是拿不准 / 非实质，不能按"无据断言"记 critical。
    return {"finding_id": finding_id, "grounded": bool(raw.get("grounded", not parts)) and not parts,
            "model_grounded": raw.get("grounded") is True, "unsupported_parts": parts,
            "notes": str(raw.get("notes") or "")}, CallFailure()


def _errors_for_finding(finding_id: str, statement: str, judgments: list[Judgment], coverage: dict[str, Any] | None,
                        *, allow_unsupported: bool = True) -> list[ErrorRecord]:
    out: list[ErrorRecord] = []
    base_extra = {"finding": finding_id}

    def method(j: Judgment) -> str:
        return "pixels" if j.is_image else "model"

    for j in judgments:
        for p in j.contradicted:
            kind = {"misread_fine_text": ErrorType.MISREAD_FINE_TEXT, "sloppy_numeric": ErrorType.SLOPPY_NUMERIC}.get(p.hint or "", ErrorType.WRONG_CITATION)
            out.append(error(kind, f"{j.evidence_id} 与该部分相反：{p.quote[:200]}", span=p.part, evidence_ids=[j.evidence_id],
                             method=method(j), extra={**base_extra, "quote": p.quote[:500]}))
    if coverage is not None and coverage["unsupported_parts"]:
        cited = [j.evidence_id for j in judgments]
        uncertain = coverage.get("model_grounded") is True
        for part in coverage["unsupported_parts"]:
            if not uncertain and FEATURES["limitation_rule"] and is_limitation_part(part) and is_limitation_statement(statement):
                out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "限制语（交代调查范围、不含具体事实），仅备注", span=part,
                                 evidence_ids=cited, method="model", severity=Severity.MINOR,
                                 extra={**base_extra, "notes": coverage.get("notes", ""), "limitation_part": True}))
                continue
            if uncertain:
                out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "核验模型列为拿不准、但整句判为有支持（仅备注）", span=part,
                                 evidence_ids=cited, method="model", severity=Severity.MINOR,
                                 extra={**base_extra, "notes": coverage.get("notes", ""), "uncertain": True}))
                continue
            out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=part, evidence_ids=cited, method="model",
                             extra={**base_extra, "notes": coverage.get("notes", "")}))
    elif (allow_unsupported and coverage is None and judgments and not any(j.supported for j in judgments)
          and all(j.verdict in ("not_found", "contradicted") for j in judgments)):
        out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持该句", span=statement,
                         evidence_ids=[j.evidence_id for j in judgments], method="model", extra=base_extra))
    supporting = [j for j in judgments if j.supported]
    hedged = re.search(r"据|称|报道|引述|表示|声称|According to|reported|said|claims?", statement) is not None
    meta = FEATURES["limitation_rule"] and is_limitation_statement(statement)    # 交代调查范围的句子不是在转述来源
    if not hedged and not meta and any(j.source_reports_claim and j.statement_asserts_fact for j in supporting):
        out.append(error(ErrorType.ATTRIBUTION_DRIFT, "来源只是转述说法，该句却写成了事实", span=statement,
                         evidence_ids=[j.evidence_id for j in supporting if j.source_reports_claim], method="model", extra=base_extra))
    if any(j.scope_overreach for j in supporting):
        # 单条来源上的"以偏概全"提示只是线索（视频帧类报告几乎条条被提示），记 minor，不单独把报告推成 negative
        out.append(error(ErrorType.SCOPE_OVERREACH, "核验模型提示：由局部/少数样本推断了整体", span=statement, severity=Severity.MINOR,
                         evidence_ids=[j.evidence_id for j in supporting if j.scope_overreach], method="model", extra=base_extra))
    out = apply_pixel_rules(out, statement, judgments)           # verifier@0.4.0 第 3 处：像素只核句子所指的那张图
    return apply_date_rule(out, statement, judgments)            # verifier@0.4.0 第 5 处：页面声明的日期与 dates 卡一致不算 sloppy_numeric


@dataclass
class _FindingOutcome:
    judgments: list[Judgment] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] | None = None
    errors: list[ErrorRecord] = field(default_factory=list)


_FIRM = (Severity.CRITICAL, Severity.MAJOR)
_CONFIRM_GROUP = {ErrorType.WRONG_CITATION: "contra", ErrorType.SLOPPY_NUMERIC: "contra", ErrorType.MISREAD_FINE_TEXT: "contra"}


def _group(e: ErrorRecord) -> str:
    return _CONFIRM_GROUP.get(e.type, e.type.value)


def _needs_confirm(e: ErrorRecord) -> bool:
    return e.severity in _FIRM and e.method in ("model", "pixels")


async def _verify_finding(finding_id: str, statement: str, texts: list[EvidenceText], judge: JsonJudge, *,
                          task: str | None = None) -> _FindingOutcome:
    """一条 Finding 的第二层核验；0.3.4 起，第一遍出了模型层实锤错误就独立复核一遍，没复现的降为 minor。"""

    outcome = await _verify_finding_once(finding_id, statement, texts, judge, task=task)
    firm = [e for e in outcome.errors if _needs_confirm(e)]
    if not (FEATURES["confirm_firm"] and firm):
        return outcome
    second = await _verify_finding_once(finding_id, statement, texts, judge, task=task, sample=1)
    if second.pending and not second.errors:
        return outcome                      # 复核没跑成：保留第一遍（不因调用失败放过错误）
    again = {_group(e) for e in second.errors if e.severity in _FIRM}
    kept: list[ErrorRecord] = []
    for e in outcome.errors:
        if _needs_confirm(e) and _group(e) not in again:
            e = replace(e, severity=Severity.MINOR, detail="（复核未复现，仅备注）" + e.detail,
                        extra={**e.extra, "unconfirmed": True, "original_severity": e.severity.value})
        elif _needs_confirm(e):
            e = replace(e, extra={**e.extra, "confirmed": True})
        kept.append(e)
    outcome.errors = kept
    return outcome


async def _verify_finding_once(finding_id: str, statement: str, texts: list[EvidenceText], judge: JsonJudge, *,
                               task: str | None = None, sample: int = 0) -> _FindingOutcome:
    """一条 Finding：各 E 的 judge_pair 并发（按 texts 顺序收集），全部完成后再汇总覆盖。"""

    outcome = _FindingOutcome()
    judgments = list(await asyncio.gather(*(judge_pair(judge, finding_id, statement, text, task=task, sample=sample, others=texts)
                                            for text in texts)))
    outcome.judgments = judgments
    for judgment in judgments:
        if judgment.verdict == "needs_pixels":
            outcome.pending.append({"finding": finding_id, "evidence_id": judgment.evidence_id, "reason": "image_without_pixels"})
        elif judgment.verdict == "failed":
            failure = judgment.call_failure or CallFailure(errors=dict(judgment.call_error))
            outcome.pending.append({"finding": finding_id, "evidence_id": judgment.evidence_id, "reason": "judge_output_invalid",
                                    **failure.pending_fields()})
    usable = [j for j in judgments if j.verdict in ("supported", "contradicted", "not_found")]
    if not usable:
        return outcome
    if any(j.verdict == "failed" for j in judgments):
        # 有一条 E 没判成，就说不清"没被支持的部分"是真没支持还是那条 E 没看；只报反驳，不报无据
        outcome.errors = _errors_for_finding(finding_id, statement, usable, None, allow_unsupported=False)
        return outcome
    coverage, failure = await _judge_coverage_checked(judge, finding_id, statement, usable, sample=sample, task=task)
    if coverage is None:
        outcome.pending.append({"finding": finding_id, "reason": "coverage_output_invalid", **failure.pending_fields()})
    outcome.coverage = coverage
    outcome.errors = _errors_for_finding(finding_id, statement, usable, coverage)
    return outcome


# ---------------------------------------------------------------- 程序规则（候选，见 PROGRAM_RULES）

_PUNCT = re.compile(r"[\s\u3000，。、；：:,.;!?！？（）()\[\]【】“”\"'‘’《》<>—–\-·…]+")
_DATE = DATE_NUMERIC             # 10-07 起日期、套话、回指、结论句都是中英双语（lang.py）；中文部分与以前逐字相同
_TASK_WORDS = TASK_WORDS
_BACKREF = BACKREF
_VERDICT = VERDICT


def _norm(text: str) -> str:
    return _PUNCT.sub("", text or "").lower()


def _dates(text: str) -> set[str]:
    return set(dates_in(text or ""))


def is_task_given(span: str, task: str | None) -> bool:
    """片段是否只是复述任务文字里已给出的信息：整段出现在任务里；或所含日期都在任务里、去掉日期和套话后几乎不剩内容。"""

    if not task or not span:
        return False
    ns, nt = _norm(span), _norm(task)
    if len(ns) >= 4 and ns in nt:
        return True
    dates = _dates(span)
    if not dates or not dates <= _dates(task):
        return False
    residue = _norm(_TASK_WORDS.sub("", strip_dates(span)))
    return len(residue) <= 2


def _bigrams(text: str) -> set[str]:
    """中文：字符二元组（同以前）；纯英文：实词（英文字母二元组几乎处处重合）。"""

    return grams(text, _norm)


def back_reference_support(span: str, others: dict[str, str]) -> list[str]:
    """片段含回指标记，且去掉标记后的内容大半能在其他干净 Finding / 答案句里找到 → 返回这些编号；否则空表。"""

    if not span or not _BACKREF.search(span) or not others:
        return []
    content = _BACKREF.sub("", span)
    grams = _bigrams(content)
    if not grams:
        return list(others)
    hits = [fid for fid, text in others.items() if grams & _bigrams(text)]
    covered = set().union(*(_bigrams(others[f]) for f in hits)) & grams if hits else set()
    return hits if len(covered) / len(grams) >= 0.5 else []


def is_verdict_sentence(span: str) -> bool:
    return bool(span) and _VERDICT.search(span.strip()) is not None


def apply_program_rules(errors: list[ErrorRecord], statements: dict[str, str], task: str | None, *, answer: bool,
                        key: str = "finding") -> list[ErrorRecord]:
    """候选程序规则：把命中规则的模型层"无支持"实锤降为 minor。statements 是同一单元全部 Finding / 答案句。"""

    if not any(PROGRAM_RULES.values()):
        return errors
    firm_by = {}
    for e in errors:
        if e.severity in _FIRM:
            firm_by.setdefault(str(e.extra.get(key) or e.extra.get("finding")), []).append(e)
    out: list[ErrorRecord] = []
    for e in errors:
        if not (e.type is ErrorType.UNSUPPORTED_ASSERTION and e.severity in _FIRM and e.method == "model" and isinstance(e.span, str)
                and not e.extra.get("check")):           # 案卷断言（标签 / 日期等）不归第二层规则管
            out.append(e)
            continue
        fid = str(e.extra.get(key) or e.extra.get("finding"))
        clean = {f: t for f, t in statements.items() if f != fid and not firm_by.get(f)}
        rule, info = None, {}
        if PROGRAM_RULES["task_given"] and is_task_given(e.span, task):
            rule = "task_given"
        elif PROGRAM_RULES["back_reference"]:
            refs = back_reference_support(e.span, clean)
            if refs:
                rule, info = "back_reference", {"refers_to": refs[:6]}
        if rule is None and PROGRAM_RULES["verdict_sentence"] and answer and is_verdict_sentence(e.span) and len(clean) >= 2:
            rule, info = "verdict_sentence", {"supported_by": list(clean)[:6]}
        if rule is None:
            out.append(e)
            continue
        note = {"task_given": "复述任务给出的信息", "back_reference": "回指同单元已建立的内容",
                "verdict_sentence": "对声明的总结论，依据在同一答案的其他句子"}[rule]
        out.append(replace(e, severity=Severity.MINOR, detail=f"（程序规则：{note}，仅备注）" + e.detail,
                           extra={**e.extra, "program_rule": rule, "original_severity": e.severity.value, **info}))
    return out


def feature_version() -> str:
    """当前开关组合对应的版本标签（reanchor 用）。"""

    from .inference import INFERENCE_CHECK
    from .semantic_coverage import SEMANTIC_COVERAGE
    from .verdict import VERIFIER_VERSION
    if not (FEATURES["limitation_rule"] and FEATURES["confirm_firm"]):
        return "verifier@0.3.3"
    flags = (("rules", all(PROGRAM_RULES.values())), ("infer", INFERENCE_CHECK["enabled"]), ("semcov", SEMANTIC_COVERAGE["enabled"]))
    from .joint_sources import JOINT_SOURCES
    tags = [t for t, on in flags if on]
    joint = "+joint" if JOINT_SOURCES["enabled"] else ""          # 候选：多来源合并复核（默认关闭，不进正式版本号）
    if len(tags) == len(flags):
        return VERIFIER_VERSION + joint
    return (VERIFIER_VERSION + "-partial:" + "+".join(tags) if tags else "verifier@0.3.5") + joint


async def verify_anchored(report: dict[str, Any], lookup: EvidenceLookup, judge: JsonJudge, *,
                          skip_evidence_ids: set[str] | None = None, task: str | None = None, answer: bool = False) -> AnchoredResult:
    """对每条 Finding：逐条可读 E 判"部分"，再汇总覆盖。skip_evidence_ids 是第一层已判为不存在/不可见的 E。
    task：报告所接的任务文字（answer 单元是总任务）；只随来自输入材料的图片对进 payload，用来理解指代（verifier@0.3.3）。

    各 Finding、各 (Finding, E) 并发跑（上限见 judge_concurrency）；产物顺序与串行一致。
    """

    result = AnchoredResult()
    skip = skip_evidence_ids or set()
    jobs = []
    statements: dict[str, str] = {}
    for index, finding in enumerate(report.get("findings") or []):
        if not isinstance(finding, dict):
            continue
        finding_id = str(finding.get("finding_id") or f"finding[{index}]")
        statement = str(finding.get("statement") or "")
        statements[finding_id] = statement
        texts: list[EvidenceText] = []
        for eid in finding.get("evidence_ids") or []:
            if not isinstance(eid, str) or eid in skip or not lookup.has(eid):
                continue
            text = lookup.get(eid)
            if not text.is_text and not text.is_image:
                continue                         # 视频/音频原件：本层不判，靠派生的帧/转写
            texts.append(text)
        jobs.append(_verify_finding(finding_id, statement, texts, judge, task=task))
    for outcome in await asyncio.gather(*jobs):
        result.judgments.extend(outcome.judgments)
        result.pending.extend(outcome.pending)
        if outcome.coverage is not None:
            result.coverage.append(outcome.coverage)
        result.errors.extend(outcome.errors)
    result.errors = apply_program_rules(result.errors, statements, task, answer=answer)
    from .inference import INFERENCE_CHECK, apply_inference_check     # 循环引用：inference 依赖本模块
    if INFERENCE_CHECK["enabled"]:
        result.errors = await apply_inference_check(result.errors, statements, [j.to_dict() for j in result.judgments], judge)
    from .joint_sources import JOINT_SOURCES, apply_joint_check       # 候选（默认关闭）：多来源合并复核
    if JOINT_SOURCES["enabled"]:
        result.errors = await apply_joint_check(result.errors, statements, lookup, judge)
    return result


def count_pair_calls(report: dict[str, Any], lookup: EvidenceLookup, *, skip_evidence_ids: set[str] | None = None) -> int:
    """不调模型，估算 verify_anchored 要发的调用数：每条可判 (Finding, E) 一次 + 每条有可判 E 的 Finding 一次汇总。"""

    skip = skip_evidence_ids or set()
    total = 0
    for finding in report.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        pairs = 0
        for eid in finding.get("evidence_ids") or []:
            if not isinstance(eid, str) or eid in skip or not lookup.has(eid):
                continue
            text = lookup.get(eid)
            if text.is_text or (text.is_image and text.image_path):
                pairs += 1
        total += pairs + (1 if pairs else 0)
    return total


__all__ = ["COVERAGE_MAX_IMAGES", "coverage_system_prompt", "pair_system_prompt", "COVERAGE_SCHEMA", "COVERAGE_SYSTEM_PROMPT", "COVERAGE_SYSTEM_PROMPTS", "BUDGET_EXIT_CODE", "DEFAULT_CONCURRENCY", "JudgeBudgetExhausted", "budget_message", "PAIR_SCHEMA", "PAIR_SYSTEM_PROMPT",
           "PAIR_SYSTEM_PROMPTS", "AnchoredResult", "CALL_ATTEMPTS", "CallFailure", "JsonJudge",
           "Judgment", "Part", "count_pair_calls", "judge_concurrency", "judge_coverage", "judge_pair", "judge_semaphore", "quote_in_source",
           "verify_anchored", "PROGRAM_RULES", "apply_program_rules", "is_task_given", "back_reference_support", "is_verdict_sentence"]
