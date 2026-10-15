"""核验器程序规则的中英双语词表与切分（2026-10-07）。

大白话：hi@0.7.0 / gia@0.8.0 起模型输入输出全英文（规格 harness_params.language=en），教师写的 Finding、答案都是英文。
核验器第一层（程序规则）原来只认中文：限制语"未能/无法确认"、日期"2016年7月14日"、区间"12.5–14 秒"、
回指"上述"、结论句"该声明不实"、句子按"。！？"切……碰到英文要么全漏（答案整段被当成一句），要么误伤
（"含拉丁字母词"被当成具体事实）。这里集中放双语版本，各规则模块从这里取；中文行为逐字不变。

只认措辞，不判语义；拿不准的仍交给模型层。
"""

from __future__ import annotations

import re
from typing import Iterable

CJK = re.compile(r"[㐀-鿿豈-﫿]")


def is_cjk(text: str) -> bool:
    """文字里有汉字就按中文规则（中文 Finding 里夹英文专名是常态；纯英文才走英文规则）。"""

    return bool(CJK.search(text or ""))


# ---------------------------------------------------------------- 限制语（交代调查范围 / 局限）

LIMITATION_ZH = (r"未能|未核验|未取得|未获得|未覆盖|未检查|未查到|无法(?:确认|核实|判断|证实|排除)|不能(?:确认|确定|证实|据此|直接|排除|算作)"
                 r"|尚不能|尚无法|尚未|仅(?:覆盖|核查|检查|核验|抽查|查看)|只(?:覆盖|核查|检查|核验|抽查|查看)|证据不足|不足以|局限|限制[:：]"
                 r"|(?:没有|并未|未)明确|没有直接|无直接")
LIMITATION_EN = (r"\b(?:could ?n[o']t|can ?n[o']t|cannot|unable to|(?:was|were) not able to"
                 r"|did ?n[o']t (?:obtain|find|verify|check|confirm|locate|access|examine|inspect|establish|retrieve)"
                 r"|(?:was|were|is|are|has|have|had)(?: been)? not (?:verified|confirmed|checked|obtained|found|established|covered|examined|inspected|retrieved|accessed)"
                 r"|not (?:independently )?(?:verified|confirmed|established)|unverified|unconfirmed|not verifiable"
                 r"|remains? (?:unclear|unknown|unverified|unconfirmed|uncertain|open)|(?:is|was) unclear|uncertain(?:ty)?"
                 r"|only (?:covered|checked|examined|reviewed|sampled|inspected|looked at|covers|checks|examines)"
                 r"|insufficient|not enough (?:evidence|information)|does not (?:establish|show|prove)|do not (?:establish|show|prove)"
                 r"|no direct|not directly|limitations?:|limited to|beyond the scope|outside the scope"
                 r"|cannot be (?:confirmed|determined|ruled out|established|verified)|could not be (?:confirmed|determined|ruled out|established|verified|obtained))\b")
LIMITATION = re.compile(f"{LIMITATION_ZH}|{LIMITATION_EN}", re.I)

# "具体事实"（限制语里夹带的数字、专名、转折）：中文沿用 0.3.4 的口径；英文看数字、数词、转折连词与句中大写专名
CONCRETE_ZH = re.compile(r"\d|[A-Za-z]{2,}|但|但是|然而|不过|却|一|二|两|三|四|五|六|七|八|九|十|百|千|万")
_CONCRETE_EN = re.compile(r"\d|\b(?:but|however|although|though|yet|whereas|nevertheless|nonetheless|while"
                          r"|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_NOT_PROPER = {"I", "A", "An", "The", "This", "That", "These", "Those", "It", "We", "No", "Not", "Only", "However", "But"}


def proper_nouns(text: str) -> list[str]:
    """英文句中的大写词（排除句首词与 I / The 等）：近似专名。"""

    out: list[str] = []
    for sentence in split_sentences(text):
        words = _WORD.findall(sentence)
        out.extend(w for w in words[1:] if w[0].isupper() and w not in _NOT_PROPER)
    return out


def has_concrete(text: str) -> bool:
    text = text or ""
    if is_cjk(text):
        return bool(CONCRETE_ZH.search(text))
    return bool(_CONCRETE_EN.search(text)) or bool(proper_nouns(text))


# ---------------------------------------------------------------- 日期

_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
DATE_NUMERIC = re.compile(r"(\d{4})\s*[年\-/.]\s*(\d{1,2})\s*[月\-/.]\s*(\d{1,2})\s*日?")
_DATE_MDY = re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DATE_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON},?\s+(\d{{4}})\b", re.I)


def date_spans(text: str) -> list[tuple[int, int, str]]:
    """(起, 止, YYYY-MM-DD)：数字日期（2016-07-14 / 2016年7月14日）与英文月份日期（July 14, 2016 / 14 July 2016）。"""

    out: list[tuple[int, int, str]] = []
    for m in DATE_NUMERIC.finditer(text or ""):
        y, mo, d = m.groups()
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            out.append((m.start(), m.end(), f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"))
    for m in _DATE_MDY.finditer(text or ""):
        mon, d, y = m.groups()
        out.append((m.start(), m.end(), f"{int(y):04d}-{_MONTHS[mon.lower().rstrip('.')]:02d}-{int(d):02d}"))
    for m in _DATE_DMY.finditer(text or ""):
        d, mon, y = m.groups()
        out.append((m.start(), m.end(), f"{int(y):04d}-{_MONTHS[mon.lower().rstrip('.')]:02d}-{int(d):02d}"))
    return sorted(out)


def dates_in(text: str) -> list[str]:
    return [d for _, _, d in date_spans(text)]


def strip_dates(text: str) -> str:
    text = text or ""
    for start, end, _ in reversed(date_spans(text)):
        text = text[:start] + " " + text[end:]
    return text


# ---------------------------------------------------------------- 时间区间（秒）

_SEC_ZH = r"(?:秒|s)"
_SEC_EN = r"(?:(?:s|sec|secs|seconds?)\b)"
INTERVAL_ZH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*[–\-~至到]\s*(\d+(?:\.\d+)?)\s*(?:秒|s)")
_INTERVAL_EN = re.compile(rf"(?:from\s+|between\s+)?(\d+(?:\.\d+)?)\s*{_SEC_EN}?\s*(?:[–\-~]|to|and|until|through)\s*(\d+(?:\.\d+)?)\s*{_SEC_EN}", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2}(?:\.\d+)?)\s*(?:[–\-~]|to|至|到)\s*(\d{1,2}):(\d{2}(?:\.\d+)?)")


def intervals_in(text: str) -> list[tuple[float, float]]:
    """秒数区间：12.5–14 秒 / 12.5 s–14 s / from 12.5 to 14.0 seconds / between 12.5 and 14 s / 0:12–0:14。"""

    out = [(float(a), float(b)) for a, b in INTERVAL_ZH.findall(text or "")]
    out += [(float(a), float(b)) for a, b in _INTERVAL_EN.findall(text or "")]
    out += [(int(m1) * 60 + float(s1), int(m2) * 60 + float(s2)) for m1, s1, m2, s2 in _CLOCK.findall(text or "")]
    return list(dict.fromkeys(pair for pair in out if pair[0] <= pair[1]))


# ---------------------------------------------------------------- 句子切分

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "st", "no", "vs", "etc", "e.g", "i.e", "jr", "sr", "inc", "ltd", "co", "fig",
           "approx", "u.s", "u.k", "a.m", "p.m", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
           "nov", "dec", "rep", "sen", "gov", "gen", "col", "lt", "sgt", "mt", "ft", "vol", "pp", "ca"}
_ZH_END = "。！？；"
_TERMINATOR_ZH = re.compile(r"(?<=[。！？!?；;])")
_EN_END = ".!?;"


def split_sentences(paragraph: str) -> list[str]:
    """按中文句末（。！？；）与英文句末（. ! ? ; 后跟空白，且不是缩写、小数、单字母缩写）切句，分隔符留在前一句。"""

    text = paragraph or ""
    if is_cjk(text):
        # 中文段落：与 0.3.6 以前逐字相同（answer_anchored._TERMINATOR）
        return [p for p in _TERMINATOR_ZH.split(text) if p.strip()]
    pieces: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cut = False
        if ch in _ZH_END:
            cut = True
        elif ch in "!?;":
            cut = i + 1 >= n or text[i + 1].isspace()
        elif ch == ".":
            if i + 1 < n and not text[i + 1].isspace():
                cut = False                                   # 1.5 / e.g. / U.S. / example.com
            else:
                word = re.search(r"([A-Za-z][A-Za-z.]*)$", text[start:i])
                token = word.group(1).lower() if word else ""
                cut = not (token in _ABBREV or (len(token) == 1 and token.isalpha()))
        if cut:
            j = i + 1
            while j < n and text[j] in "”\"'’)）]】":           # 句末引号 / 括号跟着前一句
                j += 1
            pieces.append(text[start:j])
            start = j
            i = j
            continue
        i += 1
    if start < n:
        pieces.append(text[start:])
    return [p for p in pieces if p.strip()]


# ---------------------------------------------------------------- 其他措辞

TASK_WORDS_ZH = r"声明日期|声明发布|发布日期|声明原文|声明|日期|发布于|发布|为|于|是|的|即|在|晚于|早于|当天|当日"
TASK_WORDS_EN = (r"\b(?:the|a|an|this|that|claim(?:'s)?|claimed|statement|date[ds]?|dated|published|posted|issued|made|on|in|at|of|"
                 r"is|was|were|be|been|it|as|later|earlier|than|before|after|same|day|original|text|by)\b")
TASK_WORDS = re.compile(f"{TASK_WORDS_ZH}|{TASK_WORDS_EN}", re.I)

BACKREF_ZH = r"上述|前述|前文|如前所述|已确认的|已核实的|已建立的|(?:现有|当前|已有)(?:证据|材料)(?:能够|可以|足以)?(?:确认|支持|表明)"
BACKREF_EN = (r"\b(?:the above|above-mentioned|aforementioned|as (?:noted|described|stated|shown|established|mentioned|found) (?:above|earlier|before)"
              r"|as (?:already )?established|(?:previously|already) (?:established|confirmed|verified|identified|noted|shown)"
              r"|these (?:differences|findings|observations)|the (?:existing|current|available|gathered) evidence (?:confirms|supports|shows|indicates|establishes))\b")
BACKREF = re.compile(f"{BACKREF_ZH}|{BACKREF_EN}", re.I)

VERDICT_ZH = (r"^(?:结论[:：]\s*)?(?:该|此|这则|这一|这条)?(?:声明|说法|帖子|传言)?(?:的)?(?:核心)?"
              r"(?:错误|不实|属实|成立|不成立|具有误导性|有误导性|有证据支持|部分属实|缺乏证据|证据不足|不能成立|被证伪)")
VERDICT_EN = (r"^(?:(?:conclusion|verdict|rating|label|overall|in summary|in short)\s*[:,]\s*)?(?:the|this|that)?\s*(?:core\s+)?"
              r"(?:claim|statement|post|rumou?r|assertion)?\s*(?:is|was|appears|remains)?\s*(?:to be\s+)?"
              r"(?:false|true|misleading|mostly false|mostly true|partly true|partially true|half true|a mixture|mixed|unsupported|unproven"
              r"|refuted|supported|accurate|inaccurate|not supported|not true|fabricated|correct|incorrect|lacks evidence|unsubstantiated)\b")
VERDICT = re.compile(f"{VERDICT_ZH}|{VERDICT_EN}", re.I)

REFUTE_ZH = r"反驳|驳斥|否定|否认|排除|推翻|不支持|并非|不是|不实|不成立|误称|谣传|所谓|声称|宣称|解释"
REFUTE_EN = (r"\b(?:refut\w*|rebut\w*|den(?:y|ies|ied)|debunk\w*|reject\w*|rule[sd]? out|dismiss\w*|contradict\w*|disprov\w*|"
             r"not (?:true|the case|supported)|false(?:ly)?|so-called|alleged(?:ly)?|claim(?:s|ed)?|purport\w*|rumou?r\w*|misattribut\w*|explain\w*)\b")
REFUTE = re.compile(f"{REFUTE_ZH}|{REFUTE_EN}", re.I)

SIBLING_DEP_ZH = r"W\d+-\d+|上一(个)?任务|另一(个)?任务|兄弟|待.{0,12}(确认|完成|结果)后|根据.{0,12}的?结果|依据.{0,8}报告"
SIBLING_DEP_EN = (r"\b(?:previous|other|another|sibling|parallel) (?:task|workspace)s?\b|\bonce .{0,30}\b(?:confirmed|completed|finished|available)\b"
                  r"|\b(?:based on|using|depending on|after) (?:the )?(?:results?|findings?|output|report) (?:of|from)\b")
SIBLING_DEP = re.compile(f"({SIBLING_DEP_ZH}|{SIBLING_DEP_EN})", re.I)

HEDGE_ZH = r"尚未|未能|未经|未独立|无法确认|不能确认|未核实|未确认|仍未"
HEDGE_EN = (r"\b(?:not (?:yet |independently )?(?:verified|confirmed|established)|unverified|unconfirmed|could not (?:be )?(?:verify|verified|confirm|confirmed)"
            r"|cannot (?:be )?(?:verify|verified|confirm|confirmed)|remains? (?:unverified|unconfirmed|unclear))\b")
HEDGE = re.compile(f"({HEDGE_ZH}|{HEDGE_EN})", re.I)

ANSWER_SKIP = re.compile(r"^(结论|判定|标签|总结|conclusion|verdict|rating|label|summary)\s*[:：]?\s*$", re.I)

# 标签前的否定 / 限定：中文看紧贴的两个字，英文看前一个词（"not false""partly true""mostly false"）及词内前缀（un-/in-）
LABEL_NEGATORS_ZH = ("不", "非", "未", "无", "没有", "并非", "部分", "基本", "大体", "多数")
LABEL_NEGATORS_EN = {"not", "no", "never", "partly", "partially", "mostly", "largely", "half", "somewhat", "neither", "nor", "isn't",
                     "wasn't", "aren't", "weren't", "hardly", "mainly", "predominantly"}


def _starts_with_negator(form: str) -> bool:
    first = form.split()[0] if form.split() else ""
    return form.startswith(LABEL_NEGATORS_ZH) or first in LABEL_NEGATORS_EN


def label_hit(form: str, answer: str) -> bool:
    """form 在 answer 里出现，且至少有一处前面没有紧贴否定 / 限定词（form 自身以这些词开头时不查）。

    拉丁字母写法额外要求词边界（"True" 不算进 "untrue"，"Correct" 不算进 "incorrect"）。"""

    f, a = form.lower().strip(), answer.lower()
    if not f:
        return False
    latin = bool(re.match(r"[a-z]", f))
    start = a.find(f)
    while start != -1:
        ok = True
        if latin:
            if start > 0 and (a[start - 1].isalpha() or a[start - 1] in "-'’"):
                ok = False                                  # 词内：untrue / incorrect / half-true
            elif not _starts_with_negator(f):
                prev = re.findall(r"[a-z'’]+", a[max(0, start - 30):start])
                ok = not (prev and prev[-1] in LABEL_NEGATORS_EN)
        else:
            before = a[max(0, start - 2):start]
            ok = _starts_with_negator(f) or not any(before.endswith(neg) for neg in LABEL_NEGATORS_ZH)
        if ok:
            return True
        start = a.find(f, start + 1)
    return False


# ---------------------------------------------------------------- 文本重合

_STOP_EN = {"the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "was", "were", "are", "be", "been", "it", "this",
            "that", "these", "those", "with", "by", "as", "from", "which", "who", "whether", "not", "no", "its", "their", "his", "her",
            "has", "have", "had", "than", "such", "any", "all", "each", "both", "can", "could", "does", "did", "do", "if", "so", "also"}


def content_words(text: str) -> set[str]:
    return {w for w in (x.lower().strip("'’-") for x in _WORD.findall(text or "")) if len(w) > 2 and w not in _STOP_EN}


def grams(text: str, norm: callable = None) -> set[str]:  # type: ignore[valid-type]
    """重合比较的单位：中文用字符二元组（与以前相同）；纯英文用实词（英文字母二元组几乎处处重合，不能用）。"""

    if is_cjk(text or ""):
        t = norm(text) if norm else re.sub(r"\s+", "", text or "")
        return {t[i:i + 2] for i in range(len(t) - 1)}
    return content_words(text)


def any_form_in(forms: Iterable[str], text: str) -> bool:
    low = (text or "").lower()
    return any(f and f.lower() in low for f in forms)


__all__ = ["BACKREF", "HEDGE", "LIMITATION", "REFUTE", "SIBLING_DEP", "TASK_WORDS", "VERDICT", "ANSWER_SKIP", "content_words",
           "date_spans", "dates_in", "grams", "has_concrete", "intervals_in", "is_cjk", "label_hit", "proper_nouns",
           "split_sentences", "strip_dates", "any_form_in"]
