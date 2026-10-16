"""多引擎以图搜图结果的统一写法（10-16，FOREN_REVERSE_NORMALIZE=1 才用；默认关，关时一切照旧）。

同一个说法在各引擎里意思不同：GCV 的 full / partial / page、SerpAPI 版 Lens 的 exact / visual、Serper 版 Lens 与
Yandex 根本不报告是不是同一张图（以前代码把 Serper 的每条都写成 "visually similar"）、TinEye 只收副本。这里把它们
换成与引擎无关的写法，供 multi_reverse 合并时用：

- 网址：去掉跟踪参数（utm_*、fbclid、gclid、yclid、ref_src、spm …；YouTube 的 si / feature），其余参数原样保留
  （v=、文章 id 等）。合并去重与显示的网址都用去过参数的。
- extra.engine_match：{引擎: exact / partial / similar / page / unknown}，各引擎原来报告的匹配类型；不报告的写 unknown。
- extra.date / date_kind（page_date / crawl_date / approx_relative）：Lens 的相对日期（"3 years ago"）按取到结果那天
  （硬盘缓存里的条目按缓存的 saved_at）换算成约略年份 / 年月，原文留在 date_raw，换算基准在 date_basis。
- 摘要统一几句事实：自核结果（thumb_verify）、哪些引擎找到、各引擎报告的匹配类型、日期及其类型、网页文字。
  只写引擎说了什么、我们比出了什么，不写推断。
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from ..contracts import CandidateSource
from .google_vision import youtube_id

LABELS = {"gcv": "Google Cloud Vision", "lens": "Google Lens", "tineye": "TinEye", "yandex": "Yandex"}
MATCH_WORDS = ("exact", "partial", "similar", "page", "unknown")
DATE_KINDS = ("page_date", "crawl_date", "approx_relative")

# 跟踪参数：不影响网页是哪一页，只记来源 / 点击。与 recording.snapshot_url 的集合一致，另加几项常见的。
TRACKING_PARAMS = re.compile(
    r"^(utm_[a-z0-9_]+|fbclid|gclid|gbraid|wbraid|dclid|msclkid|yclid|ysclid|mc_cid|mc_eid|_hsenc|_hsmi|igshid|"
    r"ref_src|ref_url|spm|scm|share_source|share_medium|vd_source|_ga|mkt_tok)$", re.I)
# 只在 YouTube 上算跟踪参数（别的站点上 si / feature 可能有意义）
YOUTUBE_TRACKING = re.compile(r"^(si|feature|pp)$", re.I)

_MATCH_TEXT = {"exact": "same image", "partial": "cropped or edited copy", "similar": "visually similar image",
               "page": "page with a matching image", "unknown": "match type not reported"}
# 只写量出来的：感知哈希近不近。10-16 Yandex 实测：判 same 的里有一张是局部改过的副本（同一张照片手里贴了东西），
# 判 similar 的多是同一张照片裁过 / 加框 / 加字，判 different 的里有加了旗子和单边黑框的副本——所以不写"同一张 / 不是同一张"。
# no match 补一句事实（10-16 第二轮）：e107 的真来源判成 no match，因为查询图加了黑边、贴了旗。
_VERIFIED_TEXT = {"same": "near-identical", "similar": "partial match, not near-identical",
                  "different": "no match (cropping, an added border or a local edit can also prevent a hash match)"}


def enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("FOREN_REVERSE_NORMALIZE", "").strip().lower() in {"1", "on", "true", "yes"}


# ------------------------------------------------------------------ 网址


def clean_url(url: str) -> str:
    """去掉跟踪参数；其余参数的顺序与写法原样保留，fragment 也保留。"""

    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.query:
        return url
    host = (parts.hostname or "").lower()
    youtube = host.endswith("youtube.com") or host == "youtu.be"
    kept = []
    for pair in parts.query.split("&"):
        if not pair:
            continue
        name = unquote_plus(pair.split("=", 1)[0])
        if TRACKING_PARAMS.match(name) or (youtube and YOUTUBE_TRACKING.match(name)):
            continue
        kept.append(pair)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "&".join(kept), parts.fragment))


def identity(url: str) -> str:
    """合并去重用的网页身份：YouTube 视频按编号；其余去跟踪参数、去 fragment、去末尾 /、去 www.、小写。"""

    vid = youtube_id(url)
    if vid:
        return f"youtube:{vid}"
    return clean_url(url).split("#", 1)[0].rstrip("/").replace("://www.", "://").lower()


# ------------------------------------------------------------------ 匹配类型


def engine_match(name: str, candidate: CandidateSource) -> str:
    """某个引擎对这条结果报告的匹配类型，统一成 exact / partial / similar / page / unknown。"""

    extra = candidate.extra or {}
    if name == "lens":
        if extra.get("match") == "exact":
            return "exact"
        # Serper 不分 exact / visual（代码给每条都写了 visual）；SerpAPI 的 visual_matches 才是"相似"
        return "similar" if extra.get("match") == "visual" and extra.get("lens_backend") == "serpapi" else "unknown"
    if name == "gcv":
        kind = extra.get("match")
        if kind in ("thumbnail", "similar_thumbnail"):
            kind = extra.get("thumb_match") or ("similar" if kind == "similar_thumbnail" else None)
        return {"full": "exact", "partial": "partial", "page": "page", "similar": "similar"}.get(kind, "unknown")
    if name == "tineye":
        return "exact"          # TinEye 只收同一张图的副本（可能缩放 / 调色过），不报告"相似"
    return "unknown"            # Yandex 等：不报告是不是同一张图


# ------------------------------------------------------------------ 日期

_REL = re.compile(r"^\s*(\d+|an?|one)\s+(minute|hour|day|week|month|year)s?\s+ago\s*$", re.I)
_ABS_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%b %Y", "%B %Y")


def _shift_months(day: _dt.date, months: int) -> tuple[int, int]:
    total = day.year * 12 + (day.month - 1) - months
    return total // 12, total % 12 + 1


def relative_date(text: str, basis: _dt.date) -> str | None:
    """"3 years ago" → "2023"（basis 2026-10-16）；月 / 周 → "YYYY-MM"；天及更短 → "YYYY-MM-DD"；不认识返回 None。"""

    low = text.strip().lower()
    if low in {"today", "just now"}:
        return basis.isoformat()
    if low == "yesterday":
        return (basis - _dt.timedelta(days=1)).isoformat()
    m = _REL.match(low)
    if not m:
        return None
    n = 1 if m.group(1) in {"a", "an", "one"} else int(m.group(1))
    unit = m.group(2)
    if unit == "year":
        return f"{basis.year - n:04d}"
    if unit == "month":
        y, mo = _shift_months(basis, n)
        return f"{y:04d}-{mo:02d}"
    if unit == "week":
        d = basis - _dt.timedelta(weeks=n)
        return f"{d.year:04d}-{d.month:02d}"
    return (basis - _dt.timedelta(days=n if unit == "day" else 0)).isoformat()


def absolute_date(text: str) -> str | None:
    for fmt in _ABS_FORMATS:
        try:
            parsed = _dt.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
        return parsed.strftime("%Y-%m") if "%d" not in fmt else parsed.isoformat()
    return None


def _basis(value: Any) -> _dt.date | None:
    if isinstance(value, str) and len(value) >= 10:
        try:
            return _dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def engine_dates(name: str, candidate: CandidateSource) -> list[dict[str, Any]]:
    """这个引擎给这条结果的日期（可能没有）：[{date, date_kind, date_raw, date_basis, engine}]。"""

    extra = candidate.extra or {}
    out: list[dict[str, Any]] = []
    if name == "lens" and isinstance(extra.get("lens_date"), str) and extra["lens_date"].strip():
        raw = extra["lens_date"].strip()
        basis = _basis(extra.get("fetched_on"))
        approx = relative_date(raw, basis) if basis else None
        if approx:
            out.append({"engine": name, "date": approx, "date_kind": "approx_relative", "date_raw": raw,
                        "date_basis": basis.isoformat()})
        elif _REL.match(raw.lower()):
            out.append({"engine": name, "date": None, "date_kind": "approx_relative", "date_raw": raw, "date_basis": None})
        else:
            out.append({"engine": name, "date": absolute_date(raw) or raw, "date_kind": "page_date", "date_raw": raw})
    if name == "tineye" and isinstance(extra.get("crawl_date"), str) and extra["crawl_date"]:
        out.append({"engine": name, "date": extra["crawl_date"], "date_kind": "crawl_date", "date_raw": extra["crawl_date"]})
    return out


def _date_sentence(d: Mapping[str, Any]) -> str:
    label = LABELS.get(d["engine"], d["engine"])
    if d["date_kind"] == "crawl_date":       # 原有写法保留
        return f"TinEye first crawled this page with the image on {d['date']} (a crawl date, not a publication date)."
    if d["date_kind"] == "approx_relative":
        if d.get("date"):
            return (f"Approximate date (from \"{d['date_raw']}\" shown by {label} on {d['date_basis']}): "
                    f"about {d['date']}.")
        return f"Relative date shown by {label}: \"{d['date_raw']}\" (retrieval date unknown)."
    return f"Page date shown by {label}: {d['date_raw']}."


# ------------------------------------------------------------------ 合并与摘要


def merge_slots(lists: Mapping[str, list[CandidateSource]], order: Sequence[str]) -> list[dict[str, Any]]:
    """各引擎名次轮流合并、按 identity 去重；返回 [{"base", "found_by", "per_engine": {引擎: 该引擎的候选}}]。"""

    slots: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    depth = max((len(v) for v in lists.values()), default=0)
    for i in range(depth):
        for name in order:
            items = lists.get(name) or []
            if i >= len(items):
                continue
            c = items[i]
            ident = identity(c.source_url)
            slot = slots.get(ident)
            if slot is None:
                slots[ident] = {"base": c, "found_by": [name], "per_engine": {name: c}}
                ordered.append(ident)
            elif name not in slot["found_by"]:
                slot["found_by"].append(name)
                slot["per_engine"][name] = c
                base = slot["base"]
                if base.candidate_type != "video" and c.candidate_type == "video":
                    slot["base"] = replace(c, search_snippet=c.search_snippet or base.search_snippet)
                elif not base.search_title and c.search_title:
                    slot["base"] = replace(base, search_title=c.search_title)
    return [slots[i] for i in ordered]


def build_candidate(slot: Mapping[str, Any], verification: Mapping[str, Any] | None = None) -> CandidateSource:
    """一个合并后的网页 → 统一写法的 CandidateSource（网址去跟踪参数，摘要只写事实）。"""

    base: CandidateSource = slot["base"]
    found: list[str] = list(slot["found_by"])
    per: Mapping[str, CandidateSource] = slot["per_engine"]
    matches = {name: engine_match(name, per[name]) for name in found}
    dates = [d for name in found for d in engine_dates(name, per[name])]
    verified = (verification or {}).get("verified_match") or "unchecked"

    sentences = []
    if verified in _VERIFIED_TEXT:
        sentences.append(f"Thumbnail compared with the query image: {_VERIFIED_TEXT[verified]}.")
    sentences.append("Found by " + ", ".join(f"{LABELS.get(n, n)} ({_MATCH_TEXT[matches[n]]})" for n in found) + ".")
    sentences.extend(_date_sentence(d) for d in dates)
    tineye = (per.get("tineye").extra or {}) if per.get("tineye") is not None else {}
    earliest = tineye.get("earliest_crawl_date")
    if earliest and earliest != tineye.get("crawl_date"):
        sentences.append(f"Earliest TinEye crawl date among all copies: {earliest} (a crawl date, not a publication date).")
    lens = (per.get("lens").extra or {}) if per.get("lens") is not None else {}
    if isinstance(lens.get("source"), str) and lens["source"].strip():
        sentences.append(f"Site: {lens['source'].strip()}.")
    gcv = (per.get("gcv").extra or {}) if per.get("gcv") is not None else {}
    if gcv.get("best_guess"):
        sentences.append(f"Google's best guess for the query image: {gcv['best_guess']}.")
    yandex = per.get("yandex")
    if yandex is not None and yandex.search_snippet:
        sentences.append(f"Page text: {yandex.search_snippet.strip()}")

    url = clean_url(base.source_url)
    extra: dict[str, Any] = {**(base.extra or {}), "found_by": found, "engine_match": matches,
                             "verified_match": verified}
    for key in ("match", "lens_date"):          # 引擎各自的旧字段收进 engine_match / date，不再并列
        extra.pop(key, None)
    if verification and verified != "unchecked":
        extra["verified_scores"] = {"dhash": verification.get("dhash"), "phash": verification.get("phash")}
        extra["verified_thumbnail_url"] = verification.get("thumbnail_url")
        if verification.get("verified_engine_fallback"):        # r1：第一家引擎的缩略图没对上，用的是另一家的缩略图
            extra["verified_engine_fallback"] = True
    elif verification and verification.get("reason"):
        extra["verified_reason"] = verification["reason"]
    if dates:
        first = dates[0]
        extra.update(date=first["date"], date_kind=first["date_kind"], date_raw=first["date_raw"])
        if first.get("date_basis"):
            extra["date_basis"] = first["date_basis"]
        if len(dates) > 1:
            extra["dates"] = dates
    if tineye.get("crawl_date"):
        extra["crawl_date"] = tineye["crawl_date"]
        extra["earliest_crawl_date"] = tineye.get("earliest_crawl_date")
    if url != base.source_url:
        extra["url_tracking_removed_from"] = base.source_url
    return replace(base, source_url=url, search_snippet=" ".join(sentences), extra=extra)


def same_first(candidates: list[CandidateSource]) -> list[CandidateSource]:
    """verified_match=same 的排最前（各自保持原来的轮流顺序），其余照旧。"""

    same = [c for c in candidates if (c.extra or {}).get("verified_match") == "same"]
    rest = [c for c in candidates if (c.extra or {}).get("verified_match") != "same"]
    return same + rest


__all__ = ["DATE_KINDS", "LABELS", "MATCH_WORDS", "absolute_date", "build_candidate", "clean_url", "enabled",
           "engine_dates", "engine_match", "identity", "merge_slots", "relative_date", "same_first"]
