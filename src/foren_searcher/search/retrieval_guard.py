"""检索护栏：核查站点黑名单 + 按声明日期截断（总 TodoList 阶段 04，漏洞审查 F4）。

两条规则都套在 SearchService 的 provider / opener 外面，不改 SEARCH 的接口：

- **黑名单**：来源域名命中核查站点（Snopes、PolitiFact、FactCheck.org、Full Fact、AFP Factuel、Lead Stories、
  Reuters / AP 的 fact-check 栏目……）时，``exclude`` 模式直接丢掉候选、拒绝打开；``mark`` 模式照常返回，但在
  candidate.extra["fact_check_site"] 打标，并写日志（测试时报"带 / 不带黑名单"两组结果用）。
- **日期截断**：案卷 source_policy.date_cutoff 为 YYYY-MM-DD 时，候选的 publishedDate、打开后页面的
  article:published_time / datePublished / <time datetime> / yt-dlp upload_date 晚于截止日的一律拒绝；
  抽不到日期的放行并打 ``date_unknown`` 标记（宁可漏过，不凭空猜日期）。

每次拒绝或打标都追加一行到 ``retrieval_guard.jsonl``（stage / url / reason / matched），供导出和论文统计
"每个 run 命中核查站点的比例"。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from .contracts import CandidateSource, OpenedSource, SearchError, SearchFailure

BlocklistMode = Literal["exclude", "mark", "off"]

# 域名按后缀匹配（snopes.com 也覆盖 www.snopes.com / es.snopes.com）；带路径的条目只拦该栏目
# （reuters.com/fact-check 不拦 Reuters 的普通新闻，因为新闻本身可以是一手来源）。
FACT_CHECK_DOMAINS: tuple[str, ...] = (
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
    "checkyourfact.com", "truthorfiction.com", "hoax-slayer.net", "hoaxeye.com",
    "factcheck.afp.com", "factuel.afp.com", "faktencheck.afp.com", "factchequeado.com", "verificado.com.mx",
    "boomlive.in", "altnews.in", "factly.in", "thequint.com/news/webqoof", "vishvasnews.com", "newschecker.in",
    "logically.ai", "logicallyfacts.com", "factcheck.kz", "maldita.es", "newtral.es", "chequeado.com",
    "aosfatos.org", "lupa.uol.com.br", "africacheck.org", "dubawa.org", "pesacheck.org",
    "mythdetector.ge", "stopfake.org", "correctiv.org/faktencheck", "mimikama.org",
    "factcheck.kr", "factchecklab.org", "tfc-taiwan.org.tw", "mygopen.com", "cofacts.tw", "piyao.org.cn",
    "reuters.com/fact-check", "apnews.com/hub/ap-fact-check", "apnews.com/ap-fact-check",
    "usatoday.com/story/news/factcheck", "washingtonpost.com/politics/fact-checker",
    "bbc.com/news/reality_check", "bbc.co.uk/news/reality_check", "bbc.com/news/bbcverify", "bbc.co.uk/news/bbcverify",
    "channel4.com/news/factcheck", "theferret.scot/fact-check", "euvsdisinfo.eu", "faktisk.no",
    "factcheck.org.au", "aap.com.au/factcheck", "rmit.edu.au/news/factlab", "abc.net.au/news/factcheck",
    "healthfeedback.org", "sciencefeedback.co", "climatefeedback.org", "poynter.org/ifcn",
    # 2026-09-29 补：案卷对照发现的二手核查（同一声明被其它核查机构 / 媒体核查栏目复核）。整站只收专职核查机构；
    # 综合媒体只收其核查栏目路径，栏目外的普通报道（如 ndtv / etvbharat 的辟谣新闻）不拦，残余泄漏风险在案卷 notes 注明。
    "youturn.in", "haqcheck.org", "factcrescendo.com", "factchecker.in", "dfrac.org", "digiteye.in", "thip.media",
    "misbar.com", "fatabyyano.net", "verafiles.org", "annielab.org",
    "newsmobile.in/nm-fact-checker", "telugupost.com/english-factcheck", "telugupost.com/fact-check",
    "timesofindia.indiatimes.com/times-fact-check", "indiatoday.in/fact-check", "newsmeter.in/fact-check",
    "thelogicalindian.com/fact-check", "rappler.com/newsbreak/fact-check",
    # 2026-10-04 补：机制对照 r0cx 里 MinnPost 的 Fact Briefs 栏目（对同一声明的二手核查）漏过了域名名单。
    "minnpost.com/fact-briefs",
)
# 2026-10-04：综合媒体的核查栏目很多（名单永远列不全），再按路径兜底——某一段路径就是核查栏目名，
# 或文章 slug 以 fact-check 开头，就当核查页。只看路径不看域名，普通新闻不受影响。命中时返回 "path:<那一段>"。
_FACT_CHECK_SEGMENT = re.compile(
    r"^(?:fact[-_]?checks?|fact[-_]?checkers?|factcheck(?:s|er|ers|ing)?|fact[-_]?briefs?|reality[-_]?check|"
    r"faktencheck|faktisk|verificacion|verificação|hoax[-_]?check|viral[-_]?check|ap[-_]fact[-_]check|"
    r"times[-_]fact[-_]check|webqoof)$|^fact[-_]?check[-_]", re.I)

_META_DATE = re.compile(
    r"""<meta[^>]+(?:property|name|itemprop)\s*=\s*["'](?:article:published_time|og:article:published_time|"""
    r"""datePublished|dateCreated|date|pubdate|publishdate|publish_date|DC\.date\.issued|sailthru\.date|"""
    r"""parsely-pub-date|article\.published|cXenseParse:recs:publishtime)["'][^>]+content\s*=\s*["']([^"']+)["']""",
    re.I)
_META_DATE_REV = re.compile(
    r"""<meta[^>]+content\s*=\s*["']([^"']+)["'][^>]+(?:property|name|itemprop)\s*=\s*["'](?:article:published_time|"""
    r"""datePublished|dateCreated|pubdate|publishdate|publish_date)["']""", re.I)
_JSONLD_DATE = re.compile(r""""datePublished"\s*:\s*"([^"]+)\"""", re.I)
_TIME_TAG = re.compile(r"""<time[^>]+datetime\s*=\s*["']([^"']+)["']""", re.I)
_ISO_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def parse_date(value: Any) -> date | None:
    """把常见日期写法解析成 date；解析不了返回 None（不猜）。"""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = _ISO_PREFIX.match(text) or _COMPACT.match(text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:40].rstrip("."), fmt).date()
        except ValueError:
            continue
    return None


def page_published_date(raw: bytes | str | None) -> date | None:
    """从页面 HTML 的 meta / JSON-LD / <time> 里抽发布日期；抽不到返回 None。"""

    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    head = text[:400_000]
    for pattern in (_META_DATE, _META_DATE_REV, _JSONLD_DATE, _TIME_TAG):
        for found in pattern.findall(head):
            parsed = parse_date(found)
            if parsed is not None:
                return parsed
    return None


def fact_check_match(url: str | None, domains: tuple[str, ...] = FACT_CHECK_DOMAINS) -> str | None:
    """URL 命中哪条黑名单条目；不命中返回 None。存档地址按原址判断。"""

    if not url:
        return None
    url = re.sub(r"^(https?://)?web\.archive\.org/web/[^/]+/", "", url.strip(), flags=re.I)
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlsplit(url)
    except ValueError:      # 从文本里抓出的残缺 URL（如 "https://[E3]"）会让 urlsplit 报 Invalid IPv6 URL：当作不命中
        return None
    host = parts.netloc.lower().split("@")[-1].split(":")[0].removeprefix("www.")
    path = parts.path.lower().rstrip("/")
    for entry in domains:
        entry_host, _, entry_path = entry.lower().partition("/")
        if host != entry_host and not host.endswith("." + entry_host):
            continue
        if not entry_path:
            return entry
        if path == "/" + entry_path or path.startswith("/" + entry_path + "/"):
            return entry
    if domains is FACT_CHECK_DOMAINS:          # 调用方自带名单时只按名单判（测试 / 特殊用途）
        for segment in path.split("/"):
            if segment and _FACT_CHECK_SEGMENT.search(segment):
                return f"path:{segment}"
    return None


@dataclass
class RetrievalGuard:
    blocklist: BlocklistMode = "exclude"
    date_cutoff: date | None = None
    log_path: Path | None = None
    domains: tuple[str, ...] = FACT_CHECK_DOMAINS
    counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.blocklist not in {"exclude", "mark", "off"}:
            raise ValueError(f"blocklist 应为 exclude / mark / off，得到 {self.blocklist!r}")
        if isinstance(self.date_cutoff, str):
            parsed = parse_date(self.date_cutoff)
            if parsed is None:
                raise ValueError(f"date_cutoff 无法解析：{self.date_cutoff!r}")
            self.date_cutoff = parsed
        self.counts = {"fact_check_excluded": 0, "fact_check_marked": 0, "after_cutoff": 0, "date_unknown": 0}

    # ------------------------------------------------------------ 记录

    def record(self, stage: str, url: str | None, reason: str, **extra: Any) -> None:
        assert self.counts is not None
        self.counts[reason] = self.counts.get(reason, 0) + 1
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": stage, "url": url, "reason": reason, **extra}, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        return {"blocklist": self.blocklist, "date_cutoff": self.date_cutoff.isoformat() if self.date_cutoff else None,
                "counts": dict(self.counts or {})}

    # ------------------------------------------------------------ 判定

    def candidate_decision(self, candidate: CandidateSource) -> tuple[CandidateSource | None, str | None]:
        """返回 (保留的候选或 None, 拒绝原因)。mark 模式只打标不丢。"""

        matched = fact_check_match(candidate.source_url, self.domains) if self.blocklist != "off" else None
        if matched and self.blocklist == "exclude":
            return None, "fact_check_excluded"
        extra = dict(candidate.extra)
        if matched:
            extra["fact_check_site"] = matched
        if self.date_cutoff is not None:
            published = parse_date(extra.get("publishedDate"))
            if published is not None and published > self.date_cutoff:
                return None, "after_cutoff"
            if published is None and not candidate.extra.get("date_unknown"):
                extra["date_unknown"] = True
                assert self.counts is not None
                self.counts["date_unknown"] += 1          # 只计数不逐条写日志（打开时若仍抽不到日期才写）
        return (candidate if extra == candidate.extra else replace(candidate, extra=extra)), None

    def opened_decision(self, candidate: CandidateSource, opened: OpenedSource) -> str | None:
        """打开之后再看一次：重定向目标、页面里的发布日期、视频的上传日期。返回拒绝原因或 None。"""

        if self.blocklist == "exclude" and fact_check_match(opened.final_url, self.domains):
            return "fact_check_excluded"
        if self.date_cutoff is None:
            return None
        published: date | None = None
        video = opened.video_material or {}
        for key in ("upload_date", "release_date"):
            published = published or parse_date(video.get(key))
        if published is None:
            for material in opened.materials:
                if material.kind in {"page", "document"} and material.artifact_bytes:
                    published = page_published_date(material.artifact_bytes)
                    if published is not None:
                        break
        if published is None:
            published = parse_date(candidate.extra.get("publishedDate"))
        if published is not None and published > self.date_cutoff:
            return "after_cutoff"
        if published is None:
            self.record("source_open", opened.final_url, "date_unknown")
        return None


class GuardedProvider:
    def __init__(self, provider: Any, guard: RetrievalGuard) -> None:
        self.provider, self.guard = provider, guard

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> Any:
        batch = await self.provider.search(request, offset=offset, limit=limit, image_path=image_path)
        kept = []
        for candidate in batch.candidates:
            decided, reason = self.guard.candidate_decision(candidate)
            if decided is None:
                self.guard.record("provider_candidate", candidate.source_url, reason or "excluded",
                                  matched=fact_check_match(candidate.source_url, self.guard.domains))
                continue
            if decided.extra.get("fact_check_site") and not candidate.extra.get("fact_check_site"):
                self.guard.record("provider_candidate", candidate.source_url, "fact_check_marked",
                                  matched=decided.extra["fact_check_site"])
            kept.append(decided)
        return replace(batch, candidates=tuple(kept))


class GuardedOpener:
    def __init__(self, opener: Any, guard: RetrievalGuard) -> None:
        self.opener, self.guard = opener, guard

    async def open(self, candidate: CandidateSource, request: Any) -> OpenedSource:
        decided, reason = self.guard.candidate_decision(candidate)
        if decided is None:
            self.guard.record("source_open", candidate.source_url, reason or "excluded",
                              matched=fact_check_match(candidate.source_url, self.guard.domains))
            raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED", _message(reason)))
        opened = await self.opener.open(candidate, request)
        reason = self.guard.opened_decision(candidate, opened)
        if reason is not None:
            self.guard.record("source_open", candidate.source_url, reason, final_url=opened.final_url,
                              matched=fact_check_match(opened.final_url, self.guard.domains))
            raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED", _message(reason)))
        return opened


def _message(reason: str | None) -> str:
    if reason == "after_cutoff":
        return "This source was published after the claim date and is outside the investigation's time window"
    return "Fact-checking sites are excluded from this investigation's evidence sources; find primary sources instead"


def apply_retrieval_guard(service: Any, guard: RetrievalGuard) -> Any:
    """把护栏套在 SearchService 的 provider 与 opener 外面（可叠在案卷的 FilteredProvider 之上）。"""

    if guard.blocklist == "off" and guard.date_cutoff is None:
        return service
    service.provider = GuardedProvider(service.provider, guard)
    service.opener = GuardedOpener(service.opener, guard)
    return service


def context_mentions_fact_check(text: str, domains: tuple[str, ...] = FACT_CHECK_DOMAINS) -> list[str]:
    """给已有训练行打 contaminated_context 用：文本里出现了哪些核查站点域名。"""

    lowered = text.lower()
    hosts = sorted({entry.split("/", 1)[0] for entry in domains})
    return [host for host in hosts if host in lowered]
