"""r1 建卷：E / L 族"可回答"探测（负责人 10-17 定的新定义）。

可回答 = GT 可回答，且引擎前 N（=10）条召回里有"载图页"：
- 以图搜一次（E：Google Lens + Yandex 合并后前 10；L：Yandex 前 10，output_type=image）、文字搜（ddgs 前 10，查询词由
  GT 关键词生成：GT 页面标题，L 族标题里没有 GT 地点时补上地点；L 族另加一条"声明画面词 + GT 地点"——L 的 GT 来源页常是地点
  介绍页、不载这张图）。任一次命中即可回答。
- 载图页 = 页面上的图（引擎原图 → 缩略图 → 页内主图，取到一张比上为止）与给定图比对 same / edited（与 walker 的 compare_media
  图片模式同一套：SIFT 配准后逐像素差异，差异面积 ≤ 2% same、≤ 35% edited，配准没做成且 > 10% different），且页面文字含 GT 要点
  （E：页面声明了发布日期 / 平台上传日期；L：页面文字写明 GT 地点或其别名）。
- GT 可回答：E 有标注的最早来源网址；L 案卷 checks 里有 entity_in_evidence 的 GT 地点。

探测走 walker 同一条 search 层（run_cases.search_service_for：同一 lens_cache、同一 FOREN_REVERSE_NORMALIZE 归一化与缩略图自核、
案卷的 exclude_original 排除、核查站护栏），以图搜的查询图按 SearchService._prepare_query_image 同一办法转成 ≤ 500 KiB JPEG——
Lens 硬盘缓存按这份 JPEG 的 sha256 存，全量时 walker 对同一张图的以图搜直接命中，不再扣 Serper。不录检索快照（walker 看不到探测）。
Serper：Lens 只对 E 开、每张图 ≤ 1 次（FOREN_LENS_MAX_SEARCHES=1），全程另有 --lens-budget 次数上限（缓存命中不算）；
图床 imgbb 只在 E 族的探测里开（同 walker_run.r1_search_policy）。

输出 experiments/sft_data/r1_answerable_probe.json（每个候选的漏斗：以图搜 / 文字搜各自的召回数、取到图的页数、比对结果分布、
GT 要点匹配数、载图页清单），不改案卷文件。（任务里写的 cases/r1_answerable_probe.json 会被 export 的 load_cases 当成案卷读，
所以放在 sft_data 根下。）

    python scripts/check_unanswerable.py [--cases averimatec_e120 …] [--extra-dirs claude/_r0plus_work/unused_cases] [--lens-budget 66]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
for sub in (ROOT / "src", ROOT / "scripts", SFT):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

OUT = SFT / "r1_answerable_probe.json"
LIMIT = 10
LENS_COST = 3                      # Serper：一次 Lens = 3 点（没有结果也扣）
PAGE_CONCURRENCY = 5
L_QUERY = "both"                   # --l-query
SAME_AREA, EDITED_AREA = 0.02, 0.35   # = walker._SAME_AREA / _EDITED_AREA


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- GT（同 r0g 的 check_unanswerable）

def norm(url: str) -> str:
    from walker_d2_cases import _strip_archive

    u = _strip_archive(html.unescape(url or ""))
    m = re.search(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/)|youtu\.be/)([\w-]{6,})", u)
    if m:
        return f"youtube.com/watch?v={m.group(1)}"
    return u.lower().split("?", 1)[0].split("#", 1)[0].rstrip("/")


def url_words(gt: str) -> str:
    path = gt.split("/", 1)[1] if "/" in gt else gt
    words = [w for w in re.split(r"[^0-9A-Za-z]+", path) if len(w) >= 3 and not w.isdigit() and not re.fullmatch(r"[0-9a-f]{8,}", w)]
    return " ".join(words[:14]) or gt


_SHORTENERS = re.compile(r"^(?:https?://)?(?:www\.)?(?:cutt\.ly|bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|is\.gd|buff\.ly|rb\.gy)/", re.I)


def resolve_short(raw: str) -> str:
    if not _SHORTENERS.match(raw.strip()):
        return raw
    import httpx

    url = raw.strip() if raw.strip().startswith("http") else "https://" + raw.strip()
    try:
        with httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
            final = str(client.get(url).url)
    except Exception:  # noqa: BLE001
        return raw
    return final if final and not _SHORTENERS.match(final) else raw


def gt_source(case: dict[str, Any], rows: list[dict[str, Any]], e_sources: dict[int, str]) -> str | None:
    sid = case["source_id"]
    index = int(sid[1:])
    if sid.startswith("e"):
        raw = e_sources.get(index)
        return resolve_short(raw) if raw else None
    row = rows[index]
    question = next((q for q in row.get("questions") or [] if re.search(r"where|location|place|city", q.get("question") or "", re.I)), {})
    raw = ((question.get("answers") or [{}])[0]).get("source_url") or ""
    return raw if raw.strip() else None


def gt_place(case: dict[str, Any]) -> list[str]:
    """L 族 GT 地点（checks 里 entity_in_evidence 的 text + aliases）。"""

    for check in case.get("checks") or []:
        if check.get("kind") == "entity_in_evidence":
            params = check.get("params") or {}
            names = [params.get("text"), *(params.get("aliases") or [])]
            return [n for n in dict.fromkeys(str(x).strip() for x in names if x) if n]
    return []


_ERROR_TITLE = re.compile(r"wayback machine|just a moment|access denied|404|400 bad request|bad request|not found|attention required|"
                          r"request could not be satisfied|^error\b|forbidden|captcha|are you a robot|page unavailable|"
                          r"site can.t be reached", re.I)


def page_title(raw: str) -> str | None:
    import httpx

    candidates = [raw.strip()] if raw.startswith("http") else ["https://" + raw.strip(), "http://" + raw.strip()]
    if "web.archive.org" not in raw:
        candidates.append("https://web.archive.org/web/2024/" + re.sub(r"^https?://", "", raw.strip()))
    for url in candidates:
        try:
            with httpx.Client(timeout=25, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
                text = client.get(url).text[:400000]
        except Exception:  # noqa: BLE001
            continue
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        if m:
            title = " ".join(html.unescape(m.group(1)).split())
            if title and not _ERROR_TITLE.search(title):
                return title[:300]
    return None


_CLAIM_DROP = re.compile(r"(?i)\b(?:image|picture|photo|photos|pictures|shows?|showing|said to be|a|an|the|of|in|at|on|from|"
                         r"is|are|was|were|taken|after|being|its|it|which|where|who|that|this|by|with|and|for|to)\b")


def claim_place_query(claim: str, places: list[str]) -> str:
    """L 族：声明里画面内容的普通词（去掉虚词、引子、专名，前 8 个）+ GT 地点。"""

    # 声明里大写开头的词多是声明点名的地点 / 人名 / 日期（Joshimath、Capitol、January），去掉，只留画面内容的普通词
    words = [w for w in re.split(r"[^\w'-]+", _CLAIM_DROP.sub(" ", claim)) if len(w) > 2 and not w[0].isupper() and not w[0].isdigit()][:8]
    return " ".join(words + [places[0]])


def text_query(gt_norm: str, title: str | None, family: str, places: list[str], claim: str = "", l_query: str = "title") -> tuple[str, str]:
    if family == "L" and l_query == "claim" and places:
        return claim_place_query(claim, places)[:480], "claim+place"
    domain = gt_norm.split("/", 1)[0]
    if title and (len(title.split()) < 3 or domain.split(".")[0] in title.lower().replace(" ", "")[:len(domain) + 3]
                  and len(title.split()) <= 4 or re.fullmatch(r"(?i)maintenance|youtube|home|vk\.com \| vk", title)):
        title = None
    query = title or (url_words(gt_norm) if "youtube.com/watch" not in gt_norm else gt_norm)
    kind = "title" if title else "url"
    if family == "L" and places and not any(p.lower() in query.lower() for p in places):
        query = f"{query} {places[0]}"
        kind += "+place"
    return query[:480], kind


def hit_rank(candidates: list[str], gt: str) -> int | None:
    from urllib.parse import unquote

    path = gt.split("/", 1)[1] if "/" in gt else ""
    for rank, url in enumerate(candidates, 1):
        c = norm(url)
        full = unquote(html.unescape(url or "")).lower()
        if c and (c == gt or c.startswith(gt) or (len(c) > 15 and gt.startswith(c) and "/" in c)):
            return rank
        if len(path) >= 12 and path in full:
            return rank
    return None


# ---------------------------------------------------------------- 查询图与比对

def query_jpeg(source: Path, target: Path) -> Path:
    """= SearchService._prepare_query_image：≤ 2048 像素、≤ 500 KiB 的 JPEG（Lens 缓存键是它的 sha256）。"""

    from PIL import Image

    with Image.open(source) as image:
        image.load()
        converted = image.convert("RGB")
        converted.thumbnail((2048, 2048))
        for quality in (90, 80, 70, 60, 50):
            converted.save(target, "JPEG", quality=quality, optimize=True)
            if target.stat().st_size <= 500 * 1024:
                break
            converted.thumbnail((max(320, int(converted.width * 0.8)), max(320, int(converted.height * 0.8))))
    return target


def lens_cached(jpeg: Path) -> bool:
    from foren_searcher.datagen.run_cases import lens_cache_dir

    cache = lens_cache_dir(ROOT)
    key = hashlib.sha256(jpeg.read_bytes()).hexdigest()
    return bool(cache and (cache / key[:2] / f"{key}_en.json").is_file())


def compare_images(path_a: Path, data_b: bytes) -> dict[str, Any]:
    """compare_media 图片模式的核心（backends.local._diff_images：配准请求开、无区域）+ walker._compare_class 的分级。"""

    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    import io

    from foren_searcher.inspect import media_diff

    frames = []
    try:
        for raw_bytes in (path_a.read_bytes(), data_b):
            with Image.open(io.BytesIO(raw_bytes)) as raw:
                rgb = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:  # noqa: BLE001
        return {"class": "failed", "area": None, "error": f"decode: {type(exc).__name__}"}
    a_native, b_native = frames
    registration = media_diff.estimate_registration([(a_native, b_native)], (a_native.shape[1], a_native.shape[0]),
                                                    (b_native.shape[1], b_native.shape[0]), True)
    compare = media_diff.region_mask(registration.a_size, None) & registration.valid_mask()
    if not compare.any():
        return {"class": "failed", "area": None, "error": "no overlap after registration"}
    aligned, _ = registration.align(b_native)
    diff = media_diff.difference(media_diff.blurred(media_diff.resize(a_native, registration.a_size)), media_diff.blurred(aligned))
    threshold, _noise = media_diff.estimate_threshold([(diff, compare)])
    mask = media_diff.changed_mask(diff, compare, threshold)
    area, _bbox = media_diff.measure(mask, int(compare.sum()))
    info = registration.public() if hasattr(registration, "public") else {}
    if info.get("requested") and info.get("applied") is False and area > 0.10:
        cls = "different"
    else:
        cls = "same" if area <= SAME_AREA else "edited" if area <= EDITED_AREA else "different"
    return {"class": cls, "area": round(float(area), 4), "registered": bool(info.get("applied")), "inliers": info.get("inliers")}


# ---------------------------------------------------------------- 页面

_DATE_LINE = re.compile(r"- (published date declared by the page|platform upload date): (\d{4}-\d{2}-\d{2})")


def gt_point(family: str, text: str, raw_html: bytes | None, places: list[str]) -> dict[str, Any]:
    if family == "E":
        from foren_searcher.search.openers.dates import page_dates

        declared = page_dates(raw_html) if raw_html else {"published": None}
        return {"match": bool(declared.get("published")), "published": declared.get("published")}
    low = text.lower()
    hits = [p for p in places if re.search(r"(?<![a-z])" + re.escape(p.lower()) + r"(?![a-z])", low)]
    return {"match": bool(hits), "places": hits}


async def platform_date(opener: Any, url: str) -> str | None:
    """视频平台页（YouTube 等）的上传日期：走 walker 的 SEARCH dates 同一条链。"""

    from foren_searcher.search.contracts import CandidateSource, OpenSourceRequest

    try:
        opened = await asyncio.wait_for(opener.open(CandidateSource(url, candidate_type="page"),
                                                    OpenSourceRequest(url, "text", ("dates",))), 90)
    except Exception:  # noqa: BLE001
        return None
    for material in opened.materials:
        for m in _DATE_LINE.finditer(material.content or ""):
            return m.group(2)
    return None


async def eval_page(opener: Any, candidate: Any, family: str, image: Path, places: list[str], sem: asyncio.Semaphore) -> dict[str, Any]:
    """一页：取图（引擎原图 → 缩略图 → 页内主图）比对，读页文字查 GT 要点。"""

    from foren_searcher.search.contracts import OpenSourceRequest

    out: dict[str, Any] = {"url": candidate.source_url, "rank": candidate.provider_rank, "title": (candidate.search_title or "")[:160],
                           "images": [], "compare": None, "gt": None}
    async with sem:
        try:
            request = OpenSourceRequest(candidate.source_url, "image")
        except ValueError:
            out["error"] = "bad url"
            return out
        try:
            opened = await asyncio.wait_for(opener.open(candidate, request), 150)
            materials = list(opened.materials)
        except Exception as exc:  # noqa: BLE001
            materials = []
            out["error"] = f"{type(exc).__name__}: {str(getattr(exc, 'error', exc))[:160]}"
        text, raw_html = "", None
        pictures: list[tuple[str, bytes]] = []
        for m in materials:
            if m.kind in ("page", "document") and m.status in ("ready", "partial"):
                text += "\n" + (m.content or "")
                raw_html = raw_html or (m.artifact_bytes if m.kind == "page" else None)
            elif m.kind in ("image", "thumbnail") and m.status == "ready" and m.artifact_bytes:
                pictures.append((m.kind if m.kind == "thumbnail" else ("page_main_image" if "page main image" in (m.locator or "")
                                                                         else "engine_image"), m.artifact_bytes))
            elif m.kind in ("image", "thumbnail"):
                out["images"].append({"slot": m.kind, "status": m.status,
                                      "error": getattr(getattr(m, "error", None), "code", None)})
        best = None
        for slot, data in pictures:
            result = await asyncio.to_thread(compare_images, image, data)
            out["images"].append({"slot": slot, "status": "ready", **result})
            if best is None or ["failed", "different", "edited", "same"].index(result["class"]) > \
                    ["failed", "different", "edited", "same"].index(best["class"]):
                best = {**result, "slot": slot}
            if result["class"] in ("same", "edited"):
                break
        if (best is None or best["class"] not in ("same", "edited")) and candidate.image_url and text:
            # 引擎原图 / 缩略图没比上：再取这一页的页内主图（walker 的 E6a / L6a 同一个打开器）
            page_cand = dataclasses.replace(candidate, image_url=None, thumbnail_url=None, candidate_type="page",
                                            extra={k: v for k, v in candidate.extra.items() if k != "_search_cached_materials"})
            try:
                main = await asyncio.wait_for(opener.open(page_cand, request), 150)
                for m in main.materials:
                    if m.kind == "image" and m.status == "ready" and m.artifact_bytes:
                        result = await asyncio.to_thread(compare_images, image, m.artifact_bytes)
                        out["images"].append({"slot": "page_main_image", "status": "ready", **result})
                        if best is None or ["failed", "different", "edited", "same"].index(result["class"]) > \
                                ["failed", "different", "edited", "same"].index(best["class"]):
                            best = {**result, "slot": "page_main_image"}
                    elif m.kind == "image":
                        out["images"].append({"slot": "page_main_image", "status": m.status,
                                              "error": getattr(getattr(m, "error", None), "code", None)})
            except Exception as exc:  # noqa: BLE001
                out["images"].append({"slot": "page_main_image", "status": "failed", "error": type(exc).__name__})
        out["compare"] = best
        gt = gt_point(family, text, raw_html, places)
        if family == "E" and not gt["match"] and re.search(r"youtube\.com|youtu\.be|vimeo\.com|dailymotion|tiktok|rumble|bilibili",
                                                           candidate.source_url, re.I):
            upload = await platform_date(opener, candidate.source_url)
            gt = {"match": bool(upload), "published": None, "upload": upload}
        out["gt"] = gt
        out["text_chars"] = len(text)
        out["carrying"] = bool(best and best["class"] in ("same", "edited") and gt["match"])
    return out


def funnel(entry: dict[str, Any]) -> dict[str, Any]:
    pages = entry.get("pages") or []
    classes: dict[str, int] = {}
    for p in pages:
        c = (p.get("compare") or {}).get("class") or "no_image"
        classes[c] = classes.get(c, 0) + 1
    return {"recalled": entry.get("count"), "pages_evaluated": len(pages),
            "image_obtained": sum(1 for p in pages if p.get("compare")),
            "compare": classes,
            "same_or_edited": sum(1 for p in pages if (p.get("compare") or {}).get("class") in ("same", "edited")),
            "gt_point": sum(1 for p in pages if (p.get("gt") or {}).get("match")),
            "carrying": sum(1 for p in pages if p.get("carrying"))}


# ---------------------------------------------------------------- 一个候选

class LensBudget:
    def __init__(self, limit: int) -> None:
        self.limit, self.used, self.cache_hits, self.skipped = limit, 0, 0, []


def media_path(case_file: Path, rel: str) -> Path:
    for base in (case_file.parent, SFT / "cases"):
        p = (base / rel).resolve()
        if p.is_file():
            return p
    return (case_file.parent / rel).resolve()


async def one(case_file: Path, rows: list[dict[str, Any]], e_sources: dict[int, str], budget: LensBudget) -> dict[str, Any]:
    from foren_searcher import GlobalStore
    from foren_searcher.datagen.cases import Case
    from foren_searcher.datagen.run_cases import search_service_for
    from foren_searcher.search.contracts import SearchQueryRequest

    data = json.loads(case_file.read_text(encoding="utf-8"))
    case = Case.from_dict(data)
    family = "E" if case.source_id.startswith("e") else "L"
    row: dict[str, Any] = {"case_id": case.case_id, "case": case_file.stem, "family": family,
                           "case_file": str(case_file.relative_to(ROOT)).replace("\\", "/")}
    raw = gt_source(data, rows, e_sources)
    places = gt_place(data) if family == "L" else []
    gt_ok = bool(raw) and (family == "E" or bool(places))
    row.update({"gt_url": raw, "gt_norm": norm(raw) if raw else None, "gt_places": places, "gt_answerable": gt_ok})
    if not gt_ok:
        row.update({"answerable": False, "reason": "GT 不可回答（标注里没有 GT 来源网址 / GT 地点）", "searches": []})
        return row
    title = await asyncio.to_thread(page_title, raw)
    claim = (data.get("claims") or [""])[0]
    query, kind = text_query(row["gt_norm"], title, family, places, claim, "title")
    queries = [("text", query, kind)]
    if family == "L" and L_QUERY == "both" and places:
        # L 族另一条：声明画面词 + GT 地点（GT 来源页常是地点的介绍页、不载这张图，例如 l623 的堪萨斯议会大厦导览页）
        q2, k2 = text_query(row["gt_norm"], title, family, places, claim, "claim")
        queries.append(("text_claim", q2, k2))
    row.update({"gt_title": title, "text_query": query, "text_query_kind": kind,
                "text_queries": [{"kind": k, "query": q, "query_kind": qk} for k, q, qk in queries]})
    image = media_path(case_file, case.media[0].path)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        jpeg = query_jpeg(image, out / "query.jpg")
        engines = "yandex"
        lens_note = None
        os.environ["FOREN_REVERSE_NORMALIZE"] = "1"
        os.environ.pop("FOREN_LENS_IMAGE_HOST", None)
        if family == "E":
            cached = lens_cached(jpeg)
            if cached:
                engines, lens_note = "lens,yandex", "cache_hit"
                budget.cache_hits += 1
            elif budget.used < budget.limit:
                engines, lens_note = "lens,yandex", "billed"
                budget.used += 1
                os.environ["FOREN_LENS_IMAGE_HOST"] = "imgbb"
                os.environ["FOREN_LENS_IMAGE_FALLBACK"] = "off"
            else:
                lens_note = "skipped_budget"
                budget.skipped.append(case_file.stem)
        os.environ["FOREN_LENS_MAX_SEARCHES"] = "1"
        row.update({"image_engines": engines, "lens": lens_note, "query_jpeg_sha256": hashlib.sha256(jpeg.read_bytes()).hexdigest()})
        store = GlobalStore(out / "store.sqlite3", out / "artifacts")
        search, desc, _ = search_service_for(case, store, out, recording=None, image_search=engines)
        row["source_policy"] = desc
        sem = asyncio.Semaphore(PAGE_CONCURRENCY)
        searches = []
        for skind, request, path in (("image", SearchQueryRequest("image", "E1", "image"), jpeg),
                                     *((k, SearchQueryRequest("text", q, "text"), None) for k, q, _qk in queries)):
            entry: dict[str, Any] = {"kind": skind}
            try:
                batch = await asyncio.wait_for(search.provider.search(request, offset=0, limit=LIMIT, image_path=path), 240)
                cands = list(batch.candidates)[:LIMIT]
                entry.update({"count": len(cands), "engine": f"{batch.provider}/{batch.engine}",
                              "gt_url_rank": hit_rank([c.source_url for c in cands], row["gt_norm"])})
                if skind == "image":
                    errors = (batch.raw_response or {}).get("errors") if isinstance(getattr(batch, "raw_response", None), dict) else None
                    if errors:
                        entry["engine_errors"] = {k: str(v)[:160] for k, v in errors.items()}
                entry["pages"] = list(await asyncio.gather(*(eval_page(search.opener, c, family, image, places, sem) for c in cands),
                                                           return_exceptions=True))
                entry["pages"] = [p if isinstance(p, dict) else {"url": None, "error": f"{type(p).__name__}: {p}"[:200], "compare": None,
                                                                 "gt": None, "images": []} for p in entry["pages"]]
            except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                entry.update({"count": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "pages": []})
            entry["funnel"] = funnel(entry)
            searches.append(entry)
        if family == "E" and lens_note == "billed" and not lens_cached(jpeg):
            row["lens"] = "billed_no_cache"      # 发出了但没写进缓存（报错）：仍按扣过算
        close = getattr(search, "aclose", None)
        if callable(close):
            await close()
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
    row["searches"] = searches
    hits = [s["kind"] for s in searches if (s.get("funnel") or {}).get("carrying")]
    ok = [s for s in searches if s.get("count") is not None]
    if hits:
        row["answerable"] = True
        row["reason"] = "载图页：" + "、".join(f"{s['kind']} {s['funnel']['carrying']} 页" for s in searches if s["kind"] in hits)
    elif not ok:
        row["answerable"] = None
        row["reason"] = "两次检索都报错，未判定"
    else:
        row["answerable"] = False
        row["reason"] = "前 10 召回里没有载图页（" + "；".join(
            f"{s['kind']} 召回 {s['funnel']['recalled']}、比上 {s['funnel']['same_or_edited']}、GT 要点 {s['funnel']['gt_point']}"
            for s in searches) + "）"
    row["carrying_pages"] = [{"search": s["kind"], "rank": p["rank"], "url": p["url"], "compare": p["compare"], "gt": p["gt"]}
                             for s in searches for p in s.get("pages") or [] if p.get("carrying")]
    return row


def summarize(rows: list[dict[str, Any]], budget: LensBudget | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fam in ("E", "L"):
        fr = [r for r in rows if r["family"] == fam]
        stage = {"candidates": len(fr), "gt_answerable": sum(1 for r in fr if r.get("gt_answerable")),
                 "image_carrying": sum(1 for r in fr if any(s["kind"] == "image" and (s.get("funnel") or {}).get("carrying")
                                                            for s in r.get("searches") or [])),
                 "text_carrying": sum(1 for r in fr if any(s["kind"].startswith("text") and (s.get("funnel") or {}).get("carrying")
                                                           for s in r.get("searches") or [])),
                 "any_same_or_edited": sum(1 for r in fr if any((s.get("funnel") or {}).get("same_or_edited")
                                                                for s in r.get("searches") or [])),
                 "answerable": sum(1 for r in fr if r.get("answerable") is True),
                 "unanswerable": sum(1 for r in fr if r.get("answerable") is False),
                 "undetermined": sum(1 for r in fr if r.get("answerable") is None)}
        out[fam] = stage
    if budget is not None:
        out["lens"] = {"billed_searches": budget.used, "credits_estimate": budget.used * LENS_COST, "cache_hits": budget.cache_hits,
                       "skipped_budget": budget.skipped, "budget": budget.limit}
    return out


async def main_async(args: argparse.Namespace) -> int:
    from walker_d2_cases import averimatec_picks

    rows = json.loads((SFT / "data" / "annotations" / "averimatec_train.json").read_text(encoding="utf-8"))
    e_sources = {i: (a.get("source_url") or s) for i, _r, a, s in averimatec_picks()}
    files: list[Path] = []
    if args.cases:
        for name in args.cases:
            hit = [p for d in [SFT / "cases", *(ROOT / x for x in args.extra_dirs)] for p in [d / f"{name}.json"] if p.is_file()]
            if not hit:
                raise SystemExit(f"找不到案卷 {name}")
            files.append(hit[0])
    else:
        files = sorted((SFT / "cases").glob("averimatec_[el]*.json"))
        for d in args.extra_dirs:
            files += sorted((ROOT / d).glob("averimatec_[el]*.json"))
    seen: set[str] = set()
    files = [f for f in files if not (f.stem in seen or seen.add(f.stem))]
    out_path = Path(args.out)
    old = json.loads(out_path.read_text(encoding="utf-8")).get("rows") if out_path.is_file() and not args.fresh else []
    done = {r["case"]: r for r in old or []}
    budget = LensBudget(args.lens_budget - sum(1 for r in done.values() if str(r.get("lens") or "").startswith("billed")))
    results = dict(done)

    def write() -> None:
        ordered = sorted(results.values(), key=lambda r: (r["family"], int(re.sub(r"\D", "", r["case"].split("_")[-1]) or 0)))
        payload = {"generated": now(), "method": __doc__.split("\n\n")[0] + "\n" + __doc__.split("\n\n")[1], "limit": LIMIT,
                   "lens_budget": args.lens_budget, "summary": summarize(ordered, None),
                   "lens": {"billed_searches": sum(1 for r in ordered if str(r.get("lens") or "").startswith("billed")),
                            "cache_hits": sum(1 for r in ordered if r.get("lens") == "cache_hit"),
                            "skipped_budget": [r["case"] for r in ordered if r.get("lens") == "skipped_budget"]},
                   "answerable_cases": [r["case"] for r in ordered if r.get("answerable") is True],
                   "unanswerable_cases": [r["case"] for r in ordered if r.get("answerable") is False],
                   "rows": ordered}
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    for i, path in enumerate(files, 1):
        if path.stem in done and not args.redo:
            continue
        started = time.monotonic()
        try:
            row = await one(path, rows, e_sources, budget)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 —— 10-17：一次 TLS 读被取消漏出 CancelledError，整批停了
            row = {"case": path.stem, "case_id": None, "family": "E" if path.stem.split("_")[-1].startswith("e") else "L",
                   "answerable": None, "reason": f"探测出错：{type(exc).__name__}: {str(exc)[:300]}", "searches": []}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = now()
        results[path.stem] = row
        f = {s["kind"]: s.get("funnel") for s in row.get("searches") or []}
        print(f"[{i}/{len(files)}] {path.stem}: answerable={row['answerable']}；{row['reason']}；lens={row.get('lens')}；"
              f"{row['seconds']:.0f}s；image {f.get('image')}；text {f.get('text')}", flush=True)
        write()
    write()
    print(json.dumps(summarize(list(results.values()), budget), ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", default=None, help="案卷名（不带 .json）；默认 cases/ 下全部 averimatec_e* / l* 加 --extra-dirs")
    parser.add_argument("--extra-dirs", nargs="*", default=[], help="另外的候选案卷目录（相对仓库根），例如 claude/_r0plus_work/unused_cases")
    parser.add_argument("--lens-budget", type=int, default=66, help="本次探测 Lens 实发次数上限（缓存命中不算；已有结果里扣过的算进去）")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--redo", action="store_true", help="已有结果的候选也重探")
    parser.add_argument("--l-query", choices=["title", "both"], default="both", help="L 族文字搜：只用 GT 标题（+地点），或另加一条 声明画面词 + GT 地点（默认）")
    parser.add_argument("--fresh", action="store_true", help="不读已有输出")
    args = parser.parse_args()
    global L_QUERY
    L_QUERY = args.l_query
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
