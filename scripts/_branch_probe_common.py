"""长尾分支"分支可达"探测的公共部分（2026-10-18，《长尾分支补量方案与三包合一前核对》）。只探测、不建卷、不跑 walker、不生成轨迹。

做法照 scripts/check_unanswerable.py：检索走 walker 同一条 search 层（run_cases.search_service_for：同一 provider 链、
同一案卷过滤 / 核查站护栏、同一页面打开器），按候选程序化确认某个分支的触发条件在检索结果里真的会出现。

费用与对外发送的边界（负责人 10-18）：
- **Google Lens（Serper）一律关**：引擎显式写 yandex（不用 auto——auto 见到 serper key 会开 Lens），并把
  FOREN_LENS_MAX_SEARCHES 设 0、清掉 FOREN_LENS_IMAGE_HOST（不传图床）。所以本组探测 Serper 0 点。
- 以图搜只发 FMNV 公开视频（YouTube 真新闻）的帧到 Yandex；候选先排除 EVID-Bench（test）全文里出现过的编号，
  EVID 的帧 / 视频一律不碰。要发帧的探测必须显式给 --allow-yandex，不给就只做文字部分。
- 候选池：FMNV 真新闻（A 族那一半哈希）、FMNV 假标题 / 真标题（C 族）、GroundLie360、AVeriTeC 训练集；都排除主目录
  cases/ 与 .wt_g1、.wt_g2 案卷里已用的编号（只读那两个工作树的案卷文件名与 source_id / family_key）。
- 输出写 claude/_longtail_work/probe_<分支>.json；C 族三个探测共用一次页面扫描（缓存 probe_c_scan.json）。
"""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
for sub in (ROOT / "src", ROOT / "scripts", SFT):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))

OUT_DIR = ROOT / "claude" / "_longtail_work"
G1_CASES = ROOT / ".wt_g1" / "experiments" / "sft_data" / "cases"
G2_CASES = ROOT / ".wt_g2" / "experiments" / "sft_data" / "cases"      # 10-18 20:43 g2 定向补的 FMNV-A 40 个也算已用
FMNV_JSON = ROOT / "plan-harness-1" / "datasets" / "FMNV" / "data.json"
FMNV_ZIP = Path(r"D:\BaiduNetdiskDownload\FMNV.zip")
G360_CSV = ROOT / "plan-harness-1" / "datasets" / "GroundLie360" / "data.csv.gz"
G360_INDEX = ROOT / "plan-harness-1" / "datasets" / "GroundLie360" / "videos_index.json"
AVERITEC = SFT / "data" / "annotations" / "averitec_train.json"
LIMIT = 10
READ_PAGE_CHARS = 6000             # = specs.read_page_chars（READ_RECORD 一页的字符数；正文更长才有 next_offset）
PAGE_CONCURRENCY = 5
# = walker._VIDEO_LINK / _PDF_NO_TEXT（只读拷贝，探测不 import walker 的私有名以外的东西）
VIDEO_LINK = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)[\w-]{6,}|youtu\.be/[\w-]{6,}|"
                        r"vimeo\.com/\d{5,}|dailymotion\.com/video/\w+)")
PDF_NO_TEXT = re.compile(r"no extractable text layer|text extraction failed|pages_with_text=0", re.I)
_STOP = set("a an the of in at on from is are was were to and or for with by as that this it its be been has have had will would "
            "can could not no but so if we you they he she i me my our your their them his her us just like get got do does did "
            "there here what which who whom when where why how all any some more most very also than then now up out into about "
            "over after before said says say going gonna want know think one two".split())


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ordered(ids: list[str], seed: str) -> list[str]:
    """= walker_d2_cases._ordered：按 sha1(seed + id) 排，结果可复现。"""

    return sorted(ids, key=lambda x: hashlib.sha1((seed + x).encode("utf-8")).hexdigest())


def no_lens_env() -> None:
    """本组探测的 Serper 硬保护：Lens 计数上限 0、不设图床。"""

    os.environ["FOREN_LENS_MAX_SEARCHES"] = "0"
    os.environ.pop("FOREN_LENS_IMAGE_HOST", None)
    os.environ["FOREN_LENS_IMAGE_FALLBACK"] = "off"
    os.environ.setdefault("FOREN_REVERSE_NORMALIZE", "1")


def serper_balance() -> int | None:
    """Serper 余额（免费接口；探测前后各读一次，证明 0 点）。读不到返回 None。"""

    try:
        import httpx

        from foren_searcher.search.providers.keys import find_key

        key = find_key("serper", ROOT)
        if not key:
            return None
        r = httpx.get("https://google.serper.dev/account", headers={"X-API-KEY": key}, timeout=20)
        return int(r.json().get("balance"))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 已用编号与 EVID

def evid_text() -> str:
    folder = ROOT / "experiments" / "evid" / "cases"
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.glob("*.json")) if folder.is_dir() else ""


def used_ids() -> set[str]:
    """主目录 cases/ 与 .wt_g1 案卷里出现过的原始编号（FMNV 视频编号、GroundLie360 编号、AVeriTeC 下标），只读。"""

    out: set[str] = set()
    for folder in (SFT / "cases", G1_CASES, G2_CASES):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for value in (case.get("source_id"), str(case.get("family_key") or "").partition(":")[2], path.stem):
                v = str(value or "")
                if not v:
                    continue
                out.add(v)
                out.add(re.sub(r"^(?:a|c|v|e|l)_", "", v))
                out.add(re.sub(r"^(?:fmnv|groundlie360|averitec|averimatec)_(?:[ac]_|c_|v)?", "", v))
    return out


# ---------------------------------------------------------------- 候选池

def fmnv_rows() -> dict[str, dict[str, Any]]:
    return {r["video_id"]: r for r in json.loads(FMNV_JSON.read_text(encoding="utf-8"))}


def fmnv_split() -> dict[str, str]:
    out: dict[str, str] = {}
    if not FMNV_ZIP.is_file():
        return out
    with zipfile.ZipFile(FMNV_ZIP) as zf:
        for split in ("train", "val", "test"):
            name = f"FMNV/data-split/vid_{split}.txt"
            try:
                for line in zf.read(name).decode("utf-8", errors="replace").split():
                    out[line.strip()] = split
            except KeyError:
                continue
    return out


def pool_fmnv_a(n: int) -> list[dict[str, Any]]:
    """A 族：FMNV 真新闻里哈希分到 A 的那一半（= walker_d2_cases.expand_pools 的 split-a），排除 EVID 与已用；FMNV 训练划分优先。"""

    rows, evid, used, split = fmnv_rows(), evid_text(), used_ids(), fmnv_split()
    real = [v for v, r in rows.items() if r.get("label") == "true" and v not in evid and v not in used]
    a_real = [v for v in real if int(hashlib.sha1(("split-a:" + v).encode()).hexdigest(), 16) % 2 == 0]
    # A 族的原片要在视频平台上找得到（案卷 url_pattern 是 youtube.com/watch?v=<id>）：只要 11 位 YouTube 编号，
    # 纯数字的是 X / Twitter 帖子编号（FMNV 真新闻里约一半），视频搜（yt-dlp）找不到
    a_real = [v for v in a_real if re.fullmatch(r"[\w-]{11}", v) and not v.isdigit()]
    ranked = ordered(a_real, "probe-a")
    ranked = [v for v in ranked if split.get(v) == "train"] + [v for v in ranked if split.get(v) != "train"]
    return [{"pool": "fmnv_real_a", "video_id": v, "title": rows[v].get("title"), "subject": rows[v].get("subject"),
             "transcript": rows[v].get("audio_transcript") or "", "fmnv_split": split.get(v)} for v in ranked[:n]]


def pool_fmnv_c(n: int) -> list[dict[str, Any]]:
    """C 族：FMNV 假标题（class ft）与真新闻里哈希分到 C 的那一半，排除 EVID 与已用。"""

    rows, evid, used = fmnv_rows(), evid_text(), used_ids()
    real = [v for v, r in rows.items() if r.get("label") == "true" and v not in evid and v not in used]
    c_real = [v for v in real if int(hashlib.sha1(("split-a:" + v).encode()).hexdigest(), 16) % 2 == 1]
    fake = [v for v, r in rows.items() if r.get("class") == "ft" and v not in evid and v not in used]
    picks = ordered(fake, "probe-cf")[: (n + 1) // 2] + ordered(c_real, "probe-cr")[: n // 2]
    return [{"pool": "fmnv_c", "id": v, "claim": rows[v].get("title"), "label": rows[v].get("label"), "gt_urls": []} for v in picks]


def pool_g360(n: int) -> list[dict[str, Any]]:
    """GroundLie360：本机有视频的条目（C 族视频声明），排除 EVID 与已用。"""

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(G360_CSV, "rt", encoding="utf-8", newline="") as stream:
        rows = {str(r["video_id"]): r for r in csv.DictReader(stream)}
    index = json.loads(G360_INDEX.read_text(encoding="utf-8")) if G360_INDEX.is_file() else {}
    evid, used = evid_text(), used_ids()
    ids = [v for v in rows if v in index and v not in evid and v not in used and (rows[v].get("title") or "").strip()]
    return [{"pool": "groundlie360", "id": v, "claim": rows[v]["title"].strip(), "label": rows[v].get("first_level_annotation"),
             "gt_urls": []} for v in ordered(ids, "probe-g360")[:n]]


def pool_averitec(n: int) -> list[dict[str, Any]]:
    """AVeriTeC 训练集（文字声明，C 族 CLAIM_TEXT）：GT 来源网址 = 各问答的 source_url；排除已有 averitec_* 案卷的下标。"""

    rows = json.loads(AVERITEC.read_text(encoding="utf-8"))
    used = used_ids()
    out = []
    for i in ordered([str(i) for i in range(len(rows))], "probe-averitec"):
        if i in used:
            continue
        row = rows[int(i)]
        urls = [a.get("source_url") for q in row.get("questions") or [] for a in q.get("answers") or [] if a.get("source_url")]
        urls = [u for u in dict.fromkeys(urls) if u and u.startswith("http")]
        if not urls:
            continue
        out.append({"pool": "averitec", "id": i, "claim": row.get("claim"), "label": row.get("label"), "gt_urls": urls[:6],
                    "fact_check_article": row.get("fact_checking_article")})
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 合成案卷（只在内存里，不写 cases/）

def make_case(kind: str, cand: dict[str, Any]) -> Any:
    """A 族照 fmnv_a_* 案卷、C 族照 groundlie360_c_* 案卷的写法拼一个 Case（search_service_for 只看 source_policy 与 case_id）。"""

    from foren_searcher.datagen.cases import Case
    from walker_d2_cases import CLAIM_TEXT_EN, CLAIM_VIDEO_EN, PROVENANCE_EN

    if kind == "A":
        data = {"case_id": f"fmnv:a_{cand['video_id']}", "dataset": "fmnv", "source_id": f"a_{cand['video_id']}",
                "split": "dev_train", "task_kind": "provenance", "task_text": PROVENANCE_EN, "task_text_en": PROVENANCE_EN,
                "family_key": f"fmnv:{cand['video_id']}", "media": [], "claims": [],
                "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None}}
    else:
        claim = str(cand.get("claim") or "")
        policy = {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None}
        if cand.get("pool") == "averitec" and cand.get("fact_check_article"):
            from check_unanswerable import norm

            policy = {"mode": "exclude_original", "excluded_keys": [norm(cand["fact_check_article"])], "date_cutoff": None}
        task = (CLAIM_TEXT_EN.format(claim=claim, date="unknown") if cand.get("pool") == "averitec"
                else CLAIM_VIDEO_EN.format(claim=claim))
        data = {"case_id": f"{cand['pool']}:probe_{cand['id']}", "dataset": cand["pool"], "source_id": f"probe_{cand['id']}",
                "split": "dev_train", "task_kind": "claim_check", "task_text": task, "task_text_en": task,
                "family_key": f"{cand['pool']}:{cand['id']}", "media": [], "claims": [claim], "source_policy": policy}
    return Case.from_dict(data)


async def open_search(case: Any, tmp: Path, engines: str) -> Any:
    """walker 同一条 search 层；engines 只许 'yandex' 或 'off'（Lens 永不开）。"""

    from foren_searcher import GlobalStore
    from foren_searcher.datagen.run_cases import search_service_for

    if engines not in ("yandex", "off"):
        raise SystemExit(f"探测只许 yandex / off，得到 {engines!r}（Lens 不开）")
    no_lens_env()
    store = GlobalStore(tmp / "store.sqlite3", tmp / "artifacts")
    search, desc, _guard = search_service_for(case, store, tmp, recording=None, image_search=engines)
    return search, store, desc


async def close_search(search: Any, store: Any) -> None:
    close = getattr(search, "aclose", None)
    if callable(close):
        try:
            await close()
        except Exception:  # noqa: BLE001
            pass
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------- A 族：查询词与帧

def youtube_id(url: str | None) -> str | None:
    m = re.search(r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/)|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else None


def transcript_query(text: str, words: int = 12) -> str:
    """S1 槽写查询词的替身：取转写里前 12 个内容词（学生在 A2 ASR 之后看得到的那段话）。"""

    toks = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+|\d{2,}", text or "") if w.lower() not in _STOP and len(w) > 2]
    return " ".join(toks[:words])


def fmnv_frame(video_id: str, target: Path) -> Path | None:
    """FMNV.zip 里预抽好的帧（FMNV/videos_frame/<id>/<帧号>.jpg）取中间那张；没有就 None。只读 zip，不解压视频。"""

    if not FMNV_ZIP.is_file():
        return None
    prefix = f"FMNV/videos_frame/{video_id}/"
    with zipfile.ZipFile(FMNV_ZIP) as zf:
        names = sorted((n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".jpg")),
                       key=lambda n: int(re.sub(r"\D", "", n.rsplit("/", 1)[1]) or 0))
        if not names:
            return None
        target.write_bytes(zf.read(names[len(names) // 2]))
    from check_unanswerable import query_jpeg

    return query_jpeg(target, target.with_name("query.jpg"))


async def video_search(search: Any, request: Any, image_path: Path | None) -> dict[str, Any]:
    try:
        batch = await asyncio.wait_for(search.provider.search(request, offset=0, limit=LIMIT, image_path=image_path), 240)
        cands = list(batch.candidates)[:LIMIT]
        return {"count": len(cands), "engine": f"{batch.provider}/{batch.engine}",
                "results": [{"rank": i, "url": c.source_url, "type": c.candidate_type, "engine": c.engine,
                             "yt": c.platform_video_id or youtube_id(c.source_url), "title": (c.search_title or "")[:120]}
                            for i, c in enumerate(cands, 1)]}
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
        return {"count": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "results": []}


def rank_of(entry: dict[str, Any], video_id: str) -> int | None:
    return next((r["rank"] for r in entry.get("results") or [] if r.get("yt") == video_id), None)


async def a_probe_one(cand: dict[str, Any], modes: list[str], allow_yandex: bool) -> dict[str, Any]:
    """A 族一个候选：按 modes（text_transcript / text_title / text_image / image）各搜一次 output_type=video，看原片编号进不进前 10。

    text_transcript = 转写前 12 个内容词（学生 A2 ASR 后可见，S1 槽写查询词的替身，偏保守）；
    text_title = FMNV 标题（学生看不到标题，只当上界——画面文字里常有标题条）；
    text_image / image = 中间一帧 + 转写查询词 / 只用帧，以图搜只走 Yandex（--allow-yandex 才发帧）。"""

    from foren_searcher.search.contracts import SearchQueryRequest

    vid = cand["video_id"]
    row: dict[str, Any] = {k: v for k, v in cand.items() if k != "transcript"}
    q_tr = transcript_query(cand.get("transcript") or "")
    q_title = " ".join(str(cand.get("title") or "").split())[:300]
    row["queries"] = {"transcript": q_tr, "title": q_title}
    needs_image = any(m in ("text_image", "image") for m in modes)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        frame = fmnv_frame(vid, tmpdir / "frame.jpg") if needs_image else None
        row["frame"] = bool(frame)
        engines = "yandex" if (needs_image and allow_yandex and frame) else "off"
        search, store, desc = await open_search(make_case("A", cand), tmpdir, engines)
        row["image_engines"] = engines
        searches: dict[str, Any] = {}
        for mode in modes:
            if mode == "text_transcript":
                if not q_tr:
                    searches[mode] = {"skipped": "no transcript"}
                    continue
                req, img = SearchQueryRequest("text", q_tr, "video"), None
            elif mode == "text_title":
                req, img = SearchQueryRequest("text", q_title, "video"), None
            elif mode in ("text_image", "image"):
                if engines != "yandex":
                    searches[mode] = {"skipped": "no frame" if not frame else "image search off (needs --allow-yandex)"}
                    continue
                req = (SearchQueryRequest("text_image", {"evidence_id": "E1", "text": q_tr or q_title}, "video") if mode == "text_image"
                       else SearchQueryRequest("image", "E1", "video"))
                img = frame
            else:
                raise ValueError(mode)
            entry = await video_search(search, req, img)
            entry["original_rank"] = rank_of(entry, vid)
            searches[mode] = entry
        await close_search(search, store)
    row["searches"] = searches
    row["original_in_top10"] = {m: s.get("original_rank") for m, s in searches.items() if "skipped" not in s}
    return row


async def a_probe(n: int, modes: list[str], allow_yandex: bool, out_name: str, redo: bool = False,
                  only: list[str] | None = None) -> dict[str, Any]:
    path = OUT_DIR / out_name
    old = {r["video_id"]: r for r in (json.loads(path.read_text(encoding="utf-8")).get("rows") or [])} if path.is_file() and not redo else {}
    before = serper_balance()
    cands = pool_fmnv_a(n)
    if only:
        cands = [c for c in cands if c["video_id"] in set(only)]
    rows = []
    for i, cand in enumerate(cands, 1):
        if cand["video_id"] in old:
            rows.append(old[cand["video_id"]])
            continue
        started = time.monotonic()
        try:
            row = await a_probe_one(cand, modes, allow_yandex)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row = {**{k: v for k, v in cand.items() if k != "transcript"}, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = now()
        rows.append(row)
        old[cand["video_id"]] = row
        print(f"[{out_name} {i}/{len(cands)}] {cand['video_id']}: 原片排名 {row.get('original_in_top10')}；{row['seconds']:.0f}s", flush=True)
        write_json(path, {"generated": now(), "rows": list(old.values())})
    after = serper_balance()
    summary = {m: {"searched": sum(1 for r in rows if m in (r.get("original_in_top10") or {})),
                   "original_in_top10": sum(1 for r in rows if (r.get("original_in_top10") or {}).get(m)),
                   "errors": sum(1 for r in rows if ((r.get("searches") or {}).get(m) or {}).get("error"))}
               for m in modes}
    payload = {"generated": now(), "modes": modes, "allow_yandex": allow_yandex, "candidates": len(rows),
               "serper_balance_before": before, "serper_balance_after": after, "summary": summary,
               "reachable": [r["video_id"] for r in rows if any((r.get("original_in_top10") or {}).values())],
               "rows": list(old.values())}
    write_json(path, payload)
    return payload


# ---------------------------------------------------------------- C 族：页面性质扫描（三个 C 探测共用）

async def page_props(opener: Any, url: str, title: str | None, rank: int | None, sem: asyncio.Semaphore,
                     candidate: Any | None = None) -> dict[str, Any]:
    """用 walker 的页面打开器打开一页：PDF 有没有文字层、正文多长（几页 READ_RECORD）、正文里有没有视频链接。"""

    from foren_searcher.search.contracts import CandidateSource, OpenSourceRequest

    out: dict[str, Any] = {"url": url, "rank": rank, "title": (title or "")[:140]}
    async with sem:
        try:
            cand = candidate or CandidateSource(url, candidate_type="page")
            opened = await asyncio.wait_for(opener.open(cand, OpenSourceRequest(url, "text")), 120)
            mats = list(opened.materials)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {str(getattr(exc, 'error', exc))[:160]}"
            return out
    text = "\n".join(m.content or "" for m in mats if m.kind in ("page", "document"))
    docs = [m for m in mats if m.kind == "document"]
    out.update({
        "kinds": [f"{m.kind}:{m.status}" for m in mats],
        "is_pdf": any(m.media_type == "application/pdf" for m in docs),
        "pdf_no_text": any(m.media_type == "application/pdf" and PDF_NO_TEXT.search((m.content or "") + " " + m.locator) for m in docs),
        "chars": len(text),
        "read_pages": (len(text) + READ_PAGE_CHARS - 1) // READ_PAGE_CHARS if text else 0,
        "video_links": sorted(set(VIDEO_LINK.findall(text)))[:5],
    })
    out["long"] = out["chars"] > READ_PAGE_CHARS
    return out


def claim_query(claim: str, words: int = 12) -> str:
    """C 族首条查询的替身：= walker C4_search_fallback（声明前 12 个词）。真实首条查询由 S3 槽写。"""

    return " ".join(str(claim or "").split()[:words])[:300] or "news report"


async def scan_c_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    from foren_searcher.search.contracts import SearchQueryRequest

    case = make_case("C", cand)
    row: dict[str, Any] = {**{k: v for k, v in cand.items() if k != "transcript"}, "case_id": case.case_id,
                           "query": claim_query(cand["claim"])}
    with tempfile.TemporaryDirectory() as tmp:
        search, store, desc = await open_search(case, Path(tmp), "off")
        row["source_policy"] = desc
        sem = asyncio.Semaphore(PAGE_CONCURRENCY)
        try:
            batch = await asyncio.wait_for(search.provider.search(SearchQueryRequest("text", row["query"], "text"),
                                                                  offset=0, limit=LIMIT, image_path=None), 240)
            cands = list(batch.candidates)[:LIMIT]
            row["search"] = {"count": len(cands), "engine": f"{batch.provider}/{batch.engine}"}
            pages = await asyncio.gather(*(page_props(search.opener, c.source_url, c.search_title, i, sem,
                                                      dataclasses.replace(c, extra={k: v for k, v in c.extra.items()
                                                                                    if k != "_search_cached_materials"}))
                                           for i, c in enumerate(cands, 1)), return_exceptions=True)
            row["pages"] = [p if isinstance(p, dict) else {"error": f"{type(p).__name__}"} for p in pages]
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row["search"] = {"count": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            row["pages"] = []
        gt = await asyncio.gather(*(page_props(search.opener, u, None, None, sem) for u in cand.get("gt_urls") or []),
                                  return_exceptions=True)
        row["gt_pages"] = [p if isinstance(p, dict) else {"error": f"{type(p).__name__}"} for p in gt]
        await close_search(search, store)
    return row


def c_candidates(n_per_pool: int, pools: list[str]) -> list[dict[str, Any]]:
    fns = {"fmnv_c": pool_fmnv_c, "groundlie360": pool_g360, "averitec": pool_averitec}
    out: list[dict[str, Any]] = []
    for name in pools:
        out += fns[name](n_per_pool)
    return out


async def c_scan(n_per_pool: int, pools: list[str], redo: bool = False) -> list[dict[str, Any]]:
    """三个 C 探测共用：结果按候选缓存在 probe_c_scan.json（重跑只补没探过的）。"""

    path = OUT_DIR / "probe_c_scan.json"
    cache = {r["case_id"]: r for r in (json.loads(path.read_text(encoding="utf-8")).get("rows") or [])} if path.is_file() and not redo else {}
    rows = []
    cands = c_candidates(n_per_pool, pools)
    for i, cand in enumerate(cands, 1):
        cid = f"{cand['pool']}:probe_{cand['id']}"
        if cid in cache:
            rows.append(cache[cid])
            continue
        started = time.monotonic()
        try:
            row = await scan_c_candidate(cand)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row = {**cand, "case_id": cid, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "pages": [], "gt_pages": []}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = now()
        rows.append(row)
        cache[cid] = row
        pages = row.get("pages") or []
        print(f"[c_scan {i}/{len(cands)}] {cid}: 召回 {len(pages)}；PDF 无文字层 {sum(1 for p in pages if p.get('pdf_no_text'))}；"
              f"长页 {sum(1 for p in pages if p.get('long'))}；带视频链接 {sum(1 for p in pages if p.get('video_links'))}；"
              f"{row['seconds']:.0f}s", flush=True)
        write_json(path, {"generated": now(), "rows": list(cache.values())})
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def c_probe_main(key: str, out_name: str, doc: str, top_default: int, note: str) -> int:
    """三个 C 探测的入口：一次页面扫描（缓存）→ 按 key 汇总 → claude/_longtail_work/<out_name>。"""

    import argparse

    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="每个池的候选个数")
    parser.add_argument("--pools", nargs="*", default=["fmnv_c", "groundlie360", "averitec"])
    parser.add_argument("--top", type=int, default=top_default, help="检索前几条算'walker 会读到'")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    before = serper_balance()
    rows = asyncio.run(c_scan(args.n, args.pools, args.redo))
    rows = [r for r in rows if r.get("pool") in args.pools]
    after = serper_balance()
    summary = c_branch_summary(rows, key, args.top)
    payload = {"generated": now(), "branch_key": key, "top": args.top, "note": note, "candidates": len(rows),
               "serper_balance_before": before, "serper_balance_after": after, **summary}
    write_json(OUT_DIR / out_name, payload)
    print(json.dumps({k: payload[k] for k in ("candidates", "top", "by_pool", "serper_balance_before", "serper_balance_after")},
                     ensure_ascii=False))
    print("可达案例：" + "、".join(r["case_id"] for r in payload["reachable"]))
    return 0


def c_branch_summary(rows: list[dict[str, Any]], key: str, top: int) -> dict[str, Any]:
    """某个页面性质（pdf_no_text / long / video_links）在检索前 top 条与 GT 页里的出现情况，按池分。"""

    by_pool: dict[str, dict[str, Any]] = {}
    reachable = []
    for r in rows:
        pool = r.get("pool", "?")
        s = by_pool.setdefault(pool, {"candidates": 0, "searched": 0, "hit_in_top": 0, "hit_in_gt": 0, "pages_opened": 0, "pages_hit": 0})
        s["candidates"] += 1
        pages = [p for p in r.get("pages") or [] if (p.get("rank") or 99) <= top]
        if (r.get("search") or {}).get("count") is not None:
            s["searched"] += 1
        s["pages_opened"] += sum(1 for p in pages if "error" not in p)
        hits = [p for p in pages if p.get(key)]
        s["pages_hit"] += len(hits)
        gt_hits = [p for p in r.get("gt_pages") or [] if p.get(key)]
        if hits:
            s["hit_in_top"] += 1
            reachable.append({"case_id": r["case_id"], "claim": (r.get("claim") or "")[:120],
                              "pages": [{"rank": p["rank"], "url": p["url"], key: p.get(key), "chars": p.get("chars")} for p in hits]})
        if gt_hits:
            s["hit_in_gt"] += 1
    return {"by_pool": by_pool, "reachable": reachable}
