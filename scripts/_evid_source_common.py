"""找原片实验 A6 / A7 两个脚本共用的东西（10-06）：样本集、GT 读取、走真实后端的视频搜索、绕开桥接的直接搜索。

搜索一律用 run 里同一个 YtDlpSearchProvider（经 SafeHttpClient 桥接）；direct_search 用裸 yt_dlp（不经桥接），
只在桥接返回 0 条时补查一次，用来判断"零结果"是 YouTube 本身还是我们的桥接造成的。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.search.contracts import SearchFailure, SearchQueryRequest  # noqa: E402
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider  # noqa: E402

DATASET = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "evid_bench.json"
SCORES = ROOT / "experiments" / "evid" / "scores"
AB_IDS = tuple(f"video_{2 + round(i * 128 / 19):03d}" for i in range(20))     # 与 _launch_evid.EVID_AB_IDS 相同
FORBIDDEN_FROM = 131                                                          # 131–222 不碰


def load_sources(ids: list[str]) -> dict[str, list[str]]:
    for sample in ids:
        if int(sample.split("_")[-1]) >= FORBIDDEN_FROM:
            raise SystemExit(f"{sample}：找原片实验不碰 131–222")
    entries = {e["id"]: e for e in json.loads(DATASET.read_text(encoding="utf-8"))}
    missing = [i for i in ids if i not in entries]
    if missing:
        raise SystemExit(f"evid_bench.json 里没有：{missing}")
    return {i: list(entries[i].get("source") or []) for i in ids}


async def provider_search(provider: YtDlpSearchProvider, query: str, limit: int = 10) -> tuple[list[str], str | None]:
    """返回 (视频 ID 列表按排名, 错误)。"""

    try:
        batch = await provider.search(SearchQueryRequest("text", query[:500], "video"), offset=0, limit=limit)
    except SearchFailure as exc:
        return [], f"{exc.error.code}: {exc.error.message[:120]}"
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {str(exc)[:120]}"
    return [c.platform_video_id for c in batch.candidates if c.platform_video_id], None


def direct_search(query: str, limit: int = 10) -> tuple[list[str], str | None]:
    """裸 yt_dlp 的 ytsearch（不经 SafeHttpClient 桥接）。"""

    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "skip_download": True, "no_warnings": True,
                        "ignoreconfig": True, "playlistend": limit, "socket_timeout": 20}) as client:
            raw = client.extract_info(f"ytsearch{limit}:{query[:500]}", download=False)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {str(exc)[:120]}"
    return [e.get("id") for e in (raw or {}).get("entries") or [] if isinstance(e, dict) and e.get("id")], None


def oembed(video_id: str) -> dict[str, Any]:
    """公开 oEmbed：标题与频道名；404 / 401 = 已删除或私享（这时搜不到是正常的）。"""

    import httpx

    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = httpx.get(url, timeout=15, follow_redirects=True)
    except httpx.HTTPError as exc:
        return {"available": None, "error": type(exc).__name__}
    if response.status_code != 200:
        return {"available": False, "status": response.status_code}
    data = response.json()
    return {"available": True, "title": data.get("title"), "channel": data.get("author_name")}


def rank_of(ids: list[str], sources: list[str]) -> int | None:
    for position, video_id in enumerate(ids, start=1):
        if video_id in sources:
            return position
    return None


async def gather_limited(jobs: list[Any], width: int = 4) -> list[Any]:
    gate = asyncio.Semaphore(width)

    async def one(job: Any) -> Any:
        async with gate:
            return await job

    return await asyncio.gather(*(one(job) for job in jobs))


def write_outputs(stem: str, payload: dict[str, Any], markdown: list[str]) -> Path:
    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = SCORES / f"{stem}.md"
    md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return md


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


__all__ = ["AB_IDS", "ROOT", "SCORES", "direct_search", "gather_limited", "load_sources", "now", "oembed",
           "provider_search", "rank_of", "write_outputs"]
