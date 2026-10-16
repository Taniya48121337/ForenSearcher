"""以图搜图结果的自核：下载结果缩略图，和查询图比感知哈希，判断是不是同一张图（10-16）。

各引擎说"同一张图"的口径不同（GCV full / partial、Lens exact / visual、Serper 与 Yandex 不说），这里不信引擎，
自己比一次：

- 只核合并后排在前 N 条的结果（默认 10），取 thumbnail_url，没有再取 image_url。
- 下载走 SafeHttpClient（逐跳校验公网地址、有大小上限、超时）；失败记 unchecked，不影响搜索本身。
- 比法：dHash（16x16，256 位）与 pHash（32x32 DCT 取 8x8，64 位），两者都算归一化汉明距离。先去掉上下 / 左右
  纯色边（YouTube 缩略图的黑边）。"same" 必须两种哈希都够近；只凭一种哈希够近最多算 "similar"。
  阈值用本机正例 / 负例校准（scripts/_calibrate_thumb_hash.py，结果写在 SAME_* / SIMILAR_* 旁边）。
- 比对结果按（查询图 sha256，缩略图 URL）存硬盘缓存，重跑不重下；只存比成了的。

这里的判断只写进 extra.verified_match 与分数；摘要里由 multi_reverse 写成一句事实陈述。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import SearchFailure

# 本机校准（scripts/_calibrate_thumb_hash.py，10-16；d3 + 金标 v1 共 160 张，evid 40 个 run 的帧做难负例）：
# 正例（同一张图缩到 96–200 px、JPEG q30–75）158–159/160 判 same；裁 3% 156/160；裁 6% 122/160 same、其余多为 similar；
# 加 4:3 黑边 151/160。负例：不同案例的图 23000 对、同一视频相隔几秒的帧 1998 对，没有一对判 same
# （最近的负例 dHash 0.285 / pHash 0.254，所以 same 取 0.25 / 0.22，留余量）；similar 两种哈希都要近。
SAME_DHASH = 0.25
SAME_PHASH = 0.22
SIMILAR_DHASH = 0.36
SIMILAR_PHASH = 0.32
CACHE_VERSION = 1
VERDICTS = ("same", "similar", "different", "unchecked")


def _load_gray(data: bytes) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        gray = im.convert("L")
        gray.load()
    return np.asarray(gray, dtype=np.float32)


def trim_borders(gray: np.ndarray, *, tolerance: float = 8.0, max_fraction: float = 0.3,
                 min_fraction: float = 0.03) -> np.ndarray:
    """去掉信箱式黑边 / 白边（上下一对或左右一对），每边最多去掉 max_fraction。

    先在缩到长边 256 px 的副本上找边（原图与缩略图找出的边比例一致）。只认近黑（≤24）或近白（≥231）的纯色边，
    且上下（左右）两边都要有、宽度差不超过 3%——暗场景的画面边缘不是黑边，单边的也不去；
    某一边不到 min_fraction 的不去（JPEG 边缘的几行噪声会让原图和缩略图各去一点，反而对不齐）。
    """

    h, w = gray.shape
    if h < 8 or w < 8:
        return gray
    scale = 256.0 / max(h, w)
    small = _resize(gray, max(8, round(w * scale)), max(8, round(h * scale))) if scale < 1 else gray
    sh, sw = small.shape

    def flat(line: np.ndarray, ref: float) -> bool:
        return float(np.abs(line - ref).mean()) < tolerance and float(line.std()) < tolerance

    ref = float(np.median(np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])))
    if 24 < ref < 231:
        return gray
    top, bottom, left, right = 0, sh, 0, sw
    while top < int(sh * max_fraction) and flat(small[top], ref):
        top += 1
    while bottom > sh - int(sh * max_fraction) and flat(small[bottom - 1], ref):
        bottom -= 1
    while left < int(sw * max_fraction) and flat(small[top:bottom, left], ref):
        left += 1
    while right > sw - int(sw * max_fraction) and flat(small[top:bottom, right - 1], ref):
        right -= 1
    cut = [top / sh, 1 - bottom / sh, left / sw, 1 - right / sw]
    for i, j in ((0, 1), (2, 3)):            # 成对的边才去
        if min(cut[i], cut[j]) < min_fraction or abs(cut[i] - cut[j]) > 0.03:
            cut[i] = cut[j] = 0.0
    t, b, l, r = round(cut[0] * h), h - round(cut[1] * h), round(cut[2] * w), w - round(cut[3] * w)
    return gray[t:b, l:r] if b - t >= 8 and r - l >= 8 else gray


def _resize(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(gray.astype(np.uint8), mode="L").resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def dhash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    small = _resize(gray, size + 1, size)
    return (small[:, 1:] > small[:, :-1]).ravel()


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def phash(gray: np.ndarray) -> np.ndarray:
    small = _resize(gray, 32, 32)
    coeffs = _DCT32 @ small @ _DCT32.T
    low = coeffs[:8, :8].ravel()[1:]                  # 去掉直流分量
    return low > np.median(low)


def fingerprint(data: bytes) -> dict[str, np.ndarray]:
    gray = trim_borders(_load_gray(data))
    return {"dhash": dhash(gray), "phash": phash(gray)}


def distances(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: round(float(np.count_nonzero(a[k] != b[k])) / a[k].size, 4) for k in ("dhash", "phash")}


def verdict(dist: dict[str, float]) -> str:
    d, p = dist["dhash"], dist["phash"]
    if d <= SAME_DHASH and p <= SAME_PHASH:
        return "same"
    if d <= SIMILAR_DHASH and p <= SIMILAR_PHASH:
        return "similar"
    return "different"


def thumb_url_of(candidate: Any) -> str | None:
    return getattr(candidate, "thumbnail_url", None) or getattr(candidate, "image_url", None)


class ThumbnailVerifier:
    """下载前 N 条结果的缩略图，与查询图比感知哈希；结果 {url: {"verified_match", "dhash", "phash", ...}}。"""

    def __init__(self, http: Any, *, top_n: int = 10, timeout_s: float = 8.0, total_timeout_s: float = 25.0,
                 max_bytes: int = 3 * 1024 * 1024, cache_dir: str | Path | None = None, concurrency: int = 5) -> None:
        self._http = http                       # SafeHttpClient（或同接口：async fetch(url, maximum_bytes=)）
        self.top_n = max(0, int(top_n))
        self.timeout_s = timeout_s
        self.total_timeout_s = total_timeout_s
        self.max_bytes = max_bytes
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.stats: dict[str, Any] = {"top_n": self.top_n, "checked": 0, "downloads": 0, "cache_hits": 0,
                                      "verdicts": {v: 0 for v in VERDICTS}, "unchecked_reasons": {},
                                      "cache_dir": str(self._cache_dir) if self._cache_dir else None}

    async def aclose(self) -> None:
        close = getattr(self._http, "aclose", None)
        if callable(close):
            await close()

    def skipped(self, reason: str) -> dict[str, Any]:
        return self._unchecked(reason)

    async def verify(self, query_image: bytes, urls: list[str | None], *,
                     total_timeout_s: float | None = None) -> list[dict[str, Any]]:
        """按顺序给每个 URL 一个结果；None（没有缩略图）记 unchecked/no_thumbnail。

        total_timeout_s：这次最多花多久（默认 self.total_timeout_s，取两者小的）；到点没比完的记 unchecked/total_timeout。
        """

        try:
            query = fingerprint(query_image)
        except Exception:  # noqa: BLE001  查询图读不了：全部 unchecked
            return [self._unchecked("query_unreadable") for _ in urls]
        qsha = hashlib.sha256(query_image).hexdigest()

        async def one(url: str | None) -> dict[str, Any]:
            if not url:
                return self._unchecked("no_thumbnail")
            cached = self._cache_get(qsha, url)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return self._count(cached)
            async with self._sem:
                try:
                    fetched = await asyncio.wait_for(self._http.fetch(url, maximum_bytes=self.max_bytes), self.timeout_s)
                except asyncio.TimeoutError:
                    return self._unchecked("timeout")
                except SearchFailure as exc:
                    return self._unchecked(exc.error.code.lower())
                except Exception as exc:  # noqa: BLE001  缩略图下载出任何错都只记 unchecked
                    return self._unchecked(type(exc).__name__)
            self.stats["downloads"] += 1
            try:
                dist = distances(query, await asyncio.to_thread(fingerprint, fetched.content))
            except Exception:  # noqa: BLE001
                return self._unchecked("not_an_image")
            result = {"verified_match": verdict(dist), "dhash": dist["dhash"], "phash": dist["phash"],
                      "thumbnail_url": url}
            self._cache_put(qsha, url, result)
            return self._count(result)

        tasks = [asyncio.ensure_future(one(u)) for u in urls]
        limit = self.total_timeout_s if total_timeout_s is None else min(self.total_timeout_s, max(0.0, total_timeout_s))
        done, pending = await asyncio.wait(tasks, timeout=limit) if tasks else (set(), set())
        for task in pending:
            task.cancel()
        out = []
        for task in tasks:
            if task in done and not task.cancelled() and task.exception() is None:
                out.append(task.result())
            else:
                out.append(self._unchecked("total_timeout"))
        return out

    def _unchecked(self, reason: str) -> dict[str, Any]:
        reasons = self.stats["unchecked_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        self.stats["verdicts"]["unchecked"] += 1
        return {"verified_match": "unchecked", "reason": reason}

    def _count(self, result: dict[str, Any]) -> dict[str, Any]:
        self.stats["checked"] += 1
        self.stats["verdicts"][result["verified_match"]] += 1
        return result

    # ------------------------------------------------------------ 硬盘缓存

    def _cache_path(self, qsha: str, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        key = hashlib.sha256(f"{qsha}\n{url}".encode("utf-8")).hexdigest()
        return self._cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, qsha: str, url: str) -> dict[str, Any] | None:
        path = self._cache_path(qsha, url)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION or data.get("query_sha256") != qsha \
                or data.get("thumbnail_url") != url:
            return None
        try:
            dist = {"dhash": float(data["dhash"]), "phash": float(data["phash"])}
        except (KeyError, TypeError, ValueError):
            return None
        # 存的是距离；结论按当前阈值重算（改阈值后旧缓存不会给出过时的结论）
        return {"verified_match": verdict(dist), **dist, "thumbnail_url": url}

    def _cache_put(self, qsha: str, url: str, result: dict[str, Any]) -> None:
        path = self._cache_path(qsha, url)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({"version": CACHE_VERSION, "query_sha256": qsha, **result,
                                       "thresholds": {"same": [SAME_DHASH, SAME_PHASH],
                                                      "similar": [SIMILAR_DHASH, SIMILAR_PHASH]},
                                       "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            self.stats["cache_write_errors"] = self.stats.get("cache_write_errors", 0) + 1


__all__ = ["SAME_DHASH", "SAME_PHASH", "SIMILAR_DHASH", "SIMILAR_PHASH", "ThumbnailVerifier", "distances",
           "dhash", "fingerprint", "phash", "thumb_url_of", "trim_borders", "verdict"]
