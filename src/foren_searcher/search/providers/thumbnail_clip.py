"""本机以图搜视频（找原片实验 A3 / imageclip，2026-10-06）：缩略图帧与查询图的 CLIP 相似度重排。

旧 plan-harness-1 流水线的 thumbnail_rerank（public_thumbnail_clip_robust_copy_v1）把查询帧与候选缩略图做 CLIP
比对，video_090 这类样本靠它把原片排进前面。这里用现有的 .models/inspect/clip-vit-base-patch32（locate_frames
用的同一份，FOREN_CLIP_MODEL），不另下权重。

- text_image → video：文字部分查 YouTube 前 30 条（YtDlpSearchProvider），每条取 4 张缩略图（hqdefault 与
  YouTube 自动截的 hq1 / hq2 / hq3，后三张是视频 1/4、2/4、3/4 处的真实帧），与查询图的余弦相似度取最大，按它重排。
- image → video：没有文字可查，只重排本 run 已有的 YouTube 候选（store.candidates）；一个都没有时告诉 HI 先用文字找。
- 其他输出类型：不支持（报 INVALID_INPUT，说明只能找视频）。

查询图只在本机用；联网的只有文字查询与公开缩略图下载。候选 provider 记 "thumbnail_clip"，extra 里有 clip_score、
text_rank，漏斗据此归因。
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
from pathlib import Path
from typing import Any

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest

POOL_FOR_TEXT = 30
THUMBS = ("hqdefault", "hq1", "hq2", "hq3")
THUMB_URL = "https://i.ytimg.com/vi/{video_id}/{name}.jpg"
_MODEL_LOCK = threading.Lock()
_MODEL: dict[str, Any] = {}

NOT_VIDEO = ("Image-input search here only finds videos: use output_type=video. To find pages or images, describe the "
             "image in a text query instead.")
EMPTY_POOL = ("Image-only search ranks the video candidates already found in this investigation, and there are none yet. "
              "Use input_type=text_image with 3-8 distinctive words describing the video, or run a text video search first.")


def _clip_path() -> str:
    path = os.environ.get("FOREN_CLIP_MODEL")
    if not path:
        root = Path(".models") / "inspect" / "clip-vit-base-patch32"
        if root.is_dir():
            return str(root.resolve())
        raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "image search model is not available on this machine"))
    return path


def _load_clip() -> tuple[Any, Any, str]:
    with _MODEL_LOCK:
        if "model" not in _MODEL:
            import torch
            from transformers import AutoProcessor, CLIPModel

            path = _clip_path()
            try:
                from ...inspect.backends.local import _resolve_local_model

                path = _resolve_local_model(path)
            except Exception:  # noqa: BLE001  解析失败就按原路径加载
                pass
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()
            _MODEL.update(model=model, processor=processor, device=device)
        return _MODEL["model"], _MODEL["processor"], _MODEL["device"]


def _features(images: list[Any]) -> Any:
    import torch

    model, processor, device = _load_clip()
    with _MODEL_LOCK, torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        features = model.get_image_features(**inputs)
        features = getattr(features, "pooler_output", features)
        return (features / features.norm(dim=-1, keepdim=True)).float().cpu()


def clip_scores(query_image: bytes, thumbs: dict[str, list[bytes]]) -> dict[str, float]:
    """每个视频：它的几张缩略图与查询图余弦相似度的最大值。"""

    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(query_image)) as raw:
        query = ImageOps.exif_transpose(raw).convert("RGB")
    owners: list[str] = []
    images: list[Any] = []
    for video_id, blobs in thumbs.items():
        for blob in blobs:
            try:
                with Image.open(io.BytesIO(blob)) as raw:
                    images.append(raw.convert("RGB").copy())
                owners.append(video_id)
            except Exception:  # noqa: BLE001  坏图跳过
                continue
    if not images:
        return {}
    query_features = _features([query])
    scores: dict[str, float] = {}
    for start in range(0, len(images), 32):
        chunk = _features(images[start:start + 32])
        for owner, value in zip(owners[start:start + 32], (chunk @ query_features.T).squeeze(-1).tolist()):
            scores[owner] = max(scores.get(owner, -1.0), float(value))
    return scores


async def _download_thumbs(video_ids: list[str], timeout_s: float = 10.0) -> dict[str, list[bytes]]:
    import httpx

    out: dict[str, list[bytes]] = {video_id: [] for video_id in video_ids}
    limit = asyncio.Semaphore(12)
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 ForenSearcher thumbnail rerank"}) as client:
        async def one(video_id: str, name: str) -> None:
            async with limit:
                try:
                    response = await client.get(THUMB_URL.format(video_id=video_id, name=name))
                except httpx.HTTPError:
                    return
                if response.status_code == 200 and response.content and len(response.content) < 2_000_000:
                    out[video_id].append(response.content)

        await asyncio.gather(*(one(video_id, name) for video_id in video_ids for name in THUMBS))
    return out


class ThumbnailClipProvider:
    """image / text_image → video 的本机重排 provider。store 用来取本 run 已有的候选（纯图片查询）。"""

    def __init__(self, text_video_provider: Any, store: Any | None = None, *, pool: int = POOL_FOR_TEXT,
                 scorer: Any | None = None, downloader: Any | None = None) -> None:
        self.text_video_provider, self.store, self.pool = text_video_provider, store, pool
        self._score = scorer or clip_scores
        self._download = downloader or _download_thumbs
        self.stats = {"text_image": 0, "image": 0, "empty_pool": 0, "failures": 0, "last_error": None}

    def _existing_pool(self) -> list[CandidateSource]:
        if self.store is None:
            return []
        rows: list[tuple[Any, ...]] = []
        try:
            with self.store._connect() as connection:      # GlobalStore 的短连接（_ClosingConnection，退出即关）
                rows = [tuple(row) for row in connection.execute(
                    "SELECT source_url, search_title, platform_video_id, thumbnail_url FROM candidates "
                    "WHERE platform='youtube' AND platform_video_id IS NOT NULL ORDER BY id").fetchall()]
        except Exception:  # noqa: BLE001
            rows = []
        out: list[CandidateSource] = []
        seen: set[str] = set()
        for url, title, video_id, thumb in rows:
            if video_id in seen:
                continue
            seen.add(video_id)
            out.append(CandidateSource(str(url), title, None, "yt-dlp", "existing_pool", len(out) + 1, "video",
                                       thumbnail_url=thumb, platform="youtube", platform_video_id=str(video_id)))
        return out

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> CandidateBatch:
        if getattr(request, "output_type", None) != "video":
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", NOT_VIDEO))
        if image_path is None:
            raise SearchFailure(SearchError("upload", "INVALID_INPUT", "image query has no prepared image"))
        if request.input_type == "text_image":
            self.stats["text_image"] += 1
            text = str(request.query.get("text") if isinstance(request.query, dict) else "")
            batch = await self.text_video_provider.search(SearchQueryRequest("text", text, "video"), offset=0,
                                                          limit=self.pool, image_path=None)
            base = [c for c in batch.candidates if c.platform_video_id]
        else:
            self.stats["image"] += 1
            base = self._existing_pool()
            if not base:
                self.stats["empty_pool"] += 1
                raise SearchFailure(SearchError("search", "INVALID_INPUT", EMPTY_POOL))
        if not base:
            return CandidateBatch("thumbnail_clip", "youtube_thumbnail_clip", (), {"base": 0})
        ids = list(dict.fromkeys(c.platform_video_id for c in base if c.platform_video_id))
        thumbs = await self._download(ids)
        try:
            scores = await asyncio.to_thread(self._score, Path(image_path).read_bytes(), thumbs)
        except SearchFailure:
            raise
        except Exception as exc:  # noqa: BLE001
            self.stats["failures"] += 1
            self.stats["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", f"image ranking failed ({type(exc).__name__})")) from exc
        by_id = {c.platform_video_id: c for c in base}
        order = sorted(ids, key=lambda vid: (-scores.get(vid, -1.0), base.index(by_id[vid])))
        self.stats["last_scored"] = len(scores)
        self.stats["last_top"] = [[vid, round(scores[vid], 4)] for vid in order[:3] if vid in scores]
        ranked = []
        for position, video_id in enumerate(order, start=1):
            item = by_id[video_id]
            ranked.append(CandidateSource(
                item.source_url, item.search_title, item.search_snippet, "thumbnail_clip", "youtube_thumbnail_clip",
                position, "video", thumbnail_url=item.thumbnail_url, platform="youtube", platform_video_id=video_id,
                extra={**dict(item.extra or {}), "clip_score": round(scores[video_id], 4) if video_id in scores else None,
                       "text_rank": item.provider_rank, "base": item.engine}))
        page = ranked[offset:offset + limit]
        return CandidateBatch("thumbnail_clip", "youtube_thumbnail_clip", tuple(page),
                              {"base": len(base), "scored": len(scores)})

    async def aclose(self) -> None:
        close = getattr(self.text_video_provider, "aclose", None)
        if callable(close):
            await close()

    def summary(self) -> dict[str, Any]:
        return dict(self.stats)


IGNORED_TEXT_NOTE = " The text part of text_image is ignored."
TEXT_ALSO_NOTE = (" With input_type=text_image the text part also runs as an ordinary text search, and the two result lists"
                  " are merged.")


def text_also_note(note: str) -> str:
    return note.replace(IGNORED_TEXT_NOTE, "") + TEXT_ALSO_NOTE


class ImageQueryRouter:
    """image / text_image 输入交给以图搜图 provider；文字输入照旧走内层 provider。

    text_also=True（10-13，多引擎以图搜图 / Google Cloud Vision）：text_image 的文字部分**同时**交给内层的文字搜索，
    两边结果交错合并。以图搜图引擎不看文字，evidM27a 里 HI 惯用 text_image 找视频，打开以图搜图后这些查询只剩网页
    结果（新闻页、Pinterest），YouTube 文字搜索整个没了：14 个里 172 / 194 / 205 的视频候选从 20 / 10 / 12 降到 0 / 0 / 2。
    """

    def __init__(self, inner: Any, image_provider: Any, *, text_also: bool = False) -> None:
        self.inner, self.image_provider = inner, image_provider
        self.text_also = text_also
        self.stats: dict[str, Any] = {"text_image_both": 0, "text_side_failed": 0, "image_side_failed": 0}

    @property
    def provider(self) -> Any:          # run_cases._find_video_router 沿 .provider 往里找
        return self.inner

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> Any:
        kind = getattr(request, "input_type", None)
        if kind == "text_image" and self.text_also:
            return await self._both(request, offset=offset, limit=limit, image_path=image_path)
        if kind in {"image", "text_image"}:
            return await self.image_provider.search(request, offset=offset, limit=limit, image_path=image_path)
        return await self.inner.search(request, offset=offset, limit=limit, image_path=image_path)

    async def _both(self, request: Any, *, offset: int, limit: int, image_path: Path | None) -> Any:
        self.stats["text_image_both"] += 1
        text_request = SearchQueryRequest("text", request.query["text"], request.output_type)
        image_task = self.image_provider.search(request, offset=offset, limit=limit, image_path=image_path)
        text_task = self.inner.search(text_request, offset=offset, limit=limit, image_path=None)
        image_out, text_out = await asyncio.gather(image_task, text_task, return_exceptions=True)
        for value in (image_out, text_out):
            if isinstance(value, BaseException) and not isinstance(value, Exception):
                raise value                                   # cancellation
        image_ok = not isinstance(image_out, BaseException)
        text_ok = not isinstance(text_out, BaseException)
        if not image_ok:
            self.stats["image_side_failed"] += 1
        if not text_ok:
            self.stats["text_side_failed"] += 1
        if not image_ok and not text_ok:
            raise image_out                                    # type: ignore[misc]
        if not text_ok:
            return image_out
        if not image_ok:
            return text_out
        # video output: the text search returns platform videos, the image engines mostly pages; interleave from the
        # text side so the first few opened results include both kinds
        first, second = (text_out, image_out) if request.output_type == "video" else (image_out, text_out)
        merged, seen = [], set()
        for k in range(max(len(first.candidates), len(second.candidates))):
            for batch in (first, second):
                if k < len(batch.candidates):
                    c = batch.candidates[k]
                    key = c.platform_video_id or c.source_url
                    if key not in seen:
                        seen.add(key)
                        merged.append(c)
        return CandidateBatch(f"{image_out.provider}+{text_out.provider}", f"{image_out.engine}+{text_out.engine}",
                              tuple(merged), {"image": image_out.raw_response, "text": text_out.raw_response},
                              {"text_image": "image engines + text search", "image": image_out.request_metadata,
                               "text": text_out.request_metadata})

    async def aclose(self) -> None:
        for provider in (self.inner, self.image_provider):
            close = getattr(provider, "aclose", None)
            if callable(close):
                await close()


__all__ = ["EMPTY_POOL", "IGNORED_TEXT_NOTE", "ImageQueryRouter", "TEXT_ALSO_NOTE", "text_also_note", "NOT_VIDEO", "ThumbnailClipProvider", "clip_scores"]
