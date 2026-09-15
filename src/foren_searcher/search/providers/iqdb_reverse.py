"""无需密钥的 IQDB 反向图片 Provider，支持插画/动漫库和可选 3D 库。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from lxml import html

from ..contracts import CandidateBatch, CandidateSource, SearchError, SearchFailure, SearchQueryRequest
from ._http import request_bounded
from .yandex_reverse import _check_response


class IqdbReverseImageProvider:
    """上传一张图片，解析 IQDB 专用索引返回的相似来源。"""

    def __init__(
        self,
        timeout_s: float = 30.0,
        *,
        index: str = "artwork",
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        """验证索引和响应上限，并创建或接管异步 HTTP 客户端。"""

        if index not in {"artwork", "3d"}:
            raise ValueError("index must be artwork or 3d")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self._index = index
        self._base_url = "https://iqdb.org/" if index == "artwork" else "https://3d.iqdb.org/"
        self._max_response_bytes = max_response_bytes
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "ForenSearcher/0.1"},
        )
        # 同一图片的翻页应复用首次上传结果，避免反复上传和触发限流。
        self._cache_lock = asyncio.Lock()
        self._cached_key: str | None = None
        self._cached_result: tuple[list[dict[str, Any]], str] | None = None

    async def aclose(self) -> None:
        """只关闭本实例自己创建的客户端。"""

        if self._owns_client:
            await self._client.aclose()

    async def search(
        self,
        request: SearchQueryRequest,
        *,
        offset: int,
        limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """验证图片输入、复用上传缓存，再切出请求页并规范化候选。"""

        if request.input_type != "image" or image_path is None:
            raise SearchFailure(SearchError(
                "upload", "INVALID_INPUT", "IQDB reverse search requires an image-only query",
            ))
        image_bytes = image_path.read_bytes()
        image_key = str(image_path.resolve())
        async with self._cache_lock:
            # 锁保证并发翻页不会同时上传同一图片。
            reused = image_key == self._cached_key and self._cached_result is not None
            if reused:
                matches, result_url = self._cached_result  # type: ignore[misc]
            else:
                matches, result_url = await self._upload(image_bytes)
                self._cached_key = image_key
                self._cached_result = (matches, result_url)

        selected = matches[offset:offset + limit]
        candidates = tuple(
            _normalize_iqdb(match, request.output_type, rank, self._base_url)
            for rank, match in enumerate(selected, start=offset + 1)
        )
        return CandidateBatch(
            "iqdb-reverse",
            f"iqdb-{self._index}",
            candidates,
            {"result_url": result_url, "result_count": len(matches), "results": selected},
            {"index": self._index, "uploaded": not reused, "reused_provider_result": reused},
        )

    async def _upload(self, image_bytes: bytes) -> tuple[list[dict[str, Any]], str]:
        """上传图片并从 IQDB HTML 表格中提取匹配记录。"""

        try:
            response = await request_bounded(
                self._client, "POST", self._base_url,
                maximum_bytes=self._max_response_bytes, provider_name="IQDB", stage="upload",
                files={"file": ("query.jpg", image_bytes, "image/jpeg")},
            )
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("upload", "TIMEOUT", "IQDB image upload timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "IQDB image upload failed")) from exc
        _check_response(response, self._max_response_bytes, "IQDB")
        try:
            document = html.fromstring(response.content)
        except (TypeError, ValueError) as exc:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "IQDB returned invalid HTML")) from exc
        tables = document.xpath('//div[@id="pages"]/div/table')
        if not tables:
            raise SearchFailure(SearchError("search", "PROVIDER_ERROR", "IQDB result structure was not recognized"))
        matches: list[dict[str, Any]] = []
        # 第一张表是查询摘要，后续表才是单个匹配结果。
        for table in tables[1:]:
            rows = table.xpath("./tr")
            if len(rows) < 5:
                continue
            label = _row_text(rows[0])
            if label == "No relevant matches":
                continue
            links = rows[1].xpath(".//a/@href")
            images = rows[1].xpath(".//img/@src")
            if not links or not images:
                continue
            similarity_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", _row_text(rows[4]))
            matches.append({
                "label": label,
                "url": urljoin(self._base_url, links[0]),
                "thumbnail": urljoin(self._base_url, images[0]),
                "source": _row_text(rows[2]),
                "size": _row_text(rows[3]),
                "similarity": float(similarity_match.group(1)) if similarity_match else None,
            })
        return matches, str(response.url)


def _normalize_iqdb(
    match: dict[str, Any], output_type: str, rank: int, base_url: str,
) -> CandidateSource:
    """把 IQDB 私有字段转换成系统统一的 CandidateSource。"""

    source_url = str(match["url"])
    thumbnail = str(match["thumbnail"])
    source = str(match.get("source") or "IQDB match")
    similarity = match.get("similarity")
    label = str(match.get("label") or "Visual match")
    snippet = f"{label}; {similarity:g}% similarity" if isinstance(similarity, (int, float)) else label
    return CandidateSource(
        source_url=source_url,
        search_title=source,
        search_snippet=snippet,
        provider="iqdb-reverse",
        engine="iqdb-3d" if "3d.iqdb.org" in base_url else "iqdb-artwork",
        provider_rank=rank,
        candidate_type="image" if output_type == "image" else "page",
        image_url=thumbnail if output_type == "image" else None,
        thumbnail_url=thumbnail,
        extra={"match_label": label, "similarity": similarity, "size": match.get("size")},
    )


def _row_text(row: Any) -> str:
    """合并一个 lxml 行节点的所有可见文字并压缩空白。"""

    return " ".join(" ".join(row.xpath(".//text()")).split())
