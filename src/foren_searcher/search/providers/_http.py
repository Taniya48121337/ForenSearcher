"""Provider 适配器共用的有界 HTTP 响应读取器。"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ..contracts import SearchError, SearchFailure


async def request_bounded(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    maximum_bytes: int,
    provider_name: str,
    stage: Literal["search", "upload"] = "search",
    **kwargs: Any,
) -> httpx.Response:
    """流式读取解码后的响应体，一旦超过上限立即停止。"""

    # stream=True 防止 httpx 在检查大小之前就把超大响应全部载入内存。
    request = client.build_request(method, url, **kwargs)
    response = await client.send(request, stream=True, follow_redirects=False)
    try:
        declared = response.headers.get("Content-Length")
        if declared is not None:
            # Content-Length 只是快速拒绝；它不可信，后面仍累计实际字节。
            try:
                if int(declared) > maximum_bytes:
                    raise _too_large(provider_name, stage)
            except ValueError:
                pass
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > maximum_bytes:
                raise _too_large(provider_name, stage)
            chunks.append(chunk)
        headers = response.headers.copy()
        # 新 Response 已是解码后的完整字节，旧传输编码头不能继续保留。
        for name in ("content-encoding", "content-length", "transfer-encoding"):
            if name in headers:
                del headers[name]
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=b"".join(chunks),
            request=response.request,
            extensions=response.extensions,
        )
    finally:
        await response.aclose()


def _too_large(provider_name: str, stage: Literal["search", "upload"]) -> SearchFailure:
    """构造统一的“Provider 响应过大”结构化异常。"""

    return SearchFailure(SearchError(
        stage,
        "PROVIDER_ERROR",
        f"{provider_name} response exceeded the size limit",
    ))
