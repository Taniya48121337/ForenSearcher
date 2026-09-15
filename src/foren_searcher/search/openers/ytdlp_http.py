"""将 yt-dlp 的同步请求接入来源获取使用的受限异步 HTTP 客户端。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Future
import io
from email.message import Message
import threading
from types import SimpleNamespace
from typing import Any
from urllib.request import Request as UrllibRequest

from yt_dlp.networking import Request, Response
from yt_dlp.networking.exceptions import HTTPError

from ..contracts import SearchError, SearchFailure
from ..policy import SearchPolicy
from .http import SafeHttpClient


class _Body(io.RawIOBase):
    def __init__(self, bridge: _Bridge, context: Any, response: Any) -> None:
        super().__init__()
        self.bridge, self.context, self.response = bridge, context, response
        self.chunks = response.aiter_bytes(chunk_size=16*1024).__aiter__()
        self.buffer = b""
        self.received = 0
        self.finished = False
        self.released = False

    async def read_async(self, size: int | None) -> bytes:
        if self.closed or self.released:
            return b""
        while not self.finished and (size is None or size < 0 or len(self.buffer) < size):
            try:
                chunk = await self.chunks.__anext__()
            except StopAsyncIteration:
                self.finished = True
                break
            self.received += len(chunk)
            self.bridge.received += len(chunk)
            if self.received > self.bridge.policy.maximum_html_bytes or self.bridge.received > self.bridge.policy.maximum_document_bytes:
                raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "video metadata exceeds the configured read budget"))
            self.buffer += chunk
        take = len(self.buffer) if size is None or size < 0 else size
        result, self.buffer = self.buffer[:take], self.buffer[take:]
        return result

    def read(self, size: int | None = -1) -> bytes:
        return self.bridge.run(lambda:self.read_async(size))

    async def release(self) -> None:
        if not self.released:
            self.released = True
            await self.context.__aexit__(None, None, None)

    def close(self) -> None:
        if not self.closed:
            if not self.bridge.stopped.is_set():
                self.bridge.run(self.release)
            super().close()


class _Bridge:
    def __init__(self, http: SafeHttpClient, policy: SearchPolicy) -> None:
        self.http, self.policy = http, policy
        self.loop = asyncio.get_running_loop()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.pending: set[Future[Any]] = set()
        self.bodies: list[_Body] = []
        self.received = 0
        self.failure: SearchFailure | None = None
        self.cookiejar: Any = None
        self.default_headers: dict[str, str] = {}

    def run(self, coroutine: Callable[[], Any]) -> Any:
        with self.lock:
            if self.stopped.is_set():
                raise SearchFailure(SearchError("fetch", "TIMEOUT", "video metadata extraction was stopped"))
            future = asyncio.run_coroutine_threadsafe(coroutine(), self.loop)
            self.pending.add(future)
        try:
            return future.result(timeout=self.policy.source_timeout_s)
        except SearchFailure as exc:
            self.failure = exc
            raise
        finally:
            with self.lock:
                self.pending.discard(future)
            if not future.done():
                future.cancel()

    async def open_async(self, request: Request) -> Response:
        if request.data is not None and not isinstance(request.data, bytes):
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "metadata request body must be bytes"))
        context = self.http.metadata_stream(request.url, method=request.method,
            data=request.data, headers={**self.default_headers, **dict(request.headers)})
        response = await context.__aenter__()
        body = _Body(self, context, response)
        self.bodies.append(body)
        result = Response(body, str(response.url), dict(response.headers), status=response.status_code)
        result.headers = Message()
        for key,value in response.headers.multi_items():
            result.headers.add_header(key, value)
        self.received += sum(len(key)+len(value) for key,value in response.headers.multi_items())
        if self.received > self.policy.maximum_document_bytes:
            raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "video metadata exceeds the configured read budget"))
        if self.cookiejar is not None:
            self.cookiejar.extract_cookies(SimpleNamespace(info=lambda:result.headers), UrllibRequest(str(response.url)))
        if response.status_code >= 400:
            raise HTTPError(result)
        return result

    def urlopen(self, request: Any) -> Response:
        if isinstance(request, str):
            request = Request(request)
        elif isinstance(request, UrllibRequest):
            request = Request(request.full_url, data=request.data, headers=dict(request.header_items()), method=request.get_method())
        if not isinstance(request, Request):
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "unsupported metadata request type"))
        if self.cookiejar is not None:
            cookie_request = UrllibRequest(request.url)
            self.cookiejar.add_cookie_header(cookie_request)
            if cookie_request.has_header("Cookie") and "Cookie" not in request.headers:
                request.headers["Cookie"] = cookie_request.get_header("Cookie")
        return self.run(lambda:self.open_async(request))

    async def stop(self) -> None:
        with self.lock:
            self.stopped.set()
            pending = tuple(self.pending)
        for future in pending:
            future.cancel()
        await asyncio.gather(*(asyncio.wrap_future(future) for future in pending), return_exceptions=True)
        await asyncio.gather(*(body.release() for body in self.bodies), return_exceptions=True)
        self.bodies.clear()


async def extract_metadata(http: SafeHttpClient, policy: SearchPolicy,
                           factory: Callable[[Mapping[str, Any]], Any] | None,
                           options: Mapping[str, Any], target: str) -> Any:
    """提取结束或取消后关闭所有响应；工作线程无法再发起网络请求。"""
    bridge = _Bridge(http, policy)
    def extract() -> Any:
        if bridge.stopped.is_set():
            raise SearchFailure(SearchError("fetch", "TIMEOUT", "video metadata extraction was stopped"))
        from yt_dlp import YoutubeDL
        with (factory or YoutubeDL)(options) as client:
            bridge.cookiejar = getattr(client, "cookiejar", None)
            bridge.default_headers = dict(getattr(client,"params",{}).get("http_headers") or {})
            client.urlopen = bridge.urlopen
            raw = client.extract_info(target, download=False)
            sanitizer = getattr(client, "sanitize_info", None)
            return sanitizer(raw) if callable(sanitizer) else raw
    try:
        async with asyncio.timeout(policy.source_timeout_s):
            value = await asyncio.to_thread(extract)
        if bridge.failure is not None:
            raise bridge.failure
        return value
    except TimeoutError as exc:
        raise SearchFailure(SearchError("fetch", "TIMEOUT", "video metadata extraction timed out")) from exc
    except Exception:
        if bridge.failure is not None:
            raise bridge.failure
        raise
    finally:
        await bridge.stop()
