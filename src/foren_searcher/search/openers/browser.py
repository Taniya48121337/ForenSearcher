"""面向 JavaScript 动态页面的可选隔离 Playwright 后备打开器。"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from html import escape
from typing import Any

from ..contracts import SearchError, SearchFailure
from ..policy import SearchPolicy
from .http import SafeHttpClient


class PlaywrightRenderer:
    """在无头 Chromium 中渲染页面，同时复用 SafeHttpClient 的 URL 策略。"""

    def __init__(self, http: SafeHttpClient, policy: SearchPolicy | None = None) -> None:
        """保存安全客户端，并用信号量限制浏览器并发。"""

        self.http, self.policy = http, policy or http.policy
        self._semaphore = asyncio.Semaphore(self.policy.dynamic_concurrency)

    async def render(self, url: str) -> bytes:
        """渲染主页面和可读 iframe，返回有大小上限的 UTF-8 HTML。"""

        # 浏览器启动前先验证入口 URL。
        await self.http.validate_url(url)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise SearchFailure(SearchError("open", "EXTRACT_FAILED", "dynamic page requires the Playwright extra")) from exc
        try:
            async with self._semaphore, async_playwright() as playwright:
                browser = None
                context = None
                try:
                    browser = await playwright.chromium.launch(headless=True)
                    context = await browser.new_context(service_workers="block")
                    page = await context.new_page()

                    async def guard(route: Any) -> None:
                        """拦截每个子资源请求，阻止 WebSocket 和非公网 URL。"""

                        try:
                            if route.request.resource_type == "websocket":
                                await route.abort()
                                return
                            await self.http.validate_url(route.request.url)
                            await route.continue_()
                        except SearchFailure:
                            await route.abort()

                    await page.route("**/*", guard)
                    await page.goto(url, wait_until="networkidle", timeout=int(self.policy.dynamic_timeout_s * 1000))
                    documents = [await page.content()]
                    # iframe 可能承载正文，因此经 URL 验证后也纳入抽取文档。
                    for frame in page.frames:
                        if frame == page.main_frame:
                            continue
                        frame_url = frame.url
                        if frame_url.startswith(("http://", "https://")):
                            await self.http.validate_url(frame_url)
                        try:
                            frame_html = await frame.content()
                        except Exception:
                            continue
                        documents.append(
                            f'<section data-search-frame-url="{escape(frame_url, quote=True)}">{frame_html}</section>'
                        )
                    result = "\n".join(documents).encode("utf-8")
                    if len(result) > self.policy.maximum_html_bytes:
                        raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "rendered page exceeds the configured size limit"))
                    return result
                finally:
                    # 即使提取或取消失败也尽力关闭上下文和浏览器进程。
                    if context is not None:
                        with suppress(Exception):
                            await context.close()
                    if browser is not None:
                        with suppress(Exception):
                            await browser.close()
        except SearchFailure:
            raise
        except Exception as exc:
            # Playwright 异常类型不稳定，统一收敛成 SEARCH 公开错误码。
            code = "TIMEOUT" if "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower() else "EXTRACT_FAILED"
            raise SearchFailure(SearchError("open", code, f"dynamic page rendering failed ({type(exc).__name__})")) from exc  # type: ignore[arg-type]
