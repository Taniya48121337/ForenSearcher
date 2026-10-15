"""带 SSRF 防护和体积上限的 HTTP 客户端。

任何来源 URL 在首个请求及每次跳转前都会重新解析域名；连接建立后还核对真实远端
IP，防止 DNS 重绑定把公开域名转向内网。响应按流读取，以免未知站点耗尽内存。
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import ssl
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit

import httpx

from ..contracts import SearchError, SearchFailure
from ..policy import SearchPolicy

# 下载媒体时允许带上的平台请求头（yt-dlp 格式的 http_headers）；不含 Cookie / Authorization 等凭据。
DOWNLOAD_HEADER_ALLOWLIST = frozenset({"user-agent", "accept", "accept-language", "sec-fetch-mode", "referer", "origin"})


@dataclass(frozen=True, slots=True)
class FetchedResource:
    """已经完整读入内存的小型 HTTP 资源。"""

    requested_url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    content: bytes
    media_type: str
    encoding: str | None


@dataclass(frozen=True, slots=True)
class DownloadedResource:
    """已经直接流式写入文件的大型 HTTP 资源元数据。"""

    requested_url: str
    final_url: str
    status_code: int
    headers: dict[str, str]
    media_type: str
    byte_size: int


class SafeHttpClient:
    """SEARCH 所有普通 HTTP 获取操作的安全边界。"""

    def __init__(self, policy: SearchPolicy | None = None, *, allow_private_network: bool = False, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """建立禁用环境代理和自动跳转的 httpx 客户端。"""

        self.policy = policy or SearchPolicy()
        self.allow_private_network = allow_private_network
        # 10-12 服务器：FOREN_HTTP_PROXY（如 http://10.0.0.1:7890）设了才走代理。经代理时连接对端是代理本身，
        # 所以不再核对"实际连接 IP 是公网"，DNS 解析不了也放行（由代理解析）；URL 里的字面 IP 仍须是公网。
        self.proxy = os.environ.get("FOREN_HTTP_PROXY") or None
        timeout = httpx.Timeout(self.policy.source_timeout_s, connect=min(10.0, self.policy.source_timeout_s))
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False,
            headers={"User-Agent": self.policy.user_agent}, transport=transport, trust_env=False,
            **({"proxy": self.proxy} if self.proxy and transport is None else {}))

    async def aclose(self) -> None:
        """释放连接池。"""

        await self._client.aclose()

    async def validate_url(self, url: str) -> None:
        """验证 URL 语法，并确保其所有解析地址都是公网 IP。"""

        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "source must be an HTTP(S) URL without embedded credentials"))
        if self.allow_private_network:
            # 仅测试或明确受控部署可打开此选项。
            return
        try:
            literal = ipaddress.ip_address(parsed.hostname)
            addresses = {literal}
        except ValueError:
            try:
                # getaddrinfo 是阻塞调用，放在线程中避免卡住事件循环。
                rows = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
                addresses = {ipaddress.ip_address(row[4][0]) for row in rows}
            except socket.gaierror as exc:
                if self.proxy:
                    return                      # 经代理：本机解析不了的域名交给代理
                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source hostname could not be resolved")) from exc
        # 只要某个解析结果不是公网就拒绝，避免客户端随机命中内网地址。
        if not addresses or any(not address.is_global for address in addresses):
            raise SearchFailure(SearchError("validate", "ACCESS_RESTRICTED", "source resolves to a non-public network address"))

    async def fetch(self, url: str, *, maximum_bytes: int) -> FetchedResource:
        """安全获取有界资源并返回内存字节。"""

        current = url
        for redirect_count in range(self.policy.maximum_redirects + 1):
            await self.validate_url(current)
            try:
                async with self._client.stream("GET", current) as response:
                    if not self.allow_private_network and not self.proxy:
                        # 请求建立后核对实际连接 IP，补上 DNS 校验与连接间的竞态窗口。
                        network_stream = response.extensions.get("network_stream")
                        server_addr = network_stream.get_extra_info("server_addr") if network_stream is not None else None
                        if server_addr:
                            try:
                                connected = ipaddress.ip_address(server_addr[0])
                            except (ValueError, TypeError):
                                connected = None
                            if connected is not None and not connected.is_global:
                                raise SearchFailure(SearchError("validate", "ACCESS_RESTRICTED", "connection reached a non-public network address"))
                    header_size = sum(len(k) + len(v) for k, v in response.headers.items())
                    # 超大响应头同样能消耗资源，所以独立限制为 64 KiB。
                    if header_size > 64 * 1024:
                        raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "response headers exceed the safety limit"))
                    if response.status_code in {301, 302, 303, 307, 308}:
                        # 禁用 httpx 自动跳转后，才能在每个新 URL 前重新做安全验证。
                        location = response.headers.get("location")
                        if not location:
                            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "redirect response has no location"))
                        current = urljoin(current, location)
                        continue
                    if response.status_code in {401, 403, 407, 451}:
                        raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED", f"source denied access (HTTP {response.status_code})"))
                    if response.status_code in {404, 410}:
                        raise SearchFailure(SearchError("open", "NOT_FOUND", f"source is unavailable (HTTP {response.status_code})"))
                    if response.status_code == 429:
                        retry_after = _retry_after(response.headers.get("retry-after"))
                        raise SearchFailure(SearchError("fetch", "RATE_LIMITED", "source rate limited the request"), retry_after_s=retry_after)
                    if response.status_code >= 400:
                        raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", f"source returned HTTP {response.status_code}"))
                    declared = response.headers.get("content-length")
                    # Content-Length 仅作提前拒绝，实际读取仍逐块累计。
                    if declared and declared.isdigit() and int(declared) > maximum_bytes:
                        raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "source exceeds the configured size limit"))
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > maximum_bytes:
                            raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "source exceeds the configured size limit"))
                        chunks.append(chunk)
                    media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip().lower()
                    return FetchedResource(url, str(response.url), response.status_code, dict(response.headers), b"".join(chunks), media_type, response.encoding)
            except SearchFailure:
                raise
            except ssl.SSLCertVerificationError as exc:
                raise SearchFailure(SearchError("fetch", "AUTH_ERROR", "source TLS certificate could not be verified")) from exc
            except ssl.SSLError as exc:
                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source TLS connection failed")) from exc
            except httpx.TimeoutException as exc:
                raise SearchFailure(SearchError("fetch", "TIMEOUT", "source request timed out")) from exc
            except httpx.HTTPError as exc:
                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", f"source request failed ({type(exc).__name__})")) from exc
        raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source exceeded the redirect limit"))

    @asynccontextmanager
    async def metadata_stream(self, url: str, *, method: str = "GET", data: bytes | None = None,
                              headers: Mapping[str, str] | None = None) -> AsyncIterator[httpx.Response]:
        """供同步提取器按需读取；调用者负责正文总量，连接和每次跳转在此检查。"""
        if method not in {"GET", "HEAD", "POST"} or (data is not None and len(data) > self.policy.maximum_html_bytes):
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "unsupported metadata request method or body size"))
        current = url
        forwarded = {key:value for key,value in (headers or {}).items()
                     if key.lower() not in {"host", "content-length", "connection", "transfer-encoding"}}
        try:
            for _ in range(self.policy.maximum_redirects + 1):
                await self.validate_url(current)
                async with self._client.stream(method, current, headers=forwarded, content=data) as response:
                    if not self.allow_private_network and not self.proxy:
                        network = response.extensions.get("network_stream")
                        address = network.get_extra_info("server_addr") if network is not None else None
                        if address:
                            try:
                                connected = ipaddress.ip_address(address[0])
                            except (ValueError, TypeError):
                                connected = None
                            if connected is not None and not connected.is_global:
                                raise SearchFailure(SearchError("validate", "ACCESS_RESTRICTED", "connection reached a non-public network address"))
                    if sum(len(key)+len(value) for key,value in response.headers.items()) > 64*1024:
                        raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "response headers exceed the safety limit"))
                    if response.status_code in {301,302,303,307,308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "redirect response has no location"))
                        target = urljoin(current, location)
                        before, after = urlsplit(current), urlsplit(target)
                        if (before.scheme,before.hostname,before.port) != (after.scheme,after.hostname,after.port):
                            forwarded = {key:value for key,value in forwarded.items() if key.lower() not in {"authorization", "cookie"}}
                        if (response.status_code == 303 and method != "HEAD") or (response.status_code in {301,302} and method == "POST"):
                            method, data = "GET", None
                            forwarded = {key:value for key,value in forwarded.items() if not key.lower().startswith("content-")}
                        current = target
                        continue
                    # HTTP 错误正文仍供 yt-dlp 按其来源协议解析；网络/权限检查不可跳过。
                    yield response
                    return
            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source exceeded the redirect limit"))
        except ssl.SSLCertVerificationError as exc:
            raise SearchFailure(SearchError("fetch", "AUTH_ERROR", "source TLS certificate could not be verified")) from exc
        except ssl.SSLError as exc:
            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source TLS connection failed")) from exc
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("fetch", "TIMEOUT", "metadata request timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", f"metadata request failed ({type(exc).__name__})")) from exc

    async def download_to_file(self, url: str, *, destination: Path, maximum_bytes: int,
                               byte_range: tuple[int, int] | None = None,
                               on_chunk: Callable[[int], None] | None = None,
                               extra_headers: Mapping[str, str] | None = None) -> DownloadedResource:
        """把公网资源流式写入一个必须尚不存在的目标文件。

        extra_headers（10-05）：平台元数据给出的请求头（yt-dlp 格式的 http_headers，如 User-Agent），
        只收 DOWNLOAD_HEADER_ALLOWLIST 里的键（都是无凭据的普通请求头，跳转时照常带上）。"""
        current = url
        headers = {key: value for key, value in (extra_headers or {}).items()
                   if isinstance(key, str) and isinstance(value, str) and key.lower() in DOWNLOAD_HEADER_ALLOWLIST}
        created_destination = False
        if byte_range is not None:
            start, length = byte_range
            if start < 0 or length <= 0 or length > maximum_bytes:
                raise ValueError("invalid or oversized byte range")
            headers["Range"] = f"bytes={start}-{start + length - 1}"
        try:
            for _redirect_count in range(self.policy.maximum_redirects + 1):
                await self.validate_url(current)
                try:
                    async with self._client.stream("GET", current, headers=headers) as response:
                        if not self.allow_private_network and not self.proxy:
                            network_stream = response.extensions.get("network_stream")
                            server_addr = network_stream.get_extra_info("server_addr") if network_stream is not None else None
                            if server_addr:
                                try:
                                    connected = ipaddress.ip_address(server_addr[0])
                                except (ValueError, TypeError):
                                    connected = None
                                if connected is not None and not connected.is_global:
                                    raise SearchFailure(SearchError("validate", "ACCESS_RESTRICTED", "connection reached a non-public network address"))
                        header_size = sum(len(k) + len(v) for k, v in response.headers.items())
                        if header_size > 64 * 1024:
                            raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "response headers exceed the safety limit"))
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "redirect response has no location"))
                            current = urljoin(current, location)
                            continue
                        if response.status_code in {401, 403, 407, 451}:
                            raise SearchFailure(SearchError("open", "ACCESS_RESTRICTED", f"source denied access (HTTP {response.status_code})"))
                        if response.status_code in {404, 410}:
                            raise SearchFailure(SearchError("open", "NOT_FOUND", f"source is unavailable (HTTP {response.status_code})"))
                        if response.status_code == 429:
                            raise SearchFailure(SearchError("fetch", "RATE_LIMITED", "source rate limited the request"))
                        if response.status_code >= 400:
                            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", f"source returned HTTP {response.status_code}"))
                        if byte_range is not None:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", response.headers.get("content-range", ""))
                            if response.status_code != 206 or match is None or (int(match[1]), int(match[2])) != (start, start + length - 1):
                                raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source did not honor the requested byte range"))
                        elif response.status_code == 206:
                            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source returned an unsolicited partial download"))
                        declared = response.headers.get("content-length")
                        if declared and declared.isdigit() and int(declared) > maximum_bytes:
                            raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "source exceeds the configured size limit"))
                        size = 0
                        # xb 可防止覆盖已有文件；超限异常会由外层统一删除半成品。
                        with destination.open("xb") as handle:
                            created_destination = True
                            async for chunk in response.aiter_bytes():
                                size += len(chunk)
                                if size > maximum_bytes:
                                    raise SearchFailure(SearchError("fetch", "UNSUPPORTED_FORMAT", "source exceeds the configured size limit"))
                                if on_chunk is not None:
                                    on_chunk(len(chunk))
                                handle.write(chunk)
                        if byte_range is not None and size != length:
                            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source returned an incomplete byte range"))
                        media_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip().lower()
                        return DownloadedResource(url, str(response.url), response.status_code, dict(response.headers), media_type, size)
                except SearchFailure:
                    raise
                except ssl.SSLCertVerificationError as exc:
                    raise SearchFailure(SearchError("fetch", "AUTH_ERROR", "source TLS certificate could not be verified")) from exc
                except ssl.SSLError as exc:
                    raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source TLS connection failed")) from exc
                except httpx.TimeoutException as exc:
                    raise SearchFailure(SearchError("fetch", "TIMEOUT", "source request timed out")) from exc
                except httpx.HTTPError as exc:
                    raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", f"source request failed ({type(exc).__name__})")) from exc
            raise SearchFailure(SearchError("fetch", "NETWORK_ERROR", "source exceeded the redirect limit"))
        except BaseException:
            # 包括协程取消在内的任何中断都不能留下不完整下载。
            if created_destination:
                destination.unlink(missing_ok=True)
            raise


def _retry_after(value: str | None) -> float | None:
    """容错解析 Retry-After 秒数。"""

    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
