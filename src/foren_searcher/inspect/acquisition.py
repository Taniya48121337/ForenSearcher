"""受控获取公开视频文件和完整 HLS VOD。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ..evidence_resolver import sniff_media_kind
from ..search.contracts import SearchFailure
from ..search.openers.http import SafeHttpClient
from ..search.policy import SearchPolicy
from .backends.base import BackendFailure
from .policy import InspectPolicy
from .hls import HlsDownloader


@dataclass(frozen=True, slots=True)
class AcquiredMedia:
    """下载完成但尚未提交为 Evidence 的媒体文件及其元数据。"""

    path: Path
    media_type: str
    suffix: str
    label: str
    details: dict[str, object]


class DirectMediaAcquirer:
    """取得视频文件或 HLS 分片；每次获取与跳转均执行公网地址检查。"""

    def __init__(self, *, allow_private_network: bool = False) -> None:
        """记录网络策略，并跟踪尚未关闭的临时 HTTP 客户端。"""

        self._allow_private_network = allow_private_network
        self._clients: set[SafeHttpClient] = set()

    async def aclose(self) -> None:
        """关闭所有仍在进行获取的客户端。"""

        clients, self._clients = tuple(self._clients), set()
        for client in clients:
            await client.aclose()

    async def acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy) -> AcquiredMedia:
        """在时间和字节预算内下载、验证视频，并返回获取结果。"""

        worker = asyncio.create_task(self._acquire(source_url, destination=destination, policy=policy))
        try:
            done, _ = await asyncio.wait({worker}, timeout=policy.acquisition_timeout_s)
            if not done:
                raise BackendFailure("TIMEOUT", "media acquisition timed out", stage="acquire", retryable=True)
            return await worker
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy) -> AcquiredMedia:

        # 复用 SEARCH 的 SSRF 安全客户端，但使用 INSPECT 的获取超时。
        http = SafeHttpClient(
            SearchPolicy(source_timeout_s=policy.acquisition_timeout_s),
            allow_private_network=self._allow_private_network,
        )
        self._clients.add(http)
        path = destination.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        try:
            resource = await http.download_to_file(source_url, destination=path,
                maximum_bytes=min(policy.maximum_file_bytes, policy.maximum_download_bytes))
            with path.open("rb") as handle:
                header = handle.read(32).lstrip(b"\xef\xbb\xbf \r\n\t")
            if header.startswith(b"#EXTM3U"):
                if path.stat().st_size > 8 * 1024**2:
                    raise BackendFailure("RESOURCE_LIMIT", "HLS playlist exceeds size limit", stage="acquire")
                initial_content = path.read_bytes()
                path.unlink()
                details = await HlsDownloader(http, policy).acquire(resource.final_url, path, initial_content=initial_content)
                details["requested_url"] = source_url
                return AcquiredMedia(path, "video/x-matroska", ".mkv", "acquired-video.mkv", details)
        except TimeoutError as exc:
            path.unlink(missing_ok=True)
            raise BackendFailure("TIMEOUT", "media acquisition timed out", stage="acquire", retryable=True) from exc
        except SearchFailure as exc:
            path.unlink(missing_ok=True)
            mapping = {
                "ACCESS_RESTRICTED": "ACCESS_RESTRICTED",
                "TIMEOUT": "TIMEOUT",
                "UNSUPPORTED_FORMAT": "RESOURCE_LIMIT",
            }
            code = mapping.get(exc.error.code, "MATERIAL_UNAVAILABLE")
            raise BackendFailure(code, exc.error.message, stage="acquire", retryable=code in {"TIMEOUT", "MATERIAL_UNAVAILABLE"}) from exc
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            # 每次 acquire 独占客户端；无论成败都立即释放连接池。
            self._clients.discard(http)
            await http.aclose()
        # MIME 可伪造，因此结合文件魔数/探测判断下载内容确实是视频。
        if sniff_media_kind(path, resource.media_type) != "video":
            path.unlink(missing_ok=True)
            raise BackendFailure("MATERIAL_UNAVAILABLE", "candidate URL did not return a directly decodable video", stage="acquire")
        suffix = Path(urlsplit(resource.final_url).path).suffix.lower()
        # 不可信 URL 后缀只保留很短的正常扩展名，其余回退为 .bin。
        if not suffix or len(suffix) > 12:
            suffix = ".bin"
        return AcquiredMedia(
            path, resource.media_type or "application/octet-stream", suffix,
            Path(urlsplit(resource.final_url).path).name or "acquired video",
            {"requested_url": source_url, "final_url": resource.final_url, "byte_size": resource.byte_size},
        )
