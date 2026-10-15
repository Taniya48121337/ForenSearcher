"""受控获取公开视频文件和完整 HLS VOD。"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..store.resolver import sniff_media_kind
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

    async def acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy,
                      headers: Mapping[str, str] | None = None, page_url: str | None = None,
                      video_only_height: int | None = None) -> AcquiredMedia:
        """在时间和字节预算内下载、验证视频，并返回获取结果。

        10-05：headers 是平台元数据给出的下载请求头（yt-dlp 格式的 http_headers）。page_url 是 YouTube 视频页：
        直链下载被拒（HTTP 403 等）或直链已过期时，改由 yt-dlp 按页面重新解析并下载（evidb7 冒烟里直链 5/5 被 403）。
        10-12 video_only_height（原 DETECT 的低清兜底，下沉到这里）：只要画面时下 ≤ 该高度的单路视频流、不要声音。
        YouTube 页面才走这条；其他地址照常整文件下载。
        """

        if video_only_height:
            youtube = next((u for u in (page_url, source_url)
                            if u and (urlsplit(u).hostname or "").lower() in YTDLP_HOSTS), None)
            if youtube is not None:
                worker = asyncio.create_task(asyncio.to_thread(ytdlp_download, youtube, destination, policy,
                                                               video_only_height=int(video_only_height)))
                try:
                    done, _ = await asyncio.wait({worker}, timeout=policy.acquisition_timeout_s)
                    if not done:
                        raise BackendFailure("TIMEOUT", "media acquisition timed out", stage="acquire", retryable=True)
                    acquired = await worker
                finally:
                    if not worker.done():
                        worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                acquired.details["video_only_height"] = int(video_only_height)
                return acquired
        worker = asyncio.create_task(self._acquire_with_fallback(source_url, destination=destination, policy=policy,
                                                                 headers=headers, page_url=page_url))
        try:
            done, _ = await asyncio.wait({worker}, timeout=policy.acquisition_timeout_s)
            if not done:
                raise BackendFailure("TIMEOUT", "media acquisition timed out", stage="acquire", retryable=True)
            return await worker
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _acquire_with_fallback(self, source_url: str, *, destination: Path, policy: InspectPolicy,
                                     headers: Mapping[str, str] | None, page_url: str | None) -> AcquiredMedia:
        try:
            acquired = await self._acquire(source_url, destination=destination, policy=policy, headers=headers)
        except BackendFailure as exc:
            if not page_url or exc.code not in YTDLP_FALLBACK_CODES:
                raise
            direct_error = f"{exc.code}: {exc}"
            try:
                acquired = await asyncio.to_thread(ytdlp_download, page_url, destination, policy)
            except BackendFailure as fallback:
                raise BackendFailure(fallback.code, f"direct download failed ({direct_error}); yt-dlp download failed "
                                     f"({fallback})", stage="acquire", retryable=fallback.retryable) from fallback
            acquired.details["direct_error"] = direct_error
            return acquired
        acquired.details.setdefault("method", "direct")
        return acquired

    async def _acquire(self, source_url: str, *, destination: Path, policy: InspectPolicy,
                       headers: Mapping[str, str] | None = None) -> AcquiredMedia:

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
                maximum_bytes=min(policy.maximum_file_bytes, policy.maximum_download_bytes), extra_headers=headers)
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


# 直链下载失败时改走 yt-dlp 的错误类别：被拒（403 等）、直链失效 / 不是视频；超时与超限不重试。
YTDLP_FALLBACK_CODES = frozenset({"ACCESS_RESTRICTED", "MATERIAL_UNAVAILABLE"})
YTDLP_HOSTS = frozenset({"www.youtube.com", "youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"})
# 优先 ≤720p 的音画合一 mp4（itag 18 等，不需要合并）；没有时才下分离流交给 ffmpeg 合并。
YTDLP_FORMAT = ("18/b[ext=mp4][height<=720][vcodec!=none][acodec!=none]/b[height<=720][vcodec!=none][acodec!=none]"
                "/bv*[height<=720]+ba/b")
# 本机没有 ffmpeg 程序时（10-05 晚：yt-dlp 2026.08.19 换客户端后 YouTube 不再给音画合一格式，合并要 ffmpeg）：
# 先试音画合一；没有就分别下视频流、音频流，用 PyAV 不重新编码地封装成 mkv（PyAV 是项目依赖）。
YTDLP_MUXED_FORMAT = "18/b[ext=mp4][height<=720][vcodec!=none][acodec!=none]/b[height<=720][vcodec!=none][acodec!=none]"
# 优先 H.264（解码快、各工具都认）；YouTube 只给 AV1 / VP9 时也照收（PyAV 用 libdav1d 等解码）。
YTDLP_VIDEO_FORMAT = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][ext=mp4]/bv*[height<=720]/bv*"
YTDLP_AUDIO_FORMAT = "ba[ext=m4a]/ba"
# 10-07：YouTube 对某些客户端给的流地址时不时回 403（evidsA8：16 次下载里 3 次声音被拒只剩画面、2 次整个被拒）。
# 默认客户端被 403 拒时，依次指定这几个客户端各重试一次；不认识的客户端名 yt-dlp 只告警跳过。
YTDLP_RETRY_CLIENTS: tuple[str, ...] = ("tv", "mweb", "web_safari", "android_vr", "ios", "web_embedded")
_YTDLP_MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime"}
_AUDIO_SUFFIXES = (".m4a", ".webm", ".mp4", ".opus", ".mka", ".aac", ".ogg")


def _ytdlp_failure(exc: Exception) -> BackendFailure:
    message = str(exc)
    folded = message.casefold()
    code = ("ACCESS_RESTRICTED" if any(x in folded for x in ("403", "forbidden", "sign in", "login", "private",
                                                               "age-restricted", "members-only", "bot"))
            else "TIMEOUT" if "timed out" in folded else "MATERIAL_UNAVAILABLE")
    return BackendFailure(code, f"yt-dlp: {message[:300]}", stage="acquire", retryable=code in {"TIMEOUT", "MATERIAL_UNAVAILABLE"})


# 10-11：EVID 全量里 find_source 有一批原片下载失败（189 的原片排第 1 却没下下来），以前只在 403 时换客户端。
# 现在除了确定换客户端也没用的错误（私享、会员、超时长/超大小、已删除、没有所选格式），都换客户端再试（如 bot 验证）。
_YTDLP_TERMINAL = ("private video", "members-only", "members only", "longer than", "larger than", "max_filesize",
                   "has been removed", "account associated with this video has been terminated", "copyright",
                   "requested format is not available")      # 调用方自己会换成分离流


def _client_retryable(message: str) -> bool:
    folded = message.casefold()
    return not any(x in folded for x in _YTDLP_TERMINAL)


def _forbidden(message: str) -> bool:
    folded = message.casefold()
    return "403" in folded or "forbidden" in folded


def copy_stream_template(output: Any, stream: Any) -> Any:
    """封装用的输出流：只拷贝码流参数，不查编码器。

    PyAV 默认按解码器名字找同名编码器，AV1 的解码器叫 libdav1d、没有同名编码器，会报 UnknownCodecError
    （10-05 晚本机自检）。opaque=True 直接拷贝 codec 参数；老版本 PyAV 没有这个参数时退回原调用。"""

    add = getattr(output, "add_stream_from_template", None)
    if not callable(add):
        return output.add_stream(template=stream)
    try:
        return add(stream, opaque=True)
    except TypeError:
        return add(stream)


def remux_video_audio(video: Path, audio: Path, output: Path) -> None:
    """把单独的视频流与音频流按时间交错、不重新编码地封装成 mkv（只用 PyAV，不需要 ffmpeg 程序）。"""

    import heapq

    import av

    with av.open(str(video)) as vin, av.open(str(audio)) as ain, av.open(str(output), "w", format="matroska") as out:
        sources = []
        for container, streams in ((vin, vin.streams.video), (ain, ain.streams.audio)):
            if not streams:
                continue
            stream = streams[0]
            target = copy_stream_template(out, stream)
            sources.append((container.demux(stream), stream, target))
        if not sources or sources[0][1].type != "video":
            raise BackendFailure("MATERIAL_UNAVAILABLE", "downloaded video stream has no video track", stage="acquire")

        def advance(index: int):
            packets, stream, _ = sources[index]
            for packet in packets:
                if packet.dts is None:          # demux 结束时的空包
                    continue
                return (float(packet.dts * stream.time_base), index, packet)
            return None

        heap = [item for item in (advance(i) for i in range(len(sources))) if item is not None]
        heapq.heapify(heap)
        while heap:
            _, index, packet = heapq.heappop(heap)
            packet.stream = sources[index][2]
            out.mux(packet)
            following = advance(index)
            if following is not None:
                heapq.heappush(heap, following)


class _QuietLogger:
    """10-11：yt-dlp 即使 quiet 也会把每次尝试的 ERROR 打到控制台（换客户端重试时一屏都是）；改为只记下来，最终失败时随异常给出。"""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        self.errors.append(str(msg)[:300])


def ytdlp_download(page_url: str, destination: Path, policy: InspectPolicy, *, factory: Any = None,
                   ffmpeg: str | bool | None = None, video_only_height: int | None = None) -> AcquiredMedia:
    """用 yt-dlp 自己的下载器取 YouTube 视频（同步，调用方放进线程）。只接受 https 的 YouTube 页面地址。

    yt-dlp 会按当前客户端重新解析签名直链并带上对应请求头，直链过期、请求头不符这类 403 由它处理；
    大小与时长上限沿用 InspectPolicy。有 ffmpeg 程序时让 yt-dlp 合并分离流；没有时见 YTDLP_MUXED_FORMAT 的说明。
    factory / ffmpeg 供测试注入（默认 yt_dlp.YoutubeDL / PATH 里的 ffmpeg；ffmpeg=False 表示当作没有）。
    """

    parts = urlsplit(page_url)
    if parts.scheme != "https" or (parts.hostname or "").lower() not in YTDLP_HOSTS:
        raise BackendFailure("ACCESS_RESTRICTED", "yt-dlp download is only enabled for https YouTube pages", stage="acquire")
    if factory is None:
        from yt_dlp import YoutubeDL as factory  # noqa: N813
    ffmpeg_path = (shutil.which("ffmpeg") if ffmpeg is None else ffmpeg) or None
    work = destination.resolve().parent / (destination.name + ".ytdlp")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    maximum_bytes = min(policy.maximum_file_bytes, policy.maximum_download_bytes)

    def too_long(info: Mapping[str, Any], *, incomplete: bool = False) -> str | None:
        duration = info.get("duration")
        if isinstance(duration, (int, float)) and duration > policy.maximum_duration_s:
            return f"video is longer than {policy.maximum_duration_s:.0f} s"
        return None

    clients_used: dict[str, str] = {}
    retries: list[dict[str, str]] = []

    def attempt(selector: str, name: str, client: str | None) -> Mapping[str, Any]:
        options = {"format": selector, "outtmpl": str(work / f"{name}.%(ext)s"), "noplaylist": True, "quiet": True,
                   "no_warnings": True, "noprogress": True, "ignoreconfig": True, "cachedir": False,
                   "max_filesize": maximum_bytes, "socket_timeout": 30, "retries": 5, "fragment_retries": 5,
                   "match_filter": too_long, "overwrites": True, "logger": _QuietLogger()}
        if os.environ.get("FOREN_HTTP_PROXY"):          # 10-12 服务器代理（见 search/openers/http.py）
            options["proxy"] = os.environ["FOREN_HTTP_PROXY"]
        if ffmpeg_path:
            options.update(ffmpeg_location=str(ffmpeg_path), merge_output_format="mp4")
        if client:
            options["extractor_args"] = {"youtube": {"player_client": [client]}}
        with factory(options) as ydl:
            info = ydl.extract_info(page_url, download=True)
        if not isinstance(info, Mapping):
            raise BackendFailure("MATERIAL_UNAVAILABLE", "yt-dlp returned no video information", stage="acquire")
        return info

    def run(selector: str, name: str) -> Mapping[str, Any]:
        """默认客户端下载；被 403 拒时依次换 YTDLP_RETRY_CLIENTS 重试，都不行再报最初那次的错。"""

        try:
            info = attempt(selector, name, None)
            clients_used[name] = "default"
            return info
        except BackendFailure:
            raise
        except Exception as exc:  # noqa: BLE001 —— yt-dlp 各站点报错不统一，归并成稳定类别
            first = exc
            if not _client_retryable(str(exc)):
                raise _ytdlp_failure(exc) from exc
        for client in YTDLP_RETRY_CLIENTS:
            for leftover in work.glob(f"{name}.*"):
                leftover.unlink(missing_ok=True)
            try:
                info = attempt(selector, name, client)
            except Exception as exc:  # noqa: BLE001
                retries.append({"stream": name, "client": client, "error": str(exc)[:160]})
                continue
            clients_used[name] = client
            retries.append({"stream": name, "client": client, "error": ""})
            return info
        raise _ytdlp_failure(first) from first

    def produced(name: str, suffixes: tuple[str, ...]) -> Path | None:
        files = sorted((p for p in work.iterdir() if p.is_file() and p.stem == name and p.suffix.lower() in suffixes),
                       key=lambda p: p.stat().st_size, reverse=True)
        return files[0] if files else None

    try:
        method = "yt-dlp"
        if video_only_height:
            # 10-11 比对原片只要画面（原 DETECT；10-12 起 SEARCH 内容比对与 INSPECT 低清兜底共用）：≤ video_only_height 的单路视频流，不要声音，文件小、请求少
            h = int(video_only_height)
            info = run(f"bv*[height<={h}][vcodec^=avc1]/bv*[height<={h}]/b[height<={h}]/wv*/w", "video")
            result = produced("video", tuple(_YTDLP_MEDIA_TYPES))
            method = f"yt-dlp (video only <= {h}p)"
        elif ffmpeg_path:
            info = run(YTDLP_FORMAT, "video")
            result = produced("video", tuple(_YTDLP_MEDIA_TYPES))
        else:
            try:
                info = run(YTDLP_MUXED_FORMAT, "video")
                result = produced("video", tuple(_YTDLP_MEDIA_TYPES))
            except BackendFailure as exc:
                if "requested format is not available" not in str(exc).casefold() and not _forbidden(str(exc)):
                    raise                           # 音画合一被拒（403）也改走分离流：分离流常走别的地址
                info = run(YTDLP_VIDEO_FORMAT, "video")
                video = produced("video", tuple(_YTDLP_MEDIA_TYPES))
                try:
                    run(YTDLP_AUDIO_FORMAT, "audio")
                    audio = produced("audio", _AUDIO_SUFFIXES)
                except BackendFailure:
                    audio = None                    # 没有音轨也能取帧 / 视觉对齐；asr 会如实报没有音频
                if video is not None and audio is not None:
                    merged = work / "merged.mkv"
                    try:
                        remux_video_audio(video, audio, merged)
                    except BackendFailure:
                        raise
                    except Exception as merge_error:  # noqa: BLE001
                        raise BackendFailure("MATERIAL_UNAVAILABLE", f"could not combine video and audio streams "
                                             f"({type(merge_error).__name__}: {merge_error})", stage="acquire") from merge_error
                    result, method = merged, "yt-dlp+pyav"
                else:
                    result, method = video, "yt-dlp (video only)"
        if result is None:
            raise BackendFailure("MATERIAL_UNAVAILABLE", "yt-dlp produced no video file (filtered by size or duration limit?)",
                                 stage="acquire")
        if result.stat().st_size > maximum_bytes:
            raise BackendFailure("RESOURCE_LIMIT", "downloaded video exceeds the size limit", stage="acquire")
        target = destination.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        shutil.move(str(result), str(target))
        suffix = result.suffix.lower()
        if sniff_media_kind(target, _YTDLP_MEDIA_TYPES[suffix]) != "video":
            target.unlink(missing_ok=True)
            raise BackendFailure("MATERIAL_UNAVAILABLE", "yt-dlp output is not a decodable video", stage="acquire")
        label = str(info.get("title") or info.get("id") or "acquired video")[:200]
        details = {"requested_url": page_url, "final_url": str(info.get("webpage_url") or page_url),
                   "byte_size": target.stat().st_size, "method": method,
                   "format_id": info.get("format_id"), "duration_s": info.get("duration"),
                   "ytdlp_client": dict(clients_used)}
        if retries:
            details["ytdlp_retries"] = retries
        return AcquiredMedia(target, _YTDLP_MEDIA_TYPES[suffix], suffix, label, details)
    finally:
        shutil.rmtree(work, ignore_errors=True)
