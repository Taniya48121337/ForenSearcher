"""通过受限 HTTP 客户端取得 HLS VOD，再由隔离子进程封装本地分片。"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from urllib.parse import urljoin

from yt_dlp.utils import parse_m3u8_attributes

from ..search.openers.http import SafeHttpClient
from .backends.base import BackendFailure
from .policy import InspectPolicy


def _failure(message: str, code: str = "UNSUPPORTED_MEDIA") -> BackendFailure:
    return BackendFailure(code, message, stage="acquire")


def _lines(content: bytes) -> list[str]:
    try:
        lines = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    except UnicodeError as exc:
        raise _failure("HLS playlist is not UTF-8") from exc
    if not lines or lines[0] != "#EXTM3U":
        raise _failure("HLS response is not a playlist")
    return lines


def _duration(lines: list[str]) -> float:
    values = []
    for line in lines:
        if line.startswith("#EXTINF:"):
            try:
                value = float(line.split(":", 1)[1].split(",", 1)[0])
            except ValueError as exc:
                raise _failure("HLS segment duration is invalid") from exc
            if not math.isfinite(value) or value <= 0:
                raise _failure("HLS segment duration must be finite and positive")
            values.append(value)
    return sum(values)


class HlsDownloader:
    """每次调用独占下载预算和暂存目录；远程清单不会直接交给解码器。"""

    def __init__(self, http: SafeHttpClient, policy: InspectPolicy) -> None:
        self.http = http
        self.policy = policy
        self.maximum_download_bytes = min(policy.maximum_file_bytes, policy.maximum_download_bytes)
        self.transferred = 0
        self.resource_count = 0
        self.playlists: list[dict[str, Any]] = []
        self.files: dict[tuple[str, tuple[int, int] | None], str] = {}
        self._initial: dict[str, bytes] = {}
        self._pending: list[asyncio.Task[None]] = []
        self._network_slots = asyncio.Semaphore(4)

    async def _load(self, url: str) -> tuple[list[str], str]:
        if len(self.playlists) >= 32:
            raise _failure("HLS playlist request limit exceeded", "RESOURCE_LIMIT")
        if url in self._initial:
            content, final_url = self._initial.pop(url), url
        else:
            resource = await self.http.fetch(url, maximum_bytes=min(8 * 1024**2, self._remaining()))
            content, final_url = resource.content, resource.final_url
        self._account(len(content))
        lines = _lines(content)
        self.playlists.append({"url": final_url, "content": content.decode("utf-8-sig")})
        return lines, final_url

    def _account(self, count: int) -> None:
        self.transferred += count
        if self.transferred > self.maximum_download_bytes:
            raise _failure("HLS aggregate download size exceeds configured limit", "RESOURCE_LIMIT")

    def _remaining(self) -> int:
        remaining = self.maximum_download_bytes - self.transferred
        if remaining <= 0:
            raise _failure("HLS aggregate download size exceeds configured limit", "RESOURCE_LIMIT")
        return remaining

    async def _download(self, url: str, byte_range: tuple[int, int] | None, root: Path, suffix: str) -> str:
        key = (url, byte_range)
        if key in self.files:
            return self.files[key]
        if self.resource_count >= 20_000:
            raise _failure("HLS resource count exceeds configured limit", "RESOURCE_LIMIT")
        remaining = self._remaining()
        if byte_range is not None and byte_range[1] > remaining:
            raise _failure("HLS byte range exceeds remaining download budget", "RESOURCE_LIMIT")
        name = f"resource-{self.resource_count:05d}{suffix}"
        self.resource_count += 1
        self.files[key] = name
        async def fetch() -> None:
            async with self._network_slots:
                resource = await self.http.download_to_file(url, destination=root/name,
                    maximum_bytes=self.maximum_download_bytes, byte_range=byte_range, on_chunk=self._account)
                if not resource.byte_size:
                    raise _failure("HLS media resource is empty", "MATERIAL_UNAVAILABLE")
        self._pending.append(asyncio.create_task(fetch()))
        return name

    async def _finish_downloads(self) -> None:
        tasks, self._pending = self._pending, []
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _range(value: str | None, url: str, previous: tuple[str, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        try:
            size_text, *offset_text = value.split("@")
            size = int(size_text)
            if len(offset_text) > 1:
                raise ValueError
            if offset_text:
                offset = int(offset_text[0])
            elif previous is not None and previous[0] == url:
                offset = previous[1]
            else:
                raise ValueError
            if size <= 0 or offset < 0:
                raise ValueError
        except ValueError as exc:
            raise _failure("HLS byte range has an invalid or missing offset") from exc
        return offset, size

    async def _media(self, lines: list[str], url: str, root: Path, name: str) -> tuple[Path, float]:
        if "#EXT-X-ENDLIST" not in lines:
            raise _failure("HLS playlist is still live; a complete VOD is required")
        duration = _duration(lines)
        if duration <= 0 or duration > self.policy.maximum_duration_s:
            raise _failure("HLS duration exceeds the configured range", "RESOURCE_LIMIT")
        result = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-PLAYLIST-TYPE:VOD"]
        pending_duration: str | None = None
        pending_range: str | None = None
        previous_range: tuple[str, int] | None = None
        map_range: tuple[str, int] | None = None
        has_map = False
        count = 0
        ended = False
        for line in lines[1:]:
            if line.startswith(("#EXT-X-KEY:", "#EXT-X-SESSION-KEY:")):
                if parse_m3u8_attributes(line.split(":", 1)[1]).get("METHOD") != "NONE":
                    raise _failure("encrypted HLS is not supported")
            elif line in {"#EXT-X-GAP", "#EXT-X-I-FRAMES-ONLY"}:
                raise _failure("HLS playlist does not contain a complete continuous video")
            elif line.startswith("#EXT-X-MAP:"):
                attrs = parse_m3u8_attributes(line.split(":", 1)[1])
                if not attrs.get("URI"):
                    raise _failure("HLS initialization map has no URI")
                source = urljoin(url, attrs["URI"])
                window = self._range(attrs.get("BYTERANGE"), source, map_range)
                local = await self._download(source, window, root, ".mp4")
                map_range = (source, sum(window)) if window else None
                result.append(f'#EXT-X-MAP:URI="{local}"')
                has_map = True
            elif line.startswith("#EXTINF:"):
                if pending_duration is not None:
                    raise _failure("HLS segment URI is missing")
                pending_duration = line.split(":", 1)[1].split(",", 1)[0]
            elif line.startswith("#EXT-X-BYTERANGE:"):
                pending_range = line.split(":", 1)[1]
            elif line.startswith(("#EXT-X-TARGETDURATION:", "#EXT-X-MEDIA-SEQUENCE:", "#EXT-X-DISCONTINUITY-SEQUENCE:")):
                try:
                    if int(line.split(":", 1)[1]) < 0:
                        raise ValueError
                except ValueError as exc:
                    raise _failure("HLS numeric playlist tag is invalid") from exc
                result.append(line)
            elif line == "#EXT-X-DISCONTINUITY":
                result.append(line)
            elif line == "#EXT-X-ENDLIST":
                ended = True
            elif not line.startswith("#"):
                if pending_duration is None or ended or "{$" in line:
                    raise _failure("HLS segment has invalid placement or an unresolved URI")
                source = urljoin(url, line)
                window = self._range(pending_range, source, previous_range)
                local = await self._download(source, window, root, ".m4s" if has_map else ".ts")
                previous_range = (source, sum(window)) if window else None
                result.extend([f"#EXTINF:{pending_duration},", local])
                pending_duration = pending_range = None
                count += 1
        if pending_duration is not None or pending_range is not None or not count:
            raise _failure("HLS playlist has an incomplete segment")
        if not any(line.startswith("#EXT-X-TARGETDURATION:") for line in result):
            raise _failure("HLS playlist has no target duration")
        result.append("#EXT-X-ENDLIST")
        await self._finish_downloads()
        path = root/name
        path.write_text("\n".join(result) + "\n", encoding="utf-8")
        return path, duration

    async def _localize(self, url: str, root: Path, depth: int = 0) -> tuple[Path, float]:
        if depth >= 4:
            raise _failure("HLS playlist nesting exceeds the supported depth", "RESOURCE_LIMIT")
        lines, base = await self._load(url)
        variants: list[tuple[float, dict[str, str], str]] = []
        renditions: list[dict[str, str]] = []
        pending: dict[str, str] | None = None
        for line in lines:
            if line.startswith("#EXT-X-STREAM-INF:"):
                pending = parse_m3u8_attributes(line.split(":", 1)[1])
            elif line.startswith("#EXT-X-MEDIA:"):
                renditions.append(parse_m3u8_attributes(line.split(":", 1)[1]))
            elif line.startswith("#EXT-X-SESSION-KEY:"):
                if parse_m3u8_attributes(line.split(":", 1)[1]).get("METHOD") != "NONE":
                    raise _failure("encrypted HLS is not supported")
            elif not line.startswith("#") and pending is not None:
                try:
                    bandwidth = float(pending.get("AVERAGE-BANDWIDTH") or pending["BANDWIDTH"])
                    if not math.isfinite(bandwidth) or bandwidth <= 0:
                        raise ValueError
                except (KeyError, ValueError) as exc:
                    raise _failure("HLS variant bandwidth is invalid") from exc
                variants.append((bandwidth, pending, urljoin(base, line)))
                pending = None
        if pending is not None:
            raise _failure("HLS variant URI is missing")
        if not variants:
            return await self._media(lines, base, root, f"media-{depth}.m3u8")
        variants.sort(key=lambda item: item[0], reverse=True)
        selected = variants[-1]
        selected_lines: list[str] | None = None
        selected_url = ""
        # 用来源声明的带宽估计容量，实际下载仍逐字节执行总上限。
        for candidate in variants:
            candidate_lines, candidate_url = await self._load(candidate[2])
            if any(line.startswith("#EXT-X-STREAM-INF:") for line in candidate_lines):
                return await self._localize(candidate_url, root, depth + 1)
            selected, selected_lines, selected_url = candidate, candidate_lines, candidate_url
            if _duration(candidate_lines) * candidate[0] / 8 <= self._remaining():
                break
        assert selected_lines is not None
        video, duration = await self._media(selected_lines, selected_url, root, f"video-{depth}.m3u8")
        group = selected[1].get("AUDIO")
        audio = [item for item in renditions if item.get("TYPE") == "AUDIO" and item.get("GROUP-ID") == group]
        external = sorted(audio, key=lambda item: item.get("DEFAULT") == "YES", reverse=True)
        if not external or not external[0].get("URI"):
            return video, duration
        audio_lines, audio_url = await self._load(urljoin(base, external[0]["URI"]))
        audio_path, audio_duration = await self._media(audio_lines, audio_url, root, f"audio-{depth}.m3u8")
        if abs(duration - audio_duration) > max(1.0, duration * 0.01):
            raise _failure("HLS audio and video playlists cover different durations")
        master = root/f"master-{depth}.m3u8"
        master.write_text('#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="audio",DEFAULT=YES,AUTOSELECT=YES,URI="' + audio_path.name +
            '"\n#EXT-X-STREAM-INF:BANDWIDTH=' + str(int(selected[0])) + ',AUDIO="audio"\n' + video.name + '\n', encoding="utf-8")
        return master, duration

    async def acquire(self, url: str, destination: Path, *, initial_content: bytes | None = None) -> dict[str, Any]:
        if initial_content is not None:
            self._initial[url] = initial_content
        with tempfile.TemporaryDirectory(prefix="hls-", dir=destination.parent) as temporary:
            root = Path(temporary)
            try:
                playlist, duration = await self._localize(url, root)
            finally:
                pending, self._pending = self._pending, []
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            process_options: dict[str, Any] = {"creationflags":0x08000000} if os.name == "nt" else {}
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", "foren_searcher.inspect.hls_remux",
                str(playlist), str(destination), str(self.policy.maximum_file_bytes),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                **process_options)
            try:
                stdout, _ = await process.communicate()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            if process.returncode:
                raise _failure("HLS local remux failed: " + stdout.decode("utf-8", errors="replace")[:300], "PROCESS_FAILED")
            return {"transport":"hls", "scope":"complete_vod_playlist", "requested_url":url,
                "manifest_duration_s":duration, "downloaded_bytes":self.transferred,
                "resource_count":self.resource_count, "byte_size":destination.stat().st_size,
                "remux":"PyAV stream copy", "playlists":self.playlists}
