"""本机图片临时传到 imgbb，换一个公开网址给 Serper 的 Google Lens（10-16，负责人同意后加）。

Serper 只能按公开 URL 搜；本机材料的帧没有公开地址。SerpAPI 额度用完以后，要让本机帧也能走 Lens，
就先把帧传到 imgbb，拿到网址再交给 Serper。

- 默认关闭。打开：环境变量 FOREN_LENS_IMAGE_HOST=imgbb，并提供 key（IMGBB_API_KEY 或 .local/imgbb_key.txt，
  在 https://api.imgbb.com/ 免费申请）。
- POST https://api.imgbb.com/1/upload?key=..&expiration=600，表单 image=<base64> → data.url。
  图片 600 秒（10 分钟）后由 imgbb 自动删除；最短 60 秒，可用 FOREN_LENS_IMAGE_TTL_S 调（60–600）。
- 只在"走 Serper、且这张图没有自己的公开地址"时上传；Lens 结果有硬盘缓存，同一张图不会重复上传。
- 备用图床 litterbox（10-16 加）：imgbb 失败才用。POST https://litterbox.catbox.moe/resources/internals/api.php，
  reqtype=fileupload&time=1h，文件字段 fileToUpload → 正文就是网址。不要 key；最短只能 1 小时（没有更短的档，
  也没有删除接口）。FOREN_LENS_IMAGE_FALLBACK=off 关掉备用。
  10-16 本机实测：imgbb 对所有新图片的字节上传都回 400 code 111 "Internal upload error"（key 有效：按网址上传、
  已存过的图都能成功；假 key 回的是 code 100），litterbox 同时可用。

注意：删除以前，拿到网址的人都能看到这张图（网址随机，不会被列出来）。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx

from ..contracts import SearchError, SearchFailure
from ._http import request_bounded
from .keys import first_key_line

IMGBB_API = "https://api.imgbb.com/1/upload"
HOST_MODES = ("off", "imgbb")
DEFAULT_TTL_S = 600
LITTERBOX_API = "https://litterbox.catbox.moe/resources/internals/api.php"
LITTERBOX_TIME = "1h"                 # litterbox 最短的一档（可选 1h / 12h / 24h / 72h）
FALLBACK_MODES = ("off", "litterbox")
IMGBB_BAD_KEY = 100                   # imgbb 的 key 无效：HTTP 400 + error.code 100 "Invalid API v1 key."


def find_imgbb_key(root: str | Path | None = None) -> str | None:
    key = os.environ.get("IMGBB_API_KEY", "").strip()
    if not key and root is not None:
        path = Path(root) / ".local" / "imgbb_key.txt"
        if path.is_file():
            key = first_key_line(path.read_text(encoding="utf-8-sig"))
    return key or None


def resolve_host(root: str | Path | None = None) -> tuple[str | None, str | None, int]:
    """FOREN_LENS_IMAGE_HOST → (图床或 None, key, 过期秒数)。设了 imgbb 却没有 key 时报错。"""

    mode = os.environ.get("FOREN_LENS_IMAGE_HOST", "off").strip().lower() or "off"
    if mode not in HOST_MODES:
        raise ValueError(f"FOREN_LENS_IMAGE_HOST 不认识 {mode!r}；可选 {HOST_MODES}")
    try:
        ttl = int(os.environ.get("FOREN_LENS_IMAGE_TTL_S", "").strip() or DEFAULT_TTL_S)
    except ValueError as exc:
        raise ValueError("FOREN_LENS_IMAGE_TTL_S 要是 60–600 的整数") from exc
    ttl = max(60, min(ttl, DEFAULT_TTL_S))
    if mode == "off":
        return None, None, ttl
    key = find_imgbb_key(root)
    if not key:
        raise ValueError("FOREN_LENS_IMAGE_HOST=imgbb 需要 IMGBB_API_KEY 或 .local/imgbb_key.txt")
    return mode, key, ttl


def resolve_fallback() -> str | None:
    """imgbb 打开时的备用图床：默认 litterbox；FOREN_LENS_IMAGE_FALLBACK=off 关掉。图床整个关着时返回 None。"""

    if (os.environ.get("FOREN_LENS_IMAGE_HOST", "off").strip().lower() or "off") == "off":
        return None
    mode = os.environ.get("FOREN_LENS_IMAGE_FALLBACK", "litterbox").strip().lower() or "litterbox"
    if mode not in FALLBACK_MODES:
        raise ValueError(f"FOREN_LENS_IMAGE_FALLBACK 不认识 {mode!r}；可选 {FALLBACK_MODES}")
    return None if mode == "off" else mode


class ImgbbHost:
    name = "imgbb"

    def __init__(self, key: str, *, client: httpx.AsyncClient, expiration_s: int = DEFAULT_TTL_S,
                 max_response_bytes: int = 64 * 1024) -> None:
        if not key:
            raise ValueError("imgbb needs an API key")
        self._key = key
        self._client = client
        self.expiration_s = max(60, min(int(expiration_s), DEFAULT_TTL_S))
        self._max = max_response_bytes
        self.stats: dict[str, int] = {"uploads": 0, "errors": 0}

    @property
    def label(self) -> str:
        return f"imgbb (expires {self.expiration_s}s)"

    async def upload(self, image: bytes) -> str:
        """返回公开网址；失败抛 SearchFailure(upload, ...)。

        10-16 本机实测 imgbb 对一次 base64 表单上传回过 400 "Internal upload error"：先按文件（multipart）传，
        不成再按 base64 表单传一次；key 无效（401/403，或 400 + code 100）不再试。
        """

        last: SearchFailure | None = None
        for form in ("file", "base64"):
            try:
                return await self._upload_once(image, form)
            except SearchFailure as exc:
                last = exc
                self.stats[f"{form}_errors"] = self.stats.get(f"{form}_errors", 0) + 1
                if exc.error.code == "AUTH_ERROR":
                    break
        self.stats["errors"] += 1
        assert last is not None
        raise last

    async def _upload_once(self, image: bytes, form: str) -> str:
        kwargs: dict = ({"files": {"image": ("query.jpg", image, "image/jpeg")}} if form == "file"
                        else {"data": {"image": base64.b64encode(image).decode("ascii")}})
        try:
            response = await request_bounded(
                self._client, "POST", IMGBB_API, maximum_bytes=self._max, provider_name="imgbb", stage="upload",
                params={"key": self._key, "expiration": str(self.expiration_s)}, **kwargs)
        except httpx.TimeoutException as exc:
            raise SearchFailure(SearchError("upload", "TIMEOUT", "imgbb upload timed out")) from exc
        except httpx.HTTPError as exc:
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "imgbb upload failed")) from exc
        try:
            data = json.loads(response.content)
        except ValueError:
            data = None
        url = ((data.get("data") or {}).get("url")) if isinstance(data, dict) and isinstance(data.get("data"), dict) else None
        if response.status_code != 200 or not isinstance(url, str) or not url.startswith("https://"):
            err = data.get("error") if isinstance(data, dict) else None
            message = err.get("message") if isinstance(err, dict) else ""
            bad_key = response.status_code in (401, 403) or (isinstance(err, dict) and err.get("code") == IMGBB_BAD_KEY)
            raise SearchFailure(SearchError("upload", "AUTH_ERROR" if bad_key else "PROVIDER_ERROR",
                                            f"imgbb upload ({form}) returned HTTP {response.status_code}: {message or 'no url'}"[:200]))
        self.stats["uploads"] += 1
        return url


class LitterboxHost:
    """imgbb 的备用图床：litterbox.catbox.moe，不要 key，1 小时后自动删除（最短一档）。"""

    name = "litterbox"
    label = f"litterbox (expires {LITTERBOX_TIME})"

    def __init__(self, *, client: httpx.AsyncClient, max_response_bytes: int = 64 * 1024) -> None:
        self._client = client
        self._max = max_response_bytes
        self.stats: dict[str, int] = {"uploads": 0, "errors": 0}

    async def upload(self, image: bytes) -> str:
        try:
            response = await request_bounded(
                self._client, "POST", LITTERBOX_API, maximum_bytes=self._max, provider_name="litterbox", stage="upload",
                data={"reqtype": "fileupload", "time": LITTERBOX_TIME},
                files={"fileToUpload": ("query.jpg", image, "image/jpeg")})
        except httpx.TimeoutException as exc:
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("upload", "TIMEOUT", "litterbox upload timed out")) from exc
        except httpx.HTTPError as exc:
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("upload", "NETWORK_ERROR", "litterbox upload failed")) from exc
        except SearchFailure:
            self.stats["errors"] += 1
            raise
        text = response.content.decode("utf-8", "replace").strip()
        if response.status_code != 200 or not text.startswith("https://litter.catbox.moe/"):
            self.stats["errors"] += 1
            raise SearchFailure(SearchError("upload", "PROVIDER_ERROR",
                                            f"litterbox upload returned HTTP {response.status_code}: {text[:120] or 'no url'}"[:200]))
        self.stats["uploads"] += 1
        return text


class ImageHostChain:
    """按顺序试几个图床（imgbb → litterbox）：前一个失败才用下一个；key 无效的图床本 run 里不再试。

    last_used 是最近一次成功上传实际用的图床名。
    """

    def __init__(self, hosts: list) -> None:
        if not hosts:
            raise ValueError("ImageHostChain needs at least one host")
        self._hosts = list(hosts)
        self.label = " → ".join(h.label for h in self._hosts)
        self.last_used: str | None = None
        self.stats: dict = {"uploads": 0, "errors": 0, "fallbacks": 0,
                            "by_host": {h.name: h.stats for h in self._hosts}}

    async def upload(self, image: bytes) -> str:
        last: SearchFailure | None = None
        for index, host in enumerate(list(self._hosts)):
            try:
                url = await host.upload(image)
            except SearchFailure as exc:
                last = exc
                if exc.error.code == "AUTH_ERROR" and len(self._hosts) > 1:
                    self._hosts.remove(host)              # 比如 imgbb key 无效：以后直接用备用
                    self.stats.setdefault("dropped", []).append(f"{host.name}: {exc.error.message}"[:200])
                continue
            self.stats["uploads"] += 1
            self.stats["fallbacks"] += int(index > 0)
            self.last_used = host.name
            return url
        self.stats["errors"] += 1
        assert last is not None
        raise SearchFailure(SearchError(last.error.stage, last.error.code,
                                        f"all image hosts failed; last: {last.error.message}"[:200]))


__all__ = ["DEFAULT_TTL_S", "FALLBACK_MODES", "HOST_MODES", "ImageHostChain", "ImgbbHost", "LitterboxHost",
           "find_imgbb_key", "resolve_fallback", "resolve_host"]
