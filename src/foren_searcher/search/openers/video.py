"""使用 yt-dlp 获取视频来源元数据和一条指定语言字幕。"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any, Callable, cast

from PIL import Image, UnidentifiedImageError

from ..contracts import AUTO_SUBTITLE_LANGUAGE, CandidateSource, OpenSourceRequest, OpenedMaterial, OpenedSource, SearchError, SearchFailure, SearchRequest
from ..policy import SearchPolicy
from .http import SafeHttpClient
from .ytdlp_http import extract_metadata


class VideoSourceOpener:
    """把平台视频页面转换为元数据、封面和可选字幕材料。"""

    def __init__(self, http: SafeHttpClient, policy: SearchPolicy | None = None, youtube_dl_factory: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        """注入安全 HTTP 与可替换的 yt-dlp 工厂。"""

        self.http, self.policy, self._factory = http, policy or http.policy, youtube_dl_factory

    def supports(self, candidate: CandidateSource, output_type: str) -> bool:
        """本打开器只服务 video 输出。"""

        return output_type == "video"

    async def open(self, candidate: CandidateSource, request: SearchRequest) -> OpenedSource:
        """提取视频元数据，再独立尝试封面和字幕。"""

        await self.http.validate_url(candidate.source_url)
        try:
            info = await extract_metadata(self.http, self.policy, self._factory,
                {"skip_download":True, "quiet":True, "noplaylist":True, "ignoreconfig":True,
                 "socket_timeout":self.policy.source_timeout_s, "no_warnings":True}, candidate.source_url)
        except SearchFailure:
            raise
        except Exception as exc:
            # 将 yt-dlp 各站点不一致的消息归并为稳定错误类别。
            message = str(exc).casefold()
            code = "NOT_FOUND" if "unavailable" in message or "deleted" in message else "ACCESS_RESTRICTED" if any(x in message for x in ("private", "login", "sign in", "members-only", "age-restricted")) else "TIMEOUT" if "timed out" in message else "NETWORK_ERROR"
            raise SearchFailure(SearchError("open", code, f"video source metadata failed ({type(exc).__name__})")) from exc  # type: ignore[arg-type]
        if not isinstance(info, Mapping) or not info.get("id") or not info.get("webpage_url"):
            raise SearchFailure(SearchError("extract", "EXTRACT_FAILED", "source did not identify a concrete video"))
        # 只挑选模型判断来源可信度和内容所需的有限字段。
        fields = {
            "title": info.get("title"), "webpage_url": info.get("webpage_url"),
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"), "release_date": info.get("release_date"),
            "duration": info.get("duration"), "description": info.get("description"),
            "extractor": info.get("extractor_key") or info.get("extractor"),
            "live_status": info.get("live_status"), "video_id": info.get("id"),
        }
        clean_fields = {key: value for key, value in fields.items() if value is not None}
        raw_json = json.dumps(info, ensure_ascii=False, default=str).encode("utf-8")
        # 精简 content 给模型阅读，完整原始 JSON 仍作为 Artifact 保存。
        content = json.dumps(clean_fields, ensure_ascii=False, indent=2)
        title = str(info.get("title") or candidate.search_title or candidate.source_url)
        materials: list[OpenedMaterial] = [OpenedMaterial(
            "page", "ready", content, raw_json, "application/json", ".json", ("text",),
            title, "video source fields: " + ", ".join(clean_fields), "SEARCH/open/yt-dlp",
            "source_metadata")]
        cover_url = info.get("thumbnail")
        if isinstance(cover_url, str):
            try:
                cover = await self.http.fetch(cover_url, maximum_bytes=self.policy.maximum_image_bytes)
                try:
                    # 完整解码确保封面不是 HTML 错误页或截断图片。
                    with Image.open(io.BytesIO(cover.content)) as image:
                        image.load()
                        width, height, image_format = image.width, image.height, image.format
                except (UnidentifiedImageError, OSError) as exc:
                    raise SearchFailure(SearchError("extract", "UNSUPPORTED_FORMAT", "video cover is not a complete supported image")) from exc
                materials.append(OpenedMaterial("cover", "ready", json.dumps({"source_url": cover.final_url, "width": width, "height": height, "format": image_format}), cover.content,
                    Image.MIME.get(image_format or "", cover.media_type), "." + (image_format or "jpg").lower(), ("image",), f"cover: {title}", "complete cover image",
                    "SEARCH/open/yt-dlp+http", "source_material"))
            except SearchFailure as exc:
                materials.append(OpenedMaterial("cover", "failed", error=exc.error))
        else:
            materials.append(OpenedMaterial("cover", "absent"))
        if isinstance(request, OpenSourceRequest) and request.include == ("subtitles",):
            # 字幕只有直接 OpenSourceRequest 明确要求时才下载。
            materials.append(await self._subtitle(info, request.subtitle_language, title))
        else:
            materials.append(OpenedMaterial("subtitle", "not_requested"))
        formats = info.get("formats")
        download_check = {"status": "formats_found", "reason": None} if isinstance(formats, list) else {"status": "not_checked", "reason": None}
        return OpenedSource(str(info.get("webpage_url")), tuple(materials), download_check,
            {"status": "not_requested", "evidence_id": None, "error": None})

    async def _subtitle(self, info: Mapping[str, Any], language: str | None, title: str) -> OpenedMaterial:
        """选择一条语言明确的字幕轨道并通过安全 HTTP 下载。"""

        manual_value = info.get("subtitles")
        automatic_value = info.get("automatic_captions")
        manual = cast(Mapping[str, Any], manual_value) if isinstance(manual_value, Mapping) else {}
        automatic = cast(Mapping[str, Any], automatic_value) if isinstance(automatic_value, Mapping) else {}
        # 保留 Provider 顺序，同时合并人工字幕与自动字幕语言并去重。
        languages = list(dict.fromkeys([*manual.keys(), *automatic.keys()]))
        if language == AUTO_SUBTITLE_LANGUAGE:
            # 10-13 compact_tool_params：自动带的字幕只取原语言；选不出就当没有（不报错，HI 没有要字幕）
            original = info.get("language")
            manual_langs = list(manual.keys())
            orig = [x for x in automatic.keys() if isinstance(x, str) and x.endswith("-orig")]
            if isinstance(original, str) and original in languages:
                chosen = original
            elif len(orig) == 1:
                chosen = orig[0]
            elif len(manual_langs) == 1:
                chosen = manual_langs[0]
            elif len(languages) == 1:
                chosen = languages[0]
            else:
                return OpenedMaterial("subtitle", "absent")
        elif language is not None:
            if language not in languages:
                return OpenedMaterial("subtitle", "absent")
            chosen = language
        else:
            original = info.get("language")
            if isinstance(original, str) and original in languages:
                chosen = original
            elif len(languages) == 1:
                chosen = languages[0]
            elif not languages:
                return OpenedMaterial("subtitle", "absent")
            else:
                # 多语言时不能擅自猜测，返回可行动的参数错误。
                return OpenedMaterial("subtitle", "failed", error=SearchError("validate", "INVALID_INPUT", "multiple subtitle languages are available; specify one: " + ", ".join(languages[:20])))
        tracks = manual.get(chosen) or automatic.get(chosen) or []
        if not isinstance(tracks, list) or not tracks:
            return OpenedMaterial("subtitle", "absent")
        # 优先选择常见文本字幕格式，避免图片字幕等不可直接分析格式。
        preferred = next((track for track in tracks if isinstance(track, Mapping) and track.get("ext") in {"vtt", "srt", "ttml"}), tracks[0])
        if not isinstance(preferred, Mapping) or not isinstance(preferred.get("url"), str):
            return OpenedMaterial("subtitle", "failed", error=SearchError("extract", "EXTRACT_FAILED", "selected subtitle track has no downloadable URL"))
        try:
            resource = await self.http.fetch(preferred["url"], maximum_bytes=self.policy.maximum_document_bytes)
        except SearchFailure as exc:
            return OpenedMaterial("subtitle", "failed", error=exc.error)
        generated = chosen not in manual
        header = json.dumps({"language": chosen, "automatic": generated, "format": preferred.get("ext")}, ensure_ascii=False)
        raw_text = resource.content.decode(resource.encoding or "utf-8", errors="replace")
        return OpenedMaterial("subtitle", "ready", header + "\n" + raw_text, resource.content,
            resource.media_type or "text/vtt", "." + str(preferred.get("ext") or "txt"), ("text",),
            f"{title} subtitles ({chosen})", "platform subtitle timestamps", "SEARCH/open/yt-dlp+http", "source_transcript")
