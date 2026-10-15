"""SEARCH 的请求、候选、材料和响应契约。

阅读顺序可以理解为：调用方先提交 ``SearchQueryRequest``（搜索新来源）或
``OpenSourceRequest``（打开指定来源）；Provider 返回 ``CandidateBatch``；打开器把
来源变成 ``OpenedSource``；Service 最后向 Workspace 返回 ``SearchResponse``。
所有 dataclass 都在创建时校验不变量，因此后续代码可以相信对象内部字段彼此一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit

from ..contracts import EvidenceCard

InputType = Literal["text", "image", "text_image"]
OutputType = Literal["text", "image", "video"]
MaterialStatus = Literal["ready", "partial", "not_requested", "absent", "failed"]
StopReason = Literal["target_reached", "candidates_exhausted", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class SearchError:
    """可序列化的 SEARCH 错误，stage 表示失败环节，code 表示稳定分类。"""

    stage: Literal["validate", "search", "upload", "open", "fetch", "extract", "save"]
    code: Literal[
        "INVALID_INPUT", "NOT_FOUND", "ACCESS_RESTRICTED", "TIMEOUT",
        "NETWORK_ERROR", "UNSUPPORTED_FORMAT", "AUTH_ERROR", "QUOTA_EXCEEDED",
        "RATE_LIMITED", "EXTRACT_FAILED", "SAVE_FAILED", "PROVIDER_ERROR",
    ]
    message: str

    def __post_init__(self) -> None:
        """阻止未知阶段、未知错误码和空消息进入系统。"""

        if self.stage not in {"validate", "search", "upload", "open", "fetch", "extract", "save"}:
            raise ValueError("invalid SEARCH error stage")
        if self.code not in {
            "INVALID_INPUT", "NOT_FOUND", "ACCESS_RESTRICTED", "TIMEOUT", "NETWORK_ERROR",
            "UNSUPPORTED_FORMAT", "AUTH_ERROR", "QUOTA_EXCEEDED", "RATE_LIMITED",
            "EXTRACT_FAILED", "SAVE_FAILED", "PROVIDER_ERROR",
        }:
            raise ValueError("invalid SEARCH error code")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("SEARCH error message must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """转成只含普通 JSON 类型的字典。"""

        return {"stage": self.stage, "code": self.code, "message": self.message}


class SearchFailure(RuntimeError):
    """SEARCH 内部携带结构化 SearchError 的异常。"""

    def __init__(self, error: SearchError, *, retry_after_s: float | None = None) -> None:
        """保存错误和 Provider 建议的最短重试等待时间。"""

        if retry_after_s is not None and (
            not isinstance(retry_after_s, (int, float))
            or isinstance(retry_after_s, bool)
            or retry_after_s < 0
        ):
            raise ValueError("retry_after_s must be a non-negative number or null")
        super().__init__(error.message)
        self.error = error
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True)
class SearchQueryRequest:
    """主动搜索请求，支持文本、图片或文字加图片三种输入。"""

    input_type: InputType
    query: str | dict[str, str]
    output_type: OutputType

    def __post_init__(self) -> None:
        """依据 input_type 校验 query 的具体形状和长度。"""

        _validate_output_type(self.output_type)
        if self.input_type == "text":
            if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 500:
                raise ValueError("text query must be 1..500 characters")
        elif self.input_type == "image":
            if not _is_evidence_id(self.query):
                raise ValueError("image query must be one Evidence ID")
        elif self.input_type == "text_image":
            if not isinstance(self.query, dict) or set(self.query) != {"evidence_id", "text"}:
                raise ValueError("text_image query must contain exactly evidence_id and text")
            if not _is_evidence_id(self.query.get("evidence_id")):
                raise ValueError("text_image.evidence_id must be an Evidence ID")
            text = self.query.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError("text_image.text must be 1..500 characters")
        else:
            raise ValueError("input_type must be text, image, or text_image")

    def to_dict(self) -> dict[str, Any]:
        """生成 Provider 和审计表可直接使用的请求字典。"""

        return {"input_type": self.input_type, "query": self.query, "output_type": self.output_type}


# 10-13（规格 compact_tool_params，hi@0.10.0 起）：SearchService.auto_subtitles 打开时，打开视频的请求自动带
# include=("subtitles",) 与这个语言值：选原语言（平台标的原语言 / 自动字幕的 -orig / 唯一一种），选不出就不取字幕
AUTO_SUBTITLE_LANGUAGE = "auto"


@dataclass(frozen=True, slots=True)
class OpenSourceRequest:
    """直接打开 URL 或已有 R 候选的请求。"""

    source: str
    output_type: OutputType
    include: tuple[str, ...] = ()
    subtitle_language: str | None = None

    def __post_init__(self) -> None:
        """校验来源格式，并限制字幕选项只能用于视频。"""

        _validate_output_type(self.output_type)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a URL or R identifier")
        if not (_is_http_url(self.source) or _is_candidate_id(self.source)):
            raise ValueError("source must be an HTTP(S) URL or R identifier")
        if self.include not in ((), ("subtitles",), ("dates",)):
            raise ValueError("include must be empty, ('subtitles',) or ('dates',)")
        if self.include == ("dates",):
            # 10-13 日期查询（openers/dates.py）：只给文字结果；不带字幕语言
            if self.output_type != "text":
                raise ValueError("dates can only be requested with output_type text")
            if self.subtitle_language is not None:
                raise ValueError("subtitle_language requires include=('subtitles',)")
            return
        if self.subtitle_language is not None:
            if self.include != ("subtitles",) or not isinstance(self.subtitle_language, str) or not self.subtitle_language.strip():
                raise ValueError("subtitle_language requires include=('subtitles',)")
        if self.include and self.output_type != "video":
            raise ValueError("subtitles can only be requested for video output")

    def to_dict(self) -> dict[str, Any]:
        """把元组等不可直接表达的字段转换为 JSON 友好形式。"""

        return {"source": self.source, "output_type": self.output_type,
                "include": list(self.include), "subtitle_language": self.subtitle_language}


SearchRequest: TypeAlias = SearchQueryRequest | OpenSourceRequest


def parse_search_request(value: Any) -> SearchRequest:
    """从外部字典严格解析两种请求，拒绝拼错或多余字段。"""

    if not isinstance(value, dict):
        raise ValueError("SEARCH request must be an object")
    keys = set(value)
    if "source" in keys:
        # 出现 source 就按“打开来源”解析；其余请求必须是查询搜索。
        allowed = {"source", "output_type", "include", "subtitle_language"}
        required = {"source", "output_type"}
        if not required <= keys or not keys <= allowed:
            raise ValueError(f"open request fields must be within {sorted(allowed)} and include {sorted(required)}")
        include = value.get("include", ())
        if not isinstance(include, (list, tuple)):
            raise ValueError("include must be a list")
        return OpenSourceRequest(value["source"], value["output_type"], tuple(include), value.get("subtitle_language"))
    expected = {"input_type", "query", "output_type"}
    if keys != expected:
        raise ValueError(f"search request fields must be {sorted(expected)}")
    return SearchQueryRequest(value["input_type"], value["query"], value["output_type"])


@dataclass(frozen=True, slots=True)
class CandidateSource:
    """Provider 返回但尚未打开的候选来源。"""

    source_url: str
    search_title: str | None = None
    search_snippet: str | None = None
    provider: str = "direct"
    engine: str = "direct"
    provider_rank: int = 1
    candidate_type: Literal["page", "document", "image", "video", "unknown"] = "unknown"
    image_url: str | None = None
    thumbnail_url: str | None = None
    platform: str | None = None
    platform_video_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 URL 安全形状、Provider 元数据和候选类型。"""

        if not _is_http_url(self.source_url):
            raise ValueError("candidate source_url must be an HTTP(S) URL without credentials")
        for name in ("image_url", "thumbnail_url"):
            value = getattr(self, name)
            if value is not None and not _is_http_url(value):
                raise ValueError(f"candidate {name} must be an HTTP(S) URL without credentials")
        for name in ("provider", "engine"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate {name} must be non-empty")
        if not isinstance(self.provider_rank, int) or isinstance(self.provider_rank, bool) or self.provider_rank <= 0:
            raise ValueError("candidate provider_rank must be a positive integer")
        if self.candidate_type not in {"page", "document", "image", "video", "unknown"}:
            raise ValueError("invalid candidate_type")
        if not isinstance(self.extra, dict):
            raise ValueError("candidate extra must be an object")


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """一次 Provider 调用的规范化候选与原始响应。"""

    provider: str
    engine: str
    candidates: tuple[CandidateSource, ...]
    raw_response: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """确保批次来源明确，并且 candidates 全是规范对象。"""

        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("candidate batch provider must be non-empty")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("candidate batch engine must be non-empty")
        if not isinstance(self.candidates, tuple) or not all(isinstance(item, CandidateSource) for item in self.candidates):
            raise ValueError("candidate batch candidates must be a tuple of CandidateSource values")
        if not isinstance(self.raw_response, dict) or not isinstance(self.request_metadata, dict):
            raise ValueError("candidate batch metadata must be objects")


@dataclass(frozen=True, slots=True)
class OpenedMaterial:
    """打开器产生、尚未写入 Global Store 的单份材料。"""

    kind: Literal["page", "document", "image", "cover", "thumbnail", "subtitle", "dates"]
    status: MaterialStatus
    content: str = ""
    artifact_bytes: bytes = b""
    media_type: str = "application/octet-stream"
    suffix: str = ".bin"
    modalities: tuple[str, ...] = ("text",)
    label: str = "source material"
    locator: str = "complete material"
    acquired_by: str = "SEARCH/open"
    production_method: str = "source_excerpt"
    error: SearchError | None = None

    def __post_init__(self) -> None:
        """保证 status 与内容、字节和 error 的组合不会自相矛盾。"""

        if self.kind not in {"page", "document", "image", "cover", "thumbnail", "subtitle", "dates"}:
            raise ValueError("invalid material kind")
        if self.status not in {"ready", "partial", "not_requested", "absent", "failed"}:
            raise ValueError("invalid material status")
        if self.status in {"ready", "partial"} and (not self.artifact_bytes or not self.content):
            raise ValueError("ready/partial material requires artifact bytes and content")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed material requires an error")
        if self.status in {"not_requested", "absent"} and self.error is not None:
            raise ValueError("not_requested/absent material cannot have an error")


@dataclass(frozen=True, slots=True)
class OpenedSource:
    """一次来源打开操作的内部结果，可包含多个材料。"""

    final_url: str
    materials: tuple[OpenedMaterial, ...]
    download_check: dict[str, Any] | None = None
    video_material: dict[str, Any] | None = None
    discovered_links: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MaterialResult:
    """材料写入 Store 后对 Workspace 返回的公开结果。"""

    kind: str
    status: MaterialStatus
    evidence: tuple[EvidenceCard, ...] = ()
    error: SearchError | None = None

    def __post_init__(self) -> None:
        """校验状态、Evidence 和错误三者的合法组合。"""

        if self.kind not in {"page", "document", "image", "cover", "thumbnail", "subtitle", "video", "dates"}:
            raise ValueError("invalid material kind")
        if self.status not in {"ready", "partial", "not_requested", "absent", "failed"}:
            raise ValueError("invalid material status")
        if self.status in {"ready", "partial"} and not self.evidence:
            raise ValueError("ready/partial result requires Evidence")
        if self.status == "ready" and self.error is not None:
            raise ValueError("ready result cannot have an error")
        if self.status == "partial" and self.error is None:
            raise ValueError("partial result requires an error")
        if self.status == "failed" and (self.evidence or self.error is None):
            raise ValueError("failed result requires only an error")
        if self.status in {"not_requested", "absent"} and (self.evidence or self.error):
            raise ValueError("not_requested/absent result cannot carry Evidence/error")

    def to_dict(self) -> dict[str, Any]:
        """递归序列化 EvidenceCard 和 SearchError。"""

        return {"kind": self.kind, "status": self.status,
                "evidence": [item.to_dict() for item in self.evidence],
                "error": self.error.to_dict() if self.error else None}


# source = 材料的大部分画面在这段视频里；partial = 一部分画面在（多来源剪辑的来源之一）；none = 没对上；
# download_failed = 低清画面下载失败；not_checked = 本次没比（超出每次 / 每个材料的比对名额，或时长差得太远）
CONTENT_MATCH_STATUSES = frozenset({"source", "partial", "none", "download_failed", "not_checked"})


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一个 Candidate 的最终搜索结果及其各类材料。"""

    candidate_id: str
    source_url: str
    search_title: str | None
    search_snippet: str | None
    materials: tuple[MaterialResult, ...]
    download_check: dict[str, Any] | None = None
    video_material: dict[str, Any] | None = None
    # 10-12（DETECT find_source 并入 SEARCH）：视频结果与本调查材料视频的画面比对，见 search/source_match.py
    content_match: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """验证材料不重复，以及视频下载状态字段始终成对出现。"""

        if not _is_candidate_id(self.candidate_id):
            raise ValueError("candidate_id must be an R identifier")
        kinds = [item.kind for item in self.materials]
        if len(kinds) != len(set(kinds)):
            raise ValueError("materials must contain each kind at most once")
        if (self.download_check is None) != (self.video_material is None):
            raise ValueError("video status fields must be provided together")
        if self.download_check is not None:
            if set(self.download_check) != {"status", "reason"}:
                raise ValueError("download_check requires exactly status and reason")
            status = self.download_check.get("status")
            if status not in {"not_checked", "formats_found", "blocked", "unknown"}:
                raise ValueError("invalid download_check.status")
            if "reason" not in self.download_check:
                raise ValueError("download_check requires reason")
            if self.download_check.get("reason") is not None and not isinstance(self.download_check.get("reason"), str):
                raise ValueError("download_check.reason must be text or null")
        if self.video_material is not None:
            if set(self.video_material) != {"status", "evidence_id", "error"}:
                raise ValueError("video_material requires exactly status, evidence_id, and error")
            status = self.video_material.get("status")
            evidence_id = self.video_material.get("evidence_id")
            error = self.video_material.get("error")
            if status not in {"not_requested", "ready", "failed"}:
                raise ValueError("invalid video_material.status")
            if status == "ready" and not _is_evidence_id(evidence_id):
                raise ValueError("ready video material requires evidence_id")
            if status == "failed" and not isinstance(error, dict):
                raise ValueError("failed video material requires error")
            if status != "ready" and evidence_id is not None:
                raise ValueError("non-ready video material cannot contain evidence_id")
            if status != "failed" and error is not None:
                raise ValueError("non-failed video material cannot contain error")
        if self.content_match is not None:
            if self.video_material is None:
                raise ValueError("content_match is only for video results")
            if not {"material", "status"} <= set(self.content_match):
                raise ValueError("content_match requires material and status")
            if self.content_match["status"] not in CONTENT_MATCH_STATUSES:
                raise ValueError("invalid content_match.status")

    def to_dict(self) -> dict[str, Any]:
        """序列化结果，并只在视频场景加入视频专属字段。"""

        value: dict[str, Any] = {"candidate_id": self.candidate_id, "source_url": self.source_url,
                 "search_title": self.search_title, "search_snippet": self.search_snippet,
                 "materials": [item.to_dict() for item in self.materials]}
        if self.download_check is not None:
            value["download_check"] = self.download_check
        if self.video_material is not None:
            value["video_material"] = self.video_material
        if self.content_match is not None:
            value["content_match"] = self.content_match
        return value


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """一次 SEARCH 动作的顶层响应。"""

    output_type: OutputType
    stop_reason: StopReason
    results: tuple[SearchResult, ...]
    error: SearchError | None = None

    def __post_init__(self) -> None:
        """校验停止原因、顶层错误和 output_type 对应的材料排列。"""

        _validate_output_type(self.output_type)
        if self.stop_reason not in {"target_reached", "candidates_exhausted", "timeout", "error"}:
            raise ValueError("invalid SEARCH stop reason")
        if self.stop_reason in {"error", "timeout"} and self.error is None:
            raise ValueError("error/timeout response requires a top-level error")
        if self.stop_reason in {"target_reached", "candidates_exhausted"} and self.error is not None:
            raise ValueError("normally completed response cannot contain a top-level error")
        if self.stop_reason == "target_reached" and not self.results:
            raise ValueError("target_reached response requires results")
        # 每种 output_type 都有固定材料槽位，调用方无需猜测缺了哪一项。
        expected = {
            "text": ("page", "document"),
            "image": ("image", "page", "thumbnail"),
            "video": ("page", "cover", "subtitle"),
        }[self.output_type]
        for result in self.results:
            kinds = tuple(item.kind for item in result.materials)
            if kinds != expected and not (self.output_type == "text" and kinds == ("dates",)):
                raise ValueError("result material kinds do not match output_type")
            has_video_fields = result.download_check is not None and result.video_material is not None
            if has_video_fields != (self.output_type == "video"):
                raise ValueError("video status fields do not match output_type")

    def to_dict(self) -> dict[str, Any]:
        """生成模型动作观察值使用的 JSON 字典。"""

        return {"output_type": self.output_type, "stop_reason": self.stop_reason,
                "results": [item.to_dict() for item in self.results],
                "error": self.error.to_dict() if self.error else None}


def _validate_output_type(value: Any) -> None:
    """集中校验三个公开输出类型。"""

    if not isinstance(value, str) or value not in {"text", "image", "video"}:
        raise ValueError("output_type must be text, image, or video")


def _is_evidence_id(value: Any) -> bool:
    """检查严格 E<n>，并拒绝 E01 之类带前导零的写法。"""

    return isinstance(value, str) and value.startswith("E") and value[1:].isdigit() and str(int(value[1:])) == value[1:]


def _is_candidate_id(value: Any) -> bool:
    """检查严格 R<n>，并拒绝前导零。"""

    return isinstance(value, str) and value.startswith("R") and value[1:].isdigit() and str(int(value[1:])) == value[1:]


def _is_http_url(value: str) -> bool:
    """只接受无用户名和密码的 HTTP(S) URL。"""

    try:
        # 读取 port 会触发 urllib 对非法端口的 ValueError。
        parsed = urlsplit(value)
        _ = parsed.port
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False
