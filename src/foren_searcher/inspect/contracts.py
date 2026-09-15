"""十种 INSPECT 操作的严格公开契约。

外部字典先由 ``parse_inspect_call`` 转成强类型参数；后端结果再组合成具体 Result，
最后由 ``InspectResponse`` 检查状态、错误和 Evidence 引用闭包。契约层不做媒体处理，
只保证进入 Service 的请求和离开 Service 的响应形状可靠。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, TypeAlias, cast

from ..contracts import EvidenceCard
from ..ids import is_evidence_id

InspectOperation = Literal["get_frames", "get_pages", "crop", "locate_regions", "ocr", "asr", "audio_qa", "visual_qa", "locate_frames", "align_media"]
InspectStatus = Literal["ready", "partial", "failed"]
_OPERATIONS = {"get_frames", "get_pages", "crop", "locate_regions", "ocr", "asr", "audio_qa", "visual_qa", "locate_frames", "align_media"}
_OCR_COVERAGE_NOTE = (
    "coverage 仅描述视频的时间采样；静态图片为 null 表示不适用，"
    "不表示文字识别完整或不完整。OCR 本身仍可能漏字。"
)
_FRAME_RETRIEVAL_NOTE = (
    "按单帧语义或外观相似度检索候选，score 仅用于排序，不是匹配正确的概率。"
    "相似人物、物体或场景不能证明来自同一具体片段；单帧检索不验证连续时间对应。"
    "视频间的片段对应可用 align_media 提出时序候选，再比较实际细节；"
    "首次出现和前后范围仍需检查候选之外的时段。"
)


def inspect_action_schema() -> dict[str, Any]:
    """与公开参数类型对应的模型工具 schema；跨字段数值关系由解析器检查。"""

    text = {"type": "string", "minLength": 1}
    evidence_id = {"type": "string", "pattern": "^E[1-9][0-9]*$"}
    positive = {"type": "integer", "minimum": 1}
    seconds = {"type": "number", "minimum": 0}
    time_range = {"type": "array", "minItems": 2, "maxItems": 2, "items": seconds,
                  "description": "[start, end] 秒，start < end"}

    def obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": list(required),
                "additionalProperties": False}

    queries = []
    for kind, fields in (("text", ("text",)), ("image", ("evidence_id",)),
                         ("text_image", ("text", "evidence_id"))):
        queries.append(obj({"type": {"const": kind}, **{
            name: evidence_id if name == "evidence_id" else text for name in fields
        }}, ("type", *fields)))

    frames = obj({"times_s": {"type": "array", "minItems": 1, "items": seconds,
                              "description": "帧时间满足 0 <= t < 视频时长；结束时刻本身没有帧。时长未知时可省略 times_s，按 count 采样。"},
                  "range_s": time_range, "count": positive})
    frames["not"] = {"anyOf": [{"required": ["times_s", "range_s"]}, {"required": ["times_s", "count"]}]}
    pages = obj({"pages": {"type": "array", "minItems": 1, "items": positive},
                 "page_range": {"type": "array", "minItems": 2, "maxItems": 2, "items": positive},
                 "count": positive})
    pages["not"] = {"anyOf": [{"required": ["pages", "page_range"]}, {"required": ["pages", "count"]}]}
    operations = {
        "get_frames": ("视频 E → 指定时刻或区间的图片 E；times_s 与 range_s/count 互斥。time_s 是真实帧起点，frame_duration_s 是容器提供的该帧时长，未知为 null；它不是抽样间隔或目标场景总时长", frames),
        "get_pages": ("PDF E → 页码从 1 开始的图片 E；pages 与 page_range/count 互斥", pages),
        "crop": ("单张图片 E → 裁剪图片 E；box=[x1,y1,x2,y2]，0≤x1<x2≤1，0≤y1<y2≤1",
                 obj({"box": {"type": "array", "minItems": 4, "maxItems": 4,
                               "items": {"type": "number", "minimum": 0, "maximum": 1}}}, ("box",))),
        "locate_regions": ("单张图片中的对象区域定位", obj({"query": text, "top_k": positive}, ("query",))),
        "ocr": ("单张图片或单个视频中的文字；PDF 先 get_pages；图片使用 params={}，视频才可指定 range_s。" + _OCR_COVERAGE_NOTE, obj({"range_s": time_range})),
        "asr": ("音频/视频中的语音转写", obj({"range_s": time_range, "language": text})),
        "audio_qa": ("音频/视频声音问答", obj({"question": text, "range_s": time_range,
                                            "audio_track": {"type": "integer", "minimum": 0}}, ("question",))),
        "visual_qa": ("一张或多张图片的画面内容或比较问答；视频先 get_frames，PDF 先 get_pages。文字、数字和书面日期须用 ocr 读取核验；本操作不提供可信文字转录或原文摘录", obj({"question": text}, ("question",))),
        "locate_frames": (_FRAME_RETRIEVAL_NOTE,
                          obj({"query": {"oneOf": queries}, "range_s": time_range, "top_k": positive}, ("query",))),
        "align_media": ("两个音视频 E 的候选对应片段，targets 顺序分别为 a/b；区间和锚点是匹配估计。matches 不是剪辑事件清单，候选可重叠且不要求全部同时成立。多个候选不表示发生了拼接；相似静态画面可能有多个对应，未匹配区间不等于已确认删减，需结合实际帧或声音复核",
                        obj({"mode": {"enum": ["visual", "audio"]}, "range_a_s": time_range,
                             "range_b_s": time_range, "top_k": positive}, ("mode",))),
    }
    alternatives = []
    for name, (description, params) in operations.items():
        count = 2 if name == "align_media" else 1
        targets: dict[str, Any] = {"type":"array", "minItems":count, "items":evidence_id}
        if name == "visual_qa":
            targets["uniqueItems"] = True
        else:
            targets["maxItems"] = count
        alternatives.append(obj({"targets":targets,
            "request":obj({"operation":{"const":name, "description":description}, "params":params},
                          ("operation", "params"))}, ("targets", "request")))
    return {"oneOf": alternatives}


class InspectValidationError(ValueError):
    """请求校验错误；operation 在可识别时帮助响应保留操作名。"""

    def __init__(self, message: str, operation: InspectOperation | None = None) -> None:
        """保存人类可读消息和可选操作。"""

        super().__init__(message)
        self.operation = operation


def _exact(value: Any, allowed: set[str], required: set[str], context: str) -> dict[str, Any]:
    """要求字典只含允许字段，并包含全部必填字段。"""

    if not isinstance(value, dict) or not required <= set(value) or not set(value) <= allowed:
        raise ValueError(f"{context} fields must be within {sorted(allowed)} and include {sorted(required)}")
    return value


def _number(value: Any, name: str) -> float:
    """解析有限非负数，同时排除作为 int 子类的 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _positive(value: Any, name: str) -> int:
    """解析严格正整数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    """解析可为零的整数，例如音轨下标。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _text(value: Any, name: str) -> str:
    """解析并去除首尾空白的非空文本。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _range(value: Any, name: str) -> tuple[float, float]:
    """解析 ``[start, end]`` 秒区间，并要求 start < end。"""

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two times")
    start, end = _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")
    if start >= end:
        raise ValueError(f"{name} must satisfy start < end")
    return start, end


def _optional_range(value: Any, name: str) -> tuple[float, float] | None:
    """允许 None 的时间区间解析包装器。"""

    return None if value is None else _range(value, name)


def _box(value: Any) -> tuple[float, float, float, float]:
    """解析 0..1 归一化裁剪框 ``x1,y1,x2,y2``。"""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must contain four normalized coordinates")
    x1, y1, x2, y2 = tuple(_number(x, "box coordinate") for x in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("box must satisfy 0 <= min < max <= 1")
    return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class GetFramesParams:
    """按明确时间点，或按区间和数量采样视频帧。"""

    times_s: tuple[float, ...] | None = None
    range_s: tuple[float, float] | None = None
    count: int | None = None


@dataclass(frozen=True, slots=True)
class GetPagesParams:
    """按明确页码，或按页范围和数量渲染 PDF 页面。"""

    pages: tuple[int, ...] | None = None
    page_range: tuple[int, int] | None = None
    count: int | None = None


@dataclass(frozen=True, slots=True)
class CropParams:
    """对单张图片使用归一化坐标裁剪。"""

    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class LocateRegionsParams:
    """用文本查询在图片中定位若干区域。"""

    query: str
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class OcrParams:
    """对图片/PDF/视频做 OCR，可限制视频时间范围。"""

    range_s: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class AsrParams:
    """对音视频做语音识别，可限制时间范围与语言。"""

    range_s: tuple[float, float] | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class AudioQaParams:
    """针对指定音频范围和音轨提出自然语言问题。"""

    question: str
    range_s: tuple[float, float] | None = None
    audio_track: int | None = None


@dataclass(frozen=True, slots=True)
class VisualQaParams:
    """针对一个或多个视觉 Evidence 提问。"""

    question: str


@dataclass(frozen=True, slots=True)
class TextFrameQuery:
    """仅文本的帧定位查询。"""

    type: Literal["text"]
    text: str


@dataclass(frozen=True, slots=True)
class ImageFrameQuery:
    """以一张 Evidence 图片为查询的帧定位。"""

    type: Literal["image"]
    evidence_id: str


@dataclass(frozen=True, slots=True)
class TextImageFrameQuery:
    """同时使用参考图片和文字约束的帧定位。"""

    type: Literal["text_image"]
    evidence_id: str
    text: str


FrameQuery: TypeAlias = TextFrameQuery | ImageFrameQuery | TextImageFrameQuery


@dataclass(frozen=True, slots=True)
class LocateFramesParams:
    """在视频时间区间内返回与查询最相关的若干帧。"""

    query: FrameQuery
    range_s: tuple[float, float] | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class AlignMediaParams:
    """以视觉或音频方式对齐两份媒体的时间轴。"""

    mode: Literal["visual", "audio"]
    range_a_s: tuple[float, float] | None = None
    range_b_s: tuple[float, float] | None = None
    top_k: int | None = None


InspectParams: TypeAlias = GetFramesParams | GetPagesParams | CropParams | LocateRegionsParams | OcrParams | AsrParams | AudioQaParams | VisualQaParams | LocateFramesParams | AlignMediaParams


def _parse_query(value: Any) -> FrameQuery:
    """严格解析 locate_frames 内嵌的三种查询联合。"""

    if not isinstance(value, dict):
        raise ValueError("query must be an object")
    kind = value.get("type")
    if kind == "text":
        _exact(value, {"type", "text"}, {"type", "text"}, "text query")
        return TextFrameQuery("text", _text(value["text"], "query.text"))
    if kind == "image":
        _exact(value, {"type", "evidence_id"}, {"type", "evidence_id"}, "image query")
        if not is_evidence_id(value["evidence_id"]):
            raise ValueError("query.evidence_id must be canonical")
        return ImageFrameQuery("image", value["evidence_id"])
    if kind == "text_image":
        _exact(value, {"type", "evidence_id", "text"}, {"type", "evidence_id", "text"}, "text_image query")
        if not is_evidence_id(value["evidence_id"]):
            raise ValueError("query.evidence_id must be canonical")
        return TextImageFrameQuery("text_image", value["evidence_id"], _text(value["text"], "query.text"))
    raise ValueError("query.type must be text, image, or text_image")


def _parse_params(operation: InspectOperation, raw: Any) -> InspectParams:
    """依据 operation 把 params 精确分派到对应参数 dataclass。"""

    if operation == "get_frames":
        # times_s 与 range_s/count 是两套互斥选帧方式。
        value = _exact(raw, {"times_s", "range_s", "count"}, set(), "get_frames params")
        times = value.get("times_s")
        parsed = None
        if times is not None:
            if not isinstance(times, list) or not times: raise ValueError("times_s must be a non-empty list")
            parsed = tuple(_number(x, "times_s item") for x in times)
            if value.get("range_s") is not None or value.get("count") is not None: raise ValueError("times_s is mutually exclusive with range_s/count")
        return GetFramesParams(parsed, _optional_range(value.get("range_s"), "range_s"), None if value.get("count") is None else _positive(value["count"], "count"))
    if operation == "get_pages":
        value = _exact(raw, {"pages", "page_range", "count"}, set(), "get_pages params")
        pages, page_range = value.get("pages"), value.get("page_range")
        parsed_pages = None
        if pages is not None:
            if not isinstance(pages, list) or not pages: raise ValueError("pages must be a non-empty list")
            parsed_pages = tuple(_positive(x, "page") for x in pages)
            if page_range is not None or value.get("count") is not None: raise ValueError("pages is mutually exclusive with page_range/count")
        parsed_range = None
        if page_range is not None:
            if not isinstance(page_range, (list, tuple)) or len(page_range) != 2: raise ValueError("page_range must contain two pages")
            start, end = _positive(page_range[0], "page_range[0]"), _positive(page_range[1], "page_range[1]")
            if start > end: raise ValueError("page_range must satisfy start <= end")
            parsed_range = (start, end)
        return GetPagesParams(parsed_pages, parsed_range, None if value.get("count") is None else _positive(value["count"], "count"))
    if operation == "crop":
        value = _exact(raw, {"box"}, {"box"}, "crop params"); return CropParams(_box(value["box"]))
    if operation == "locate_regions":
        value = _exact(raw, {"query", "top_k"}, {"query"}, "locate_regions params"); return LocateRegionsParams(_text(value["query"], "query"), None if value.get("top_k") is None else _positive(value["top_k"], "top_k"))
    if operation == "ocr":
        value = _exact(raw, {"range_s"}, set(), "ocr params"); return OcrParams(_optional_range(value.get("range_s"), "range_s"))
    if operation == "asr":
        value = _exact(raw, {"range_s", "language"}, set(), "asr params"); return AsrParams(_optional_range(value.get("range_s"), "range_s"), None if value.get("language") is None else _text(value["language"], "language"))
    if operation == "audio_qa":
        value = _exact(raw, {"question", "range_s", "audio_track"}, {"question"}, "audio_qa params"); return AudioQaParams(_text(value["question"], "question"), _optional_range(value.get("range_s"), "range_s"), None if value.get("audio_track") is None else _nonnegative_int(value["audio_track"], "audio_track"))
    if operation == "visual_qa":
        value = _exact(raw, {"question"}, {"question"}, "visual_qa params"); return VisualQaParams(_text(value["question"], "question"))
    if operation == "locate_frames":
        value = _exact(raw, {"query", "range_s", "top_k"}, {"query"}, "locate_frames params"); return LocateFramesParams(_parse_query(value["query"]), _optional_range(value.get("range_s"), "range_s"), None if value.get("top_k") is None else _positive(value["top_k"], "top_k"))
    value = _exact(raw, {"mode", "range_a_s", "range_b_s", "top_k"}, {"mode"}, "align_media params")
    if value["mode"] not in {"visual", "audio"}: raise ValueError("mode must be visual or audio")
    return AlignMediaParams(value["mode"], _optional_range(value.get("range_a_s"), "range_a_s"), _optional_range(value.get("range_b_s"), "range_b_s"), None if value.get("top_k") is None else _positive(value["top_k"], "top_k"))


def _jsonify(value: Any) -> Any:
    """递归把 dataclass、元组和领域对象转换为 JSON 友好值。"""

    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(cast(Any, value)).items() if item is not None}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_jsonify(x) for x in value]
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class InspectRequest:
    """已解析的具体操作及其强类型参数。"""

    operation: InspectOperation
    params: InspectParams

    def to_dict(self) -> dict[str, Any]:
        """序列化操作及参数。"""

        return {"operation": self.operation, "params": _jsonify(self.params)}


@dataclass(frozen=True, slots=True)
class InspectCall:
    """一次 INSPECT 调用：若干目标 E 加一个操作请求。"""

    targets: tuple[str, ...]
    request: InspectRequest

    def to_dict(self) -> dict[str, Any]:
        """生成执行审计可保存的公开字典。"""

        return {"targets": list(self.targets), "request": self.request.to_dict()}


def parse_inspect_call(value: Any) -> InspectCall:
    """严格解析顶层调用、目标数量和操作专属参数。"""

    if not isinstance(value, dict) or set(value) != {"targets", "request"}:
        raise InspectValidationError("INSPECT fields must be targets and request")
    targets = value["targets"]
    if not isinstance(targets, list) or not targets or not all(is_evidence_id(x) for x in targets):
        raise InspectValidationError("targets must be a non-empty list of canonical Evidence IDs")
    request = value["request"]
    if not isinstance(request, dict) or set(request) != {"operation", "params"}:
        raise InspectValidationError("request fields must be operation and params")
    raw_operation = request["operation"]
    if not isinstance(raw_operation, str) or raw_operation not in _OPERATIONS:
        raise InspectValidationError("unknown INSPECT operation")
    operation = cast(InspectOperation, raw_operation)
    expected = 2 if operation == "align_media" else None if operation == "visual_qa" else 1
    # align_media 固定两个目标；visual_qa 可多目标；其余操作固定一个。
    if expected is not None and len(targets) != expected:
        raise InspectValidationError(f"{operation} requires exactly {expected} target(s)", operation)
    if operation == "visual_qa" and len(set(targets)) != len(targets):
        raise InspectValidationError("visual_qa targets must not contain duplicates", operation)
    try:
        params = _parse_params(operation, request["params"])
    except (TypeError, ValueError) as exc:
        raise InspectValidationError(str(exc), operation) from exc
    return InspectCall(tuple(targets), InspectRequest(operation, params))


@dataclass(frozen=True, slots=True)
class InspectFeedback:
    """对调用方可见的警告或错误，包含阶段、定位和重试提示。"""

    severity: Literal["warning", "error"]
    stage: Literal["validate", "resolve", "acquire", "probe", "decode", "process", "save"]
    code: Literal["INVALID_INPUT", "NOT_FOUND", "ACCESS_RESTRICTED", "MATERIAL_UNAVAILABLE", "UNSUPPORTED_MEDIA", "NO_AUDIO", "OUT_OF_RANGE", "RESOURCE_LIMIT", "TIMEOUT", "DECODE_FAILED", "PROCESS_FAILED", "SAVE_FAILED", "COUNT_REDUCED"]
    message: str
    target_evidence_id: str | None = None
    locator: dict[str, Any] | None = None
    retryable: bool = False
    def __post_init__(self) -> None:
        """COUNT_REDUCED 只能是警告，其余反馈码必须是错误。"""

        if self.code == "COUNT_REDUCED" and self.severity != "warning":
            raise ValueError("COUNT_REDUCED feedback must be a warning")
        if self.code != "COUNT_REDUCED" and self.severity != "error":
            raise ValueError(f"{self.code} feedback must be an error")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("feedback message must be non-empty text")
        if self.target_evidence_id is not None and not is_evidence_id(self.target_evidence_id):
            raise ValueError("feedback target_evidence_id must be canonical")

    def to_dict(self) -> dict[str, Any]:
        """序列化反馈并去掉 None 字段。"""

        return _jsonify(asdict(self))


@dataclass(frozen=True, slots=True)
class TimeCoverage:
    """ASR 等时间操作实际处理和失败的区间。"""

    range_s: tuple[float, float]
    processed_ranges: tuple[tuple[float, float], ...]
    failed_ranges: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化覆盖范围。"""

        return _jsonify(asdict(self))


@dataclass(frozen=True, slots=True)
class OcrSampling:
    """视频 OCR 各采样阶段使用的帧数统计。"""

    base_fps: float
    base_frames: int
    scene_frames: int
    refinement_frames: int
    processed_frames: int


@dataclass(frozen=True, slots=True)
class OcrVideoCoverage:
    """视频 OCR 的时间范围、采样统计和失败区间。"""

    range_s: tuple[float, float]
    sampling: OcrSampling
    failed_ranges: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化 OCR 覆盖信息。"""

        return _jsonify(asdict(self))


@dataclass(frozen=True, slots=True)
class LocateFramesCoverage:
    """帧定位的粗采样数、精化区间和失败区间。"""

    range_s: tuple[float, float]
    coarse_frames: int
    refined_ranges: tuple[tuple[float, float], ...]
    failed_ranges: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化帧定位覆盖信息。"""

        return _jsonify(asdict(self))


@dataclass(frozen=True, slots=True)
class AlignmentSideCoverage:
    """媒体对齐某一侧的处理范围和处理量。"""

    range_s: tuple[float, float]
    unit: Literal["frames", "seconds"]
    value: int | float
    failed_ranges: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """把处理量包装成带单位对象，避免数字含义不清。"""

        return {"range_s": list(self.range_s), "processed": {"unit": self.unit, "value": self.value}, "failed_ranges": [list(x) for x in self.failed_ranges]}


class _Result:
    """各操作结果的共享序列化和 Evidence 引用接口。"""

    def to_dict(self) -> dict[str, Any]:
        """默认把结果 dataclass 递归序列化。"""

        return _jsonify(asdict(cast(Any, self)))

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """返回结果内直接引用的派生 E；无引用的结果默认返回空。"""

        return ()


@dataclass(frozen=True, slots=True)
class FrameResult:
    """抽取帧的时间点、尺寸及容器声明的单帧时长；缺失时长保持未知。"""

    evidence_id: str
    time_s: float
    width: int
    height: int
    frame_duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class GetFramesResult(_Result):
    """GET_FRAMES 返回的有序帧集合。"""

    frames: tuple[FrameResult, ...]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出所有新帧 Evidence。"""

        return tuple(x.evidence_id for x in self.frames)


@dataclass(frozen=True, slots=True)
class PageResult:
    """一张渲染 PDF 页面的 E ID、页码和像素尺寸。"""

    evidence_id: str
    page_number: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class GetPagesResult(_Result):
    """GET_PAGES 返回的有序页面集合。"""

    pages: tuple[PageResult, ...]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出所有新页面 Evidence。"""

        return tuple(x.evidence_id for x in self.pages)


@dataclass(frozen=True, slots=True)
class CropResult(_Result):
    """CROP 的派生图片及实际像素裁剪框。"""

    evidence_id: str
    width: int
    height: int
    box_px: tuple[int, int, int, int]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """返回唯一裁剪图 E。"""

        return (self.evidence_id,)


@dataclass(frozen=True, slots=True)
class RegionResult:
    """一个匹配区域、对应裁剪 E、文本和相关性分数。"""

    crop_evidence_id: str
    box: tuple[float, float, float, float]
    box_px: tuple[int, int, int, int]
    matched_text: str
    score: float


@dataclass(frozen=True, slots=True)
class LocateRegionsResult(_Result):
    """LOCATE_REGIONS 的原图、查询和区域列表。"""

    image_evidence_id: str; query: str; regions: tuple[RegionResult, ...]
    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """响应需闭包原图 E 以及每张区域裁剪 E。"""

        return (self.image_evidence_id,) + tuple(x.crop_evidence_id for x in self.regions)


@dataclass(frozen=True, slots=True)
class OcrItem:
    """一段 OCR 文本及其归一化框和可选置信度。"""

    text: str
    box: tuple[float, float, float, float]
    confidence: float | None


@dataclass(frozen=True, slots=True)
class OcrImageResult:
    """一张图片或视频帧的 OCR 结果。"""

    image_evidence_id: str
    time_s: float | None
    items: tuple[OcrItem, ...]


@dataclass(frozen=True, slots=True)
class OcrResult(_Result):
    """OCR 总结果；静态图片无 coverage，视频有采样覆盖。"""

    coverage: OcrVideoCoverage | None
    results: tuple[OcrImageResult, ...]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出被 OCR 的图片或派生帧 E。"""

        return tuple(x.image_evidence_id for x in self.results)


@dataclass(frozen=True, slots=True)
class AsrSegment:
    """一段带起止秒数的语音转写。"""

    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class AsrResult(_Result):
    """ASR 的覆盖范围、语言、音轨和分段文本。"""

    coverage: TimeCoverage
    language: str | None
    audio_track: int
    segments: tuple[AsrSegment, ...]


@dataclass(frozen=True, slots=True)
class AudioQaResult(_Result):
    """音频问答答案及其分析范围和限制。"""

    range_s: tuple[float, float]
    audio_track: int
    answer: str
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisualQaResult(_Result):
    """视觉问答使用的图片 E、答案和限制。"""

    image_evidence_ids: tuple[str, ...]
    answer: str
    limitations: tuple[str, ...]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """返回回答实际观察的全部图片 E。"""

        return self.image_evidence_ids


@dataclass(frozen=True, slots=True)
class LocatedFrame:
    """一张匹配帧的 E ID、视频时间点和 0..1 相似度。"""

    evidence_id: str
    time_s: float
    score: float

    def __post_init__(self) -> None:
        """确保 ID 规范、时间非负且相似度分数位于 0..1。"""

        if not is_evidence_id(self.evidence_id):
            raise ValueError("evidence_id must be canonical")
        _number(self.time_s, "time_s")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(float(self.score)) or not 0 <= self.score <= 1:
            raise ValueError("score must be a finite number in [0, 1]")


@dataclass(frozen=True, slots=True)
class LocateFramesResult(_Result):
    """LOCATE_FRAMES 的采样覆盖与排序候选。"""

    coverage: LocateFramesCoverage
    candidates: tuple[LocatedFrame, ...]

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出所有候选帧 Evidence。"""

        return tuple(x.evidence_id for x in self.candidates)


@dataclass(frozen=True, slots=True)
class AlignmentAnchor:
    """媒体 A/B 的一对对应时间点和可选可视化帧 E。"""

    time_a_s: float
    time_b_s: float
    image_a_evidence_id: str | None
    image_b_evidence_id: str | None


@dataclass(frozen=True, slots=True)
class AlignmentMatch:
    """两段对应区间、相似分数及若干锚点。"""

    range_a_s: tuple[float, float]
    range_b_s: tuple[float, float]
    score: float
    anchors: tuple[AlignmentAnchor, ...]


@dataclass(frozen=True, slots=True)
class AlignMediaResult(_Result):
    """ALIGN_MEDIA 的模式、两侧覆盖和匹配区间。"""

    mode: Literal["visual", "audio"]
    coverage_a: AlignmentSideCoverage
    coverage_b: AlignmentSideCoverage
    matches: tuple[AlignmentMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        """将 A/B 覆盖信息放在同一个 coverage 对象中。"""

        matches = _jsonify(self.matches)
        for match in matches:
            anchors = match["anchors"]
            for anchor in anchors:
                anchor["time_b_minus_a_s"] = round(anchor["time_b_s"] - anchor["time_a_s"], 9)
            # 偏移不恒定也可能是整体变速；拟合只描述候选锚点，不判定编辑方式。
            match["time_mapping"] = None
            if len(anchors) >= 2:
                mean_a = sum(a["time_a_s"] for a in anchors) / len(anchors)
                mean_b = sum(a["time_b_s"] for a in anchors) / len(anchors)
                variance = sum((a["time_a_s"] - mean_a) ** 2 for a in anchors)
                if variance > 1e-12:
                    scale = sum((a["time_a_s"] - mean_a) * (a["time_b_s"] - mean_b) for a in anchors) / variance
                    offset = mean_b - scale * mean_a
                    match["time_mapping"] = {
                        "formula": "time_b_s = scale * time_a_s + offset_s",
                        "scale": round(scale, 9), "offset_s": round(offset, 9), "anchor_count": len(anchors),
                        "playback_rate_b_relative_to_a": round(1 / scale, 9) if scale > 1e-12 else None,
                        "rate_interpretation": "scale 是对应区间的 B 时长/A 时长；播放倍速是其倒数。例如 scale=2/3 表示 B 为 A 的 1.5 倍速。该倍速仅描述候选锚点拟合，不证明整个区间连续或实际编辑过程。",
                        "max_anchor_residual_s": round(max(abs(a["time_b_s"] - scale * a["time_a_s"] - offset) for a in anchors), 9),
                        "limitation": "仅描述已列候选锚点的线性时间关系；残差不是剪辑边界误差或未采样区间的连续性保证。"}
        relationships = []
        for i, left in enumerate(self.matches):
            if len(left.anchors) < 2:
                continue
            left_times = [anchor.time_b_s for anchor in left.anchors]
            for j in range(i + 1, len(self.matches)):
                right = self.matches[j]
                if len(right.anchors) < 2:
                    continue
                right_times = [anchor.time_b_s for anchor in right.anchors]
                start = max(min(left_times), min(right_times))
                end = min(max(left_times), max(right_times))
                if start < end:
                    relationships.append({
                        "match_indices": [i, j], "overlap_in_b_s": [start, end],
                        "relationship": "overlapping_candidates",
                        "interpretation": "matches 的这两个零起始下标对应同一B时段的候选；它们不要求同时成立，不能按列表顺序解释为先后发生的剪辑段。重复画面可有多个来源位置。"})
        return {"mode": self.mode, "coverage": {"a": self.coverage_a.to_dict(), "b": self.coverage_b.to_dict()},
                "matches": matches, "candidate_relationships": relationships}

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出视觉对齐锚点引用的所有帧 E。"""

        return tuple(x for match in self.matches for anchor in match.anchors for x in (anchor.image_a_evidence_id, anchor.image_b_evidence_id) if x)


InspectResult: TypeAlias = GetFramesResult | GetPagesResult | CropResult | LocateRegionsResult | OcrResult | AsrResult | AudioQaResult | VisualQaResult | LocateFramesResult | AlignMediaResult


@dataclass(frozen=True, slots=True)
class InspectResponse:
    """一次 INSPECT 的顶层响应和完整 Evidence 引用闭包。"""

    operation: InspectOperation | None
    status: InspectStatus
    result: InspectResult | None
    result_evidence_id: str | None
    evidence: tuple[EvidenceCard, ...]
    feedback: tuple[InspectFeedback, ...]
    execution_id: int | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """校验状态与错误一致，并确保结果引用都附有 EvidenceCard。"""

        errors = any(x.severity == "error" for x in self.feedback)
        if self.status == "failed" and (self.result is not None or not errors):
            raise ValueError("failed response requires null result and error feedback")
        if self.status == "partial" and (self.result is None or not errors):
            raise ValueError("partial response requires a result and error feedback")
        if self.status == "ready" and (self.result is None or errors):
            raise ValueError("ready response requires a result and no error feedback")
        card_ids = {x.evidence_id for x in self.evidence}
        referenced = set(self.result.referenced_evidence_ids()) if self.result else set()
        if self.result_evidence_id:
            referenced.add(self.result_evidence_id)
        if not referenced <= card_ids:
            raise ValueError("INSPECT response contains unclosed Evidence references")

    def to_dict(self) -> dict[str, Any]:
        """递归序列化结果、证据卡和反馈。"""

        return {"operation": self.operation, "status": self.status, "result": self.result.to_dict() if self.result else None, "result_evidence_id": self.result_evidence_id, "evidence": [x.to_dict() for x in self.evidence], "feedback": [x.to_dict() for x in self.feedback]}


def invalid_response(operation: InspectOperation | None, message: str) -> InspectResponse:
    """为请求解析失败构造契约合法的 failed 响应。"""

    return InspectResponse(operation, "failed", None, None, (), (InspectFeedback("error", "validate", "INVALID_INPUT", message),))
