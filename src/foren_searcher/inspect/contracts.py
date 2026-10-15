"""INSPECT 操作的严格公开契约（十种默认操作，另有需显式开启的 diff_media）。

外部字典先由 ``parse_inspect_call`` 转成强类型参数；后端结果再组合成具体 Result，
最后由 ``InspectResponse`` 检查状态、错误和 Evidence 引用闭包。契约层不做媒体处理，
只保证进入 Service 的请求和离开 Service 的响应形状可靠。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, TypeAlias, cast

from ..contracts import EvidenceCard
from ..i18n import tr
from ..ids import is_evidence_id

InspectOperation = Literal["get_frames", "get_pages", "crop", "locate_regions", "ocr", "asr", "audio_qa", "visual_qa", "locate_frames", "align_media", "diff_media", "compare_media", "video_structure", "locate"]
InspectStatus = Literal["ready", "partial", "failed"]
_OPERATIONS = {"get_frames", "get_pages", "crop", "locate_regions", "ocr", "asr", "audio_qa", "visual_qa", "locate_frames", "align_media", "diff_media",
               "compare_media", "video_structure", "locate"}
# 需显式开启的操作：enabled_operations 为 None（默认）时不开放，旧规格与旧运行的工具目录保持不变。
# 由规格 harness_params（如 hi@0.4.2 的 inspect_diff_media）或调用方显式传入 enabled_operations 开启。
OPT_IN_OPERATIONS: frozenset[str] = frozenset({"diff_media", "compare_media", "video_structure", "locate"})
DEFAULT_OPERATIONS: frozenset[str] = frozenset(_OPERATIONS) - OPT_IN_OPERATIONS
# 10-12 合并后的工具目录（规格 hi@0.8.0 起，见 DETECT并入SEARCH与INSPECT_合并方案 §7）：模型只看到这 6 个操作。
# get_frames 兼管 PDF 页（pages / page_range）与裁剪（box），locate 兼管视频找帧与图片定位，compare_media 兼管
# align_media / diff_media / DETECT compare_source，video_structure 取代 DETECT generation_scan。旧操作仍可解析执行，
# 供旧规格与旧 run 复现，但不进入合并目录。
MERGED_OPERATIONS: frozenset[str] = frozenset({"get_frames", "ocr", "asr", "locate", "video_structure", "compare_media"})


def is_merged_catalog(enabled: frozenset[str] | set[str] | None) -> bool:
    """compare_media 开着就是合并目录（hi@0.8.0 起）。"""

    return bool(enabled) and "compare_media" in enabled


# 固定两个目标（顺序为 a/b）的操作。
_PAIR_OPERATIONS = frozenset({"align_media", "diff_media", "compare_media"})
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
_DIFF_MEDIA_NOTE = (
    "逐帧计算两个视频（或两张图片）在何时、何处存在像素差异，targets 顺序为 a/b，须同为视频或同为图片。"
    "返回帧级差异片段（左闭右开的起止时间与帧号、峰值面积、差异框）、逐帧差异面积时间线和“A帧|B帧|差异叠加”对照图 E。"
    "需要确定两段已对齐媒体从哪一帧开始、到哪一帧结束不同或差异在画面哪里时，用它精确计算，不要靠逐张取帧目测。"
    "mapping 表示 time_b_s = scale × time_a_s + offset_s，可直接填 align_media 某个候选 time_mapping 中的 scale 与 offset_s，默认恒等；"
    "region 为 A 画面上的归一化 x/y/w/h；min_area 为差异面积占比下限（默认 0.0005）；register 默认 true，画面尺寸或取景不同时先做全局配准。"
    "结果只说明哪里像素不同，不说明差异内容或哪一侧被编辑。"
)
# English counterparts (spec harness_params.language="en"; see i18n.py).
_OCR_COVERAGE_NOTE_EN = (
    "coverage only describes temporal sampling of a video; null for a still image means not applicable, "
    "not that text recognition is complete or incomplete. OCR itself can still miss characters."
)
_FRAME_RETRIEVAL_NOTE_EN = (
    "Retrieves candidates by single-frame semantic or visual similarity; score is only for ranking and is not "
    "the probability that a match is correct. Similar people, objects or scenes do not prove they come from the "
    "same specific clip; single-frame retrieval does not verify continuous temporal correspondence. For clip "
    "correspondence between videos, use align_media to propose temporal candidates, then compare the actual "
    "details; first appearance and the extent before/after still require checking periods outside the candidates."
)
_DIFF_MEDIA_NOTE_EN = (
    "Computes frame by frame when and where two videos (or two images) differ in pixels; targets are ordered a/b "
    "and must both be videos or both be images. Returns frame-level difference segments (half-open start/end "
    "times and frame numbers, peak area, difference boxes), a per-frame difference-area timeline and an "
    "\"A frame | B frame | difference overlay\" comparison image E. When you need to determine at which frame two "
    "aligned media start and stop differing, or where in the picture they differ, compute it with this "
    "operation instead of eyeballing frames one by one. mapping means time_b_s = scale × time_a_s + offset_s; "
    "you can copy scale and offset_s from a candidate's time_mapping in align_media; default is identity. "
    "region is a normalized x/y/w/h on picture A; min_area is the lower bound on the difference-area fraction "
    "(default 0.0005); register defaults to true and performs global registration first when frame size or "
    "framing differs. The result only says where pixels differ, not what the difference is or which side was edited."
)


_COMPARE_MEDIA_NOTE = (
    "比较两个视频 A（待查）与 B（候选原片或另一版本），targets 顺序为 a/b；也可比较两张图片。"
    "一次给出：A 的哪些时段在 B 里有对应（逐段给出 B 时段与时间映射 scale/offset，含变速、倒放、定格、左右镜像）、"
    "A 里有而 B 里没有的时段（插入或替换）、B 里有而 A 里跳过的时段（删减）、顺序变化与重复使用、"
    "同一时刻画面中不同的区域、同一位置人脸身份不同的人、同一时刻声音不同的时段，并附 A|B 对照图 E。"
    "B 不是 A 的来源时直接说明。range_a_s 给出时，在该段内逐帧精算差异的起止帧；region/min_area 同 diff_media。"
    "mode 默认 auto（画面 + 声音）；audio 只比较声音。结果只说明两份文件哪里相同、哪里不同，不判定哪一份是原件或差异如何产生。"
)
_COMPARE_MEDIA_NOTE_EN = (
    "Compares video A (questioned) with video B (a candidate original or another version), targets ordered a/b; two "
    "images can also be compared. One call reports: which spans of A have a counterpart in B (each with its B span and "
    "time mapping scale/offset, including speed changes, reversal, freeze frames and left-right mirroring), spans of A "
    "that are not in B (inserted or replaced footage), spans of B that A skips (cuts), re-ordering and re-used footage, "
    "regions that differ at the same moment, people whose face identity differs from the face at the same position in "
    "B, and spans where the sound differs at the same moment, with A|B comparison image E. If B is not the source of A "
    "it says so. With range_a_s, frame-exact difference boundaries are computed inside that span; region/min_area as in "
    "frame differencing. mode defaults to auto (picture and sound); audio compares sound only. The result only says "
    "where the two files are the same and where they differ; it does not decide which one is the original or how a "
    "difference was made."
)
# 10-13（规格 harness_params.compact_tool_params，hi@0.10.0 起）：compare_media 不再有 mode 参数——画面对不上且两边都有
# 声音时自动只比声音（原 mode=audio 并入默认）。
_COMPARE_MEDIA_COMPACT_TAIL = "结果只说明两份文件哪里相同、哪里不同，不判定哪一份是原件或差异如何产生。"
_COMPARE_MEDIA_COMPACT_NOTE = _COMPARE_MEDIA_NOTE.replace(
    "mode 默认 auto（画面 + 声音）；audio 只比较声音。",
    "画面与声音都比较；画面对不上而两份都有声音时，自动只比较声音（声音在哪个偏移下对得上、哪些时段不同）。")
_COMPARE_MEDIA_COMPACT_NOTE_EN = _COMPARE_MEDIA_NOTE_EN.replace(
    "mode defaults to auto (picture and sound); audio compares sound only.",
    "Picture and sound are both compared; when the pictures do not match and both files have sound, the sound is "
    "compared on its own (the offset at which it lines up and the spans that differ).")
_VIDEO_STRUCTURE_NOTE = (
    "单个视频的结构：镜头切分（起止、时长）与每个镜头的中间帧图 E；恰好约 10 秒的镜头（常见 AI 视频生成器的固定片长）、"
    "画面定格的时段、重复出现的同一画面、黑场。不需要原片。这些是线索，不是结论；没有线索不说明视频未经剪辑。"
)
_VIDEO_STRUCTURE_NOTE_EN = (
    "The structure of one video: shots between hard cuts (start, end, duration) with a contact-sheet image E of each "
    "shot's middle frame; shots lasting about 10 s (the fixed clip length of common AI video generators), frozen spans, "
    "the same picture re-used in two shots, and black spans. No original needed. These are cues, not conclusions; no "
    "cue does not show the video is unedited."
)
_LOCATE_NOTE = (
    "视频 E：按文字、图片或文字+图片查询找最相关的帧（返回时刻与帧 E）；图片 E：按文字查询定位画面中的对象区域（返回框与裁剪 E）。"
    "score 只用于排序；相似不证明来自同一片段。"
)
_LOCATE_NOTE_EN = (
    "Video E: find the frames that best match a text, image or text+image query (returns times and frame E). "
    "Image E: locate the regions matching a text query (returns boxes and crop E). score is only for ranking; "
    "similarity does not prove the same clip."
)
_GET_FRAMES_MERGED_NOTE = (
    "取图：视频 E 用 times_s 或 range_s/count 取帧（time_s 是真实帧起点）；PDF E 用 pages 或 page_range/count 渲染页面（页码从 1 开始）；"
    "图片 E 用 box 裁剪。box=[x1,y1,x2,y2]（0..1 归一化）也可与视频帧一起用，返回放大后的局部；PDF 页先渲染再对页面图片 E 用 box。"
)
_GET_FRAMES_MERGED_NOTE_EN = (
    "Images from media: for a video E, frames at times_s or over range_s/count (time_s is the actual frame start); for a "
    "PDF E, pages by pages or page_range/count (numbered from 1); for an image E, a crop by box. box=[x1,y1,x2,y2] "
    "(normalised 0..1) also works with video frames and returns the enlarged part; for a PDF, render the page first and "
    "then use box on the page image E."
)
_ASR_MERGED_NOTE = (
    "音频/视频的声音：带起止时间的语音转写，另附非语音声音的时间轴（有声但无人说话的段如音乐或其他声音、静音段）。"
)
_ASR_MERGED_NOTE_EN = (
    "The sound of audio/video: speech transcription in timed segments, plus a timeline of non-speech sound (spans with "
    "sound but no speech, such as music or other sounds, and silent spans)."
)


def inspect_action_schema(merged: bool = False, compact: bool = False) -> dict[str, Any]:
    """与公开参数类型对应的模型工具 schema；跨字段数值关系由解析器检查。

    merged=True（hi@0.8.0 起的合并目录）时 get_frames 多 pages/page_range/box，asr 说明含非语音声音，
    并加入 locate / video_structure / compare_media；实际展示哪些操作仍由 enabled_operations 过滤。
    compact=True（hi@0.10.0 起）：compare_media 不列 mode（声音单比并入默认，见 _COMPARE_MEDIA_COMPACT_NOTE）。"""

    text = {"type": "string", "minLength": 1}
    evidence_id = {"type": "string", "pattern": "^E[1-9][0-9]*$"}
    positive = {"type": "integer", "minimum": 1}
    seconds = {"type": "number", "minimum": 0}
    time_range = {"type": "array", "minItems": 2, "maxItems": 2, "items": seconds,
                  "description": tr("[start, end] 秒，start < end", "[start, end] in seconds, start < end")}

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
                              "description": tr("帧时间满足 0 <= t < 视频时长；结束时刻本身没有帧。时长未知时可省略 times_s，按 count 采样。",
                                                 "Frame times satisfy 0 <= t < video duration; the end instant itself has no frame. "
                                                 "If the duration is unknown, omit times_s and sample by count.")},
                  "range_s": time_range, "count": positive})
    frames["not"] = {"anyOf": [{"required": ["times_s", "range_s"]}, {"required": ["times_s", "count"]}]}
    box = {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1}}
    if merged:
        frames = obj({"times_s": frames["properties"]["times_s"], "range_s": time_range, "count": positive,
                      "pages": {"type": "array", "minItems": 1, "items": positive},
                      "page_range": {"type": "array", "minItems": 2, "maxItems": 2, "items": positive},
                      "box": box})
        frames["not"] = {"anyOf": [{"required": ["times_s", "range_s"]}, {"required": ["times_s", "count"]},
                                   {"required": ["pages", "page_range"]}, {"required": ["pages", "count"]},
                                   {"required": ["times_s", "pages"]}, {"required": ["range_s", "pages"]},
                                   {"required": ["range_s", "page_range"]}, {"required": ["times_s", "page_range"]}]}
    pages = obj({"pages": {"type": "array", "minItems": 1, "items": positive},
                 "page_range": {"type": "array", "minItems": 2, "maxItems": 2, "items": positive},
                 "count": positive})
    pages["not"] = {"anyOf": [{"required": ["pages", "page_range"]}, {"required": ["pages", "count"]}]}
    operations = {
        "get_frames": (tr("视频 E → 指定时刻或区间的图片 E；times_s 与 range_s/count 互斥。time_s 是真实帧起点，frame_duration_s 是容器提供的该帧时长，未知为 null；它不是抽样间隔或目标场景总时长",
                           "video E → image E at given times or over a range; times_s is mutually exclusive with range_s/count. "
                           "time_s is the actual frame start; frame_duration_s is that frame's duration from the container, null if "
                           "unknown; it is not the sampling interval or the total duration of the target scene"), frames),
        "get_pages": (tr("PDF E → 页码从 1 开始的图片 E；pages 与 page_range/count 互斥",
                          "PDF E → image E per page, pages numbered from 1; pages is mutually exclusive with page_range/count"), pages),
        "crop": (tr("单张图片 E → 裁剪图片 E；box=[x1,y1,x2,y2]，0≤x1<x2≤1，0≤y1<y2≤1",
                    "single image E → cropped image E; box=[x1,y1,x2,y2], 0≤x1<x2≤1, 0≤y1<y2≤1"),
                 obj({"box": {"type": "array", "minItems": 4, "maxItems": 4,
                               "items": {"type": "number", "minimum": 0, "maximum": 1}}}, ("box",))),
        "locate_regions": (tr("单张图片中的对象区域定位", "locate object regions in a single image"), obj({"query": text, "top_k": positive}, ("query",))),
        "ocr": (tr("单张图片或单个视频中的文字；PDF 先 get_pages；图片使用 params={}，视频才可指定 range_s。" + _OCR_COVERAGE_NOTE,
                   "text in a single image or a single video; for a PDF, get_pages first; images use params={}, only videos may "
                   "set range_s. " + _OCR_COVERAGE_NOTE_EN), obj({"range_s": time_range})),
        "asr": (tr("音频/视频中的语音转写", "speech transcription of audio/video"), obj({"range_s": time_range, "language": text})),
        "audio_qa": (tr("音频/视频声音问答", "question answering about the sound of audio/video"), obj({"question": text, "range_s": time_range,
                                            "audio_track": {"type": "integer", "minimum": 0}}, ("question",))),
        "visual_qa": (tr("一张或多张图片的画面内容或比较问答；视频先 get_frames，PDF 先 get_pages。文字、数字和书面日期须用 ocr 读取核验；本操作不提供可信文字转录或原文摘录",
                          "question answering about the visual content of one or more images, or comparing them; for a video, "
                          "get_frames first; for a PDF, get_pages first. Text, numbers and written dates must be read and verified "
                          "with ocr; this operation provides no reliable text transcription or verbatim excerpt"), obj({"question": text}, ("question",))),
        "locate_frames": (tr(_FRAME_RETRIEVAL_NOTE, _FRAME_RETRIEVAL_NOTE_EN),
                          obj({"query": {"oneOf": queries}, "range_s": time_range, "top_k": positive}, ("query",))),
        "align_media": (tr("两个音视频 E 的候选对应片段，targets 顺序分别为 a/b；区间和锚点是匹配估计。matches 不是剪辑事件清单，候选可重叠且不要求全部同时成立。多个候选不表示发生了拼接；相似静态画面可能有多个对应，未匹配区间不等于已确认删减，需结合实际帧或声音复核",
                            "candidate corresponding clips of two audio/video E, targets ordered a/b; ranges and anchors are match "
                            "estimates. matches is not a list of edit events; candidates may overlap and need not all hold at once. "
                            "Multiple candidates do not mean splicing occurred; similar static pictures may have several "
                            "correspondences, and an unmatched range is not a confirmed deletion; re-check against actual frames "
                            "or sound"),
                        obj({"mode": {"enum": ["visual", "audio"]}, "range_a_s": time_range,
                             "range_b_s": time_range, "top_k": positive}, ("mode",))),
        "diff_media": (tr(_DIFF_MEDIA_NOTE, _DIFF_MEDIA_NOTE_EN), obj({
            "mapping": obj({"offset_s": {"type": "number", "description": tr("秒，可为负", "seconds, may be negative")},
                            "scale": {"type": "number", "minimum": 0, "maximum": 100, "description": tr("正数，默认 1", "positive, default 1")}}),
            "range_a_s": time_range,
            "region": obj({key: {"type": "number", "minimum": 0, "maximum": 1} for key in ("x", "y", "w", "h")},
                          ("x", "y", "w", "h")),
            "min_area": {"type": "number", "minimum": 0, "maximum": 1, "description": "0 < min_area < 1"},
            "register": {"type": "boolean"}})),
    }
    region_obj = obj({key: {"type": "number", "minimum": 0, "maximum": 1} for key in ("x", "y", "w", "h")}, ("x", "y", "w", "h"))
    if merged:
        operations["get_frames"] = (tr(_GET_FRAMES_MERGED_NOTE, _GET_FRAMES_MERGED_NOTE_EN), frames)
        operations["asr"] = (tr(_ASR_MERGED_NOTE, _ASR_MERGED_NOTE_EN), operations["asr"][1])
    operations["locate"] = (tr(_LOCATE_NOTE, _LOCATE_NOTE_EN),
                            obj({"query": {"oneOf": queries}, "range_s": time_range, "top_k": positive}, ("query",)))
    operations["video_structure"] = (tr(_VIDEO_STRUCTURE_NOTE, _VIDEO_STRUCTURE_NOTE_EN), obj({"range_s": time_range}))
    compare_params = {"mode": {"enum": ["auto", "visual", "audio"]}, "range_a_s": time_range, "region": region_obj,
                      "min_area": {"type": "number", "minimum": 0, "maximum": 1, "description": "0 < min_area < 1"}}
    if compact:
        compare_params.pop("mode")
        operations["compare_media"] = (tr(_COMPARE_MEDIA_COMPACT_NOTE, _COMPARE_MEDIA_COMPACT_NOTE_EN), obj(compare_params))
    else:
        operations["compare_media"] = (tr(_COMPARE_MEDIA_NOTE, _COMPARE_MEDIA_NOTE_EN), obj(compare_params))
    alternatives = []
    for name, (description, params) in operations.items():
        count = 2 if name in _PAIR_OPERATIONS else 1
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


def _signed(value: Any, name: str) -> float:
    """解析可为负的有限数，同时排除 bool。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


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
    # 10-12 合并目录：PDF 页与裁剪也走 get_frames（服务按目标媒体类型改派 get_pages / crop）
    pages: tuple[int, ...] | None = None
    page_range: tuple[int, int] | None = None
    box: tuple[float, float, float, float] | None = None


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
    # 10-12：合并目录下由服务置 True，结果另附非语音声音时间轴（不对模型开放）
    sounds: bool | None = None


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


@dataclass(frozen=True, slots=True)
class DiffMapping:
    """diff_media 的线性时间映射：time_b_s = scale * time_a_s + offset_s。"""

    offset_s: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class DiffRegion:
    """A 画面上的归一化比较区域（左上角 x/y 与宽高 w/h）。"""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class DiffMediaParams:
    """逐帧像素差异：时间映射、A 时间范围、比较区域、面积下限和是否配准。"""

    mapping: DiffMapping | None = None
    range_a_s: tuple[float, float] | None = None
    region: DiffRegion | None = None
    min_area: float | None = None
    register: bool | None = None


@dataclass(frozen=True, slots=True)
class LocateParams:
    """合并目录的 locate：视频按查询找帧（同 locate_frames），图片按文字定位区域（同 locate_regions）。"""

    query: FrameQuery
    range_s: tuple[float, float] | None = None
    top_k: int | None = None


@dataclass(frozen=True, slots=True)
class VideoStructureParams:
    """单个视频的镜头结构，可限定时间范围。"""

    range_s: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class CompareMediaParams:
    """A/B 比较：模式、逐帧精算的 A 时段、比较区域与面积下限。"""

    # "auto_fallback" 只由 InspectService._translate 在 compact_params 打开时写入（模型不能直接给）：
    # 同 auto，另外画面对不上且两边都有声音时只比声音
    mode: Literal["auto", "visual", "audio", "auto_fallback"] | None = None
    range_a_s: tuple[float, float] | None = None
    region: DiffRegion | None = None
    min_area: float | None = None


InspectParams: TypeAlias = GetFramesParams | GetPagesParams | CropParams | LocateRegionsParams | OcrParams | AsrParams | AudioQaParams | VisualQaParams | LocateFramesParams | AlignMediaParams | DiffMediaParams | LocateParams | VideoStructureParams | CompareMediaParams


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
        value = _exact(raw, {"times_s", "range_s", "count", "pages", "page_range", "box"}, set(), "get_frames params")
        times = value.get("times_s")
        parsed = None
        if times is not None:
            if not isinstance(times, list) or not times: raise ValueError("times_s must be a non-empty list")
            parsed = tuple(_number(x, "times_s item") for x in times)
            if value.get("range_s") is not None or value.get("count") is not None: raise ValueError("times_s is mutually exclusive with range_s/count")
        page_params = None
        if value.get("pages") is not None or value.get("page_range") is not None:
            if times is not None or value.get("range_s") is not None:
                raise ValueError("pages/page_range are for PDF Evidence and cannot be combined with times_s/range_s")
            page_params = _parse_params("get_pages", {k: value[k] for k in ("pages", "page_range", "count") if k in value})
        box = None if value.get("box") is None else _box(value["box"])
        return GetFramesParams(parsed, _optional_range(value.get("range_s"), "range_s"),
                               None if value.get("count") is None else _positive(value["count"], "count"),
                               page_params.pages if page_params else None, page_params.page_range if page_params else None, box)
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
    if operation == "diff_media":
        return _parse_diff_params(raw)
    if operation == "locate":
        value = _exact(raw, {"query", "range_s", "top_k"}, {"query"}, "locate params")
        return LocateParams(_parse_query(value["query"]), _optional_range(value.get("range_s"), "range_s"),
                            None if value.get("top_k") is None else _positive(value["top_k"], "top_k"))
    if operation == "video_structure":
        value = _exact(raw, {"range_s"}, set(), "video_structure params")
        return VideoStructureParams(_optional_range(value.get("range_s"), "range_s"))
    if operation == "compare_media":
        value = _exact(raw, {"mode", "range_a_s", "region", "min_area"}, set(), "compare_media params")
        mode = value.get("mode")
        if mode is not None and mode not in {"auto", "visual", "audio"}:
            raise ValueError("mode must be auto, visual or audio")
        diff = _parse_diff_params({k: value[k] for k in ("region", "min_area") if k in value})
        return CompareMediaParams(mode, _optional_range(value.get("range_a_s"), "range_a_s"), diff.region, diff.min_area)
    value = _exact(raw, {"mode", "range_a_s", "range_b_s", "top_k"}, {"mode"}, "align_media params")
    if value["mode"] not in {"visual", "audio"}: raise ValueError("mode must be visual or audio")
    return AlignMediaParams(value["mode"], _optional_range(value.get("range_a_s"), "range_a_s"), _optional_range(value.get("range_b_s"), "range_b_s"), None if value.get("top_k") is None else _positive(value["top_k"], "top_k"))


def _parse_diff_params(raw: Any) -> DiffMediaParams:
    """严格解析 diff_media 参数；所有字段可省略，省略即用默认值。"""

    value = _exact(raw, {"mapping", "range_a_s", "region", "min_area", "register"}, set(), "diff_media params")
    mapping = None
    if value.get("mapping") is not None:
        item = _exact(value["mapping"], {"offset_s", "scale"}, set(), "diff_media mapping")
        offset = _signed(item.get("offset_s", 0.0), "mapping.offset_s")
        scale = _signed(item.get("scale", 1.0), "mapping.scale")
        if not 0 < scale <= 100:
            raise ValueError("mapping.scale must satisfy 0 < scale <= 100")
        mapping = DiffMapping(offset, scale)
    region = None
    if value.get("region") is not None:
        item = _exact(value["region"], {"x", "y", "w", "h"}, {"x", "y", "w", "h"}, "diff_media region")
        x, y, w, h = (_number(item[key], f"region.{key}") for key in ("x", "y", "w", "h"))
        if w <= 0 or h <= 0 or x + w > 1 + 1e-9 or y + h > 1 + 1e-9:
            raise ValueError("region must satisfy w > 0, h > 0, x + w <= 1 and y + h <= 1")
        region = DiffRegion(x, y, w, h)
    min_area = None
    if value.get("min_area") is not None:
        min_area = _number(value["min_area"], "min_area")
        if not 0 < min_area < 1:
            raise ValueError("min_area must satisfy 0 < min_area < 1")
    register = value.get("register")
    if register is not None and not isinstance(register, bool):
        raise ValueError("register must be a boolean")
    return DiffMediaParams(mapping, _optional_range(value.get("range_a_s"), "range_a_s"), region, min_area, register)


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
    expected = 2 if operation in _PAIR_OPERATIONS else None if operation == "visual_qa" else 1
    # align_media / diff_media 固定两个目标；visual_qa 可多目标；其余操作固定一个。
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


# 10-15（A7）：一次 OCR 调用 = 一个结果 E（kind=ocr）。观察行只列出现最多的若干条不同文字和总数；结果 E 里有每个
# 文字框的文字、坐标、置信度，READ_RECORD 能读全。
OCR_RECORD_KIND = "ocr"
OCR_OBSERVATION_TEXTS = 12


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
        """列出被 OCR 的图片 E。

        10-15（A7）：视频 OCR 不再把每个采样帧存成 E（以前一次调用 500 多个帧 E，把视图里的其他卡片挤掉），
        行里的 image_evidence_id 就是输入视频本身、靠 time_s 定位，不算结果引用。"""

        return tuple(dict.fromkeys(x.image_evidence_id for x in self.results if x.time_s is None))

    def text_summary(self, limit: int = OCR_OBSERVATION_TEXTS) -> dict[str, Any]:
        """识别出的文字汇总（观察行和记录开头用），见 ocr_text_summary。"""

        return ocr_text_summary(self.to_dict(), limit)


def _ocr_key(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def ocr_text_summary(result: dict[str, Any], limit: int = OCR_OBSERVATION_TEXTS) -> dict[str, Any]:
    """按"不同文字"汇总一次 OCR 结果（字典形态：OcrResult.to_dict() 或记录里的同名字段）。

    texts：出现帧数最多的 limit 条，再按首次出现的时刻排（图片按阅读顺序）；每条带首末时刻、帧数、最高置信度。
    计数：text_boxes 文字框总数、frames 处理的图片 / 帧数、frames_with_text 有字的帧数、distinct_texts 不同文字数。"""

    rows = [row for row in (result.get("results") or []) if isinstance(row, dict)]
    stats: dict[str, dict[str, Any]] = {}
    order = 0
    boxes = 0
    for row in rows:
        time_s = row.get("time_s")
        for item in row.get("items") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            boxes += 1
            key = _ocr_key(text)
            entry = stats.get(key)
            if entry is None:
                entry = stats[key] = {"text": text, "first_s": time_s, "last_s": time_s, "frames": 0, "max_confidence": None,
                                      "_order": order, "_rows": set()}
                order += 1
            if id(row) not in entry["_rows"]:
                entry["_rows"].add(id(row))
                entry["frames"] += 1
            if isinstance(time_s, (int, float)):
                entry["first_s"] = time_s if entry["first_s"] is None else min(entry["first_s"], time_s)
                entry["last_s"] = time_s if entry["last_s"] is None else max(entry["last_s"], time_s)
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                entry["max_confidence"] = round(max(float(confidence), entry["max_confidence"] or 0.0), 3)
    picked = sorted(stats.values(), key=lambda e: (-e["frames"], e["_order"]))[:max(0, limit)]
    picked.sort(key=lambda e: (e["first_s"] if isinstance(e["first_s"], (int, float)) else 0.0, e["_order"]))
    texts = []
    for entry in picked:
        out = {"text": entry["text"], "frames": entry["frames"], "max_confidence": entry["max_confidence"]}
        if entry["first_s"] is not None:
            out["first_s"] = round(float(entry["first_s"]), 3)
            out["last_s"] = round(float(entry["last_s"]), 3)
        texts.append(out)
    return {"text_boxes": boxes, "frames": len(rows), "frames_with_text": sum(1 for row in rows if row.get("items")),
            "distinct_texts": len(stats), "video": any(row.get("time_s") is not None for row in rows), "texts": texts}


def ocr_summary_line(summary: dict[str, Any], record_id: str | None, source: str | None = None) -> str:
    """观察行 / 历史摘要里的一句话：总数 + 列出的文字（带时刻）+ 去哪读全部。"""

    def when(entry: dict[str, Any]) -> str:
        if entry.get("first_s") is None:
            return ""
        a, b = float(entry["first_s"]), float(entry["last_s"])
        return f" ({a:.1f} s)" if abs(b - a) < 0.05 else f" ({a:.1f}–{b:.1f} s)"

    def n(count: Any, word: str, plural: str | None = None) -> str:
        count = int(count or 0)
        return f"{count} {word if count == 1 else (plural or word + 's')}"

    video = bool(summary.get("video")) or any(e.get("first_s") is not None for e in summary.get("texts") or [])
    unit = "frame" if video else "image"
    head = (f"OCR{(' of ' + source) if source else ''}: {n(summary.get('text_boxes'), 'text box', 'text boxes')}, "
            f"{n(summary.get('distinct_texts'), 'distinct text')} in {summary.get('frames_with_text', 0)} of "
            f"{n(summary.get('frames'), unit)}")
    texts = summary.get("texts") or []
    if not texts:
        return head + "; no text was recognised (OCR can miss text, so this does not show there is none)."
    shown = "; ".join(f"\"{e['text']}\"{when(e)}" for e in texts)
    more = int(summary.get("distinct_texts", 0)) - len(texts)
    tail = (f"; {more} more distinct texts" if more > 0 else "")
    where = (f" READ_RECORD {record_id} lists every text box with its {'time, ' if video else ''}position and confidence."
             if record_id else "")
    return f"{head}. Texts{' (most frequent)' if more > 0 else ''}: {shown}{tail}.{where}"


@dataclass(frozen=True, slots=True)
class AsrSegment:
    """一段带起止秒数的语音转写。"""

    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class SoundSegment:
    """非语音声音时间轴的一段：speech（有人说话）、other_sound（有声但无人说话，如音乐或其他声音）、silence。"""

    start_s: float
    end_s: float
    kind: Literal["speech", "other_sound", "silence"]
    label: str | None = None


@dataclass(frozen=True, slots=True)
class AsrResult(_Result):
    """ASR 的覆盖范围、语言、音轨和分段文本；合并目录下另附声音时间轴。"""

    coverage: TimeCoverage
    language: str | None
    audio_track: int
    segments: tuple[AsrSegment, ...]
    sounds: tuple[SoundSegment, ...] | None = None
    sound_method: str | None = None


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
                        "rate_interpretation": tr("scale 是对应区间的 B 时长/A 时长；播放倍速是其倒数。例如 scale=2/3 表示 B 为 A 的 1.5 倍速。该倍速仅描述候选锚点拟合，不证明整个区间连续或实际编辑过程。",
                                                  "scale is B duration / A duration over the corresponding range; the playback rate is its "
                                                  "reciprocal. For example scale=2/3 means B plays at 1.5x of A. This rate only describes the "
                                                  "fit of the candidate anchors and does not prove the whole range is continuous or the actual "
                                                  "editing process."),
                        "max_anchor_residual_s": round(max(abs(a["time_b_s"] - scale * a["time_a_s"] - offset) for a in anchors), 9),
                        "limitation": tr("仅描述已列候选锚点的线性时间关系；残差不是剪辑边界误差或未采样区间的连续性保证。",
                                         "only describes the linear time relation of the listed candidate anchors; the residual is not "
                                         "an edit-boundary error or a continuity guarantee for unsampled ranges.")}
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
                        "interpretation": tr("matches 的这两个零起始下标对应同一B时段的候选；它们不要求同时成立，不能按列表顺序解释为先后发生的剪辑段。重复画面可有多个来源位置。",
                                              "these two zero-based indices into matches are candidates for the same period of B; they "
                                              "need not both hold and must not be read, by list order, as edit segments occurring in "
                                              "sequence. A repeated picture can have several source positions.")})
        return {"mode": self.mode, "coverage": {"a": self.coverage_a.to_dict(), "b": self.coverage_b.to_dict()},
                "matches": matches, "candidate_relationships": relationships}

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出视觉对齐锚点引用的所有帧 E。"""

        return tuple(x for match in self.matches for anchor in match.anchors for x in (anchor.image_a_evidence_id, anchor.image_b_evidence_id) if x)


@dataclass(frozen=True, slots=True)
class DiffImage:
    """一张“A帧|B帧|差异叠加”对照图 E 及其对应的 A/B 帧时间。"""

    evidence_id: str
    kind: Literal["first", "peak", "last", "single"]
    segment_index: int | None
    frame_a: int | None
    time_a_s: float | None
    frame_b: int | None
    time_b_s: float | None
    changed_area: float


@dataclass(frozen=True, slots=True)
class DiffSegment:
    """连续差异帧组成的一个片段；时间区间左闭右开，帧号 end 为不含。"""

    start_frame_a: int
    end_frame_a: int
    start_s: float
    end_s: float
    end_reason: Literal["no_difference", "range_end", "b_unavailable"]
    start_b_s: float
    end_b_s: float
    differing_frames: int
    first_diff_frame_a: int
    first_diff_s: float
    first_diff_frame_b: int | None
    first_diff_b_s: float
    last_diff_frame_a: int
    last_diff_s: float
    last_diff_frame_b: int | None
    last_diff_b_s: float
    peak_frame_a: int
    peak_s: float
    peak_frame_b: int | None
    peak_b_s: float
    peak_area: float
    mean_area: float
    union_bbox: tuple[float, float, float, float]
    image_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiffMediaResult(_Result):
    """DIFF_MEDIA 的差异片段、时间线、覆盖、阈值与配准信息及对照图。"""

    media_kind: Literal["video", "image"]
    differs: bool
    summary: str
    segments: tuple[DiffSegment, ...]
    image_comparison: dict[str, Any] | None
    timeline: dict[str, Any] | None
    coverage: dict[str, Any]
    frame_rate: dict[str, Any] | None
    mapping: dict[str, Any] | None
    threshold: dict[str, Any]
    min_area: float
    region: tuple[float, float, float, float] | None
    registration: dict[str, Any]
    segment_definition: str | None
    images: tuple[DiffImage, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """固定字段顺序；None 字段保留为 null，便于模型区分“不适用”和“缺失”。"""

        return {"media_kind": self.media_kind, "differs": self.differs, "summary": self.summary,
                "segments": [_jsonify(x) for x in self.segments],
                "image_comparison": _jsonify(self.image_comparison), "timeline": _jsonify(self.timeline),
                "coverage": _jsonify(self.coverage), "frame_rate": _jsonify(self.frame_rate),
                "mapping": _jsonify(self.mapping), "threshold": _jsonify(self.threshold), "min_area": self.min_area,
                "region": list(self.region) if self.region is not None else None,
                "registration": _jsonify(self.registration), "segment_definition": self.segment_definition,
                "images": [{key: value for key, value in asdict(x).items()} for x in self.images],
                "limitations": list(self.limitations)}

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        """列出全部对照图 E（片段内的引用是其子集）。"""

        return tuple(dict.fromkeys([x.evidence_id for x in self.images]
                                   + [e for segment in self.segments for e in segment.image_evidence_ids]))


@dataclass(frozen=True, slots=True)
class ReportImage:
    """compare_media / video_structure 附带的一张对照图或镜头缩略图 E。"""

    evidence_id: str
    kind: str
    index: int | None
    caption: str


@dataclass(frozen=True, slots=True)
class CompareMediaResult(_Result):
    """compare_media：摘要行、结构化报告（对应段、差异、人脸、声音）、对照图。"""

    media_kind: Literal["video", "image", "audio"]
    summary_lines: tuple[str, ...]
    report: dict[str, Any]
    images: tuple[ReportImage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"media_kind": self.media_kind, "summary_lines": list(self.summary_lines), **_jsonify(self.report),
                "images": [_jsonify(x) for x in self.images]}

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        nested = []
        diff = self.report.get("frame_diff") if isinstance(self.report, dict) else None
        if isinstance(diff, dict):
            nested = [x.get("evidence_id") for x in diff.get("images", []) if isinstance(x, dict) and x.get("evidence_id")]
        return tuple(dict.fromkeys([x.evidence_id for x in self.images] + nested))


@dataclass(frozen=True, slots=True)
class VideoStructureResult(_Result):
    """video_structure：摘要行、镜头与线索、镜头缩略图。"""

    summary_lines: tuple[str, ...]
    report: dict[str, Any]
    images: tuple[ReportImage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"summary_lines": list(self.summary_lines), **_jsonify(self.report), "images": [_jsonify(x) for x in self.images]}

    def referenced_evidence_ids(self) -> tuple[str, ...]:
        return tuple(x.evidence_id for x in self.images)


InspectResult: TypeAlias = GetFramesResult | GetPagesResult | CropResult | LocateRegionsResult | OcrResult | AsrResult | AudioQaResult | VisualQaResult | LocateFramesResult | AlignMediaResult | DiffMediaResult | CompareMediaResult | VideoStructureResult


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
