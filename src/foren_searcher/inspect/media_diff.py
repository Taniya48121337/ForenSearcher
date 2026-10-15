"""diff_media 的确定性像素差异计算：解码游标、全局配准、逐帧差异掩码、片段与对照图。

本模块只做数值处理，不碰 Store、不分配 E 编号；``LocalInspectBackend.diff_media`` 负责编排并产出
``PendingArtifact``。所有阈值与形态学参数固定或由抽样帧确定性地估计，同一输入总得到同一结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import av
import cv2
import numpy as np

from .backends.base import BackendFailure

WORK_WIDTH = 480                 # 差异计算的工作分辨率宽度上限
MAX_DIFF_FRAMES = 1800           # 单次最多比较的 A 帧数；超过时等间隔抽帧并给出 COUNT_REDUCED
TIMELINE_MAX_VALUES = 600        # 时间线最多保留的数值个数；超过时按桶取最大值
MAX_SEGMENTS = 32                # 结果中最多列出的片段数（控制观察长度）
MAX_DIFF_IMAGES = 8              # 对照图总数上限（另受 maximum_visual_qa_images 约束）
DEFAULT_MIN_AREA = 0.0005
THRESHOLD_BASE, THRESHOLD_SLOPE, THRESHOLD_MAX = 16, 4, 48
REGISTRATION_SAMPLES = 6
SEEK_AFTER_S = 30.0              # 起点超过该秒数才跳转解码；更早的起点从头顺序解码以得到准确帧号
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

THRESHOLD_METHOD = (
    f"自动：clip({THRESHOLD_BASE} + {THRESHOLD_SLOPE} × 噪声估计, {THRESHOLD_BASE}, {THRESHOLD_MAX})；"
    "噪声估计为若干抽样帧对差值第 75 百分位的中位数（同源同编码的媒体噪声接近 0，阈值即下限）"
)
PREPROCESSING = (
    f"两侧缩放到工作分辨率（宽 ≤{WORK_WIDTH} 像素，B 先按 registration 对齐到 A 的画面）；各做 3×3 高斯模糊；"
    "逐像素取 B/G/R 三通道绝对差的最大值，与阈值比较后做 3×3 椭圆核开运算去除孤立压缩噪点"
)
SEGMENT_DEFINITION = (
    "片段是差异面积 ≥ min_area 的连续已比较 A 帧。start_s 是首个差异帧的起始时间；end_s 是其后第一个"
    "已比较且无差异帧的起始时间（end_reason=no_difference），或比较范围末尾（range_end），或其后第一个"
    "因映射时间超出 B 而未比较的帧（b_unavailable）；区间左闭右开，end_frame_a 不含。不做平滑，也不合并"
    "被无差异帧隔开的片段（merge_gap_frames=0）。start_b_s 是首个差异帧对应的 B 帧时间，end_b_s 是按 mapping "
    "换算的 end_s。"
)
# 10-05：英文版（规格 language="en"，见 i18n.tr）；旧常量保留给旧规格
THRESHOLD_METHOD_EN = (
    f"automatic: clip({THRESHOLD_BASE} + {THRESHOLD_SLOPE} × noise estimate, {THRESHOLD_BASE}, {THRESHOLD_MAX}); "
    "the noise estimate is the median over sampled frame pairs of the 75th percentile of differences (near 0 for media of the same source and encoding, so the threshold is the lower bound)"
)
PREPROCESSING_EN = (
    f"both sides scaled to the working resolution (width ≤{WORK_WIDTH} px, B first aligned to the A picture by registration); 3×3 Gaussian blur on each; "
    "per pixel the maximum absolute difference over the B/G/R channels, compared with the threshold, followed by a 3×3 elliptical opening to remove isolated compression noise"
)
SEGMENT_DEFINITION_EN = (
    "A segment is a run of consecutive compared A frames whose changed area is ≥ min_area. start_s is the start time of the first differing frame; end_s is the start time of the "
    "first later compared frame without difference (end_reason=no_difference), or the end of the compared range (range_end), or the first later "
    "frame not compared because its mapped time is outside B (b_unavailable); intervals are half-open and end_frame_a is excluded. No smoothing, and segments "
    "separated by frames without difference are not merged (merge_gap_frames=0). start_b_s is the B frame time of the first differing frame; end_b_s is end_s "
    "converted by the mapping."
)
LIMITATIONS_EN = (
    "Results rest on a per-frame pixel difference threshold: compression noise, brightness or colour adjustments and registration errors can produce differences that are not edits, and can also hide real low-contrast differences.",
    "It only says when and where in the picture A and B differ in pixels, not what the difference is or which side was edited; confirm the content by viewing the composites or fetching the original frames with get_frames at those times.",
    "Frames whose changed area is below min_area count as no difference and are not reported; no difference does not prove the two sides are identical.",
    "The A/B frame correspondence is fully determined by the mapping (time_b_s = scale × time_a_s + offset_s, identity by default); when the mapping is wrong, differences may come from time misalignment rather than different content.",
    "Registration estimates a single global homography; local motion, parallax or zoom cannot be described by one transform, and residual misalignment shows up as differences at edges.",
)


def _tr(zh: str, en: str) -> str:
    from ..i18n import tr

    return tr(zh, en)


def limitations() -> tuple[str, ...]:
    from ..i18n import english

    return LIMITATIONS_EN if english() else LIMITATIONS


def threshold_method() -> str:
    return _tr(THRESHOLD_METHOD, THRESHOLD_METHOD_EN)


def preprocessing() -> str:
    return _tr(PREPROCESSING, PREPROCESSING_EN)


def segment_definition() -> str:
    return _tr(SEGMENT_DEFINITION, SEGMENT_DEFINITION_EN)


LIMITATIONS = (
    "结果基于逐帧像素差阈值：压缩噪声、亮度或色彩调整、配准误差可能产生并非编辑的差异，也可能掩盖低对比度的真实差异。",
    "它只说明 A/B 在何时、画面何处像素不同，不说明差异是什么内容，也不说明哪一侧被编辑；内容需查看对照图或按时间用 get_frames 取原帧确认。",
    "差异面积小于 min_area 的帧记为无差异，不会报告；无差异不证明两侧完全一致。",
    "A/B 的帧对应完全由 mapping 决定（time_b_s = scale × time_a_s + offset_s，默认恒等）；mapping 不准时，差异可能来自时间错位而不是内容不同。",
    "配准只估计一个全局单应性；局部运动、视差或镜头变焦无法由单一变换描述，残余错位会表现为边缘处的差异。",
)


def work_size(width: int, height: int) -> tuple[int, int]:
    """保持宽高比，把宽度限制到 WORK_WIDTH。"""

    scale = min(1.0, WORK_WIDTH / max(1, width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """尺寸已符合时原样返回，否则用 INTER_AREA 缩放。"""

    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def blurred(image: np.ndarray) -> np.ndarray:
    """差异计算前的轻度模糊；A/B 两侧使用同一处理。"""

    return cv2.GaussianBlur(image, (3, 3), 0)


# ---------------------------------------------------------------- 解码


@dataclass
class DecodedFrame:
    """一帧的解码序号（或按帧率估计的帧号）、相对时间和原始 PyAV 帧。"""

    index: int
    time_s: float
    raw: Any

    def bgr(self) -> np.ndarray:
        return self.raw.to_ndarray(format="bgr24")


class FrameStream:
    """按时间顺序迭代视频帧；起点较早时从头解码以得到准确帧号，否则跳转并按帧率估计帧号。"""

    def __init__(self, path: Path, start_s: float, fps: float | None) -> None:
        try:
            self.container = av.open(str(path))
            if not self.container.streams.video:
                raise BackendFailure("DECODE_FAILED", "container has no video stream", stage="decode")
            self.stream = self.container.streams.video[0]
            if self.stream.time_base is None:
                raise BackendFailure("DECODE_FAILED", "video stream has no time base", stage="decode")
            self.origin = (self.stream.start_time * self.stream.time_base
                           if self.stream.start_time is not None else Fraction(0))
            self.fps = fps if fps and math.isfinite(fps) and fps > 0 else None
            self.exact_index = start_s <= SEEK_AFTER_S or self.fps is None
            if not self.exact_index:
                target = Fraction(str(max(0.0, start_s - 1.0))) + self.origin
                self.container.seek(max(0, int(target / self.stream.time_base)), stream=self.stream,
                                    backward=True, any_frame=False)
        except BackendFailure:
            self.close()
            raise
        except (OSError, ValueError, IndexError, av.FFmpegError) as exc:
            self.close()
            raise BackendFailure("DECODE_FAILED", f"video decode failed: {type(exc).__name__}", stage="decode") from exc
        self._decoder: Iterator[Any] = iter(self.container.decode(self.stream))
        self._count = -1

    @property
    def index_basis(self) -> str:
        return "decode_order" if self.exact_index else "estimated_from_time_x_fps"

    def close(self) -> None:
        container = getattr(self, "container", None)
        if container is not None:
            container.close()
            self.container = None

    def next(self) -> DecodedFrame | None:
        try:
            for raw in self._decoder:
                if raw.pts is None or raw.time_base is None:
                    continue
                self._count += 1
                time_s = float(raw.pts * raw.time_base - self.origin)
                index = self._count if self.exact_index else round(time_s * float(self.fps or 0))
                return DecodedFrame(index, time_s, raw)
        except (OSError, ValueError, av.FFmpegError) as exc:
            raise BackendFailure("DECODE_FAILED", f"video decode failed: {type(exc).__name__}", stage="decode") from exc
        return None


class NearestCursor:
    """为单调不减的目标时间返回 B 中时间最近的帧（等距时取较早帧）。"""

    def __init__(self, stream: FrameStream) -> None:
        self.stream = stream
        self.previous: DecodedFrame | None = None
        self.current: DecodedFrame | None = stream.next()

    def nearest(self, target_s: float) -> DecodedFrame | None:
        while self.current is not None and self.current.time_s < target_s - 1e-9:
            self.previous, self.current = self.current, self.stream.next()
        candidates = [frame for frame in (self.previous, self.current) if frame is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda frame: (abs(frame.time_s - target_s), frame.time_s))


def frames_at(path: Path, times: list[float]) -> dict[float, np.ndarray]:
    """取若干时间附近最近的帧，返回 {请求时间: BGR 原分辨率帧}；相邻请求顺序解码，远处才跳转。"""

    found: dict[float, np.ndarray] = {}
    if not times:
        return found
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise BackendFailure("DECODE_FAILED", "video stream has no time base", stage="decode")
            origin = stream.start_time * stream.time_base if stream.start_time is not None else Fraction(0)
            decoder: Iterator[Any] | None = None
            previous: tuple[float, Any] | None = None
            current: tuple[float, Any] | None = None
            exhausted = False
            for requested in sorted(set(times)):
                anchor = previous or current
                if decoder is None or anchor is None or requested < anchor[0] or requested - anchor[0] > 10.0:
                    container.seek(max(0, int((Fraction(str(max(0.0, requested - 0.5))) + origin) / stream.time_base)),
                                   stream=stream, backward=True, any_frame=False)
                    decoder, previous, current, exhausted = iter(container.decode(stream)), None, None, False
                while not exhausted and (current is None or current[0] < requested - 1e-9):
                    following = None
                    for raw in decoder:
                        if raw.pts is not None and raw.time_base is not None:
                            following = (float(raw.pts * raw.time_base - origin), raw)
                            break
                    if following is None:
                        exhausted = True
                        break
                    previous, current = current, following
                # 连续解码时 previous/current 是相邻两帧，二者之一必是离请求时间最近的帧。
                candidates = [value for value in (previous, current) if value is not None]
                if candidates:
                    _, raw = min(candidates, key=lambda value: (abs(value[0] - requested), value[0]))
                    found[requested] = raw.to_ndarray(format="bgr24")
    except BackendFailure:
        raise
    except (OSError, ValueError, IndexError, av.FFmpegError) as exc:
        raise BackendFailure("DECODE_FAILED", f"video decode failed: {type(exc).__name__}", stage="decode") from exc
    return found


# ---------------------------------------------------------------- 配准


@dataclass
class Registration:
    """B→A 的几何对应：mode=none/resize 时直接缩放到 A 的工作尺寸，homography 时用 transform 变换。"""

    a_size: tuple[int, int]          # A 工作尺寸
    b_size: tuple[int, int]          # B 自身工作尺寸（单应性的源坐标系）
    mode: str
    transform: np.ndarray | None
    info: dict[str, Any] = field(default_factory=dict)

    def align(self, b_native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把 B 原帧变到 A 工作坐标系，返回 (对齐图, 有效像素掩码)。"""

        width, height = self.a_size
        if self.transform is None:
            return resize(b_native, self.a_size), np.ones((height, width), dtype=bool)
        source = resize(b_native, self.b_size)
        aligned = cv2.warpPerspective(source, self.transform, self.a_size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return aligned, self.valid_mask()

    def valid_mask(self) -> np.ndarray:
        width, height = self.a_size
        if self.transform is None:
            return np.ones((height, width), dtype=bool)
        cached = self.info.get("_valid")
        if cached is None:
            ones = np.full((self.b_size[1], self.b_size[0]), 255, dtype=np.uint8)
            warped = cv2.warpPerspective(ones, self.transform, self.a_size, flags=cv2.INTER_NEAREST)
            # 边缘插值像素不可靠，向内收缩两像素。
            cached = cv2.erode(warped, _KERNEL, iterations=2) > 0
            self.info["_valid"] = cached
        return cached

    def panel_transform(self, b_native_size: tuple[int, int], panel_size: tuple[int, int]) -> np.ndarray | None:
        """把 B 原分辨率坐标直接映射到 A 面板坐标的单应性（用于对照图）。"""

        if self.transform is None:
            return None
        source = np.diag([self.b_size[0] / b_native_size[0], self.b_size[1] / b_native_size[1], 1.0])
        target = np.diag([panel_size[0] / self.a_size[0], panel_size[1] / self.a_size[1], 1.0])
        return target @ self.transform @ source

    def public(self) -> dict[str, Any]:
        return {key: value for key, value in self.info.items() if not key.startswith("_")}


def _corners(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def estimate_registration(samples: list[tuple[np.ndarray, np.ndarray]], a_native: tuple[int, int],
                          b_native: tuple[int, int], requested: bool) -> Registration:
    """用若干 A/B 帧对估计一个全局单应性；与直接缩放差别不足 1 个工作像素时不做变换。"""

    a_size, b_size = work_size(*a_native), work_size(*b_native)
    same_size = tuple(a_native) == tuple(b_native)
    plain = "none" if same_size else "resize"
    info: dict[str, Any] = {"requested": requested, "applied": False, "mode": plain,
                            "method": _tr("SIFT 特征 + 比值检验 + RANSAC 全局单应性（B→A），多帧对匹配点合并估计",
                                          "SIFT features + ratio test + RANSAC global homography (B→A), matches of several frame pairs pooled"),
                            "a_size": list(a_native), "b_size": list(b_native), "working_size_a": list(a_size),
                            "sampled_pairs": 0, "matches": 0, "inliers": 0, "inlier_ratio": None,
                            "max_corner_shift_px": None}
    plain_reason = (_tr("A/B 尺寸相同，直接逐像素比较", "A/B have the same size and are compared pixel by pixel directly") if same_size
                    else _tr("A/B 尺寸不同，B 直接缩放到 A 的画面尺寸后比较", "A/B differ in size; B is scaled directly to the A picture size before comparing"))
    if not requested:
        info["reason"] = _tr("register=false；", "register=false; ") + plain_reason
        return Registration(a_size, b_size, plain, None, info)
    sift = getattr(cv2, "SIFT_create")(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    source_points: list[tuple[float, float]] = []
    target_points: list[tuple[float, float]] = []
    for a_frame, b_frame in samples:
        a_gray = cv2.cvtColor(resize(a_frame, a_size), cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(resize(b_frame, b_size), cv2.COLOR_BGR2GRAY)
        a_points, a_desc = sift.detectAndCompute(a_gray, None)
        b_points, b_desc = sift.detectAndCompute(b_gray, None)
        info["sampled_pairs"] += 1
        if a_desc is None or b_desc is None or len(a_desc) < 2 or len(b_desc) < 2:
            continue
        for pair in matcher.knnMatch(b_desc, a_desc, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                source_points.append(b_points[pair[0].queryIdx].pt)
                target_points.append(a_points[pair[0].trainIdx].pt)
    info["matches"] = len(source_points)
    scale = np.diag([a_size[0] / b_size[0], a_size[1] / b_size[1], 1.0])
    if len(source_points) < 12:
        info["reason"] = _tr("特征匹配不足，未配准；", "too few feature matches, not registered; ") + plain_reason
        return Registration(a_size, b_size, plain, None, info)
    matrix, mask = cv2.findHomography(np.asarray(source_points, dtype=np.float32),
                                      np.asarray(target_points, dtype=np.float32), cv2.RANSAC, 2.0,
                                      maxIters=2000, confidence=0.995)
    inliers = int(mask.sum()) if mask is not None else 0
    info["inliers"] = inliers
    info["inlier_ratio"] = round(inliers / len(source_points), 4)
    usable = matrix is not None and np.isfinite(matrix).all() and inliers >= 15 and inliers >= 0.25 * len(source_points)
    if usable:
        assert matrix is not None
        projected = _project(matrix, _corners(*b_size))
        area = abs(cv2.contourArea(projected.astype(np.float32)))
        usable = (np.linalg.det(matrix[:2, :2]) > 0 and cv2.isContourConvex(projected.astype(np.float32))
                  and 0.25 <= area / (a_size[0] * a_size[1]) <= 4.0)
    if not usable:
        info["reason"] = _tr("配准内点不足或变换不合理，未配准；", "too few registration inliers or an implausible transform, not registered; ") + plain_reason
        return Registration(a_size, b_size, plain, None, info)
    assert matrix is not None
    shift = float(np.max(np.linalg.norm(_project(matrix, _corners(*b_size)) - _project(scale, _corners(*b_size)), axis=1)))
    info["max_corner_shift_px"] = round(shift, 3)
    if shift < 1.0:
        info["reason"] = _tr(f"配准结果与直接比较一致（角点最大偏移 {shift:.2f} 工作像素 < 1），未做变换；",
                             f"registration agrees with direct comparison (max corner shift {shift:.2f} working px < 1), no transform applied; ") + plain_reason
        return Registration(a_size, b_size, plain, None, info)
    normalize_a = np.diag([1 / a_size[0], 1 / a_size[1], 1.0])
    denormalize_b = np.diag([b_size[0], b_size[1], 1.0])
    normalized = normalize_a @ matrix @ denormalize_b
    normalized = normalized / normalized[2, 2]
    info.update({"applied": True, "mode": "homography",
                 "reason": _tr(f"B 相对 A 的取景或尺寸不同（角点最大偏移 {shift:.2f} 工作像素），已把 B 按单应性变换到 A 的画面后比较",
                               f"B differs from A in framing or size (max corner shift {shift:.2f} working px); B was transformed onto the A picture by the homography before comparing"),
                 "homography_b_to_a_normalized": [[round(float(x), 6) for x in row] for row in normalized]})
    return Registration(a_size, b_size, "homography", matrix, info)


# ---------------------------------------------------------------- 差异


def region_mask(size: tuple[int, int], region: tuple[float, float, float, float] | None) -> np.ndarray:
    """A 工作坐标下的比较区域掩码；region 为 None 时是整幅画面。"""

    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    if region is None:
        mask[:] = True
        return mask
    x, y, w, h = region
    left, top = math.floor(x * width), math.floor(y * height)
    right, bottom = max(left + 1, math.ceil((x + w) * width)), max(top + 1, math.ceil((y + h) * height))
    mask[top:min(height, bottom), left:min(width, right)] = True
    return mask


def difference(a_blur: np.ndarray, b_blur: np.ndarray) -> np.ndarray:
    """B/G/R 三通道绝对差的逐像素最大值。"""

    blue, green, red = cv2.split(cv2.absdiff(a_blur, b_blur))
    return cv2.max(cv2.max(blue, green), red)


def estimate_threshold(diffs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, float]:
    """由抽样帧对的差值分布确定阈值：噪声估计为各帧对第 75 百分位的中位数。"""

    levels = [float(np.percentile(diff[mask], 75)) for diff, mask in diffs if mask.any()]
    noise = float(np.median(levels)) if levels else 0.0
    value = int(min(THRESHOLD_MAX, max(THRESHOLD_BASE, round(THRESHOLD_BASE + THRESHOLD_SLOPE * noise))))
    return value, round(noise, 3)


def changed_mask(diff: np.ndarray, compare: np.ndarray, threshold: int) -> np.ndarray:
    """阈值化、限制到比较区域，再用开运算去掉孤立噪点。"""

    binary = ((diff >= threshold) & compare).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL) > 0


def measure(mask: np.ndarray, compared_pixels: int) -> tuple[float, tuple[float, float, float, float] | None]:
    """差异面积占比和归一化差异框（相对 A 画面）。"""

    count = int(mask.sum())
    if count == 0 or compared_pixels <= 0:
        return 0.0, None
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    box = (columns[0] / width, rows[0] / height, (columns[-1] + 1) / width, (rows[-1] + 1) / height)
    return count / compared_pixels, tuple(round(float(v), 4) for v in box)  # type: ignore[return-value]


def union_box(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


# ---------------------------------------------------------------- 片段与时间线


def build_segments(records: list[dict[str, Any]], min_area: float, end_frame: int, end_s: float,
                   mapping: tuple[float, float], b_duration: float | None) -> list[dict[str, Any]]:
    """把已比较且面积 ≥ min_area 的连续帧合成片段；不做平滑与跨无差异帧合并。"""

    scale, offset = mapping
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(records):
        record = records[index]
        if not (record["compared"] and record["area"] >= min_area):
            index += 1
            continue
        run = []
        while index < len(records) and records[index]["compared"] and records[index]["area"] >= min_area:
            run.append(records[index])
            index += 1
        if index < len(records):
            following = records[index]
            stop_frame, stop_s = following["frame_a"], following["time_a"]
            reason = "no_difference" if following["compared"] else "b_unavailable"
        else:
            stop_frame, stop_s, reason = end_frame, end_s, "range_end"
        stop_b = scale * stop_s + offset
        if b_duration is not None:
            stop_b = min(stop_b, b_duration)
        peak = max(run, key=lambda item: item["area"])
        first, last = run[0], run[-1]
        segments.append({
            "start_frame_a": first["frame_a"], "end_frame_a": stop_frame,
            "start_s": round(first["time_a"], 6), "end_s": round(stop_s, 6), "end_reason": reason,
            "start_b_s": round(first["time_b"], 6), "end_b_s": round(max(0.0, stop_b), 6),
            "differing_frames": len(run),
            "first_diff_frame_a": first["frame_a"], "first_diff_s": round(first["time_a"], 6),
            "first_diff_frame_b": first["frame_b"], "first_diff_b_s": round(first["time_b"], 6),
            "last_diff_frame_a": last["frame_a"], "last_diff_s": round(last["time_a"], 6),
            "last_diff_frame_b": last["frame_b"], "last_diff_b_s": round(last["time_b"], 6),
            "peak_frame_a": peak["frame_a"], "peak_s": round(peak["time_a"], 6),
            "peak_frame_b": peak["frame_b"], "peak_b_s": round(peak["time_b"], 6),
            "peak_area": round(peak["area"], 5), "mean_area": round(sum(r["area"] for r in run) / len(run), 5),
            "union_bbox": union_box([r["bbox"] for r in run if r["bbox"] is not None]),
            "_records": {"first": first, "peak": peak, "last": last},
        })
    return segments


def timeline(records: list[dict[str, Any]], stride: int, fps: float | None) -> dict[str, Any]:
    """逐帧差异面积；数值多于 TIMELINE_MAX_VALUES 时按桶取最大值（短暂差异不会被平均掉）。"""

    values = [round(r["area"], 5) if r["compared"] else None for r in records]
    bucket = max(1, math.ceil(len(values) / TIMELINE_MAX_VALUES))
    if bucket > 1:
        grouped = []
        for start in range(0, len(values), bucket):
            chunk = [v for v in values[start:start + bucket] if v is not None]
            grouped.append(max(chunk) if chunk else None)
        values = grouped
    return {"unit": "changed_area_fraction", "aggregation": "per_compared_frame" if bucket == 1 else "max_per_bucket",
            "frames_per_value": stride * bucket, "first_frame_a": records[0]["frame_a"] if records else None,
            "start_s": round(records[0]["time_a"], 6) if records else None, "fps_a": fps, "values": values,
            "note": _tr("第 k 个值对应 A 帧 first_frame_a + k × frames_per_value 起的区间；null 表示映射时间超出 B、未比较。",
                        "the k-th value covers the interval starting at A frame first_frame_a + k × frames_per_value; null means the mapped time is outside B and was not compared.")}


def unavailable_ranges(records: list[dict[str, Any]], end_s: float) -> list[list[float]]:
    """映射到 B 之外、未比较的 A 时间区间（左闭右开）。"""

    ranges: list[list[float]] = []
    for position, record in enumerate(records):
        if record["compared"]:
            continue
        stop = records[position + 1]["time_a"] if position + 1 < len(records) else end_s
        if ranges and abs(ranges[-1][1] - record["time_a"]) < 1e-9:
            ranges[-1][1] = round(stop, 6)
        else:
            ranges.append([round(record["time_a"], 6), round(stop, 6)])
    return ranges


# ---------------------------------------------------------------- 对照图


def _label(image: np.ndarray, text: str) -> None:
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (0, 0), (width + 8, height + baseline + 8), (0, 0, 0), -1)
    cv2.putText(image, text, (4, height + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_box(image: np.ndarray, box: tuple[float, float, float, float] | None, color: tuple[int, int, int],
              thickness: int) -> None:
    if box is None:
        return
    height, width = image.shape[:2]
    cv2.rectangle(image, (int(box[0] * width), int(box[1] * height)),
                  (max(0, int(math.ceil(box[2] * width)) - 1), max(0, int(math.ceil(box[3] * height)) - 1)),
                  color, thickness)


ZOOM_TRIGGER = 0.25             # 差异框宽或高小于画面该比例时加放大行
ZOOM_MIN_FRACTION = 0.12        # 放大区域至少占画面宽 / 高的比例
_CJK_FONTS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/System/Library/Fonts/PingFang.ttc", "C:/Windows/Fonts/msyh.ttc")


@lru_cache(maxsize=1)
def _cjk_font() -> Any:
    """查找可显示中文的字体；找不到时返回 None，标签退回 ASCII。"""

    from PIL import ImageFont

    candidates = list(_CJK_FONTS) + sorted(str(p) for root in ("/usr/share/fonts", "/usr/local/share/fonts")
                                           if Path(root).is_dir()
                                           for p in Path(root).rglob("*") if p.suffix.lower() in {".ttc", ".ttf", ".otf"}
                                           and any(key in p.name.lower() for key in ("cjk", "wqy", "hei", "song")))
    for candidate in candidates:
        try:
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, 16)
        except OSError:
            continue
    return None


def _text_label(image: np.ndarray, text: str, fallback: str) -> None:
    """左上角黑底白字标签；有中文字体时写中文，否则写 ASCII 版本。"""

    from ..i18n import english

    font = None if english() else _cjk_font()       # 英文档：图上的标签也写英文（ASCII 版本）
    if font is None:
        _label(image, fallback)
        return
    from PIL import Image, ImageDraw

    picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    left, top, right, bottom = draw.textbbox((4, 2), text, font=font)
    draw.rectangle((0, 0, right + 4, bottom + 4), fill=(0, 0, 0))
    draw.text((4, 2), text, font=font, fill=(255, 255, 255))
    image[:] = cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def zoom_window(bbox: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    """差异框较小时返回放大区域（归一化 x0,y0,x1,y1）：外扩差异框，至少占画面 12%，并平移到画面内。"""

    if bbox is None or (bbox[2] - bbox[0] >= ZOOM_TRIGGER and bbox[3] - bbox[1] >= ZOOM_TRIGGER):
        return None
    window = []
    for low, high in ((bbox[0], bbox[2]), (bbox[1], bbox[3])):
        size = min(1.0, max(ZOOM_MIN_FRACTION, (high - low) * 1.5))
        start = min(max(0.0, (low + high) / 2 - size / 2), 1.0 - size)
        window.append((start, start + size))
    return (window[0][0], window[1][0], window[0][1], window[1][1])


def composite(a_native: np.ndarray, b_native: np.ndarray, registration: Registration, mask: np.ndarray,
              bbox: tuple[float, float, float, float] | None, region: tuple[float, float, float, float] | None,
              labels: tuple[str, str, str]) -> tuple[np.ndarray, dict[str, Any] | None]:
    """拼出“A 帧 | B 帧 | B 对齐图上红色标差异像素、绿色画差异框”三栏对照图。

    差异框宽或高小于画面 25% 时再加第二行：差异区域放大后的“A 局部 | 对齐后的 B 局部”，差异像素用红色描边。
    返回 (图像, 放大信息或 None)。
    """

    panel_height = min(360, a_native.shape[0])
    a_panel_size = (max(1, round(a_native.shape[1] * panel_height / a_native.shape[0])), panel_height)
    b_panel_size = (max(1, round(b_native.shape[1] * panel_height / b_native.shape[0])), panel_height)
    a_panel = resize(a_native, a_panel_size).copy()
    b_panel = resize(b_native, b_panel_size).copy()
    transform = registration.panel_transform((b_native.shape[1], b_native.shape[0]), a_panel_size)
    if transform is None:
        overlay = resize(b_native, a_panel_size).copy()
    else:
        overlay = cv2.warpPerspective(b_native, transform, a_panel_size, flags=cv2.INTER_LINEAR)
    changed = cv2.resize(mask.astype(np.uint8), a_panel_size, interpolation=cv2.INTER_NEAREST) > 0
    overlay[changed] = (0.4 * overlay[changed] + 0.6 * np.array((0, 0, 255))).astype(np.uint8)
    _draw_box(overlay, bbox, (0, 255, 0), 2)
    if region is not None:
        box = (region[0], region[1], region[0] + region[2], region[1] + region[3])
        _draw_box(a_panel, box, (0, 255, 255), 1)
        _draw_box(overlay, box, (0, 255, 255), 1)
    for panel, text in zip((a_panel, b_panel, overlay), labels):
        _label(panel, text)
    gap = np.full((panel_height, 6, 3), 255, dtype=np.uint8)
    top = np.hstack([a_panel, gap, b_panel, gap, overlay])
    window = zoom_window(bbox)
    if window is None:
        return top, None
    zoom_row, info = _zoom_row(a_native, b_native, registration, mask, window, panel_height, top.shape[1])
    separator = np.full((6, top.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([top, separator, zoom_row]), info


def _zoom_row(a_native: np.ndarray, b_native: np.ndarray, registration: Registration, mask: np.ndarray,
              window: tuple[float, float, float, float], height: int, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    """放大行：A 与对齐后的 B 在同一区域的局部，按原始像素放大并用红线描出差异像素。"""

    a_height, a_width = a_native.shape[:2]
    x0, y0 = math.floor(window[0] * a_width), math.floor(window[1] * a_height)
    x1, y1 = max(x0 + 1, math.ceil(window[2] * a_width)), max(y0 + 1, math.ceil(window[3] * a_height))
    # B 对齐到 A 的原分辨率坐标后再取同一块区域。
    transform = registration.panel_transform((b_native.shape[1], b_native.shape[0]), (a_width, a_height))
    b_aligned = (resize(b_native, (a_width, a_height)) if transform is None
                 else cv2.warpPerspective(b_native, transform, (a_width, a_height), flags=cv2.INTER_LINEAR))
    crop_width, crop_height = x1 - x0, y1 - y0
    factor = min(height / crop_height, (width - 6) / 2 / crop_width)
    size = (max(1, round(crop_width * factor)), max(1, round(crop_height * factor)))
    interpolation = cv2.INTER_NEAREST if factor >= 4 else cv2.INTER_LANCZOS4
    work_height, work_width = mask.shape
    wx0, wy0 = math.floor(window[0] * work_width), math.floor(window[1] * work_height)
    wx1, wy1 = max(wx0 + 1, math.ceil(window[2] * work_width)), max(wy0 + 1, math.ceil(window[3] * work_height))
    outline = cv2.resize(mask[wy0:wy1, wx0:wx1].astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for source in (a_native, b_aligned):
        crop = cv2.resize(source[y0:y1, x0:x1], size, interpolation=interpolation)
        cv2.drawContours(crop, contours, -1, (0, 0, 255), 2)
        crops.append(crop)
    label = f"放大 x{factor:.1f}，区域 x {window[0]:.2f}–{window[2]:.2f}，y {window[1]:.2f}–{window[3]:.2f}"
    _text_label(crops[0], "A " + label, f"A zoom x{factor:.1f}, x {window[0]:.2f}-{window[2]:.2f}, y {window[1]:.2f}-{window[3]:.2f}")
    _text_label(crops[1], "B（已对齐）", "B (aligned)")
    row = np.full((height, width, 3), 40, dtype=np.uint8)
    pieces = np.hstack([crops[0], np.full((size[1], 6, 3), 255, dtype=np.uint8), crops[1]])
    row[:pieces.shape[0], :pieces.shape[1]] = pieces
    info = {"factor": round(factor, 2), "window": [round(v, 4) for v in window],
            "note": _tr("第二行为差异区域放大：左为 A 局部，右为对齐后的 B 局部，红线描出差异像素；factor 相对 A 原始像素",
                        "the second row enlarges the difference region: left the A detail, right the aligned B detail, red lines outline differing pixels; factor is relative to A's original pixels")}
    return row, info
