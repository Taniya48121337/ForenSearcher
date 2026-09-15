"""本地缩略帧采样与 SIFT 几何匹配；中间特征不生成 Evidence。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import av
import cv2
import numpy as np

from .backends.base import BackendFailure, PreparedMedia


@dataclass(frozen=True)
class AlignmentFrame:
    requested_s: float
    time_s: float
    image: np.ndarray


def sample_alignment_frames(media: PreparedMedia, times: tuple[float, ...],
                            interval: tuple[float, float]) -> tuple[list[AlignmentFrame], list[BackendFailure]]:
    """密集采样顺序解码，稀疏采样才跳转；只保留最长边320像素的工作图。"""

    frames: list[AlignmentFrame] = []
    issues: list[BackendFailure] = []
    seen: set[float] = set()
    try:
        with av.open(str(media.path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise BackendFailure("DECODE_FAILED", "video stream has no time base", stage="decode")
            origin = stream.start_time * stream.time_base if stream.start_time is not None else Fraction(0)
            decoder: Any = None
            current: tuple[float, av.VideoFrame] | None = None
            previous: tuple[float, av.VideoFrame] | None = None
            last_request = -float("inf")
            for requested in times:
                try:
                    if decoder is None or requested - last_request > 2:
                        container.seek(max(0, int((Fraction(str(requested)) + origin) / stream.time_base)),
                                       stream=stream, backward=True, any_frame=False)
                        decoder = iter(container.decode(stream))
                        current = previous = None
                    last_request = requested
                    while current is None or current[0] < requested:
                        if current is not None:
                            previous = current
                        current = None
                        for raw in decoder:
                            if raw.pts is None or raw.time_base is None:
                                continue
                            actual = float(raw.pts * raw.time_base - origin)
                            if actual < interval[0]:
                                continue
                            if actual >= interval[1]:
                                break
                            current = (actual, raw)
                            break
                        if current is None:
                            break
                    candidates = [value for value in (previous, current) if value is not None]
                    if not candidates:
                        raise BackendFailure("DECODE_FAILED", "no frame near requested time", stage="decode")
                    stamp, raw = min(candidates, key=lambda value: abs(value[0] - requested))
                    if round(stamp, 9) in seen:
                        continue
                    seen.add(round(stamp, 9))
                    width, height = raw.width, raw.height
                    scale = min(1.0, 320 / max(width, height))
                    image = raw.reformat(width=max(1, round(width * scale)), height=max(1, round(height * scale)),
                                         format="bgr24").to_ndarray()
                    frames.append(AlignmentFrame(requested, stamp, image))
                except (OSError, ValueError, av.FFmpegError, BackendFailure) as exc:
                    issues.append(BackendFailure("DECODE_FAILED", f"alignment sample unavailable: {type(exc).__name__}",
                        stage="decode", retryable=True, locator={"requested_time_s": requested}))
                    decoder = None
    except (OSError, ValueError, IndexError, av.FFmpegError) as exc:
        raise BackendFailure("DECODE_FAILED", f"alignment decode failed: {type(exc).__name__}", stage="decode") from exc
    return frames, issues


def geometric_similarities(left: list[AlignmentFrame], right: list[AlignmentFrame]) -> np.ndarray:
    """FLANN检索少量候选，再用SIFT比值检验与RANSAC验证局部几何对应。"""

    matrix = np.zeros((len(left), len(right)), dtype=np.float32)
    if not matrix.size:
        return matrix
    sift = getattr(cv2, "SIFT_create")(nfeatures=300)
    features = []
    for sequence in (left, right):
        group = []
        previous_image = None
        for frame in sequence:
            if previous_image is not None and np.array_equal(previous_image, frame.image):
                group.append(group[-1])
                continue
            points, descriptors = sift.detectAndCompute(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), None)
            group.append((np.array([point.pt for point in points], dtype=np.float32), descriptors))
            previous_image = frame.image
        features.append(group)
    train_side = int(len(right) > len(left))
    train, query = features[train_side], features[1 - train_side]
    indices = [i for i, (_, desc) in enumerate(train) if desc is not None and len(desc) >= 2]
    if not indices:
        return matrix
    index = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
    index.add([train[i][1] for i in indices])
    index.train()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for j, (points_q, desc_q) in enumerate(query):
        if desc_q is None or len(desc_q) < 8:
            continue
        votes = np.zeros(len(train), dtype=np.float32)
        for neighbors in index.knnMatch(desc_q, k=min(4, sum(len(train[i][1]) for i in indices))):
            visited = set()
            for neighbor in neighbors:
                if neighbor.imgIdx not in visited:
                    votes[indices[neighbor.imgIdx]] += 1 / (1 + neighbor.distance)
                    visited.add(neighbor.imgIdx)
        for i in np.argsort(votes)[-8:]:
            points_t, desc_t = train[i]
            if desc_t is None or len(desc_t) < 2 or votes[i] <= 0:
                continue
            matches = [pair[0] for pair in matcher.knnMatch(desc_q, desc_t, k=2)
                       if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
            if len(matches) < 12:
                continue
            source = points_q[[m.queryIdx for m in matches]]
            target = points_t[[m.trainIdx for m in matches]]
            transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 3, maxIters=500)
            inliers = int(mask.sum()) if mask is not None else 0
            if transform is None or not np.isfinite(transform).all() or inliers < 12 or inliers < .4 * len(matches):
                continue
            # 单个标志或少量共线点不足以代表整个画面对应。
            assert mask is not None
            keep = mask.ravel().astype(bool)
            sequences = (left, right)
            if any(cv2.contourArea(cv2.convexHull(points[keep])) < .02 * frame.image.shape[0] * frame.image.shape[1]
                   for points, frame in ((source, sequences[1 - train_side][j]), (target, sequences[train_side][i]))):
                continue
            support = inliers / max(20, min(len(desc_q), len(desc_t)))
            score = .9 + .1 * min(1.0, support)
            if train_side == 0:
                matrix[i, j] = score
            else:
                matrix[j, i] = score
    return matrix
