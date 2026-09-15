"""从帧相似度矩阵提取单调连续的局部时间对应路径。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalMatch:
    pairs: tuple[tuple[int, int], ...]
    score: float


def _pair_gains(similarities: np.ndarray, threshold: float, relative_similarity: bool) -> np.ndarray:
    gains = similarities - threshold + .02
    if relative_similarity and similarities.size:
        best = np.maximum(similarities.max(axis=1)[:, None], similarities.max(axis=0)[None, :])
        # 相似画面存在更好对应时降低收益；保留短暂退化帧穿过连续路径的可能。
        relative_distance = (1 - similarities) / np.maximum(1e-5, 2 * (1 - best))
        gains = np.minimum(gains, (1 - threshold + .02) * (1 - relative_distance))
    return gains


def path_score(similarities: np.ndarray, pairs: tuple[tuple[int, int], ...], *, threshold: float = .86,
               relative_similarity: bool = False) -> float:
    """按同一次采样规模归一化路径收益，保留动态规划的连续匹配量及步长惩罚。"""

    gains = _pair_gains(similarities, threshold, relative_similarity)
    gain = sum(float(gains[i, j]) for i, j in pairs)
    skips = sum(b[0] - a[0] + b[1] - a[1] - 2 for a, b in zip(pairs, pairs[1:]))
    maximum = min(similarities.shape) * (1 - threshold + .02)
    return float(np.clip((gain - .04 * skips) / maximum, 0, 1))


def align_sequences(similarities: np.ndarray, *, top_k: int, threshold: float = 0.86,
                    min_pairs: int = 4, relative_similarity: bool = False) -> tuple[TemporalMatch, ...]:
    """局部动态规划；允许 0.5–2 倍采样步长，单帧命中不能构成片段。"""

    if similarities.ndim != 2 or not np.isfinite(similarities).all():
        raise ValueError("similarities must be a finite matrix")
    rows, columns = similarities.shape
    available = similarities >= threshold
    gains = _pair_gains(similarities, threshold, relative_similarity)
    results = []
    for _ in range(top_k):
        scores = np.zeros((rows + 1, columns + 1), dtype=np.float64)
        previous = np.zeros((rows, columns), dtype=np.uint8)
        lengths = np.zeros((rows + 1, columns + 1), dtype=np.int32)
        for i in range(rows):
            for j in range(columns):
                if not available[i, j]:
                    continue
                # 等分画面优先保持稳定时间偏移，避免把静态片段任意伸缩。
                choices = (scores[i, j], scores[i, j - 1] - 0.04 if j else -1,
                           scores[i - 1, j] - 0.04 if i else -1)
                step = int(np.argmax(choices))
                gain = float(gains[i, j])
                scores[i + 1, j + 1] = max(0.0, gain + max(0.0, choices[step]))
                if scores[i + 1, j + 1] <= 0:
                    continue
                previous[i, j] = step + 1 if choices[step] > 0 else 0
                di, dj = ((1, 1), (1, 2), (2, 1))[step]
                lengths[i + 1, j + 1] = 1 + (lengths[i + 1 - di, j + 1 - dj] if choices[step] > 0 else 0)
        eligible = np.where(lengths >= min_pairs, scores, 0)
        end_i, end_j = np.unravel_index(np.argmax(eligible), scores.shape)
        if eligible[end_i, end_j] <= 0:
            break
        if scores[end_i, end_j] <= 0:
            break
        i, j = int(end_i) - 1, int(end_j) - 1
        path = []
        while i >= 0 and j >= 0 and available[i, j]:
            path.append((i, j))
            step = previous[i, j]
            if step == 0:
                break
            di, dj = ((1, 1), (1, 2), (2, 1))[step - 1]
            i, j = i - di, j - dj
        path.reverse()
        if len(path) < min_pairs:
            break
        pairs = tuple(path)
        results.append(TemporalMatch(pairs, path_score(similarities, pairs, threshold=threshold,
                                                      relative_similarity=relative_similarity)))
        # 只抑制这一对区间；同一内容在另一侧不同时间的重复出现仍可形成候选。
        available[path[0][0]:path[-1][0] + 1, path[0][1]:path[-1][1] + 1] = False
    return tuple(results)


def prune_dominated_paths(matches: tuple[TemporalMatch, ...], similarities: np.ndarray) -> tuple[TemporalMatch, ...]:
    """按输入分数降序去除完全被更强结构对应覆盖的次级路径，保留重复与新增对应。"""

    best_a = np.full(similarities.shape[0], -np.inf)
    best_b = np.full(similarities.shape[1], -np.inf)
    retained = []
    for match in matches:
        if all(1 - similarities[i, j] > max(1e-5, 2 * (1 - max(best_a[i], best_b[j])))
               for i, j in match.pairs):
            continue
        retained.append(match)
        for i, j in match.pairs:
            best_a[i] = max(best_a[i], similarities[i, j])
            best_b[j] = max(best_b[j], similarities[i, j])
    return tuple(retained)
