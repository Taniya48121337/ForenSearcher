"""使用局部频谱序列匹配补充波形相关，支持有限的保音高速度变化。"""

from typing import Any

import numpy as np

from .temporal import align_sequences


def _features(samples: np.ndarray, rate: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, int(np.ceil(len(samples) / rate / step)))
    times = np.minimum((np.arange(count) + 0.5) * step, (len(samples) - 1) / rate)
    width = max(64, int(0.128 * rate))
    indices = (times * rate).astype(int)[:, None] + np.arange(width)[None, :] - width // 2
    valid = (indices >= 0) & (indices < len(samples))
    windows = samples[np.clip(indices, 0, len(samples) - 1)] * valid * np.hanning(width)
    power = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(width, 1 / rate)
    edges = np.geomspace(50, rate / 2, 49)
    bands = np.stack([power[:, (frequencies >= lo) & (frequencies < hi)].sum(axis=1)
                      for lo, hi in zip(edges, edges[1:])], axis=1)
    features = np.log1p(1000 * bands / np.maximum(bands.max(axis=1, keepdims=True), 1e-12))
    features -= features.mean(axis=1, keepdims=True)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return times, features


def spectral_matches(a: np.ndarray, b: np.ndarray, rate: int,
                     range_a: tuple[float, float], range_b: tuple[float, float],
                     top_k: int) -> tuple[list[dict[str, Any]], float]:
    """在有界采样上寻找单调局部路径；空白或零散相似点不构成匹配。"""

    step = max(0.1, len(a) / rate / 600, len(b) / rate / 600)
    times_a, features_a = _features(a, rate, step)
    times_b, features_b = _features(b, rate, step)
    similarity = np.clip(features_a @ features_b.T, -1, 1)
    paths = align_sequences(similarity, top_k=top_k, threshold=0.9,
                            min_pairs=max(4, int(np.ceil(0.8 / step))))
    matches = []
    for path in paths:
        pairs = path.pairs
        indices = sorted(set(np.linspace(0, len(pairs) - 1, min(6, len(pairs))).astype(int)))
        start_a = max(range_a[0], range_a[0] + times_a[pairs[0][0]] - step / 2)
        end_a = min(range_a[1], range_a[0] + times_a[pairs[-1][0]] + step / 2)
        start_b = max(range_b[0], range_b[0] + times_b[pairs[0][1]] - step / 2)
        end_b = min(range_b[1], range_b[0] + times_b[pairs[-1][1]] + step / 2)
        matches.append({"range_a_s":[start_a, end_a], "range_b_s":[start_b, end_b],
            "score":path.score, "anchors":[{
                "time_a_s":range_a[0] + float(times_a[pairs[i][0]]),
                "time_b_s":range_b[0] + float(times_b[pairs[i][1]]),
                "image_a_handle":None, "image_b_handle":None} for i in indices]})
    return matches, step
