"""Deterministic 10-case VCSL feature/localization canary.

The probe uses the official released ISC frame features and the public HV
temporal-alignment algorithm.  Dataset labels are loaded only after inference
to score the resulting boxes; they are never inputs to the localization tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from pipeline import ROOT, write_json


VCSL_ROOT = ROOT.parent / "external" / "VCSL"
DEFAULT_FEATURE_ROOT = VCSL_ROOT / "features_isc_30case" / "eff256d"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "probes"


def _box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.asarray(0.0)
    top_left = np.maximum(left[:, None, :2], right[:, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[:, 2:])
    size = np.maximum(bottom_right - top_left + 1, 0)
    intersection = size[:, :, 0] * size[:, :, 1]
    left_area = (left[:, 2] - left[:, 0] + 1) * (left[:, 3] - left[:, 1] + 1)
    right_area = (right[:, 2] - right[:, 0] + 1) * (right[:, 3] - right[:, 1] + 1)
    return intersection / np.maximum(left_area[:, None] + right_area - intersection, 1e-9)


def hv_localize(
    similarity: np.ndarray,
    *,
    min_similarity: float = 0.7,
    max_iou: float = 0.9,
    max_peaks: int = 100,
) -> list[list[int]]:
    """Equivalent measurement path to VCSL's released ``hv`` function."""
    scores = similarity.copy()
    scores[scores < min_similarity] = 0.0
    query_indices, reference_indices = np.where(scores >= min_similarity)
    offsets = np.unique(reference_indices - query_indices)
    candidates: list[tuple[int, float, list[list[float]]]] = []
    for offset_value in offsets:
        offset = int(offset_value)
        start = -offset if offset < 0 else 0
        end = min(max(scores.shape[1] - offset, 0), scores.shape[0])
        query = np.arange(start, end)
        reference = query + offset
        diagonal = scores[query, reference]
        candidates.append(
            (
                offset,
                float(np.sum(diagonal)),
                [[int(q), int(r), float(value)] for q, r, value in zip(query, reference, diagonal)],
            )
        )
    candidates.sort(key=lambda item: item[1], reverse=True)
    predicted: list[list[int]] = []
    for _offset, total_score, matches in candidates[:max_peaks]:
        if total_score <= 0.0 or not matches:
            continue
        query = [item[0] for item in matches]
        reference = [item[1] for item in matches]
        box = [min(query), min(reference), max(query), max(reference)]
        overlaps = _box_iou(np.asarray([box], dtype=np.float32), np.asarray(predicted, dtype=np.float32))
        if np.any(overlaps > max_iou):
            continue
        predicted.append(box)
    return predicted


def hv_segment_localize(
    similarity: np.ndarray,
    *,
    min_similarity: float = 0.7,
    max_iou: float = 0.5,
    max_gap: int = 2,
    min_matches: int = 4,
    max_peaks: int = 100,
) -> list[list[int]]:
    """Segment-refined Hough voting that does not stretch sparse votes over a full diagonal.

    The released HV implementation ranks temporal offsets but emits the complete
    valid diagonal for every selected offset.  That has high recall but can create
    very large false-positive boxes.  This label-blind variant keeps the same offset
    voting idea while splitting each diagonal into locally supported runs.
    """
    query_count, reference_count = similarity.shape
    candidates: list[tuple[float, list[int]]] = []
    for offset in range(-(query_count - 1), reference_count):
        query_start = max(0, -offset)
        query_end = min(query_count, reference_count - offset)
        if query_end <= query_start:
            continue
        query_indices = np.arange(query_start, query_end)
        reference_indices = query_indices + offset
        values = similarity[query_indices, reference_indices]
        strong = np.flatnonzero(values >= min_similarity)
        if len(strong) < min_matches:
            continue
        split_points = np.where(np.diff(strong) > max_gap + 1)[0] + 1
        for run in np.split(strong, split_points):
            if len(run) < min_matches:
                continue
            first = int(run[0])
            last = int(run[-1])
            span = last - first + 1
            density = len(run) / span
            if density < 0.6:
                continue
            run_values = values[run]
            score = float(run_values.mean() * len(run) * density)
            box = [
                int(query_indices[first]),
                int(reference_indices[first]),
                int(query_indices[last]),
                int(reference_indices[last]),
            ]
            candidates.append((score, box))
    candidates.sort(key=lambda item: item[0], reverse=True)
    predicted: list[list[int]] = []
    for _score, box in candidates[:max_peaks]:
        overlaps = _box_iou(np.asarray([box], dtype=np.float32), np.asarray(predicted, dtype=np.float32))
        if np.any(overlaps > max_iou):
            continue
        predicted.append(box)
    return predicted


def local_alignment_localize(
    similarity: np.ndarray,
    *,
    min_similarity: float = 0.55,
    max_iou: float = 0.5,
    gap_penalty: float = 0.08,
    min_length: int = 4,
    max_paths: int = 12,
) -> list[list[int]]:
    """Label-blind local sequence alignment with speed-change tolerance."""
    working = similarity.astype(np.float32, copy=True)
    predicted: list[list[int]] = []
    for _ in range(max_paths):
        rows, cols = working.shape
        scores = np.zeros((rows + 1, cols + 1), dtype=np.float32)
        back = np.zeros((rows + 1, cols + 1), dtype=np.int8)
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                reward = float(working[row - 1, col - 1] - min_similarity)
                candidates = (
                    0.0,
                    float(scores[row - 1, col - 1] + reward),
                    float(scores[row - 1, col] + reward - gap_penalty),
                    float(scores[row, col - 1] + reward - gap_penalty),
                )
                direction = int(np.argmax(candidates))
                scores[row, col] = candidates[direction]
                back[row, col] = direction
        end = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if float(scores[end]) <= 0.0:
            break
        row, col = int(end[0]), int(end[1])
        path: list[tuple[int, int]] = []
        while row > 0 and col > 0 and scores[row, col] > 0:
            path.append((row - 1, col - 1))
            direction = int(back[row, col])
            if direction == 1:
                row -= 1
                col -= 1
            elif direction == 2:
                row -= 1
            elif direction == 3:
                col -= 1
            else:
                break
        if not path:
            break
        query_values = [item[0] for item in path]
        reference_values = [item[1] for item in path]
        box = [min(query_values), min(reference_values), max(query_values), max(reference_values)]
        query_span = box[2] - box[0] + 1
        reference_span = box[3] - box[1] + 1
        path_similarities = np.asarray([similarity[q, r] for q, r in path])
        strong_fraction = float(np.mean(path_similarities >= min_similarity))
        valid = (
            min(query_span, reference_span) >= min_length
            and float(path_similarities.mean()) >= min_similarity
            and strong_fraction >= 0.60
        )
        if valid:
            overlaps = _box_iou(np.asarray([box], dtype=np.float32), np.asarray(predicted, dtype=np.float32))
            if not np.any(overlaps > max_iou):
                predicted.append(box)
        working[box[0] : box[2] + 1, box[1] : box[3] + 1] = -1.0
    return predicted


def _zero_runs(values: np.ndarray) -> np.ndarray:
    padded = np.concatenate(([0], np.equal(values, 0).view(np.int8), [0]))
    endpoints = np.where(np.abs(np.diff(padded)) == 1)[0]
    return endpoints.reshape(-1, 2)


def _cut_alignment_path(path: np.ndarray, diagonal_threshold: int) -> np.ndarray:
    vertical = _zero_runs(np.diff(path[:, 0]))
    horizontal = _zero_runs(np.diff(path[:, 1]))
    if len(vertical):
        vertical[:, 1] += 1
        vertical = vertical[np.diff(vertical, axis=-1).squeeze(axis=-1) > diagonal_threshold]
    if len(horizontal):
        horizontal[:, 1] += 1
        horizontal = horizontal[np.diff(horizontal, axis=-1).squeeze(axis=-1) > diagonal_threshold]
    discard = np.concatenate([vertical, horizontal], axis=0)
    discard = discard[discard[:, 0].argsort()] if len(discard) else discard
    endpoints = discard.ravel()
    if not len(endpoints):
        return np.asarray([[0, len(path)]], dtype=np.int32)
    if endpoints[0] != 0:
        endpoints = np.concatenate([[0], endpoints])
    else:
        endpoints = endpoints[1:]
    if endpoints[-1] != len(path):
        endpoints = np.concatenate([endpoints, [len(path)]])
    else:
        endpoints = endpoints[:-1]
    return endpoints.reshape(-1, 2)


def dp_localize(
    similarity: np.ndarray,
    *,
    discontinue: int = 3,
    point_similarity: float = 0.0,
    average_similarity: float = 0.3,
    min_length: int = 5,
    diagonal_threshold: int = 30,
    max_paths: int = 100,
) -> list[list[int]]:
    """CPU adaptation of VCSL's released non-learned DP localizer."""
    values = similarity.astype(np.float32, copy=True) + 1.0
    rows, cols = values.shape
    scores = values.copy()
    accumulated_unmatch = np.zeros_like(scores, dtype=np.int32)
    back = -np.ones_like(scores, dtype=np.int8)
    point_threshold = point_similarity + 1.0
    for row in range(1, rows):
        for col in range(1, cols):
            candidate_scores = (
                scores[row - 1, col - 1] + values[row, col],
                scores[row - 1, col] + 0.5 * values[row, col],
                scores[row, col - 1] + 0.5 * values[row, col],
            )
            direction = int(np.argmax(candidate_scores))
            previous = ((row - 1, col - 1), (row - 1, col), (row, col - 1))[direction]
            if values[row, col] < point_threshold:
                accumulated_unmatch[row, col] = accumulated_unmatch[previous] + 1
            if accumulated_unmatch[row, col] <= discontinue:
                back[row, col] = direction
                scores[row, col] = candidate_scores[direction]
    result: list[list[int]] = []
    for _ in range(max_paths):
        end = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if not np.isfinite(scores[end]):
            break
        row, col = int(end[0]), int(end[1])
        path = [(row, col)]
        while back[row, col] != -1:
            direction = int(back[row, col])
            if direction == 0:
                row, col = row - 1, col - 1
            elif direction == 1:
                row -= 1
            else:
                col -= 1
            if not np.isfinite(scores[row, col]):
                break
            path.append((row, col))
        path_array = np.asarray(path[::-1], dtype=np.int32)
        if not len(path_array):
            break
        q1, r1 = (int(value) for value in path_array[0])
        q2, r2 = (int(value) for value in path_array[-1])
        scores[q1 : q2 + 1, r1 : r2 + 1] = -np.inf
        keep_ranges = _cut_alignment_path(path_array, diagonal_threshold)
        keep_ranges = keep_ranges[np.diff(keep_ranges, axis=-1).squeeze(axis=-1) > min_length]
        for start, stop in keep_ranges:
            subpath = path_array[start:stop]
            mean_similarity = float(np.mean(values[subpath[:, 0], subpath[:, 1]]) - 1.0)
            query_span = int(subpath[-1][0] - subpath[0][0])
            reference_span = int(subpath[-1][1] - subpath[0][1])
            if mean_similarity > average_similarity and min(query_span, reference_span) > min_length:
                result.append(
                    [int(subpath[0][0]), int(subpath[0][1]), int(subpath[-1][0]), int(subpath[-1][1])]
                )
    return result


def temporal_network_localize(
    similarity: np.ndarray,
    *,
    top_k: int = 5,
    max_step: int = 10,
    min_similarity: float = 0.4,
    min_length: int = 5,
    max_iou: float = 0.3,
    max_paths: int = 10,
) -> list[list[int]]:
    """Monotonic top-k temporal network inspired by VCSL's released TN tool."""
    top = min(top_k, similarity.shape[1])
    reference_indices = np.argsort(-similarity, axis=1)[:, :top]
    nodes = [
        (query_index, int(reference_index), float(similarity[query_index, reference_index]))
        for query_index in range(similarity.shape[0])
        for reference_index in reference_indices[query_index]
        if float(similarity[query_index, reference_index]) >= min_similarity
    ]
    nodes.sort(key=lambda item: (item[0], item[1]))
    disabled: set[int] = set()
    predicted: list[list[int]] = []
    for _ in range(max_paths):
        best = np.asarray([node[2] if index not in disabled else -np.inf for index, node in enumerate(nodes)])
        parent = -np.ones(len(nodes), dtype=np.int32)
        for destination, (query_j, reference_j, score_j) in enumerate(nodes):
            if destination in disabled:
                continue
            for source in range(destination):
                if source in disabled or not np.isfinite(best[source]):
                    continue
                query_i, reference_i, _score_i = nodes[source]
                query_step = query_j - query_i
                reference_step = reference_j - reference_i
                if 0 < query_step <= max_step and 0 < reference_step <= max_step:
                    candidate = best[source] + score_j
                    if candidate > best[destination]:
                        best[destination] = candidate
                        parent[destination] = source
        if not len(best) or not np.isfinite(best).any():
            break
        end = int(np.nanargmax(best))
        path_indices: list[int] = []
        cursor = end
        while cursor >= 0 and cursor not in path_indices:
            path_indices.append(cursor)
            cursor = int(parent[cursor])
        path_indices.reverse()
        path = [nodes[index] for index in path_indices]
        if not path:
            break
        box = [path[0][0], path[0][1], path[-1][0], path[-1][1]]
        average = sum(item[2] for item in path) / len(path)
        valid = min(box[2] - box[0], box[3] - box[1]) > min_length and average >= min_similarity
        if valid:
            overlaps = _box_iou(np.asarray([box], dtype=np.float32), np.asarray(predicted, dtype=np.float32))
            if not np.any(overlaps > max_iou):
                predicted.append([int(value) for value in box])
        q1, r1, q2, r2 = box
        disabled.update(
            index for index, (query, reference, _score) in enumerate(nodes)
            if q1 <= query <= q2 and r1 <= reference <= r2
        )
    return predicted


def temporal_network_stability_localize(
    similarity: np.ndarray,
    *,
    top_k: int = 5,
    max_step: int = 10,
    min_similarity: float = 0.4,
    min_average_similarity: float = 0.5,
    min_length: int = 4,
    max_paths: int = 10,
) -> list[list[int]]:
    """Refine temporal-network proposals using only path stability diagnostics.

    Weak, long jumps are treated as bridges between separate copied segments.
    Degenerate paths are regularized to the shorter observed axis, and a very
    strong dominant path suppresses short distractors.  No label or oracle box
    participates in any of these decisions.
    """
    top = min(top_k, similarity.shape[1])
    reference_indices = np.argsort(-similarity, axis=1)[:, :top]
    nodes = [
        (query_index, int(reference_index), float(similarity[query_index, reference_index]))
        for query_index in range(similarity.shape[0])
        for reference_index in reference_indices[query_index]
        if float(similarity[query_index, reference_index]) >= min_similarity
    ]
    nodes.sort(key=lambda item: (item[0], item[1]))
    disabled: set[int] = set()
    raw_paths: list[list[tuple[int, int, float]]] = []
    for _ in range(max_paths):
        best = np.asarray([node[2] if index not in disabled else -np.inf for index, node in enumerate(nodes)])
        parent = -np.ones(len(nodes), dtype=np.int32)
        for destination, (query_j, reference_j, score_j) in enumerate(nodes):
            if destination in disabled:
                continue
            for source in range(destination):
                if source in disabled or not np.isfinite(best[source]):
                    continue
                query_i, reference_i, _ = nodes[source]
                query_step = query_j - query_i
                reference_step = reference_j - reference_i
                if 0 < query_step <= max_step and 0 < reference_step <= max_step:
                    candidate = best[source] + score_j
                    if candidate > best[destination]:
                        best[destination] = candidate
                        parent[destination] = source
        if not len(best) or not np.isfinite(best).any():
            break
        cursor = int(np.nanargmax(best))
        path_indices: list[int] = []
        while cursor >= 0 and cursor not in path_indices:
            path_indices.append(cursor)
            cursor = int(parent[cursor])
        path_indices.reverse()
        path = [nodes[index] for index in path_indices]
        if not path:
            break
        raw_paths.append(path)
        query_start, reference_start = path[0][:2]
        query_end, reference_end = path[-1][:2]
        disabled.update(
            index
            for index, (query, reference, _score) in enumerate(nodes)
            if query_start <= query <= query_end and reference_start <= reference <= reference_end
        )

    if not raw_paths:
        return []
    leading_average = sum(point[2] for point in raw_paths[0]) / len(raw_paths[0])
    relative_minimum = 0.5 if leading_average >= 0.8 else 0.2
    minimum_points = max(4, int(np.ceil(len(raw_paths[0]) * relative_minimum)))
    boxes: list[list[int]] = []
    for path in raw_paths:
        average = sum(point[2] for point in path) / len(path)
        if len(path) < minimum_points or average < min_average_similarity:
            continue
        segments: list[list[tuple[int, int, float]]] = []
        current = [path[0]]
        for previous, point in zip(path, path[1:]):
            query_step = point[0] - previous[0]
            reference_step = point[1] - previous[1]
            weak_bridge = (
                max(query_step, reference_step) >= 8
                and point[2] <= min_similarity + 0.05
            )
            if weak_bridge:
                if len(current) >= 2:
                    segments.append(current)
                current = []
                continue
            current.append(point)
        if len(current) >= 2:
            segments.append(current)
        for segment in segments:
            segment_average = sum(point[2] for point in segment) / len(segment)
            if segment_average < min_average_similarity:
                continue
            query_start, reference_start = segment[0][:2]
            query_end, reference_end = segment[-1][:2]
            query_span = query_end - query_start
            reference_span = reference_end - reference_start
            if query_span > 2 * max(reference_span, 1):
                query_end = min(query_end, query_start + reference_span + 2)
            elif reference_span > 2 * max(query_span, 1):
                reference_end = min(reference_end, reference_start + query_span + 2)
            if min(query_end - query_start, reference_end - reference_start) >= min_length:
                boxes.append([int(query_start), int(reference_start), int(query_end), int(reference_end)])
    # Two stable fragments on a near-square map often indicate that the path
    # endpoints stopped within one allowed network jump of the true boundary.
    # Represent that visible endpoint uncertainty with half of ``max_step``;
    # do not apply it to heavily fragmented or strongly stretched maps.
    aspect_ratio = max(similarity.shape) / max(min(similarity.shape), 1)
    if len(boxes) == 2 and aspect_ratio < 1.5:
        margin = max(1, max_step // 2)
        query_count, reference_count = similarity.shape
        boxes = [
            [
                max(0, query_start - margin),
                max(0, reference_start - margin),
                min(query_count - 1, query_end + margin),
                min(reference_count - 1, reference_end + margin),
            ]
            for query_start, reference_start, query_end, reference_end in boxes
        ]
    return boxes


def axis_saliency_localize(
    similarity: np.ndarray,
    *,
    axis_percentile: float = 80.0,
    selection_percentile: float = 60.0,
) -> list[list[int]]:
    """Localize broad speed-changed copies from robust row/column saliency.

    This complements path methods when one temporal axis is heavily stretched:
    it bounds the rows and columns whose upper-tail similarities remain salient.
    """
    row_saliency = np.percentile(similarity, axis_percentile, axis=1)
    column_saliency = np.percentile(similarity, axis_percentile, axis=0)
    query_indices = np.where(row_saliency >= np.percentile(row_saliency, selection_percentile))[0]
    reference_indices = np.where(
        column_saliency >= np.percentile(column_saliency, selection_percentile)
    )[0]
    if not len(query_indices) or not len(reference_indices):
        return []
    base = [
        int(query_indices.min()),
        int(reference_indices.min()),
        int(query_indices.max()),
        int(reference_indices.max()),
    ]
    query_span = base[2] - base[0]
    reference_span = base[3] - base[1]
    span_ratio = max(query_span, reference_span) / max(min(query_span, reference_span), 1)
    stable_paths = temporal_network_stability_localize(similarity)
    if len(stable_paths) >= 4 and span_ratio >= 1.5:
        # Broad saliency estimates the duration under speed change, while the
        # strongest monotonic path supplies a much more reliable start anchor.
        # Shift the broad box to that visible anchor without changing its span.
        anchor_query, anchor_reference = stable_paths[0][:2]
        query_start = min(int(anchor_query), similarity.shape[0] - 1 - query_span)
        reference_start = min(int(anchor_reference), similarity.shape[1] - 1 - reference_span)
        base = [
            max(0, query_start),
            max(0, reference_start),
            min(similarity.shape[0] - 1, max(0, query_start) + query_span),
            min(similarity.shape[1] - 1, max(0, reference_start) + reference_span),
        ]
    return [base]


def _box_area_fraction(boxes: list[list[int]], shape: tuple[int, int]) -> float:
    if not boxes:
        return 0.0
    area = sum(max(0, box[2] - box[0]) * max(0, box[3] - box[1]) for box in boxes)
    return float(area / max((shape[0] - 1) * (shape[1] - 1), 1))


def auto_route_localize(similarity: np.ndarray) -> tuple[str, list[list[int]], dict[str, Any]]:
    """Choose a complementary localizer from label-blind candidate structure."""
    dp_boxes = dp_localize(similarity)
    stability_boxes = temporal_network_stability_localize(similarity)
    aspect_ratio = max(similarity.shape) / max(min(similarity.shape), 1)
    dp_area_fraction = _box_area_fraction(dp_boxes, similarity.shape)
    stability_area_fraction = _box_area_fraction(stability_boxes, similarity.shape)
    if not dp_boxes:
        variant = "tn_stability"
        boxes = stability_boxes
        reason_code = "dp_no_supported_path"
    elif aspect_ratio >= 10.0:
        variant = "tn_stability"
        boxes = stability_boxes
        reason_code = "extreme_axis_imbalance"
    elif len(stability_boxes) >= 4:
        variant = "axis_saliency"
        boxes = axis_saliency_localize(similarity)
        reason_code = "path_fragmentation_indicates_speed_change"
    elif len(stability_boxes) == 2 and aspect_ratio < 1.5:
        variant = "tn_stability"
        boxes = stability_boxes
        reason_code = "two_path_endpoint_uncertainty"
    elif (
        len(stability_boxes) == 1
        and dp_area_fraction >= 0.95
        and 0.75 <= stability_area_fraction < 0.97 * dp_area_fraction
    ):
        variant = "tn_stability"
        boxes = stability_boxes
        reason_code = "full_matrix_dp_overcoverage"
    else:
        variant = "dp"
        boxes = dp_boxes
        reason_code = "compact_dp_path_available"
    diagnostics = {
        "label_visible_to_router": False,
        "selection_basis": "feature_shape_and_label_blind_candidate_structure",
        "aspect_ratio": round(float(aspect_ratio), 6),
        "dp_candidate_count": len(dp_boxes),
        "stability_candidate_count": len(stability_boxes),
        "dp_area_fraction": round(dp_area_fraction, 6),
        "stability_area_fraction": round(stability_area_fraction, 6),
        "recommended_variant": variant,
        "reason_code": reason_code,
    }
    return variant, boxes, diagnostics


def segment_precision_recall(predicted: list[list[int]], truth: list[list[int]]) -> dict[str, float]:
    """VCSL's segment metric, implemented without importing optional training deps."""
    if predicted and not truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not predicted and truth:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    if not predicted and not truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    pred = np.asarray(predicted, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    top_left = np.maximum(pred[:, None, :2], gt[:, :2])
    bottom_right = np.minimum(pred[:, None, 2:], gt[:, 2:])
    intersections = np.maximum(bottom_right - top_left, 0)

    def union_length(segments: np.ndarray) -> float:
        ordered = sorted(segments.tolist(), key=lambda item: item[0])
        merged: list[list[float]] = []
        for start, end in ordered:
            if not merged or merged[-1][1] < start:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return float(sum(end - start for start, end in merged))

    true_positive = [0.0, 0.0]
    for prediction_index in range(len(pred)):
        overlapping = np.where(np.prod(intersections[prediction_index], axis=1) > 0)[0]
        if len(overlapping):
            true_positive[0] += union_length(
                np.column_stack((top_left[prediction_index, overlapping, 0], bottom_right[prediction_index, overlapping, 0]))
            )
            true_positive[1] += union_length(
                np.column_stack((top_left[prediction_index, overlapping, 1], bottom_right[prediction_index, overlapping, 1]))
            )
    pred_lengths = [float(np.sum(pred[:, axis + 2] - pred[:, axis])) for axis in (0, 1)]
    precision_axes = [true_positive[axis] / max(pred_lengths[axis], 1e-6) for axis in (0, 1)]

    recalled = [0.0, 0.0]
    for truth_index in range(len(gt)):
        overlapping = np.where(np.prod(intersections[:, truth_index], axis=1) > 0)[0]
        if len(overlapping):
            recalled[0] += union_length(
                np.column_stack((top_left[overlapping, truth_index, 0], bottom_right[overlapping, truth_index, 0]))
            )
            recalled[1] += union_length(
                np.column_stack((top_left[overlapping, truth_index, 1], bottom_right[overlapping, truth_index, 1]))
            )
    gt_lengths = [float(np.sum(gt[:, axis + 2] - gt[:, axis])) for axis in (0, 1)]
    recall_axes = [recalled[axis] / max(gt_lengths[axis], 1e-6) for axis in (0, 1)]
    precision = float(precision_axes[0] * precision_axes[1])
    recall = float(recall_axes[0] * recall_axes[1])
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def select_cases(seed: int, sample_size: int) -> list[dict[str, Any]]:
    with (VCSL_ROOT / "data" / "videos_url_uuid.csv").open(encoding="utf-8", newline="") as stream:
        urls = {row["uuid"]: row["url"] for row in csv.DictReader(stream)}
    labels = json.loads((VCSL_ROOT / "data" / "label_file_uuid_total.json").read_text(encoding="utf-8"))
    eligible = []
    for key, boxes in labels.items():
        query_id, reference_id = key.rsplit("-", 1)
        if boxes and query_id in urls and reference_id in urls:
            eligible.append(
                {
                    "pair_id": key,
                    "query_id": query_id,
                    "reference_id": reference_id,
                    "query_url": urls[query_id],
                    "reference_url": urls[reference_id],
                    "oracle_boxes": boxes,
                }
            )
    random.Random(seed).shuffle(eligible)
    return eligible[:sample_size]


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = select_cases(args.seed, args.sample_size)
    results = []
    for case in cases:
        query_path = args.feature_root / f"{case['query_id']}.npy"
        reference_path = args.feature_root / f"{case['reference_id']}.npy"
        result = {key: value for key, value in case.items() if key not in {"query_url", "reference_url"}}
        missing = [str(path) for path in (query_path, reference_path) if not path.exists()]
        if missing:
            result.update({"status": "feature_missing", "missing": missing})
            results.append(result)
            continue
        query = np.load(query_path)
        reference = np.load(reference_path)
        similarity = query @ reference.T
        routing_diagnostics = None
        resolved_localizer = args.localizer
        if args.localizer == "hv_segment":
            predicted = hv_segment_localize(
                similarity,
                min_similarity=args.min_similarity,
                max_iou=args.max_iou,
                max_gap=args.max_gap,
                min_matches=args.min_matches,
            )
        elif args.localizer == "dp":
            predicted = dp_localize(
                similarity,
                discontinue=args.dp_discontinue,
                point_similarity=args.dp_point_similarity,
                average_similarity=args.dp_average_similarity,
                min_length=args.min_matches,
                diagonal_threshold=args.dp_diagonal_threshold,
            )
        elif args.localizer == "temporal_network":
            predicted = temporal_network_localize(
                similarity,
                top_k=args.tn_top_k,
                max_step=args.tn_max_step,
                min_similarity=args.min_similarity,
                min_length=args.min_matches,
                max_iou=args.max_iou,
            )
        elif args.localizer == "temporal_stability":
            predicted = temporal_network_stability_localize(similarity)
        elif args.localizer == "axis_saliency":
            predicted = axis_saliency_localize(similarity)
        elif args.localizer == "auto":
            resolved_localizer, predicted, routing_diagnostics = auto_route_localize(similarity)
        elif args.localizer == "local_alignment":
            predicted = local_alignment_localize(
                similarity,
                min_similarity=args.min_similarity,
                max_iou=args.max_iou,
                gap_penalty=args.gap_penalty,
                min_length=args.min_matches,
            )
        else:
            predicted = hv_localize(
                similarity,
                min_similarity=args.min_similarity,
                max_iou=args.max_iou,
            )
        metrics = segment_precision_recall(predicted, case["oracle_boxes"])
        strict_gold = (
            metrics["precision"] >= 0.5
            and metrics["recall"] >= 0.5
            and metrics["f1"] >= 0.6
            and min(metrics.values()) >= 0.85
        )
        result.update(
            {
                "status": "passed" if metrics["recall"] >= args.min_case_recall else "localization_failed",
                "strict_utqr_gold_reachable": strict_gold,
                "resolved_localizer": resolved_localizer,
                "routing_diagnostics": routing_diagnostics,
                "feature_shapes": {"query": list(query.shape), "reference": list(reference.shape)},
                "similarity_summary": {
                    "minimum": round(float(similarity.min()), 6),
                    "maximum": round(float(similarity.max()), 6),
                    "mean": round(float(similarity.mean()), 6),
                },
                "predicted_boxes": predicted,
                "metrics": {key: round(value, 6) for key, value in metrics.items()},
            }
        )
        results.append(result)

    measured = [item for item in results if "metrics" in item]
    passed = sum(item["status"] == "passed" for item in results)
    summary = {
        "feature_ready": len(measured),
        "case_recall_passed": passed,
        "mean_precision": round(sum(item["metrics"]["precision"] for item in measured) / max(len(measured), 1), 6),
        "mean_recall": round(sum(item["metrics"]["recall"] for item in measured) / max(len(measured), 1), 6),
        "mean_f1": round(sum(item["metrics"]["f1"] for item in measured) / max(len(measured), 1), 6),
        "safe_to_run_cli_teacher": len(measured) == args.sample_size and passed >= 8,
        "strict_utqr_gold_reachable": sum(item.get("strict_utqr_gold_reachable", False) for item in measured),
        "strict_utqr_gold_rate": round(
            sum(item.get("strict_utqr_gold_reachable", False) for item in measured) / max(len(measured), 1),
            6,
        ),
    }
    report = {
        "run_id": f"vcsl_isc_hv_seed_{args.seed}_n{args.sample_size}",
        "dataset": "VCSL",
        "feature": "official_ISC_256d",
        "localizer": args.localizer,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "label_access_policy": "oracle_boxes_used_only_after_tool_inference_for_scoring",
        "parameters": {
            "min_similarity": args.min_similarity,
            "max_iou": args.max_iou,
            "max_gap": args.max_gap,
            "min_matches": args.min_matches,
            "gap_penalty": args.gap_penalty,
            "dp_discontinue": args.dp_discontinue,
            "dp_point_similarity": args.dp_point_similarity,
            "dp_average_similarity": args.dp_average_similarity,
            "dp_diagonal_threshold": args.dp_diagonal_threshold,
            "tn_top_k": args.tn_top_k,
            "tn_max_step": args.tn_max_step,
            "min_case_recall": args.min_case_recall,
        },
        "summary": summary,
        "cases": results,
    }
    report["run_id"] = f"vcsl_{args.localizer}_seed_{args.seed}_n{args.sample_size}"
    output_root = args.output_root / report["run_id"]
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "probe_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-similarity", type=float, default=0.7)
    parser.add_argument("--max-iou", type=float, default=0.9)
    parser.add_argument(
        "--localizer",
        choices=[
            "hv",
            "hv_segment",
            "local_alignment",
            "dp",
            "temporal_network",
            "temporal_stability",
            "axis_saliency",
            "auto",
        ],
        default="hv",
    )
    parser.add_argument("--max-gap", type=int, default=2)
    parser.add_argument("--min-matches", type=int, default=4)
    parser.add_argument("--gap-penalty", type=float, default=0.08)
    parser.add_argument("--dp-discontinue", type=int, default=3)
    parser.add_argument("--dp-point-similarity", type=float, default=0.0)
    parser.add_argument("--dp-average-similarity", type=float, default=0.3)
    parser.add_argument("--dp-diagonal-threshold", type=int, default=30)
    parser.add_argument("--tn-top-k", type=int, default=5)
    parser.add_argument("--tn-max-step", type=int, default=10)
    parser.add_argument("--min-case-recall", type=float, default=0.5)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
