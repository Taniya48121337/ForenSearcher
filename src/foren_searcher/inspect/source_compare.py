"""compare_media: compare a video A with a candidate original B and report what corresponds and what differs.

Moved from detect/compare.py (DETECT compare_source, 10-10..10-11) and extended on 10-12 so that one INSPECT
operation covers what align_media + diff_media + compare_source did separately (see the project doc
"DETECT并入SEARCH与INSPECT_合并方案_20261012" §7.4):

1. near-duplicate piecewise alignment: 64x36 grey thumbnails (normalised cross-correlation); for every half second of A
   the best moments of B (top candidates, so static pictures do not jump); matches are chained into *correspondence
   runs* whose B time follows a straight line in A time;
2. transformations: a horizontally flipped copy of every B thumbnail is matched too (mirroring); when fewer than 30% of
   A's samples are explained this way the caller falls back to the SIFT/RANSAC alignment of align_media (crops,
   borders, picture-in-picture) and passes its matches in as runs;
3. many-to-many: every A run is reported with its own B range, so repeated or re-used footage shows up as several runs;
4. footage of B missing from A (cuts) and 5. re-ordering: from the order of the runs in B;
6. speed and 7. reversal: each run's slope (B seconds per A second) is fitted, so slowed / sped-up / reversed / frozen
   runs are reported as such;
8. frame-exact boundaries: the caller runs media_diff inside a run when asked (range_a_s) or for short equal pairs;
9. regions and faces at the same moment (ECC affine registration, minimum over neighbouring B frames, MTCNN+FaceNet);
10. audio: windowed log-spectrogram similarity at the matched times (audio_compare());
11. image pairs are handled by the caller (diff_media's image path).

The comparer only reports differences between two files. Its text states facts (what spans, regions, faces and
sounds differ, how long a shot lasts); it does not decide which file is the original or how a difference was made.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np


class CompareUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class CompareConfig:
    thumb_step_a: float = 0.5
    thumb_step_b: float = 0.1           # B up to medium_b_s: fast pans decorrelate 64x36 thumbnails within ~2 frames
    medium_b_s: float = 300.0
    thumb_step_b_medium: float = 0.25
    long_b_s: float = 1200.0            # longer B: thumbnails every 0.5 s
    match_ncc: float = 0.80
    top_candidates: int = 8             # B moments kept per A sample (static pictures have many near-equal matches)
    tie_margin: float = 0.05
    altered_ncc: float = 0.45           # a gap that lines up in time with B and is this similar is 'altered', not 'replaced'
    min_run_points: int = 3
    min_segment_s: float = 1.5
    detail_step: float = 1.0
    max_detail: int = 300
    window_s: float = 0.25
    work_width: int = 640
    min_region_area: float = 0.0015
    min_region_area_persistent: float = 0.0008   # 10-11: a small box that stays put over >= 5 samples is kept
    max_region_area: float = 0.45
    face_step: float = 2.0
    max_face_frames: int = 90
    face_min_px: int = 40
    face_same_person: float = 0.6       # FaceNet cosine: same person above this
    face_swapped: float = 0.45          # median similarity below this -> identity differs
    face_swapped_weak: float = 0.68     # 10-11: below this over >= 4 comparisons -> likely differs (partial swap)
    face_weak_min_compared: int = 4
    max_images: int = 8
    time_budget_s: float = 900.0
    generator_shot_s: tuple[float, float] = (9.8, 10.2)


@dataclass
class _Sample:
    t: float
    idx: int
    tb_pred: float
    jpeg: bytes
    face: bool
    flip: bool = False
    window: list = field(default_factory=list)   # (ncc, b_idx, frame640, full_frame_or_None)


@dataclass
class Run:
    """A stretch of A whose B time follows a straight line: tb = slope * ta + offset."""

    ta: list[float]
    tb: list[float]
    flip: bool = False
    engine: str = "near_duplicate"
    score: float | None = None
    altered: bool = False               # lines up in time with B but the picture differs (thumbnail NCC below match level)
    similarity: float | None = None
    kind_override: str | None = None    # 'slowed' for strong slow motion that the slope alone would call frozen

    @property
    def a_start(self) -> float:
        return self.ta[0]

    @property
    def a_end(self) -> float:
        return self.ta[-1]

    def fit(self) -> tuple[float, float]:
        if len(self.ta) < 2 or self.ta[-1] - self.ta[0] < 1e-6:
            return 1.0, self.tb[0] - self.ta[0]
        slope, offset = np.polyfit(np.asarray(self.ta), np.asarray(self.tb), 1)
        return float(slope), float(offset)

    def b_at(self, t: float) -> float:
        slope, offset = self.fit()
        return slope * t + offset

    @property
    def kind(self) -> str:
        if self.kind_override:
            return self.kind_override
        slope, _ = self.fit()
        if abs(slope) < 0.15:
            return "frozen"
        if slope < 0:
            return "reversed"
        if slope < 0.85:
            return "slowed"
        if slope > 1.18:
            return "sped_up"
        return "normal"


# ----------------------------------------------------------------------------- small helpers

def _thumb(frame_bgr) -> np.ndarray:
    import cv2

    h, w = frame_bgr.shape[:2]
    g = cv2.cvtColor(frame_bgr[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)], cv2.COLOR_BGR2GRAY)
    v = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-3 else v * 0


def _small_gray(frame_bgr) -> np.ndarray:
    import cv2

    v = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-3 else v * 0


def _resize_w(frame, width):
    import cv2

    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, max(2, round(h * width / w))), interpolation=cv2.INTER_AREA)


def jpeg(frame_bgr, quality=92) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CompareUnavailable("ENCODE_FAILED", "JPEG encoding failed")
    return buf.tobytes()


def unjpeg(data: bytes):
    import cv2

    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _plausible_homography(H: np.ndarray) -> bool:
    """A crop / zoom / inset / letterbox is close to a similarity: both scales positive and comparable, little shear or
    perspective. Wild homographies from a handful of chance matches are rejected before the costly re-alignment."""

    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-9:
        return False
    H = H / H[2, 2]
    if abs(H[2, 0]) > 2e-3 or abs(H[2, 1]) > 2e-3:
        return False
    sx, sy = float(np.hypot(H[0, 0], H[1, 0])), float(np.hypot(H[0, 1], H[1, 1]))
    det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
    return det > 0 and 0.2 <= sx <= 6 and 0.2 <= sy <= 6 and 0.5 <= sx / sy <= 2.0


def video_info(path: Path) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise CompareUnavailable("UNSUPPORTED", f"cannot open video {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or not math.isfinite(fps) or fps <= 0 or n <= 0:
        raise CompareUnavailable("UNSUPPORTED", f"video {path.name} has no frame rate / frame count")
    return float(fps), n, w, h


def _position(obs) -> str | None:
    """Rough place of a face track in the frame from its boxes (frame size from the stored A frame)."""

    try:
        a = unjpeg(obs[0]["a_jpeg"])
        H, W = a.shape[:2]
    except Exception:  # noqa: BLE001
        return None
    cx = float(np.median([(o["box"][0] + o["box"][2]) / 2 for o in obs])) / W
    cy = float(np.median([(o["box"][1] + o["box"][3]) / 2 for o in obs])) / H
    size = float(np.median([o["box"][2] - o["box"][0] for o in obs])) / W
    side = "left" if cx < 0.36 else "right" if cx > 0.64 else "centre"
    height = "upper" if cy < 0.33 else "lower" if cy > 0.67 else "middle"
    depth = "close to the camera" if size > 0.10 else "farther from the camera" if size < 0.05 else "mid-distance"
    return f"{side} of the frame, {height} part, face {100 * size:.0f}% of the frame width ({depth})"


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def grab(path: Path, indices: list[int]) -> dict:
    import cv2

    want, out = set(indices), {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(path))
    idx, last = -1, max(want)
    try:
        while idx < last:
            if not cap.grab():
                break
            idx += 1
            if idx in want:
                ok, bgr = cap.retrieve()
                if ok:
                    out[idx] = bgr
    finally:
        cap.release()
    return out


def label(img, text):
    import cv2

    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], 12 + 9 * len(text)), 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _pad_w(img, width):
    if img.shape[1] >= width:
        return img[:, :width]
    pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=img.dtype)
    return np.hstack([img, pad])


def _face_pair(a, box_a, b, box_b, text):
    import cv2

    def crop(img, box):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        H, W = img.shape[:2]
        c = img[max(0, int(y0 - 0.6 * h)):min(H, int(y1 + 0.9 * h)), max(0, int(x0 - 0.6 * w)):min(W, int(x1 + 0.6 * w))]
        return cv2.resize(c, (200, 240)) if c.size else np.zeros((240, 200, 3), np.uint8)

    pair = np.hstack([crop(a, box_a), crop(b, box_b)])
    return label(pair, text)


# ----------------------------------------------------------------------------- faces

class FaceIdentity:
    """MTCNN + FaceNet(VGGFace2). Loaded lazily on first use."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._mtcnn = self._net = None

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("facenet_pytorch") is not None

    def _load(self):
        if self._net is None:
            from facenet_pytorch import MTCNN, InceptionResnetV1

            self._mtcnn = MTCNN(keep_all=True, device=self.device, post_process=False)
            self._net = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()

    def detect(self, frame_bgr, min_px: int) -> list[tuple[float, float, float, float]]:
        import cv2
        from PIL import Image

        self._load()
        boxes, probs = self._mtcnn.detect(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if boxes is None:
            return []
        return [tuple(float(v) for v in b) for b, p in zip(boxes, probs)
                if p is not None and p >= 0.9 and (b[2] - b[0]) >= min_px]

    def embed(self, frame_bgr, boxes) -> np.ndarray:
        import cv2
        import torch

        self._load()
        if not boxes:
            return np.zeros((0, 512), np.float32)
        H, W = frame_bgr.shape[:2]
        crops = []
        for x0, y0, x1, y1 in boxes:
            m = 0.15 * (x1 - x0)
            c = frame_bgr[max(0, int(y0 - m)):min(H, int(y1 + m)), max(0, int(x0 - m)):min(W, int(x1 + m))]
            crops.append(cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (160, 160), interpolation=cv2.INTER_AREA))
        with torch.inference_mode():
            x = (torch.tensor(np.stack(crops), device=self.device).permute(0, 3, 1, 2).float() - 127.5) / 128
            e = self._net(x)
            e = e / e.norm(dim=1, keepdim=True)
        return e.cpu().numpy()

    def identity(self) -> dict:
        return {"detector": "MTCNN (facenet-pytorch)", "embedding": "InceptionResnetV1 VGGFace2", "device": self.device}


# ----------------------------------------------------------------------------- alignment: runs

def build_runs(ta: np.ndarray, cand_b: list[list[tuple[float, float, bool]]], cfg: CompareConfig) -> list[Run]:
    """Chain per-sample B candidates (tb, score, flipped) into straight-line runs.

    A sample may continue the current run if one of its candidates lies on the run's line (residual tolerance grows
    with the gap); otherwise the run closes and the sample's best candidate starts a new run. Up to two unmatched
    samples may be skipped inside a run (fast motion between B samples)."""

    runs: list[Run] = []
    current: Run | None = None
    misses = 0
    for i, t in enumerate(ta):
        cands = cand_b[i]
        if not cands:
            misses += 1
            if current is not None and misses > 2:
                runs.append(current)
                current = None
            continue
        chosen = None
        if current is not None:
            gap = t - current.ta[-1]
            if len(current.ta) >= 2:
                slope, offset = current.fit() if len(current.ta) < 8 else Run(current.ta[-8:], current.tb[-8:]).fit()
                pred = slope * t + offset
                tol = 0.3 + 0.12 * gap * max(1.0, abs(slope))
                best = min(cands, key=lambda c: abs(c[0] - pred))
                if abs(best[0] - pred) <= tol and best[2] == current.flip:
                    chosen = best
            else:
                last = current.tb[-1]
                ok = [c for c in cands if c[2] == current.flip and abs(c[0] - last) <= 4.0 * gap + 0.35]
                if ok:
                    chosen = min(ok, key=lambda c: abs(abs(c[0] - last) - gap))   # prefer normal speed
        if chosen is not None:
            current.ta.append(float(t))
            current.tb.append(float(chosen[0]))
            misses = 0
            continue
        if current is not None:
            runs.append(current)
        best = cands[0]
        current = Run([float(t)], [float(best[0])], flip=bool(best[2]))
        misses = 0
    if current is not None:
        runs.append(current)
    return [r for r in runs if len(r.ta) >= cfg.min_run_points]


def run_summary(run: Run, step_a: float) -> dict:
    slope, offset = run.fit()
    b0, b1 = run.b_at(run.a_start), run.b_at(run.a_end)
    return {"a_range_s": [_r(max(0.0, run.a_start - step_a / 2), 1), _r(run.a_end + step_a / 2, 1)],
            "b_range_s": [_r(max(0.0, min(b0, b1)), 1), _r(max(0.0, b0, b1), 1)],
            "kind": run.kind, "mirrored": run.flip, "engine": run.engine, "picture_differs": run.altered,
            "thumbnail_similarity": _r(run.similarity, 2) if run.similarity is not None else None,
            "mapping": {"formula": "time_b_s = scale * time_a_s + offset_s", "scale": _r(slope, 3), "offset_s": _r(offset, 2)},
            "samples": len(run.ta)}


def structure_from_runs(runs: list[Run], ta: np.ndarray, dur_a: float, dur_b: float, cfg: CompareConfig) -> dict:
    """A-only segments, B gaps between consecutive runs (cuts), re-orderings and repeats."""

    step = cfg.thumb_step_a
    matched = np.zeros(len(ta), dtype=bool)
    owner = np.full(len(ta), -1)
    for k, run in enumerate(runs):
        lo, hi = run.a_start - step / 2 - 1e-6, run.a_end + step / 2 + 1e-6
        sel = (ta >= lo) & (ta <= hi)
        matched |= sel
        owner[sel & (owner < 0)] = k
    # A-only spans
    segments = []
    i, n = 0, len(ta)
    while i < n:
        if matched[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not matched[j + 1]:
            j += 1
        start = max(0.0, ta[i] - step / 2)
        end = min(dur_a, ta[j] + step / 2)
        prev_run = next((runs[owner[k]] for k in range(i - 1, -1, -1) if owner[k] >= 0), None)
        next_run = next((runs[owner[k]] for k in range(j + 1, n) if owner[k] >= 0), None)
        before = prev_run.b_at(prev_run.a_end) if prev_run else None
        after = next_run.b_at(next_run.a_start) if next_run else None
        if end - start >= cfg.min_segment_s:
            dur = end - start
            context = None
            if before is not None and after is not None:
                gap = after - before
                if abs(gap) <= 1.0:
                    context = "inserted"          # B continues around it with no gap
                elif abs(gap - dur) <= 1.0:
                    context = "replaced"          # lines up with a B span of the same length that looks different
                elif gap > 1.0:
                    context = "replaces_longer_b_span"
                else:
                    context = "b_jumps_back"
            segments.append({"a_range_s": [_r(start, 1), _r(end, 1)], "duration_s": _r(dur, 1),
                             "b_before_s": _r(before, 1), "b_after_s": _r(after, 1),
                             "b_gap_s": _r(after - before, 1) if before is not None and after is not None else None,
                             "context": context,
                             "single_shot_generator_length": bool(cfg.generator_shot_s[0] - 1.3 <= dur <= cfg.generator_shot_s[1] + 0.4)})
        i = j + 1
    # cuts / reorder / repeats from consecutive forward runs
    moving = [r for r in runs if r.kind in {"normal", "slowed", "sped_up"}]
    moving.sort(key=lambda r: r.a_start)
    cuts, reorders = [], []
    for r1, r2 in zip(moving, moving[1:]):
        e1, s2 = r1.b_at(r1.a_end), r2.b_at(r2.a_start)
        a_gap = r2.a_start - r1.a_end
        b_jump = s2 - e1 - a_gap * max(0.0, r1.fit()[0])
        if b_jump > 1.0:
            cuts.append({"a_at_s": _r((r1.a_end + r2.a_start) / 2, 1), "b_missing_s": [_r(e1, 1), _r(s2, 1)],
                         "duration_s": _r(s2 - e1, 1)})
        elif s2 < e1 - 1.0:
            reorders.append({"a_at_s": _r((r1.a_end + r2.a_start) / 2, 1), "b_from_s": _r(e1, 1), "b_to_s": _r(s2, 1)})
    repeats = []
    for x in range(len(runs)):
        for y in range(x + 1, len(runs)):
            bx = sorted((runs[x].b_at(runs[x].a_start), runs[x].b_at(runs[x].a_end)))
            by = sorted((runs[y].b_at(runs[y].a_start), runs[y].b_at(runs[y].a_end)))
            ov = min(bx[1], by[1]) - max(bx[0], by[0])
            if ov >= 1.0:
                repeats.append({"b_range_s": [_r(max(bx[0], by[0]), 1), _r(min(bx[1], by[1]), 1)],
                                "a_ranges_s": [[_r(runs[x].a_start, 1), _r(runs[x].a_end, 1)],
                                               [_r(runs[y].a_start, 1), _r(runs[y].a_end, 1)]]})
    covered = [sorted((r.b_at(r.a_start), r.b_at(r.a_end))) for r in runs]
    extent = [min(c[0] for c in covered), max(c[1] for c in covered)] if covered else None
    return {"matched": matched, "segments_not_in_b": segments, "b_cuts": cuts, "reorders": reorders,
            "repeats": repeats[:10], "b_extent_s": [_r(extent[0], 1), _r(extent[1], 1)] if extent else None}


# ----------------------------------------------------------------------------- comparer

class SourceComparer:
    def __init__(self, faces: FaceIdentity | None = None, config: CompareConfig | None = None):
        self.faces = faces
        self.cfg = config or CompareConfig()
        self._a_cache = None

    def identity(self) -> dict:
        return {"name": "compare_media", "version": "2026-10-12",
                "alignment": "64x36 grey NCC (normal + mirrored B) chained into straight-line runs; SIFT fallback",
                "regions": "ECC affine + min over neighbouring frames",
                "faces": self.faces.identity() if self.faces else None}

    # .................................................................. decoding passes

    def _pass_a(self, path, fps, n, deadline, cancel):
        import cv2

        cfg = self.cfg
        dur = n / fps
        thumb_idx = {min(n - 1, int(round(t * fps))): t for t in np.arange(0, max(dur - 0.05, 0.01), cfg.thumb_step_a)}
        step = max(cfg.detail_step, dur / cfg.max_detail)
        detail_idx = {min(n - 1, int(round(t * fps))): t for t in np.arange(min(0.25, dur / 2), max(dur - 0.05, 0.01), step)}
        face_every = max(1, int(round(max(cfg.face_step, dur / cfg.max_face_frames) / step)))
        thumbs, details = {}, {}
        cap = cv2.VideoCapture(str(path))
        last = max(list(thumb_idx) + list(detail_idx))
        idx = -1
        try:
            while idx < last:
                if not cap.grab():
                    break
                idx += 1
                if idx not in thumb_idx and idx not in detail_idx:
                    continue
                if idx % 64 == 0:
                    self._check(deadline, cancel)
                ok, bgr = cap.retrieve()
                if not ok:
                    continue
                if idx in thumb_idx:
                    thumbs[idx] = _thumb(bgr)
                if idx in detail_idx:
                    k = len(details)
                    details[idx] = (bgr if (self.faces is not None and k % face_every == 0) else None,
                                    _resize_w(bgr, cfg.work_width), _small_gray(bgr))
        finally:
            cap.release()
        return thumbs, details

    def _thumbs_b(self, path, fps, n, deadline, cancel, warp=None):
        import cv2

        cfg = self.cfg
        dur = n / fps
        step = (cfg.thumb_step_b if dur <= cfg.medium_b_s else cfg.thumb_step_b_medium if dur <= cfg.long_b_s
                else cfg.thumb_step_a)
        want = {min(n - 1, int(round(t * fps))) for t in np.arange(0, max(dur - 0.05, 0.01), step)}
        thumbs, flipped = {}, {}
        cap = cv2.VideoCapture(str(path))
        last, idx = max(want), -1
        try:
            while idx < last:
                if not cap.grab():
                    break
                idx += 1
                if idx not in want:
                    continue
                if idx % 64 == 0:
                    self._check(deadline, cancel)
                ok, bgr = cap.retrieve()
                if ok:
                    if warp is not None:
                        bgr, _ = warp.align(bgr)
                    thumbs[idx] = _thumb(bgr)
                    flipped[idx] = _thumb(cv2.flip(bgr, 1))
        finally:
            cap.release()
        return thumbs, flipped

    @staticmethod
    def _check(deadline, cancel):
        if cancel is not None and cancel.is_set():
            raise CompareUnavailable("CANCELLED", "comparison was cancelled")
        if time.monotonic() > deadline:
            raise CompareUnavailable("TIME_LIMIT", "compare_media time budget exceeded")

    # .................................................................. main entry

    def align(self, path_a: Path, path_b: Path, cancel: Event | None = None) -> dict:
        """Near-duplicate alignment only (what SEARCH's source matching uses)."""

        return self._align(path_a, path_b, cancel, keep_details=False)

    def match(self, path_a: Path, path_b: Path, cancel: Event | None = None, *, registration_below: float = 0.3) -> dict:
        """SEARCH content matching (10-12, was the first half of DETECT find_source): how much of A is in B.

        Near-duplicate alignment first; when under registration_below of A matches, B is also tried warped onto A's
        framing (crop / zoom / letterbox) and the better alignment is kept. Returns plain numbers only."""

        aligned = self.align(path_a, path_b, cancel)
        fraction, engine = self.match_fraction(aligned), "near_duplicate"
        if fraction < registration_below:
            try:
                found = self.find_registration(path_a, path_b, cancel)[:2]
            except CompareUnavailable:
                found = []
            for reg in found:
                warped = self._align(path_a, path_b, cancel, keep_details=False, warp=reg)
                if self.match_fraction(warped) >= fraction + 0.1:
                    aligned, fraction, engine = warped, self.match_fraction(warped), "registered"
        step = self.cfg.thumb_step_a
        runs = sorted(aligned["runs"], key=lambda r: r.a_start)
        return {"fraction": round(fraction, 3), "engine": engine,
                "a_duration_s": _r(aligned["dur_a"], 1), "b_duration_s": _r(aligned["dur_b"], 1),
                "runs": [{k: v for k, v in run_summary(r, step).items() if k in {"a_range_s", "b_range_s", "kind", "mirrored"}}
                         for r in runs[:8]],
                "align_s": aligned["align_s"]}

    def _align(self, path_a, path_b, cancel, keep_details: bool, warp=None) -> dict:
        cfg = self.cfg
        started = time.monotonic()
        deadline = started + cfg.time_budget_s
        fa, na, wa, ha = video_info(path_a)
        fb, nb, wb, hb = video_info(path_b)
        dur_a, dur_b = na / fa, nb / fb
        key = (str(path_a), na, keep_details)
        if self._a_cache and self._a_cache[0] == key:
            thumbs_a, details = self._a_cache[1]
        else:
            thumbs_a, details = self._pass_a(path_a, fa, na, deadline, cancel) if keep_details else (
                self._pass_a_thumbs(path_a, fa, na, deadline, cancel), {})
            self._a_cache = (key, (thumbs_a, details))      # several B are tried against the same A
        thumbs_b, flipped_b = self._thumbs_b(path_b, fb, nb, deadline, cancel, warp)
        if not thumbs_a or not thumbs_b:
            raise CompareUnavailable("INSUFFICIENT_FRAMES", "could not decode frames from one of the videos")
        ka, kb = sorted(thumbs_a), sorted(thumbs_b)
        A = np.stack([thumbs_a[i] for i in ka])
        B = np.stack([thumbs_b[i] for i in kb])
        Bf = np.stack([flipped_b[i] for i in kb])
        ta = np.asarray([i / fa for i in ka])
        tbg = np.asarray([j / fb for j in kb])
        cand: list[list[tuple[float, float, bool]]] = []
        best_s = np.empty(len(ka), dtype=np.float32)
        K = cfg.top_candidates
        for s in range(0, len(ka), 512):
            S = A[s:s + 512] @ B.T
            Sf = A[s:s + 512] @ Bf.T
            for r in range(S.shape[0]):
                row, rowf = S[r], Sf[r]
                top = row.max()
                topf = rowf.max()
                best_s[s + r] = max(top, topf)
                items: list[tuple[float, float, bool]] = []
                for vec, flag, peak in ((row, False, top), (rowf, True, topf)):
                    if peak < cfg.match_ncc:
                        continue
                    idx = np.argpartition(-vec, min(K, len(vec) - 1))[:K]
                    for j in idx[np.argsort(-vec[idx])]:
                        if vec[j] >= max(cfg.match_ncc, peak - cfg.tie_margin):
                            items.append((float(tbg[j]), float(vec[j]), flag))
                # a mirrored match must clearly beat the unmirrored one (symmetric pictures match both ways)
                if any(not c[2] for c in items) and topf <= top + 0.02:
                    items = [c for c in items if not c[2]]
                items.sort(key=lambda c: -c[1])
                cand.append(items)
        runs = build_runs(ta, cand, cfg)
        runs = self._check_frozen(runs, ta, A)
        runs = self._fill_altered(runs, ta, A, tbg, B, Bf)
        return {"fa": fa, "fb": fb, "na": na, "nb": nb, "dur_a": dur_a, "dur_b": dur_b, "size_a": (wa, ha),
                "size_b": (wb, hb), "ta": ta, "best_s": best_s, "runs": runs, "details": details,
                "align_s": round(time.monotonic() - started, 1), "deadline": deadline, "started": started}

    @staticmethod
    def _check_frozen(runs: list[Run], ta: np.ndarray, A: np.ndarray) -> list[Run]:
        """A 'frozen' run (B time standing still) is only a freeze when A itself stands still; a moving A whose
        samples all resemble one B frame (self-similar or low-texture footage) is a chance match and is dropped."""

        out = []
        index = {round(float(t), 4): i for i, t in enumerate(ta)}
        for run in runs:
            if run.kind != "frozen":
                out.append(run)
                continue
            ids = [index.get(round(t, 4)) for t in run.ta]
            ids = [i for i in ids if i is not None]
            steady = [float(A[i] @ A[j]) for i, j in zip(ids, ids[1:])]
            if steady and float(np.median(steady)) >= 0.985:
                out.append(run)
                continue
            # A moves while B advances slowly but steadily (VideoSham 0069: 10 s of A show B 0-1 s): strong slow motion
            slope, _ = run.fit()
            steps = np.diff(np.asarray(run.tb))
            if (slope > 0 and max(run.tb) - min(run.tb) >= 0.25 and len({round(t, 2) for t in run.tb}) >= 3
                    and float((steps >= -0.05).mean()) >= 0.9):
                run.kind_override = "slowed"
                out.append(run)
        return out

    def _fill_altered(self, runs: list[Run], ta: np.ndarray, A: np.ndarray, tbg: np.ndarray, B: np.ndarray,
                      Bf: np.ndarray) -> list[Run]:
        """A gap between two runs that keeps B's timeline (B advances as much as A) is compared directly with the B
        moments it lines up with. Moderately similar thumbnails mean the same shot with part of the picture changed
        (an edited object, a large overlay): such a gap becomes an 'altered' run, so the region pass looks at it.
        Low similarity leaves it a replaced span."""

        cfg = self.cfg
        if len(runs) < 2:
            return runs
        ordered = sorted(runs, key=lambda r: r.a_start)
        extra = []
        for r1, r2 in zip(ordered, ordered[1:]):
            if r1.kind != "normal" or r2.kind != "normal" or r1.flip != r2.flip:
                continue
            a_gap = r2.a_start - r1.a_end
            b_gap = r2.b_at(r2.a_start) - r1.b_at(r1.a_end)
            if a_gap < 1.0 or abs(b_gap - a_gap) > 1.0:
                continue
            slope = b_gap / a_gap if a_gap > 0 else 1.0
            sel = np.flatnonzero((ta > r1.a_end + 1e-6) & (ta < r2.a_start - 1e-6))
            pts, sims = [], []
            Bx = Bf if r1.flip else B
            for i in sel:
                tb = r1.b_at(r1.a_end) + slope * (ta[i] - r1.a_end)
                j = int(np.argmin(np.abs(tbg - tb)))
                lo, hi = max(0, j - 1), min(len(tbg), j + 2)
                sim = float(np.max(Bx[lo:hi] @ A[i]))
                pts.append((float(ta[i]), float(tbg[lo + int(np.argmax(Bx[lo:hi] @ A[i]))])))
                sims.append(sim)
            if pts and float(np.median(sims)) >= cfg.altered_ncc:
                run = Run([p[0] for p in pts], [p[1] for p in pts], flip=r1.flip, altered=True,
                          similarity=float(np.median(sims)))
                if len(run.ta) >= 2:
                    extra.append(run)
        return sorted(runs + extra, key=lambda r: r.a_start)

    def _pass_a_thumbs(self, path, fps, n, deadline, cancel):
        import cv2

        cfg = self.cfg
        dur = n / fps
        thumb_idx = {min(n - 1, int(round(t * fps))) for t in np.arange(0, max(dur - 0.05, 0.01), cfg.thumb_step_a)}
        thumbs = {}
        cap = cv2.VideoCapture(str(path))
        last, idx = max(thumb_idx), -1
        try:
            while idx < last:
                if not cap.grab():
                    break
                idx += 1
                if idx not in thumb_idx:
                    continue
                if idx % 64 == 0:
                    self._check(deadline, cancel)
                ok, bgr = cap.retrieve()
                if ok:
                    thumbs[idx] = _thumb(bgr)
        finally:
            cap.release()
        return thumbs

    def find_registration(self, path_a: Path, path_b: Path, cancel: Event | None = None, *, b_step_s: float = 1.0,
                          max_b: int = 400):
        """Candidate B->A homographies for an A that shows B cropped, scaled, letterboxed or framed differently.

        Three A frames (25/50/75%) are matched by SIFT + ratio test + RANSAC against B frames every b_step_s; the B
        frame with the most inliers gives each candidate. Panning footage makes a crop ambiguous in time (the same
        window of a neighbouring B frame also fits), so the caller tries each distinct candidate and keeps the one under
        which the whole of A aligns best. Returns media_diff.Registration objects (B native -> A work size)."""

        import cv2

        from . import media_diff

        fa, na, wa, ha = video_info(path_a)
        fb, nb, wb, hb = video_info(path_b)
        a_size, b_size = media_diff.work_size(wa, ha), media_diff.work_size(wb, hb)
        # features at a larger size than the work size: an A shown as a small inset of B (picture-in-picture, VCSL
        # pair 1) leaves too few SIFT matches at 480 px (best 17 inliers vs 23-42 at 960 px). Long B: 640 px (time).
        reg_w = 960 if nb / fb <= 300 else 640

        def reg_size(w, h):
            return (reg_w, max(2, int(round(reg_w * h / max(w, 1)))))

        a_reg, b_reg = reg_size(wa, ha), reg_size(wb, hb)
        sift = cv2.SIFT_create(nfeatures=3000 if reg_w > 640 else 1500)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        to_reg_b = np.diag([b_reg[0] / b_size[0], b_reg[1] / b_size[1], 1.0])
        from_reg_a = np.diag([a_size[0] / a_reg[0], a_size[1] / a_reg[1], 1.0])

        def features(frame, size):
            g = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            return sift.detectAndCompute(g, None)

        a_idx = [int(na * q) for q in (0.25, 0.5, 0.75)]
        a_frames = grab(path_a, a_idx)
        step = max(b_step_s, (nb / fb) / max_b)
        b_idx = sorted({min(nb - 1, int(round(t * fb))) for t in np.arange(0, nb / fb, step)})
        b_frames = grab(path_b, b_idx)
        b_feats = {j: features(f, b_reg) for j, f in b_frames.items()}
        candidates = []
        for i in a_idx:
            if i not in a_frames:
                continue
            kp_a, des_a = features(a_frames[i], a_reg)
            if des_a is None or len(kp_a) < 12:
                continue
            best = None
            for j, (kp_b, des_b) in b_feats.items():
                if cancel is not None and cancel.is_set():
                    raise CompareUnavailable("CANCELLED", "comparison was cancelled")
                if des_b is None or len(kp_b) < 12:
                    continue
                pairs = matcher.knnMatch(des_b, des_a, k=2)
                good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
                if len(good) < 12 or (best is not None and len(good) <= best[0]):
                    continue
                src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
                if H is None or not _plausible_homography(H):
                    continue
                inliers = int(mask.sum())
                if inliers >= 20 and (best is None or inliers > best[0]):
                    best = (inliers, from_reg_a @ H @ to_reg_b, j / fb)
            if best is not None:
                candidates.append(best)
        self.last_registration_candidates = [(c[0], round(c[2], 2)) for c in candidates]
        if not candidates:
            return []
        corners = np.float32([[0, 0], [b_size[0], 0], [b_size[0], b_size[1]], [0, b_size[1]]]).reshape(-1, 1, 2)

        def close(h1, h2):
            return float(np.max(np.abs(cv2.perspectiveTransform(corners, h1) - cv2.perspectiveTransform(corners, h2)))) < 12
        support = [sum(close(c[1], d[1]) for d in candidates) for c in candidates]
        out = []
        for k in sorted(range(len(candidates)), key=lambda k: (-support[k], -candidates[k][0])):
            inliers, H, _ = candidates[k]
            if any(close(H, r.transform) for r in out):
                continue
            shift = float(np.max(np.abs(cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                                        - corners.reshape(-1, 2) * np.array([a_size[0] / b_size[0], a_size[1] / b_size[1]]))))
            if shift < 4:
                continue          # same framing: the plain comparison already applies
            info = {"applied": True, "mode": "homography", "inliers": inliers, "supporting_frames": support[k],
                    "max_corner_shift_px": round(shift, 1), "method": "SIFT + ratio test + RANSAC homography (B -> A)"}
            out.append(media_diff.Registration(a_size, b_size, "homography", H, info))
        return out

    @staticmethod
    def match_fraction(aligned: dict) -> float:
        ta = aligned["ta"]
        if not len(ta):
            return 0.0
        step = ta[1] - ta[0] if len(ta) > 1 else 0.5
        m = np.zeros(len(ta), dtype=bool)
        for run in aligned["runs"]:
            m |= (ta >= run.a_start - step / 2 - 1e-6) & (ta <= run.a_end + step / 2 + 1e-6)
        return float(m.mean())

    def compare(self, path_a: Path, path_b: Path, cancel: Event | None = None, *,
                external_runs: list[Run] | None = None, registration: Any = None) -> tuple[dict, list[dict]]:
        """Full comparison. external_runs (from align_media's SIFT alignment) replace the near-duplicate runs; with a
        registration (media_diff.Registration, B -> A) the detail pass warps B frames before differencing."""

        cfg = self.cfg
        aligned = self._align(path_a, path_b, cancel, keep_details=True)
        self.last_engine = "near_duplicate"
        if external_runs is not None:
            aligned["runs"] = external_runs
        elif registration is None and self.match_fraction(aligned) < 0.8:
            # cropped / scaled / letterboxed A: one homography, then the same alignment on warped B frames
            best_fraction = self.match_fraction(aligned)
            for found in self.find_registration(path_a, path_b, cancel)[:3]:
                self._a_cache = None
                warped = self._align(path_a, path_b, cancel, keep_details=True, warp=found)
                if self.match_fraction(warped) >= best_fraction + 0.1:
                    best_fraction = self.match_fraction(warped)
                    aligned, registration = warped, found
                    self.last_engine = "registered"
                    self.last_registration = found.public()
        runs: list[Run] = aligned["runs"]
        ta, dur_a, dur_b = aligned["ta"], aligned["dur_a"], aligned["dur_b"]
        structure = structure_from_runs(runs, ta, dur_a, dur_b, cfg)
        matched = structure["matched"]
        frac = float(matched.mean()) if len(matched) else 0.0
        timing = {"align_s": aligned["align_s"]}
        if frac < 0.3:
            result = self._report(structure, runs, [], None, [], aligned)
            result["timings_s"] = {**timing, "total_s": round(time.monotonic() - aligned["started"], 1)}
            return result, []
        samples = []
        step_a = cfg.thumb_step_a
        for a_idx, (full, small_bgr, gray) in sorted(aligned["details"].items()):
            t = a_idx / aligned["fa"]
            run = next((r for r in runs if r.a_start - step_a <= t <= r.a_end + step_a), None)
            if run is None:
                continue
            s = _Sample(t=t, idx=a_idx, tb_pred=float(run.b_at(t)), jpeg=b"", face=full is not None, flip=run.flip)
            s.altered = run.altered
            s.registration = registration.get(id(run)) if isinstance(registration, dict) else registration
            s.a_small = small_bgr
            s.a_gray = gray
            s.a_full = jpeg(full) if full is not None else None
            samples.append(s)
        if external_runs is not None:
            # feature-alignment anchors are only accurate to its sampling interval: search B more widely, then refit
            saved = cfg.window_s
            cfg.window_s = 0.8
            try:
                self._pass_b_details(path_b, aligned["fb"], aligned["nb"], samples, aligned["deadline"], cancel, registration)
            finally:
                cfg.window_s = saved
            for run in runs:
                pts = [(s.t, s.b_idx / aligned["fb"]) for s in samples
                       if getattr(s, "b_ncc", 0) >= 0.8 and run.a_start - step_a <= s.t <= run.a_end + step_a]
                if len(pts) >= 3:
                    slope, offset = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
                    run.tb = [float(slope * t + offset) for t in run.ta]
        else:
            self._pass_b_details(path_b, aligned["fb"], aligned["nb"], samples, aligned["deadline"], cancel, registration)
        if external_runs is not None:
            # feature-aligned runs are checked against the registered frames: a run whose frames do not look alike
            # (median detail NCC < 0.6) is a wrong correspondence and is dropped
            def run_of(t):
                return next((r for r in runs if r.a_start - step_a <= t <= r.a_end + step_a), None)
            keep = []
            for run in runs:
                vals = [getattr(s, "b_ncc", 0.0) for s in samples if run_of(s.t) is run]
                if vals and float(np.median(vals)) >= 0.6 and not (run.kind == "frozen" and len(run.ta) < 6):
                    keep.append(run)
            if len(keep) != len(runs):
                runs = keep
                aligned["runs"] = runs
                structure = structure_from_runs(runs, ta, dur_a, dur_b, cfg)
                samples = [s for s in samples if run_of(s.t) is not None]
                if float(structure["matched"].mean() if len(structure["matched"]) else 0.0) < 0.3:
                    result = self._report(structure, runs, [], None, [], aligned)
                    result["timings_s"] = {**timing, "total_s": round(time.monotonic() - aligned["started"], 1)}
                    return result, []
        timing["detail_s"] = round(time.monotonic() - aligned["started"], 1)
        regions = self._regions(samples)
        persons = self._faces(samples) if self.faces is not None else None
        timing["faces_s"] = round(time.monotonic() - aligned["started"], 1)
        images = self._images(structure["segments_not_in_b"], regions, persons, samples, path_a, aligned["fa"],
                              path_b, aligned["fb"])
        result = self._report(structure, runs, regions, persons, samples, aligned)
        result["timings_s"] = {**timing, "total_s": round(time.monotonic() - aligned["started"], 1)}
        return result, images

    # .................................................................. B detail pass

    def _pass_b_details(self, path, fps, n, samples, deadline, cancel, registration=None):
        import cv2

        cfg = self.cfg
        windows: dict[int, list[int]] = {}
        for k, s in enumerate(samples):
            c = int(round(s.tb_pred * fps))
            for j in range(max(0, c - int(cfg.window_s * fps) - 2), min(n, c + int(cfg.window_s * fps) + 3)):
                windows.setdefault(j, []).append(k)
        if not windows:
            return
        ends: dict[int, int] = {}
        for j, ks in windows.items():
            for k in ks:
                ends[k] = max(ends.get(k, j), j)
        cap = cv2.VideoCapture(str(path))
        idx, last = -1, max(windows)
        try:
            while idx < last:
                if not cap.grab():
                    break
                idx += 1
                ks = windows.get(idx)
                if not ks:
                    continue
                if idx % 32 == 0:
                    self._check(deadline, cancel)
                ok, bgr = cap.retrieve()
                if not ok:
                    continue
                variants = {}
                for k in ks:
                    s = samples[k]
                    reg = getattr(s, "registration", None)
                    key = (s.flip, id(reg))
                    if key not in variants:
                        frame = cv2.flip(bgr, 1) if s.flip else bgr
                        if reg is not None:
                            aligned, _ = reg.align(frame)
                            frame = cv2.resize(aligned, (s.a_small.shape[1], s.a_small.shape[0]))
                        variants[key] = (frame, _resize_w(frame, cfg.work_width), _small_gray(frame))
                    frame, small, gray = variants[key]
                    s.window.append((float(s.a_gray @ gray), idx, small, frame if s.face else None))
                for k in ks:
                    if ends[k] == idx:
                        self._finish_sample(samples[k])
        finally:
            cap.release()
        for s in samples:
            if s.window:
                self._finish_sample(s)

    def _finish_sample(self, s: _Sample) -> None:
        """Keep the best B frame (+-2 neighbours at work size, best at full size for faces), drop the rest."""

        win = sorted(s.window, key=lambda x: x[1])
        s.window = []
        if not win:
            return
        k = int(np.argmax([w[0] for w in win]))
        s.b_ncc = win[k][0]
        s.b_idx = win[k][1]
        s.b_small = [w[2] for w in win[max(0, k - 2):k + 3]]
        s.b_best_small = win[k][2]
        s.b_full = jpeg(win[k][3]) if win[k][3] is not None else None

    # .................................................................. regions

    def _diff(self, s) -> dict | None:
        import cv2

        cfg = self.cfg
        # the run already fixed which B moment this is; b_ncc only guards against a wrong neighbour frame
        if getattr(s, "b_ncc", 0.0) < (0.5 if getattr(s, "altered", False) else 0.65):
            return None
        a = s.a_small
        H, W = a.shape[:2]
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(s.b_best_small, (W, H)), cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(cv2.GaussianBlur(ga, (5, 5), 0), cv2.GaussianBlur(gb, (5, 5), 0), warp,
                                           cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), None, 5)
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)
        ab = cv2.GaussianBlur(a, (5, 5), 0)
        d = None
        for b in s.b_small:
            bw = cv2.warpAffine(cv2.resize(b, (W, H)), warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
            dk = cv2.absdiff(ab, cv2.GaussianBlur(bw, (5, 5), 0)).max(axis=2)
            d = dk if d is None else np.minimum(d, dk)
        m = int(0.02 * min(W, H))
        d[:m], d[-m:], d[:, :m], d[:, -m:] = 0, 0, 0, 0
        noise = float(np.median(d))
        thr = float(min(60, max(24, 20 + 3 * noise)))
        mask = (d > thr).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []
        for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4]:
            area = stats[i, cv2.CC_STAT_AREA] / (W * H)
            if area < cfg.min_region_area * 0.6:
                break
            x, y, bw_, bh = (int(v) for v in stats[i, :4])
            boxes.append((x / W, y / H, (x + bw_) / W, (y + bh) / H, float(area)))
        return {"area": float(mask.mean()), "noise": noise, "boxes": boxes,
                "scale": float(math.sqrt(abs(np.linalg.det(warp[:, :2]))))}

    def _regions(self, samples) -> list[dict]:
        cfg = self.cfg
        tracks: list[dict] = []
        for s in samples:
            s.diff = self._diff(s)
            if not s.diff or s.diff["area"] > cfg.max_region_area:
                continue
            for box in s.diff["boxes"]:
                placed = False
                for tr in tracks:
                    if tr["last_t"] >= s.t - 2.6 * cfg.detail_step and _iou(tr["box"], box) >= 0.2:
                        tr["obs"].append((s.t, box))
                        tr["last_t"] = s.t
                        tr["box"] = box
                        placed = True
                        break
                if not placed:
                    tracks.append({"obs": [(s.t, box)], "last_t": s.t, "box": box})
        out = []
        for tr in tracks:
            obs = tr["obs"]
            areas = [b[4] for _, b in obs]
            if len(obs) < 2:
                continue
            steady = (float(np.mean([_iou(obs[i][1], obs[i + 1][1]) for i in range(len(obs) - 1)]))
                      if len(obs) >= 5 else 0.0)
            med_area = float(np.median(areas))
            if med_area < cfg.min_region_area and not (steady >= 0.5 and med_area >= cfg.min_region_area_persistent):
                continue
            xs0 = np.median([b[0] for _, b in obs]); ys0 = np.median([b[1] for _, b in obs])
            xs1 = np.median([b[2] for _, b in obs]); ys1 = np.median([b[3] for _, b in obs])
            peak_t, peak_box = max(obs, key=lambda o: o[1][4])
            out.append({"a_range_s": [_r(obs[0][0], 1), _r(obs[-1][0], 1)], "samples": len(obs),
                        "box": [_r(xs0, 3), _r(ys0, 3), _r(xs1, 3), _r(ys1, 3)],
                        "area": _r(float(np.median(areas)), 4), "peak_t": _r(peak_t, 2),
                        "peak_box": [_r(v, 3) for v in peak_box[:4]],
                        "_score": len(obs) * float(np.median(areas)) ** 0.5 * (1 + steady)})
        out.sort(key=lambda r: -r["_score"])
        return out

    # .................................................................. faces

    def _faces(self, samples) -> list[dict]:
        cfg = self.cfg
        persons: list[dict] = []
        for s in samples:
            if not s.face or s.a_full is None or getattr(s, "b_full", None) is None or \
                    getattr(s, "b_ncc", 0) < (0.55 if getattr(s, "altered", False) else 0.85):
                continue
            a = unjpeg(s.a_full)
            b = unjpeg(s.b_full)
            fa_ = self.faces.detect(a, cfg.face_min_px)
            if not fa_:
                continue
            sx, sy = a.shape[1] / b.shape[1], a.shape[0] / b.shape[0]
            fb_raw = self.faces.detect(b, max(20, int(cfg.face_min_px / max(sx, 1))))
            fb_ = [(x0 * sx, y0 * sy, x1 * sx, y1 * sy) for x0, y0, x1, y1 in fb_raw]
            ea = self.faces.embed(a, fa_)
            eb = self.faces.embed(b, fb_raw)
            for k, box in enumerate(fa_):
                j = max(range(len(fb_)), key=lambda j: _iou(box, fb_[j]), default=None)
                sim = float(ea[k] @ eb[j]) if j is not None and _iou(box, fb_[j]) >= 0.3 else None
                target = None
                for p in persons:
                    if float(ea[k] @ p["centroid"]) >= cfg.face_same_person:
                        target = p
                        break
                if target is None:
                    target = {"centroid": ea[k].copy(), "obs": []}
                    persons.append(target)
                target["obs"].append({"t": s.t, "box": box, "sim": sim, "b_box": fb_raw[j] if sim is not None else None,
                                      "a_jpeg": s.a_full, "b_jpeg": s.b_full, "px": box[2] - box[0]})
                c = target["centroid"] * (len(target["obs"]) - 1) + ea[k]
                target["centroid"] = c / (np.linalg.norm(c) + 1e-9)
        out = []
        for p in persons:
            sims = [o["sim"] for o in p["obs"] if o["sim"] is not None]
            if not sims:
                continue
            med = float(np.median(sims))
            strong = len(sims) >= 2 and med < cfg.face_swapped
            weak = not strong and len(sims) >= cfg.face_weak_min_compared and med < cfg.face_swapped_weak
            ex = sorted((o for o in p["obs"] if o["sim"] is not None), key=lambda o: (o["sim"], -o["px"]))
            out.append({"observations": len(p["obs"]), "compared": len(sims), "median_similarity": _r(med, 2),
                        "a_times_s": [_r(p["obs"][0]["t"], 1), _r(p["obs"][-1]["t"], 1)],
                        "identity_differs": bool(strong or weak), "strength": "strong" if strong else "moderate" if weak else None,
                        "position": _position(p["obs"]),
                        "_examples": ex[:3] if (strong or weak) else ex[-2:]})
        out.sort(key=lambda p: (not p["identity_differs"], -p["compared"]))
        return out

    # .................................................................. evidence images

    def _images(self, segments, regions, persons, samples, path_a, fa, path_b, fb) -> list[dict]:
        import cv2

        cfg = self.cfg
        images = []
        frames_needed_a = {}
        segments = [x for x in segments if x["duration_s"] >= 3.0]
        for k, seg in enumerate(segments[:4]):
            s0, s1 = seg["a_range_s"]
            for t in (s0 + 0.25 * (s1 - s0), s0 + 0.75 * (s1 - s0)):
                frames_needed_a[int(round(t * fa))] = ("segment", k)
        grabbed = grab(path_a, sorted(frames_needed_a)) if frames_needed_a else {}
        before, need_b, b_ref = {}, {}, {}
        for k, seg in enumerate(segments[:4]):
            if seg["b_before_s"] is None:
                continue
            t = seg["b_before_s"]
            if seg["b_gap_s"] is not None and abs(seg["b_gap_s"] - seg["duration_s"]) <= 1.0 and seg["b_gap_s"] > 1.0:
                t = (seg["b_before_s"] + seg["b_after_s"]) / 2
            b_ref[k] = t
            need_b[int(round(t * fb))] = k
        grabbed_b = grab(path_b, sorted(need_b)) if need_b else {}
        for idx, k in need_b.items():
            if idx in grabbed_b:
                before[k] = grabbed_b[idx]
        for k, seg in enumerate(segments[:4]):
            frames = [grabbed[i] for i, v in sorted(frames_needed_a.items()) if v == ("segment", k) and i in grabbed]
            if not frames:
                continue
            tiles = [label(_resize_w(f, 480), f"A {t:.1f}s") for f, t in
                     zip(frames, (seg["a_range_s"][0] + 0.25 * seg["duration_s"], seg["a_range_s"][0] + 0.75 * seg["duration_s"]))]
            if k in before:
                same = b_ref.get(k) != seg["b_before_s"]
                tiles.append(label(_resize_w(before[k], 480), f"B {b_ref[k]:.1f}s" + (" (same moment)" if same else " (last matching moment before)")))
            h = min(t.shape[0] for t in tiles)
            strip = np.hstack([t[:h] for t in tiles])
            images.append({"kind": "segment", "index": k, "jpeg": jpeg(strip, 88),
                           "caption": f"A {seg['a_range_s'][0]}-{seg['a_range_s'][1]} s has no counterpart in B: two frames from A"
                                      + (f", then B at {b_ref[k]:.1f} s for context" if k in before else "")})
        by_t = {round(s.t, 2): s for s in samples}
        for k, reg in enumerate(regions[:4]):
            s = by_t.get(round(reg["peak_t"], 2))
            if s is None or getattr(s, "b_best_small", None) is None:
                continue
            a = s.a_small
            H, W = a.shape[:2]
            b = cv2.resize(s.b_best_small, (W, H))
            x0, y0, x1, y1 = reg["peak_box"]
            mx, my = max(0.08, (x1 - x0) * 0.6), max(0.08, (y1 - y0) * 0.6)
            cx0, cy0 = max(0, int((x0 - mx) * W)), max(0, int((y0 - my) * H))
            cx1, cy1 = min(W, int((x1 + mx) * W)), min(H, int((y1 + my) * H))
            ca, cb = a[cy0:cy1, cx0:cx1], b[cy0:cy1, cx0:cx1]
            scale = 300 / max(1, ca.shape[0])
            ca = cv2.resize(ca, (max(1, int(ca.shape[1] * scale)), 300))
            cb = cv2.resize(cb, (max(1, int(cb.shape[1] * scale)), 300))
            fa_box, fb_box = a.copy(), b.copy()
            for img in (fa_box, fb_box):
                cv2.rectangle(img, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (0, 0, 255), 2)
            top = np.hstack([label(_resize_w(fa_box, 400), f"A {s.t:.1f}s"), label(_resize_w(fb_box, 400), f"B {s.b_idx / fb:.1f}s")])
            bottom = np.hstack([label(ca, "A detail"), label(cb, "B detail")])
            if bottom.shape[1] > top.shape[1]:
                q = top.shape[1] / bottom.shape[1]
                bottom = cv2.resize(bottom, (top.shape[1], max(1, int(bottom.shape[0] * q))))
            bottom = _pad_w(bottom, top.shape[1])
            images.append({"kind": "region", "index": k, "jpeg": jpeg(np.vstack([top, bottom]), 88),
                           "caption": f"A {reg['a_range_s'][0]}-{reg['a_range_s'][1]} s: region in red differs from B at the same moment "
                                      f"(top: whole frames A | B; bottom: enlarged A | B)"})
        for k, p in enumerate(persons or []):
            if not p["identity_differs"] or len(images) >= cfg.max_images:
                continue
            tiles = []
            context = None
            for o in p["_examples"][:2]:
                a = unjpeg(o["a_jpeg"])
                b = unjpeg(o["b_jpeg"])
                tiles.append(_face_pair(a, o["box"], b, o["b_box"], f"{o['t']:.0f}s sim {o['sim']:.2f}"))
                if context is None:     # 10-11: the whole A frame with this face boxed, so the person can be placed
                    context = a.copy()
                    x0, y0, x1, y1 = (int(v) for v in o["box"])
                    th = max(2, context.shape[1] // 320)
                    cv2.rectangle(context, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), th)
                    context = label(cv2.resize(context, (max(1, int(context.shape[1] * 240 / context.shape[0])), 240)),
                                    f"A {o['t']:.0f}s whole frame")
            if tiles:
                row = np.hstack(([context] if context is not None else []) + tiles)
                images.append({"kind": "person", "index": k, "jpeg": jpeg(row, 90),
                               "caption": f"person {k + 1} ({p.get('position') or 'in frame'}): whole frame of A with the face boxed "
                                          "in red, then pairs of the face in A (left) and the face at the same position in B (right); "
                                          f"identity similarity {p['median_similarity']}"})
        return images[:cfg.max_images]

    # .................................................................. report

    def _report(self, structure, runs, regions, persons, samples, aligned) -> dict:
        cfg = self.cfg
        matched = structure["matched"]
        frac = float(matched.mean()) if len(matched) else 0.0
        dur_a, dur_b = aligned["dur_a"], aligned["dur_b"]
        best_s = aligned["best_s"]
        match = {"fraction_of_a_matched": _r(frac, 3), "a_duration_s": _r(dur_a, 1), "b_duration_s": _r(dur_b, 1),
                 "a_size": list(aligned["size_a"]), "b_size": list(aligned["size_b"]),
                 "median_ncc": _r(float(np.median(best_s)), 3) if len(best_s) else None}
        lines = []
        run_rows = [run_summary(r, cfg.thumb_step_a) for r in runs]
        if frac < 0.3:
            lines.append(f"B is probably not the source of A: only {100 * frac:.0f}% of A's half-second samples have a "
                         "consistent match in B. Nothing else is reported.")
            return {"summary_lines": lines, "source_match": False, "match": match,
                    "correspondences": run_rows[:10], "segments_not_in_b": [], "b_cuts": [], "reorders": [], "repeats": [],
                    "changed_regions": [], "faces": None, "coverage": {},
                    "limitations": ["Only the same footage is matched (also mirrored, re-encoded, cropped or with overlays "
                                    "when the SIFT fallback ran); a re-shot or heavily re-edited version is not recognised."]}
        ext = [max(0.0, x) for x in structure["b_extent_s"]]
        lines.append(f"B matches A: {100 * frac:.0f}% of A's half-second samples are found in B "
                     f"(A {dur_a:.0f} s, B {dur_b:.0f} s; the matched footage lies in B {ext[0]:.0f}-{ext[1]:.0f} s).")
        if dur_b - (ext[1] - ext[0]) > 5:
            lines.append(f"A uses B {ext[0]:.0f}-{ext[1]:.0f} s only; the rest of B ({dur_b:.0f} s in all) is not in A.")
        # 10-14：恢复 DETECT v2.4 的输出写法（结论与"该怎么写"紧挨着数据）。合并时改成"只陈述事实、归因放提示"后，27B 不再把
        # 变化区域读成换物、把低相似度读成换脸（evidM27a–f：读成字幕 / 台标 / 美颜 / 变速），比对上原片的样本也拿不到分。
        # 顺序：新增片段 → 变化区域 → 人脸 → 剪辑结构；3 s 以下的零碎差异各并成一行（多为剪辑点附近的对齐误差）。
        segments = list(structure["segments_not_in_b"])
        long_segments = [s for s in segments if s["duration_s"] >= 3.0]
        short_segments = [s for s in segments if s["duration_s"] < 3.0]
        for seg in long_segments:
            ctx = {"inserted": f"; B continues from {seg['b_before_s']} s to {seg['b_after_s']} s around it with no gap, so this footage was added to A",
                   "replaced": f"; it lines up with B {seg['b_before_s']}-{seg['b_after_s']} s, which looks different: that footage was replaced or heavily altered (compare the A | B image)",
                   "replaces_longer_b_span": f"; B continues from {seg['b_before_s']} s to {seg['b_after_s']} s around it, so it replaces about {seg['b_gap_s']} s of B",
                   "b_jumps_back": f"; around it B jumps from {seg['b_before_s']} s back to {seg['b_after_s']} s"}.get(seg["context"] or "", "")
            gen = (" Its length (about 10 s, one shot) matches the fixed clip length of AI video generators, so it is most likely "
                   "an AI-generated clip made to fit this scene; say what it shows."
                   if seg["single_shot_generator_length"] else " Say what it shows and whether it belongs with the footage around it.")
            lines.append(f"Not in B: A {seg['a_range_s'][0]}-{seg['a_range_s'][1]} s ({seg['duration_s']} s){ctx}.{gen}")
        for reg in regions[:6]:
            x0, y0, x1, y1 = reg["box"]
            lines.append(f"Changed region: A {reg['a_range_s'][0]}-{reg['a_range_s'][1]} s, x {x0:.2f}-{x1:.2f} y {y0:.2f}-{y1:.2f} "
                         f"({100 * reg['area']:.1f}% of the frame) differs from B at the same moment in {reg['samples']} samples "
                         "while the rest of the frame matches: an object there was replaced, added or altered (typical of AI "
                         "video editing); name the object in B and what A shows instead.")
        if persons is not None:
            for k, p in enumerate(persons):
                if p["identity_differs"]:
                    verdict = ("this person's face was replaced (AI face swap)" if p.get("strength") == "strong" else
                               "this person's face was most likely replaced (AI face swap; a partial swap keeps some likeness)")
                    lines.append(f"Face identity differs: person {k + 1} ({p.get('position') or 'in frame'}; A {p['a_times_s'][0]}-"
                                 f"{p['a_times_s'][1]} s, {p['compared']} comparisons) has median similarity {p['median_similarity']} "
                                 f"to the face at the same position in B (unchanged people here are usually above 0.8): {verdict}; "
                                 "describe the person (hair, clothing, role, where they sit or stand) from the image.")
        for cut in structure["b_cuts"][:8]:
            lines.append(f"Missing from A: B {cut['b_missing_s'][0]}-{cut['b_missing_s'][1]} s ({cut['duration_s']} s) is skipped "
                         f"at A {cut['a_at_s']} s (A continues straight from before it to after it).")
        for ro in structure["reorders"][:6]:
            lines.append(f"Order differs: at A {ro['a_at_s']} s, A goes from B {ro['b_from_s']} s back to earlier B {ro['b_to_s']} s.")
        for rp in structure["repeats"][:4]:
            lines.append(f"Repeated: B {rp['b_range_s'][0]}-{rp['b_range_s'][1]} s appears in A twice "
                         f"(A {rp['a_ranges_s'][0][0]}-{rp['a_ranges_s'][0][1]} s and {rp['a_ranges_s'][1][0]}-{rp['a_ranges_s'][1][1]} s).")
        retimed_short = []
        for r in run_rows:
            if r["picture_differs"] or (r["kind"] == "normal" and not r["mirrored"]):
                continue
            length = r["a_range_s"][1] - r["a_range_s"][0]
            what = {"frozen": "shows a single B frame held still (a freeze)",
                    "reversed": "plays B backwards",
                    "slowed": f"plays B slower ({r['mapping']['scale']}x of B's speed)",
                    "sped_up": f"plays B faster ({r['mapping']['scale']}x of B's speed)"}.get(r["kind"], "matches B")
            mirror = " mirrored left-right" if r["mirrored"] else ""
            if length < 3.0 and r["kind"] in {"reversed", "slowed", "sped_up"} and not r["mirrored"]:
                retimed_short.append(f"A {r['a_range_s'][0]}-{r['a_range_s'][1]} s")
                continue
            lines.append(f"A {r['a_range_s'][0]}-{r['a_range_s'][1]} s {what}{mirror} (B {r['b_range_s'][0]}-{r['b_range_s'][1]} s).")
        if short_segments:
            lines.append(f"{len(short_segments)} shorter mismatch(es) under 3 s (often fast motion or a re-timed cut): "
                         + ", ".join(f"{x['a_range_s'][0]}-{x['a_range_s'][1]} s" for x in short_segments[:6]) + ".")
        if retimed_short:
            lines.append(f"{len(retimed_short)} short span(s) under 3 s align with B at another speed or backwards (usually "
                         "imprecise alignment around a cut, not an edit): " + ", ".join(retimed_short[:6]) + ".")
        if persons is not None:
            same = [p for p in persons if not p["identity_differs"]]
            if same:
                lines.append(f"{len(same)} other face track(s) match B (median similarity "
                             + ", ".join(str(p["median_similarity"]) for p in same[:4]) + ").")
            if not persons:
                lines.append("No face could be compared (no frontal face found in matched frames).")
        if len(lines) == 1:
            lines.append("No added, missing, re-timed or re-ordered footage, changed region or changed face was found at the "
                         "sampled moments.")
        sampled = [s for s in samples if getattr(s, "diff", None)]
        return {
            "summary_lines": lines,
            "source_match": True,
            "match": match,
            "correspondences": run_rows[:20],
            "segments_not_in_b": structure["segments_not_in_b"],
            "b_cuts": structure["b_cuts"][:20],
            "reorders": structure["reorders"][:20],
            "repeats": structure["repeats"],
            "changed_regions": [{k: v for k, v in r.items() if not k.startswith("_")} for r in regions[:8]],
            "faces": None if persons is None else [{k: v for k, v in p.items() if not k.startswith("_")} for p in persons[:8]],
            "coverage": {"thumb_step_a_s": cfg.thumb_step_a, "detail_samples": len(sampled),
                         "detail_step_s": _r((dur_a / max(1, len(samples))) if samples else None, 2),
                         "face_samples": sum(1 for s in samples if s.face)},
            "limitations": [
                "Differences are measured between two files; which one is the original is not decided by this tool.",
                f"Time boundaries are accurate to about {cfg.thumb_step_a} s; pass range_a_s for frame-exact differences inside one span.",
                "Regions and faces are checked only at the sampled moments listed in coverage; short changes between samples can be missed.",
                "Overlays (subtitles, logos), crops, colour grading or a different upload of the same footage also produce differences.",
                "A difference shows that A's pixels are not B's; it does not by itself show how they were produced.",
            ],
        }


# ----------------------------------------------------------------------------- audio

def _spectrogram(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Log band energies (32 bands, 64 ms frames, 32 ms hop) and per-frame RMS."""

    n, hop = 512, 256
    if len(samples) < n:
        return np.zeros((0, 32), np.float32), np.zeros(0, np.float32), hop / rate
    count = 1 + (len(samples) - n) // hop
    window = np.hanning(n).astype(np.float32)
    frames = np.lib.stride_tricks.as_strided(samples, shape=(count, n), strides=(samples.strides[0] * hop, samples.strides[0]))
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    edges = np.unique(np.geomspace(2, spec.shape[1] - 1, 33).astype(int))
    bands = np.stack([spec[:, edges[i]:max(edges[i] + 1, edges[i + 1])].sum(axis=1) for i in range(len(edges) - 1)], axis=1)
    logb = np.log10(bands + 1e-9).astype(np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)
    return logb, rms, hop / rate


def _window_similarity(fa: np.ndarray, fb: np.ndarray) -> float:
    if not len(fa) or not len(fb):
        return 0.0
    m = min(len(fa), len(fb))
    x, y = fa[:m] - fa[:m].mean(axis=1, keepdims=True), fb[:m] - fb[:m].mean(axis=1, keepdims=True)
    num = (x * y).sum(axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-9
    return float(np.mean(num / den))


def audio_compare(samples_a: np.ndarray, samples_b: np.ndarray, rate: int, mapping: Callable[[float], float | None],
                  dur_a: float, *, window_s: float = 1.0, hop_s: float = 0.5, max_lag_s: float = 0.3,
                  threshold: float = 0.55) -> dict:
    """Compare A's sound with B's sound at the corresponding moment (mapping: A time -> B time, None if none).

    Returns per-window similarity and merged spans where they differ. Silence on one side only is reported as such.
    The measure is spectral (robust to re-encoding); speech replaced by other speech, music laid over, or a different
    sound track all score low."""

    fa, rms_a, hop = _spectrogram(samples_a.astype(np.float32), rate)
    fb, rms_b, _ = _spectrogram(samples_b.astype(np.float32), rate)
    if not len(fa) or not len(fb):
        return {"compared_windows": 0, "spans_differ": [], "similarity": []}
    quiet_a = float(np.percentile(rms_a, 95)) * 0.03 + 1e-5
    quiet_b = float(np.percentile(rms_b, 95)) * 0.03 + 1e-5
    w = max(1, int(round(window_s / hop)))
    lag = int(round(max_lag_s / hop))
    rows = []
    t = 0.0
    while t + window_s <= dur_a + 1e-6:
        tb = mapping(t)
        if tb is None:
            t += hop_s
            continue
        ia, ib = int(round(t / hop)), int(round(tb / hop))
        if ia + w > len(fa) or ib < 0 or ib + w > len(fb):
            t += hop_s
            continue
        seg_a = fa[ia:ia + w]
        loud_a = float(np.mean(rms_a[ia:ia + w])) > quiet_a
        best, best_loud_b = -1.0, False
        for d in range(-lag, lag + 1):
            j = ib + d
            if j < 0 or j + w > len(fb):
                continue
            sim = _window_similarity(seg_a, fb[j:j + w])
            if sim > best:
                best, best_loud_b = sim, float(np.mean(rms_b[j:j + w])) > quiet_b
        if not loud_a and not best_loud_b:
            state = "both_silent"
        elif loud_a and not best_loud_b:
            state = "sound_only_in_a"
        elif not loud_a and best_loud_b:
            state = "silent_in_a"
        else:
            state = "same" if best >= threshold else "differs"
        rows.append((t, tb, best, state))
        t += hop_s
    spans = []
    cur = None
    for t, tb, sim, state in rows:
        bad = state in {"differs", "sound_only_in_a", "silent_in_a"}
        if bad and cur is not None and cur["state"] == state and t <= cur["a_end"] + 1e-6:
            cur["a_end"] = t + window_s
            cur["b_end"] = tb + window_s
            cur["sims"].append(sim)
        elif bad:
            if cur is not None:
                spans.append(cur)
            cur = {"state": state, "a_start": t, "a_end": t + window_s, "b_start": tb, "b_end": tb + window_s, "sims": [sim]}
        elif cur is not None:
            spans.append(cur)
            cur = None
    if cur is not None:
        spans.append(cur)
    out = [{"a_range_s": [_r(s["a_start"], 1), _r(s["a_end"], 1)], "b_range_s": [_r(s["b_start"], 1), _r(s["b_end"], 1)],
            "state": s["state"], "median_similarity": _r(float(np.median(s["sims"])), 2)}
           for s in spans if s["a_end"] - s["a_start"] >= 1.5 * window_s - 1e-6]
    same = [r for r in rows if r[3] == "same"]
    return {"compared_windows": len(rows), "same_windows": len(same), "spans_differ": out,
            "median_similarity": _r(float(np.median([r[2] for r in rows])), 2) if rows else None,
            "threshold": threshold, "window_s": window_s}


def audio_offset(samples_a: np.ndarray, samples_b: np.ndarray, rate: int) -> tuple[float, float]:
    """Global B-minus-A offset (seconds) from cross-correlating the band-energy envelopes; returns (offset, score)."""

    fa, _, hop = _spectrogram(samples_a.astype(np.float32), rate)
    fb, _, _ = _spectrogram(samples_b.astype(np.float32), rate)
    if not len(fa) or not len(fb):
        return 0.0, 0.0
    ea = fa.mean(axis=1) - fa.mean()
    eb = fb.mean(axis=1) - fb.mean()
    n = len(ea) + len(eb) - 1
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(eb, size) * np.conj(np.fft.rfft(ea, size)), size)
    corr = np.concatenate([corr[-(len(ea) - 1):], corr[:len(eb)]]) if len(ea) > 1 else corr[:len(eb)]
    k = int(np.argmax(corr))
    shift = k - (len(ea) - 1)
    denom = np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9
    return shift * hop, float(corr[k] / denom)
