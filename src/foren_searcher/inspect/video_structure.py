"""video_structure: the shot structure of one video, without an original to compare with.

Replaces DETECT generation_scan v2 (detect/shotscan.py, 10-10) and extends it (10-12) to what the editing datasets need
from a single video:

* shots between hard cuts (HSV histogram correlation of consecutive frames), with start/end/duration and a key frame;
* shots lasting 9.8-10.2 s between two hard cuts are marked (the fixed output length of common AI video generators; on
  the EVID dev set 11 of 18 inserted generated clips were such shots, with 4 other such shots in 18 videos), stated as
  a fact about the shot, not as a verdict;
* frozen spans: consecutive frames (near) identical for >= 0.8 s while the shots around move (freeze frames,
  duplicated / slowed frames);
* repeated shots: two non-adjacent shots whose key frames are the same picture (re-used footage, loops);
* black or blank spans (>= 0.2 s) and an abrupt ending on black.

A 10-second shot, a frozen span or a repeated shot says where to look, not what happened; since 10-14 the lines say so
next to the fact (as DETECT generation_scan v2 did) and each 10-second shot gets its own beginning / middle / end image.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from .source_compare import CompareUnavailable, _r, _resize_w, _thumb, grab, jpeg, label

GENERATOR_SHOT_S = (9.8, 10.2)


def analyse(path: Path, *, range_s: tuple[float, float] | None = None, cancel: Event | None = None,
            time_budget_s: float = 600.0, cut_corr: float = 0.6, min_shot_s: float = 0.3,
            freeze_s: float = 0.8, black_s: float = 0.2, max_sheet_tiles: int = 24) -> tuple[dict, list[dict]]:
    import cv2

    started = time.monotonic()
    deadline = started + time_budget_s
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or n <= 0:
        cap.release()
        raise CompareUnavailable("UNSUPPORTED", "video has no frame rate / frame count")
    lo_f = int((range_s[0] if range_s else 0.0) * fps)
    hi_f = int((range_s[1] if range_s else n / fps) * fps)
    prev_hist = prev_small = None
    cuts: list[float] = []
    still: list[bool] = []          # frame i equals frame i-1
    dark: list[bool] = []
    motion: list[float] = []
    samples: list[tuple[float, np.ndarray]] = []     # (time, thumbnail) every 2 frames, for repeated shots
    sample_every = 2
    idx = -1
    try:
        while True:
            if not cap.grab():
                break
            idx += 1
            if idx < lo_f:
                continue
            if idx >= hi_f:
                break
            if idx % 128 == 0:
                if cancel is not None and cancel.is_set():
                    raise CompareUnavailable("CANCELLED", "video_structure was cancelled")
                if time.monotonic() > deadline:
                    raise CompareUnavailable("TIME_LIMIT", "video_structure time budget exceeded")
            ok, bgr = cap.retrieve()
            if not ok:
                still.append(False)
                dark.append(False)
                motion.append(0.0)
                continue
            small = cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA)
            if (idx - lo_f) % sample_every == 0:
                samples.append((idx / fps, _thumb(bgr)))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dark.append(float(gray.mean()) < 18.0 and float(gray.std()) < 8.0)
            if prev_hist is not None and float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)) < cut_corr:
                cuts.append(idx / fps)
            diff = float(np.abs(gray - prev_small).mean()) if prev_small is not None else 0.0
            still.append(prev_small is not None and diff < 0.6)
            motion.append(diff)
            prev_hist, prev_small = hist, gray
    finally:
        cap.release()
    start_s = lo_f / fps
    dur_end = start_s + len(still) / fps
    bounds: list[float] = []
    for x in [start_s, *cuts, dur_end]:
        if bounds and x - bounds[-1] < min_shot_s:
            continue
        bounds.append(x)
    if len(bounds) < 2:
        bounds = [start_s, dur_end]
    shots = list(zip(bounds[:-1], bounds[1:]))
    times = start_s + np.arange(len(still)) / fps
    motion_arr = np.asarray(motion)

    def spans(flags: list[bool], min_len: float) -> list[tuple[float, float]]:
        out, i, arr = [], 0, np.asarray(flags, dtype=bool)
        while i < len(arr):
            if not arr[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(arr) and arr[j + 1]:
                j += 1
            a, b = float(times[i]), float(times[j] + 1 / fps)
            if b - a >= min_len:
                out.append((a, b))
            i = j + 1
        return out

    # a key frame inside a held picture re-encodes it once (a single small jump); it must not split the freeze
    for i in range(1, len(still) - 1):
        if not still[i] and still[i - 1] and still[i + 1] and motion[i] < 2.0:
            still[i] = True
    frozen = []
    for a, b in spans(still, freeze_s):
        around = motion_arr[(times >= a - 2) & (times < a)].tolist() + motion_arr[(times > b) & (times <= b + 2)].tolist()
        moving_around = bool(around) and float(np.median(around)) > 1.0
        frozen.append({"range_s": [_r(a, 2), _r(b, 2)], "duration_s": _r(b - a, 2),
                       "surrounding_footage_moves": moving_around})
    black = [{"range_s": [_r(a, 2), _r(b, 2)], "duration_s": _r(b - a, 2)} for a, b in spans(dark, black_s)]
    ends_black = bool(black) and black[-1]["range_s"][1] >= dur_end - 0.1
    # key frames and repeats
    mids = [int(round(((a + b) / 2) * fps)) for a, b in shots]
    frames = grab(path, sorted(set(mids)))
    by_shot: dict[int, list[np.ndarray]] = {}
    for t, v in samples:
        k = next((i for i, (a, b) in enumerate(shots) if a <= t < b), None)
        if k is not None:
            by_shot.setdefault(k, []).append(v)
    repeats = []
    keys = sorted(by_shot)
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            if keys[y] - keys[x] < 2:
                continue                      # adjacent shots: a cut between two angles of one scene
            X, Y = np.stack(by_shot[keys[x]]), np.stack(by_shot[keys[y]])
            S = X @ Y.T
            hits = int((S.max(axis=0) >= 0.9).sum())
            if min(len(X), len(Y)) >= 8 and hits >= max(8, 0.5 * min(len(X), len(Y))):
                repeats.append({"shots": [keys[x] + 1, keys[y] + 1],
                                "a_ranges_s": [[_r(shots[keys[x]][0], 2), _r(shots[keys[x]][1], 2)],
                                               [_r(shots[keys[y]][0], 2), _r(shots[keys[y]][1], 2)]]})
    shot_rows = []
    for k, (a, b) in enumerate(shots):
        length = b - a
        edge = a <= start_s + 0.05 or b >= dur_end - 0.05
        shot_rows.append({"shot": k + 1, "range_s": [_r(a, 2), _r(b, 2)], "duration_s": _r(length, 2),
                          "generator_length": bool(GENERATOR_SHOT_S[0] <= length <= GENERATOR_SHOT_S[1] and not edge)})
    flagged = [s for s in shot_rows if s["generator_length"]]
    lines = [f"{len(shots)} shot(s) between hard cuts in {start_s:.0f}-{dur_end:.0f} s "
             f"(median shot {float(np.median([b - a for a, b in shots])):.1f} s)."]
    for s in flagged:
        # 10-14：恢复 DETECT generation_scan v2 的写法（结论后面紧跟"该看什么"）
        lines.append(f"Shot {s['shot']} (A {s['range_s'][0]}-{s['range_s'][1]} s) lasts {s['duration_s']} s between two hard cuts: "
                     "the fixed clip length of common AI video generators (10 s). Check whether its content belongs with the "
                     "surrounding footage (new people, places, camera moves or objects not seen elsewhere).")
    # 10-14：静止 / 重复画面 / 黑场合成一行。DETECT generation_scan v2 不报这些；逐条列出时 HI 把它们当伪造点写进答案
    # （evidM27h video_209："opening built from frozen stills"，没去找原片）
    other = []
    if frozen:
        longest = max(frozen, key=lambda f: f["duration_s"])
        other.append(f"{len(frozen)} frozen span(s) (longest {longest['duration_s']} s at "
                     f"{longest['range_s'][0]}-{longest['range_s'][1]} s)")
    if repeats:
        other.append(f"{len(repeats)} pair(s) of shots showing the same picture")
    if black:
        other.append(f"{len(black)} black span(s)" + (" including the very end" if ends_black else ""))
    if other:
        lines.append("Also measured: " + ", ".join(other) + ". These are ordinary editing (static shots, cutaways, "
                     "fades) and are not a sign of manipulation by themselves.")
    # 10-14：每个约 10 s 的镜头单独一张图（开头 / 中间 / 结尾三帧），排在总览图前面（DETECT generation_scan v2 的做法）
    images = []
    for s in flagged[:4]:
        a, b = s["range_s"]
        picks = [min(b - 0.05, a + 0.4), (a + b) / 2, max(a + 0.05, b - 0.4)]
        got = grab(path, [int(round(t * fps)) for t in picks])
        tiles_s = [label(_resize_w(got[int(round(t * fps))], 320), f"A {t:.1f}s") for t in picks if int(round(t * fps)) in got]
        if tiles_s:
            h = min(t.shape[0] for t in tiles_s)
            images.append({"kind": "shot", "index": s["shot"] - 1, "jpeg": jpeg(np.hstack([t[:h] for t in tiles_s]), 85),
                           "caption": f"shot {a}-{b} s ({s['duration_s']} s): beginning, middle, end"})
    # contact sheet(s): one tile per shot
    tiles = []
    for k, m in enumerate(mids):
        if m in frames:
            a, b = shots[k]
            tiles.append(label(_resize_w(frames[m], 240), f"{k + 1}: {a:.1f}-{b:.1f}s"))
    for sheet, start in enumerate(range(0, len(tiles), max_sheet_tiles)):
        if sheet >= 2:
            break
        group = tiles[start:start + max_sheet_tiles]
        h = min(t.shape[0] for t in group)
        rows = [np.hstack([t[:h] for t in group[r:r + 6]]) for r in range(0, len(group), 6)]
        width = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)]) if r.shape[1] < width else r
                for r in rows]
        images.append({"kind": "shots", "index": sheet, "jpeg": jpeg(np.vstack(rows), 85),
                       "caption": f"middle frame of shots {start + 1}-{start + len(group)}, labelled with shot number and time"})
    result = {"summary_lines": lines, "shots": shot_rows[:200], "generator_length_shots": flagged,
              "frozen_spans": frozen[:50], "repeated_shots": repeats[:20], "black_spans": black[:50],
              "ends_on_black": ends_black,
              "coverage": {"range_s": [_r(start_s, 2), _r(dur_end, 2)], "frames": len(still), "fps": _r(fps, 3)},
              "limitations": [
                  "Cuts are found from colour-histogram changes; dissolves, fades and cuts between similar pictures can be missed.",
                  "A 10-second shot, a frozen span or a repeated picture is a cue about editing or generation, not proof; "
                  "real footage also contains static shots and 10-second takes.",
                  "No such cue does not show that the video is authentic or unedited."],
              "timings_s": {"total_s": round(time.monotonic() - started, 1)}}
    return result, images
