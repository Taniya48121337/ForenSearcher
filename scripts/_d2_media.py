"""D2 案卷的媒体处理（2026-10-13）：只用 PyAV（本机不一定有 ffmpeg 程序），把原片去掉若干段后重新编码成材料视频。

cut_segments(src, dst, cuts) — cuts 是原片时间轴上的 [start, end) 秒段列表；输出视频的时间轴是去掉这些段后接起来的。
返回 {"duration_in", "duration_out", "removed": [[s, e], ...], "cut_points_out": [材料里每个剪点的时刻]}。
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any


def probe_duration(path: str | Path) -> float | None:
    import av

    with av.open(str(path)) as container:
        if container.duration:
            return float(container.duration / 1_000_000)
        vs = next(iter(container.streams.video), None)
        if vs is not None and vs.duration and vs.time_base:
            return float(vs.duration * vs.time_base)
    return None


def _removed_before(t: float, cuts: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(t, e) - s) for s, e in cuts if s < t)


def _inside(t: float, cuts: list[tuple[float, float]]) -> bool:
    return any(s <= t < e for s, e in cuts)


def cut_segments(src: str | Path, dst: str | Path, cuts: list[tuple[float, float]], *, max_height: int = 720) -> dict[str, Any]:
    import av

    cuts = sorted((float(s), float(e)) for s, e in cuts)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    with av.open(str(src)) as inp:
        vin = next(iter(inp.streams.video), None)
        ain = next(iter(inp.streams.audio), None)
        if vin is None:
            raise RuntimeError(f"{src} has no video stream")
        fps = vin.average_rate or vin.guessed_rate or Fraction(30, 1)
        fps = Fraction(fps).limit_denominator(1001)
        width, height = vin.codec_context.width, vin.codec_context.height
        if height > max_height:
            width, height = int(round(width * max_height / height)), max_height
        width, height = width - width % 2, height - height % 2
        with av.open(str(tmp), "w") as out:
            vout = None
            for codec in ("libx264", "h264", "mpeg4"):
                try:
                    vout = out.add_stream(codec, rate=fps)
                    break
                except Exception:  # noqa: BLE001 —— 编码器不可用就换下一个
                    continue
            if vout is None:
                raise RuntimeError("no usable video encoder in this PyAV build")
            vout.width, vout.height, vout.pix_fmt = width, height, "yuv420p"
            if vout.codec_context.name in ("libx264", "h264"):
                vout.options = {"crf": "20", "preset": "veryfast"}
            else:
                vout.bit_rate = 4_000_000
            aout = resampler = None
            if ain is not None:
                rate = ain.codec_context.sample_rate or 44100
                aout = out.add_stream("aac", rate=rate)
                aout.layout = "stereo"
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
            streams = [vin] + ([ain] if ain is not None else [])
            next_vpts = 0
            last_t = 0.0
            for packet in inp.demux(*streams):
                for frame in packet.decode():
                    if frame.pts is None or frame.time is None:
                        continue
                    t = float(frame.time)
                    if _inside(t, cuts):
                        continue
                    if packet.stream.type == "video":
                        last_t = max(last_t, t)
                        new_t = t - _removed_before(t, cuts)
                        pts = int(round(new_t * fps))
                        if pts < next_vpts:          # 时间戳回退或重复：跳过，保证单调
                            continue
                        img = frame.reformat(width=width, height=height, format="yuv420p")
                        img.pts = pts
                        img.time_base = Fraction(1, 1) / fps
                        for p in vout.encode(img):
                            out.mux(p)
                        next_vpts = pts + 1
                    elif aout is not None and resampler is not None:
                        frame.pts = None
                        for rf in resampler.resample(frame):
                            rf.pts = None
                            for p in aout.encode(rf):
                                out.mux(p)
            for p in vout.encode(None):
                out.mux(p)
            if aout is not None and resampler is not None:
                for rf in resampler.resample(None):
                    rf.pts = None
                    for p in aout.encode(rf):
                        out.mux(p)
                for p in aout.encode(None):
                    out.mux(p)
    tmp.replace(dst)
    d_in = probe_duration(src)
    removed = [[round(s, 3), round(e, 3)] for s, e in cuts]
    points = [round(s - _removed_before(s, cuts), 3) for s, _ in cuts]
    return {"duration_in": d_in, "duration_out": probe_duration(dst), "removed": removed, "cut_points_out": points,
            "video_codec": vout.codec_context.name, "size": [width, height], "fps": float(fps)}
