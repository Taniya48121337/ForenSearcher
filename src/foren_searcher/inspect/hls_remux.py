"""仅处理已下载文件的 HLS 无损封装子进程。"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any
from urllib.parse import unquote

import av
from yt_dlp.utils import parse_m3u8_attributes


def _epochs(playlist: Path) -> list[tuple[Path, Fraction]]:
    root = playlist.resolve().parent
    original_files = {path.resolve() for path in root.iterdir() if path.is_file()}

    def local(name: str) -> Path:
        path = (root/name).resolve()
        if path not in original_files:
            raise ValueError("playlist references a file outside downloaded inputs")
        return path

    def split(path: Path, prefix: str) -> list[tuple[Path, Fraction]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = [line for line in lines if line.startswith(("#EXTM3U", "#EXT-X-VERSION:", "#EXT-X-TARGETDURATION:"))]
        total = sum((Fraction(line.split(":",1)[1].split(",",1)[0]) for line in lines if line.startswith("#EXTINF:")), Fraction(0))
        if "#EXT-X-DISCONTINUITY" not in lines:
            return [(path, total)]
        result: list[tuple[Path, Fraction]] = []
        body: list[str] = []
        init: str | None = None
        duration = Fraction(0)

        def flush() -> None:
            nonlocal body, duration
            if duration:
                output = root/f"{prefix}-epoch-{len(result)}.m3u8"
                output.write_text("\n".join(header + body + ["#EXT-X-ENDLIST"]) + "\n", encoding="utf-8")
                result.append((output, duration))
            body = [init] if init else []
            duration = Fraction(0)

        for line in lines:
            if line == "#EXT-X-DISCONTINUITY":
                flush()
            elif line.startswith("#EXT-X-MAP:"):
                init = line
                body.append(line)
            elif line.startswith("#EXTINF:"):
                duration += Fraction(line.split(":",1)[1].split(",",1)[0])
                body.append(line)
            elif line and not line.startswith("#"):
                local(line)
                body.append(line)
        flush()
        return result

    lines = playlist.read_text(encoding="utf-8").splitlines()
    stream = next((index for index,line in enumerate(lines) if line.startswith("#EXT-X-STREAM-INF:")), None)
    if stream is None:
        return split(playlist, "video")
    video = split(local(lines[stream+1]), "video")
    audio_tag = next(line for line in lines if line.startswith("#EXT-X-MEDIA:"))
    attrs = parse_m3u8_attributes(audio_tag.split(":",1)[1])
    audio = split(local(attrs["URI"]), "audio")
    if len(video) != len(audio):
        raise ValueError("audio and video discontinuities do not align")
    output = []
    for index, ((video_path, duration), (audio_path, audio_duration)) in enumerate(zip(video, audio, strict=True)):
        if abs(duration-audio_duration) > 1:
            raise ValueError("audio and video epochs cover different durations")
        path = root/f"combined-epoch-{index}.m3u8"
        path.write_text('#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="audio",DEFAULT=YES,URI="' + audio_path.name +
            '"\n#EXT-X-STREAM-INF:BANDWIDTH=1000000,AUDIO="audio"\n' + video_path.name + '\n', encoding="utf-8")
        output.append((path, duration))
    return output


def remux(playlist: Path, destination: Path, maximum_bytes: int) -> None:
    root = playlist.resolve().parent
    epochs = _epochs(playlist)
    allowed = {path.resolve() for path in root.iterdir() if path.is_file()}

    def open_local(url, flags, options):
        value = unquote(url.removeprefix("file:"))
        path = Path(value).resolve()
        if flags != 1 or path not in allowed:
            raise PermissionError("HLS decoder requested a file outside its downloaded inputs")
        return path.open("rb")

    with av.open(str(destination), "w", format="matroska", options={"avoid_negative_ts":"make_non_negative"}) as output:
        target_streams: list[Any] = []
        descriptions = []
        position = Fraction(0)
        packets = 0
        for epoch, duration in epochs:
            with av.open(str(epoch), format="hls", io_open=open_local,
                         options={"protocol_whitelist":"file", "allowed_extensions":"ts,m4s,mp4,aac"}) as source:
                if not source.streams.video:
                    raise ValueError("HLS input has no video stream")
                streams = [stream for stream in source.streams if stream.type in {"video", "audio"}]
                current = [(stream.type, stream.codec_context.name, stream.codec_context.extradata,
                            getattr(stream.codec_context, "width", None), getattr(stream.codec_context, "height", None),
                            getattr(stream.codec_context, "sample_rate", None)) for stream in streams]
                if not target_streams:
                    target_streams = [output.add_stream_from_template(stream) for stream in streams]
                    descriptions = current
                elif current != descriptions:
                    raise ValueError("HLS codec parameters change across discontinuities")
                mapping = {stream.index:target for stream,target in zip(streams, target_streams, strict=True)}
                origin = Fraction(source.start_time or 0, av.time_base)
                for packet in source.demux():
                    if packet.stream.index not in mapping or packet.dts is None:
                        continue
                    if packet.time_base is not None:
                        shift = round((position-origin) / packet.time_base)
                        packet.dts += shift
                        if packet.pts is not None:
                            packet.pts += shift
                    packet.stream = mapping[packet.stream.index]
                    output.mux(packet)
                    packets += 1
                    if packets % 128 == 0 and destination.stat().st_size > maximum_bytes:
                        raise ValueError("remux output exceeds configured size limit")
                position += duration
        if not packets:
            raise ValueError("HLS input contains no packets")
    if destination.stat().st_size > maximum_bytes:
        raise ValueError("remux output exceeds configured size limit")


if __name__ == "__main__":
    try:
        remux(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    except Exception as exc:
        print(json.dumps({"error":type(exc).__name__, "message":str(exc)[:200]}))
        sys.exit(1)
