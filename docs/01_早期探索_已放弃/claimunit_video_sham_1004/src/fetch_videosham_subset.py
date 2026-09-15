"""Select and range-fetch a balanced VideoSham subset from official release ZIPs.

Fetching is opt-in because the official Adobe Research License limits use to
noncommercial research.  The HTTP reader is seekable, allowing ``zipfile`` to
read only the central directory and selected members instead of downloading all
four multi-gigabyte archives.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import av

from pipeline import ROOT, write_json
from videosham_batch import (
    EDITED_ROOT,
    SOURCE_ROOT,
    _case_number,
    _rows,
    _time_window,
    capability_family,
    preflight_case,
)


RELEASE_BASE = "https://github.com/adobe-research/VideoSham-dataset/releases/download/v1.0"
ARCHIVES = {
    "source_part1": f"{RELEASE_BASE}/Unedited-Part1.zip",
    "source_part2": f"{RELEASE_BASE}/Unedited-Part2.zip",
    "edited_part1": f"{RELEASE_BASE}/Processed-Part1.zip",
    "edited_part2": f"{RELEASE_BASE}/Processed-Part2.zip",
}
DEFAULT_OUTPUT = ROOT / "outputs" / "scale_readiness" / "videosham_subset_acquisition.json"
TARGET_FINAL_COUNTS = {
    "temporal_omission": 3,
    "temporal_warp": 5,
    "audio_replacement": 4,
    "spatial_removal": 4,
    "spatial_insertion": 4,
    "appearance_change": 5,
    "text_edit": 5,
}


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, *, timeout_s: int = 60) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.position = 0
        self.length = self._discover_length()
        self.bytes_transferred = 0
        self.request_count = 0

    def _request(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "ForenSearcher-VideoSham-Research-Subset/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range", "")
            if status != 206 or not content_range.startswith(f"bytes {start}-"):
                raise OSError(f"release server did not honor byte range {start}-{end}: status={status}")
            payload = response.read()
        self.request_count += 1
        self.bytes_transferred += len(payload)
        return payload

    def _discover_length(self) -> int:
        request = urllib.request.Request(
            self.url,
            headers={"Range": "bytes=0-0", "User-Agent": "ForenSearcher-VideoSham-Research-Subset/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            status = getattr(response, "status", response.getcode())
            content_range = response.headers.get("Content-Range", "")
            if status != 206 or "/" not in content_range:
                raise OSError(f"release server does not expose a range-readable ZIP: status={status}")
            return int(content_range.rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"unknown seek mode: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        end = min(self.position + size, self.length) - 1
        if end < self.position:
            return b""
        payload = self._request(self.position, end)
        self.position += len(payload)
        return payload


def select_balanced_missing_rows(seed: int = 20260820) -> list[dict[str, str]]:
    rows = _rows()
    existing = {
        number
        for row in rows
        if (number := _case_number(row))
        and (SOURCE_ROOT / f"{number}_unedited.mp4").exists()
        and (EDITED_ROOT / f"{number}_processed.mp4").exists()
    }
    existing_counts = Counter(
        capability_family(row)
        for row in rows
        if _case_number(row) in existing
    )
    candidates: dict[str, list[dict[str, str]]] = {family: [] for family in TARGET_FINAL_COUNTS}
    for row in rows:
        number = _case_number(row)
        family = capability_family(row)
        if (
            number
            and number not in existing
            and family in candidates
            and _time_window(row["Task"]) is not None
        ):
            candidates[family].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for family, target in TARGET_FINAL_COUNTS.items():
        need = max(0, target - existing_counts.get(family, 0))
        values = list(candidates[family])
        rng.shuffle(values)
        if len(values) < need:
            raise ValueError(f"VideoSham has only {len(values)} parseable missing rows for {family}, need {need}")
        selected.extend(values[:need])
    return selected


def _member_case_number(name: str) -> str | None:
    basename = Path(name).name
    match = re.match(r"(?P<number>\d{1,4})_(?:unedited|processed)\.mp4$", basename, re.IGNORECASE)
    return match.group("number").zfill(4) if match else None


def validate_video_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(Path(path).relative_to(ROOT)), "exists": path.exists()}
    if not path.exists():
        return {**result, "valid": False, "reason": "missing"}
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return {**result, "valid": False, "reason": "no_video_stream"}
            stream = container.streams.video[0]
            frame = next(container.decode(video=0), None)
            duration_s = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else float(container.duration / av.time_base)
                if container.duration is not None
                else None
            )
            result.update(
                {
                    "bytes": path.stat().st_size,
                    "video_codec": stream.codec_context.name,
                    "width": int(stream.codec_context.width),
                    "height": int(stream.codec_context.height),
                    "duration_s": round(duration_s, 3) if duration_s is not None else None,
                    "audio_stream_present": bool(container.streams.audio),
                    "first_frame_decoded": frame is not None,
                }
            )
            result["valid"] = bool(
                result["bytes"] > 0
                and result["width"] > 0
                and result["height"] > 0
                and result["first_frame_decoded"]
                and (duration_s is None or duration_s > 0.0)
            )
            return result
    except Exception as error:
        return {
            **result,
            "valid": False,
            "reason": "decode_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _index_archives(timeout_s: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    members: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    for archive_name, url in ARCHIVES.items():
        reader = HTTPRangeReader(url, timeout_s=timeout_s)
        with zipfile.ZipFile(reader) as archive:
            for info in archive.infolist():
                number = _member_case_number(info.filename)
                if number is None:
                    continue
                kind = "source" if "unedited" in Path(info.filename).name.lower() else "edited"
                key = f"{kind}:{number}"
                if key in members:
                    raise ValueError(f"duplicate release member for {key}")
                members[key] = {
                    "archive_name": archive_name,
                    "url": url,
                    "member_name": info.filename,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "crc32": info.CRC,
                }
        telemetry[archive_name] = {
            "archive_size": reader.length,
            "index_bytes_transferred": reader.bytes_transferred,
            "index_requests": reader.request_count,
        }
    return members, telemetry


def _extract_member(
    member: dict[str, Any],
    destination: Path,
    *,
    timeout_s: int,
) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 0:
        return {"status": "already_present", "bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    reader = HTTPRangeReader(member["url"], timeout_s=timeout_s)
    try:
        with zipfile.ZipFile(reader) as archive, archive.open(member["member_name"]) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if partial.stat().st_size != int(member["uncompressed_size"]):
            raise OSError(f"extracted size mismatch for {member['member_name']}")
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "status": "downloaded",
        "bytes": destination.stat().st_size,
        "range_bytes_transferred": reader.bytes_transferred,
        "range_requests": reader.request_count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = select_balanced_missing_rows(args.seed)
    selection = [
        {
            "case_number": _case_number(row),
            "video_name": row["Video-Name"],
            "family": capability_family(row),
            "task": row["Task"],
        }
        for row in selected
    ]
    report: dict[str, Any] = {
        "source": "official_adobe_research_videosham_release_v1.0",
        "license": "Adobe Research License Version 1.1; noncommercial research only",
        "seed": args.seed,
        "target_final_family_counts": TARGET_FINAL_COUNTS,
        "selected_new_cases": len(selection),
        "selection_family_counts": dict(Counter(item["family"] for item in selection)),
        "selection": selection,
        "fetch_requested": bool(args.fetch),
    }
    if not args.fetch:
        write_json(args.output, report)
        return report
    if not args.accept_adobe_research_license_noncommercial:
        raise PermissionError(
            "fetch requires --accept-adobe-research-license-noncommercial; downloading exercises rights under that license"
        )

    members, index_telemetry = _index_archives(args.timeout_s)
    missing_release_members: list[str] = []
    planned: list[tuple[dict[str, Any], Path, str, str]] = []
    for item in selection:
        number = item["case_number"]
        for kind, destination in (
            ("source", SOURCE_ROOT / f"{number}_unedited.mp4"),
            ("edited", EDITED_ROOT / f"{number}_processed.mp4"),
        ):
            key = f"{kind}:{number}"
            if key not in members:
                missing_release_members.append(key)
            else:
                planned.append((members[key], destination, kind, number))
    if missing_release_members:
        raise FileNotFoundError(f"selected cases are absent from official release archives: {missing_release_members}")

    fetched = []
    for member, destination, kind, number in planned:
        result = _extract_member(member, destination, timeout_s=args.timeout_s)
        fetched.append(
            {
                "case_number": number,
                "kind": kind,
                "destination": str(destination.relative_to(ROOT)),
                "release_member": member["member_name"],
                **result,
            }
        )
        report.update({"archive_index_telemetry": index_telemetry, "fetched": fetched})
        write_json(args.output, report)
    selected_by_number = {_case_number(row): row for row in selected}
    validations = []
    for number, row in selected_by_number.items():
        source_validation = validate_video_file(SOURCE_ROOT / f"{number}_unedited.mp4")
        edited_validation = validate_video_file(EDITED_ROOT / f"{number}_processed.mp4")
        preflight = preflight_case(row)
        validations.append(
            {
                "case_number": number,
                "family": capability_family(row),
                "source": source_validation,
                "edited": edited_validation,
                "tool_preflight_eligible": preflight["cli_eligible"],
                "tool_preflight_reasons": preflight["preflight_reasons"],
            }
        )
    report["media_validation"] = validations
    report["summary"] = {
        "downloaded_files": sum(item["status"] == "downloaded" for item in fetched),
        "already_present_files": sum(item["status"] == "already_present" for item in fetched),
        "downloaded_bytes": sum(item["bytes"] for item in fetched if item["status"] == "downloaded"),
        "valid_media_pairs": sum(
            item["source"]["valid"] and item["edited"]["valid"] for item in validations
        ),
        "tool_preflight_eligible_pairs": sum(item["tool_preflight_eligible"] for item in validations),
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--accept-adobe-research-license-noncommercial", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=90)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report.get("summary", {"selected_new_cases": report["selected_new_cases"], "selection_family_counts": report["selection_family_counts"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
