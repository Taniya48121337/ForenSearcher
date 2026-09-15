"""List or extract selected members from a remote ZIP using HTTP byte ranges.

This avoids downloading the complete 1.3+ GB VideoSham archives.  Only the ZIP
central directory and the compressed bytes of matching members are requested.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import urllib.request
import zipfile
from pathlib import Path


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            self.url = response.geturl()
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.pos + offset
        elif whence == os.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.pos = min(target, self.size)
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size is None or size < 0:
            end = self.size - 1
        else:
            end = min(self.size - 1, self.pos + size - 1)
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request) as response:
            payload = response.read()
            status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(f"server ignored byte range (HTTP {status})")
        expected = end - self.pos + 1
        if len(payload) != expected:
            raise RuntimeError(f"short range read: expected {expected}, got {len(payload)}")
        self.pos = end + 1
        return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--match", default=r"1004\.mp4$", help="case-insensitive regex")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    reader = HTTPRangeReader(args.url)
    pattern = re.compile(args.match, re.IGNORECASE)
    with zipfile.ZipFile(reader) as archive:
        matches = [item for item in archive.infolist() if pattern.search(item.filename)]
        if not matches:
            candidates = [item.filename for item in archive.infolist() if "1004" in item.filename]
            raise SystemExit(f"no match; 1004-like members: {candidates}")
        for item in matches:
            print(f"{item.filename}\t{item.file_size}\tcompressed={item.compress_size}")
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                target = args.output_dir / Path(item.filename).name
                with archive.open(item) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                print(f"extracted\t{target}")


if __name__ == "__main__":
    main()
