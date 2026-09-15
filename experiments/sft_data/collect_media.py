"""Extract selected public dataset members without downloading complete archives."""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
TRUE_URL = "https://whueducn-my.sharepoint.com/:u:/g/personal/kaipengniu_whu_edu_cn/EcMBGGGA7VVHsTwLui4gJMYBZDH-p3A8GJiXHamIiVTkoA?download=1"
IMAGE_URL = "https://huggingface.co/datasets/Rui4416/AVerImaTeC/resolve/main/images.zip"


class RangeReader(io.RawIOBase):
    def __init__(self, url: str):
        self.client = httpx.Client(timeout=90, follow_redirects=True)
        response = self.client.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError("Archive does not support byte ranges")
        self.url = str(response.url)
        self.size = int(response.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.transferred = len(response.content)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        self.pos = offset + (self.pos if whence == os.SEEK_CUR else self.size if whence == os.SEEK_END else 0)
        if self.pos < 0:
            raise ValueError("Negative seek")
        return self.pos

    def read(self, size=-1):
        if self.pos >= self.size or size == 0:
            return b""
        end = self.size - 1 if size < 0 else min(self.size - 1, self.pos + size - 1)
        response = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}", "Accept-Encoding": "identity"})
        response.raise_for_status()
        expected_range = f"bytes {self.pos}-{end}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
            raise RuntimeError(f"Unexpected archive range: {response.status_code} {response.headers.get('Content-Range')}")
        content = response.content
        if len(content) != end - self.pos + 1:
            raise RuntimeError("Incomplete range response")
        self.pos = end + 1
        self.transferred += len(content)
        return content

    def close(self):
        self.client.close()
        super().close()


def write_member(archive, member, destination):
    if destination.exists():
        return
    info = archive.getinfo(member)
    if info.file_size > 160 * 1024**2:
        raise RuntimeError(f"Sample exceeds 160 MiB: {member}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = archive.read(member)
    destination.write_bytes(data)
    print(json.dumps({"member": member, "file": str(destination), "bytes": len(data)}), flush=True)


def collect_true(count):
    folder = ROOT / "data" / "true"
    folder.mkdir(parents=True, exist_ok=True)
    with RangeReader(TRUE_URL) as reader, zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        (folder / "archive_members.json").write_text(json.dumps(names), encoding="utf-8")
        roster = next(n for n in names if n.endswith("train_val_set.txt") and not n.startswith("__MACOSX"))
        ids = archive.read(roster).decode("utf-8-sig").split()
        chosen = random.Random(20260912).sample(sorted(ids), count)
        print(json.dumps({"dataset": "TRUE", "train_count": len(ids), "selected": chosen, "archive_bytes": reader.size}), flush=True)
        records = []
        for cid in chosen:
            annotation = next(n for n in names if "/train_val/" in "/" + n and n.endswith("/" + cid + ".json") and "__MACOSX" not in n)
            media = next(n for n in names if "/train_val_video/" in "/" + n and n.endswith("/" + cid + ".mp4") and "__MACOSX" not in n)
            write_member(archive, annotation, folder / cid / "reference.json")
            write_member(archive, media, folder / cid / "inputs" / "A.mp4")
            records.append({"case_id": cid, "annotation_member": annotation, "video_member": media,
                            "dataset_url": TRUE_URL, "split": "train_val", "usage": "development"})
        (folder / "samples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps({"transferred_bytes": reader.transferred}), flush=True)


def collect_images(count):
    rows = json.loads((ROOT / "data/annotations/averimatec_train.json").read_text(encoding="utf-8"))
    indices = random.Random(20260912).sample(range(len(rows)), count)
    folder = ROOT / "data" / "averimatec"
    folder.mkdir(parents=True, exist_ok=True)
    with RangeReader(IMAGE_URL) as reader, zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        records = []
        for index in indices:
            row = rows[index]
            case = folder / str(index)
            case.mkdir(exist_ok=True)
            (case / "reference.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            for number, filename in enumerate(row["claim_images"]):
                member = next(n for n in names if n == filename or n.endswith("/" + filename))
                write_member(archive, member, case / "inputs" / f"image_{number}.jpg")
            records.append({"row_index": index, "split": "train", "usage": "development", "archive_url": IMAGE_URL})
        (folder / "samples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps({"dataset": "AVerImaTeC", "selected": indices, "transferred_bytes": reader.transferred}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["true", "averimatec"])
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    (collect_true if args.dataset == "true" else collect_images)(args.count)
