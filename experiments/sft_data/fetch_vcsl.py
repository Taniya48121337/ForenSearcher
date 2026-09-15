"""Acquire the previously sampled VCSL videos through their published URLs."""
from __future__ import annotations

import json
import argparse
import csv
import random
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-pairs", type=int, default=0)
    parser.add_argument("--retry-transient", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "data" / "vcsl"
    samples = json.loads((folder / "samples.json").read_text())
    official = ROOT.parents[1] / "external/VCSL/data"
    labels = json.loads((official / "label_file_uuid_total.json").read_text())
    if args.extra_pairs:
        frames = {x["uuid"]: int(x["frame_count"]) for x in csv.DictReader((official / "frames_all.csv").open())}
        urls = {x["uuid"]: x["url"] for x in csv.DictReader((official / "videos_url_uuid.csv").open())}
        candidates = [x for x in csv.DictReader((official / "pair_file_train.csv").open())
                      if frames.get(x["query_id"], 99999) <= 90 and frames.get(x["reference_id"], 99999) <= 90]
        random.Random(20260912).shuffle(candidates)
        seen = {x[k] for x in samples for k in ("query_id", "reference_id")}
        added = 0
        for row in candidates:
            ids = [row["query_id"], row["reference_id"]]
            if any(x in seen for x in ids):
                continue
            samples.append({**row, "urls": [urls[x] for x in ids], "seconds_approx": [frames[x] for x in ids]})
            seen.update(ids)
            added += 1
            if added >= args.extra_pairs:
                break
        (folder / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    previous = [json.loads(line) for line in (folder / "acquisition.jsonl").read_text().splitlines()] if (folder / "acquisition.jsonl").exists() else []
    attempted = {(x["case"], x["input"]) for x in previous}
    if args.retry_transient:
        latest = {(x["case"], x["input"]): x for x in previous}
        attempted = {key for key, record in latest.items()
                     if record["status"] == "downloaded" or "bytes read" not in record.get("error", "")}
    results = []
    for i, sample in enumerate(samples):
        case = folder / f"pair_{i + 1}"
        inputs = case / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        key = sample["query_id"] + "-" + sample["reference_id"]
        reference = {**sample, "dataset": "VCSL", "split": "train", "usage": "development",
                     "selection": "seed_20260912_nonoverlapping_pairs_both_90s_or_less",
                     "annotation": labels.get(key), "condition": "provided_pair_video_only"}
        (case / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        for name, url in zip(["A", "B"], sample["urls"]):
            if (i + 1, name) in attempted:
                continue
            options = {"format": "bestvideo[ext=mp4]/best[ext=mp4]/best", "outtmpl": str(inputs / (name + ".%(ext)s")),
                       "socket_timeout": 25, "retries": 1, "fragment_retries": 1, "noplaylist": True,
                       "max_filesize": 160 * 1024**2, "quiet": True, "noprogress": True, "no_warnings": True}
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(url, download=True)
                record = {"case": i + 1, "input": name, "status": "downloaded", "url": url,
                          "duration": info.get("duration"), "format_id": info.get("format_id"),
                          "files": [str(p) for p in inputs.glob(name + ".*")]}
            except Exception as exc:
                record = {"case": i + 1, "input": name, "status": "unavailable", "url": url,
                          "error": str(exc)[:700]}
            results.append(record)
            with (folder / "acquisition.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
