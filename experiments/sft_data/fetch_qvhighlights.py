"""Stream the official archive until three distinct training videos are acquired."""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
URL = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"


class StreamReader:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.buffer = bytearray()
        self.received = 0

    def read(self, size):
        while len(self.buffer) < size:
            try:
                chunk = next(self.chunks)
            except StopIteration:
                break
            self.received += len(chunk)
            if self.received > 1024**3:
                raise RuntimeError("Archive prefix exceeded 1 GiB; retain obtained samples and reassess acquisition")
            self.buffer.extend(chunk)
        output = bytes(self.buffer[:size])
        del self.buffer[:size]
        return output


def main(validation=False):
    rows = [json.loads(line) for line in (ROOT / "data/annotations/qvhighlights_train.jsonl").read_text().splitlines()]
    by_video = {}
    for row in rows:
        by_video.setdefault(row["vid"], []).append(row)
    folder = ROOT / "data/qvhighlights"
    folder.mkdir(parents=True, exist_ok=True)
    samples, seen = [], set()
    if validation:
        if (folder / "validation_samples.json").exists():
            print((folder / "validation_samples.json").read_text(encoding="utf-8"))
            return
        existing = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
        seen.update(item["vid"].rsplit("_", 2)[0] for item in existing)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        with client.stream("GET", URL) as response:
            response.raise_for_status()
            reader = StreamReader(response.iter_bytes(65536))
            with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                for member in archive:
                    vid = Path(member.name).stem
                    source_id = vid.rsplit("_", 2)[0]
                    if not member.isfile() or vid not in by_video or source_id in seen:
                        continue
                    if member.size > 160 * 1024**2:
                        continue
                    row = sorted(by_video[vid], key=lambda x: x["qid"])[0]
                    case = folder / str(row["qid"])
                    inputs = case / "inputs"
                    inputs.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    with (inputs / "A.mp4").open("wb") as output:
                        while data := stream.read(1024**2):
                            output.write(data)
                    (case / "reference.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                    record = {"qid": row["qid"], "vid": vid, "member": member.name, "url": URL,
                              "split": "train", "usage": "construction_validation" if validation else "development", "bytes": member.size,
                              "selection": "archive_order_new_training_source" if validation else "archive_order_three_distinct_training_source_videos"}
                    samples.append(record)
                    seen.add(source_id)
                    (folder / ("validation_samples.json" if validation else "samples.json")).write_text(json.dumps(samples, indent=2), encoding="utf-8")
                    print(json.dumps(record), flush=True)
                    if len(samples) == (1 if validation else 3):
                        break
            print(json.dumps({"downloaded_archive_prefix_bytes": reader.received, "samples": len(samples)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="store_true")
    main(parser.parse_args().validation)
