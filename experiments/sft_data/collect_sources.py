"""Download official annotation files and record their public source addresses."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "averitec_train.json": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json",
    "averimatec_train.json": "https://huggingface.co/datasets/Rui4416/AVerImaTeC/resolve/main/train.json",
    "qvhighlights_train.jsonl": "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_train_release.jsonl",
    "qvhighlights_readme.md": "https://raw.githubusercontent.com/jayleicn/moment_detr/main/README.md",
    "true_readme.md": "https://raw.githubusercontent.com/MeiTaylor/TRUE-3MFact/main/TRUE_Dataset/TRUE_Dataset.md",
}


def main() -> None:
    folder = ROOT / "data" / "annotations"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        for name, url in SOURCES.items():
            path = folder / name
            if path.is_file():
                print(f"Already available: {name}", flush=True)
                continue
            response = client.get(url)
            response.raise_for_status()
            path.write_bytes(response.content)
            record = {"file": name, "url": url, "resolved_url": str(response.url),
                      "retrieved_at": datetime.now(timezone.utc).isoformat(), "bytes": len(response.content)}
            records.append(record)
            print(json.dumps(record), flush=True)
    with (folder / "sources.jsonl").open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
