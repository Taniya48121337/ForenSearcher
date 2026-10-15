"""GroundLie360 / FMNV -> ForenSearcher cases (2026-10-12).

    python scripts/_bench_build_cases.py --dataset g360 --n 20      # pilot: 10 Fake (spread over types) + 10 Real
    python scripts/_bench_build_cases.py --dataset fmnv --n 100

Only samples whose video file is on disk under plan-harness-1/datasets/<GroundLie360|FMNV>/ are eligible (searched
recursively by file stem = video_id). The sample is deterministic and nested: a larger --n keeps the smaller one's picks
as long as the same videos are on disk. Cases go to experiments/bench/<dataset>/cases/ (split=test); the list of chosen
IDs to experiments/bench/<dataset>/selection_n<N>.json. Labels stay out of the case text (reference_path only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH, DATASETS, DIRS, LOADERS, pick, task_text, video_index  # noqa: E402
from foren_searcher.datagen.cases import Case, Media, SourcePolicy  # noqa: E402

MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
               ".m4v": "video/mp4", ".avi": "video/x-msvideo", ".flv": "video/x-flv"}


def build(sample, video: Path, out_dir: Path) -> Case:
    folder = DATASETS / DIRS[sample.dataset]
    reference = folder / ("data.json" if sample.dataset == "fmnv" else
                          "data.csv" if (folder / "data.csv").is_file() else "data.csv.gz")
    text = task_text(sample)
    return Case(
        case_id=sample.case_id, dataset=sample.dataset, source_id=sample.sample_id, split="test",
        task_kind="claim_check", task_text=text, task_text_en=text,
        family_key=f"{sample.dataset}:{sample.sample_id}",
        media=(Media(ref="V0", path=Path(os.path.relpath(video, out_dir)).as_posix(),
                     media_type=MEDIA_TYPES.get(video.suffix.lower(), "video/mp4")),),
        source_policy=SourcePolicy(mode="unrestricted"),
        reference_path=Path(os.path.relpath(reference, out_dir)).as_posix(),
        notes=f"{DIRS[sample.dataset]} test sample; labels only in reference_path.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args(argv)
    samples = LOADERS[args.dataset]()
    videos = video_index(args.dataset)
    chosen = pick(args.dataset, samples, set(videos), args.n)
    out_dir = BENCH / args.dataset / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in chosen:
        case = build(sample, videos[sample.sample_id], out_dir)
        (out_dir / f"{args.dataset}_{sample.sample_id}.json").write_text(
            json.dumps(case.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    selection = BENCH / args.dataset / f"selection_n{args.n}.json"
    selection.write_text(json.dumps([s.sample_id for s in chosen], indent=1), encoding="utf-8")
    labels = {}
    for s in chosen:
        key = s.label + (f"/{s.types[0]}" if s.types else "")
        labels[key] = labels.get(key, 0) + 1
    print(f"{DIRS[args.dataset]}: {len(samples)} annotated, {len(videos)} video files found, "
          f"{len(set(videos) & {s.sample_id for s in samples})} matched; chose {len(chosen)}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(labels.items())))
    if len(chosen) < args.n:
        print(f"!! only {len(chosen)} of {args.n}: download more videos into {DATASETS / DIRS[args.dataset]}")
    print(f"cases -> {out_dir}; selection -> {selection}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
