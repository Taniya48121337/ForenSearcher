"""Extract only the deterministic VCSL canary features from the local archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np

from pipeline import ROOT, write_json
from vcsl_probe import select_cases


DEFAULT_ARCHIVE = ROOT.parent / "external" / "VCSL" / "vcsl_isc_features_download"
DEFAULT_OUTPUT = ROOT.parent / "external" / "VCSL" / "features_isc_30case" / "eff256d"
DEFAULT_REPORT = ROOT / "outputs" / "scale_readiness" / "vcsl_30case_feature_extraction.json"


def target_ids(seed: int, sample_size: int) -> set[str]:
    return {
        case[key]
        for case in select_cases(seed, sample_size)
        for key in ("query_id", "reference_id")
    }


def extract(archive: Path, output: Path, ids: set[str]) -> dict:
    archive = archive.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            Path(member.name).name.removesuffix(".npy"): member
            for member in handle.getmembers()
            if member.isfile() and member.name.endswith(".npy")
        }
        missing = sorted(ids - set(members))
        if missing:
            raise FileNotFoundError(f"target VCSL features absent from archive: {missing}")
        for video_id in sorted(ids):
            destination = output / f"{video_id}.npy"
            if not destination.exists():
                source = handle.extractfile(members[video_id])
                if source is None:
                    raise OSError(f"cannot read archive member for {video_id}")
                destination.write_bytes(source.read())
            extracted.append(video_id)

    invalid = []
    shapes = {}
    for video_id in extracted:
        path = output / f"{video_id}.npy"
        array = np.load(path, allow_pickle=False)
        shapes[video_id] = list(array.shape)
        if array.ndim != 2 or array.shape[1] != 256 or not np.isfinite(array).all():
            invalid.append(video_id)
    return {
        "archive": str(archive),
        "output": str(output),
        "target_feature_count": len(ids),
        "extracted_or_already_present": len(extracted),
        "invalid_feature_ids": invalid,
        "all_target_features_valid": len(extracted) == len(ids) and not invalid,
        "shape_examples": dict(list(shapes.items())[:5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()
    report = extract(args.archive, args.output, target_ids(args.seed, args.sample_size))
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
