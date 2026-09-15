"""Acquire separate source cases for checking the current construction procedure."""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

from collect_media import IMAGE_URL, TRUE_URL, ROOT, RangeReader, write_member

SEED = 20260913


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def collect(dataset, *, selection_name="validation_samples.json", seed=SEED):
    folder = ROOT / "data" / dataset
    selection_path = folder / selection_name
    if selection_path.exists():
        print(selection_path.read_text(encoding="utf-8"), flush=True)
        return
    rng = random.Random(seed)
    record = {"dataset": dataset, "usage": "construction_validation", "selection_seed": seed,
              "scope": "new_case_for_current_method_not_population_generalization_proof"}
    if dataset in {"averitec", "averimatec"}:
        rows = json.loads((ROOT / "data/annotations" / f"{dataset}_train.json").read_text(encoding="utf-8"))
        existing = {int(path.name) for path in folder.iterdir() if path.is_dir() and path.name.isdigit()}
        used_images = {image for index in existing for image in rows[index].get("claim_images", [])}
        used_articles = {rows[index].get("fact_checking_article", rows[index].get("article")) for index in existing}
        used_articles.discard(None)
        candidates = [i for i, row in enumerate(rows) if i not in existing
                      and row.get("questions")
                      and row.get("fact_checking_article", row.get("article")) not in used_articles
                      and not used_images.intersection(row.get("claim_images", []))]
        index = rng.choice(candidates)
        case = folder / str(index)
        save(case / "reference.json", rows[index])
        record.update(case=str(index), row_index=index, split="train")
        if dataset == "averimatec":
            with RangeReader(IMAGE_URL) as reader, zipfile.ZipFile(reader) as archive:
                names = archive.namelist()
                for number, filename in enumerate(rows[index]["claim_images"]):
                    member = next(n for n in names if n == filename or n.endswith("/" + filename))
                    write_member(archive, member, case / "inputs" / f"image_{number}.jpg")
                record.update(archive_url=IMAGE_URL, transferred_bytes=reader.transferred)
    elif dataset == "true":
        existing = {path.name for path in folder.iterdir() if path.is_dir()}
        with RangeReader(TRUE_URL) as reader, zipfile.ZipFile(reader) as archive:
            names = archive.namelist()
            roster = next(n for n in names if n.endswith("train_val_set.txt") and not n.startswith("__MACOSX"))
            ids = archive.read(roster).decode("utf-8-sig").split()
            cid = rng.choice(sorted(set(ids) - existing))
            annotation = next(n for n in names if "/train_val/" in "/" + n and n.endswith("/" + cid + ".json") and "__MACOSX" not in n)
            media = next(n for n in names if "/train_val_video/" in "/" + n and n.endswith("/" + cid + ".mp4") and "__MACOSX" not in n)
            write_member(archive, annotation, folder / cid / "reference.json")
            write_member(archive, media, folder / cid / "inputs/A.mp4")
            record.update(case=cid, annotation_member=annotation, video_member=media,
                          archive_url=TRUE_URL, transferred_bytes=reader.transferred, split="train_val")
    elif dataset == "videosham":
        source = ROOT.parents[1] / "claimunit_video_sham_1004/datasets/videosham"
        rows = list(csv.DictReader((source / "VideoSourceandEdits.csv").open(encoding="utf-8-sig")))
        development = {"1028", "0069", "2010"}
        source_key = "Name/Id in source dataset"
        used_sources = {row[source_key] for row in rows if Path(row["Video-Name"]).stem in development}
        candidates = [row for row in rows if row[source_key] not in used_sources
                      and (source / "media/source" / (Path(row["Video-Name"]).stem + "_unedited.mp4")).is_file()
                      and (source / "media/edited" / (Path(row["Video-Name"]).stem + "_processed.mp4")).is_file()]
        row = rng.choice(candidates)
        record.update(case=Path(row["Video-Name"]).stem, source_group=row[source_key], split="local_public_subset",
                      selection="available_pairs_excluding_current_development_source_groups",
                      limitation="Materials existed in an older project; this is not a never-used project test set.")
    elif dataset == "vcsl":
        references = {cid: json.loads((folder / cid / "reference.json").read_text(encoding="utf-8"))
                      for cid in ("pair_1", "pair_4", "pair_5", "pair_6")}
        used = {references[cid][key] for cid in ("pair_1", "pair_4", "pair_6") for key in ("query_id", "reference_id")}
        selected = references["pair_5"]
        if used.intersection({selected["query_id"], selected["reference_id"]}):
            raise ValueError("VCSL validation pair shares a development source")
        if not all((folder / "pair_5/inputs" / name).is_file() for name in ("A.mp4", "B.mp4")):
            raise FileNotFoundError("Validation pair is incomplete")
        record.update(case="pair_5", split="train", query_id=selected["query_id"], reference_id=selected["reference_id"],
                      selection="previously_acquired_unrun_pair_with_disjoint_video_ids")
    else:
        raise ValueError(dataset)
    save(selection_path, [record])
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=["averitec", "averimatec", "true", "videosham", "vcsl"])
    parser.add_argument("--selection-name", default="validation_samples.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if Path(args.selection_name).name != args.selection_name or not args.selection_name.endswith(".json"):
        parser.error("selection-name must be a JSON filename")
    for dataset in args.datasets:
        collect(dataset, selection_name=args.selection_name, seed=args.seed)
