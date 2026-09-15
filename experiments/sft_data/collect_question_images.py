"""Fetch exactly the official image inputs referenced by selected local questions."""
import argparse
import json
import zipfile

from collect_media import IMAGE_URL, ROOT, RangeReader, write_member


def main(entries):
    with RangeReader(IMAGE_URL) as reader, zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        for entry in entries:
            cid, qi = entry.split(":")
            case = ROOT / "data/averimatec" / cid
            reference = json.loads((case / "reference.json").read_text(encoding="utf-8"))
            question = reference["questions"][int(qi)]
            target = case / "question_inputs" / ("q" + qi)
            target.mkdir(parents=True, exist_ok=True)
            mapping = []
            for i, filename in enumerate(question.get("input_images", [])):
                member = next(n for n in names if n == filename or n.endswith("/" + filename))
                destination = target / f"image_{i}.jpg"
                write_member(archive, member, destination)
                mapping.append({"local_file": destination.name, "official_input_image": filename,
                                "archive_member": member})
            (target / "mapping.json").write_text(json.dumps({"case": cid, "question_index": int(qi),
                "mapping": mapping, "source_url": IMAGE_URL}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"case": cid, "question": int(qi), "images": len(mapping)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", nargs="+")
    main(parser.parse_args().entries)
