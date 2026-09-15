"""Match retained decision IDs to actual processor checks without counting removed rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(corpus):
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    checked, batches = {}, []
    for folder in sorted(ROOT.glob("student_inputs_*")):
        summary_path, row_path = folder / "summary.json", folder / "rows.jsonl"
        if not summary_path.is_file() or not row_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("truncation") is not False:
            continue
        model = summary["model"]
        batches.append({"summary": summary_path.relative_to(ROOT).as_posix(), "model": model})
        for line in row_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") != "processed" or not row.get("target_text_exact") or not row.get("source_tokens_masked"):
                continue
            key = (row["file"], row["run"], row["request"])
            checked.setdefault(model, {})[key] = row
    retained = []
    for entry in manifest["files"]:
        for line in (corpus / entry["file"]).read_text(encoding="utf-8").splitlines():
            metadata = json.loads(line)["metadata"]
            retained.append((entry["file"], metadata.get("run", metadata.get("construction")),
                             metadata.get("request", metadata.get("state"))))
    results = []
    for model, rows in sorted(checked.items()):
        covered = [rows[key] for key in retained if key in rows]
        missing = [dict(zip(("file", "run", "request"), key)) for key in retained if key not in rows]
        results.append({"model": model, "retained_rows": len(retained), "covered_rows": len(covered),
                        "missing": missing, "max_input_tokens": max((row["input_tokens"] for row in covered), default=0),
                        "above_context_tokens": {str(n): sum(row["input_tokens"] > n for row in covered)
                                                 for n in (16384, 32768, 65536)}})
    return {"models": results, "batches": batches,
            "scope": "Retained decision IDs matched to completed, untruncated processor checks. Assumes source request payloads and targets retain their recorded content; not a student training or GPU capacity measurement."}


if __name__ == "__main__":
    corpus = ROOT / "corpus"
    result = summarize(corpus)
    (corpus / "processing_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
