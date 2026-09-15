"""Compute temporal overlap from explicitly transcribed report intervals."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def temporal_iou(prediction, reference):
    start, end = map(float, prediction)
    gold_start, gold_end = map(float, reference)
    if end <= start or gold_end <= gold_start:
        raise ValueError("Intervals must have positive duration")
    intersection = max(0.0, min(end, gold_end) - max(start, gold_start))
    return intersection / (end - start + gold_end - gold_start - intersection)


def main():
    gold = json.loads((ROOT / "data/qvhighlights/0/reference.json").read_text(encoding="utf-8"))["relevant_windows"]
    predictions = [
        {"run": "qvhighlights_0_r1", "request": "request_0012", "reported_interval_s": [45, 51],
         "interpretation": "literal_approximate_interval_from_report", "endpoint_bounds": None},
        {"run": "qvhighlights_0_r2", "request": "request_0012", "reported_interval_s": [38, 51.5],
         "interpretation": "reported_envelope_includes_uncertain_start", "endpoint_bounds": {"start": [38, 41], "end": [51, 51.48]}},
    ]
    for item in predictions:
        source = ROOT / "runs" / item["run"] / "requests" / item["request"] / "response.json"
        item["report_text"] = source.read_text(encoding="utf-8")
        item["best_temporal_iou"] = max(temporal_iou(item["reported_interval_s"], window) for window in gold)
        if item["endpoint_bounds"]:
            values = [max(temporal_iou([s, e], window) for window in gold)
                      for s in item["endpoint_bounds"]["start"] for e in item["endpoint_bounds"]["end"]]
            item["iou_range_from_reported_endpoint_bounds"] = [min(values), max(values)]
    result = {"dataset": "QVHighlights", "qid": 0, "gold_intervals_s": gold, "predictions": predictions,
              "scope": "same_development_case_descriptive_comparison; not aggregate mAP or proof of semantic correctness",
              "prediction_extraction": "curator_literal_transcription_of_report; no hidden annotation entered actor input"}
    (ROOT / "temporal_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in item.items() if k != "report_text"} for item in predictions]))


if __name__ == "__main__":
    main()
