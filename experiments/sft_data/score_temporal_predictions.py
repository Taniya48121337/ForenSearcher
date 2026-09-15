"""Measure literal predictions without converting anchor spans into full intervals."""
import importlib.util
import json

import numpy as np

from temporal_metrics import ROOT, temporal_iou


def main():
    entries = json.loads((ROOT / "temporal_predictions.json").read_text(encoding="utf-8"))
    metric_path = ROOT.parents[1] / "external/VCSL/vcsl/metric.py"
    spec = importlib.util.spec_from_file_location("vcsl_official_metric", metric_path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    results = []
    for entry in entries:
        report = ROOT / "runs" / entry["run"] / "requests" / entry["request"] / "response.json"
        text = report.read_text(encoding="utf-8")
        if entry.get("excerpt") and entry["excerpt"] not in text:
            raise ValueError(f"Prediction excerpt does not match actual report: {entry['run']}")
        reference = json.loads((ROOT / "data" / entry["dataset"] / entry["case"] / "reference.json").read_text(encoding="utf-8"))
        row = {**entry, "report_path": str(report.relative_to(ROOT)), "prediction_source": "curator_transcription_of_actual_teacher_report"}
        if entry["dataset"] == "qvhighlights":
            gold = reference["relevant_windows"]
            row.update(reference_intervals_s=gold, best_temporal_iou=max(temporal_iou(entry["interval_s"], g) for g in gold))
        else:
            row.update(reference_segments=reference["annotation"],
                raw_coordinate_precision_recall=metric.precision_recall(np.asarray([entry["segment_xyxy"]], dtype=np.float32),
                                                                        np.asarray(reference["annotation"], dtype=np.float32)),
                metric_source="external/VCSL/vcsl/metric.py:precision_recall",
                interpretation_limit="Official metric function on original annotation coordinates; no independent original-to-downloaded timeline calibration. This is not a full-benchmark result.")
        results.append(row)
    (ROOT / "temporal_results.json").write_text(json.dumps({"results": results,
        "scope": "descriptive_case_metrics; numerical_overlap_does_not_prove_every_semantic_claim_or_continuity"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: row[k] for k in ("run", "best_temporal_iou", "raw_coordinate_precision_recall") if k in row} for row in results]))


if __name__ == "__main__":
    main()
