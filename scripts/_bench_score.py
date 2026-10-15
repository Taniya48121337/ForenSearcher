"""Score GroundLie360 / FMNV runs (2026-10-12).

    python scripts/_bench_score.py --dataset g360 --round g27a --n 20
    python scripts/_bench_score.py --dataset fmnv --round f27a --n 20 --no-extract     # fixed answer lines only

The answer's fixed lines (VERDICT / TYPES / ...) are parsed first. When they are missing, Sonnet (claude-cli, the same
judge as EVID) only extracts the conclusion the answer states; it does not judge correctness. A run without a final
answer, or with no stated verdict, counts as wrong. The answer used is the accepted FINISH or, failing that, the last
FINISH the framework rejected for formatting (same rule as the EVID score).

Writes experiments/bench/<dataset>/scores/<round>.jsonl and <round>.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import (BENCH, DIRS, EXTRACT_PROMPT, FMNV_CODES, G360_TYPES, LOADERS, binary_metrics,  # noqa: E402
                           parse_answer, span_f1, token_f1, type_metrics, video_index)

PAPER = {
    "g360": "GroundLie360 paper, zero-shot FakeMark (GPT-4o-mini + InternVL-8B) on all 2,044: Acc 62.90, Fake-P 67.81, "
            "Fake-R 52.81, F1 59.38; fake-type macro-F1 29.23; title-grounding F1 22.39. Supervised FakingRecipe "
            "(15% test split): Acc 76.14, F1 74.94.",
    "fmnv": "FMNV paper: supervised models only (see paper); no zero-shot agent baseline reported.",
}
EXTRACT_SCHEMA = {
    "g360": {"type": "object", "required": ["verdict"], "properties": {
        "verdict": {"type": "string", "enum": ["Real", "Fake", "unknown"]},
        "types": {"type": "array", "items": {"type": "string"}}, "title_words": {"type": "string"},
        "false_speech_s": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        "edited_s": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}},
    "fmnv": {"type": "object", "required": ["verdict", "type"], "properties": {
        "verdict": {"type": "string", "enum": ["Real", "Fake", "unknown"]},
        "type": {"type": "string", "enum": ["CD", "CE", "SV", "CA", "none"]}}},
}


def run_dir(dataset: str, sample_id: str, round_tag: str) -> Path:
    from foren_searcher.datagen.run_cases import _safe

    return BENCH / dataset / "runs" / f"{_safe(dataset)}_{_safe(sample_id)}_{_safe(round_tag)}_1"


def read_answer(path: Path) -> tuple[str, str]:
    from _evid_score import read_run

    info = read_run(path, [])
    if info["decision"] == "FINISH" and info["answer"].strip():
        return info["answer"], "finish"
    if info.get("rejected_finish_answer"):
        return info["rejected_finish_answer"], "rejected_finish"
    return "", info["status"]


async def extract(judge, dataset: str, answer: str) -> dict | None:
    try:
        value = await judge.judge(system_prompt=EXTRACT_PROMPT[dataset], payload={"ANSWER": answer[-6000:]},
                                  images=[], schema=EXTRACT_SCHEMA[dataset])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(value, dict):
        return None
    verdict = str(value.get("verdict") or "").strip().capitalize()
    pred = {"parsed": False, "extracted": True, "verdict": verdict if verdict in ("Real", "Fake") else None}
    if dataset == "g360":
        types = value.get("types") or []
        pred["types"] = [t for t in G360_TYPES if t in types]
        pred["title_words"] = str(value.get("title_words") or "")
        pred["false_speech_s"] = [list(map(float, x))[:2] for x in value.get("false_speech_s") or [] if isinstance(x, list) and len(x) >= 2]
        pred["edited_s"] = [list(map(float, x))[:2] for x in value.get("edited_s") or [] if isinstance(x, list) and len(x) >= 2]
    else:
        t = str(value.get("type") or "").upper()
        pred["type"] = t if t in FMNV_CODES else None
    return pred


def fps_of(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else None
    except Exception:  # noqa: BLE001
        return None


def edited_seconds(gt: dict, fps: float | None) -> list[list[float]]:
    if not fps:
        return []
    spans = []
    for flag, (a, b) in zip(gt.get("edited_scenes") or [], gt.get("scene_frames") or []):
        if flag:
            spans.append([(a - 1) / fps, b / fps])
    return spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--round", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--judge", default="claude-cli:sonnet")
    args = parser.parse_args(argv)
    ds = args.dataset
    ids = json.loads((BENCH / ds / f"selection_n{args.n}.json").read_text(encoding="utf-8"))
    samples = {s.sample_id: s for s in LOADERS[ds]()}
    videos = video_index(ds) if ds == "g360" else {}
    judge = None
    if not args.no_extract:
        from foren_searcher.datagen.verify.judges import build_judge

        kind, _, model = args.judge.partition(":")
        judge = build_judge(kind, model=model, cache_path=BENCH / ds / "scores" / "extract_cache.jsonl")

    rows = []
    for sid in ids:
        s = samples[sid]
        answer, source = read_answer(run_dir(ds, sid, args.round))
        pred = parse_answer(ds, answer) if answer else {"parsed": False, "verdict": None}
        if answer and not pred["parsed"] and judge is not None:
            pred = asyncio.run(extract(judge, ds, answer)) or pred
        row = {"sample_id": sid, "label": s.label, "gt_types": list(s.types), "answer_source": source,
               "pred_verdict": pred.get("verdict"), "parsed": pred.get("parsed"), "extracted": pred.get("extracted", False)}
        if ds == "g360":
            row["pred_types"] = pred.get("types") or []
            row["title_f1"] = token_f1(pred.get("title_words", ""), s.gt["title_spans"]) if s.label == "Fake" else None
            row["speech_f1"] = span_f1(pred.get("false_speech_s") or [], s.gt["false_speech_s"])
            row["edit_f1"] = span_f1(pred.get("edited_s") or [], edited_seconds(s.gt, fps_of(videos.get(sid))))
        else:
            row["pred_type"] = pred.get("type")
            row["gt_class"] = s.types[0] if s.types else "Real"
            row["pred_class"] = (pred.get("type") or "?") if pred.get("verdict") == "Fake" else (pred.get("verdict") or "?")
        row["answer"] = answer[-3000:]
        rows.append(row)

    out = BENCH / ds / "scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.round}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    b = binary_metrics(rows)
    lines = [f"# {DIRS[ds]} — round {args.round} ({len(rows)} samples)", "",
             f"Answers: {sum(r['answer_source'] == 'finish' for r in rows)} FINISH, "
             f"{sum(r['answer_source'] == 'rejected_finish' for r in rows)} rejected FINISH used, "
             f"{sum(not r['answer_source'] in ('finish', 'rejected_finish') for r in rows)} without an answer (counted wrong); "
             f"fixed lines parsed {sum(bool(r['parsed']) for r in rows)}, extracted by judge {sum(bool(r['extracted']) for r in rows)}.",
             "", "## Real / Fake", "",
             "| n | Acc | Fake P | Fake R | Fake F1 | Real F1 | macro-F1 | verdict given |", "|---|---|---|---|---|---|---|---|",
             f"| {b['n']} | {100 * b['acc']:.1f} | {100 * b['fake_p']:.1f} | {100 * b['fake_r']:.1f} | {100 * b['fake_f1']:.1f} | "
             f"{100 * b['real_f1']:.1f} | {100 * b['macro_f1']:.1f} | {100 * b['parsed']:.0f}% |", ""]
    if ds == "g360":
        tm = type_metrics(rows, G360_TYPES, "gt_types", "pred_types")
        lines += ["## Fake types (multi-label, all samples)", "", "| type | support | P | R | F1 |", "|---|---|---|---|---|"]
        for t, v in tm["per_type"].items():
            lines.append(f"| {t} | {v['support']} | {100 * v['p']:.1f} | {100 * v['r']:.1f} | {100 * v['f1']:.1f} |")
        lines.append(f"| **macro** | | | | **{100 * tm['macro_f1']:.1f}** |")
        for key, name in (("title_f1", "title words (token F1, Fake samples)"), ("speech_f1", "false-speech time (0.1 s F1)"),
                          ("edit_f1", "edited time (0.1 s F1)")):
            vals = [r[key] for r in rows if r[key] is not None]
            if vals:
                lines.append(f"- grounding, {name}: {100 * sum(vals) / len(vals):.1f} over {len(vals)} samples")
    else:
        classes = ("Real", *FMNV_CODES)
        correct = sum(1 for r in rows if r["pred_class"] == r["gt_class"])
        tm = type_metrics([{**r, "g": [r["gt_class"]], "p": [r["pred_class"]]} for r in rows], classes, "g", "p")
        lines += ["## 5 classes (Real, CD, CE, SV, CA)", "", f"Accuracy {100 * correct / len(rows):.1f}; "
                  f"macro-F1 {100 * tm['macro_f1']:.1f}", "", "| class | support | P | R | F1 |", "|---|---|---|---|---|"]
        for t, v in tm["per_type"].items():
            lines.append(f"| {t} | {v['support']} | {100 * v['p']:.1f} | {100 * v['r']:.1f} | {100 * v['f1']:.1f} |")
    lines += ["", "## Per sample", "", "| sample | label | GT types | predicted | types predicted |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sample_id']} | {r['label']} | {', '.join(r['gt_types']) or '—'} | {r['pred_verdict'] or '—'} | "
                     f"{', '.join(r.get('pred_types') or ([r['pred_type']] if r.get('pred_type') else [])) or '—'} |")
    lines += ["", "Reference: " + PAPER[ds]]
    (out / f"{args.round}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics = {"dataset": ds, "round": args.round, "n": len(rows), "binary": b}
    if ds == "g360":
        metrics["types_macro_f1"] = tm["macro_f1"]
        for key in ("title_f1", "speech_f1", "edit_f1"):
            vals = [r[key] for r in rows if r[key] is not None]
            metrics[key] = sum(vals) / len(vals) if vals else None
    else:
        metrics["class5_acc"] = correct / len(rows)
        metrics["class5_macro_f1"] = tm["macro_f1"]
    (out / f"{args.round}.metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"-> {out / (args.round + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
