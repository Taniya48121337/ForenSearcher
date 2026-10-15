"""diff_media 对照小实验的取证指标（2026-10-04）：同一批 videosham 案例，hi@0.4.1（无 diff_media）vs hi@0.4.2（开 diff_media）。

只读 run 目录与案卷，不调模型。输出：
- 每个 run 的 HI 步数、INSPECT 各操作次数（diff_media 用了几次）、报告状态；
- 对取景一致、时长相同的成对视频（4126、1028、2010），用 diff_media 同一套像素差异算法算出"标准差异段"，
  再从最终答案里抽出"a–b 秒"区间，逐个标准段报起止边界差几帧（找不到对应区间记"未给出"）。

    python scripts/_diff_media_ab.py --rounds r0dv1 r0dv2 --out experiments/sft_data/diff_media_ab.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from foren_searcher.datagen.runlog import load_run
from foren_searcher.inspect import media_diff as M

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:秒)?\s*(?:–|—|-|~|～|至|到)\s*(\d+(?:\.\d+)?)\s*秒")
# 10-12：英文答案（hi@0.7.0 起）的 "2.0–4.04 s" / "from 2.0 s to 4.04 s"
_RANGE_EN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:–|—|-|~|to|until|through)\s*(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|seconds)")
_BRACKET = re.compile(r"[\[［]\s*(\d+(?:\.\d+)?)\s*[,，]\s*(\d+(?:\.\d+)?)\s*[)）\]］]")
# "从 2.333 秒附近延伸至素材结束" 这类开区间：终点取素材时长
_TO_END = re.compile(r"(?:从|自)\s*(\d+(?:\.\d+)?)\s*秒[^。；;\n]{0,12}?(?:至|到)(?:素材|视频|片段|画面)?(?:结束|末尾|结尾|最后)")


def case_media(case_id: str) -> list[Path]:
    case = json.loads((SFT / "cases" / (case_id.replace(":", "_") + ".json")).read_text(encoding="utf-8"))
    return [(SFT / "cases" / m["path"]).resolve() for m in case.get("media") or [] if str(m.get("media_type", "")).startswith("video")]


def _frames(path: Path) -> tuple[list[Any], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out, fps


_TRUTH: dict[str, dict[str, Any] | None] = {}


def truth(case_id: str) -> dict[str, Any] | None:
    """恒等时间映射下的逐帧差异段；两段帧数或帧率不一致（需要先对齐）时返回 None。"""

    if case_id in _TRUTH:
        return _TRUTH[case_id]
    media = case_media(case_id)
    result = None
    if len(media) == 2 and all(p.exists() for p in media):
        (a, fa), (b, fb) = _frames(media[0]), _frames(media[1])
        if a and b and abs(fa - fb) < 0.01 and abs(len(a) - len(b)) <= 1:
            n = min(len(a), len(b))
            size = M.work_size(a[0].shape[1], a[0].shape[0])
            A = [M.blurred(M.resize(x, size)) for x in a[:n]]
            B = [M.blurred(M.resize(x, size)) for x in b[:n]]
            comp = M.region_mask(size, None)
            th, _ = M.estimate_threshold([(M.difference(A[i], B[i]), comp) for i in range(0, n, max(1, n // 12))])
            area = [M.measure(M.changed_mask(M.difference(A[i], B[i]), comp, th), int(comp.sum()))[0] for i in range(n)]
            segs, start = [], None
            for i, v in enumerate(area + [0.0]):
                if v >= 0.0005 and start is None:
                    start = i
                elif v < 0.0005 and start is not None:
                    segs.append((start, i))
                    start = None
            result = {"fps": fa, "frames": n, "threshold": th, "segments": [(s, e, round(s / fa, 3), round(e / fa, 3)) for s, e in segs]}
    _TRUTH[case_id] = result
    return result


def summarize(run_dir: Path) -> dict[str, Any]:
    rl = load_run(run_dir)
    try:
        ops: Counter[str] = Counter()
        hi_steps = 0
        statuses: Counter[str] = Counter()
        answer = ""
        for d in rl.decisions:
            if d.role == "HI" and d.kind == "step":
                hi_steps += 1
                if d.action == "INSPECT":
                    ops[str(((d.output.get("arguments") or {}).get("request") or {}).get("operation"))] += 1
            elif d.role == "HI" and d.kind == "report":
                statuses[str(d.output.get("status"))] += 1
            elif d.kind == "answer":
                answer = str(d.output.get("answer") or d.output.get("final_answer") or json.dumps(d.output, ensure_ascii=False))
        return {"run": rl.name, "case_id": rl.case_id, "hi_steps": hi_steps, "ops": dict(ops), "statuses": dict(statuses), "answer": answer}
    finally:
        rl.close()


def boundary_errors(answer: str, t: dict[str, Any]) -> list[str]:
    end = t["frames"] / t["fps"]
    ranges = [(float(a), float(b)) for a, b in _RANGE.findall(answer) + _RANGE_EN.findall(answer) + _BRACKET.findall(answer)]
    ranges += [(float(a), end) for a in _TO_END.findall(answer)]
    rows = []
    for k, (s, e, ss, es) in enumerate(t["segments"], 1):
        best = min(ranges, key=lambda r: abs(r[0] - ss) + abs(r[1] - es), default=None)
        if best is None or abs(best[0] - ss) > 1.0:
            rows.append(f"段{k} {ss:.3f}–{es:.3f}s：答案未给出")
            continue
        ds, de = round((best[0] - ss) * t["fps"]), round((best[1] - es) * t["fps"])
        rows.append(f"段{k} {ss:.3f}–{es:.3f}s：答案 {best[0]:g}–{best[1]:g}s（起点差 {ds:+d} 帧，终点差 {de:+d} 帧）")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", nargs="+", required=True)
    p.add_argument("--runs", default=str(SFT / "runs"))
    p.add_argument("--prefix", default="videosham_", help="只看这个前缀的 run（默认 videosham_）")
    p.add_argument("--out", default=str(SFT / "diff_media_ab.md"))
    args = p.parse_args()
    lines = ["# diff_media 对照：取证指标（不依赖核验器）", ""]
    for rnd in args.rounds:
        runs = sorted(d for d in Path(args.runs).glob(f"{args.prefix}*_{rnd}_*")
                      if d.is_dir() and ".failed" not in d.name and (d / "trajectory.jsonl").is_file())
        lines += [f"## 轮次 {rnd}（{len(runs)} 个 run）", "", "| run | HI 步数 | INSPECT 操作 | 报告状态 |", "|---|---|---|---|"]
        details = []
        for d in runs:
            try:
                s = summarize(d)
            except Exception as exc:  # 单个 run 读不了不影响整张表
                lines.append(f"| {d.name} | 读取失败：{type(exc).__name__} | — | — |")
                continue
            ops = "；".join(f"{k} {v}" for k, v in sorted(s["ops"].items(), key=lambda x: -x[1])) or "—"
            lines.append(f"| {s['run']} | {s['hi_steps']} | {ops} | {s['statuses']} |")
            t = truth(s["case_id"]) if s["case_id"] else None
            if t:
                segs = "，".join(f"{ss:.3f}–{es:.3f}s（帧 {a}–{b - 1}）" for a, b, ss, es in t["segments"]) or "无差异"
                whole = len(t["segments"]) == 1 and t["segments"][0][0] == 0 and t["segments"][0][1] >= t["frames"] - 1
                note = "（整段都有差异：取景/画面整体不同，区间边界不是这题的考点）" if whole else ""
                details.append(f"- **{s['run']}**：标准差异段（阈值 {t['threshold']}）{segs}{note}")
                details += [f"  - {r}" for r in boundary_errors(s["answer"], t)]
        lines += ["", "区间准确度（仅取景一致、时长相同的成对视频）：", "", *(details or ["- 无可比的成对视频"]), ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
