"""DETECT face_swap_scan 开发验证（10-10）：双击 scripts\\_detect_faceswap_dev.cmd。

不看原片时，换脸只能靠检测器。这里在开发集上量"按人聚合的 DFDC B7 分"能不能把被换脸的人和真人分开：
  被换的人  6 个换脸样本里，与原片同位置人脸身份相似度中位 < 0.45 的人（用 compare_source 标）
  真人      同样 6 个样本里身份没变的人；这 6 个样本的原片里的人；插入 / 改物体样本里的人；EVID 002–130 的 12 个真实视频
输出 outputs/detect/faceswap_dev/{persons.jsonl, summary.md, <视频>_<人>.jpg}
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _detect_dev_probe as probe  # noqa: E402

OUT = ROOT / "outputs" / "detect" / "faceswap_dev"
EXTRA_REAL = ["004", "012", "020", "030", "040", "050", "064", "076", "088", "098", "110", "124", "112", "026", "056", "018", "060"]


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%H:%M:%S')} {text}\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    probe.ensure_packages()
    import numpy as np
    import torch

    from foren_searcher.detect.compare import FaceIdentity, SourceComparer
    from foren_searcher.detect.faceswap import DFDCScorer, FaceSwapScanner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    faces = FaceIdentity(device)
    scanner = FaceSwapScanner(faces, DFDCScorer(probe.DF_MODELS / "dfdc", device))
    captured = {}

    class Capturing(SourceComparer):
        def _faces(self, samples):
            out = super()._faces(samples)
            captured["persons"] = out
            return out

    comparer = Capturing(faces)
    entries = {e["id"].removeprefix("video_"): e for e in json.loads((probe.DATASET / "evid_bench.json").read_text(encoding="utf-8"))}
    done_path = OUT / "persons.jsonl"
    done = set()
    if done_path.is_file():
        done = {json.loads(l)["video"] for l in done_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    jobs = []
    for s, kind in probe.DEV.items():
        jobs.append((f"video_{s}", kind, probe.VIDEOS / f"video_{s}.mp4", entries[s]["source"][0] if kind == "swap" else None))
        if kind == "swap":
            orig = probe.fetch_original(entries[s]["source"][0])
            if orig:
                jobs.append((f"orig_{s}", "original_of_swap", orig, None))
    jobs += [(f"video_{s}", "dev_real", probe.VIDEOS / f"video_{s}.mp4", None) for s in EXTRA_REAL]
    for name, kind, path, source in jobs:
        if name in done or not path.is_file():
            continue
        started = time.monotonic()
        try:
            result, images = scanner.scan(path)
            persons = result["persons"]
            labels = [None] * len(persons)
            if source:
                orig = probe.fetch_original(source)
                captured.clear()
                comparer.compare(path, orig)
                ref = captured.get("persons") or []
                # 用 compare 的人（有身份相似度）去对 scan 的人：两边都按 FaceNet 质心聚类，取质心最接近的
                cents = getattr(scanner, "_last_centroids", [])
                for k, c in enumerate(cents[:len(persons)]):
                    best = max(ref, key=lambda r: float(c @ r["_centroid"]), default=None)
                    if best is not None and float(c @ best["_centroid"]) >= 0.6:
                        labels[k] = {"median_similarity": best["median_similarity"], "swapped": best["identity_differs"]}
            for k, p in enumerate(persons):
                row = {"video": name, "kind": kind, "person": k + 1, **p,
                       "label": (labels[k] or {}).get("swapped") if source else (False if kind != "swap" else None),
                       "identity_similarity": (labels[k] or {}).get("median_similarity") if source else None}
                with done_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not persons:
                with done_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"video": name, "kind": kind, "person": None}) + "\n")
            for img in images[:4]:
                (OUT / f"{name}_{img['index'] + 1}.jpg").write_bytes(img["jpeg"])
            log(f"[{name}] {kind}：{len(persons)} 人，{time.monotonic() - started:.0f}s；" + " | ".join(result["summary_lines"][:4]))
        except Exception as exc:  # noqa: BLE001
            log(f"!!! {name} 失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")
    rows = [json.loads(l) for l in done_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("person")]
    pos = [r["median_score"] for r in rows if r["label"] is True]
    neg = [r["median_score"] for r in rows if r["label"] is False]
    auc = (sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))) if pos and neg else None
    lines = ["# face_swap_scan 开发验证", "", f"被换的人 {len(pos)}，真人 {len(neg)}，按人中位分 AUROC {auc}", "",
             "| 阈值 | 被换的人检出 | 真人误报 |", "|---|---|---|"]
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        lines.append(f"| {th} | {sum(p >= th for p in pos)}/{len(pos)} | {sum(q >= th for q in neg)}/{len(neg)} |")
    lines += ["", "| 视频 | 类别 | 人 | 帧 | 脸 px | 中位分 | 上四分位 | 身份相似度 | 标签 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['video']} | {r['kind']} | {r['person']} | {r['observations']} | {r['median_face_px']} | {r['median_score']} | "
                     f"{r['p75_score']} | {r.get('identity_similarity')} | {r['label']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
