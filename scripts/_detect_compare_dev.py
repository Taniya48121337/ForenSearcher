"""DETECT compare_source 离线验收（10-10）：双击 scripts\\_detect_compare_dev.cmd。

对开发集 18 个样本（与 _detect_dev_probe 相同），拿 GT 原片当 B，跑 detect/compare.py 的 SourceComparer，
把 summary_lines 和 GT 并排写进 outputs/detect/compare_dev/summary.md，证据图存成 <样本>_<序号>_<类型>.jpg。
这一步只验"拿到原片之后工具能不能把三类改动直接说清楚"，不跑 agent。
  --only 201 213   只跑指定样本      --no-faces   不做人脸身份比对
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _detect_dev_probe as probe  # noqa: E402

OUT = ROOT / "outputs" / "detect" / "compare_dev"


def log(text: str) -> None:
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "log.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%H:%M:%S')} {text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-faces", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    probe.ensure_packages()
    import torch

    from foren_searcher.detect.compare import FaceIdentity, SourceComparer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    comparer = SourceComparer(None if args.no_faces else FaceIdentity(device))
    entries = {e["id"].removeprefix("video_"): e for e in json.loads((probe.DATASET / "evid_bench.json").read_text(encoding="utf-8"))}
    lines = ["# compare_source 在开发集上的输出（B = GT 原片）", ""]
    for sample in (args.only or list(probe.DEV)):
        e = entries[sample]
        source = e["source"][0]
        b = probe.fetch_original(source)
        lines += [f"## {sample}（{probe.DEV[sample]}，原片 {source}）", "", "GT："]
        lines += [f"- {g}" for g in e["groundtruth"]]
        if b is None:
            lines += ["", "原片下载不到，跳过。", ""]
            log(f"[{sample}] 原片下载不到")
            continue
        started = time.monotonic()
        try:
            result, images = comparer.compare(probe.VIDEOS / f"video_{sample}.mp4", b)
        except Exception as exc:  # noqa: BLE001
            log(f"!!! [{sample}] 失败：{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")
            lines += ["", f"失败：{type(exc).__name__}: {exc}", ""]
            continue
        for k, img in enumerate(images):
            (OUT / f"{sample}_{k}_{img['kind']}.jpg").write_bytes(img["jpeg"])
        (OUT / f"{sample}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        lines += ["", "工具输出："] + [f"- {x}" for x in result["summary_lines"]]
        lines += [f"- 用时 {result['timings_s']}；证据图 {len(images)} 张", ""]
        log(f"[{sample}] {time.monotonic() - started:.0f}s：" + " | ".join(result["summary_lines"]))
        (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
