"""生成金标"复核"页：在 Claude 预标注的基础上逐条确认 / 改判（2026-10-04）。

用法：python scripts/_gold_review.py --batch experiments/sft_data/gold_set/v1p2_hard
需要批次目录里有 items.json（_gold_sample.py 产出）和 review_spec.json（每条的分组、要判断的问题、
Claude 预判、核验器结论、证据高亮、截断补录、辅助放大图；videos 给出每个案例原视频相对批次目录的路径、帧率、时长）。输出 复核.html，双击即可用，不联网。
导出的 labels_<缩写>.json 与 标注.html 同格式（多一个 review 字段），可直接给 _gold_eval.py。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("gold_review_template.html")


def build(batch: Path) -> Path:
    items = json.loads((batch / "items.json").read_text(encoding="utf-8"))
    spec = json.loads((batch / "review_spec.json").read_text(encoding="utf-8"))
    missing = [it["id"] for it in items["items"] if it["id"] not in spec["items"]]
    if missing:
        raise SystemExit(f"review_spec.json 缺这些条目：{missing}")
    for it in items["items"]:
        for src in spec["items"][it["id"]].get("images", []):
            if not (batch / "media" / src).exists():
                raise SystemExit(f"{it['id']} 的辅助图不存在：media/{src}")
    for cid, vids in (spec.get("videos") or {}).items():
        for v in vids:
            if not (batch / v["src"]).exists():
                print(f"提示：{cid} 的视频 {v['src']} 在本机找不到，复核页里这段视频会播不出来")
    data = {"meta": items["meta"], "items": items["items"], "spec": spec}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    out = batch / "复核.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    out = build(Path(p.parse_args().batch))
    print(f"已生成 {out}（{out.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()
