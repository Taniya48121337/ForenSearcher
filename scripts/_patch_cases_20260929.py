"""案卷修补（2026-09-29，案卷 × 检索护栏对照）：两个案卷的必需证据组在默认 --blocklist exclude 下全部被拦、做不到，
补上非核查机构的一手 / 参考来源。先预览，--apply 才写；已改过的（notes 带标记）跳过，可重复跑。

    python scripts/_patch_cases_20260929.py [--apply]

- averimatec:699：REQ1 组原有 altnews / newsmobile / telugupost / toi 四个，全是其它核查机构或媒体核查栏目，09-29 起都在黑名单里。
  补 Jatland Wiki「Lohagarh Fort」条目（写明巴拉特布尔宫内铁柱 Vijay Stambha 刻的是贾特王族世系），must_cite=True 进同组。
  REQ2 原只有 thewire（可选），补 Wikipedia「Iron pillar of Delhi」（铭文为笈多体梵文、记旃陀罗王），两者合成一个可选 any_one_of 组。
- true:39132300：唯一必需项是 snopes（TRUE 声明本身出自 Snopes，属原始核查），被整站拦截。ghostface_site 页写明
  Fun World 1991 年「Fantastic Faces」系列即已发售，改 must_cite=True，与 snopes 组成 any_one_of（旧 run 引 snopes 仍算满足）。
两处来源 09-29 用 WebFetch 核对过页面内容。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CASES = ROOT / "experiments" / "sft_data" / "cases"
MARK = "（2026-09-29 护栏对照修补"

from foren_searcher.datagen.cases import Case  # noqa: E402


def patch_699(d: dict) -> None:
    d["required_evidence"] += [
        {"key": "jatland_lohagarh", "description": "Jatland Wiki「Lohagarh Fort」条目：巴拉特布尔宫内铁柱（Vijay Stambha）刻贾特王族世系",
         "url_pattern": "jatland.com/home/lohagarh_fort*", "acquired_by": None, "must_cite": True, "supports": ["REQ1"]},
        {"key": "wiki_delhi_pillar", "description": "Wikipedia「Iron pillar of Delhi」：铭文内容（笈多体梵文，记旃陀罗王）",
         "url_pattern": "en.wikipedia.org/wiki/iron_pillar_of_delhi*", "acquired_by": None, "must_cite": False, "supports": ["REQ2"]},
    ]
    d["required_evidence_rule"] = "any_one_of:altnews,newsmobile,telugupost,toi,jatland_lohagarh; any_one_of:thewire,wiki_delhi_pillar"
    d["notes"] += (f"{MARK}：REQ1 组原四项均为二手核查、默认黑名单下全被拦，补 jatland_lohagarh（must）；REQ2 补 wiki_delhi_pillar，"
                   "与 thewire 合为可选组。）")


def patch_39132300(d: dict) -> None:
    for r in d["required_evidence"]:
        if r["key"] == "ghostface_site":
            r["must_cite"] = True
            r["description"] = "GhostFace.co.uk 面具历史页（Fun World 1991 年 Fantastic Faces 系列）"
    d["required_evidence_rule"] = "any_one_of:snopes,ghostface_site"
    d["notes"] += (f"{MARK}：snopes 为 TRUE 声明的原始核查、默认黑名单整站拦截，唯一必需组做不到；ghostface_site 改 must，"
                   "与 snopes 组成 any_one_of。）")


PATCHES = {"averimatec_699.json": patch_699, "true_39132300.json": patch_39132300}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--cases", default=str(CASES))
    args = parser.parse_args()
    root = Path(args.cases)
    for name, fn in PATCHES.items():
        path = root / name
        data = json.loads(path.read_text(encoding="utf-8"))
        if MARK in data.get("notes", ""):
            print(f"{name}: 已修补，跳过")
            continue
        fn(data)
        Case.from_dict(data)        # 校验
        print(f"{name}: rule → {data['required_evidence_rule']}")
        if args.apply:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.apply:
        print("预览；加 --apply 写入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
