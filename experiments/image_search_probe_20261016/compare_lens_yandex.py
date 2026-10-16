"""10-16 第二轮：Lens（Serper）与 Yandex 同一批图的召回对比，离线合并（不重新搜索），写 lens_vs_yandex.md。

  python experiments\\image_search_probe_20261016\\compare_lens_yandex.py

- Yandex：yandex/results.json（第一轮）；Lens：lens/results.json（第二轮，含 Lens 自己的原始顺序 lens_raw）。
- 合并：reverse_normalize.merge_slots（引擎顺序与正式 run 一样：lens 在 yandex 前），build_candidate，same_first。
  每条的核对判定取两边已有的结果（base 引擎优先）；两边都没核过的记 unchecked。
  注意：第一轮 Yandex 只存了核对后重排的顺序，离线合并把它当作 Yandex 的名次（same 的几条可能比原始名次靠前）。
- 真来源核对判定"对不对"：人工逐张看过真来源的缩略图（lens/gt_thumbs.jpg），结论写在 JUDGED 里。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from recall_yandex import norm  # noqa: E402

from foren_searcher.search.contracts import CandidateSource  # noqa: E402
from foren_searcher.search.providers import reverse_normalize as rn  # noqa: E402

SPECIAL = {
    "e107": "查询图本身被改过（单边黑框 + 贴旗），核对漏判",
    "e193": "真来源在前 10 之外（Yandex 第 26）",
    "e122": "真来源在前 10 之外（Yandex 第 20）",
    "l274": "Yandex 结果里有篡改副本被判 near-identical（第一轮第 8 名）",
}
# 人工看过真来源缩略图后的结论：{(图, 引擎): "对" / "错" + 说明}，由 lens/gt_thumbs.jpg 与 yandex/sheet_*.jpg 得出
JUDGED: dict[tuple[str, str], str] = {
    ("e107", "yandex"): "Yandex 错（同一张照片被加框贴旗，判 no match）",
    ("e107", "lens"): "Lens 错（同上）",
    ("e297", "yandex"): "Yandex 对（同一张照片，near-identical）",
    ("e260", "yandex"): "Yandex 对（同一张照片，near-identical）",
    ("e260", "lens"): "Lens 对（页面缩略图是同一事件的另一张照片，no match）",
    ("e505", "yandex"): "Yandex 对（同一张照片，near-identical）",
    ("e372", "lens"): "Lens 对（页面缩略图是同一现场的另一张照片，no match）",
    ("e403", "lens"): "Lens 对（同一张照片，near-identical）",
    ("evid_103", "lens"): "Lens 对（页面缩略图是同一场抗议的另一张照片，no match）",
}


def gt_date(case: str) -> str | None:
    if case.startswith("evid"):
        return None
    d = json.loads((ROOT / f"experiments/sft_data/cases/averimatec_{case}.json").read_text(encoding="utf-8"))
    for r in d.get("required_evidence") or []:
        m = re.search(r"（(\d{4}-\d{2}-\d{2})）", r.get("description", "")) if r.get("key") == "earliest_source" else None
        if m:
            return m.group(1)
    return None


def url_year(url: str) -> str | None:
    m = re.search(r"/(20\d{2}|19\d{2})(?:/|-)(?:\d{2}|[a-z]{3})", url.lower())
    return m.group(1) if m else None


def _cand(url: str, provider: str, rank: int, ctype: str = "page", **extra) -> CandidateSource:
    return CandidateSource(url, None, None, provider=provider, engine=provider, provider_rank=rank, candidate_type=ctype, extra=extra)


def merged_rank(y: dict, lz: dict) -> tuple[int | None, str | None]:
    lens_list = [CandidateSource(**c) for c in lz["lens_raw"]]
    yandex_list = [_cand(r["url"], "yandex-reverse", r["rank"]) for r in y["results"]]
    checks = {}
    for row, eng in ((lz, "lens"), (y, "yandex")):
        for r in row["results"]:
            if r["verified"] and r["verified"] != "unchecked":
                checks.setdefault((eng, rn.identity(r["url"])), {"verified_match": r["verified"], **(r["scores"] or {}),
                                                               "thumbnail_url": None})
    slots = rn.merge_slots({"lens": lens_list, "yandex": yandex_list}, ["lens", "yandex"])
    out = []
    for i, slot in enumerate(slots):
        ident = rn.identity(slot["base"].source_url)
        check = None
        if i < 10:
            check = next((checks[(e, ident)] for e in slot["found_by"] if (e, ident) in checks), None)
        out.append(rn.build_candidate(slot, check))
    merged = rn.same_first(out)
    item = {"family": y["family"], "id": y["id"], "gt_norm": y.get("gt_norm"), "gt_full": y.get("gt_full"), "gt_ids": y.get("gt_ids")}
    from recall_yandex import gt_hit
    for rank, c in enumerate(merged, start=1):
        if gt_hit(item, c.source_url):
            return rank, c.extra.get("verified_match")
    return None, None


def gt_of(row: dict) -> dict | None:
    return next((r for r in row["results"] if r["gt"]), None)


def main() -> int:
    yandex = {r["id"]: r for r in json.loads((HERE / "yandex/results.json").read_text(encoding="utf-8"))}
    lens = {r["id"]: r for r in json.loads((HERE / "lens/results.json").read_text(encoding="utf-8"))}
    ids = [i for i in yandex if i in lens]
    lines = ["# Lens（Serper）与 Yandex 以图搜图召回对比（2026-10-16）", "",
             f"同一批 {len(ids)} 张查询图（字节相同）。Yandex 为第一轮结果，Lens 为第二轮；合并为离线按 multi_reverse 逻辑合并。",
             "名次是统一写法下的最终名次（核对判 near-identical 的排最前）。", "",
             "| 图 | 族 | Yandex 名次 | Lens 名次 | 合并名次 | 真来源核对判定（Yandex / Lens） | 判定对不对 | 经图床 |",
             "|---|---|---|---|---|---|---|---|"]
    stats: dict[str, dict[str, int]] = {}
    only_lens, only_yandex = [], []
    for i in ids:
        y, lz = yandex[i], lens[i]
        yg, lg = gt_of(y), gt_of(lz)
        mr, mv = merged_rank(y, lz)
        host = lz["lens_summary"].get("image_host_used") or ("缓存" if lz["lens_summary"].get("disk_hits") else "否")
        verdicts = f"{yg['verified'] if yg else '–'} / {lg['verified'] if lg else '–'}"
        judged = "；".join(v for k, v in JUDGED.items() if k[0] == i) or "–"
        lines.append(f"| {i}{' ※' if i in SPECIAL else ''} | {y['family']} | {yg['rank'] if yg else '–'} | {lg['rank'] if lg else '–'} | "
                     f"{mr or '–'} | {verdicts} | {judged} | {host} |")
        if i in SPECIAL:
            continue
        s = stats.setdefault(y["family"], {"n": 0, "yandex": 0, "lens": 0, "merged": 0})
        s["n"] += 1
        yin, lin = bool(yg and yg["rank"] <= 10), bool(lg and lg["rank"] <= 10)
        s["yandex"] += yin
        s["lens"] += lin
        s["merged"] += bool(mr and mr <= 10)
        if lin and not yin:
            only_lens.append(i)
        if yin and not lin:
            only_yandex.append(i)
    lines += ["", "※ 不计入召回，见下文\"单列\"。", "", "## 前 10 召回（不含 ※ 的几张）", "",
              "| 族 | 张数 | Yandex 单独 | Lens 单独 | 两者合并 |", "|---|---|---|---|---|"]
    for fam in ("E", "L", "EVID"):
        s = stats.get(fam)
        if s:
            lines.append(f"| {fam} | {s['n']} | {s['yandex']}/{s['n']} | {s['lens']}/{s['n']} | {s['merged']}/{s['n']} |")
    lines += ["", f"- Lens 前 10 有、Yandex 前 10 没有的真来源：{len(only_lens)} 个（{', '.join(only_lens) or '无'}）",
              f"- Yandex 前 10 有、Lens 前 10 没有的真来源：{len(only_yandex)} 个（{', '.join(only_yandex) or '无'}）", "",
              "## 单列（不计入召回）", ""]
    for i, why in SPECIAL.items():
        if i in lens:
            yg, lg = gt_of(yandex[i]), gt_of(lens[i])
            mr, _ = merged_rank(yandex[i], lens[i])
            lines.append(f"- **{i}**：{why}。Yandex {yg['rank'] if yg else '–'}，Lens {lg['rank'] if lg else '–'}，合并 {mr or '–'}"
                         f"；真来源核对 Yandex {yg['verified'] if yg else '–'} / Lens {lg['verified'] if lg else '–'}。")
    lines += ["", "## Lens 结果里的日期", "",
              "Serper 版 Lens 的 date 字段；约略年份按取到结果那天（date_basis）换算。\"网址年份\"是结果网址里带的年份（有的话），"
              "用来粗查换算对不对；E 族另列 GT 最早出处日期。", "",
              "| 图 | 名次 | 结果网站 | Lens 原文 | 换算 | 网址年份 | 换算对不对 | GT 最早出处 |", "|---|---|---|---|---|---|---|---|"]
    n_dates = 0
    for i in ids:
        for r in lens[i]["results"]:
            if not r.get("date_raw"):
                continue
            n_dates += 1
            uy = url_year(r["url"])
            approx = r.get("date") or "–"
            ok = "–" if not uy or not r.get("date") else ("对" if str(r["date"])[:4] == uy
                                                          else f"差 {abs(int(str(r['date'])[:4]) - int(uy))} 年")
            lines.append(f"| {i} | {r['rank']}{' (GT)' if r['gt'] else ''} | {norm(r['url']).split('/')[0]} | {r['date_raw']} | "
                         f"{approx} ({r.get('date_kind')}) | {uy or '–'} | {ok} | {gt_date(i) or '–'} |")
    if not n_dates:
        lines.append("| （Lens 结果里没有日期） | | | | | | | |")
    lines += ["", f"Lens 结果共 {sum(len(lens[i]['results']) for i in ids)} 条，带日期的 {n_dates} 条。", ""]
    ledger = json.loads((HERE / "lens/ledger.json").read_text(encoding="utf-8"))
    credits = [e.get("credits") if e.get("credits") is not None else e["balance_before"] - e["balance_after"] for e in ledger["entries"]]
    empty = [i for i in ids if lens[i]["n"] == 0]
    uploads = sum(lens[i]["lens_summary"].get("host_uploads", 0) for i in ids)
    hosts = sorted({lens[i]["lens_summary"].get("image_host_used") for i in ids if lens[i]["lens_summary"].get("image_host_used")})
    lines += ["## 额度与图床", "",
              f"- Serper Lens 请求 {ledger['requests']} 次（硬上限 20，lens/ledger.json），每次扣 {sorted(set(credits))} 个额度，"
              f"合计 {sum(credits)}；余额 {ledger['entries'][0]['balance_before']} → {ledger['entries'][-1].get('balance_after')}"
              "（余额接口约 20 秒后才更新，第一次的读数是事后补记的）。",
              f"- 图床上传 {uploads} 次，用的是 {', '.join(hosts) or '无'}（imgbb 10 分钟后删除；litterbox 备用没有用上）。",
              f"- 返回 0 条结果、照样扣了额度的：{', '.join(empty) or '无'}（原因未查：可能是 Google 当时取不到 imgbb 的图，"
              "也可能 Lens 确实没找到；e297 的同一张图 Yandex 找到了 3 条 near-identical）。",
              "- 硬盘缓存命中 0 次（这批图以前没用 Lens 搜过）。", "",
              "## 观察", "",
              "- Serper 版 Lens 的结果多是\"同一场景的其他照片\"（visual matches），不是同一张图：真来源页面的缩略图常是该页的"
              "另一张照片（e372、e260、evid_103），核对判 no match 对缩略图是对的，但不说明页面里没有查询图。",
              "- 合并时同一网页的核对取 base 引擎（Lens 在前）的缩略图：e260 的真来源 Yandex 缩略图判 near-identical、Lens 缩略图判"
              " no match，合并后按 Lens 判，名次从 Yandex 的 3 / Lens 的 2 落到 5。可以考虑 base 对不上时再试其他引擎的缩略图。",
              "- 第一轮 Yandex 只存了核对后重排的顺序，离线合并把它当作 Yandex 原始名次。", ""]
    (HERE / "lens_vs_yandex.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
