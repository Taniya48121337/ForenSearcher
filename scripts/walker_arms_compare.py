"""S5 答案的 GT 用法对照（2026-10-14）：同一条轨迹、同一个 FINISH 状态下的四版答案，分别核验后比较。

    python scripts\\walker_arms_compare.py --round wk11 --cases <案例…>

各版（walker_run.py d1 --arms 产生）：
  d0  <round>    首稿：教师只看状态写答案，不用 GT，不重写
  b   <round>b   原方法：首稿漏了 GT 要点就把 GT 词（裁决标签候选词）提示给教师重写；核验后把全部核验意见（含"应有的
                 结论 'misleading'"这类 GT 值）给教师再重写一次
  g   <round>g   证据门控：漏掉的要点只有在可见 Finding 里确实有时才提示，且只指到 E、不说 GT；核验后只把"证据不支持 /
                 引错 / 说过头"这类意见给教师重写（不看整体判定、不给 GT 类意见）
  n   <round>n   择优：首稿 + 另抽 N-1 份（同一提示），按 GT 命中数挑一份；不提示、不重写

指标：
  核验 positive   answer 单元判定（Sonnet 核验，与以前各轮相同）；b / g 给重写前 → 重写后
  GT 命中         程序核对（与 oracle 的 _gt_misses 相同）：最终答案里有没有 GT 要点
  无据命中        答案命中了某条 GT 要点，但学生可见的 Finding 里没有这条（= 很可能是 GT 提示带出来的，不是证据推出来的）
  证据问题        核验意见里非 minor 的"证据不支持 / 引错 / 说过头"条数（不含 GT 类意见）
产物：experiments/sft_data/walker_<round>_arms.md / .json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SFT))
sys.path.insert(0, str(ROOT / "scripts"))

ARMS = [("d0", "", "首稿（不用 GT）"), ("b", "b", "原方法"), ("g", "g", "证据门控"), ("n", "n", "择优")]


def _units(tag: str, suffix: str = "") -> dict[str, dict[str, Any]]:
    files = [p for p in SFT.glob(f"verdicts_units_{tag}_judged.*.jsonl{suffix}") if ".summary" not in p.name]
    if not files:
        return {}
    path = max(files, key=lambda p: p.stat().st_mtime)
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("unit") == "answer":
            out[str((row.get("subject") or {}).get("run"))] = row
    return out


def _finish_dir(run_dir: Path) -> Path | None:
    try:
        links = json.loads((run_dir / "execution_links.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    finish = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH" and l.get("request_id")]
    return run_dir / "requests" / finish[-1]["request_id"] if finish else None


def _answer(run_dir: Path) -> str:
    try:
        links = json.loads((run_dir / "execution_links.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    finish = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH"]
    return str((finish[-1].get("raw_output") or {}).get("answer") or "") if finish else ""


def _family(case: str) -> str:
    if case.startswith("averimatec_e"):
        return "E 最早出处"
    if case.startswith("averimatec_l"):
        return "L 拍摄地"
    if case.startswith("fmnv_a_"):
        return "A 找原片"
    if case.startswith(("fmnv_c_", "groundlie360")):
        return "C 核查声明"
    return "B 给定媒体"


def main() -> int:
    from oracle_generator import _gt_misses, _gt_points  # type: ignore
    from walker_run import support_errors  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    tag = args.round
    units = {arm: _units(tag + sfx) for arm, sfx, _ in ARMS}
    pre = {arm: _units(tag + sfx, ".pre_revise") for arm, sfx, _ in ARMS if arm in ("b", "g")}
    rows = []
    for case in args.cases:
        live = RUNS / f"{case}_{tag}_1"
        fdir = _finish_dir(live) if (live / "result.json").is_file() else None
        if not fdir or not (fdir / "slot_S5_arms.json").is_file():
            continue
        cfg = json.loads((live / "run_config.json").read_text(encoding="utf-8")) if (live / "run_config.json").is_file() else {}
        points = _gt_points(cfg.get("case_file"))
        gt_check = json.loads((fdir / "slot_S5_gt_check.json").read_text(encoding="utf-8")) if (fdir / "slot_S5_gt_check.json").is_file() else {}
        support = gt_check.get("support") or {}
        arms_meta = json.loads((fdir / "slot_S5_arms.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {"case": case, "family": _family(case), "points": [p["kind"] for p in points],
                               "visible_support": {k: bool(v) for k, v in support.items()}, "arms": {}}
        for arm, sfx, _ in ARMS:
            run = f"{case}_{tag + sfx}_1"
            run_dir = RUNS / run
            answer = _answer(run_dir) if run_dir.is_dir() else ""
            missed = {m["kind"] for m in _gt_misses(points, answer)} if answer else {p["kind"] for p in points}
            hit = [p["kind"] for p in points if p["kind"] not in missed]
            unit = units[arm].get(run) or {}
            before = (pre.get(arm) or {}).get(run)
            row["arms"][arm] = {
                "run": run, "exists": run_dir.is_dir(), "disposition": unit.get("disposition"),
                "disposition_before_revise": (before or unit).get("disposition") if arm in ("b", "g") else None,
                "revised": (run_dir / "execution_links.json.orig").exists(),
                "gt_hit": hit, "gt_all": not missed and bool(points),
                "hit_without_visible_support": [k for k in hit if not support.get(k)],
                "support_errors": len(support_errors(unit.get("errors") or [])),
                "hinted": arms_meta.get(arm, {}).get("hinted"), "picked": arms_meta.get(arm, {}).get("picked"),
                "words": len(answer.split()),
            }
        rows.append(row)

    def mark(a: dict[str, Any]) -> str:
        d = {"positive": "✓", "negative": "✗"}.get(a["disposition"] or "", "?")
        if a["disposition_before_revise"] and a["revised"]:
            d = {"positive": "✓", "negative": "✗"}.get(a["disposition_before_revise"], "?") + "→" + d
        g = "GT✓" if a["gt_all"] else ("GT½" if a["gt_hit"] else "GT✗")
        extra = "，无据" if a["hit_without_visible_support"] else ""
        extra += f"，证据问题 {a['support_errors']}" if a["support_errors"] else ""
        return f"{d} {g}{extra}"

    teacher = "?"
    for r in rows:
        try:
            cfg = json.loads((RUNS / r["arms"]["d0"]["run"] / "run_config.json").read_text(encoding="utf-8"))
            teacher = str((cfg.get("walker") or {}).get("teacher_model") or teacher)
            break
        except (OSError, ValueError):
            continue
    lines = [f"# walker {tag}：S5 答案 GT 用法对照（教师 {teacher}）", "",
             "同一条轨迹、同一个 FINISH 状态下的几版答案分别核验。✓/✗ = 核验 answer 单元 positive / negative（b、g 为"
             "重写前→重写后）；GT✓/½/✗ = 程序核对 GT 要点全中 / 部分 / 没中；无据 = 命中的 GT 要点在学生可见的 Finding 里没有；"
             "证据问题 = 非 minor 的证据不支持类意见条数。", "",
             "| 案例 | 族 | GT 要点（可见证据里有？） | " + " | ".join(f"{a}·{n}" for a, _, n in ARMS) + " |",
             "|---|---|---|" + "---|" * len(ARMS)]
    for r in rows:
        pts = "、".join(f"{k}{'有' if r['visible_support'].get(k) else '无'}" for k in r["points"]) or "—"
        lines.append(f"| {r['case']} | {r['family']} | {pts} | " + " | ".join(mark(r["arms"][a]) for a, _, _ in ARMS) + " |")
    n = len(rows)
    lines += ["", "## 汇总", "", "| 版本 | 核验 positive（重写前→后） | GT 全中 | 无据命中 | 有证据问题的答案 | 平均词数 |", "|---|---|---|---|---|---|"]
    summary = {}
    for arm, _, name in ARMS:
        xs = [r["arms"][arm] for r in rows if r["arms"][arm]["exists"]]
        pos = sum(x["disposition"] == "positive" for x in xs)
        pos0 = sum((x["disposition_before_revise"] or x["disposition"]) == "positive" for x in xs)
        s = {"n": len(xs), "positive": pos, "positive_before_revise": pos0, "gt_all": sum(x["gt_all"] for x in xs),
             "hit_without_support": sum(bool(x["hit_without_visible_support"]) for x in xs),
             "with_support_errors": sum(x["support_errors"] > 0 for x in xs),
             "words": round(sum(x["words"] for x in xs) / max(1, len(xs)), 1)}
        summary[arm] = s
        pos_txt = f"{pos0}→{pos}/{len(xs)}" if arm in ("b", "g") else f"{pos}/{len(xs)}"
        lines.append(f"| {arm}·{name} | {pos_txt} | {s['gt_all']}/{len(xs)} | {s['hit_without_support']} | "
                     f"{s['with_support_errors']} | {s['words']} |")
    fams: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    lines += ["", "## 分族（核验 positive，最终）", "", "| 族 | 案例数 | " + " | ".join(a for a, _, _ in ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    for fam, rs in sorted(fams.items()):
        lines.append(f"| {fam} | {len(rs)} | " + " | ".join(str(sum(r["arms"][a]["disposition"] == "positive" for r in rs))
                                                           for a, _, _ in ARMS) + " |")
    vis = sum(any(r["visible_support"].values()) for r in rows)
    lines += ["", f"{n} 个案例里，{vis} 个的可见 Finding 至少含一条 GT 要点（门控提示只在这些上能给出指向 E 的提示）。"]
    out = SFT / f"walker_{tag}_arms"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"round": tag, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
