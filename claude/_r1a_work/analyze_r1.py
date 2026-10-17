"""r1 全量：三个洞前后对照 + 附表（C 画面类看帧、E 答案层 GT 一致率、各族桶）+ 花费。

    python claude/_r1a_work/analyze_r1.py --export <r1 导出目录> [--round r1] [--unanswerable experiments/sft_data/unanswerable_r1.json]
→ claude/_r1a_work/r1_analysis.json，并打印 markdown 表

口径：
- 洞 1：E / L 可回答案例（不在 unanswerable 里）accepted 报告里的 COMPLETED 数 / accepted 报告数。
- 洞 2：可回答 accepted 报告里事后 status_rewrite 改写的占比（metadata.status_rewritten）；循环内改写另列；另给全轮改写份数
  （walker_<轮次>_status_rewrite.json / 各 run 的 status_rewrite.json）。
- 洞 3：第 2 轮派工（transport 里 G2_* 规则的 INVESTIGATE）行数、其中进 accepted 的 GIA plan 行；第 2 轮工作区里任务 Scope 第一项是
  "fetch the main image of … and compare" 的，有没有真做 S2f_compare_page_image（比例）。
- C 画面类：claim_kinds 含 visual 的 C 案例，W1 里有没有取帧（C2v / C2w / get_frames / video_structure）且交报告前回看（C9v）。
- E 答案层：E 可回答案例的最终答案（核验后重写过的取重写版）按 oracle_generator._gt_points / _gt_misses 查 date / label 两点。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SFT))


def family(case_id: str) -> str:
    ds, _, sid = str(case_id).partition(":")
    if ds in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if ds == "averimatec":
        return "E" if sid.startswith("e") else "L"
    return "C"


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def target(row: dict) -> dict:
    for m in reversed(row.get("messages") or []):
        if m.get("role") == "assistant":
            try:
                return json.loads(m["content"]) if isinstance(m["content"], str) else {}
            except ValueError:
                return {}
    return {}


def run_requests(run_dir: Path) -> list[dict]:
    out = []
    for rd in sorted((run_dir / "requests").glob("request_*"), key=lambda p: int(p.name.split("_")[-1])):
        t = load(rd / "transport.json") or {}
        q = load(rd / "request.json") or {}
        r = load(rd / "response.json") or {}
        payload = q.get("payload") or {}
        cost = sum(float(s.get("cost_usd") or 0.0) for s in t.get("slots") or [])
        out.append({"request": rd.name, "ws": payload.get("workspace_id") or "GIA", "rule": (t.get("oracle") or {}).get("rule"),
                    "task": next((t for t in (payload.get("task"), (payload.get("model_view") or {}).get("task")) if isinstance(t, str)), None),
                    "response": r, "cost": cost})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--unanswerable", default=str(SFT / "unanswerable_r1.json"))
    ap.add_argument("--json", default=str(ROOT / "claude" / "_r1a_work" / "r1_analysis.json"))
    args = ap.parse_args()
    exp = Path(args.export)
    un = set((load(Path(args.unanswerable)) or {}).get("unanswerable_case_ids") or [])
    buckets = {}
    for role in ("hi", "gia"):
        for b in ("accepted", "held", "rejected"):
            buckets[(role, b)] = jsonl(exp / f"{role}_{b}.jsonl")
    out: dict = {"export": exp.name, "round": args.round, "unanswerable_cases": len(un)}

    # ---- 各族桶
    fam_b: dict = defaultdict(Counter)
    for (role, b), rows in buckets.items():
        for r in rows:
            fam_b[f"{role}:{family(r['metadata'].get('case_id'))}"][b] += 1
    out["buckets_by_family"] = {k: dict(v) for k, v in sorted(fam_b.items())}
    rep_b: dict = defaultdict(Counter)
    for b in ("accepted", "held", "rejected"):
        for r in buckets[("hi", b)]:
            if r["metadata"].get("unit") == "report":
                rep_b[family(r["metadata"].get("case_id"))][f"{b}:{target(r).get('status')}"] += 1
    out["report_buckets_by_family"] = {k: dict(v) for k, v in sorted(rep_b.items())}

    # ---- 洞 1 / 洞 2
    hi = buckets[("hi", "accepted")]
    reports = [r for r in hi if r["metadata"].get("unit") == "report"]
    ans = [r for r in reports if r["metadata"].get("case_id") not in un]
    by = Counter((family(r["metadata"]["case_id"]), target(r).get("status")) for r in ans)
    out["answerable_accepted_reports"] = {f: {"COMPLETED": by[(f, "COMPLETED")], "INCOMPLETE": by[(f, "INCOMPLETE")]} for f in "BCEL"}
    out["hole1_EL_completed"] = {f: f"{by[(f, 'COMPLETED')]}/{by[(f, 'COMPLETED')] + by[(f, 'INCOMPLETE')]}" for f in "EL"}
    el_cases = defaultdict(set)
    for r in ans:
        f = family(r["metadata"]["case_id"])
        if f in "EL" and target(r).get("status") == "COMPLETED":
            el_cases[f].add(r["metadata"]["case_id"])
    out["hole1_EL_completed_cases"] = {f: sorted(v) for f, v in el_cases.items()}
    post = [r for r in ans if r["metadata"].get("status_rewritten")]
    inloop = [r for r in ans if r["metadata"].get("status_inloop_rewritten")]
    out["hole2_posthoc_rewrite_accepted_answerable"] = f"{len(post)}/{len(ans)} = {len(post) / max(1, len(ans)):.1%}"
    out["hole2_posthoc_by_family"] = dict(Counter(family(r["metadata"]["case_id"]) for r in post))
    out["hole2_inloop_rewrite_accepted_answerable"] = f"{len(inloop)}/{len(ans)}"
    runs = sorted(p for p in RUNS.glob(f"*_{args.round}_1") if p.is_dir())
    sr = [(p.name, load(p / "status_rewrite.json")) for p in runs if (p / "status_rewrite.json").is_file()]
    n_rewritten = sum(len((s or {}).get("rewrites") or []) for _, s in sr)
    all_reports = 0
    for p in runs:
        links = load(p / "execution_links.json") or []
        all_reports += sum(1 for l in links if (l.get("raw_output") or {}).get("status") in ("COMPLETED", "INCOMPLETE"))
    out["hole2_posthoc_rewrite_all_reports"] = f"{n_rewritten}/{all_reports}（runs with status_rewrite.json: {len(sr)}）"
    out["hole2_posthoc_rewrite_runs"] = [n for n, _ in sr]

    # ---- 洞 3 / 第 2 轮 Scope 第一项 / C 画面类 / 花费
    round2 = []
    scope_compare = {"tasks": 0, "did_compare": 0, "runs": []}
    c_visual = {"cases": 0, "frames": 0, "review": 0, "both": 0, "missing": []}
    cost = 0.0
    cost_by_run = {}
    from foren_searcher.datagen.walker import claim_kinds  # type: ignore
    for p in runs:
        reqs = run_requests(p)
        run_cost = sum(r["cost"] for r in reqs)
        cost += run_cost
        cost_by_run[p.name] = round(run_cost, 4)
        cfg = load(p / "run_config.json") or {}
        case_id = cfg.get("case_id") or ""
        fam = family(case_id)
        for r in reqs:
            if str(r["rule"] or "").startswith("G2_"):
                round2.append({"run": p.name, "family": fam, "rule": r["rule"]})
        ws_rules: dict = defaultdict(list)
        ws_task: dict = {}
        ws_ops: dict = defaultdict(set)
        for r in reqs:
            ws_rules[r["ws"]].append(r["rule"])
            req = ((r["response"].get("arguments") or {}).get("request") or {}) if isinstance(r["response"], dict) else {}
            if req.get("operation"):
                ws_ops[r["ws"]].add(req["operation"])
            if r["task"] and r["ws"] not in ws_task:
                ws_task[r["ws"]] = r["task"]
        for ws, rules in ws_rules.items():
            if ws.startswith("W2") and re.search(r"Scope:\s*first fetch the main image of", ws_task.get(ws) or "", re.I):
                scope_compare["tasks"] += 1
                done = "S2f_compare_page_image" in rules
                scope_compare["did_compare"] += int(done)
                scope_compare["runs"].append({"run": p.name, "ws": ws, "did_compare": done})
        if fam == "C":
            case = load(Path(cfg.get("case_file") or "")) if cfg.get("case_file") else None
            if case is None:
                stem = p.name[: -len(f"_{args.round}_1")]
                case = load(SFT / "cases" / f"{stem}.json")
            claim = " ".join((case or {}).get("claims") or []) or (case or {}).get("task_text_en") or ""
            try:
                kinds = claim_kinds(claim)
            except Exception:  # noqa: BLE001
                kinds = {"visual"}
            has_video = any(str(m.get("media_type", "")).startswith("video") for m in (case or {}).get("media") or [])
            if "visual" in kinds and has_video:
                w1 = [x for ws, rs in ws_rules.items() if ws.startswith("W1") for x in rs]
                frames = any(str(x).startswith(("C2v", "C2w")) for x in w1) or bool(
                    {"get_frames", "video_structure"} & set().union(*[o for ws, o in ws_ops.items() if ws.startswith("W1")] or [set()]))
                review = any(str(x).startswith("C9v") for x in w1)
                c_visual["cases"] += 1
                c_visual["frames"] += int(frames)
                c_visual["review"] += int(review)
                c_visual["both"] += int(frames and review)
                if not (frames and review):
                    c_visual["missing"].append({"run": p.name, "frames": frames, "review": review})
    out["hole3_round2_dispatch"] = {"count": len(round2), "by_family": dict(Counter(r["family"] for r in round2)),
                                    "runs": [r["run"] for r in round2]}
    gia = buckets[("gia", "accepted")]
    plans2 = [r for r in gia if r["metadata"].get("unit") == "plan" and int(r["metadata"].get("round") or 1) >= 2]
    out["hole3_round2_plan_accepted"] = {"count": len(plans2), "by_family": dict(Counter(family(r["metadata"]["case_id"]) for r in plans2))}
    out["round2_scope_first_compare"] = {**scope_compare, "share": f"{scope_compare['did_compare']}/{scope_compare['tasks']}"}
    out["c_visual_frames"] = {**c_visual, "share_frames_and_review": f"{c_visual['both']}/{c_visual['cases']}"}
    out["openrouter_slot_cost_usd"] = round(cost, 2)
    out["runs"] = len(runs)

    # ---- E 答案层与 GT
    from oracle_generator import _gt_misses, _gt_points  # type: ignore
    e_rows = []
    for p in runs:
        cfg = load(p / "run_config.json") or {}
        case_id = cfg.get("case_id") or ""
        if family(case_id) != "E" or case_id in un:
            continue
        stem = p.name[: -len(f"_{args.round}_1")]
        case_path = SFT / "cases" / f"{stem}.json"
        links = load(p / "execution_links.json") or []
        fin = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH"]
        answer = str((fin[-1].get("raw_output") or {}).get("answer") or "") if fin else ""
        pts = _gt_points(str(case_path))
        missed = {m["kind"] for m in _gt_misses(pts, answer)}
        e_rows.append({"run": p.name, "date_ok": "date" not in missed if any(x["kind"] == "date" for x in pts) else None,
                       "label_ok": "label" not in missed if any(x["kind"] == "label" for x in pts) else None,
                       "revised": (p / "execution_links.json.orig").is_file()})
    n = len(e_rows)
    out["E_answer_gt"] = {"cases": n, "date_ok": sum(1 for r in e_rows if r["date_ok"]),
                          "label_ok": sum(1 for r in e_rows if r["label_ok"]),
                          "both_ok": sum(1 for r in e_rows if r["date_ok"] and r["label_ok"]), "rows": e_rows}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    Path(args.json).write_text(text, encoding="utf-8")
    keys = [k for k in out if k not in ("rows",)]
    for k in keys:
        v = out[k]
        if isinstance(v, dict) and "rows" in v:
            v = {kk: vv for kk, vv in v.items() if kk not in ("rows", "runs", "missing")}
        print(f"- {k}: {json.dumps(v, ensure_ascii=False)[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
