"""walker（程序走、教师填）与老轨迹的同案例对照（2026-10-12，walker 计划 D1 / §8 验收）。

    python scripts\\walker_compare.py --round wk1 --cases videosham_4126 ...

只读 run 目录与已有核验结果，不调模型。对每个案例并排：
- 老轨迹：r0e（hi@0.5.0 / gia@0.6.0，Codex xhigh 走全程）；qvhighlights_9167 另列 tp2（hi@0.7.0 / gia@0.8.0，与 walker 同规格）。
- walker：<round>（hi@0.7.0 / gia@0.8.0，程序走，Codex medium 只填报告 / 答案 / 第 2 轮任务）。
指标：决策数、教师（LLM）调用数、墙钟、HI 动作步、重复步（同工作区同动作同参数）、格式被拒（修复回合）、
链条（align → diff → 差分图被看到；单段定位为 locate → 边界帧）、报告状态、核验（报告 / 答案的 positive 比例与错误计数）。
写 experiments/sft_data/walker_d1_compare.md 与 .json。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"
_RESULT_EVIDENCE = re.compile(r"Result Evidence:\s*((?:E[1-9][0-9]*(?:,\s*)?)+)")


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _response(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                return None
    return None


def _observation_ids(obs: dict[str, Any] | None) -> tuple[str | None, list[str], str | None]:
    """(status, created/result E ids, result_evidence_id)，兼容截断的 preview。"""

    if not obs:
        return None, [], None
    details = obs.get("details") or {}
    status = details.get("status")
    created = list(((details.get("execution") or {}).get("created_evidence_ids")) or [])
    result_e = details.get("result_evidence_id")
    text = " ".join(str(x) for x in obs.get("items") or []) + " " + str(obs.get("preview") or "")[:3000]
    if not status:
        m = re.search(r"returned\s+(\w+)", text)
        status = m.group(1) if m else None
    m = _RESULT_EVIDENCE.search(text)
    if m:
        ids = [x.strip() for x in m.group(1).split(",") if x.strip()]
        created = created or ids
        result_e = result_e or (ids[0] if ids else None)
    return status, created, result_e


def manifest_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((SFT / "run_manifests").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("run"):
                rows[row["run"]] = row          # 同名取最后一条
    return rows


def analyze(run_dir: Path, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reqs = sorted((run_dir / "requests").glob("request_*"))
    out: dict[str, Any] = {"run": run_dir.name, "requests": len(reqs), "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                           "llm_seconds": 0.0, "repairs": 0, "hi_steps": 0, "repeats": 0, "gia": Counter(), "ops": Counter(),
                           "reports": [], "workspaces": {}, "branches": Counter(), "oracle": False}
    per_ws: dict[str, list[dict[str, Any]]] = {}
    for req in reqs:
        request = _load(req / "request.json") or {}
        payload = request.get("payload") or {}
        transport = _load(req / "transport.json") or {}
        output = _response(req / "response.json") or {}
        if transport.get("transport") == "oracle_walker":
            out["oracle"] = True
            slots = transport.get("slots") or []
            out["llm_calls"] += len(slots)
            out["llm_seconds"] += sum(float(s.get("elapsed_s") or 0) for s in slots)
            branch = (transport.get("oracle") or {}).get("branch")
            if branch:
                out["branches"][branch] += 1
        elif transport.get("forced"):
            pass
        else:
            out["llm_calls"] += 1
            out["llm_seconds"] += float(transport.get("elapsed_s") or 0)
        usage = transport.get("usage") or {}
        out["input_tokens"] += int(usage.get("input_tokens") or 0)
        out["output_tokens"] += int(usage.get("output_tokens") or 0)
        if "previous_invalid_output" in payload or "validation_error" in payload:
            out["repairs"] += 1
        ws = payload.get("workspace_id")
        if ws:
            view = payload.get("model_view") or {}
            per_ws.setdefault(ws, []).append({"output": output, "view": view, "attached": request.get("image_evidence_ids") or []})
        else:
            key = output.get("decision") or output.get("action") or "?"
            out["gia"][key] += 1
            if output.get("decision") == "FINISH":
                out["answer"] = output.get("answer", "")
    for ws, steps in per_ws.items():
        seen: set[str] = set()
        info = {"steps": 0, "align": False, "diff": False, "diff_ready": False, "diff_images_seen": False,
                "locate": False, "boundary_frames": False, "report": None, "ops": []}
        diff_images: list[str] = []
        for index, step in enumerate(steps):
            output = step["output"]
            if "findings" in output or output.get("status") in ("COMPLETED", "INCOMPLETE"):
                info["report"] = output.get("status")
                out["reports"].append(output.get("status"))
                continue
            action = output.get("action")
            if not action:
                continue
            info["steps"] += 1
            out["hi_steps"] += 1
            args = output.get("arguments") or {}
            key = json.dumps([action, args], sort_keys=True, ensure_ascii=False)
            if key in seen:
                out["repeats"] += 1
            seen.add(key)
            op = ((args.get("request") or {}).get("operation")) if action == "INSPECT" else None
            label = f"{action} {op}" if op else action
            if op == "align_media":
                label += f"({((args.get('request') or {}).get('params') or {}).get('mode')})"
            out["ops"][label] += 1
            info["ops"].append(label)
            nxt = steps[index + 1]["view"] if index + 1 < len(steps) else {}
            status, created, result_e = _observation_ids((nxt or {}).get("latest_observation"))
            if op == "align_media" and (((args.get("request") or {}).get("params") or {}).get("mode") == "visual"):
                info["align"] = True
            if op == "diff_media":
                info["diff"] = True
                if status == "ready":
                    info["diff_ready"] = True
                    diff_images = [e for e in created if e != result_e]
            if op == "compare_media" and not (((args.get("request") or {}).get("params") or {}).get("range_a_s")):
                info["compare"] = True
                info["compare_ready"] = info.get("compare_ready") or status == "ready"
            if op in ("locate_frames", "locate"):
                info["locate"] = True
            if op == "get_frames" and (info["diff"] or info["locate"] or info["align"] or info.get("compare")) and "times_s" in (((args.get("request") or {}).get("params")) or {}):
                info["boundary_frames"] = True
        if diff_images:
            later_seen = set()
            for step in steps:
                later_seen |= set(step["attached"]) | set(step["view"].get("content_observed_evidence_ids") or [])
            info["diff_images_seen"] = bool(set(diff_images) & later_seen)
        out["workspaces"][ws] = info
    row = manifests.get(run_dir.name) or {}
    out["wall_seconds"] = row.get("wall_seconds")
    if out["wall_seconds"] is None and reqs:
        first = (reqs[0] / "request.json").stat().st_mtime
        last = max(p.stat().st_mtime for p in reqs[-1].iterdir())
        out["wall_seconds"] = round(last - first, 1)
    out["llm_seconds"] = round(out["llm_seconds"], 1)
    out["gia"] = dict(out["gia"])
    out["ops"] = dict(out["ops"])
    out["branches"] = dict(out["branches"])
    return out


def chain_ok(info: dict[str, Any]) -> bool:
    """两段视频：align → diff → 差分图被看到；对齐太弱的对应题（不做 diff）：align → 以图定位 / 两侧取帧；单段：locate → 边界帧。"""

    ops = info.get("ops") or []
    if info.get("compare"):                    # 合并目录：compare_media → 两侧取帧 / 以图找图
        return bool(info.get("compare_ready")) and (info["boundary_frames"] or info["locate"]
                                                     or any(o.startswith("INSPECT get_frames") for o in info.get("ops") or []))
    if info["diff"]:
        return info["align"] and info["diff_ready"] and info["diff_images_seen"]
    if info["align"]:
        return any(o.startswith("INSPECT locate_frames") or o == "INSPECT get_frames" for o in ops)
    if info["locate"]:
        return info["boundary_frames"]
    # 第 2 轮跟进工作区：只在指定窗口取帧 / 转写
    return any(o.startswith("INSPECT get_frames") or o == "INSPECT asr" for o in ops)


def latest_verdicts(round_tag: str, kind: str) -> Path | None:
    prefix = "verdicts_units_" if kind == "units" else "verdicts_"
    pattern = str(SFT / f"{prefix}{round_tag}_judged.*.jsonl")      # 10-14：本机 claude-cli-*，服务器 local-* 都认
    files = [Path(p) for p in glob.glob(pattern) if re.search(r"\.jsonl$", p) and not re.search(r"\.v0\d+\w*\.jsonl$", p)]
    files = [p for p in files if ".summary" not in p.name and ".pre_revise" not in p.name]
    return max(files, key=os.path.getmtime) if files else None


def verdicts(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not path:
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        run = (row.get("subject") or {}).get("run")
        if run:
            out.setdefault(run, []).append({"workspace": (row.get("subject") or {}).get("workspace_id"),
                                            "disposition": row.get("disposition"), "tally": row.get("tally") or {}})
    return out


def _fmt_v(rows: list[dict[str, Any]] | None) -> str:
    if not rows:
        return "—"
    pos = sum(r["disposition"] == "positive" for r in rows)
    errs = sum(sum(int(v) for v in r["tally"].values()) for r in rows)
    return f"{pos}/{len(rows)}（错 {errs}）"


def _mins(seconds: Any) -> str:
    return "—" if seconds in (None, "") else f"{float(seconds) / 60:.1f}"


def d2_section(table: list[dict[str, Any]], round_tag: str) -> list[str]:
    """D2（10-13）：A 族（找原片）/ C 族（核查声明）/ E 族 / L 族（照片拍摄地）案例没有老轨迹可比，单列一张表。
    A 族另看原片（案卷 family_key 里的 YouTube 编号）有没有出现在检索结果里、有没有比对。"""

    rows = [s for s in table if s["tag"] == round_tag and str(s["case"]).startswith(("fmnv_", "groundlie360_", "averimatec_e", "averimatec_l"))]
    if not rows:
        return []
    out = ["", "## D2 案例（A 族找原片 / C 族核查声明，无老轨迹）", "",
           "| 案例 | 族 | 报告 | 报告核验 | 答案核验 | 教师调用 | 墙钟(分) | SEARCH | READ_RECORD | ocr | asr | compare_media | 原片 / 最早来源进检索结果（L：实际地点） |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        case_path = SFT / "cases" / f"{s['case']}.json"
        case = _load(case_path) or {}
        image = any(str(m.get("media_type", "")).startswith("image/") for m in case.get("media") or [])
        family = ("E" if image else "A") if case.get("task_kind") == "provenance" else "L" if str(s["case"]).startswith("averimatec_l") else "C"
        in_pool = "—"
        if family == "L":
            # L 族：实际拍摄地（案卷 entity_in_evidence）有没有写进最终答案
            place = next(((c.get("params") or {}).get("text") for c in case.get("checks") or [] if c.get("kind") == "entity_in_evidence"), "")
            answer = "".join(req.read_text(encoding="utf-8", errors="replace")
                             for req in sorted((RUNS / s["run"] / "requests").glob("request_*/response.json"))[-1:])
            in_pool = ("实际地点写进答案" if place and place.lower() in answer.lower() else "答案没写实际地点") if place else "—"
        if family == "E":
            pattern = next((str(r.get("url_pattern") or "").rstrip("*") for r in case.get("required_evidence") or []
                            if r.get("key") == "earliest_source"), "")
            text = "".join(req.read_text(encoding="utf-8", errors="replace")
                           for req in sorted((RUNS / s["run"] / "requests").glob("request_*/request.json")))
            in_pool = "是" if pattern and pattern.split("/", 1)[0] in text and pattern[:60] in text else "否"
        if family == "A":
            vid = str(case.get("family_key") or "").split(":", 1)[-1]
            text = ""
            for req in sorted((RUNS / s["run"] / "requests").glob("request_*/request.json")):
                text += req.read_text(encoding="utf-8", errors="replace")
            in_pool = "是" if vid and vid in text else "否"
        ops = s["ops"]
        out.append(f"| {s['case']} | {family} | {', '.join(x or '?' for x in s['reports'])} | {_fmt_v(s['verdict_reports'])} | "
                   f"{_fmt_v(s['verdict_answer'])} | {s['llm_calls']} | {_mins(s['wall_seconds'])} | {ops.get('SEARCH', 0)} | "
                   f"{ops.get('READ_RECORD', 0)} | {ops.get('INSPECT ocr', 0)} | {ops.get('INSPECT asr', 0)} | "
                   f"{ops.get('INSPECT compare_media', 0)} | {in_pool} |")
    rep = [v for s in rows for v in (s["verdict_reports"] or [])]
    ans = [v for s in rows for v in (s["verdict_answer"] or [])]
    out += ["", f"- 报告核验 positive：{sum(v['disposition'] == 'positive' for v in rep)}/{len(rep)}；"
                f"答案核验 positive：{sum(v['disposition'] == 'positive' for v in ans)}/{len(ans)}"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="wk1")
    parser.add_argument("--before", default="r0e")
    parser.add_argument("--also", nargs="*", default=["tp2"], help="另列的轮次（只在有对应 run 的案例上出现；全部案例都有的也进汇总）")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", default=str(SFT / "walker_d1_compare"))
    args = parser.parse_args()
    manifests = manifest_rows()
    tags = [args.before, *args.also, args.round]
    v_rep = {tag: verdicts(latest_verdicts(tag, "reports")) for tag in tags}
    v_ans = {tag: verdicts(latest_verdicts(tag, "units")) for tag in tags}
    table: list[dict[str, Any]] = []
    for case in args.cases:
        for tag in tags:
            run_dir = RUNS / f"{case}_{tag}_1"
            if not run_dir.is_dir():
                continue
            stats = analyze(run_dir, manifests)
            stats["case"], stats["tag"] = case, tag
            stats["chain_ok"] = all(chain_ok(i) for i in stats["workspaces"].values()) if stats["workspaces"] else False
            stats["chain_any"] = any(chain_ok(i) for i in stats["workspaces"].values())
            stats["verdict_reports"] = v_rep[tag].get(run_dir.name)
            stats["verdict_answer"] = [r for r in v_ans[tag].get(run_dir.name, [])]
            table.append(stats)

    lines = [f"# walker {args.round} 对照 {args.before}（同案例）", "",
             f"老轨迹核验：{latest_verdicts(args.before, 'reports')}", f"walker 核验：{latest_verdicts(args.round, 'reports')}", "",
             "| 案例 | 轮次 | 调查轮数 | 决策数 | 教师调用 | 墙钟(分) | HI 动作 | 重复步 | 格式拒 | 链条走到 | 报告状态（按工作区） | 报告核验 positive | 答案核验 | 动作 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in table:
        ops = "; ".join(f"{k}×{v}" for k, v in sorted(s["ops"].items(), key=lambda kv: -kv[1]))
        chain = "全部" if s["chain_ok"] else ("部分" if s["chain_any"] else "否")
        lines.append(f"| {s['case']} | {s['tag']} | {s['gia'].get('INVESTIGATE', 0)} | {s['requests']} | {s['llm_calls']} | {_mins(s['wall_seconds'])} | {s['hi_steps']} | "
                     f"{s['repeats']} | {s['repairs']} | {chain} | {', '.join(x or '?' for x in s['reports'])} | "
                     f"{_fmt_v(s['verdict_reports'])} | {_fmt_v(s['verdict_answer'])} | {ops} |")
    lines += ["", "## 汇总（只算两边都有的案例）", ""]
    both = [c for c in args.cases if any(s["case"] == c and s["tag"] == args.before for s in table)
            and any(s["case"] == c and s["tag"] == args.round for s in table)]

    def agg(tag: str) -> dict[str, Any]:
        rows = [s for s in table if s["tag"] == tag and s["case"] in both]
        if not rows:
            return {}
        rep = [v for s in rows for v in (s["verdict_reports"] or [])]
        ans = [v for s in rows for v in (s["verdict_answer"] or [])]
        n = len(rows)
        walls = [float(s["wall_seconds"]) for s in rows if s["wall_seconds"] not in (None, "")]
        return {"runs": n, "requests_mean": round(sum(s["requests"] for s in rows) / n, 1),
                "llm_calls_mean": round(sum(s["llm_calls"] for s in rows) / n, 1),
                "llm_calls_max": max(s["llm_calls"] for s in rows),
                "wall_min_mean": round(sum(walls) / len(walls) / 60, 1) if walls else None,
                "wall_min_max": round(max(walls) / 60, 1) if walls else None,
                "hi_steps_mean": round(sum(s["hi_steps"] for s in rows) / n, 1),
                "repeats": sum(s["repeats"] for s in rows), "repairs": sum(s["repairs"] for s in rows),
                "chain_all": sum(s["chain_ok"] for s in rows),
                "runs_with_round2": sum(s["gia"].get("INVESTIGATE", 0) >= 2 for s in rows),
                "finish_with_incomplete_report": sum(1 for s in rows if s["reports"] and s["reports"][-1] == "INCOMPLETE"),
                "round2_completed": sum(1 for s in rows if s["gia"].get("INVESTIGATE", 0) >= 2 and s["reports"]
                                        and s["reports"][-1] == "COMPLETED"),
                "reports": dict(Counter(x for s in rows for x in s["reports"])),
                "report_positive": f"{sum(v['disposition'] == 'positive' for v in rep)}/{len(rep)}" if rep else "—",
                "report_errors": sum(sum(int(x) for x in v["tally"].values()) for v in rep) if rep else None,
                "answer_positive": f"{sum(v['disposition'] == 'positive' for v in ans)}/{len(ans)}" if ans else "—",
                "answer_errors": sum(sum(int(x) for x in v["tally"].values()) for v in ans) if ans else None,
                "input_tokens_mean": round(sum(s["input_tokens"] for s in rows) / n),
                "output_tokens_mean": round(sum(s["output_tokens"] for s in rows) / n)}

    cols = [args.before] + [t for t in args.also if all(any(r["case"] == c and r["tag"] == t for r in table) for c in both)] + [args.round]
    summary = {t: agg(t) for t in cols}
    keys = ["runs", "requests_mean", "llm_calls_mean", "llm_calls_max", "wall_min_mean", "wall_min_max", "hi_steps_mean",
            "repeats", "repairs", "chain_all", "runs_with_round2", "round2_completed", "finish_with_incomplete_report", "reports", "report_positive", "report_errors", "answer_positive", "answer_errors",
            "input_tokens_mean", "output_tokens_mean"]
    lines += ["| 指标 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(str(summary[t].get(k, '—')) for t in cols) + " |")
    branches = Counter()
    for s in table:
        if s["tag"] == args.round:
            branches.update(s["branches"])
    lines += ["", "## walker 分支计数（§2 覆盖矩阵，本批）", ""] + [f"- {k}：{v}" for k, v in sorted(branches.items())]
    w = summary[args.round]
    if w:
        lines += ["", "## D1 验收（计划 §8）", "",
                  f"- 重复步 0：{w['repeats']}", f"- 格式拒 0：{w['repairs']}",
                  f"- 每条都有 align → diff → 读差分（两段视频）/ locate → 边界帧（单段）：{w['chain_all']}/{w['runs']}",
                  f"- 教师调用 ≤ 6：最多 {w['llm_calls_max']}", f"- 墙钟 ≤ 5 分钟：最长 {w['wall_min_max']}",
                  f"- 报告核验通过率不低于 {args.before}：{w['report_positive']} vs {summary[args.before].get('report_positive')}"]
    lines += d2_section(table, args.round)
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps({"table": table, "summary": summary, "branches": dict(branches)},
                                                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
