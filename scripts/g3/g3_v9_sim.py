"""g3 阶段 2（P2）：V9 变体（max_evidence_cards=6 + max_history_entries=4）的离线触发模拟——只读已有 run，不跑 walker、不调教师。

照《walker_wk8》7.3 的办法（用已有回放看"交报告前要引用的文字结果是否掉出视图"），具体做法照
datagen/status_rewrite.walker_gaps：按顺序重放一个 run 的 HI 请求，把每一步的 payload 与 rendered_payload 里的
model_view 按 V9 截短（evidence_view 留前 6 张——卡片本来就按显式材料 / 焦点 / 相关性 / 新旧排好序，截短 = 取前 6；
history 留最后 4 条），喂给同一个 walker.decide_hi（memory 跨步保留，与真跑一样记得早先的观察），看交报告那一步会不会改成
Q1_query_memory_before_report。

近似之处（报告里写明）：V9 真跑时前面各步看到的视图也变短，walker 的决定可能更早岔开——所以记录"第一次与原 run 不同的
步"；在交报告那步之前就岔开的，标 diverged（结果不确定）。渲染：rendered_payload 里的 model_view 同样截短，不重新渲染。

    python scripts\\g3\\g3_v9_sim.py                  # 底案池 = 主目录 r1 / .wt_g1 r1g1 / .wt_g2 r1g2 的 B 族 run（VideoSham、VCSL）
    python scripts\\g3\\g3_v9_sim.py --calibrate      # 只跑 r1 的两个 V9 底案（videosham_1028 真跑触发、vcsl_pair_5 没触发）

输出 claude/_g3_work/v9_sim.json。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
RUN_DIRS = [(MAIN / "experiments" / "sft_data" / "runs", "r1"),
            (MAIN / ".wt_g1" / "experiments" / "sft_data" / "runs", "r1g1"),
            (MAIN / ".wt_g2" / "experiments" / "sft_data" / "runs", "r1g2")]
OUT = ROOT / "claude" / "_g3_work" / "v9_sim.json"
CARDS, HISTORY = 6, 4
BASE_RE = re.compile(r"^(videosham_\d+|vcsl_pair_(?:g1_)?\d+)$")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def v9(view: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(view)
    mv = out.get("model_view")
    if isinstance(mv, dict):
        if isinstance(mv.get("evidence_view"), list):
            mv["evidence_view"] = mv["evidence_view"][:CARDS]
        if isinstance(mv.get("history"), list) and len(mv["history"]) > HISTORY:
            mv["history"] = mv["history"][-HISTORY:]
            mv["context_truncated"] = True
    return out


def simulate(run_dir: Path) -> dict[str, Any]:
    from foren_searcher.datagen import walker as w

    memory: dict[str, dict[str, Any]] = {}
    out: dict[str, Any] = {"run": run_dir.name, "workspaces": {}}
    reqs = sorted((run_dir / "requests").glob("request_*"), key=lambda p: int(p.name.split("_")[-1]))
    for rdir in reqs:
        request = load(rdir / "request.json") or {}
        payload = request.get("payload") or {}
        ws = str(payload.get("workspace_id") or "")
        if not ws:
            continue
        rule = ((load(rdir / "transport.json") or {}).get("oracle") or {}).get("rule")
        mem = memory.setdefault(ws, {})
        if ws.startswith("W1-") is False and "run_family" not in mem:
            first = next((m for k, m in memory.items() if k.startswith("W1-") and "run_family" in m), None)
            if first:
                mem["run_family"] = first["run_family"]
        info = out["workspaces"].setdefault(ws, {"steps": 0, "diverged_at": None, "report_rule": None, "sim_at_report": None,
                                                "q1_anywhere": False, "remaining_at_report": None})
        if info["report_rule"] is not None:
            continue
        info["steps"] += 1
        rendered = request.get("rendered_payload")
        try:
            decision = w.decide_hi(v9(payload), mem, v9(rendered) if isinstance(rendered, dict) else None)
            sim_rule = decision.rule
        except Exception as exc:  # noqa: BLE001
            sim_rule = f"error: {type(exc).__name__}: {str(exc)[:120]}"
        if sim_rule == "Q1_query_memory_before_report":
            info["q1_anywhere"] = True
            info["q1_request"] = rdir.name
            info["q1_hidden"] = mem.get("memory_hidden")
        is_report = bool(rule) and (rule.endswith("_report") or "report" in rule.split("_", 1)[-1] and "before_report" not in rule)
        if sim_rule != rule and info["diverged_at"] is None and not (sim_rule == "Q1_query_memory_before_report" and is_report):
            info["diverged_at"] = {"request": rdir.name, "recorded": rule, "sim": sim_rule}
        if is_report:
            info["report_rule"] = rule
            info["sim_at_report"] = sim_rule
            info["remaining_at_report"] = ((payload.get("model_view") or {}).get("runtime_constraints") or {}).get("remaining_turns")
    w1 = next((v for k, v in out["workspaces"].items() if k.startswith("W1-")), None) or {}
    triggered = w1.get("sim_at_report") == "Q1_query_memory_before_report"
    clean = triggered and w1.get("diverged_at") is None
    out["verdict"] = "trigger" if clean else ("trigger_after_divergence" if triggered else
                                             ("q1_elsewhere" if any(v["q1_anywhere"] for v in out["workspaces"].values()) else "no"))
    return out


def pool() -> list[tuple[str, Path]]:
    """每个底案取最新一批的 W1 run（r1g2 > r1g1 > r1），只看 done 的。"""

    best: dict[str, tuple[int, Path]] = {}
    for prio, (folder, tag) in enumerate(RUN_DIRS):
        if not folder.is_dir():
            continue
        for run in folder.glob(f"*_{tag}_1"):
            base = run.name[: -len(f"_{tag}_1")]
            if not BASE_RE.match(base) or not (run / "result.json").is_file():
                continue
            if base not in best or prio > best[base][0]:
                best[base] = (prio, run)
    return sorted((b, p) for b, (_, p) in best.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        runs = [(b, RUN_DIRS[0][0] / f"{b}_r1_1") for b in ("videosham_1028", "vcsl_pair_5")]
    else:
        runs = pool()
    results = []
    for base, run in runs:
        res = simulate(run)
        res["base"] = base
        results.append(res)
        w1 = next((v for k, v in res["workspaces"].items() if k.startswith("W1-")), {})
        print(f"{base:24s} {run.parent.parent.parent.parent.name if 'wt_' in str(run) else 'main'}/{run.name}: {res['verdict']:26s} "
              f"report {w1.get('report_rule')} → {w1.get('sim_at_report')}; 剩 {w1.get('remaining_at_report')}；"
              f"岔开 {w1.get('diverged_at')}", flush=True)
    summary = {"runs": len(results), **{v: sum(1 for r in results if r["verdict"] == v)
                                        for v in ("trigger", "trigger_after_divergence", "q1_elsewhere", "no")}}
    payload = {"calibrate": args.calibrate, "cards": CARDS, "history": HISTORY, "summary": summary, "results": results}
    if not args.calibrate:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        OUT.with_name("v9_sim_calibrate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
