"""EVID-Bench 正式 baseline（10-07）：双击 scripts\\_evid_final.cmd。

口径（10-07 负责人定，选项 A）：**正式分只报 131–222**。002–130 的答案在修正计划与找原片实验里看过、拿来诊断过，
不再算干净的测试集；131–222 从没看过。其中已降级出 test 的（used_for_revision 非空，现为 151）照旧不计。

两条，按顺序跑（同一张显卡，不能并行）：
  1  8B   qwen3vl-8b-bf16-32k，学生提示            轮次 evidF8     约 11 小时（91 个，同时 2 个）
  2  27B  qwen38-27b-32k（Q4，关思考），学生提示    轮次 evidF27    约 9 小时
两条都用现在的工具配置（找原片实验的结论：工具层改动不进 baseline，search_arm 不开），规格 hi@0.7.0 / gia@0.8.0，
3 轮，compact_16k，评测兜底 eval，检索录制回放：两个模型共用一个快照目录 experiments\\evid\\search_snapshots_final
（同一个检索世界；以后训练好的学生在同一目录回放）。

每条：先跑 2 个样本试跑（两个都崩就停这一条），再跑其余；失败的 run 不超过 10 个时自动重跑一次；最后 Sonnet 评分。
可以随时关窗口，再双击从断点继续；已评过分且全部完成的那条直接跳过。只跑其中一条：set FOREN_FINAL_MODELS=27b。

汇总写 experiments\\evid\\scores\\final_131_222.md（每跑完一条就更新一次）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _evid_night as N  # noqa: E402
import _launch_evid as L  # noqa: E402
from _launch import PY, ROOT, Log, env_setup  # noqa: E402

FIRST, LAST = 131, 222
SNAPSHOT_ROOT = L.EVID / "search_snapshots_final"
SCORES = L.EVID / "scores"
REPORT = SCORES / "final_131_222.md"
RETRY_AT_MOST = 10
DATASET = Path("plan-harness-1") / "datasets" / "EVID-Bench" / "evid_bench.json"
PAPER = ("论文参考（全量 222、他们的 agent、三评委）：GPT-5.5 点级 61.43% / 视频级 43.24%；"
         "Qwen3-VL-235B-A22B 点级 48.80%。这里是 131–222、Sonnet 单评委，只能大致对照。")


@dataclass
class Leg:
    key: str
    name: str
    round: str
    model: str
    config: Path
    extra: tuple[str, ...] = ()
    minutes_per_run: float = 14.2          # 单个 run 的墙钟（同时 2 个时，20 个约 = 20 × 该值 / 2）
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


LEGS = [
    Leg("8b", "Qwen3-VL-8B bf16", "evidF8", L.EVID8_BF16_MODEL, L.EVID8_CONFIG, minutes_per_run=14.2),
    Leg("27b", "Qwen3.8-27B Q4（关思考）", "evidF27", L.EVID27_MODEL, L.EVID27_CONFIG, extra=("--no-think",),
        minutes_per_run=12.0),
]


def final_ids() -> tuple[str, ...]:
    """131–222 里仍在 test 的样本（降级出 test 的不算）。"""

    sys.path.insert(0, str(ROOT / "scripts"))
    from _evid_score import revised_ids

    entries = json.loads((ROOT / DATASET).read_text(encoding="utf-8"))
    revised = revised_ids(ROOT / L.EVID / "cases")
    ids = [e["id"] for e in entries if FIRST <= int(str(e["id"]).split("_")[-1]) <= LAST and e["id"] not in revised]
    return tuple(sorted(ids))


def run_command(leg: Leg, ids: tuple[str, ...]) -> list[str]:
    specs = L.EVID_STAGE_SPECS["p1"]
    return [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(L.EVID / "cases"), "--runs-dir",
            str(L.EVID / "runs"), "--split", "test", "--round", leg.round, "--endpoint", L.OLLAMA, "--model", leg.model,
            *leg.extra, "--jobs", L.EVID8_JOBS, "--instruction-mode", "student", "--retry-failed", "--blocklist", "exclude",
            "--hi-spec", specs[0], "--gia-spec", specs[1],
            "--search-mode", "replay_or_record", "--snapshot-root", str(SNAPSHOT_ROOT),
            "--runtime-guards", L.EVID_GUARDS, "--state-renderer", L.EVID_RENDER,
            "--ids", *(f"evid:{i}" for i in ids)]


def score_command(leg: Leg, ids: tuple[str, ...]) -> list[str]:
    return [PY, "scripts/_evid_score.py", "--round", leg.round, "--ids", *ids, "--judges", L.EVID_SONNET]


def run_leg(log: Log, leg: Leg, ids: tuple[str, ...]) -> None:
    os.environ["FOREN_CONFIG"] = str(leg.config)
    todo = [s for s in ids if not N.done(leg.round, s)]
    if todo:
        if N.free_gb() < N.MIN_FREE_GB:
            leg.status, leg.notes = "skipped_disk", [f"磁盘只剩 {N.free_gb():.1f}GB"]
            log.say(f"跳过 {leg.name}：{leg.notes[-1]}")
            return
        if not N.ollama_ready(log):
            leg.status, leg.notes = "skipped_ollama", ["Ollama 不在"]
            return
        hours = leg.minutes_per_run * len(todo) / max(1, int(L.EVID8_JOBS)) / 60
        log.say(f"==== {leg.name}（轮次 {leg.round}）：待跑 {len(todo)} / {len(ids)} 个，预计约 {hours:.1f} 小时")
        if len(todo) == len(ids):
            canary = tuple(todo[:N.CANARY])
            log.run(f"[{leg.key}] 试跑 {len(canary)} 个", run_command(leg, canary), show=True)
            if not any(N.done(leg.round, s) for s in canary):
                leg.status = "failed_canary"
                leg.notes.append(f"试跑的 {', '.join(canary)} 一个都没跑完（看 experiments\\sft_data\\batch_logs\\evid_*_{leg.round}_1.log）")
                log.say(f"停掉 {leg.name}：{leg.notes[-1]}")
                return
        rest = tuple(s for s in ids if not N.done(leg.round, s))
        if rest:
            log.run(f"[{leg.key}] 其余 {len(rest)} 个", run_command(leg, rest), show=True)
        failed = tuple(s for s in ids if not N.done(leg.round, s))
        if failed and len(failed) <= RETRY_AT_MOST:
            log.run(f"[{leg.key}] 失败的 {len(failed)} 个重跑一次", run_command(leg, failed), show=True)
    finished = N.done_count(leg.round, ids)
    leg.status = "done" if finished == len(ids) else "partial"
    if finished < len(ids):
        leg.notes.append(f"完成 {finished}/{len(ids)}（没完成的按 0 计分）")
    log.run(f"[{leg.key}] Sonnet 评分（{finished}/{len(ids)} 个完成）", score_command(leg, ids), show=True)


def leg_row(leg: Leg, ids: tuple[str, ...]) -> dict:
    summary = N.score_summary(leg.round) or {}
    attempted = summary.get("all_planned") or {}
    funnel = summary.get("funnel") or {}
    behavior = summary.get("behavior") or {}
    cost = {}
    path = ROOT / N.MANIFESTS / f"{leg.round}.jsonl"
    if path.is_file():
        rows = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("status") == "done":
                rows[row["run"]] = row
        done_rows = [r for name, r in rows.items() if name in {f"evid_{s}_{leg.round}_1" for s in ids}]
        if done_rows:
            cost = {"minutes": round(sum(float(r.get("wall_seconds") or 0) for r in done_rows) / len(done_rows) / 60, 1),
                    "tokens": int(sum(float(r.get("input_tokens") or 0) for r in done_rows) / len(done_rows))}
    return {"leg": leg.name, "round": leg.round, "done": N.done_count(leg.round, ids), "n": len(ids),
            "scored": bool(summary), "point_acc": attempted.get("point_acc"), "video_acc": attempted.get("video_acc"),
            "gt_points": attempted.get("gt_points"), "passed": attempted.get("passed_points"),
            "in_pool": funnel.get("source_in_pool"), "downloaded": funnel.get("source_downloaded"),
            "compared": funnel.get("source_compared"), "repeat_rate": behavior.get("repeat_rate"),
            "crashes": behavior.get("crashes"), "by_category": summary.get("by_category") or {}, **cost}


def write_report(log: Log, ids: tuple[str, ...]) -> Path:
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    dash = lambda v: "—" if v is None else v  # noqa: E731
    rows = [leg_row(leg, ids) for leg in LEGS]
    lines = [f"# EVID-Bench 正式 baseline：131–222（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"- 样本：131–222 里仍在 test 的 {len(ids)} 个（已降级出 test 的不算）；002–130 用于过诊断，不进正式分。",
             "- 规格 hi@0.7.0 / gia@0.8.0（英文），学生提示，3 轮，compact_16k，评测兜底 eval；工具为现状配置（不开检索臂）；"
             "两个模型共用检索快照 experiments\\evid\\search_snapshots_final。",
             "- 评委：Claude Sonnet 单评委，官方 EVID 评分提示词；点级 = 通过的 GT 点 / 全部 GT 点，视频级 = 全部点都通过的视频比例；"
             "没跑完的样本按 0。", "",
             "| 模型 | 轮次 | 完成 | 点级 | 视频级 | 通过 / GT 点 | 原片进池 | 下载 | 比对 | 重复调用率 | 崩溃 | 分钟/run | 输入 token/run | 状态 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for leg, row in zip(LEGS, rows):
        state = leg.status + ("：" + "；".join(leg.notes) if leg.notes else "")
        lines.append(f"| {row['leg']} | {row['round']} | {row['done']}/{row['n']} | {pct(row['point_acc'])} | {pct(row['video_acc'])} | "
                     f"{dash(row['passed'])} / {dash(row['gt_points'])} | {dash(row['in_pool'])} | {dash(row['downloaded'])} | "
                     f"{dash(row['compared'])} | {pct(row['repeat_rate'])} | {row['crashes'] or ('无' if row['scored'] else '—')} | "
                     f"{row.get('minutes', '—')} | {row.get('tokens', '—')} | {state} |")
    lines += ["", PAPER, "", "## 按类别（点级 / 视频级）", "", "| 类别 | " + " | ".join(r["leg"] for r in rows) + " |",
              "|---|" + "---|" * len(rows)]
    categories = sorted({c for r in rows for c in r["by_category"]})
    for category in categories:
        cells = [f"{pct((r['by_category'].get(category) or {}).get('point_acc'))} / "
                 f"{pct((r['by_category'].get(category) or {}).get('video_acc'))}（{(r['by_category'].get(category) or {}).get('videos', 0)} 个）"
                 for r in rows]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "逐样本、行为面板与找原片漏斗：" + "；".join(f"experiments\\evid\\scores\\{r['round']}.sonnet.subset.md" for r in rows)]
    path = ROOT / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.with_suffix(".json")).write_text(json.dumps({"ids": list(ids), "rows": rows,
                                                       "legs": {l.key: {"status": l.status, "notes": l.notes} for l in LEGS}},
                                                      ensure_ascii=False, indent=2), encoding="utf-8")
    log.say(f"汇总写在 {path}")
    return path


def main() -> int:
    env_setup()
    ids = final_ids()
    log = N.NightLog("_EVID正式baseline日志.txt", f"EVID-Bench 正式 baseline：131–222 共 {len(ids)} 个")
    chosen = [x.strip().lower() for x in (os.environ.get("FOREN_FINAL_MODELS") or "8b,27b").split(",") if x.strip()]
    legs = [leg for leg in LEGS if leg.key in chosen]
    print(f"EVID 正式 baseline：131–222 里 {len(ids)} 个样本，依次跑 {' → '.join(l.name for l in legs)}"
          f"（8B 约 11 小时、27B 约 9 小时），每条跑完用 Sonnet 评分。")
    print(f"可以随时关窗口，再双击从断点继续。日志 {log.path.name}；汇总 {REPORT}")
    if N.free_gb() < N.MIN_FREE_GB:
        log.say(f"磁盘只剩 {N.free_gb():.1f}GB，少于 {N.MIN_FREE_GB}GB；先清出空间再双击。")
        return 1
    log.run("[0] 检索后端与下载自检", [PY, "scripts/_evid_search_probe.py"], show=True)
    log.run("[0] EVID-Bench 标注 → 案卷", [PY, "scripts/_evid_build_cases.py"])
    for leg in LEGS:
        if leg not in legs:
            leg.status = "not_selected"
            continue
        if N.score_summary(leg.round) and N.done_count(leg.round, ids) == len(ids):
            leg.status = "done"
            leg.notes.append("之前已跑完并评分")
            log.say(f"{leg.name}（{leg.round}）之前已跑完并评分，跳过。")
            continue
        if leg.key == "8b":
            if not L.bf16_preflight(log):
                leg.status, leg.notes = "skipped_model", ["bf16 模型没就绪"]
                continue
        else:
            if L.EVID27_MODEL not in (L.ollama_models() or []):
                leg.status, leg.notes = "skipped_model", [f"本机没有 {L.EVID27_MODEL}"]
                log.say(f"跳过 {leg.name}：{leg.notes[-1]}")
                continue
            if not L.evid27_probe(log):
                leg.status, leg.notes = "skipped_model", ["27B 试答失败"]
                continue
        run_leg(log, leg, ids)
        write_report(log, ids)
    path = write_report(log, ids)
    L._ollama_ps(log, "[末] 显存检查")
    log.finish(f"汇总 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
