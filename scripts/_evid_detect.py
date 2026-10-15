"""EVID-Bench 131–222 接入 DETECT 后重跑（10-08）：双击 scripts\\_evid_detect.cmd。

与正式 baseline（scripts/_evid_final.py，轮次 evidF8 / evidF27）只差一项：run_cases 加 --detect on（HI 工具目录多一个
DETECT：整段扫描 AI 生成信号 generation_scan + 单窗 generated_visual；GIA 提示加一段说明）。其余一律相同：131–222 里
仍在 test 的 91 个、规格 hi@0.7.0 / gia@0.8.0（英文）、学生提示、3 轮、compact_16k、评测兜底 eval、同一个检索快照目录
experiments\\evid\\search_snapshots_final（命中就回放，同一个检索世界）、Sonnet 单评委。

步骤：
  [0] DETECT 单元测试（tests/test_detect.py + tests/test_detect_scan_20261008.py）——不过就停
  [1] 真实素材参照：configs/detect_reference_d3_clipb32.json 不在就先建（EVID 002–130 每隔一个，约 1–2 小时，CPU）
  [2] 27B（轮次 evidD27）→ 8B（轮次 evidD8）：先试跑 2 个，再跑其余，失败 ≤10 个自动重跑一次，最后 Sonnet 评分
  [3] 对照表 experiments\\evid\\scores\\detect_131_222.md：evidD* vs evidF*，总分 / 按类别 / DETECT 用了几次、报警几处

可以随时关窗口，再双击从断点继续。只跑一条：set FOREN_FINAL_MODELS=27b。日志 _EVID_DETECT日志.txt。
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _evid_final as F  # noqa: E402
import _evid_night as N  # noqa: E402
import _launch_evid as L  # noqa: E402
from _launch import PY, ROOT, env_setup  # noqa: E402

DETECT_CONFIG = Path("configs") / "detect.evid.json"
REFERENCE = Path("configs") / "detect_reference_d3_clipb32.json"
REPORT = F.SCORES / "detect_131_222.md"
DETECT_ARGS = ("--detect", "on", "--detect-config", str(DETECT_CONFIG))

LEGS = [
    F.Leg("27b", "Qwen3.8-27B Q4（关思考）+ DETECT", "evidD27", L.EVID27_MODEL, L.EVID27_CONFIG,
          extra=("--no-think", *DETECT_ARGS), minutes_per_run=13.0),
    F.Leg("8b", "Qwen3-VL-8B bf16 + DETECT", "evidD8", L.EVID8_BF16_MODEL, L.EVID8_CONFIG,
          extra=DETECT_ARGS, minutes_per_run=15.0),
]
BASELINE = {"27b": "evidF27", "8b": "evidF8"}


def categories() -> dict[str, str]:
    entries = json.loads((ROOT / F.DATASET).read_text(encoding="utf-8"))
    return {str(e["id"]): str((e.get("task") or {}).get("category") or "?") for e in entries}


def detect_usage(round_tag: str, ids: tuple[str, ...]) -> dict[str, dict]:
    """按类别：跑完的 run 里调用过 generation_scan 的比例、报警区间数（来自 run_config.detect.calls）。"""

    cats = categories()
    out: dict[str, dict] = defaultdict(lambda: {"runs": 0, "scanned": 0, "scans": 0, "flagged": 0, "runs_flagged": 0})
    for sample in ids:
        path = N.run_dir(round_tag, sample) / "run_config.json"
        if not path.is_file():
            continue
        try:
            calls = (json.loads(path.read_text(encoding="utf-8")).get("detect") or {}).get("calls") or {}
        except ValueError:
            continue
        for key in (cats.get(sample, "?"), "all"):
            row = out[key]
            row["runs"] += 1
            row["scanned"] += bool(calls.get("generation_scan"))
            row["scans"] += int(calls.get("generation_scan") or 0)
            row["flagged"] += int(calls.get("flagged_spans") or 0)
            row["runs_flagged"] += bool(calls.get("flagged_spans"))
    return dict(out)


def write_report(log, ids: tuple[str, ...], legs: list) -> Path:
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"# EVID-Bench 131–222：接入 DETECT 前后（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"- 样本：131–222 里仍在 test 的 {len(ids)} 个；设置与正式 baseline 完全相同，只多 --detect on。",
             "- 参照：configs/detect_reference_d3_clipb32.json（EVID 002–130 真实素材）。评委 Sonnet 单评委，没跑完按 0。", "",
             "| 模型 | 轮次 | 完成 | 点级 | 视频级 | 通过 / GT 点 | 原片进池 | 下载 | 比对 | 分钟/run | 输入 token/run |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for leg in legs:
        base = F.Leg(leg.key, leg.name.replace(" + DETECT", ""), BASELINE[leg.key], leg.model, leg.config)
        for item in (base, leg):
            row = F.leg_row(item, ids)
            rows[item.round] = row
            lines.append(f"| {row['leg']} | {row['round']} | {row['done']}/{row['n']} | {pct(row['point_acc'])} | "
                         f"{pct(row['video_acc'])} | {row['passed'] if row['passed'] is not None else '—'} / "
                         f"{row['gt_points'] if row['gt_points'] is not None else '—'} | {row['in_pool'] if row['in_pool'] is not None else '—'} | "
                         f"{row['downloaded'] if row['downloaded'] is not None else '—'} | {row['compared'] if row['compared'] is not None else '—'} | "
                         f"{row.get('minutes', '—')} | {row.get('tokens', '—')} |")
    lines += ["", "## 按类别（点级 / 视频级）", "", "| 类别 | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    cats = sorted({c for r in rows.values() for c in r["by_category"]})
    for category in cats:
        cells = []
        for row in rows.values():
            item = row["by_category"].get(category) or {}
            cells.append(f"{pct(item.get('point_acc'))} / {pct(item.get('video_acc'))}（{item.get('videos', 0)}）")
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "## DETECT 用量（跑完的 run）", "", "| 轮次 | 类别 | run | 调过 generation_scan | 扫描次数 | 有报警的 run | 报警区间 |",
              "|---|---|---|---|---|---|---|"]
    for leg in legs:
        for category, item in sorted(detect_usage(leg.round, ids).items()):
            lines.append(f"| {leg.round} | {category} | {item['runs']} | {item['scanned']} | {item['scans']} | "
                         f"{item['runs_flagged']} | {item['flagged']} |")
    lines += ["", "逐样本：experiments\\evid\\scores\\<轮次>.sonnet.subset.md"]
    path = ROOT / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"ids": list(ids), "rows": rows,
                                                     "usage": {l.round: detect_usage(l.round, ids) for l in legs}},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    log.say(f"对照表写在 {path}")
    return path


SMOKE_ID, SMOKE_ROUND = "video_002", "evidDsmoke"


def smoke(log, legs: list) -> None:
    """开跑前在一个开发段样本（video_002，不计正式分）上跑一次，确认 DETECT 在真实 run 里能被调用、不报错。"""

    leg = next((l for l in legs if l.key == "27b"), legs[0] if legs else None)
    if leg is None:
        return
    if not N.done(SMOKE_ROUND, SMOKE_ID):
        if not N.ollama_ready(log):
            log.say("[1b] Ollama 不在，冒烟跳过")
            return
        probe = F.Leg(leg.key, leg.name, SMOKE_ROUND, leg.model, leg.config, extra=leg.extra)
        os.environ["FOREN_CONFIG"] = str(leg.config)
        log.run(f"[1b] 冒烟：开发段 {SMOKE_ID} 一个 run（{SMOKE_ROUND}，{leg.name}）", F.run_command(probe, (SMOKE_ID,)), show=True)
    run = N.run_dir(SMOKE_ROUND, SMOKE_ID)
    try:
        detect = json.loads((run / "run_config.json").read_text(encoding="utf-8")).get("detect") or {}
    except (OSError, ValueError):
        detect = {}
    codes: dict[str, int] = defaultdict(int)
    trajectory = run / "trajectory.jsonl"
    if trajectory.is_file():
        for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"DETECT"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            observation = (event.get("payload") or {}).get("observation") or {}
            if observation.get("action") == "DETECT":
                details = observation.get("details") or {}
                codes[(details.get("feedback") or {}).get("code") or details.get("status") or "?"] += 1
    log.say(f"[1b] 冒烟结果：完成={N.done(SMOKE_ROUND, SMOKE_ID)}；DETECT 配置={'开' if detect.get('enabled') else '关'}，"
            f"参照={((detect.get('reference') or {}).get('name'))}；调用={json.dumps(detect.get('calls'), ensure_ascii=False)}；"
            f"观察状态={json.dumps(dict(codes), ensure_ascii=False)}")


def main() -> int:
    print("10-12：DETECT 已并入 SEARCH / INSPECT（detect/ 已移走），这个脚本不再能跑。"
          "合并后的 A/B 请双击 scripts\\_merge_ab.cmd；旧结果（evidC27e / evidC27G）照常可读。")
    return 1
    env_setup()
    ids = F.final_ids()
    log = N.NightLog("_EVID_DETECT日志.txt", f"EVID-Bench 131–222 接入 DETECT 重跑：{len(ids)} 个")
    chosen = [x.strip().lower() for x in (os.environ.get("FOREN_FINAL_MODELS") or "27b,8b").split(",") if x.strip()]
    legs = [leg for leg in LEGS if leg.key in chosen]
    print(f"EVID 131–222 接入 DETECT 重跑：{len(ids)} 个样本，依次 {' → '.join(l.name for l in legs)}。"
          f"可以随时关窗口，再双击续跑。日志 {log.path.name}")
    if N.free_gb() < N.MIN_FREE_GB:
        log.say(f"磁盘只剩 {N.free_gb():.1f}GB，少于 {N.MIN_FREE_GB}GB；先清出空间再双击。")
        return 1
    code = log.run("[0] DETECT 单元测试", [PY, "-m", "pytest", "tests/test_detect.py", "tests/test_detect_scan_20261008.py",
                                        "-q", "-o", "addopts=", "-p", "no:cacheprovider", "--basetemp=.pytest-detect-20261008"],
                   show=True)
    if code != 0:
        log.finish("DETECT 单元测试没过，停。")
        return 1
    if not (ROOT / REFERENCE).is_file():
        code = log.run("[1] 建真实素材参照（EVID 002–130 每隔一个）", [PY, "scripts/detect_build_reference.py"], show=True)
        if code != 0 or not (ROOT / REFERENCE).is_file():
            log.finish("参照没建成，停。")
            return 1
    else:
        log.say(f"[1] 参照已在 {REFERENCE}，跳过")
    if os.environ.get("FOREN_DETECT_ONLY_REFERENCE"):
        log.finish("只建参照（FOREN_DETECT_ONLY_REFERENCE），不跑 EVID。")
        return 0
    log.run("[0] 检索后端与下载自检", [PY, "scripts/_evid_search_probe.py"], show=True)
    log.run("[0] EVID-Bench 标注 → 案卷", [PY, "scripts/_evid_build_cases.py"])
    smoke(log, legs)
    for leg in legs:
        if N.score_summary(leg.round) and N.done_count(leg.round, ids) == len(ids):
            leg.status = "done"
            log.say(f"{leg.name}（{leg.round}）之前已跑完并评分，跳过。")
            continue
        if leg.key == "8b":
            if not L.bf16_preflight(log):
                leg.status = "skipped_model"
                continue
        else:
            if L.EVID27_MODEL not in (L.ollama_models() or []):
                leg.status = "skipped_model"
                log.say(f"跳过 {leg.name}：本机没有 {L.EVID27_MODEL}")
                continue
            if not L.evid27_probe(log):
                leg.status = "skipped_model"
                continue
        F.run_leg(log, leg, ids)
        write_report(log, ids, legs)
    path = write_report(log, ids, legs)
    L._ollama_ps(log, "[末] 显存检查")
    log.finish(f"对照表 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
