"""EVID DETECT v2 小规模 A/B：双击 scripts\\_evid_detect2.cmd（全量：_evid_detect2_full.cmd）。

DETECT v2 = find_source / compare_source（与候选原片逐帧比对：原片里没有的片段、同一时刻不同的区域、同一位置换了人的脸，
附对照图）+ generation_scan v2（无原片时：恰好 10 s 的镜头）。配置 configs/detect.v2.json。

10-11 小 A/B（v2.2 = evidC27c，v2.3 = evidC27d；同一批 14 个，见 AB_IDS）：全量 evidC27F 里原片进了候选池却没下载 / 没试到的 9 个 AI 类样本、
138（没找到原片就写"未见剪辑"的回退）和 4 个已得分的对照。设 FOREN_DETECT2_DLPROBE=1 时先做原片下载自检（[0b]）。
其余设置与正式 baseline evidF27 相同（27B Q4 关思考、hi@0.7.0 / gia@0.8.0、3 轮、compact_16k、同一检索快照、Sonnet 单评委）。
对照表：experiments\\evid\\scores\\detect2_ab_<轮次>.md（与 evidF27 / evidC27F 同一批样本对比）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _evid_final as F  # noqa: E402
import _evid_night as N  # noqa: E402
import _launch_evid as L  # noqa: E402
from _launch import PY, ROOT, env_setup  # noqa: E402

DETECT_CONFIG = Path("configs") / "detect.v2.json"
REPORT = None   # 见 report_path()
# 10-10 第一 / 二轮小 A/B（evidC27 / evidC27b）的 12 个
AB1_IDS = ("video_154", "video_168", "video_169", "video_174", "video_180", "video_186", "video_189", "video_194",
           "video_199", "video_215", "video_159", "video_164")
# 10-11 v2.2 小 A/B（evidC27c）：全量 evidC27F 里原片进了池却没下载 / 没试到的 AI 类 9 个，
# 138（没找到原片就写成"未见剪辑"，4/5 → 0/5），以及 4 个已经得分的对照（防止改坏）
AB_IDS = ("video_172", "video_175", "video_186", "video_189", "video_194", "video_198", "video_199", "video_205",
          "video_208",                                             # 原片在池里、没比对上
          "video_138",                                             # 非 AI 类的回退
          "video_154", "video_156", "video_180", "video_209")      # 对照：无原片扫描 / 插入 / 换脸 / 局部改动
FULL = bool(os.environ.get("FOREN_DETECT2_FULL"))        # 确认有效后：131–222 全量
ROUND = os.environ.get("FOREN_DETECT2_ROUND") or ("evidC27G" if FULL else "evidC27e")
# 读过视频 / 轨迹 / GT 的样本：全量时另报不含它们的干净子集
TOUCHED = {f"video_{i}" for i in (152, 155, 160, 163, 167, 171, 175, 177, 178, 179, 181, 184, 187, 201, 205, 210, 213, 216,
                                  220, 154, 159, 164, 168, 169, 174, 180, 186, 189, 194, 199, 215,
                                  138, 156, 172, 198, 208, 209)}
LEG = F.Leg("27b", "Qwen3.8-27B Q4（关思考）+ DETECT v2", ROUND, L.EVID27_MODEL, L.EVID27_CONFIG,
            extra=("--no-think", "--detect", "on", "--detect-config", str(DETECT_CONFIG)), minutes_per_run=14.0)
BASELINES = tuple(r for r in (("evidF27", "evidD27", "evidC27F") if FULL else ("evidF27", "evidC27F", "evidC27c", "evidC27d"))
                  if r != ROUND)       # 全量只和全量比（c / d 只有 14 个样本）


def per_sample(round_tag: str) -> dict[str, dict]:
    path = ROOT / F.SCORES / f"{round_tag}.sonnet.subset.jsonl"
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r
    return out


def detect_calls(round_tag: str, sample: str) -> dict:
    path = N.run_dir(round_tag, sample) / "run_config.json"
    try:
        return (json.loads(path.read_text(encoding="utf-8")).get("detect") or {}).get("calls") or {}
    except (OSError, ValueError):
        return {}


def report_path() -> Path:
    return F.SCORES / f"detect2_ab_{ROUND}.md"


CANARY = ("video_189", "video_194")      # 原片在池里、上一轮没比对上的两个：全量先跑它们


def sample_ids() -> tuple[str, ...]:
    if not FULL:
        return AB_IDS
    ids = F.final_ids()
    return tuple([s for s in CANARY if s in ids] + [s for s in ids if s not in CANARY])


def write_report(log) -> Path:
    ids = sample_ids()
    rows = {r: per_sample(r) for r in (LEG.round, *BASELINES)}
    lines = [f"# DETECT v2 小规模 A/B（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"{LEG.round} = evidF27 的设置 + --detect on（configs/detect.v2.json）。同一批样本、同一检索快照、Sonnet 单评委。"
             "evidC27F = v2.1 全量；evidC27c = v2.2（find_source 新排序、SEARCH 后自动跑、下载换客户端、没找到原片≠真实）；"
             "evidC27d = v2.3（换脸两档阈值与位置、小而稳定的改动区域、低清单路视频重下；看图命名因 visual_qa 未启用而没生效）；"
             "evidC27e 起 = v2.4（比对图用本 run 的 27B 端点逐张描述）。", "",
             "| 样本 | 类型 | " + " | ".join((*BASELINES, LEG.round)) + " | 原片进池/下载/比对 | DETECT 调用 |",
             "|---|---|" + "---|" * (len(BASELINES) + 1) + "---|---|"]
    totals = {r: [0, 0] for r in rows}
    for s in ids:
        cells = []
        kind = "?"
        for r in (*BASELINES, LEG.round):
            item = rows[r].get(s)
            if item:
                kind = item.get("type", kind)
                cells.append(f"{item['passed_gt']}/{item['n_gt']}")
                totals[r][0] += item["passed_gt"]
                totals[r][1] += item["n_gt"]
            else:
                cells.append("—")
        f = (rows[LEG.round].get(s) or {}).get("funnel") or {}
        funnel = "/".join("✓" if f.get(k) else "·" for k in ("source_in_pool", "source_downloaded", "source_compared"))
        lines.append(f"| {s} | {kind} | {' | '.join(cells)} | {funnel} | {json.dumps(detect_calls(LEG.round, s), ensure_ascii=False)} |")
    lines.append("| 合计 | | " + " | ".join(f"{totals[r][0]}/{totals[r][1]}" for r in (*BASELINES, LEG.round)) + " | | |")
    if FULL:
        lines += ["", "## 按类别 / 干净子集（点级：通过 / GT 点）", "", "| 子集 | " + " | ".join((*BASELINES, LEG.round)) + " |",
                  "|---|" + "---|" * (len(BASELINES) + 1)]
        cats = sorted({(rows[LEG.round].get(s) or {}).get("category", "?") for s in ids})
        subsets = [(c, [s for s in ids if (rows[LEG.round].get(s) or {}).get("category") == c]) for c in cats]
        subsets += [("ai_generation 未读过的", [s for s in ids if (rows[LEG.round].get(s) or {}).get("category") == "ai_generation"
                                            and s not in TOUCHED]),
                    ("全部未读过的", [s for s in ids if s not in TOUCHED]), ("全部", list(ids))]
        for name, members in subsets:
            cells = []
            for r in (*BASELINES, LEG.round):
                got = sum((rows[r].get(s) or {}).get("passed_gt", 0) for s in members)
                tot = sum((rows[r].get(s) or {}).get("n_gt", 0) for s in members)
                cells.append(f"{got}/{tot} = {100 * got / tot:.1f}%" if tot else "—")
            lines.append(f"| {name}（{len(members)} 个） | " + " | ".join(cells) + " |")
    path = ROOT / report_path()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.say(f"对照表 {path}")
    return path


def main() -> int:
    print("10-12：DETECT 已并入 SEARCH / INSPECT（detect/ 已移走），这个脚本不再能跑。"
          "合并后的 A/B 请双击 scripts\\_merge_ab.cmd；旧结果（evidC27e / evidC27G）照常可读。")
    return 1
    env_setup()
    ids = sample_ids()
    log = N.NightLog("_EVID_DETECT2日志.txt", f"DETECT v2 {'全量' if FULL else '小规模 A/B'}：{len(ids)} 个（{LEG.round}）")
    print(f"DETECT v2 {'全量' if FULL else 'A/B'}：{len(ids)} 个样本（{LEG.round}）。可以随时关窗口，再双击续跑。日志 {log.path.name}")
    tests = ["tests/test_detect_compare_20261010.py", "tests/test_detect_find_20261011.py",
             "tests/test_evidb7_smoke_fixes_20261005.py", "tests/test_detect_scan_20261008.py"]
    code = 1
    for attempt in range(2):
        # 10-11：evidC27e 第一次双击时测试夹具写临时 mp4 偶发 FileNotFoundError（Windows 上复用同一个 basetemp 目录，
        # 上次留下的文件没删干净 / 被占用），整轮被挡。每次用新的临时目录，失败再试一次。
        basetemp = ROOT / "outputs" / "pytest_tmp" / f"detect_{time.strftime('%Y%m%d_%H%M%S')}_{attempt}"
        basetemp.parent.mkdir(parents=True, exist_ok=True)      # pytest 只建最后一级（20:06 那次 WinError 3 就是因为上级不存在）
        code = log.run(f"[0] DETECT 单元测试{'（重试）' if attempt else ''}",
                       [PY, "-m", "pytest", *tests, "-q", "-o", "addopts=", "-p", "no:cacheprovider", f"--basetemp={basetemp}"],
                       show=True)
        if code == 0:
            break
    if code != 0:
        log.finish("DETECT 单元测试没过，停。")
        return 1
    log.run("[0] EVID-Bench 标注 → 案卷", [PY, "scripts/_evid_build_cases.py"])
    if L.EVID27_MODEL not in (L.ollama_models() or []) or not L.evid27_probe(log):
        log.finish(f"27B（{L.EVID27_MODEL}）不可用，停。")
        return 1
    if not FULL and os.environ.get("FOREN_DETECT2_DLPROBE"):      # 不用模型：先看原片能不能下载，原因写进日志
        log.run("[0b] 原片下载自检", [PY, "scripts/_evid_detect2_dlprobe.py"], show=True)
    os.environ["FOREN_CONFIG"] = str(LEG.config)
    canary = tuple(s for s in CANARY if FULL and s in ids and not N.done(LEG.round, s))
    if canary:      # run_cases 会按编号排序，所以先单独跑这两个
        log.run(f"[1a] 先跑 {len(canary)} 个（确认 find_source 被调用）", F.run_command(LEG, canary), show=True)
    todo = tuple(s for s in ids if not N.done(LEG.round, s))
    if todo:
        log.run(f"[1] 跑 {len(todo)} 个", F.run_command(LEG, todo), show=True)
    failed = tuple(s for s in ids if not N.done(LEG.round, s))
    if failed and len(failed) <= 10:
        log.run(f"[1] 失败的 {len(failed)} 个重跑一次", F.run_command(LEG, failed), show=True)
    log.run("[2] Sonnet 评分", F.score_command(LEG, ids), show=True)
    path = write_report(log)
    log.finish(f"对照表 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
