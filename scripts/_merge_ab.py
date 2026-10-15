"""DETECT 并入 SEARCH / INSPECT 之后的小 A/B（10-12）：双击 scripts\\_merge_ab.cmd。

合并方案（claude/DETECT并入SEARCH与INSPECT_合并方案_20261012.md）第 4 节第 4 步与 7.6 节的验收：

1. EVID 14 个（与 evidC27e 同一批，AB_IDS）：合并目录 hi@0.8.0 / gia@0.9.0，轮次 evidM27a；
   与 evidF27（无 DETECT）、evidC27e（DETECT v2.4，同 14 个）、evidC27G（v2.4 全量）逐样本对比，Sonnet 单评委。
   另看：原片进池并下载成功的样本，SEARCH 结果里有没有 content_match=source，之后有没有调 compare_media。
2. VideoSham 5 个 + VCSL 4 个：同一个 27B、同样学生模式，旧目录（hi@0.7.0 / gia@0.8.0，--image-search off，轮次 abO27）
   与合并目录（轮次 abM27）各跑一遍；VideoSham 用 _diff_media_ab.py 算区间边界差几帧，VCSL 列出用了哪些操作和答案。

其余设置与正式 baseline evidF27 相同（27B Q4 关思考、3 轮、compact_16k、eval 兜底、核查站点屏蔽、同一检索快照）。
可以随时关窗口，再双击续跑（跑完的 run 跳过）。日志 _合并AB日志.txt；对照表 experiments\\evid\\scores\\merge_ab_<轮次>.md。

    FOREN_MERGE_ROUND   EVID 轮次（默认 evidM27b = 10-13 修复后重跑；evidM27a 是修复前那一轮，作对照列）
    FOREN_MERGE_PARTS   evid,sft（默认两部分都跑；只跑一部分时写 evid 或 sft）
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

MERGED = ("hi@0.8.0", "gia@0.9.0")
OLD = ("hi@0.7.0", "gia@0.8.0")
ROUND = os.environ.get("FOREN_MERGE_ROUND") or "evidM27b"
AUTO_TOOLS = os.environ.get("FOREN_MERGE_AUTO") or "off"     # 10-14：on = --auto-tools on（自动比对 + 结构分析）；compare = 只自动比对
PARTS = {p.strip() for p in (os.environ.get("FOREN_MERGE_PARTS") or "evid,sft").split(",") if p.strip()}
AB_IDS = ("video_172", "video_175", "video_186", "video_189", "video_194", "video_198", "video_199", "video_205",
          "video_208", "video_138", "video_154", "video_156", "video_180", "video_209")      # = _evid_detect2.AB_IDS
ALL_ROUNDS = ("evidF27", "evidC27c", "evidC27d", "evidC27F", "evidC27e", "evidC27G", "evidM27a", "evidM27b", "evidM27c", "evidM27d", "evidM27e", "evidM27f", "evidM27g", "evidM27h", "evidM27i", "evidM27j")
# 10-13：单轮 14 个样本的波动很大（同一套 DETECT v2.4：evidC27e 23/44、evidC27G 16/44），按组取平均再比
# 10-14：这 14 个样本是 DETECT v2 的调参集，C27e 23 是调参轮里最好的一轮；DETECT 组取全部 5 轮平均（约 17）
GROUPS = (("DETECT v2（5 轮）", ("evidC27c", "evidC27d", "evidC27F", "evidC27e", "evidC27G")), ("合并目录 修复前", ("evidM27a",)),
          ("合并目录 修复 1（evidM27b）", ("evidM27b",)), ("合并目录 修复 2（收尾检查 + text_image 放宽）", ("evidM27c", "evidM27d")),
          ("合并目录 修复 3（--auto-tools on：自动比对 + 结构分析，换脸行改写）", ("evidM27e", "evidM27f")),
          ("合并目录 修复 4（恢复 DETECT 输出写法 + 报告引用修补，--auto-tools on）", ("evidM27g", "evidM27h")),
          ("合并目录 修复 5（记录 = 结论、结构杂项合并、只自动比对 --auto-tools compare）", ("evidM27i", "evidM27j")))
SFT = Path("experiments") / "sft_data"
SFT_IDS = ("videosham:0069", "videosham:1007", "videosham:1028", "videosham:2010", "videosham:4126",
           "vcsl:pair_1", "vcsl:pair_4", "vcsl:pair_5", "vcsl:pair_6")
SFT_ROUNDS = {"abO27": (OLD, ("--image-search", "off")), "abM27": (MERGED, ())}
TESTS = ["tests/test_conclusion_records_20261014.py", "tests/test_detect_contract_20261014.py", "tests/test_auto_tools_20261014.py", "tests/test_merge_regression_20261013.py", "tests/test_inspect_compare_media_20261012.py", "tests/test_search_source_match_20261012.py",
         "tests/test_merged_catalog_20261012.py", "tests/test_datagen_prompt_variants.py", "tests/test_diff_media.py"]


def common(model: str) -> list[str]:
    return ["--endpoint", L.OLLAMA, "--model", model, "--no-think", "--jobs", L.EVID8_JOBS, "--instruction-mode", "student",
            "--retry-failed", "--blocklist", "exclude", "--runtime-guards", L.EVID_GUARDS, "--state-renderer", L.EVID_RENDER]


def evid_command(ids: tuple[str, ...]) -> list[str]:
    return [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(L.EVID / "cases"), "--runs-dir", str(L.EVID / "runs"),
            "--split", "test", "--round", ROUND, *common(L.EVID27_MODEL), "--hi-spec", MERGED[0], "--gia-spec", MERGED[1],
            "--search-mode", "replay_or_record", "--snapshot-root", str(F.SNAPSHOT_ROOT),
            *(["--auto-tools", AUTO_TOOLS] if AUTO_TOOLS != "off" else []), "--ids", *(f"evid:{i}" for i in ids)]


def sft_command(round_tag: str) -> list[str]:
    specs, extra = SFT_ROUNDS[round_tag]
    return [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(SFT / "cases"), "--runs-dir", str(SFT / "runs"),
            "--round", round_tag, *common(L.EVID27_MODEL), "--hi-spec", specs[0], "--gia-spec", specs[1], *extra,
            "--search-mode", "replay_or_record", "--snapshot-root", str(SFT / "search_snapshots"), "--ids", *SFT_IDS]


def per_sample(round_tag: str) -> dict[str, dict]:
    path = ROOT / F.SCORES / f"{round_tag}.sonnet.subset.jsonl"
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["sample_id"]] = row
    return out


def tool_use(run: Path) -> dict:
    """SEARCH 结果里的 content_match、INSPECT compare_media / video_structure、以图搜图的调用次数（读 store，不调模型）。

    10-13：原来按 trajectory.jsonl 里的字符串 '"operation": "compare_media"' 计数，轨迹是紧凑 JSON（冒号后没有空格），
    所以 evidM27a 一列全是 0——实际 14 个 run 里有 compare_media 调用。改为读 store.sqlite3。"""

    import sqlite3

    out = {"source_match": 0, "compare_media": 0, "video_structure": 0, "image_search": 0}
    path = run / "store.sqlite3"
    if not path.is_file():
        return out
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        try:
            out["source_match"] = connection.execute(
                "SELECT COUNT(*) FROM candidate_matches WHERE status IN ('source','partial')").fetchone()[0]
        except sqlite3.Error:
            pass
        for (request_json,) in connection.execute("SELECT request_json FROM inspect_executions"):
            try:
                operation = (json.loads(request_json).get("request") or {}).get("operation")
            except (ValueError, AttributeError):
                continue
            if operation in ("compare_media", "video_structure"):
                out[operation] += 1
        for (request_json,) in connection.execute("SELECT request_json FROM search_executions"):
            try:
                kind = json.loads(request_json).get("input_type")
            except (ValueError, AttributeError):
                continue
            out["image_search"] += kind in ("image", "text_image")
    except sqlite3.Error:
        pass
    finally:
        connection.close()
    return out


def _baselines() -> tuple[str, ...]:
    return tuple(r for r in ALL_ROUNDS if r != ROUND and (ROOT / F.SCORES / f"{r}.sonnet.subset.jsonl").is_file())


def write_report(log) -> Path:
    BASELINES = _baselines()  # noqa: N806 - 只列已有评分的轮次
    rows = {r: per_sample(r) for r in (*BASELINES, ROUND)}
    lines = [f"# DETECT 合并后小 A/B（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"{ROUND} = evidF27 的设置 + 合并目录（hi@0.8.0 / gia@0.9.0：compare_media / video_structure / locate，SEARCH 视频结果带 "
             "content_match，以图搜图默认开）。evidC27e = DETECT v2.4（同 14 个），evidC27G = v2.4 全量。Sonnet 单评委，点级 通过/GT。", "",
             "| 样本 | 类型 | " + " | ".join((*BASELINES, ROUND)) + f" | {ROUND} 工具（同源 / compare / structure / 以图搜图） |",
             "|---|---|" + "---|" * (len(BASELINES) + 1) + "---|"]
    totals = {r: [0, 0] for r in rows}
    for s in AB_IDS:
        cells, kind = [], "?"
        for r in (*BASELINES, ROUND):
            item = rows[r].get(s)
            if item:
                kind = item.get("type", kind)
                cells.append(f"{item['passed_gt']}/{item['n_gt']}")
                totals[r][0] += item["passed_gt"]
                totals[r][1] += item["n_gt"]
            else:
                cells.append("—")
        use = tool_use(N.run_dir(ROUND, s))
        lines.append(f"| {s} | {kind} | {' | '.join(cells)} | {use['source_match']} / {use['compare_media']} / "
                     f"{use['video_structure']} / {use['image_search']} |")
    lines.append("| 合计 | | " + " | ".join(f"{totals[r][0]}/{totals[r][1]}" for r in (*BASELINES, ROUND)) + " | |")
    lines += ["", "按组平均（只算 14 个都评过分的轮次）：", ""]
    for name, members in GROUPS:
        done = [r for r in members if r in totals and totals[r][1] == 44]
        if done:
            mean = sum(totals[r][0] for r in done) / len(done)
            lines.append(f"- {name}：{mean:.1f}/44（{', '.join(f'{r} {totals[r][0]}' for r in done)}）")
    lines += ["", "验收：最新一组修复的平均不低于 DETECT v2 五轮的平均（这 14 个是 DETECT 的调参集，单轮 ±4 分波动）减 2；原片进池并下载成功的样本都应调用 compare_media；"
              "非 AI 对照（138 / 154 / 156 / 180 / 209）不变差。", ""]
    sft = ROOT / SFT / "merge_ab_sft.md"
    if sft.is_file():
        lines += ["## VideoSham / VCSL（旧目录 abO27 vs 合并目录 abM27）", "", f"见 {sft.relative_to(ROOT)}", ""]
    path = ROOT / F.SCORES / f"merge_ab_{ROUND}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.say(f"对照表 {path}")
    return path


def main() -> int:
    env_setup()
    log = N.NightLog("_合并AB日志.txt", f"DETECT 合并后小 A/B：EVID {len(AB_IDS)} 个（{ROUND}）+ VideoSham / VCSL 9 个")
    print(f"合并后小 A/B（{', '.join(sorted(PARTS))}）。可以随时关窗口，再双击续跑。日志 {log.path.name}")
    basetemp = ROOT / "outputs" / "pytest_tmp" / f"merge_{time.strftime('%Y%m%d_%H%M%S')}"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    if log.run("[0] 单元测试", [PY, "-m", "pytest", *TESTS, "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                                f"--basetemp={basetemp}"], show=True) != 0:
        log.finish("单元测试没过，停。")
        return 1
    if N.free_gb() < N.MIN_FREE_GB:
        log.finish(f"磁盘只剩 {N.free_gb():.1f}GB，少于 {N.MIN_FREE_GB}GB，停。")
        return 1
    if L.EVID27_MODEL not in (L.ollama_models() or []) or not L.evid27_probe(log):
        log.finish(f"27B（{L.EVID27_MODEL}）不可用，停。")
        return 1
    os.environ["FOREN_CONFIG"] = str(L.EVID27_CONFIG)
    if "evid" in PARTS:
        log.run("[1] EVID-Bench 标注 → 案卷", [PY, "scripts/_evid_build_cases.py"])
        todo = tuple(s for s in AB_IDS if not N.done(ROUND, s))
        if todo:
            log.run(f"[1] EVID 跑 {len(todo)} 个（{ROUND}）", evid_command(todo), show=True)
        failed = tuple(s for s in AB_IDS if not N.done(ROUND, s))
        if failed:
            log.run(f"[1] 失败的 {len(failed)} 个重跑一次", evid_command(failed), show=True)
        log.run("[2] Sonnet 评分", [PY, "scripts/_evid_score.py", "--round", ROUND, "--ids", *AB_IDS, "--judges", L.EVID_SONNET],
                show=True)
    if "sft" in PARTS:
        for round_tag in SFT_ROUNDS:
            log.run(f"[3] VideoSham / VCSL：{round_tag}", sft_command(round_tag), show=True)
        out = ROOT / SFT / "merge_ab_sft.md"
        log.run("[4] VideoSham 区间边界", [PY, "scripts/_diff_media_ab.py", "--rounds", *SFT_ROUNDS, "--out", str(out)])
        vcsl = ROOT / SFT / "merge_ab_vcsl.md"
        log.run("[4] VCSL 操作与答案", [PY, "scripts/_diff_media_ab.py", "--rounds", *SFT_ROUNDS, "--prefix", "vcsl_",
                                     "--out", str(vcsl)])
    path = write_report(log)
    log.finish(f"对照表 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
