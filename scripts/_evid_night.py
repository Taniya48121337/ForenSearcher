"""找原片实验的一整晚（10-06，claude/EVID_找原片实验计划_20261006.md）：双击 scripts\\_evid_night_source.cmd。

顺序（每一步都有闸门，坏了就跳过依赖它的步骤，不空转一整晚）：

  0  不用显卡（约 30–40 分钟）
     - 检索后端自检（yt-dlp 版本、下载）与案卷
     - A6 后端召回上界：GT 标题当查询（scripts/_evid_oracle_search.py）
     - A7 旧流水线查询重放（scripts/_evid_replay_old_queries.py）
     - 检索臂自检：results10 / guardlen / imageclip / fanout5 用真实网络各走一遍（scripts/_evid_arm_selftest.py）
     - A0（evidb12bf，已跑过）按新漏斗列重新出表（Sonnet 评委走缓存，不重复花钱）
  1  A9 = results10 + guardlen + imageclip（工具层组合，候选 baseline 配置）           自检关键项没过 → 跳过
  2  A5 = 教师指令（同一 8B bf16，A0 的工具）：流程知识值多少
  3  A4 = A9 + fanout5（程序扇出，只作消融）                                       A6 < 30% 或扇出自检没过或 A9 不可靠 → 跳过
  4  A8 = 27B Q4（A0 的工具）：底座大小值多少（本机有 qwen38-27b-32k 才跑，不在夜里下载）
  5  A8G = 27B + guardlen（只差查询限长，10-07 第二晚加）：A8 的视频搜索零结果 29/73，限长后原片能多进池吗
  6  A8T = 27B + results10 + guardlen + imageclip（A9 的整套工具层放到强底座上）

每个臂：20 个对照样本（EVID_AB_IDS，002–130），3 轮，compact_16k，评测兜底 eval，学生提示（A5 除外），
检索录制回放、各臂单独的快照目录（experiments\\evid\\search_snapshots_arms\\<轮次>）。

闸门：
  - 开跑前：Ollama 在、磁盘剩余 ≥ 15GB
  - 截止时间（可选）：set FOREN_NIGHT_UNTIL=09:00 后，预计赶不上的臂不开跑（按已跑完的臂实际耗时校正）；
    不设（10-07 起默认）就没有截止，白天随时双击都能跑完
  - 金丝雀：每个臂先跑 2 个样本，两个都崩就停这个臂（不再烧 2 个多小时）
  - 跑完：没完成的 run > 30% 记为"不可靠"，依赖它的臂跳过；有余裕时把失败的 run 重跑一次
  - 断点续跑：再双击会跳过已评完分的臂，没跑完的臂从断点继续
  - 只跑其中几个 / 换顺序：set FOREN_NIGHT_ARMS=A5,A8（在 .cmd 里或命令行先设）

最后写 experiments\\evid\\scores\\night_source_<日期>.md（每个臂一行：①进池 ②下载 ③比对 ④点级 ⑤视频级 ⑥时长与 token，
加 A6 / A7 与按实验计划判读规则自动给出的结论）。
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _launch_evid as L  # noqa: E402
from _launch import PY, ROOT, Log, env_setup  # noqa: E402

IDS = L.EVID_AB_IDS
SCORES = L.EVID / "scores"
A0_ROUND = L.EVID8_BF16_ROUND            # evidb12bf：A0（同 20 个样本、同 bf16、同规格，已跑过并有 Sonnet 分）
SNAPSHOTS = L.EVID / "search_snapshots_arms"
MANIFESTS = Path("experiments") / "sft_data" / "run_manifests"
MIN_FREE_GB = 15
CANARY = 2
UNRELIABLE = 0.30
DEFAULT_UNTIL = None               # 10-07：默认不设截止（白天也跑）；夜跑想赶早上就 set FOREN_NIGHT_UNTIL=09:00


@dataclass
class Arm:
    name: str
    round: str
    question: str
    search_arm: str | None = None
    instruction: str = "student"
    model: str = L.EVID8_BF16_MODEL
    config: Path = L.EVID8_CONFIG
    extra: tuple[str, ...] = ()
    minutes: float = 160.0              # 20 个样本的预计墙钟（bf16 A0 实测 145 分钟）
    needs: tuple[str, ...] = ()         # 依赖的前置条件（见 gates）
    status: str = "pending"
    notes: list[str] = field(default_factory=list)


ARMS = [
    Arm("A9", "evidsA9", "工具层组合（results10 + guardlen + imageclip）能把原片带进池吗", "results10,guardlen,imageclip",
        minutes=170, needs=("selftest",)),
    Arm("A5", "evidsA5", "教师指令（多条改写、先核对实体）值多少：流程知识 vs 能力", instruction="teacher", minutes=175),
    Arm("A4", "evidsA4", "程序扇出 5 条改写（替模型做事，只作消融）值多少", "results10,guardlen,imageclip,fanout5",
        minutes=185, needs=("selftest", "fanout", "a6_ok", "A9")),
    Arm("A8", "evidsA8", "27B 底座（Q4）值多少", model=L.EVID27_MODEL, config=L.EVID27_CONFIG, extra=("--no-think",),
        minutes=140, needs=("model27",)),     # 10-07 实测约 120 分钟 / 20 个（含金丝雀）
    # 10-07 第二晚：A8 里 27B 会打开候选（原片进池 3 → 下载 3），但视频查询中位 12 词、零结果 29/73。
    # A8G 只加查询限长（与 A8 只差这一项）；A8T 加 A9 的整套工具层（看工具层在强底座上值多少，与 A9 对 A0 对照）。
    Arm("A8G", "evidsA8G", "27B + 查询限长（guardlen）：零结果降下来后原片能多进池吗", "guardlen",
        model=L.EVID27_MODEL, config=L.EVID27_CONFIG, extra=("--no-think",), minutes=140, needs=("model27", "selftest")),
    Arm("A8T", "evidsA8T", "27B + A9 整套工具层（results10 + guardlen + imageclip）", "results10,guardlen,imageclip",
        model=L.EVID27_MODEL, config=L.EVID27_CONFIG, extra=("--no-think",), minutes=160, needs=("model27", "selftest")),
]


class NightLog(Log):
    """追加写（再双击续跑时保留前一次的记录）。"""

    def __init__(self, name: str, title: str) -> None:  # noqa: D401
        self.path = ROOT / name
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n==== {title} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.failed: list[str] = []


def deadline() -> dt.datetime | None:
    text = os.environ.get("FOREN_NIGHT_UNTIL") or DEFAULT_UNTIL
    if not text:
        return None
    hour, minute = (int(x) for x in text.split(":", 1))
    now = dt.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + dt.timedelta(days=1)


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def run_dir(round_tag: str, sample: str) -> Path:
    return ROOT / L.EVID / "runs" / f"evid_{sample}_{round_tag}_1"


def done(round_tag: str, sample: str) -> bool:
    path = run_dir(round_tag, sample)
    return (path / "result.json").is_file() and (path / "execution_links.json").is_file()


def done_count(round_tag: str, ids: tuple[str, ...] = IDS) -> int:
    return sum(done(round_tag, sample) for sample in ids)


def score_summary(round_tag: str) -> dict | None:
    path = ROOT / SCORES / f"{round_tag}.sonnet.subset.summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def manifest_cost(round_tag: str) -> dict:
    path = ROOT / MANIFESTS / f"{round_tag}.jsonl"
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("status") == "done":
                rows[row["run"]] = row          # 重跑过的以最后一次为准
    wanted = {f"evid_{s}_{round_tag}_1" for s in IDS}
    done_rows = [r for name, r in rows.items() if name in wanted]
    if not done_rows:
        return {}
    mean = lambda key: sum(float(r.get(key) or 0) for r in done_rows) / len(done_rows)  # noqa: E731
    return {"runs": len(done_rows), "minutes_per_run": round(mean("wall_seconds") / 60, 1),
            "input_tokens_per_run": int(mean("input_tokens")), "teacher_calls_per_run": round(mean("teacher_calls"), 1)}


def read_json(path: Path) -> dict | None:
    try:
        return json.loads((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def arm_reliable(name: str) -> bool:
    """依赖的臂这次跑完了，或之前某晚已跑完（≥ 70% 的 run 完成且已评分）。"""

    arm = next(a for a in ARMS if a.name == name)
    if arm.status == "done":
        return True
    return arm.status in {"pending", "not_selected"} and bool(score_summary(arm.round)) \
        and done_count(arm.round) >= len(IDS) * (1 - UNRELIABLE)


def ollama_ready(log: Log) -> bool:
    served = L.ollama_models()
    if served is None:
        served = L.start_ollama(log)
    if served is None:
        log.say("Ollama 连不上，自动启动也没成功。")
        return False
    return True


def run_command(arm: Arm, ids: tuple[str, ...]) -> list[str]:
    specs = L.EVID_STAGE_SPECS["p1"]
    command = [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(L.EVID / "cases"), "--runs-dir",
               str(L.EVID / "runs"), "--split", "test", "--round", arm.round, "--endpoint", L.OLLAMA, "--model", arm.model,
               *arm.extra, "--jobs", L.EVID8_JOBS,
               *(("--instruction-mode", "student") if arm.instruction == "student" else ()),     # A5：不给 = 教师指令
               "--retry-failed", "--blocklist", "exclude",
               "--hi-spec", specs[0], "--gia-spec", specs[1],
               "--search-mode", "replay_or_record", "--snapshot-root", str(SNAPSHOTS / arm.round),
               "--runtime-guards", L.EVID_GUARDS, "--state-renderer", L.EVID_RENDER,
               "--ids", *(f"evid:{i}" for i in ids)]
    if arm.search_arm:
        command += ["--search-arm", arm.search_arm]
    return command


def run_arm(log: Log, arm: Arm, end: dt.datetime | None, speed: float) -> None:
    os.environ["FOREN_CONFIG"] = str(arm.config)
    todo = [s for s in IDS if not done(arm.round, s)]
    title = f"{arm.name}（轮次 {arm.round}，{arm.model}，{arm.instruction} 提示，检索臂 {arm.search_arm or 'A0'}）"
    if todo:
        remaining = arm.minutes * speed * len(todo) / len(IDS)
        finish = dt.datetime.now() + dt.timedelta(minutes=remaining)
        if end is not None and finish > end:
            arm.status = "skipped_time"
            arm.notes.append(f"预计还要 {remaining:.0f} 分钟（{len(todo)} 个 run），会超过截止 {end:%H:%M}")
            log.say(f"跳过 {title}：{arm.notes[-1]}")
            return
        if free_gb() < MIN_FREE_GB:
            arm.status = "skipped_disk"
            arm.notes.append(f"磁盘只剩 {free_gb():.1f}GB（< {MIN_FREE_GB}GB）")
            log.say(f"跳过 {title}：{arm.notes[-1]}")
            return
        if not ollama_ready(log):
            arm.status = "skipped_ollama"
            arm.notes.append("Ollama 不在")
            return
        log.say(f"==== {title}：{arm.question}；待跑 {len(todo)} 个，预计 {remaining:.0f} 分钟"
                f"（约 {finish:%H:%M} 结束{f'；截止 {end:%H:%M}' if end else ''}）")
        canary = tuple(s for s in IDS if s in todo)[:CANARY] if len(todo) == len(IDS) else ()
        if canary:
            log.run(f"[{arm.name}] 金丝雀 {len(canary)} 个", run_command(arm, canary), show=True)
            if not any(done(arm.round, s) for s in canary):
                arm.status = "failed_canary"
                arm.notes.append(f"金丝雀 {', '.join(canary)} 一个都没跑完（看 experiments\\sft_data\\batch_logs\\evid_*_{arm.round}_1.log）")
                log.say(f"停掉 {arm.name}：{arm.notes[-1]}")
                return
        rest = tuple(s for s in IDS if not done(arm.round, s))
        if rest:
            log.run(f"[{arm.name}] 其余 {len(rest)} 个", run_command(arm, rest), show=True)
        failed = tuple(s for s in IDS if not done(arm.round, s))
        if failed and len(failed) <= 6:
            spare = (end - dt.datetime.now()).total_seconds() / 60 - 30 if end else float("inf")
            if spare > arm.minutes * speed * len(failed) / len(IDS) / 2:
                log.run(f"[{arm.name}] 失败的 {len(failed)} 个重跑一次", run_command(arm, failed), show=True)
    finished = done_count(arm.round)
    if finished < len(IDS) * (1 - UNRELIABLE):
        arm.status = "unreliable"
        arm.notes.append(f"只完成 {finished}/{len(IDS)}")
    else:
        arm.status = "done"
    log.run(f"[{arm.name}] Sonnet 评分（{finished}/{len(IDS)} 个完成；没完成的按 0）",
            [PY, "scripts/_evid_score.py", "--round", arm.round, "--ids", *IDS, "--judges", L.EVID_SONNET], show=True)


# ---------------------------------------------------------------- 汇总与判读

def arm_row(name: str, round_tag: str) -> dict:
    summary = score_summary(round_tag) or {}
    funnel = summary.get("funnel") or {}
    attempted = summary.get("attempted") or {}
    return {"arm": name, "round": round_tag, "done": done_count(round_tag),
            "in_pool": funnel.get("source_in_pool"), "downloaded": funnel.get("source_downloaded"),
            "compared": funnel.get("source_compared"), "runs_with_store": funnel.get("runs_with_store"),
            "found_by": funnel.get("source_found_by"), "video_searches": funnel.get("video_searches"),
            "video_searches_empty": funnel.get("video_searches_empty"),
            "point_acc": attempted.get("point_acc"), "video_acc": attempted.get("video_acc"),
            "scored": bool(summary), **manifest_cost(round_tag)}


def verdicts(rows: dict[str, dict], a6: dict | None, a7: dict | None) -> list[str]:
    out: list[str] = []
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    a0 = rows.get("A0") or {}
    n = len(IDS)
    if a6:
        s = a6["summary"]
        recall = (s.get("hit_at_10") or 0) / n
        if recall < 0.30:
            out.append(f"A6 前 10 命中 {s.get('hit_at_10')}/{n} < 30%：后端召回不足——就算查询写对也常搜不到，优先以图搜（A3）与换搜索源，"
                       "查询策略改了收益有限。")
        else:
            out.append(f"A6 前 10 命中 {s.get('hit_at_10')}/{n} ≥ 30%：拿对的词，现在的后端能找到原片；差距主要在查询本身。")
    if a7:
        s = a7["summary"]
        gap = (s.get("new_hit") or 0) - (s.get("old_text_hit") or 0)
        out.append(f"A7 旧查询重放：新命中 {s.get('new_hit')} vs 旧·文字 {s.get('old_text_hit')}（旧·任一候选 {s.get('old_any_hit')}）——"
                   + ("两套后端一致。" if abs(gap) <= 1 else "不一致，后端行为变了，先查零结果与桥接那几列。")
                   + f" 旧查询在现在的后端零结果 {s.get('new_empty')} 条（旧日志记 {s.get('old_failures')} 条失败）。")
    base_pool = a0.get("in_pool") or 0
    a9, a4, a5, a8 = (rows.get(k) or {} for k in ("A9", "A4", "A5", "A8"))
    if a9.get("scored"):
        gain = (a9.get("in_pool") or 0) - base_pool
        out.append(f"A9（工具层组合）进池 {a9.get('in_pool')}/{n}（A0 {base_pool}）"
                   + ("：达到计划目标 ≥ 5/20，可作 baseline 候选配置，下一步按计划扩到 50 个样本。" if (a9.get("in_pool") or 0) >= 5
                      else "：工具层单独收益小（< 2）。" if gain < 2 else "：有提升但没到 5/20。"))
    if a4.get("scored"):
        out.append(f"A4（程序扇出）进池 {a4.get('in_pool')}/{n}"
                   + ("：≥ 4/20，查询的量与写法是主因——走 A5 → SFT；扇出只在论文消融里报。" if (a4.get("in_pool") or 0) >= 4
                      else "：< 4/20，多搜几条也找不到，查询之外还有瓶颈（后端或排序）。"))
    if a5.get("scored"):
        dp = (a5.get("point_acc") or 0) - (a0.get("point_acc") or 0)
        dpool = (a5.get("in_pool") or 0) - base_pool
        if dpool >= 3 or dp >= 0.05:
            out.append(f"A5（教师指令）明显高于 A0（进池 {dpool:+d}，点级 {100 * dp:+.1f} 个百分点）：差的是流程知识，教师轨迹能教。")
        else:
            out.append(f"A5（教师指令）≈ A0（进池 {dpool:+d}，点级 {100 * dp:+.1f} 个百分点）：差的更像是能力，看 A8。")
    if a8.get("scored"):
        dp = (a8.get("point_acc") or 0) - (a0.get("point_acc") or 0)
        out.append(f"A8（27B）点级 {pct(a8.get('point_acc'))}（A0 {pct(a0.get('point_acc'))}，{100 * dp:+.1f}），进池 {a8.get('in_pool')}/{n}。")
    for name, label in (("A8G", "27B + 查询限长"), ("A8T", "27B + A9 工具层")):
        row = rows.get(name) or {}
        if row.get("scored") and a8.get("scored"):
            dpool = (row.get("in_pool") or 0) - (a8.get("in_pool") or 0)
            dp = (row.get("point_acc") or 0) - (a8.get("point_acc") or 0)
            empty = f"视频搜索零结果 {row.get('video_searches_empty')}/{row.get('video_searches')}（A8 {a8.get('video_searches_empty')}/{a8.get('video_searches')}）"
            out.append(f"{name}（{label}）对 A8：进池 {row.get('in_pool')}/{n}（{dpool:+d}），下载 {row.get('downloaded')}，"
                       f"比对 {row.get('compared')}，点级 {pct(row.get('point_acc'))}（{100 * dp:+.1f}）；{empty}"
                       + ("——进池多了，但点级掉了 5 个百分点以上：工具层带来的检索收益没转成答案，不进 baseline。"
                          if dpool >= 2 and dp <= -0.05 else
                          "——工具层在强底座上有用，可作 27B baseline 候选配置。" if dpool >= 3 else
                          "——进池没明显增加，强底座上工具层也不是瓶颈。" if dpool <= 1 else "。"))
    for name in ("A9", "A4", "A5", "A8", "A8G", "A8T"):
        row = rows.get(name) or {}
        if not row.get("scored"):
            continue
        if (row.get("in_pool") or 0) >= 3 and (row.get("downloaded") or 0) * 2 <= (row.get("in_pool") or 0):
            out.append(f"{name}：原片进池 {row.get('in_pool')} 但只下载 {row.get('downloaded')}——HI 不去打开，回到 P1.4 可供性与 A5。")
        if (row.get("compared") or 0) >= 3 and (row.get("point_acc") or 0) - (a0.get("point_acc") or 0) < 0.03:
            out.append(f"{name}：对原片比对过 {row.get('compared')} 次但点级没涨——瓶颈在答案写法 / 比对能力，停止在检索上加码。")
    out.append(f"注意：每个臂只有 {n} 个样本、温度 0.6，点级准确率相差 5 个百分点以内都在噪声里；进池数（0 → k）比准确率更可靠。")
    return out


def write_report(log: Log, arms: list[Arm], skipped: dict[str, str]) -> Path:
    rows = {"A0": arm_row("A0", A0_ROUND)}
    for arm in arms:
        rows[arm.name] = arm_row(arm.name, arm.round)
    a6 = read_json(SCORES / "A6_oracle.json")
    a7 = read_json(SCORES / "A7_replay.json")
    selftest = read_json(SCORES / "arm_selftest.json")
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    dash = lambda v: "—" if v is None else v  # noqa: E731
    n = len(IDS)
    lines = [f"# 找原片实验一夜结果（{time.strftime('%Y-%m-%d %H:%M')}）", "",
             f"20 个对照样本（002–130），Qwen3-VL-8B bf16（A8 / A8G / A8T 是 27B Q4，关思考），规格 hi@0.7.0 / gia@0.8.0，3 轮，compact_16k，"
             f"Sonnet 单评委。A0 = {A0_ROUND}（已有的 bf16 基线）。", "",
             "| 臂 | 轮次 | 完成 | ① 原片进池 | ② 下载 | ③ align/diff | ④ 点级 | ⑤ 视频级 | ⑥ 分钟/run | 输入 token/run | 视频搜索（零结果） | 状态 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    status = {arm.name: arm.status + ("：" + "；".join(arm.notes) if arm.notes else "") for arm in arms}
    status["A0"] = "已有"
    for name, row in rows.items():
        state = skipped.get(name) or status.get(name, "")
        lines.append(f"| {name} | {row['round']} | {row['done']}/{n} | {row['in_pool'] if row['in_pool'] is not None else '—'} | "
                     f"{row['downloaded'] if row['downloaded'] is not None else '—'} | {row['compared'] if row['compared'] is not None else '—'} | "
                     f"{pct(row['point_acc'])} | {pct(row['video_acc'])} | {row.get('minutes_per_run', '—')} | "
                     f"{row.get('input_tokens_per_run', '—')} | {dash(row.get('video_searches'))}（{dash(row.get('video_searches_empty'))}） | "
                     f"{state} |")
    lines += ["", "进池途径（provider/engine → 个数）：" + "；".join(f"{k} {v['found_by']}" for k, v in rows.items() if v.get("found_by")), ""]
    lines += ["## 不跑模型的部分", ""]
    if a6:
        s = a6["summary"]
        lines.append(f"- A6 后端上界（GT 标题当查询）：前 10 命中 {s['hit_at_10']}/{s['samples']}，前 5 命中 {s['hit_at_5']}；"
                     f"原片仍可访问 {s['samples_with_available_source']} 个（详见 A6_oracle.md）")
    else:
        lines.append("- A6：没有结果（看日志）")
    if a7:
        s = a7["summary"]
        lines.append(f"- A7 旧查询重放：新命中 {s['new_hit']}，旧·文字 {s['old_text_hit']}，旧·任一候选 {s['old_any_hit']}，"
                     f"一致 {s['agree_with_old_text']}/{s['with_old_log']}；新零结果 {s['new_empty']} 条（详见 A7_replay.md）")
    else:
        lines.append("- A7：没有结果（看日志）")
    if selftest:
        lines.append("- 检索臂自检：" + "；".join(f"{k} {'通过' if v['ok'] else '失败'}（{v['detail'][:80]}）"
                                             for k, v in selftest.get("checks", {}).items()))
    lines += ["", "## 自动判读（按实验计划的判读规则）", "", *[f"- {v}" for v in verdicts(rows, a6, a7)], "",
              "## 每个臂的详细表", "",
              *[f"- {name}：experiments\\evid\\scores\\{row['round']}.sonnet.subset.md（末尾有行为面板与找原片漏斗逐 run 表）"
                for name, row in rows.items()]]
    stem = f"night_source_{time.strftime('%Y%m%d')}"
    (ROOT / SCORES).mkdir(parents=True, exist_ok=True)
    md = ROOT / SCORES / f"{stem}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / SCORES / f"{stem}.json").write_text(json.dumps({"rows": rows, "arms": {a.name: {"status": a.status, "notes": a.notes}
                                                                                     for a in arms}, "skipped": skipped,
                                                            "verdicts": verdicts(rows, a6, a7)}, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    log.say(f"汇总写在 {md}")
    return md


# ---------------------------------------------------------------- 主流程

def main() -> int:
    env_setup()
    end = deadline()
    log = NightLog("_EVID找原片一夜日志.txt", "找原片实验" + (f"（截止 {end:%m-%d %H:%M}）" if end else "（不设截止）"))
    print(f"找原片实验：先做不跑模型的检查（第一次约 30–40 分钟，之后几分钟），再按顺序跑 A9 → A5 → A4 → A8 → A8G → A8T（已评过分的臂跳过；8B 每臂约 2–3 小时、27B 约 2–2.5 小时，"
          + (f"预计赶不上截止 {end:%H:%M} 的会自动跳过）。" if end else "不设截止，全部跑完为止）。"))
    print(f"可以随时关窗口，再双击会从断点继续。日志 {log.path.name}；最后的汇总在 experiments\\evid\\scores\\night_source_<日期>.md")
    if free_gb() < MIN_FREE_GB:
        log.say(f"磁盘只剩 {free_gb():.1f}GB，少于 {MIN_FREE_GB}GB；先清出空间再双击。")
        return 1
    # ---- 0：不跑模型
    log.run("[0] 检索后端自检（yt-dlp 版本、搜索、下载）", [PY, "scripts/_evid_search_probe.py"], show=True)
    log.run("[0] EVID-Bench 标注 → 案卷", [PY, "scripts/_evid_build_cases.py"])
    if not (ROOT / SCORES / "A6_oracle.json").is_file():
        log.run("[0] A6 后端召回上界（GT 标题当查询）", [PY, "scripts/_evid_oracle_search.py"], show=True)
    if not (ROOT / SCORES / "A7_replay.json").is_file():
        log.run("[0] A7 旧流水线查询重放", [PY, "scripts/_evid_replay_old_queries.py"], show=True)
    selftest_ok = log.run("[0] 检索臂自检（真实网络，中性查询）", [PY, "scripts/_evid_arm_selftest.py"], show=True) == 0
    selftest = read_json(SCORES / "arm_selftest.json") or {}
    a6 = read_json(SCORES / "A6_oracle.json")
    log.run(f"[0] A0（{A0_ROUND}）按新漏斗列重新出表（Sonnet 走缓存）",
            [PY, "scripts/_evid_score.py", "--round", A0_ROUND, "--ids", *IDS, "--judges", L.EVID_SONNET], show=True)
    served = L.ollama_models() or []
    gates = {
        "selftest": selftest_ok,
        "fanout": bool(selftest.get("fanout_ok")),
        "a6_ok": a6 is None or (a6["summary"].get("hit_at_10") or 0) / len(IDS) >= 0.30,
        "model27": L.EVID27_MODEL in served,
    }
    log.say(f"闸门：{gates}")
    if not L.bf16_preflight(log):
        log.finish("bf16 模型没就绪，模型臂都没跑。")
        write_report(log, ARMS, {})
        return 1
    # ---- 1–4：模型臂
    skipped: dict[str, str] = {}
    speed = 1.0
    chosen = [x.strip().upper() for x in (os.environ.get("FOREN_NIGHT_ARMS") or "").split(",") if x.strip()]
    order = sorted(ARMS, key=lambda a: chosen.index(a.name) if a.name in chosen else len(chosen)) if chosen else ARMS
    for arm in order:
        if chosen and arm.name not in chosen:
            arm.status = "not_selected"
            arm.notes.append("FOREN_NIGHT_ARMS 没选")
            continue
        missing = [need for need in arm.needs if need in gates and not gates[need]]
        missing += [need for need in arm.needs if need.startswith("A") and need not in gates and not arm_reliable(need)]
        if missing:
            arm.status = "skipped_gate"
            reason = {"selftest": "检索臂自检关键项没过", "fanout": "扇出自检没过", "a6_ok": "A6 < 30%（后端召回不足，扇出无意义）",
                      "model27": f"本机没有 {L.EVID27_MODEL}（不在夜里下载）"}
            arm.notes.append("；".join(reason.get(m, f"{m} 不可靠或没跑") for m in missing))
            log.say(f"跳过 {arm.name}：{arm.notes[-1]}")
            continue
        if score_summary(arm.round) and done_count(arm.round) == len(IDS):
            arm.status = "done"
            arm.notes.append("之前已跑完并评分")
            log.say(f"{arm.name}（{arm.round}）之前已跑完并评分，跳过。")
            continue
        if arm.model == L.EVID27_MODEL and not L.evid27_probe(log):
            arm.status = "skipped_model"
            arm.notes.append("27B 试答失败")
            continue
        started = time.monotonic()
        before = done_count(arm.round)
        run_arm(log, arm, end, speed if arm.model == ARMS[0].model else 1.0)
        ran = done_count(arm.round) - before
        if ran >= 5 and arm.model == ARMS[0].model:          # 只用 8B 的臂校正 8B 的预计（27B 用自己的实测值）
            actual = (time.monotonic() - started) / 60 / ran * len(IDS)
            speed = max(0.6, min(1.8, actual / arm.minutes))       # 下一个臂的预计按实际快慢校正
            log.say(f"{arm.name} 实际约 {actual:.0f} 分钟 / 20 个（预计 {arm.minutes:.0f}）；后面的预计乘 {speed:.2f}")
        write_report(log, ARMS, skipped)                            # 每跑完一个臂就更新一次汇总
    md = write_report(log, ARMS, skipped)
    L._ollama_ps(log, "[末] 显存检查")
    log.finish(f"汇总 {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
