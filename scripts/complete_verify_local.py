"""补全核验（本机跑）：六步串起来，任何一步失败都停下并保留窗口输出。默认用本地核验模型，不花钱。

    python scripts\\complete_verify_local.py                              # 本地核验，全部 12 个 bridge run，六步
    python scripts\\complete_verify_local.py --runs true_4124903_r1       # 先拿 1 个 run 冒烟
    python scripts\\complete_verify_local.py --from 4                     # 从第 4 步开始（前三步已跑过）
    python scripts\\complete_verify_local.py --endpoint http://服务器:8000/v1 --model <服务上的模型名>   # 换到服务器大模型

本地核验（--judge local，默认）：连一个 OpenAI 兼容服务（5090 上的 vLLM / LM Studio / Ollama，或服务器上的 vLLM）。
  地址默认 http://127.0.0.1:8000/v1（--endpoint 或环境变量 FOREN_JUDGE_ENDPOINT 改）；服务上只有一个模型时 --model 可省略。
  开跑前先检查服务在不在，连不上直接退出、不跑半截。
  产物和缓存都带模型标签（例如 verdicts_r0_judged.local-qwen3.5-9b.jsonl），换模型不会覆盖、不会串缓存；
  09-28 以前 OpenRouter 时期的无标签产物（verdicts_r0_judged.jsonl、export\\r0_bridge12_…_compact_16k 等）原样保留。
付费核验（--judge openrouter）：要么加 --yes-spend，要么在提示时手动输入 yes，防止误花钱。
Claude CLI 核验（--judge claude-cli，2026-09-30）：用本机 Claude Code CLI（claude -p），走登录的订阅额度；--model 默认 sonnet。
  开跑前查登录方式：OAuth（订阅）直接跑；API key 登录会按量扣费，要 --yes-spend 或手动输入 yes。
  额度用完 / 限流时本步直接退出，已判过的在缓存里，重跑命令即可接着判。--skip 4 可跳过召回校准（最费调用）。
API 核验（--judge api，2026-10-15 对话 5a，r0 冻结的核验器）：Sonnet 走 OpenAI 兼容端点（--endpoint 默认 FOREN_JUDGE_API_ENDPOINT
  或 OpenRouter；--model 默认 FOREN_JUDGE_API_MODEL 或 anthropic/claude-sonnet-5.5；Anthropic 直连用 https://api.anthropic.com/v1）。
  提示与核验代码（datagen/verify、errors.py）完全不变：判题的四步经 python -m foren_searcher.datagen.api_judge <模块> 启动，
  只是把题送到 API（与 claude-cli 相同的无工具、base64 图、结构化输出、effort medium）。按量计费：--yes-spend 或手动输入 yes。
  产物 / 缓存标签 .api-claude-sonnet-5.5，不与 claude-cli 的缓存混用。
  端点是 Anthropic 直连（https://api.anthropic.com/v1）时走 Messages API（prompt caching：系统提示与同一 run 共用的证据块作
  缓存前缀，待核句放最后），产物 / 缓存标签 .api-claude-sonnet-5-5。
API 批量核验（--judge api-batch，2026-10-15 r0 成本压缩，r0 冻结的核验方式）：Anthropic 直连 + Message Batches（半价）+ prompt
  caching（1h）。判题的步骤经 python -m foren_searcher.datagen.api_judge --batch <模块> 启动：收集未命中的题 → 批次（每组先交一条
  写缓存，结束后交其余读缓存）→ 写进同一个缓存 → 再收集，直到全部命中，最后同步正式跑一遍出产物。中途关掉再跑会接着取回已交的批次。
  key：ANTHROPIC_API_KEY（或 FOREN_JUDGE_API_KEY）。按量计费：--yes-spend 或手动输入 yes。
  每次真正的调用（同步或批次）记一行到 experiments\\sft_data\\judge_usage\\<轮次>.<标签>.jsonl（run / 步骤 / 题型 / 未缓存、缓存读、
  缓存写、输出 token / 美元），跑完按 run 汇总写 <verdicts>.usage.json，并打印每案例费用。

六步：
  1 三层核验工作区报告 → verdicts
  2 answer 逐句锚定 + plan 模型评判 → units
  3 三层核验再跑一遍：按第 2 步的 plan 评判给各工作区定要求范围（没变的问题命中缓存）
  4 核验器召回率校准（注入已知错误再核验）→ recall
  5 用 --hi-spec / --gia-spec 规格 + 判定导出训练数据（默认 hi@0.2.0 + gia@0.3.0，对应默认的 r0 run；
    其他轮次要给当时的规格：r0b hi@0.3.0 / gia@0.4.0，r0c hi@0.4.0 / gia@0.5.0，之后 hi@0.4.1 / gia@0.5.0）
  6 Qwen3.5-9B processor 核对 label mask 与 token 数

并发：FOREN_JUDGE_CONCURRENCY（默认 4）；OpenRouter 时第 4 步单独用 FOREN_JUDGE_CONCURRENCY_STEP2（默认 2）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS_DIR = Path("experiments") / "sft_data" / "runs"
DEFAULT_RUNS = (   # 全部 bridge 模式 run（GIA 与 HI 均为真实教师决策）；案卷在 experiments\sft_data\cases\
    "true_4124903_r1", "true_4124903_r2", "true_17838107_r1", "true_20205316753904_validation",
    "true_20361105_visibility_validation", "true_24171541709824_r1", "true_39132300_validation_r3",
    "videosham_0069_r1", "videosham_1007_validation", "videosham_1028_r2", "videosham_2010_r1",
    "videosham_4126_split_intervals_r7",
)
CASES = str(Path("experiments") / "sft_data" / "cases")
SFT = Path("experiments") / "sft_data"
CACHE = str(SFT / "verify_cache.jsonl")                 # local 时 judges.py 自动换成按模型分开的文件
OPENROUTER_MODEL = "qwen/qwen3.8-27b"
SPECS = ("hi@0.2.0", "gia@0.3.0")
RENDERER = "compact_16k"

JUDGE_STEPS = {1, 2, 3, 4}
RECALL_STEP = 4
ESTIMATE_STEP = 2
STEP2_CONCURRENCY = "2"       # OpenRouter 召回校准的默认并发（09-25 本机 4 路跑时大量调用失败）
BUDGET_EXIT_CODE = 3          # OpenRouter 预算用尽（已完成产物已写盘）


def run_set_id(run_dirs: list[str]) -> str:
    """这一批 run 的标识：默认 12 个 bridge run → bridge12；只有 1 个 → run 名；其它 → <个数>runs-<名字哈希8位>。
    产物名带上它，不同批次（例如批量驱动只核验新跑的 run）互不覆盖。"""

    import hashlib

    names = sorted(Path(r).name for r in run_dirs)
    if names == sorted(DEFAULT_RUNS):
        return "bridge12"
    if len(names) == 1:
        return names[0]
    return f"{len(names)}runs-" + hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:8]


def output_paths(tag: str, run_dirs: list[str], *, specs: tuple[str, str] = SPECS, round_tag: str = "r0") -> dict[str, str]:
    """每个产物带核验模型标签；非默认 run 集合再带 run 集合标识；轮次标签（默认 r0）进文件名。
    09-28 以前的无标签产物（OpenRouter 时期）不会被覆盖。"""

    set_id = run_set_id(run_dirs)
    suffix = f".{tag}" if set_id == "bridge12" else f".{tag}.{set_id}"
    out_dir = f"{round_tag}_{set_id}_{specs[0]}_{specs[1]}_{RENDERER}.{tag}"
    return {
        "verdicts": str(SFT / f"verdicts_{round_tag}_judged{suffix}.jsonl"),
        "units": str(SFT / f"verdicts_units_{round_tag}_judged{suffix}.jsonl") if round_tag != "r0"
                 else str(SFT / f"verdicts_units_judged{suffix}.jsonl"),
        "recall": str(SFT / "calibration" / f"recall_{round_tag}{suffix}.json"),
        "out": str(SFT / "export" / out_dir),
    }


API_LAUNCHER = ["-m", "foren_searcher.datagen.api_judge"]      # --judge api：判题模块经它启动（给 judges 打 "api" 补丁）
BATCH_LAUNCHER = [*API_LAUNCHER, "--batch"]                      # --judge api-batch：同一模块经批量编排器跑（judge 参数仍是 api）


def freeze_applies(specs: tuple[str, str], round_tag: str, freeze_file: Path = ROOT / "specs" / "r0_freeze.json") -> bool:
    """10-15 对话 5（清单 A8）：导出要不要 --require-freeze——规格就是冻结规格、且不是 wk<数字> 的老轮次（walker_run.specs_for
    对非 wk 轮次用冻结规格）时要；wk8–wk14 也用 hi@0.10.0 / gia@0.9.0，但它们是冻结前的对照轮，重导出时不按 r0 口径筛。"""

    import re

    try:
        frozen = json.loads(freeze_file.read_text(encoding="utf-8")).get("specs") or {}
    except (OSError, ValueError):
        return False
    return (frozen.get("hi"), frozen.get("gia")) == tuple(specs) and not re.match(r"wk\d", round_tag)


def build_steps(run_dirs: list[str], judge_args: list[str], paths: dict[str, str],
                specs: tuple[str, str] = SPECS, share_caps: list[str] | None = None,
                require_freeze: bool = False) -> list[tuple[str, list[str]]]:
    v, u, r, o = paths["verdicts"], paths["units"], paths["recall"], paths["out"]
    api = "--judge" in judge_args and judge_args[judge_args.index("--judge") + 1:][:1] == ["api"]
    batch = api and "--batch-mode" in judge_args
    judge_args = [a for a in judge_args if a != "--batch-mode"]

    def judged(module: str) -> list[str]:
        return [*(BATCH_LAUNCHER if batch else API_LAUNCHER), module] if api else ["-m", module]

    reverify = [*judged("foren_searcher.datagen.verify.reverify"), "--runs", *run_dirs, "--cases", CASES, "--cache", CACHE, *judge_args,
                "--out", v, "--units", u]
    return [
        ("三层核验工作区报告 → " + v, reverify),
        ("answer 逐句锚定 + plan 模型评判 → " + u,
         [*judged("foren_searcher.datagen.verify.complete_units"), "--runs", *run_dirs, "--cases", CASES, "--verdicts", v,
          "--cache", CACHE, *judge_args, "--out", u]),
        ("三层核验按 plan 评判重定范围（没变的问题命中缓存）→ " + v, reverify),
        ("核验器召回率校准（注入已知错误再核验）→ " + r,
         [*judged("foren_searcher.datagen.verify.calibrate_recall"), "--runs", *run_dirs, "--cases", CASES, "--verdicts", v,
          "--units", u, "--cache", CACHE, *judge_args, "--per-kind", "2", "--out", r]),
        (f"用 {specs[0]} + {specs[1]} 规格 + 判定导出 → " + o,
         ["-m", "foren_searcher.datagen.export", "--runs", *run_dirs, "--cases", CASES, "--verdicts", v, "--units", u,
          "--hi-spec", specs[0], "--gia-spec", specs[1], "--renderer", RENDERER, "--out", o,
          *(["--max-dataset-share", *share_caps] if share_caps else []),
          *(["--require-freeze", "--plan-copy", "hold"] if require_freeze else [])]),
        ("Qwen3.5-9B processor 核对",
         ["-m", "foren_searcher.datagen.check_inputs", "--export", o, "--model", str(Path(".models") / "Qwen3.5-9B"),
          "--files", "hi_accepted.jsonl", "gia_accepted.jsonl", "hi_held.jsonl", "hi_rejected.jsonl"]),
    ]


def resolve_runs(names: list[str]) -> list[str]:
    out = []
    for name in names:
        path = Path(name) if (Path(name) / "store.sqlite3").is_file() else RUNS_DIR / name
        if not (path / "store.sqlite3").is_file():
            raise SystemExit(f"找不到 run：{name}（在 {RUNS_DIR} 下也没有）")
        out.append(str(path))
    return out


def step_env(index: int, env: dict[str, str], judge: str) -> dict[str, str]:
    if judge in ("api", "api-batch"):
        return {**env, "FOREN_JUDGE_STEP": str(index)}          # 用量记账按步骤分
    if judge != "openrouter" or index != RECALL_STEP:
        return env
    return {**env, "FOREN_JUDGE_CONCURRENCY": os.environ.get("FOREN_JUDGE_CONCURRENCY_STEP2", STEP2_CONCURRENCY)}


def usage_summary(log: Path, start: int, out: Path | None = None) -> dict[str, object] | None:
    """这次跑新增的 API 调用（usage 日志从 start 字节起的行）按 run 汇总：调用数、批次占比、未缓存 / 缓存读 / 缓存写 / 输出 token、
    美元；另给"同样的 token 按 5a 的同步无缓存口径（$2 / $10 每百万）"折算的费用，便于和 5a 冒烟并排。"""

    if not log.is_file():
        return None
    with log.open("rb") as handle:
        handle.seek(start)
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    runs: dict[str, dict[str, float]] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        run = str(row.get("run") or "(none)")
        if run.startswith("("):            # 预检 / 试答
            continue
        item = runs.setdefault(run, {"calls": 0, "batch_calls": 0, "uncached": 0, "cache_read": 0, "cache_write_5m": 0,
                                     "cache_write_1h": 0, "output": 0, "cost_usd": 0.0, "cost_5a_equiv_usd": 0.0})
        item["calls"] += 1
        item["batch_calls"] += row.get("mode") == "batch"
        for key in ("uncached", "cache_read", "cache_write_5m", "cache_write_1h", "output"):
            item[key] += int(row.get(key) or 0)
        item["cost_usd"] += float(row.get("cost_usd") or 0.0)
        tokens_in = sum(int(row.get(k) or 0) for k in ("uncached", "cache_read", "cache_write_5m", "cache_write_1h"))
        item["cost_5a_equiv_usd"] += (tokens_in * 2.0 + int(row.get("output") or 0) * 10.0) / 1e6
    if not runs:
        return None
    total = {k: sum(r[k] for r in runs.values()) for k in next(iter(runs.values()))}
    for item in [*runs.values(), total]:
        item["cost_usd"] = round(item["cost_usd"], 4)
        item["cost_5a_equiv_usd"] = round(item["cost_5a_equiv_usd"], 4)
    summary = {"runs": runs, "total": total, "n_runs": len(runs), "per_case_usd": round(total["cost_usd"] / len(runs), 4),
               "per_case_calls": round(total["calls"] / len(runs), 1), "log": str(log)}
    if out is not None:
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def recall_note(path: str) -> str | None:
    """召回产物里 inconclusive > 0 时提示重跑（已判过的对命中缓存）。"""

    try:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    n = int(summary.get("inconclusive") or 0)
    line = (f"  召回 {summary.get('recall_overall')}（{summary.get('n_injections')} 处注入）；"
            f"inconclusive {n}；conclusive 召回 {summary.get('recall_conclusive')}")
    if n > 0:
        line += f"\n  有 {n} 处注入因核验调用失败未判定：可重跑本步（--from {RECALL_STEP}），已判过的对会命中缓存。"
    return line


def estimate(argv: list[str], env: dict[str, str]) -> str:
    """第 2 步大约要多少次核验调用（--estimate 不连模型、不写产物）。"""

    try:
        done = subprocess.run([sys.executable, *argv, "--estimate"], env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
    except (OSError, subprocess.SubprocessError):
        return "估不出（估算脚本没跑成）"
    lines = done.stdout.splitlines()
    total = next((l.split()[1] for l in reversed(lines) if l.startswith("ESTIMATE ")), None)
    if done.returncode != 0 or total is None:
        return "估不出（估算脚本没跑成）"
    per_run = [l for l in lines if not l.startswith("ESTIMATE ")]
    return f"约 {total} 次（含会命中缓存的）\n" + "\n".join(per_run)


def confirm_spend(args: argparse.Namespace) -> bool:
    if args.yes_spend:
        return True
    print("!!! 这次核验会调用按量计费的 API（OpenRouter 每次全量重跑约几美元；API key 登录的 Claude CLI 同样按量扣费）。本地核验用 --judge local（默认）。", flush=True)
    try:
        return input("确定要花钱就输入 yes：").strip().lower() == "yes"
    except EOFError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="补全核验：六步串跑（默认本地核验模型，不花钱）")
    parser.add_argument("--from", dest="start", type=int, default=1, help="从第几步开始（1–6）")
    parser.add_argument("--to", dest="stop", type=int, default=6, help="跑到第几步为止（1–6）")
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS), help="run 名（experiments\\sft_data\\runs 下）或路径；默认 12 个 bridge run")
    parser.add_argument("--runs-glob", default=None, help="按轮次标签选 run：runs 下所有 *_<标签>_*（跳过 .failed- 目录）；给了就忽略 --runs")
    parser.add_argument("--judge", default="local", choices=["local", "openrouter", "claude-cli", "api", "api-batch"])
    parser.add_argument("--skip", type=int, nargs="*", default=[], help="跳过这些步（例如 --skip 4 不跑召回校准）")
    parser.add_argument("--endpoint", default=None, help="本地核验服务地址（默认 FOREN_JUDGE_ENDPOINT 或 http://127.0.0.1:8000/v1）；"
                        "--judge api 时是 API 端点（默认 FOREN_JUDGE_API_ENDPOINT 或 OpenRouter）")
    parser.add_argument("--model", default=None, help="核验模型名；local 时服务上只有一个模型可省略；openrouter 默认 " + OPENROUTER_MODEL)
    parser.add_argument("--think", action="store_true", help="local：打开模型思考（更准、慢几倍）")
    parser.add_argument("--yes-spend", action="store_true", help="--judge openrouter / api 时跳过确认")
    parser.add_argument("--hi-spec", default=SPECS[0], help="导出用的 HI 规格（要与 run 实际用的提示一致：r0 hi@0.2.0，r0b hi@0.3.0，09-30 以后 hi@0.4.0）")
    parser.add_argument("--gia-spec", default=SPECS[1], help="导出用的 GIA 规格（r0 gia@0.3.0，r0b gia@0.4.0，09-30 以后 gia@0.5.0）")
    parser.add_argument("--max-dataset-share", nargs="*", default=["videosham=0.34"],
                        help="导出时按角色压数据集份额（只缩权重），默认 videosham=0.34；给空（--max-dataset-share）则不压")
    parser.add_argument("--round", default="r0", help="轮次标签，进产物文件名（批量驱动会传自己的 --round）")
    parser.add_argument("--require-freeze", choices=["auto", "on", "off"], default="auto",
                        help="导出时只收 run_config.freeze.ok 的 run（r0e 豁免）：auto = 冻结规格且不是 wk<数字> 轮次时开（10-15 对话 5）")
    args = parser.parse_args()
    os.chdir(ROOT)
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    if args.judge not in ("claude-cli", "api"):
        env.setdefault("FOREN_JUDGE_CONCURRENCY", "4")
    if args.runs_glob:
        found = sorted(p.name for p in RUNS_DIR.glob(f"*_{args.runs_glob}_*") if p.is_dir() and ".failed-" not in p.name
                       and (p / "store.sqlite3").is_file() and (p / "execution_links.json").is_file())
        skipped = sorted(p.name for p in RUNS_DIR.glob(f"*_{args.runs_glob}_*") if p.is_dir() and ".failed-" not in p.name
                         and p.name not in found)
        if skipped:
            print(f"跳过没跑完的 run（没有 execution_links.json）：{', '.join(skipped)}", flush=True)
        if not found:
            print(f"runs 下没有 *_{args.runs_glob}_* 的 run", flush=True)
            return 2
        args.runs = found
    run_dirs = resolve_runs(args.runs)

    if args.judge == "local":
        from foren_searcher.datagen.verify.judges import (LocalJudgeUnavailable, judge_tag, local_endpoint,
                                                          resolve_local_model)
        try:
            model = resolve_local_model(args.endpoint, args.model)
        except LocalJudgeUnavailable as exc:
            print(f"!!! {exc}", flush=True)
            return 2
        name = f"local:{model}" + ("+think" if args.think else "")
        tag = judge_tag(name)
        judge_args = ["--judge", "local", "--model", model, "--endpoint", local_endpoint(args.endpoint)]
        if args.think:
            judge_args.append("--think")
        print(f"核验模型：{name} @ {local_endpoint(args.endpoint)}（本地，不花钱）；产物标签 .{tag}", flush=True)
    elif args.judge == "claude-cli":
        from foren_searcher.datagen.verify.judges import (ClaudeCliError, ClaudeCliJudge, ClaudeCliLimit, LocalJudgeUnavailable,
                                                          claude_auth_method, find_claude_cli, judge_tag)
        try:
            cli = find_claude_cli()
        except LocalJudgeUnavailable as exc:
            print(f"!!! {exc}", flush=True)
            return 2
        method = claude_auth_method(cli) or "未知"
        print(f"Claude CLI：{' '.join(cli)}；登录方式 {method}", flush=True)
        # 订阅登录在不同版本里报作 oauth_token（Linux / macOS）或 claude.ai（Windows 2026-09 实测）；其余（api key 等）按量扣费
        if not any(word in method.lower() for word in ("oauth", "claude.ai")):
            print("!!! 不是订阅登录：claude -p 可能按 API 用量扣费。", flush=True)
            if not confirm_spend(args):
                print("已取消，没有调用。", flush=True)
                return 1
        probe = ClaudeCliJudge(model=args.model or "sonnet", cli=cli)
        try:
            name = probe.resolve()
        except (ClaudeCliError, LocalJudgeUnavailable, ClaudeCliLimit) as exc:
            print(f"!!! Claude CLI 试答失败：{exc}", flush=True)
            return 2
        finally:
            probe.close()
        model = name.split(":", 1)[1]
        tag = judge_tag(name)
        judge_args = ["--judge", "claude-cli", "--model", model]
        env.setdefault("FOREN_JUDGE_CONCURRENCY", "3")
        print(f"核验模型：{name}（本机 Claude CLI）；产物标签 .{tag}", flush=True)
    elif args.judge in ("api", "api-batch"):
        from foren_searcher.datagen.api_judge import (ANTHROPIC_ENDPOINT, AnthropicJudge, ApiJudge, UsageLog, is_anthropic)
        from foren_searcher.datagen.verify.judges import LocalJudgeUnavailable, judge_tag
        batch = args.judge == "api-batch"
        if batch and not (args.endpoint and is_anthropic(args.endpoint)):
            # 批量只有直连 API 有：没给端点或给的不是 Anthropic 时用直连默认端点
            args.endpoint = os.environ.get("FOREN_JUDGE_API_ENDPOINT") if is_anthropic() else ANTHROPIC_ENDPOINT
        if not confirm_spend(args):
            print("已取消，没有调用任何付费接口。", flush=True)
            return 1
        direct = batch or is_anthropic(args.endpoint)
        usage_log = SFT / "judge_usage" / f"{args.round}.{'api-batch' if batch else 'api'}.jsonl"
        usage_log.parent.mkdir(parents=True, exist_ok=True)
        env["FOREN_JUDGE_USAGE_LOG"] = str(usage_log)
        env["FOREN_JUDGE_BATCH_TAG"] = args.round
        probe = (AnthropicJudge(model=args.model, endpoint=args.endpoint, usage_log=UsageLog(usage_log)) if direct
                 else ApiJudge(model=args.model, endpoint=args.endpoint))
        try:
            name = probe.resolve()
        except LocalJudgeUnavailable as exc:
            print(f"!!! API 核验预检失败：{exc}", flush=True)
            return 2
        finally:
            probe.close()
        model, endpoint = probe.model, probe.endpoint
        tag = judge_tag(name)
        judge_args = ["--judge", "api", "--model", model, "--endpoint", endpoint, *(["--batch-mode"] if batch else [])]
        env.setdefault("FOREN_JUDGE_CONCURRENCY", "6")
        if direct:
            print(f"核验模型：{name} @ {endpoint}（Anthropic 直连，{'Message Batches + ' if batch else ''}prompt caching，effort "
                  f"{probe.effort}）；产物标签 .{tag}；用量记到 {usage_log}", flush=True)
        else:
            print(f"核验模型：{name} @ {endpoint}（API，effort {probe.effort}，供应商 {','.join(probe.providers) or '默认'}；"
                  f"试答走的是 {dict(probe.providers_used) or '?'}）；产物标签 .{tag}", flush=True)
    else:
        if not Path(".local/openrouter.json").is_file():
            print("找不到 .local\\openrouter.json（OpenRouter 配置）。", flush=True)
            return 2
        if not confirm_spend(args):
            print("已取消，没有调用任何付费接口。", flush=True)
            return 1
        from foren_searcher.datagen.verify.judges import judge_tag
        model = args.model or OPENROUTER_MODEL
        tag = judge_tag(f"openrouter:{model}")
        judge_args = ["--judge", "openrouter", "--model", model, "--reasoning", "low"]

    specs = (args.hi_spec, args.gia_spec)
    paths = output_paths(tag, run_dirs, specs=specs, round_tag=args.round)
    require_freeze = args.require_freeze == "on" or (args.require_freeze == "auto" and freeze_applies(specs, args.round))
    if require_freeze:
        print("导出按 r0 冻结口径：--require-freeze --plan-copy hold（freeze.ok 不为 true 的 run 不进）", flush=True)
    steps = build_steps(run_dirs, judge_args, paths, specs, share_caps=args.max_dataset_share, require_freeze=require_freeze)
    usage_path = Path(env["FOREN_JUDGE_USAGE_LOG"]) if env.get("FOREN_JUDGE_USAGE_LOG") else None
    usage_start = usage_path.stat().st_size if usage_path is not None and usage_path.is_file() else 0
    print(f"run：{len(run_dirs)} 个（集合 {run_set_id(run_dirs)}）；步骤 {args.start}–{args.stop}", flush=True)
    for index, (title, argv) in enumerate(steps, start=1):
        if index < args.start or index > args.stop or index in args.skip:
            continue
        env_i = step_env(index, env, args.judge)
        print(f"\n[{index}/{len(steps)}] {title}", flush=True)
        print("  > python " + " ".join(argv), flush=True)
        if index in JUDGE_STEPS:
            line = f"  核验并发 {env_i['FOREN_JUDGE_CONCURRENCY']}"
            if index == ESTIMATE_STEP:
                line += "；预计核验调用：" + estimate(argv, env_i)
            print(line, flush=True)
        code = subprocess.call([sys.executable, *argv], env=env_i)
        if code == BUDGET_EXIT_CODE and args.judge == "openrouter":
            print(f"\n!!! 第 {index} 步中止：OpenRouter 预算用尽。建议改用本地核验（去掉 --judge openrouter）。", flush=True)
            return code
        if code != 0:
            print(f"\n第 {index} 步失败（退出码 {code}）。修好后可用 --from {index} 从这一步继续。", flush=True)
            if args.judge == "api":
                print("  API 限流 / 余额不足时：恢复后重跑同一命令，已判过的命中缓存（verify_cache.api-*.jsonl）。", flush=True)
            if args.judge == "claude-cli":
                print("  若上面是 ClaudeCliLimit（订阅额度 / 限流）：等额度恢复后重跑同一命令，已判过的命中缓存。", flush=True)
            return code
        if index == RECALL_STEP:
            note = recall_note(paths["recall"])
            if note:
                print(note, flush=True)
    if usage_path is not None:
        summary = usage_summary(usage_path, usage_start, Path(paths["verdicts"] + ".usage.json"))
        if summary:
            total = summary["total"]
            print(f"\nAPI 核验费用（这次新调用的）：{summary['n_runs']} 个 run，{int(total['calls'])} 次调用（批次 {int(total['batch_calls'])}），"
                  f"${total['cost_usd']:.3f}，每案例 ${summary['per_case_usd']:.3f}（同样的 token 按 5a 同步无缓存口径 "
                  f"${total['cost_5a_equiv_usd']:.3f}）；输入 token 未缓存 {int(total['uncached'])} / 缓存读 {int(total['cache_read'])} / "
                  f"缓存写 {int(total['cache_write_5m'] + total['cache_write_1h'])}，输出 {int(total['output'])}；明细 "
                  f"{paths['verdicts']}.usage.json", flush=True)
        else:
            print("\nAPI 核验费用：这次没有新调用（全部命中缓存）", flush=True)
    print("\n完成。产物：", flush=True)
    for key in ("verdicts", "units", "recall"):
        print("  " + paths[key], flush=True)
    print("  " + str(Path(paths["out"]) / "manifest.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
