"""EVID-Bench baseline 的双击启动（10-05 从 scripts\\_launch.py 拆出来单独放）。

为什么单独一个文件：_launch.py 同时被几个会话改，EVID 部分先后四次被旧底稿覆盖（丢过 --ids、自动起 Ollama、
evid_local 配置、显式 hi@0.4.2、Sonnet 评分）。EVID 的 .cmd 只调本文件；日志类、环境设置沿用 _launch.py 的。

    python scripts\\_launch_evid.py evid_smoke | evid_full | evid_score | evid_score_sonnet
    python scripts\\_launch_evid.py evid_ab_p0                            # 修正计划 P0 的 A/B：8B × 20 个样本，evidb6 vs evidb4
    python scripts\\_launch_evid.py evid_ab_bf16                          # 10-06 精度对照：8B bf16 × 同 20 个样本，evidb12bf vs evidb12（Q4）
    python scripts\\_launch_evid.py evid27_full | evid27_score_sonnet      # 旧入口，等同 evid_full / evid_score_sonnet
    10-05 晚起被测默认 Qwen3.8-27B（FOREN_EVID_MODEL_PROFILE=8b 回到 8B）；冒烟跑完直接用 Sonnet 评分。

修正优化计划（claude/EVID修正优化计划_20261005.md）的两个阶段，FOREN_EVID_STAGE 选（默认 p1）：
    p0 = 规格 hi@0.5.0 / gia@0.6.0（不改模型输入）+ 评测档强制 compact_16k + schema 约束解码；--max-rounds 2 与 evidb4 可比
         轮次 8B evidb6、27B evidq27a
    p1 = 规格 hi@0.7.0 / gia@0.8.0（输入补全：入库关键帧、原件帧优先、origin、候选可供性、跨轮交接、约束可见；
         10-05 晚起 language=en：模型读到的框架文字全英文，外部内容原样）+ 同上；轮数用配置里的 3（P1.7）；轮次 8B evidb12（evidb7 = 修下载 / 只可交报告之前的冒烟；evidb8 = 已修但 yt-dlp 仍是 2026.07.04、下载照样 403；evidb9 = yt-dlp 已更新但本机没有 ffmpeg、分离流合并失败；evidb10 = PyAV 合并遇 AV1 视频流报 UnknownCodecError；evidb11 = 下载通了，但视频 OCR 每帧 LoadImageError）、27B evidq27b
    p1zh = 同 p1 但用中文的 hi@0.6.0 / gia@0.7.0（只为日后对照英文开关）；轮次 8B evidb7z、27B evidq27bz
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _launch import PY, Log, env_setup  # noqa: E402  共用日志与环境设置

# ---------------------------------------------------------------------------- EVID-Bench baseline（10-04）

EVID = Path("experiments") / "evid"
# 修正优化计划阶段（见文件头）。p0 的规格与 run_cases 默认一致；p1 = hi@0.7.0 / gia@0.8.0（P1 输入补全 + 英文）
EVID_STAGE = (os.environ.get("FOREN_EVID_STAGE") or "p1").lower()
EVID_STAGE_SPECS = {"p0": ("hi@0.5.0", "gia@0.6.0"), "p1": ("hi@0.7.0", "gia@0.8.0"), "p1zh": ("hi@0.6.0", "gia@0.7.0")}
if EVID_STAGE not in EVID_STAGE_SPECS:
    raise SystemExit(f"FOREN_EVID_STAGE 只能是 p0 / p1 / p1zh，得到 {EVID_STAGE!r}")
EVID_SPECS = EVID_STAGE_SPECS[EVID_STAGE]
# P0.1：评测档强制渲染档位（优先于规格 harness_params；端点按档位渲染输入）。p1 的规格本身已是 compact_16k，照样显式传
EVID_RENDER = os.environ.get("FOREN_EVID_RENDER") or "compact_16k"
# P1.7：配置 max_rounds 2 → 3；p0 的 A/B 固定 2 轮与 evidb4 可比
EVID_STAGE_ARGS: tuple[str, ...] = ("--max-rounds", "2") if EVID_STAGE == "p0" else ()
# P0 A/B 用的 20 个样本：evidb4 已跑过的 002–130 里均匀取（027 已降级出 test，不在其中）；不碰 131–222
EVID_AB_IDS = tuple(f"video_{2 + round(i * 128 / 19):03d}" for i in range(20))
EVID_SONNET = "claude-cli:sonnet"                               # 官方三评委之一的同家族；本机评委太松，正式数字用它
OLLAMA = os.environ.get("FOREN_OLLAMA_ENDPOINT") or "http://127.0.0.1:11434/v1"
EVID_JUDGES = ("gemma3-12b-16k:latest", "qwen3-8b-16k:latest", "qwen3vl-8b-32k:latest")   # 本机三评委，只在 evid_score 里用
EVID_SMOKE = ("video_001", "video_027", "video_151")    # 三大类各一个；10-05 起已降级出 test，冒烟只用来看改动效果
# 评测兜底档位（run_cases --runtime-guards）：eval = 方案第 1 步（默认，不改模型输入）；eval_plus = 再加第 2 步（改输入）。
EVID_GUARDS = os.environ.get("FOREN_EVID_GUARDS") or "eval"

# 被测模型（10-05 晚起默认 27B）。FOREN_EVID_MODEL_PROFILE=8b 回到 Qwen3-VL-8B。
# 8B：evidb0–evidb5（evidb4 全量跑到 130 个停；evidb5 = 改写任务文字 + 评测兜底的冒烟）。
# 27B：Ollama 官方 qwen3.8:27b（原生看图 / 视频，Q4 约 18GB），另建 32K 上下文的 qwen38-27b-32k，与 8B 的窗口一致；
#   关思考（--no-think → reasoning_effort=none）：27B 默认思考，每次调用多出上千 token；8B 的 instruct 版本来就不思考；
#   configs/evid_local_27b.toml 只把单步超时放宽（GIA 300→900 秒、工作区 1800→3600 秒）；
#   同时跑 2 个样本：一个在下载 / 抽帧 / 搜索时另一个用显卡，结果不受影响。
EVID_MODEL_PROFILE = (os.environ.get("FOREN_EVID_MODEL_PROFILE") or "27b").lower()
EVID27_BASE = "qwen3.8:27b"
EVID27_MODEL = "qwen38-27b-32k:latest"
EVID27_MODELFILE = Path("scripts") / "ollama" / "qwen38_27b_32k.Modelfile"
EVID27_CONFIG = Path("configs") / "evid_local_27b.toml"
EVID27_JOBS = os.environ.get("FOREN_EVID27_JOBS") or "2"
EVID27_ROUND = {"p0": "evidq27a", "p1": "evidq27b", "p1zh": "evidq27bz"}[EVID_STAGE]   # evidq27a：P0（10-05 第一次 5 个 run 在 P0 之前、已失败，--retry-failed 重跑）
EVID8_MODEL = os.environ.get("FOREN_EVID_MODEL") or "qwen3vl-8b-32k:latest"
EVID8_CONFIG = Path("configs") / "evid_local.toml"              # 与 local.toml 只差 max_workspaces_per_round 2 → 3、max_rounds 3
# 10-05 晚：8B 也同时跑 2 个样本。evidb11 每个 run 约 9.6 分钟，模型调用只占 40–65%，其余是搜索联网、视频 OCR、取帧；
# 两个样本交错时显卡与 CPU / 网络各忙各的。FOREN_EVID8_JOBS=1 回到串行。
EVID8_JOBS = os.environ.get("FOREN_EVID8_JOBS") or "2"
if EVID_MODEL_PROFILE == "8b":
    EVID_MODEL = EVID8_MODEL
    EVID_MODEL_NAME = "Qwen3-VL-8B"
    EVID_CONFIG = EVID8_CONFIG
    EVID_EXTRA_ARGS: tuple[str, ...] = ("--jobs", EVID8_JOBS)
    _DEFAULT_ROUND = {"p0": "evidb6", "p1": "evidb12", "p1zh": "evidb7z"}[EVID_STAGE]
else:
    EVID_MODEL = EVID27_MODEL
    EVID_MODEL_NAME = "Qwen3.8-27B"
    EVID_CONFIG = EVID27_CONFIG
    EVID_EXTRA_ARGS = ("--no-think", "--jobs", EVID27_JOBS)
    _DEFAULT_ROUND = EVID27_ROUND
# 轮次：evidb5p / evidq27p 是 eval_plus 的 A/B 冒烟（scripts\_evid_smoke_plus.cmd 设 FOREN_EVID_ROUND）
EVID_ROUND = os.environ.get("FOREN_EVID_ROUND") or _DEFAULT_ROUND


def ollama_models() -> list[str] | None:
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(OLLAMA.rstrip("/") + "/models", timeout=5) as response:
            return [m["id"] for m in _json.loads(response.read().decode("utf-8")).get("data", [])]
    except Exception:  # noqa: BLE001
        return None


def ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe is None:
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        exe = str(local) if local.is_file() else None
    return exe


def start_ollama(log: Log) -> list[str] | None:
    """Ollama 没开：找到 ollama 命令就在后台起 ollama serve（独立进程，关掉本窗口也不停），最多等 60 秒。"""

    exe = ollama_exe()
    if exe is None:
        log.say("找不到 ollama 命令（PATH 和 %LOCALAPPDATA%\\Programs\\Ollama 里都没有）。")
        return None
    log.say(f"Ollama 没在运行，正在后台启动：{exe} serve（最多等 60 秒）")
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0       # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True)
    except OSError as exc:
        log.say(f"  启动失败：{exc}")
        return None
    for _ in range(30):
        time.sleep(2)
        served = ollama_models()
        if served is not None:
            log.say("  Ollama 已就绪。")
            return served
    return None


def evid_preflight(log: Log) -> bool:
    """Ollama 在不在（不在就自动起）；被测模型与三个评委在不在；qwen3-8b-16k 没有就按 Modelfile 建（只是给 qwen3:8b 加 16K 上下文）。"""

    served = ollama_models()
    if served is None:
        served = start_ollama(log)
    if served is None:
        log.say(f"连不上 Ollama（{OLLAMA}），自动启动也没成功。请手动打开 Ollama（开始菜单里点 Ollama，或命令行 ollama serve），"
                "等托盘出现图标后再双击本文件。")
        log.failed.append("Ollama 未就绪")
        return False
    log.write(f"Ollama 上的模型：{served}")
    if "qwen3-8b-16k:latest" not in served and "qwen3:8b" in served:
        log.run("[0] 建评委 qwen3-8b-16k（qwen3:8b + 16K 上下文）",
                ["ollama", "create", "qwen3-8b-16k", "-f", str(Path("scripts") / "ollama" / "judge_qwen3.Modelfile")])
        served = ollama_models() or served
    missing = [m for m in (EVID_MODEL, *EVID_JUDGES) if m not in served]
    if missing:
        log.failed.append("缺模型")
        log.say(f"Ollama 上缺这些模型：{missing}。之前的 scripts\\_本地模型准备.cmd 会建 qwen3vl-8b-32k 与 gemma3-12b-16k；"
                "qwen3:8b 用 ollama pull qwen3:8b。")
        return False
    return True


def model_preflight(log: Log) -> bool:
    """被测模型就绪：27B 走 evid27_preflight（版本检查 / 下载 / 建 32K 版 / 试答），8B 走 evid_preflight；再试一次 schema 解码。"""

    ready = evid_preflight(log) if EVID_MODEL_PROFILE == "8b" else evid27_preflight(log)
    if ready:
        schema_probe(log, EVID_MODEL)
    return ready


def schema_probe(log: Log, model: str) -> bool:
    """P0.2：本机 Ollama 能不能按 JSON schema（anyOf + const + enum）约束解码。不能也照跑——教师会自动退回 json_object。"""

    import json as _json
    import urllib.error
    import urllib.request

    schema = {"anyOf": [
        {"type": "object", "additionalProperties": False, "required": ["decision", "answer"],
         "properties": {"decision": {"const": "FINISH"}, "answer": {"type": "string"}}},
        {"type": "object", "additionalProperties": False, "required": ["decision", "tasks"],
         "properties": {"decision": {"type": "string", "enum": ["INVESTIGATE"]},
                        "tasks": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False,
                                  "required": ["task", "materials"], "properties": {"task": {"type": "string"},
                                  "materials": {"type": "array", "items": {"type": "string"}}}}}}}]}
    body = {"model": model, "temperature": 0.6, "max_tokens": 128, "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": schema}},
            "messages": [{"role": "user", "content": "Decide to investigate Material E1 with one task. Reply in JSON."}]}
    try:
        request = urllib.request.Request(OLLAMA.rstrip("/") + "/chat/completions", data=_json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            data = _json.loads(response.read().decode("utf-8"))
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        value = _json.loads(text)
        ok = isinstance(value, dict) and set(value) in ({"decision", "answer"}, {"decision", "tasks"})
    except urllib.error.HTTPError as exc:
        text, ok = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}", False
    except Exception as exc:  # noqa: BLE001
        text, ok = f"{type(exc).__name__}: {exc}", False
    log.say(f"schema 约束解码自检（P0.2）：{'可用' if ok else '不可用，本批自动退回 json_object'}；回答 {text.strip()[:120]!r}")
    return ok


def _evid_run(log: Log, ids: tuple[str, ...] | None, title: str, *, model: str | None = None,
              config: Path | None = None, extra_args: tuple[str, ...] | None = None, round_tag: str | None = None,
              specs: tuple[str, str] | None = None, stage_args: tuple[str, ...] | None = None,
              preflight: bool = True) -> int:
    model = model or EVID_MODEL
    round_tag = round_tag or EVID_ROUND
    specs = specs or EVID_SPECS
    if preflight and not model_preflight(log):
        return 1
    os.environ["FOREN_CONFIG"] = str(config or EVID_CONFIG)        # 子进程继承；run_config.config_source 会记下
    log.run("[0/4] 检索后端自检（文字 → 视频：yt-dlp 与 DDGS 各一次）", [PY, "scripts/_evid_search_probe.py"], show=True)
    log.run("[1/4] EVID-Bench 标注 → 案卷（experiments\\evid\\cases，split=test）", [PY, "scripts/_evid_build_cases.py"])
    run = [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", str(EVID / "cases"), "--runs-dir", str(EVID / "runs"),
           "--split", "test", "--round", round_tag, "--endpoint", OLLAMA, "--model", model,
           *(EVID_EXTRA_ARGS if extra_args is None else extra_args),
           "--instruction-mode", "student", "--retry-failed", "--blocklist", "exclude",
           "--hi-spec", specs[0], "--gia-spec", specs[1],   # 显式写出，日志里一眼可见
           "--search-mode", "replay_or_record", "--snapshot-root", str(EVID / "search_snapshots"),
           "--runtime-guards", EVID_GUARDS, "--state-renderer", EVID_RENDER,
           *(EVID_STAGE_ARGS if stage_args is None else stage_args)]
    if ids:
        run += ["--ids", *(f"evid:{i}" for i in ids)]      # run_cases 认 case_id（evid:video_001）或案卷文件名
    log.run(f"[2/4] 跑调查：{title}（{model}，{specs[0]} / {specs[1]}，只给学生提示，兜底 {EVID_GUARDS}，"
            f"渲染 {EVID_RENDER}，轮次 {round_tag}）", run, show=True)
    score = [PY, "scripts/_evid_score.py", "--round", round_tag] + (["--ids", *ids] if ids else [])
    log.run("[3/4] 答案切分、完成情况与行为面板（不调评委）", [*score, "--parse-only"])
    return 0


def evid_smoke() -> int:
    """EVID-Bench 冒烟：3 个样本（三大类各一，已降级出 test），跑完直接用 Sonnet 评分（10-05 起不再用本机三评委）。"""

    log = Log("_EVID冒烟日志.txt", f"EVID-Bench 冒烟：3 个样本 × 本机 {EVID_MODEL_NAME}（阶段 {EVID_STAGE}，规格 "
                                   f"{EVID_SPECS[0]} / {EVID_SPECS[1]}，兜底 {EVID_GUARDS}，轮次 {EVID_ROUND}）")
    print(f"正在运行：本机 Ollama 的 {EVID_MODEL} 跑 3 个 EVID 样本（{', '.join(EVID_SMOKE)}），再用 Claude Sonnet 评分。")
    print(f"中途关窗口没关系，再双击会跳过已跑完的 run。详细输出在 {log.path.name}")
    if _evid_run(log, EVID_SMOKE, "冒烟 3 个") != 0:
        log.finish("没有开跑。")
        return 1
    log.run("[4/4] Sonnet 评分（官方 EVID 评分提示词）",
            [PY, "scripts/_evid_score.py", "--round", EVID_ROUND, "--ids", *EVID_SMOKE, "--judges", EVID_SONNET], show=True)
    log.finish(f"汇总在 {EVID}\\scores\\{EVID_ROUND}.sonnet.subset.md；单个 run 在 {EVID}\\runs\\；日志 {log.path.name}")
    return 0


def evid_full() -> int:
    """EVID-Bench test 集 219 个（已降级的 3 个不跑）；断点续跑。正式分数用 evid_score_sonnet。"""

    log = Log("_EVID全量日志.txt", f"EVID-Bench 全量：219 个样本 × 本机 {EVID_MODEL_NAME}（阶段 {EVID_STAGE}，规格 "
                                   f"{EVID_SPECS[0]} / {EVID_SPECS[1]}，兜底 {EVID_GUARDS}，轮次 {EVID_ROUND}）")
    print(f"正在运行：{EVID_MODEL} 跑 EVID test 集 219 个样本"
          + ("（27B，同时 2 个，预计一天半左右）。" if EVID_MODEL_PROFILE != "8b" else "。"))
    print(f"可以随时关窗口，再双击会从断点继续。详细输出在 {log.path.name}；出正式分数请双击 scripts\\_evid_score_sonnet.cmd")
    if _evid_run(log, None, "全量 219 个") != 0:
        log.finish("没有开跑。")
        return 1
    log.finish(f"单个 run 在 {EVID}\\runs\\evid_*_{EVID_ROUND}_1；出正式分数请双击 scripts\\_evid_score_sonnet.cmd")
    return 0


def evid_score() -> int:
    """只重新打分（例如换评委、补跑了几个 run 之后）。"""

    log = Log("_EVID评分日志.txt", f"EVID-Bench 评分：轮次 {EVID_ROUND}")
    if not evid_preflight(log):
        log.finish("没有打分。")
        return 1
    score = [PY, "scripts/_evid_score.py", "--round", EVID_ROUND, "--endpoint", OLLAMA]
    log.run("[1/2] 答案切分与完成情况", [*score, "--parse-only"])
    log.run("[2/2] 三评委打分", [*score, "--judges", *(f"local:{m}" for m in EVID_JUDGES)], show=True)
    log.finish(f"汇总在 {EVID}\\scores\\{EVID_ROUND}.*.md")
    return 0


def evid_score_sonnet() -> int:
    """用 Claude Sonnet（本机 Claude Code CLI，走订阅额度）按官方提示词给 EVID 打分；只打已有的 run，不跑调查。

    10-05 冒烟看到本机三评委明显偏松（把"AI 生成"判成命中"拆除爆破配地震字幕"），正式数字以 Sonnet 为准，
    本机三评委的分数保留作对照。缓存在 experiments\\evid\\scores\\judge_cache.claude-cli-*.jsonl，可随时中断再双击续上。
    """

    log = Log("_EVID评分Sonnet日志.txt", f"EVID-Bench 评分（Sonnet）：轮次 {EVID_ROUND}")
    print("正在运行：Claude Sonnet 给已跑完的 EVID 样本打分（每个样本 1 次调用，走 Claude 订阅额度）。")
    print(f"可以随时关窗口，再双击会用缓存续上。详细输出在 {log.path.name}")
    score = [PY, "scripts/_evid_score.py", "--round", EVID_ROUND, "--judges", EVID_SONNET]
    log.run("[1/2] Sonnet 评分：冒烟 3 个（已降级出 test，只看改动效果）", [*score, "--ids", *EVID_SMOKE], show=True)
    log.run("[2/2] Sonnet 评分：test 集（不含已降级的样本；没跑的按 0，另有\"已开跑\"一行）", score, show=True)
    log.finish(f"冒烟在 {EVID}\\scores\\{EVID_ROUND}.sonnet.subset.md，test 集在 {EVID_ROUND}.sonnet.md")
    return 0


def evid_ab_p0() -> int:
    """修正优化计划 P0 的 A/B：8B × 20 个样本（evidb4 已跑过的 002–130 里均匀取），evidb6 vs evidb4，同一 Sonnet 评委。

    evidb6 = 规格 hi@0.5.0 / gia@0.6.0（不改模型输入）+ 评测档强制 compact_16k（端点按档位渲染）+ schema 约束解码，
    --max-rounds 2 与 evidb4 一致。两轮都出行为面板（覆盖率、候选下载、GIA 末次原件帧、重复率、减图、被拒率、崩溃）。
    """

    round_tag = "evidb6"
    log = Log("_EVID_P0对照日志.txt", f"EVID 修正计划 P0 A/B：{len(EVID_AB_IDS)} 个样本 × Qwen3-VL-8B，{round_tag} vs evidb4")
    print(f"正在运行：本机 8B 跑 {len(EVID_AB_IDS)} 个 EVID 样本（轮次 {round_tag}，P0：不改模型输入），"
          "再用 Claude Sonnet 给 evidb6 和 evidb4 的同一批样本打分。")
    print(f"预计 1–2 小时；中途关窗口没关系，再双击会跳过已跑完的 run。详细输出在 {log.path.name}")
    if not evid_preflight(log):
        log.finish("没有开跑。")
        return 1
    schema_probe(log, EVID8_MODEL)
    _evid_run(log, EVID_AB_IDS, f"P0 A/B {len(EVID_AB_IDS)} 个", model=EVID8_MODEL, config=EVID8_CONFIG, extra_args=(),
              round_tag=round_tag, specs=EVID_STAGE_SPECS["p0"], stage_args=("--max-rounds", "2"), preflight=False)
    for tag in ("evidb4", round_tag):
        log.run(f"[4/4] Sonnet 评分：{tag}（同一批 {len(EVID_AB_IDS)} 个）",
                [PY, "scripts/_evid_score.py", "--round", tag, "--ids", *EVID_AB_IDS, "--judges", EVID_SONNET], show=True)
    log.finish(f"对照看 {EVID}\\scores\\{round_tag}.sonnet.subset.md 与 evidb4.sonnet.subset.md（各自末尾有行为面板）；"
               f"日志 {log.path.name}")
    return 0


EVID27_MIN_OLLAMA = (0, 32, 11)        # qwen3.8 系列要 Ollama ≥ 0.32.11；旧版 pull 报 412 "requires a newer version of Ollama"


def ollama_version() -> tuple[int, ...] | None:
    import json as _json
    import re
    import urllib.request

    base = OLLAMA.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=5) as response:
            text = _json.loads(response.read().decode("utf-8")).get("version", "")
    except Exception:  # noqa: BLE001
        return None
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(x) for x in match.groups()) if match else None


def evid27_preflight(log: Log) -> bool:
    """Ollama 在不在；没有 qwen3.8:27b 就 pull（约 18GB），没有 32K 版就按 Modelfile 建；最后让模型答一句试试。"""

    served = ollama_models()
    if served is None:
        served = start_ollama(log)
    if served is None:
        log.say(f"连不上 Ollama（{OLLAMA}），自动启动也没成功。请手动打开 Ollama 后再双击本文件。")
        log.failed.append("Ollama 未就绪")
        return False
    log.write(f"Ollama 上的模型：{served}")
    exe = ollama_exe() or "ollama"
    version = ollama_version()
    log.say(f"Ollama 版本：{'.'.join(map(str, version)) if version else '未知'}")
    if EVID27_MODEL not in served and version is not None and version < EVID27_MIN_OLLAMA:
        log.failed.append("Ollama 版本太旧")
        log.say(f"qwen3.8:27b 需要 Ollama {'.'.join(map(str, EVID27_MIN_OLLAMA))} 以上，本机是 {'.'.join(map(str, version))}。请先更新 Ollama：\n"
                "  1. 右键任务栏右下角的 Ollama 图标，有\"Restart to update\"就点它；\n"
                "  2. 没有的话，到 https://ollama.com/download 下载 Windows 安装包直接安装（覆盖安装，已有模型不会丢）。\n"
                "更新完再双击本文件。")
        return False
    if EVID27_MODEL not in served:
        if EVID27_BASE not in served and f"{EVID27_BASE}:latest" not in served:
            log.say(f"本机还没有 {EVID27_BASE}，开始下载（约 18GB，视网速可能要十几分钟到一小时；中断后再双击会续传）。")
            if log.run(f"[0] 下载 {EVID27_BASE}", [exe, "pull", EVID27_BASE], show=True) != 0:
                log.say("下载没成功：检查网络，或先在命令行手动 ollama pull qwen3.8:27b。")
                return False
        if log.run(f"[0] 建 {EVID27_MODEL}（{EVID27_BASE} + 32K 上下文）",
                   [exe, "create", EVID27_MODEL.split(":")[0], "-f", str(EVID27_MODELFILE)], show=True) != 0:
            return False
        served = ollama_models() or served
    if EVID27_MODEL not in served:
        log.failed.append("缺模型")
        log.say(f"Ollama 上还是没有 {EVID27_MODEL}，没有开跑。")
        return False
    return evid27_probe(log)


def evid27_probe(log: Log) -> bool:
    """关思考发一个小请求：确认能出 JSON、没有思考段，顺便把 27B 装进显存，记下耗时。"""

    import json as _json
    import urllib.request

    body = {"model": EVID27_MODEL, "temperature": 0.6, "max_tokens": 64, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": 'Return exactly {"ok": true} and nothing else.'}]}
    started = time.monotonic()
    try:
        request = urllib.request.Request(OLLAMA.rstrip("/") + "/chat/completions", data=_json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            data = _json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.say(f"试答失败：{type(exc).__name__}: {exc}")
        log.failed.append("27B 试答")
        return False
    message = (data.get("choices") or [{}])[0].get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    log.say(f"27B 试答（含装载）{time.monotonic() - started:.0f} 秒：{(message.get('content') or '').strip()[:80]!r}"
            + (f"；仍有思考段 {len(reasoning)} 字（reasoning_effort=none 没生效，会很慢）" if reasoning else "；思考已关"))
    return True


def evid27_full() -> int:
    """兼容旧入口（scripts\\_evid27_full.cmd）：10-05 晚起 evid_full 默认就是 27B。"""

    return evid_full()


def evid27_score_sonnet() -> int:
    """兼容旧入口（scripts\\_evid27_score_sonnet.cmd）。"""

    return evid_score_sonnet()


# ---------------------------------------------------------------------------- 10-06 精度对照：bf16 vs Q4
# 现有 qwen3vl-8b-32k 建自 qwen3-vl:8b-instruct，这个标签就是 Q4_K_M（6.1GB，ID 0533d74300e4）。训练后的学生在服务器上
# 以 bf16 部署，基线也应同精度。bf16 权重 18GB；32K 上下文 KV 每个并发约 4.5GB，5090（32GB）放一个并发宽裕、两个偏紧，
# 所以本入口只用一个并发槽（OLLAMA_NUM_PARALLEL=1 只在由本脚本起 ollama serve 时生效；托盘里已开的 Ollama 用它自己的设置，
# 末尾的 ollama ps 会记下实际是否全在显卡上）。其余与 evidb12 完全一致：规格 p1、评测兜底、渲染档、检索回放、3 轮、--jobs。
EVID8_BF16_BASE = "qwen3-vl:8b-instruct-bf16"
EVID8_BF16_MODEL = "qwen3vl-8b-bf16-32k:latest"
EVID8_BF16_MODELFILE = Path("scripts") / "ollama" / "qwen3vl_8b_bf16_32k.Modelfile"
EVID8_BF16_ROUND = os.environ.get("FOREN_EVID_BF16_ROUND") or "evidb12bf"
EVID8_Q4_ROUND = "evidb12"          # 对照侧：同规格的 Q4 全量（这 20 个样本已跑过）
BF16_MIN_FREE_GB = 25               # 18GB 权重 + 余量


def _ollama_models_dir() -> Path:
    custom = os.environ.get("OLLAMA_MODELS")
    return Path(custom) if custom else Path.home() / ".ollama" / "models"


def _ollama_ps(log: Log, label: str) -> None:
    exe = ollama_exe()
    if exe:
        log.run(f"{label}：ollama ps（PROCESSOR 一栏应为 100% GPU；出现 CPU 说明显存不够、在内存里跑，会很慢）",
                [exe, "ps"], show=True)


def bf16_preflight(log: Log) -> bool:
    """Ollama 在不在（不在就以单并发起）；没有 bf16 权重就 pull（约 18GB，先查磁盘）；没有 32K 版就建；再试一次 schema 解码。"""

    os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")       # 只影响由本脚本起的 ollama serve
    served = ollama_models()
    if served is None:
        served = start_ollama(log)
    if served is None:
        log.say(f"连不上 Ollama（{OLLAMA}），自动启动也没成功。请手动打开 Ollama 后再双击本文件。")
        log.failed.append("Ollama 未就绪")
        return False
    log.write(f"Ollama 上的模型：{served}")
    exe = ollama_exe() or "ollama"
    if EVID8_BF16_MODEL not in served:
        if EVID8_BF16_BASE not in served:
            models_dir = _ollama_models_dir()
            probe = models_dir if models_dir.exists() else Path(models_dir.anchor or ".")
            free_gb = shutil.disk_usage(probe).free / 2**30
            log.say(f"Ollama 模型目录 {models_dir}，所在盘剩余 {free_gb:.1f} GB")
            if free_gb < BF16_MIN_FREE_GB:
                log.failed.append("磁盘空间不够")
                log.say(f"下载 {EVID8_BF16_BASE} 需要约 18GB，建议至少留 {BF16_MIN_FREE_GB}GB；先清出空间再双击。")
                return False
            log.say(f"本机还没有 {EVID8_BF16_BASE}，开始下载（约 18GB；中断后再双击会续传）。")
            if log.run(f"[0] 下载 {EVID8_BF16_BASE}", [exe, "pull", EVID8_BF16_BASE], show=True) != 0:
                log.say(f"下载没成功：检查网络，或先在命令行手动 ollama pull {EVID8_BF16_BASE}。")
                return False
        if log.run(f"[0] 建 {EVID8_BF16_MODEL}（{EVID8_BF16_BASE} + 32K 上下文，与 qwen3vl-8b-32k 只差精度）",
                   [exe, "create", EVID8_BF16_MODEL.split(":")[0], "-f", str(EVID8_BF16_MODELFILE)], show=True) != 0:
            return False
        served = ollama_models() or served
    if EVID8_BF16_MODEL not in served:
        log.failed.append("缺模型")
        log.say(f"Ollama 上还是没有 {EVID8_BF16_MODEL}，没有开跑。")
        return False
    schema_probe(log, EVID8_BF16_MODEL)                      # 顺便把 bf16 装进显存
    _ollama_ps(log, "[0] 装载后显存检查")
    return True


def evid_ab_bf16() -> int:
    """精度对照：同 20 个样本（EVID_AB_IDS），bf16（evidb12bf）vs Q4（evidb12），同一 Sonnet 评委，各自出行为面板。"""

    log = Log("_EVID_bf16对照日志.txt", f"EVID 精度对照：{len(EVID_AB_IDS)} 个样本 × Qwen3-VL-8B bf16（{EVID8_BF16_ROUND}）"
                                      f" vs Q4（{EVID8_Q4_ROUND}）")
    print(f"正在运行：本机 8B bf16 跑 {len(EVID_AB_IDS)} 个 EVID 样本（轮次 {EVID8_BF16_ROUND}，规格与 {EVID8_Q4_ROUND} 相同），"
          f"再用 Claude Sonnet 给两轮的同一批样本打分。第一次会先下载约 18GB 的 bf16 权重。")
    print(f"预计 2–3 小时（不含下载）；中途关窗口没关系，再双击会跳过已跑完的 run。详细输出在 {log.path.name}")
    if not bf16_preflight(log):
        log.finish("没有开跑。")
        return 1
    _evid_run(log, EVID_AB_IDS, f"bf16 对照 {len(EVID_AB_IDS)} 个", model=EVID8_BF16_MODEL, config=EVID8_CONFIG,
              extra_args=("--jobs", EVID8_JOBS), round_tag=EVID8_BF16_ROUND, specs=EVID_STAGE_SPECS["p1"],
              stage_args=(), preflight=False)
    _ollama_ps(log, "[2/4] 跑完后显存检查")
    for tag in (EVID8_Q4_ROUND, EVID8_BF16_ROUND):
        log.run(f"[4/4] Sonnet 评分：{tag}（同一批 {len(EVID_AB_IDS)} 个）",
                [PY, "scripts/_evid_score.py", "--round", tag, "--ids", *EVID_AB_IDS, "--judges", EVID_SONNET], show=True)
    log.finish(f"对照看 {EVID}\\scores\\{EVID8_BF16_ROUND}.sonnet.subset.md 与 {EVID8_Q4_ROUND}.sonnet.subset.md"
               f"（各自末尾有行为面板）；日志 {log.path.name}")
    return 0


JOBS = {"evid_smoke": evid_smoke, "evid_full": evid_full, "evid_score": evid_score, "evid_score_sonnet": evid_score_sonnet,
        "evid_ab_p0": evid_ab_p0, "evid_ab_bf16": evid_ab_bf16, "evid27_full": evid27_full,
        "evid27_score_sonnet": evid27_score_sonnet}


def main() -> int:
    job = sys.argv[1] if len(sys.argv) > 1 else ""
    if job not in JOBS or len(sys.argv) > 2:
        print("用法：python scripts\\_launch_evid.py " + " | ".join(JOBS))
        return 2
    env_setup()
    return JOBS[job]()


if __name__ == "__main__":
    raise SystemExit(main())
