"""walker D1（2026-10-12）：程序走、教师填，在 B 族（给定媒体）9 个 dev_train 案例上各跑一条，核验后与 r0e / tp2 对照。

    python scripts\\walker_run.py d1                      # 全流程：Codex 预检 → 9 个 run → Sonnet 核验 + 导出 → 对照表
    python scripts\\walker_run.py d1 --cases videosham_4126 --no-verify   # 只跑一个、不核验（冒烟）
    python scripts\\walker_run.py compare                 # 只重算对照表
    python scripts\\walker_run.py d1 --round wk11 --d2 --skip-b --teacher claude --arms
        # 10-14：教师换 Claude CLI（Sonnet）；S5 四版（d0 首稿 / b 原方法 / g 证据门控 / n 择优）分别核验 → walker_wk11_arms.md
    python scripts\\walker_run.py freeze --round wk16          # 10-15：只做冻结预检（规格 / 工具层 hash / 教师 / 核验器）
    python scripts\\walker_run.py freeze --write               # 工具层定稿后重算 specs\\r0_freeze.json 的工具层与核验器 hash
    python scripts\\walker_run.py d1 --round r0s1 --cases videosham_4126 fmnv_c_RXR0B1SxiU0
        # 10-15 对话 5a：默认教师 = 冻结文件的分槽表；--teacher-backup qwen|strong|all 整条 run 换备用
    FOREN_FREEZE_STRICT=1 python scripts\\walker_run.py d1 --round r0a --jobs 2 --cases …
        # 10-15 对话 5（本机、全部订阅）：S0–S3 OpenRouter Qwen 397B，S4/S5 Codex CLI（strong type=codex），核验 = 冻结的
        # --judge claude-cli（Sonnet，订阅）。Codex 撞 usage limit → 该 run 换 --teacher-backup strong 重跑、本批其余 run 也用备用
        # （manifest teacher_role=backup，> 20% 报警）；Claude CLI 撞额度 → 停批、写 run_manifests/<轮次>.checkpoint.json，再跑续上

r0 冻结（10-15，claude/r0冻结口径_20261015.md）：d1 开跑前在教师预检之后做冻结预检——本轮规格与 specs/*_freeze.json
一致时，比对工具层 hash、规格 hash、核验器代码 hash，看教师是不是冻结的主教师 / 备用教师、核验是不是冻结的模型；
不一致只报警（FOREN_FREEZE_STRICT=1 时不开跑）。每个 run 的 run_config 另有 tool_layer_hash 与 freeze 逐项结果（run_cases 写）。

为什么单独一个脚本、不改 run_cases.py：写这份脚本时本机有一个 EVID 任务正在跑（run_cases 每个 run 会重新加载代码），
所以只加新文件，不动已有的 src 文件。教师替换在本进程里做（monkeypatch run_cases.make_teacher → OracleGenerator），
其余 run_one 的全部流程（规格参数、渲染档位、工具、检索快照、落盘）与 Codex 教师完全相同。

新增文件：src/foren_searcher/datagen/walker.py（状态表）、experiments/sft_data/oracle_generator.py（落盘 + 语言槽）、
本脚本、scripts/walker_compare.py（对照）、scripts/_walker_d1.cmd（双击入口）。
产物：experiments/sft_data/runs/*_wk1_1、run_manifests/wk1.jsonl、verdicts_wk1_*、export/wk1_*、
experiments/sft_data/walker_d1_compare.md / .json，日志 _walker_D1日志.txt。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SFT = Path("experiments") / "sft_data"
RUNS = SFT / "runs"
CASES_DIR = SFT / "cases"
PY = sys.executable
CODEX_MODEL = os.environ.get("FOREN_CODEX_MODEL") or "gpt-6-astra"
SPECS = ["--hi-spec", "hi@0.7.0", "--gia-spec", "gia@0.8.0"]


def specs_for(round_tag: str) -> tuple[str, str]:
    """wk1–wk4：hi@0.7.0 / gia@0.8.0；wk5 起合并目录 hi@0.8.0 / gia@0.9.0（DETECT 并入 compare_media / video_structure）。"""

    import re

    m = re.match(r"wk(\d+)", round_tag)          # 10-14：wk11b / wk11g / wk11n（S5 对照的各版）按数字部分取规格
    if not m and round_tag.startswith("r1"):
        # 10-17 r1：r1* 轮次用 specs/r1_freeze.json 的规格（hi@0.11.0 / gia@0.9.0）；草稿里还没填就退回 r0
        try:
            frozen = json.loads((ROOT / "specs" / "r1_freeze.json").read_text(encoding="utf-8")).get("specs") or {}
        except (OSError, ValueError):
            frozen = {}
        if frozen.get("hi") and frozen.get("gia"):
            return frozen["hi"], frozen["gia"]
    if not m:
        # 10-15 r0 冻结：不是 wk<数字> 的轮次（第一次 SFT 的重跑等）用 specs/r0_freeze.json 的冻结规格
        try:
            frozen = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8")).get("specs") or {}
        except (OSError, ValueError):
            frozen = {}
        if frozen.get("hi") and frozen.get("gia"):
            return frozen["hi"], frozen["gia"]
    if m and int(m.group(1)) >= 8:
        # 10-13：hi@0.9.0 = 0.8.0 + SEARCH 日期查询（search_dates）；hi@0.10.0 = 0.9.0 + compact_tool_params
        # （compare_media 无 mode、打开视频自动带字幕）。wk8 还没跑过，直接用 0.10.0
        return "hi@0.10.0", "gia@0.9.0"
    if m and int(m.group(1)) >= 5:
        return "hi@0.8.0", "gia@0.9.0"
    return "hi@0.7.0", "gia@0.8.0"
# 环境变体（10-13）：案例名后缀 _<变体> = 同一案卷、改一个 harness 参数再跑一条（run 名 <案例>_<变体>_<轮次>_1）。
# v9：max_evidence_cards=6 + max_history_entries=4——卡片清单与历史都变短，长一点的工作区里早先的结果（比对报告、
# 转写、OCR）会整个掉出视图（只剩"看过的 E"编号），交报告前要 QUERY_MEMORY 查回来
VARIANTS: dict[str, list[str]] = {"v9": ["max_evidence_cards=6", "max_history_entries=4"]}
V9_CASES = ["videosham_1028_v9", "vcsl_pair_5_v9"]


def split_variant(case: str) -> tuple[str, list[str]]:
    base, _, tail = case.rpartition("_")
    if base and tail in VARIANTS:
        return base, list(VARIANTS[tail])
    return case, []


D1_CASES = ["videosham_0069", "videosham_1007", "videosham_1028", "videosham_2010", "videosham_4126",
            "vcsl_pair_1", "vcsl_pair_5", "vcsl_pair_6", "qvhighlights_9167"]
ROUND = "wk1"
LOG_NAME = "_walker_D1日志.txt"           # wk1；之后的轮次写 _walker_<轮次>日志.txt


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Log:
    def __init__(self, name: str, title: str, append: bool = True) -> None:
        # 10-15：日志名按轮次取，同轮次再跑（冒烟后全量、断点续跑）接在后面写，不覆盖前一次（r0a 冒烟日志被全量覆盖过）
        self.path = ROOT / name
        mode = "a" if append and self.path.exists() else "w"
        with self.path.open(mode, encoding="utf-8") as stream:
            stream.write(f"==== {title} {now()}\n")
        self.failed: list[str] = []

    def write(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.write(text)

    def run(self, label: str, argv: list[str], *, env: dict[str, str] | None = None, log_file: Path | None = None) -> int:
        self.say(f"{label}（{now()}）")
        self.write("  > " + " ".join(argv))
        target = log_file or self.path
        with target.open("a", encoding="utf-8") as stream:
            try:
                proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", env=env)
            except OSError as exc:
                self.say(f"  启动失败：{exc}")
                return 127
            assert proc.stdout is not None
            for line in proc.stdout:
                stream.write(line)
                stream.flush()
            code = proc.wait()
        self.say(f"  退出码 {code}")
        if code != 0:
            self.failed.append(label.split("（")[0].strip())
        return code


def slot_backend() -> str:
    kind = os.environ.get("FOREN_ORACLE_SLOT", "").lower()
    if kind == "mock":
        return "mock"
    if kind == "slots":
        return "slots"
    if os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"):
        return "endpoint"
    return "claude" if kind == "claude" else "codex"


def _oracle_module() -> Any:
    oracle_dir = str((ROOT / SFT).resolve())
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    import oracle_generator  # type: ignore

    return oracle_generator


def slot_table() -> dict[str, Any] | None:
    """10-15 对话 5a：分槽模式下当前生效的槽 → 后端表（冻结文件 + 环境变量覆盖 + 备用开关）；其余模式 None。"""

    if slot_backend() != "slots":
        return None
    return _oracle_module().load_slot_table()


def slot_teachers(table: dict[str, Any] | None = None) -> dict[str, dict[str, Any]] | None:
    """每个槽实际用的模型：{"S0": {"backend": "qwen", "model": …, "role": "primary", "endpoint_host": …}, …}。"""

    table = table or slot_table()
    if not table:
        return None
    host = _oracle_module()._host
    return {key: {"backend": name, "model": table["backends"][name]["model"], "role": table["backends"][name]["role"],
                  "type": table["backends"][name].get("type") or "endpoint",
                  "endpoint_host": host(table["backends"][name]["endpoint"])} for key, name in table["slots"].items()}


def teacher_model() -> str:
    backend = slot_backend()
    if backend == "mock":                  # 10-15：冒烟用的假槽不记成 Codex（否则冻结预检会把它当主教师）
        return "mock"
    if backend == "slots":                 # 10-15 对话 5a：<S0–S3 模型>+<S4/S5 模型>，与 OracleGenerator.model_config.model 相同
        return slot_table()["summary"]
    if backend == "claude":
        return "claude-cli:" + (os.environ.get("FOREN_ORACLE_CLAUDE_MODEL") or "sonnet")
    if backend == "endpoint":
        return os.environ.get("FOREN_ORACLE_SLOT_MODEL") or "?"
    return os.environ.get("FOREN_CODEX_MODEL") or CODEX_MODEL


def env_setup() -> dict[str, str]:
    os.chdir(ROOT)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "FOREN_CODEX_MODEL": CODEX_MODEL})
    env.setdefault("FOREN_ORACLE_SLOT_EFFORT", "medium")
    os.environ.update({k: env[k] for k in ("PYTHONPATH", "PYTHONUTF8", "PYTHONIOENCODING", "FOREN_CODEX_MODEL",
                                           "FOREN_ORACLE_SLOT_EFFORT")})
    return env


def endpoint_preflight(log: Log, cfg: dict[str, Any] | None = None, label: str = "教师端点") -> bool:
    """10-14：服务器上教师是自己部署的模型（FOREN_ORACLE_SLOT_ENDPOINT）。开跑前发一条最小请求，确认服务在、模型名对、
    能返回文本；失败就不开跑。10-15：cfg 给了就查这个后端（分槽模式每个后端各查一次，key / 思考 / 温度按后端配置）。"""

    import httpx

    if cfg is None:
        endpoint = os.environ["FOREN_ORACLE_SLOT_ENDPOINT"].rstrip("/")
        model = os.environ.get("FOREN_ORACLE_SLOT_MODEL") or ""
        cfg = {"endpoint": endpoint, "model": model, "openrouter": "openrouter.ai" in endpoint,
               "api_key": os.environ.get("FOREN_TEACHER_API_KEY"), "trust_env": "openrouter.ai" in endpoint, "temperature": 0.0}
    endpoint, model, openrouter = str(cfg["endpoint"]).rstrip("/"), str(cfg.get("model") or ""), bool(cfg.get("openrouter"))
    log.say(f"[0] 检查{label} {endpoint} 能否用 {model or '(未设 FOREN_ORACLE_SLOT_MODEL)'}（{now()}）")
    if not model:
        log.say("  没设 FOREN_ORACLE_SLOT_MODEL（vLLM 里的模型名），没有开跑。")
        return False
    key = cfg.get("api_key") or os.environ.get("FOREN_TEACHER_API_KEY")
    if not key and openrouter:
        from foren_searcher.openrouter_settings import load_api_key

        key = load_api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0), trust_env=bool(cfg.get("trust_env", openrouter)),
                          headers=headers) as client:
            models = client.get(endpoint + "/models")
            served = [m.get("id") for m in (models.json().get("data") or [])] if models.status_code < 400 else []
            if served and model not in served:
                family = model.rsplit("/", 1)[-1].split("-")[0].lower()
                shown = [m for m in served if family in str(m).lower()] if openrouter else served
                log.say(f"  服务上没有模型 {model}；有的是：{', '.join(map(str, shown))[:1500]}。没有开跑。")
                return False
            body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                                    "max_tokens": 2048 if cfg.get("reasoning_effort") else 256}
            if cfg.get("temperature") is not None:
                body["temperature"] = 0.0
            if openrouter:
                body["reasoning"] = {"effort": "low"} if cfg.get("reasoning_effort") else {"enabled": False}
            elif "api.anthropic.com" in endpoint or "api.openai.com" in endpoint:
                pass
            else:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            reply = client.post(endpoint + "/chat/completions", json=body)
    except httpx.HTTPError as exc:
        log.say(f"  连不上{label}：{type(exc).__name__}: {exc}。没有开跑。")
        return False
    if reply.status_code >= 400:
        log.say(f"  {label}返回 HTTP {reply.status_code}：{reply.text[:300]}。没有开跑。")
        return False
    data = reply.json()
    text = str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    log.say(f"  回复：{text[:80]!r}（服务端模型 {data.get('model')}，供应商 {data.get('provider') or '-'}）")
    return bool(text)


def slots_preflight(log: Log) -> bool:
    """10-15 对话 5a：分槽模式——冻结文件的槽表读得出来，每个后端各发一条最小请求。"""

    try:
        table = slot_table()
    except Exception as exc:  # noqa: BLE001
        log.say(f"[0] 读不出冻结文件的分槽表：{exc}。没有开跑。")
        return False
    assert table is not None
    for key, item in (slot_teachers(table) or {}).items():
        log.say(f"  槽 {key} → {item['backend']}（{item['role']}）{item['model']} @ {item['endpoint_host']}")
    ok = True
    for name, cfg in table["backends"].items():
        label = f"{name} 后端（{cfg['role']}）"
        if (cfg.get("type") or "endpoint") == "codex":
            ok = codex_backend_preflight(log, cfg, label=label) and ok
        else:
            ok = endpoint_preflight(log, cfg, label=label) and ok
    if any((c.get("type") or "endpoint") == "codex" for c in table["backends"].values()):
        log.say("  （Codex 后端跨进程串行：--jobs 建议 2——一个 run 等 Codex 时另一个跑 Qwen 槽和工具）")
    return ok


def codex_backend_preflight(log: Log, cfg: dict[str, Any], label: str = "Codex 后端") -> bool:
    """10-15 r0 成本压缩：分槽表里 type=codex 的后端（strong = gpt-6-astra 走 Codex CLI / ChatGPT 订阅）。
    找得到 codex、记版本与登录状态，再用冻结的模型与 effort 发一条最小请求（只读、无工具、--json），确认能回文本。
    429 / usage limit 时直接报出来：等额度恢复，或 --teacher-backup strong 整条 run 换 OpenRouter Sonnet。"""

    og = _oracle_module()
    model = str(cfg.get("model") or "")
    log.say(f"[0] 检查{label}：Codex CLI，模型 {model}，effort {cfg.get('reasoning_effort') or '-'}（{now()}）")
    try:
        codex = og.find_codex(cfg)
    except RuntimeError as exc:
        log.say(f"  {exc}。没有开跑。")
        return False
    log.say(f"  {codex}；版本 {og.codex_version(codex) or '?'}")
    try:
        status = subprocess.run([codex, "login", "status"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=60)
        text = ((status.stdout or "") + (status.stderr or "")).strip().splitlines()
        log.say(f"  登录状态：{(text[-1] if text else '?')[:200]}")
    except (OSError, subprocess.SubprocessError) as exc:
        log.say(f"  读不出登录状态（{type(exc).__name__}），继续试一条请求")
    with tempfile.TemporaryDirectory(prefix="codex_preflight_") as tmp:
        out = Path(tmp) / "reply.txt"
        command = [codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--json",
                   "-C", tmp, "-o", str(out), "-m", model]
        if cfg.get("reasoning_effort"):
            command += ["-c", f'model_reasoning_effort="{cfg["reasoning_effort"]}"']
        command += ["--", "-"]
        try:
            with og.codex_lock(serial=cfg.get("serial", True)):
                done = subprocess.run(command, input="Reply with the single word: ok", text=True, encoding="utf-8",
                                      errors="replace", capture_output=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            log.say(f"  Codex 调用失败：{type(exc).__name__}: {exc}。没有开跑。")
            return False
        reply = out.read_text(encoding="utf-8").strip() if out.is_file() else ""
        events = []
        for line in (done.stdout or "").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        served = og.codex_model_served(events, done.stderr or "")
    if done.returncode != 0 or not reply:
        tail = ((done.stdout or "") + (done.stderr or ""))[-600:]
        if og.codex_rate_limited(tail):
            log.say(f"  Codex 到了用量上限 / 429（usage limit）：{tail[-300:]!r}。等额度恢复再跑，或 --teacher-backup strong"
                    "（整条 run 换 OpenRouter Sonnet）。没有开跑。")
        else:
            log.say(f"  Codex 返回码 {done.returncode}，回复 {reply[:80]!r}：{tail[-300:]!r}。没有开跑。")
        return False
    log.say(f"  回复：{reply[:80]!r}（请求模型 {model}，Codex 输出里的模型 {served or '未给出'}）")
    if served and not og.model_matches(served, model):
        log.say(f"  Codex 实际用的模型 {served} 与冻结的 {model} 不同，没有开跑（检查 ~/.codex/config.toml 的 profile / 账号可用模型）。")
        return False
    return True


def claude_preflight(log: Log) -> bool:
    """10-14：教师换成本机 Claude CLI。发一道小题确认能用、登录有效，把别名换成实际模型名记进日志。"""

    from foren_searcher.datagen.verify.judges import ClaudeCliJudge, ClaudeCliLimit

    model = os.environ.get("FOREN_ORACLE_CLAUDE_MODEL") or "sonnet"
    log.say(f"[0] 检查 Claude CLI 教师（{model}）能否用（{now()}）")
    try:
        judge = ClaudeCliJudge(model=model, timeout=180)
        name = judge.resolve()
        judge.close()
    except (Exception, ClaudeCliLimit) as exc:  # noqa: BLE001
        log.say(f"  Claude CLI 用不了：{str(exc)[:400]}。没有开跑。")
        return False
    log.say(f"  可用：{name}")
    return True


def codex_preflight(log: Log) -> bool:
    if slot_backend() == "slots":
        return slots_preflight(log)
    if os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"):
        return endpoint_preflight(log)
    if slot_backend() == "claude":
        return claude_preflight(log)
    codex = shutil.which("codex")
    if not codex:
        log.say("找不到 codex 命令（PATH 里没有），没有开跑。")
        return False
    code = log.run(f"[0] 检查 Codex 能否用 {CODEX_MODEL}",
                   [codex, "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "-m", CODEX_MODEL, "只回复 ok"])
    if code != 0:
        log.say(f"Codex 用不了 {CODEX_MODEL}，没有开跑。看 {log.path.name} 末尾的报错。")
        return False
    return True


def claude_judge_preflight(log: Log) -> bool:
    """10-15 对话 5（A5）：核验器改回 Claude CLI 订阅。开跑前确认是订阅登录（claude.ai / OAuth，不按量扣费）、Sonnet 能答一道小题，
    免得跑完几小时轨迹才在核验那步卡在"不是订阅登录，确认花钱吗"（批处理里没人回答）或额度用完。"""

    from foren_searcher.datagen.verify.judges import (ClaudeCliJudge, ClaudeCliLimit, claude_auth_method, find_claude_cli)

    model = judge_label().split(":", 1)[1]
    log.say(f"[0] 检查核验器 Claude CLI（{model}，订阅）（{now()}）")
    try:
        cli = find_claude_cli()
        method = claude_auth_method(cli) or "未知"
        log.say(f"  {' '.join(cli)}；登录方式 {method}")
        if not any(word in method.lower() for word in ("oauth", "claude.ai")):
            log.say("  不是订阅登录（claude -p 会按 API 用量扣费），没有开跑。先在终端 claude 里 /login 用订阅账号登录。")
            return False
        judge = ClaudeCliJudge(model=model, cli=cli, timeout=180)
        name = judge.resolve()
        judge.close()
    except (Exception, ClaudeCliLimit) as exc:  # noqa: BLE001
        log.say(f"  核验器 Claude CLI 用不了：{str(exc)[:400]}。没有开跑（--no-verify 可只跑轨迹）。")
        return False
    log.say(f"  可用：{name}")
    return True


CLAUDE_LIMIT_WORDS = ("ClaudeCliLimit",)
CODEX_LIMIT_MARK = "Codex usage limit"          # oracle_generator._codex_slot 退避用尽后的报错（带进 batch 日志末尾）


def checkpoint_path(round_tag: str) -> Path:
    return SFT / "run_manifests" / f"{round_tag}.checkpoint.json"


def write_checkpoint(round_tag: str, stage: str, reason: str, log: Log) -> None:
    """10-15 对话 5：Claude CLI 额度用完时停批，记下断点。下次双击（同一命令）续跑：已完成的 run 跳过，已判的核验命中缓存。"""

    path = checkpoint_path(round_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"round": round_tag, "stage": stage, "reason": reason[-600:], "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resume": f"再跑同一命令（python scripts\\walker_run.py d1 --round {round_tag} …）：run 已完成的跳过，核验命中 Claude CLI 缓存"}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.say(f"[断点] {stage} 时 Claude CLI 到了订阅额度，停批；断点 {path}。额度恢复后再双击续跑。")
    log.failed.append("Claude CLI 额度")


def _log_tail_since(path: Path, start: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read().decode("utf-8", errors="replace")[-20000:]
    except OSError:
        return ""


def judge_preflight(log: Log) -> bool:
    """10-15 r0 成本压缩：核验器 = Anthropic 直连（批量）。开跑前就确认 ANTHROPIC_API_KEY 能用、模型在、结构化输出能回，
    免得跑完几小时轨迹才在核验那步发现 key 不对（一道小题，约 $0.001）。"""

    from foren_searcher.datagen.api_judge import ANTHROPIC_ENDPOINT, AnthropicJudge, LocalJudgeUnavailable

    model = judge_label().split(":", 1)[1]
    log.say(f"[0] 检查核验器 Anthropic 直连 {model}（批量 + 缓存）（{now()}）")
    try:
        judge = AnthropicJudge(model=model, endpoint=ANTHROPIC_ENDPOINT if "api.anthropic.com" not in
                               str(os.environ.get("FOREN_JUDGE_API_ENDPOINT") or "") else os.environ["FOREN_JUDGE_API_ENDPOINT"])
        name = judge.resolve()
        judge.close()
    except LocalJudgeUnavailable as exc:
        log.say(f"  核验器用不了：{str(exc)[:400]}。没有开跑（--no-verify 可只跑轨迹）。")
        return False
    except Exception as exc:  # noqa: BLE001
        log.say(f"  核验器预检出错：{type(exc).__name__}: {str(exc)[:300]}。没有开跑。")
        return False
    log.say(f"  可用：{name}")
    return True


# ---------------------------------------------------------------------------- r0 冻结预检（10-15）

VERIFIER_CODE_PATHS = ("datagen/verify", "datagen/errors.py")
VERIFIER_TRANSPORT_PATHS = ("datagen/api_judge.py",)     # 10-15：--judge api 的传输层（不含提示，单独记 hash）


def verifier_code_hash(paths: tuple[str, ...] = VERIFIER_CODE_PATHS) -> str:
    """核验器代码（提示都写在这些文件里）的 hash，口径与 run_cases.tool_layer_hash 相同（包内相对路径 + LF）。"""

    from foren_searcher.datagen.run_cases import _hash_files, _layer_files

    package = ROOT / "src" / "foren_searcher"
    return _hash_files(package, _layer_files(package, paths))


def _frozen_verifier() -> dict[str, Any]:
    try:
        return json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8")).get("verifier") or {}
    except (OSError, ValueError):
        return {}


def judge_kind() -> str:
    """本轮核验走哪条路：FOREN_WALKER_JUDGE 优先；没设时取冻结文件的 verifier.judge（10-15 起是 api）；都没有用 claude-cli。"""

    if os.environ.get("FOREN_WALKER_JUDGE"):
        return os.environ["FOREN_WALKER_JUDGE"]
    frozen = _frozen_verifier()
    if frozen.get("judge") == "api" and frozen.get("mode") == "batch":
        return "api-batch"                 # 10-15 r0 成本压缩：Anthropic 直连 + Message Batches + prompt caching
    return frozen.get("judge") or "claude-cli"


def judge_label() -> str:
    """本机核验用的模型：api（10-15 起默认）是 FOREN_WALKER_JUDGE_MODEL / FOREN_JUDGE_API_MODEL / 冻结的 verifier.model；
    FOREN_WALKER_JUDGE=local 时是 FOREN_WALKER_JUDGE_MODEL；claude-cli 时是 Claude CLI 的 sonnet。"""

    judge = judge_kind()
    if judge == "local":
        return "local:" + (os.environ.get("FOREN_WALKER_JUDGE_MODEL") or "?")
    if judge in ("api", "api-batch"):
        model = (os.environ.get("FOREN_WALKER_JUDGE_MODEL") or os.environ.get("FOREN_JUDGE_API_MODEL")
                 or _frozen_verifier().get("model") or "claude-sonnet-5-5")
        if judge == "api-batch":
            model = model.split("/", 1)[-1].replace(".", "-")     # 直连的模型 id（与 api_judge.anthropic_model 相同）
        return "api:" + model
    return "claude-cli:" + (os.environ.get("FOREN_WALKER_JUDGE_MODEL") or "sonnet")


def teacher_check(teacher: dict[str, Any]) -> dict[str, Any]:
    """冻结预检的教师一项。分槽冻结（teacher.slots）：当前必须是分槽模式，且每个槽的实际模型 = 冻结表里该后端的主模型
    或备用模型（按模型身份比，容许服务器 served name 的大小写 / org 前缀 / -FP8 后缀）；老格式按 primary / backup 比。"""

    current = teacher_model()
    if not teacher.get("slots"):
        role = ("primary" if current == teacher.get("primary") else "backup" if current == teacher.get("backup") else None)
        return {"expected": [teacher.get("primary"), teacher.get("backup")], "actual": current, "ok": role is not None, "role": role}
    if slot_backend() != "slots":
        return {"expected": "slots（" + json.dumps(teacher.get("slots"), ensure_ascii=False) + "）", "actual": f"{slot_backend()}:{current}",
                "ok": False, "role": None}
    og = _oracle_module()
    actual = slot_teachers() or {}
    backups = teacher.get("backup_backends") or {}
    per_slot: dict[str, Any] = {}
    for key, name in teacher["slots"].items():
        primary = (teacher.get("backends") or {}).get(name, {}).get("model")
        backup = (backups.get(name) or {}).get("model")
        item = actual.get(key) or {}
        role = ("primary" if og.model_matches(item.get("model"), primary) else
                "backup" if og.model_matches(item.get("model"), backup) else None)
        per_slot[key] = {"backend": name, "expected": [primary, backup], "actual": item.get("model"), "role": role,
                         "declared_role": item.get("role"), "ok": role is not None and item.get("backend") == name}
    roles = {v["role"] for v in per_slot.values()}
    return {"expected": "分槽表", "actual": current, "ok": all(v["ok"] for v in per_slot.values()), "slots": per_slot,
            "role": "primary" if roles == {"primary"} else "backup" if roles == {"backup"} else
                    "mixed" if None not in roles else None}


def freeze_status(round_tag: str) -> dict[str, Any]:
    """本轮规格对应的冻结文件与当前代码 / 教师 / 核验器逐项比对。没有对应冻结文件时 applies=False。"""

    from foren_searcher.datagen.run_cases import load_freeze, tool_layer_hash
    from foren_searcher.datagen.specs import Spec

    hi_spec, gia_spec = specs_for(round_tag)
    freeze = load_freeze(ROOT / "specs", hi_spec, gia_spec)
    if not freeze:
        return {"applies": False, "specs": [hi_spec, gia_spec]}
    layer = tool_layer_hash()
    teacher = freeze.get("teacher") or {}
    verifier = freeze.get("verifier") or {}
    judge = judge_label()
    items = {
        "spec_hash.hi": (freeze.get("spec_hash") or {}).get("hi"), "spec_hash.gia": (freeze.get("spec_hash") or {}).get("gia"),
        "tool_layer_hash": (freeze.get("tool_layer") or {}).get("hash"), "verifier_code_hash": verifier.get("code_hash"),
    }
    actual = {"spec_hash.hi": Spec.load(ROOT / "specs" / hi_spec).hash(), "spec_hash.gia": Spec.load(ROOT / "specs" / gia_spec).hash(),
              "tool_layer_hash": layer["hash"], "verifier_code_hash": verifier_code_hash()}
    checks = {key: {"expected": value, "actual": actual[key], "ok": value == actual[key]} for key, value in items.items()}
    transport = verifier.get("transport") or {}
    if transport.get("hash"):
        current = verifier_code_hash(tuple([transport.get("path")]) if transport.get("path") else VERIFIER_TRANSPORT_PATHS)
        checks["verifier_transport_hash"] = {"expected": transport["hash"], "actual": current, "ok": current == transport["hash"]}
    checks["teacher_model"] = teacher_check(teacher)
    alias = verifier.get("model_alias") or ""
    expected_judge = f"{verifier.get('judge')}:{verifier.get('model')}"
    judge_ok = judge in (f"{verifier.get('judge')}:{alias}", expected_judge)
    if not judge_ok and verifier.get("judge") == "api" and judge.startswith("api:"):
        judge_ok = _oracle_module().model_matches(judge.split(":", 1)[1], verifier.get("model"))
    checks["judge"] = {"expected": expected_judge, "actual": judge, "ok": judge_ok}
    failed = [key for key, item in checks.items() if not item["ok"]]
    return {"applies": True, "freeze_id": freeze.get("freeze_id"), "file": freeze.get("_file"), "specs": [hi_spec, gia_spec],
            "ok": not failed, "failed": failed, "checks": checks, "tool_layer": layer,
            "tool_layer_status": (freeze.get("tool_layer") or {}).get("status"),
            "tool_layer_frozen_groups": (freeze.get("tool_layer") or {}).get("groups") or {}}


def freeze_preflight(log: Log, round_tag: str) -> bool:
    """d1 开跑前：与冻结口径比对，结果写日志和 run_manifests/<轮次>.freeze.json。不一致只报警；FOREN_FREEZE_STRICT=1 时不开跑。"""

    status = freeze_status(round_tag)
    if not status["applies"]:
        log.say(f"[冻结] 本轮规格 {' / '.join(status['specs'])} 没有对应的冻结文件（specs\\*_freeze.json），不按 r0 口径核对")
        return True
    out = SFT / "run_manifests" / f"{round_tag}.freeze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**status, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    teacher = status["checks"]["teacher_model"]
    if status["ok"]:
        role_text = {"primary": "主教师", "backup": "备用教师，只在主教师 429 时用", "mixed": "部分后端用备用"}.get(teacher.get("role"), "?")
        log.say(f"[冻结] {status['freeze_id']} 口径一致：规格 / 工具层 / 核验器 hash 都对上；教师 {teacher['actual']}（{role_text}）；"
                f"核验 {status['checks']['judge']['actual']}")
        return True
    for key in status["failed"]:
        item = status["checks"][key]
        log.say(f"[冻结] 警告：{key} 与 {status['freeze_id']} 不一致——应为 {item['expected']}，实际 {item['actual']}")
    if "tool_layer_hash" in status["failed"]:
        frozen = status["tool_layer_frozen_groups"]
        changed = [k for k, v in status["tool_layer"]["groups"].items() if frozen.get(k) != v] or ["?"]
        log.say(f"[冻结] 工具层（{', '.join(changed)}）的代码和冻结时不同：这一轮产出的行与冻结口径不可混用（清单 A8）。"
                f"工具层定稿后用 python scripts\\walker_run.py freeze --write 重算冻结值")
    if os.environ.get("FOREN_FREEZE_STRICT") == "1":
        log.say("[冻结] FOREN_FREEZE_STRICT=1：没有开跑")
        log.failed.append("冻结口径不一致")
        return False
    return True


def freeze_job(args: argparse.Namespace) -> int:
    """只做冻结预检；--write 把当前工具层 hash 与核验器代码 hash 写回冻结文件（工具层定稿后用）。"""

    env_setup()
    status = freeze_status(args.round)
    if not status["applies"]:
        print(f"本轮规格 {' / '.join(status['specs'])} 没有对应的冻结文件")
        return 1
    print(json.dumps({k: v for k, v in status.items() if k != "tool_layer"}, ensure_ascii=False, indent=2))
    print("工具层各部分：" + json.dumps(status["tool_layer"]["groups"], ensure_ascii=False, indent=2))
    if args.write:
        path = Path(status["file"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("tool_layer", {}).update({"hash": status["tool_layer"]["hash"], "files": status["tool_layer"]["files"],
                                                  "groups": status["tool_layer"]["groups"],
                                                  "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        data.setdefault("verifier", {})["code_hash"] = status["checks"]["verifier_code_hash"]["actual"]
        if "verifier_transport_hash" in status["checks"]:
            data["verifier"].setdefault("transport", {})["hash"] = status["checks"]["verifier_transport_hash"]["actual"]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写回 {path}：tool_layer.hash = {status['tool_layer']['hash']}；verifier.code_hash = "
              f"{status['checks']['verifier_code_hash']['actual']}；verifier.transport.hash = "
              f"{(status['checks'].get('verifier_transport_hash') or {}).get('actual')}")
        return 0
    return 0 if status["ok"] else 1


# ---------------------------------------------------------------------------- 单个 run（子进程里跑）

# r1（10-17，r1 预演之二）walker 策略版本：E 族以图搜取页面上的图并与 E1 比对、没比对的只写候选；L 族报告必须写参考图比对结论；
# C 族按场景取帧 + 整段转写交给 S4；第 2 轮沿用第 1 轮的族；循环内充分性检查与事后核验同一份 payload（verifier@0.4.0）
WALKER_POLICY_VERSION = "walker-r1@2026-10-17e"


def r1_search_policy(case_file: str, round_tag: str, mode: str | None) -> dict[str, Any] | None:
    """r1 轮次（r1*）的以图搜策略（按 10-16 Lens–Yandex 实测）：Google Lens 只给 E 族（Yandex 两族都开）；图床只给 E 族的
    案例图（FOREN_LENS_IMAGE_HOST=imgbb，只在这个 run 进程里设），litterbox 备用关；结果统一写法 + 缩略图自核
    （FOREN_REVERSE_NORMALIZE=1）。L / C 族默认不用 Lens。给定媒体（B 族）不检索。r0 轮次返回 None（行为不变）。"""

    if not round_tag.startswith("r1") or mode == "provided_media":
        return None
    from foren_searcher.datagen.walker import task_family

    try:
        task = json.loads(Path(case_file).read_text(encoding="utf-8")).get("task_text_en") or ""
    except (OSError, ValueError):
        task = ""
    family = task_family(str(task), True)
    os.environ.setdefault("FOREN_REVERSE_NORMALIZE", "1")
    if family == "E":
        os.environ.setdefault("FOREN_LENS_IMAGE_HOST", "imgbb")
        os.environ["FOREN_LENS_IMAGE_FALLBACK"] = "off"
        engines = "lens,yandex"
    else:
        os.environ.pop("FOREN_LENS_IMAGE_HOST", None)
        engines = "yandex"
    return {"family": family, "image_search": engines, "image_host": os.environ.get("FOREN_LENS_IMAGE_HOST") or "off",
            "image_fallback": os.environ.get("FOREN_LENS_IMAGE_FALLBACK") or "default",
            "normalize": os.environ.get("FOREN_REVERSE_NORMALIZE"), "lens_max_searches": os.environ.get("FOREN_LENS_MAX_SEARCHES")}


def run_one_case(case_file: str, run_name: str, round_tag: str, overrides: list[str] | None = None) -> int:
    from foren_searcher.datagen import run_cases as rc

    oracle_dir = str((ROOT / SFT).resolve())
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    from oracle_generator import OracleGenerator  # type: ignore

    from foren_searcher.datagen.teacher import RoleInstructions, role_of

    def make_teacher(kind: str, *, store: Any, output: Path, instructions: dict[str, str], endpoint: str | None,
                     model: str | None, thinking: bool, render_inputs: bool = False) -> Any:
        class RoleOracle(OracleGenerator):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._roles = RoleInstructions(instructions)

            async def generate_json(self, *, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
                self.instruction = self._roles.by_role.get(role_of(payload), "")
                return await super().generate_json(system_prompt=system_prompt, payload=payload)

        return RoleOracle(store=store, output_dir=output / "requests")

    rc.make_teacher = make_teacher          # run_one 里按模块全局名取 make_teacher
    # 10-15：造数据不该被时间限制截断。wk13（OpenRouter 上的 Qwen、开思考、3 条并行）有 5 个工作区跑满 configs/local.toml
    # 的 workspace_timeout_s=1800 被关掉，没交报告，GIA 按 G9_no_reports 直接 FINISH。这里把两个时限放宽到 4 小时 / 1 小时
    import dataclasses

    import foren_searcher.config as fs_config

    original_load = fs_config.load_config

    def load_config_generous(*a: Any, **kw: Any) -> Any:
        cfg = original_load(*a, **kw)
        runtime = dataclasses.replace(cfg.runtime,
                                      workspace_timeout_s=max(int(cfg.runtime.workspace_timeout_s or 0),
                                                              int(os.environ.get("FOREN_WALKER_WORKSPACE_TIMEOUT") or 14400)),
                                      gia_turn_timeout_s=max(int(cfg.runtime.gia_turn_timeout_s or 0),
                                                             int(os.environ.get("FOREN_WALKER_GIA_TURN_TIMEOUT") or 3600)))
        return dataclasses.replace(cfg, runtime=runtime)

    fs_config.load_config = load_config_generous
    forced = [x for x in os.environ.get("FOREN_WALKER_FORCE_OPS", "").split(",") if x]
    if forced:                              # 只用于没有模型权重的机器上冒烟（例如以图找帧走 OpenCV 描述子）
        original = rc.resolve_inspect_operations

        def resolve(mode: str = "auto", model_root: Any = None) -> Any:
            ops, detail = original(mode, model_root)
            return tuple(dict.fromkeys((*ops, *forced))), {**detail, "forced_for_smoke": forced}

        rc.resolve_inspect_operations = resolve
    hi_spec, gia_spec = specs_for(round_tag)
    # wk5：S5 写完后按案卷做 GT 核对（只在语言槽之后用，walker 的决定不读案卷）
    os.environ["FOREN_ORACLE_CASE"] = str((ROOT / case_file).resolve()) if not Path(case_file).is_absolute() else case_file
    # B 族（给定媒体）不检索，以图搜图不外发调查图片；A / C 族（D2，10-13）要检索，以图搜图按本机默认（打开）
    try:
        mode = json.loads(Path(case_file).read_text(encoding="utf-8")).get("source_policy", {}).get("mode")
    except (OSError, ValueError):
        mode = None
    extra = {"image_search": "off"} if hi_spec != "hi@0.7.0" and mode == "provided_media" else {}
    policy = r1_search_policy(case_file, round_tag, mode)
    if policy:
        extra = {"image_search": policy["image_search"]}
    outcome = asyncio.run(rc.run_one(
        case_file, run_name, runs_dir=str(RUNS), round_tag=round_tag, sample=1, teacher="codex", endpoint=None, model=None,
        thinking=True, specs_root="specs", hi_spec=hi_spec, gia_spec=gia_spec, blocklist="exclude",
        search_mode="replay_or_record", tool_cache="on", inspect_ops="auto", detect="off",
        harness_overrides=list(overrides or []) or None, **extra))
    cfg_path = RUNS / run_name / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["gia_output_source"] = "oracle_walker"
        cfg["driver"] = "scripts/walker_run.py"
        cfg["walker"] = {"plan": "轨迹生成计划_walker_能力覆盖为纲_20261012", "stage": "D1", "specs": [hi_spec, gia_spec],
                         "gt_check_after_answer": bool(os.environ.get("FOREN_ORACLE_CASE")),
                         "slot_effort": os.environ.get("FOREN_ORACLE_SLOT_EFFORT", "medium"),
                         "slot_backend": slot_backend(),
                         "s5_arms": os.environ.get("FOREN_ORACLE_ARMS") == "1",
                         "slot_endpoint": os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"),
                         "slot_model": os.environ.get("FOREN_ORACLE_SLOT_MODEL"),
                         "slot_mock": os.environ.get("FOREN_ORACLE_SLOT", "").lower() == "mock",
                         "variant_overrides": list(overrides or []),
                         "teacher_model": teacher_model(),
                         # 10-15 对话 5a：每个槽实际用的模型（分槽模式；其余模式所有槽同一个教师，记为 null）
                         "slot_teachers": slot_teachers(),
                         "slot_models": {k: v["model"] for k, v in (slot_teachers() or {}).items()} or None,
                         "teacher_backup": os.environ.get("FOREN_TEACHER_BACKUP") or None,
                         "policy_version": WALKER_POLICY_VERSION if round_tag.startswith("r1") else None,
                         "search_policy": policy}
        # 10-15 r0 冻结：顶层也记教师（run_cases 记的是 configured_model.model，这里以 walker 的槽为准），
        # 以及冻结口径里它是主教师还是备用、计划用哪个核验器（核验在 run 之后，实际用的写在 verdicts 文件名与导出行里）
        cfg["teacher_model"] = teacher_model()
        freeze = rc.load_freeze(ROOT / "specs", hi_spec, gia_spec) or {}
        teacher = freeze.get("teacher") or {}
        cfg["walker"]["teacher_role"] = teacher_check(teacher).get("role") if freeze else None
        cfg["walker"]["judge_planned"] = judge_label()
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False, default=str), flush=True)
    return 0


def slot_failures(run_dir: Path) -> list[str]:
    """10-14：教师槽失败的痕迹（工作区 runtime_error）。wk10 1KkAwCyj7nI 在 oracle 加"槽失败即停"之前就被标成了 done，
    这里认出来让它重跑；新 run 由 oracle_generator 的 slot_failed 直接停成 incomplete。"""

    path = run_dir / "trajectory.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"runtime_error"' not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event_type") == "runtime_error":
            out.append(json.dumps(row.get("payload"), ensure_ascii=False)[:300])
    return out


def no_report_finish(run_dir: Path) -> bool:
    """10-15：GIA 因为工作区没交报告（超时被关）而 FINISH（规则 G9_no_reports），而工作区其实做过动作——这种 run 当半截重跑。"""

    finish = False
    for transport in (run_dir / "requests").glob("request_*/transport.json"):
        try:
            finish = finish or (json.loads(transport.read_text(encoding="utf-8")).get("oracle") or {}).get("rule") == "G9_no_reports"
        except (OSError, ValueError):
            continue
    if not finish:
        return False
    try:
        links = json.loads((run_dir / "execution_links.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(isinstance(l, dict) and l.get("role") == "HI" for l in links)


def walker_stats(run_dir: Path) -> dict[str, Any]:
    """真正的教师（LLM）调用：各请求 transport.json 的 slots；按规则 / 分支计数。"""

    stats: dict[str, Any] = {"llm_calls": 0, "llm_seconds": 0.0, "rules": {}, "branches": {}, "slot_calls": {}, "llm_cost_usd": 0.0}
    for transport in sorted((run_dir / "requests").glob("request_*/transport.json")):
        try:
            meta = json.loads(transport.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for slot in meta.get("slots") or []:
            stats["llm_calls"] += 1
            stats["llm_seconds"] += float(slot.get("elapsed_s") or 0.0)
            stats["llm_cost_usd"] += float(slot.get("cost_usd") or 0.0)
            # 10-15：按 槽|后端|模型 计数（核对"没有任何槽走 CLI"：backend 必须是 endpoint）
            key = f"{str(slot.get('name') or '?').split('_')[0]}|{slot.get('backend') or '?'}|{slot.get('model') or '?'}"
            stats["slot_calls"][key] = stats["slot_calls"].get(key, 0) + 1
        oracle = meta.get("oracle") or {}
        if oracle.get("rule"):
            stats["rules"][oracle["rule"]] = stats["rules"].get(oracle["rule"], 0) + 1
        if oracle.get("branch"):
            stats["branches"][oracle["branch"]] = stats["branches"].get(oracle["branch"], 0) + 1
    stats["llm_seconds"] = round(stats["llm_seconds"], 1)
    stats["llm_cost_usd"] = round(stats["llm_cost_usd"], 4)
    return stats


OPENROUTER_API = "https://openrouter.ai/api/v1"


def use_openrouter_teacher(model: str) -> None:
    """10-14 晚：教师走 OpenRouter（OpenAI 兼容端点）。key 从本机加密保存的 OpenRouter 设置取，放进子进程的环境变量。"""

    from foren_searcher.openrouter_settings import load_api_key

    os.environ["FOREN_ORACLE_SLOT_ENDPOINT"] = OPENROUTER_API
    os.environ["FOREN_ORACLE_SLOT_MODEL"] = model
    if not os.environ.get("FOREN_TEACHER_API_KEY"):
        os.environ["FOREN_TEACHER_API_KEY"] = load_api_key()


def d1(args: argparse.Namespace) -> int:
    skip = ROOT / SFT / f"_skip_{args.round}"
    if skip.exists():
        # 10-15：批处理里排好的某一轮不想跑了（例如 wk15），放一个 experiments/sft_data/_skip_<轮次> 文件即可，删掉就恢复
        print(f"{args.round}：有 {skip}，这一轮跳过")
        return 0
    teacher_kind = getattr(args, "teacher", None)
    if teacher_kind is None and not os.environ.get("FOREN_ORACLE_SLOT") and not os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"):
        teacher_kind = "slots"                  # 10-15 对话 5a：默认按冻结文件的分槽表，不再默认 Codex
    if teacher_kind == "slots":
        os.environ["FOREN_ORACLE_SLOT"] = "slots"
        os.environ.pop("FOREN_ORACLE_SLOT_ENDPOINT", None)
    if getattr(args, "teacher_backup", None):
        os.environ["FOREN_TEACHER_BACKUP"] = args.teacher_backup
    if getattr(args, "teacher", None) == "claude":
        os.environ["FOREN_ORACLE_SLOT"] = "claude"
    if getattr(args, "teacher", None) == "openrouter":
        if not args.teacher_model:
            print("--teacher openrouter 需要 --teacher-model（例如 qwen/qwen3.5-122b-a10b）")
            return 2
        use_openrouter_teacher(args.teacher_model)
    if getattr(args, "arms", False):
        os.environ["FOREN_ORACLE_ARMS"] = "1"
    if not getattr(args, "no_inloop_sufficiency", False):
        os.environ["FOREN_INLOOP_SUFFICIENCY"] = "1"        # 10-16 r0+：充分性检查进循环（oracle_generator._after_report）
    env = env_setup()
    log_name = LOG_NAME if args.round == ROUND else f"_walker_{args.round}日志.txt"
    slot = (f"分槽教师 {teacher_model()}（冻结文件）" if slot_backend() == "slots"
            else f"端点槽 {os.environ.get('FOREN_ORACLE_SLOT_MODEL')} @ {os.environ['FOREN_ORACLE_SLOT_ENDPOINT']}"
            if os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT") else f"Claude CLI 槽 {teacher_model()}"
            if slot_backend() == "claude" else f"Codex 槽 {env['FOREN_ORACLE_SLOT_EFFORT']}")
    slot += f"；核验 {judge_label()}"
    if os.environ.get("FOREN_ORACLE_ARMS") == "1":
        slot += "，S5 四版对照"
    log = Log(log_name, f"walker {args.round}：{len(args.cases)} 个案例（程序走、{slot}）")
    print(f"正在运行：教师预检 → 每个案例一条 walker 轨迹（每条约 2–5 分钟）→ 核验（{judge_label()}）与导出 → 与 r0e / tp2 对照。")
    print(f"中途关窗口没关系，再双击会跳过已跑完的 run。详细输出在 {log_name}")
    if env.get("FOREN_ORACLE_SLOT", "").lower() != "mock" and not codex_preflight(log):
        return 1
    if not args.no_verify and judge_kind() == "api-batch" and not judge_preflight(log):
        return 1
    if not args.no_verify and judge_kind() == "claude-cli" and not claude_judge_preflight(log):
        return 1
    if checkpoint_path(args.round).is_file():
        log.say(f"[断点] 上次停在 {json.loads(checkpoint_path(args.round).read_text(encoding='utf-8')).get('stage')}，从这里续跑")
    try:                                              # 10-15 r0 冻结：规格 / 工具层 / 教师 / 核验器
        frozen_ok = freeze_preflight(log, args.round)
    except Exception as exc:  # noqa: BLE001          # 预检本身出错只记日志，不挡开跑（STRICT 时除外）
        log.say(f"[冻结] 预检出错：{type(exc).__name__}: {exc}")
        frozen_ok = os.environ.get("FOREN_FREEZE_STRICT") != "1"
    if not frozen_ok:
        return 1
    from foren_searcher.datagen.run_cases import run_stats, run_status

    logs = SFT / "batch_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (SFT / "run_manifests").mkdir(parents=True, exist_ok=True)
    manifest = SFT / "run_manifests" / f"{args.round}.jsonl"
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    stop = threading.Event()
    state = {"rate_limited": 0, "codex_limited": False}
    total = len(args.cases)
    # 10-15 对话 5：strong = Codex CLI（订阅）撞 usage limit 时自动换备用 strong（OpenRouter Sonnet）整条重跑；之后本批其余 run 直接用备用
    # （订阅额度按小时窗口恢复，下次双击时没跑完的 run 又从主教师开始）。--teacher-backup 已显式给了就不再自动换
    auto_backup = slot_backend() == "slots" and not os.environ.get("FOREN_TEACHER_BACKUP") and not getattr(args, "no_auto_backup", False)

    def launch(index: int, run_name: str, case_file: Path, overrides: list[str], backup: bool) -> tuple[int, Path]:
        batch_log = logs / f"{run_name}.log"
        batch_log.write_text("", encoding="utf-8")
        run_env = {**env, "FOREN_TEACHER_BACKUP": "strong"} if backup else env
        code = log.run(f"[{index}/{total}] {run_name}{'（备用 strong）' if backup else ''}（日志 {batch_log}）",
                       [PY, "scripts/walker_run.py", "one", str(case_file), run_name, "--round", args.round,
                        *(["--override", *overrides] if overrides else [])], env=run_env, log_file=batch_log)
        return code, batch_log

    def tail_of(batch_log: Path, code: int) -> str:
        try:
            return batch_log.read_text(encoding="utf-8", errors="replace")[-3000:] if code else ""
        except OSError:
            return ""

    def one(index: int, case: str) -> None:
        if stop.is_set():
            return
        base, overrides = split_variant(case)
        case_file = CASES_DIR / f"{base}.json"
        run_name = f"{case}_{args.round}_1"
        run_dir = RUNS / run_name
        status = run_status(run_dir)
        if status == "done" and (slot_failures(run_dir) or no_report_finish(run_dir)):
            status = "incomplete"          # 槽失败却标了 done（wk10 1KkAwCyj7nI）/ 工作区超时没交报告（wk13）：改名重跑
        if status == "done":
            log.say(f"[{index}/{total}] {run_name} 已完成，跳过")
            return
        if status == "incomplete":
            old = run_dir.with_name(f"{run_name}.failed-{time.strftime('%Y%m%d-%H%M%S')}")
            run_dir.rename(old)
            log.say(f"[{index}/{total}] {run_name} 是半截，改名为 {old.name} 后重跑")
        started = time.monotonic()
        backup = auto_backup and state["codex_limited"]
        code, batch_log = launch(index, run_name, case_file, overrides, backup)
        tail = tail_of(batch_log, code)
        if code and auto_backup and not backup and CODEX_LIMIT_MARK in tail:
            with lock:
                state["codex_limited"] = True
            old = run_dir.with_name(f"{run_name}.failed-{time.strftime('%Y%m%d-%H%M%S')}")
            if run_dir.exists():
                run_dir.rename(old)
            log.say(f"      {run_name}：Codex 到了订阅用量上限（usage limit，退避用尽）→ 整条 run 换备用 strong 重跑"
                    f"（--teacher-backup strong；半截改名为 {old.name}）；本批其余 run 也用备用")
            backup = True
            code, batch_log = launch(index, run_name, case_file, overrides, backup)
            tail = tail_of(batch_log, code)
        wall = round(time.monotonic() - started, 1)
        try:
            walker_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")).get("walker") or {}
        except (OSError, ValueError):
            walker_cfg = {}
        row = {"run": run_name, "case_id": case, "round": args.round, "teacher": "oracle_walker", "returncode": code,
               "status": run_status(run_dir), "wall_seconds": wall, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "teacher_model": walker_cfg.get("teacher_model") or teacher_model(),
               "slot_models": walker_cfg.get("slot_models") or {k: v["model"] for k, v in (slot_teachers() or {}).items()} or None,
               # 10-15 对话 5：本条 run 用的是主教师还是备用 strong（Codex usage limit 后自动换的记 backup）；冻结口径下的逐槽角色另记
               "teacher_role": "backup" if backup else "primary", "teacher_backup": "strong" if backup else None,
               "teacher_role_slots": walker_cfg.get("teacher_role"),
               **run_stats(run_dir), **walker_stats(run_dir)}
        with lock:
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.say(f"      {run_name}：{row['status']}；{wall:.0f}s；决策 {row['teacher_calls']} 次，其中教师（LLM）调用 {row['llm_calls']} 次"
                f"{'；备用 strong' if backup else ''}")
        limited = any(x in tail for x in ("429", "Too Many Requests", "rate limit", "usage limit", "insufficient credits", "402"))
        with lock:
            state["rate_limited"] = state["rate_limited"] + 1 if limited else 0
            if state["rate_limited"] >= 2 and not stop.is_set():
                stop.set()
                log.say("连续两个 run 都因教师限额 / 余额（429 / 402）失败，停在这里：恢复后再双击，已完成的 run 会跳过。")
                log.failed.append("教师限额")

    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    if jobs == 1:
        for index, case in enumerate(args.cases, 1):
            one(index, case)
    else:
        log.say(f"并行 {jobs} 个 run")
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda item: one(*item), list(enumerate(args.cases, 1))))
    backup_share_warning(args.round, args.cases, log)
    inloop_summary(args.round, log)
    if stop.is_set():
        return 1
    if os.environ.get("FOREN_ORACLE_ARMS") == "1":
        return arms_round(args, log, env)
    if not args.no_verify:
        start = log.path.stat().st_size
        code = verify_round(args.round, log, env)
        if code and any(w in _log_tail_since(log.path, start) for w in CLAUDE_LIMIT_WORDS):
            write_checkpoint(args.round, "核验", _log_tail_since(log.path, start), log)
            return 1
        if not getattr(args, "no_revise", False):
            start = log.path.stat().st_size
            revise_round(args.round, log, env, mode=getattr(args, "revise_mode", "support"))
            if any(w in _log_tail_since(log.path, start) for w in CLAUDE_LIMIT_WORDS):
                write_checkpoint(args.round, "重写后复核", _log_tail_since(log.path, start), log)
                return 1
        if not getattr(args, "no_status_rewrite", False):
            start = log.path.stat().st_size
            status_rewrite_round(args.round, log, env)
            if any(w in _log_tail_since(log.path, start) for w in CLAUDE_LIMIT_WORDS):
                write_checkpoint(args.round, "status 改写后复核", _log_tail_since(log.path, start), log)
                return 1
        if checkpoint_path(args.round).is_file():
            checkpoint_path(args.round).rename(checkpoint_path(args.round).with_suffix(f".done-{time.strftime('%Y%m%d-%H%M%S')}.json"))
        if getattr(args, "second_judge", None):
            second_verify(args.round, args.second_judge, log, env, sample=getattr(args, "second_judge_sample", 20))
    log.run("[对照] walker vs r0e（同案例）与 tp2（qvhighlights）",
            [PY, "scripts/walker_compare.py", "--round", args.round, "--cases", *args.cases, *_compare_extra(args.round)], env=env)
    log.run("[覆盖] 各工具 / 参数变体用了几次（对照该在哪里用）", [PY, "scripts/walker_coverage.py", "--round", args.round], env=env)
    if log.failed:
        log.say(f"有 {len(log.failed)} 步没成功：{'；'.join(log.failed)}。看 {log_name}。")
    out = "walker_d1_compare" if args.round == ROUND else f"walker_{args.round}_compare"
    log.say(f"结束。对照表 {SFT}\\{out}.md；日志 {log_name}")
    return 0


BACKUP_SHARE_WARN = 0.20                 # = pack_r0.BACKUP_SHARE_WARN


def backup_share_warning(round_tag: str, cases: list[str], log: Log) -> float:
    """10-15 对话 5：本轮（这批案例）的 run 里用备用 strong 的占比（manifest 每个 run 取最后一行）；> 20% 报警，不挡后续步骤。"""

    path = SFT / "run_manifests" / f"{round_tag}.jsonl"
    wanted = {f"{c}_{round_tag}_1" for c in cases}
    latest: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("run") in wanted:
                latest[row["run"]] = row
    if not latest:
        return 0.0
    backup = sorted(r for r, row in latest.items() if row.get("teacher_role") == "backup")
    share = len(backup) / len(latest)
    if share > BACKUP_SHARE_WARN:
        log.say(f"[报警] 备用 strong（OpenRouter Sonnet）跑的 run 占 {len(backup)}/{len(latest)} = {share:.0%} > "
                f"{BACKUP_SHARE_WARN:.0%}：{', '.join(backup)[:600]}")
    elif backup:
        log.say(f"[教师] 备用 strong 跑的 run {len(backup)}/{len(latest)} = {share:.0%}（≤ {BACKUP_SHARE_WARN:.0%}）")
    return share


def merge_inloop_caches(round_tag: str, log: Log) -> dict[str, int]:
    """10-16 r0+：循环内充分性检查的判官缓存（各 run 目录 inloop_verify_cache.jsonl，同一提示、同一 payload）并进 claude-cli 主缓存，
    跑完后的全量核验对没改过的报告直接命中。只追加主缓存里没有的 key；只在 claude-cli 核验时做（同一判官）。"""

    stats = {"files": 0, "rows": 0, "added": 0}
    if judge_kind() != "claude-cli":
        return stats
    model = _frozen_verifier().get("model") or "claude-sonnet-5-5"
    target = ROOT / SFT / f"verify_cache.claude-cli-{model}.jsonl"
    have: set[str] = set()
    if target.is_file():
        for line in target.read_text(encoding="utf-8").splitlines():
            m = re.match(r'\{"key": "([0-9a-f]{64})"', line)
            if m:
                have.add(m.group(1))
    new_lines: list[str] = []
    for path in sorted(RUNS.glob(f"*_{round_tag}_*/inloop_verify_cache.jsonl")):
        if ".failed-" in str(path):
            continue
        stats["files"] += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                key = json.loads(line)["key"]
            except (ValueError, KeyError, TypeError):
                continue
            stats["rows"] += 1
            if key not in have:
                have.add(key)
                new_lines.append(line)
    if new_lines:
        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(new_lines) + "\n")
    stats["added"] = len(new_lines)
    if stats["files"]:
        log.say(f"[循环内核验] 判官缓存并进 {target.name}：{stats['files']} 个 run、{stats['rows']} 条、新增 {stats['added']} 条")
    return stats


def inloop_summary(round_tag: str, log: Log) -> dict[str, Any]:
    """10-16 r0+：本轮循环内充分性检查的汇总（各 run 的 inloop_sufficiency.json）→ walker_<轮次>_inloop.json。"""

    rows = []
    for path in sorted(RUNS.glob(f"*_{round_tag}_*/inloop_sufficiency.json")):
        if ".failed-" in str(path):
            continue
        for c in json.loads(path.read_text(encoding="utf-8")).get("checks") or []:
            rows.append({"run": path.parent.name, **{k: c.get(k) for k in ("request", "workspace", "family", "status_in", "status_out",
                                                                         "flagged", "rewritten", "followup_appended", "error",
                                                                         "limit_hit", "judge_new_calls", "judge_cache_hits",
                                                                         "scope", "key")}})
    checked = [r for r in rows if r["status_in"] == "COMPLETED"]
    summary = {"round": round_tag, "reports_seen": len(rows), "checked": len(checked),
               "rewritten": sum(1 for r in rows if r.get("rewritten")),
               "rewritten_round2": sum(1 for r in rows if r.get("rewritten") and str(r["workspace"]).startswith("W2-")),
               "followup_appended": sum(1 for r in rows if r.get("followup_appended")),
               "errors": sum(1 for r in rows if r.get("error")), "limit_hits": sum(1 for r in rows if r.get("limit_hit")),
               "judge_new_calls": sum(int(r.get("judge_new_calls") or 0) for r in rows),
               "judge_cache_hits": sum(int(r.get("judge_cache_hits") or 0) for r in rows), "rows": rows}
    (ROOT / SFT / f"walker_{round_tag}_inloop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        log.say(f"[循环内核验] 报告 {summary['reports_seen']} 份：核 {summary['checked']}（COMPLETED），改 INCOMPLETE {summary['rewritten']}"
                f"（第 2 轮 {summary['rewritten_round2']}），补跟进 {summary['followup_appended']}；出错 {summary['errors']}、限额 "
                f"{summary['limit_hits']}；判官新调用 {summary['judge_new_calls']}、命中缓存 {summary['judge_cache_hits']}")
    return summary


def verify_round(round_tag: str, log: Log, env: dict[str, str]) -> int:
    """核验 + 导出。本机默认 Claude CLI（Sonnet）；服务器没有 Claude CLI 时 FOREN_WALKER_JUDGE=local，用 8001 端口的
    本地核验器（FOREN_JUDGE_ENDPOINT / FOREN_WALKER_JUDGE_MODEL，与 server/14_run_cases.sh 的 --judge local 相同）。"""

    merge_inloop_caches(round_tag, log)
    verify_env = dict(env)
    judge = judge_kind()
    judge_args = ["--judge", judge]
    if judge in ("api", "api-batch"):
        # 10-15 对话 5a：冻结的核验器 = Sonnet 走 API（与 claude-cli 相同的提示 / 无工具 / base64 图）；按量计费，批处理不停下来问
        frozen = _frozen_verifier()
        endpoint = os.environ.get("FOREN_JUDGE_API_ENDPOINT") or frozen.get("endpoint")
        if judge == "api-batch" and endpoint and "api.anthropic.com" not in endpoint:
            endpoint = "https://api.anthropic.com/v1"          # 批量只有 Anthropic 直连
        model = os.environ.get("FOREN_WALKER_JUDGE_MODEL") or os.environ.get("FOREN_JUDGE_API_MODEL") or frozen.get("model")
        judge_args += [*(["--endpoint", endpoint] if endpoint else []), *(["--model", model] if model else []), "--yes-spend"]
        verify_env.setdefault("FOREN_JUDGE_API_EFFORT", str((frozen.get("settings") or {}).get("reasoning_effort") or "medium"))
        verify_env.setdefault("FOREN_JUDGE_CONCURRENCY", str(frozen.get("concurrency") or 6))
    elif judge == "local":
        if os.environ.get("FOREN_JUDGE_ENDPOINT"):
            judge_args += ["--endpoint", os.environ["FOREN_JUDGE_ENDPOINT"]]
        if os.environ.get("FOREN_WALKER_JUDGE_MODEL"):
            judge_args += ["--model", os.environ["FOREN_WALKER_JUDGE_MODEL"]]
        verify_env.setdefault("FOREN_JUDGE_CONCURRENCY", "4")
    else:
        verify_env.setdefault("FOREN_JUDGE_CONCURRENCY", "3")
    label = ("Sonnet 核验（verifier 与 r0e 相同）" if judge == "claude-cli" else f"Sonnet 核验（API，{judge_label()}）"
             if judge == "api" else f"Sonnet 核验（API 批量 + 缓存，{judge_label()}）" if judge == "api-batch"
             else f"核验（--judge {judge}）")
    return log.run(f"[核验] {label} + 导出",
                   [PY, "scripts/complete_verify_local.py", *judge_args, "--runs-glob", round_tag, "--round", round_tag,
                    "--hi-spec", specs_for(round_tag)[0], "--gia-spec", specs_for(round_tag)[1], "--skip", "4", "6"], env=verify_env)


def _latest_units(round_tag: str) -> Path | None:
    files = [p for p in (ROOT / SFT).glob(f"verdicts_units_{round_tag}_judged.*.jsonl")]   # 10-14：不限 claude-cli 标签
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _answer_units(path: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path or not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("unit") == "answer":
            out[str((row.get("subject") or {}).get("run"))] = row
    return out


# 10-14：带 GT 的核验意见（label / 实体 / 日期 / 区间 / 状态 / 数值核对，以及"漏了应有的事实""要求没覆盖"）——
# "只按证据重写"（support）时一律不给教师；只留"这句话所引的证据不支持 / 引错 / 说过头"这类对照证据就能看出的问题
_GT_CHECKS = {"label_expected", "entity_in_evidence", "numeric_in_evidence", "date_in_range", "interval_iou", "status_expected"}
_GT_TYPES = {"omitted_required_fact", "premature_finish"}


def support_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for e in errors:
        extra = e.get("extra") or {}
        if e.get("severity") == "minor" or e.get("type") in _GT_TYPES or extra.get("check") in _GT_CHECKS or extra.get("reference"):
            continue
        out.append(e)
    return out


def revise_round(round_tag: str, log: Log, env: dict[str, str], mode: str = "full") -> int:
    """核验后重写答案（10-13）：answer 单元判 negative 的 run，把核验器指出的错误反馈给教师重写一次答案，写回 run 目录
    （原文件留 .orig），再整轮核验 + 导出一次（没变的命中缓存）。重写前的核验文件另存 *.pre_revise。"""

    units_path = _latest_units(round_tag)
    before = _answer_units(units_path)
    if mode == "support":
        # 不看整体判定（判定本身用了 GT）：只要有"证据不支持"类的非 minor 问题就重写
        targets = {run: row for run, row in before.items() if support_errors(row.get("errors") or [])
                   and not (RUNS / run / "execution_links.json.orig").exists()}
    else:
        targets = {run: row for run, row in before.items() if row.get("disposition") == "negative"
                   and not (RUNS / run / "execution_links.json.orig").exists()}          # 每条只重写一次
    if not targets:
        log.say("[重写] 没有需要重写的答案")
        return 0
    for path in (ROOT / SFT).glob(f"verdicts*_{round_tag}_judged.*.jsonl"):
        backup = path.with_name(path.name + ".pre_revise")
        if not backup.exists():
            shutil.copy2(path, backup)
    oracle_dir = str((ROOT / SFT).resolve())
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    from oracle_generator import revise_answer  # type: ignore

    records = []
    for run, row in sorted(targets.items()):
        errors = (support_errors(row.get("errors") or []) if mode == "support"
                  else [e for e in row.get("errors") or [] if e.get("severity") != "minor"] or list(row.get("errors") or []))
        try:
            rec = revise_answer(RUNS / run, errors)
        except Exception as exc:  # noqa: BLE001 - 一条失败不影响其余
            log.say(f"[重写] {run} 失败：{exc}")
            log.failed.append(f"重写 {run}")
            continue
        records.append(rec or {"run": run, "changed": False})
        log.say(f"[重写] {run}：{'已改写' if rec and rec.get('changed') else '未改写'}（{len(errors)} 处问题）")
    verify_round(round_tag, log, env)
    after = _answer_units(_latest_units(round_tag))
    summary = {"round": round_tag, "mode": mode, "revised": [r.get("run") for r in records if r.get("changed")],
               "before": {run: row.get("disposition") for run, row in before.items()},
               "after": {run: row.get("disposition") for run, row in after.items()}}
    summary["answer_positive_before"] = sum(1 for v in summary["before"].values() if v == "positive")
    summary["answer_positive_after"] = sum(1 for v in summary["after"].values() if v == "positive")
    (ROOT / SFT / f"walker_{round_tag}_revise.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.say(f"[重写] 答案核验 positive：重写前 {summary['answer_positive_before']}/{len(before)} → 重写后 "
            f"{summary['answer_positive_after']}/{len(after)}；明细 {SFT}\\walker_{round_tag}_revise.json")
    return 0


def _latest_verdicts(round_tag: str) -> Path | None:
    files = [p for p in (ROOT / SFT).glob(f"verdicts_{round_tag}_judged.*.jsonl")]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def status_rewrite_round(round_tag: str, log: Log, env: dict[str, str]) -> int:
    """10-16 r0g 收尾（负责人定"核验器为准"）：充分性层判 completed_without_finishing（major / critical）的报告，status 改
    INCOMPLETE、status_reason 取充分性判定的说明（datagen.status_rewrite），之后 GIA 的状态一并重渲染；改了就整轮再核验一次
    （没变的单元命中缓存）。不调教师、不改核验器。改写前的核验文件另存 *.pre_status_rewrite。"""

    from foren_searcher.datagen.status_rewrite import rewrite_round

    path = _latest_verdicts(round_tag)
    if path is None:
        log.say("[status 改写] 没有核验文件，跳过")
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for p in (ROOT / SFT).glob(f"verdicts*_{round_tag}_judged.*.jsonl"):
        backup = p.with_name(p.name + ".pre_status_rewrite")
        if not backup.exists():
            shutil.copy2(p, backup)
    done = rewrite_round(rows, RUNS, source=path.name)
    summary_path = ROOT / SFT / f"walker_{round_tag}_status_rewrite.json"
    old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
    summary_path.write_text(json.dumps(old + done, ensure_ascii=False, indent=2), encoding="utf-8")
    finish = sum(len(d.get("finish_after") or []) for d in done)
    log.say(f"[status 改写] {len(done)} 份报告 COMPLETED → INCOMPLETE（{path.name}）；之后 GIA 仍 FINISH 且报告带可修缺口的 FINISH 行 "
            f"{finish} 条（导出 held finish_after_status_rewrite）；明细 {summary_path.name}")
    # 上次改了、但之后的复核没跑成（Claude CLI 中途失败）：status_rewrite.json 比核验文件新也要复核
    stale = [p for p in RUNS.glob(f"*_{round_tag}_*/status_rewrite.json") if ".failed-" not in str(p)
             and p.stat().st_mtime > path.stat().st_mtime]
    if not done and not stale:
        return 0
    return verify_round(round_tag, log, env)


ARMS = {"b": "原方法：GT 词提示重写 + 核验意见全给重写", "g": "证据门控提示 + 只按证据重写",
        "n": "N 份稿按 GT 择优，不重写"}


def arms_round(args: argparse.Namespace, log: Log, env: dict[str, str]) -> int:
    """10-14 S5 对照：轨迹（含 d0 首稿答案）在 <round>；各版拷成 <round>b / g / n，分别核验；b 按原方法重写
    （核验意见全给），g 只按证据重写；最后 walker_arms_compare.py 出对照表。"""

    oracle_dir = str((ROOT / SFT).resolve())
    if oracle_dir not in sys.path:
        sys.path.insert(0, oracle_dir)
    from oracle_generator import make_arm_run  # type: ignore

    made = 0
    for case in args.cases:
        run_dir = RUNS / f"{case}_{args.round}_1"
        if not (run_dir / "result.json").is_file() or slot_failures(run_dir):
            continue
        for arm in ARMS:
            try:
                rec = make_arm_run(run_dir, arm, args.round + arm, args.round)
            except Exception as exc:  # noqa: BLE001
                log.say(f"[对照] {run_dir.name} → {arm} 失败：{exc}")
                log.failed.append(f"拷 {arm} {run_dir.name}")
                continue
            made += bool(rec and rec.get("created"))
    log.say(f"[对照] 新拷出 {made} 条各版 run（{', '.join(args.round + a for a in ARMS)}）")
    if args.no_verify:
        return 0
    for tag in [args.round, *(args.round + a for a in ARMS)]:
        verify_round(tag, log, env)
    if not getattr(args, "no_revise", False):
        revise_round(args.round + "b", log, env, mode="full")
        revise_round(args.round + "g", log, env, mode="support")
    log.run("[对照] S5 各版 GT 用法对照", [PY, "scripts/walker_arms_compare.py", "--round", args.round, "--cases", *args.cases], env=env)
    if log.failed:
        log.say(f"有 {len(log.failed)} 步没成功：{'；'.join(log.failed)}。")
    log.say(f"结束。对照表 {SFT}\\walker_{args.round}_arms.md")
    return 0


def second_judge_sample(round_tag: str, sample: int) -> list[str]:
    """第二核验器只核 sample 个 run（10-15 r0 成本压缩，默认 20）：本轮 run 按 sha256(run 名) 排序取前 sample 个，可复现、
    不偏向哪个族。sample ≤ 0 = 全部。"""

    import hashlib

    runs = sorted(p.name for p in RUNS.glob(f"*_{round_tag}_*") if p.is_dir() and ".failed-" not in p.name
                  and (p / "result.json").is_file())
    if sample <= 0 or sample >= len(runs):
        return runs
    return sorted(sorted(runs, key=lambda r: hashlib.sha256(r.encode("utf-8")).hexdigest())[:sample])


def second_verify(round_tag: str, model: str, log: Log, env: dict[str, str], sample: int = 20) -> int:
    """10-14 晚：用第二个核验模型（OpenRouter 上的 Qwen，走与服务器本地核验器相同的 OpenAI 兼容路径 --judge local）
    把同一轮 run 再核一遍，产物标签 <轮次>qv，供 walker_judge_agreement.py 与 Sonnet 的判定逐单元对照。
    10-15 r0 成本压缩：默认不跑（--second-judge 才跑），且只抽 sample 个 run（--second-judge-sample，默认 20）。"""

    from foren_searcher.openrouter_settings import load_api_key

    verify_env = dict(env)
    verify_env["FOREN_JUDGE_API_KEY"] = os.environ.get("FOREN_TEACHER_API_KEY") or load_api_key()
    verify_env.setdefault("FOREN_JUDGE_CONCURRENCY", "6")
    runs = second_judge_sample(round_tag, sample)
    if not runs:
        log.say("[核验2] 本轮没有跑成的 run，跳过")
        return 0
    log.say(f"[核验2] 抽 {len(runs)} 个 run（--second-judge-sample {sample}）：{', '.join(runs)[:600]}")
    code = log.run(f"[核验2] {model}（OpenRouter，--judge local 路径）",
                   [PY, "scripts/complete_verify_local.py", "--judge", "local", "--endpoint", OPENROUTER_API, "--model", model,
                    "--runs", *runs, "--round", round_tag + "qv",
                    "--hi-spec", specs_for(round_tag)[0], "--gia-spec", specs_for(round_tag)[1], "--skip", "4", "5", "6"],
                   env=verify_env)
    log.run("[核验2] 与 Sonnet 判定逐单元对照", [PY, "scripts/walker_judge_agreement.py", "--round", round_tag], env=env)
    return code


def revise_job(args: argparse.Namespace) -> int:
    env = env_setup()
    log = Log(f"_walker_{args.round}_重写日志.txt", f"walker {args.round}：答案核验后重写")
    revise_round(args.round, log, env)
    log.run("[对照] 重写后对照表", [PY, "scripts/walker_compare.py", "--round", args.round, "--cases", *args.cases,
                                  *_compare_extra(args.round)], env=env)
    return 1 if log.failed else 0


def _compare_extra(round_tag: str) -> list[str]:
    """wk1 写 walker_d1_compare（与以前同名）；之后的轮次写 walker_<轮次>_compare，并把 wk1 也列进去对照。"""

    if round_tag == ROUND:
        return []
    earlier = [f"wk{i}" for i in range(1, 10) if f"wk{i}" != round_tag and (RUNS.parent / "run_manifests" / f"wk{i}.jsonl").exists()
               and int(f"wk{i}"[2:]) < int(round_tag[2:] or 0)] if round_tag.startswith("wk") and round_tag[2:].isdigit() else [ROUND]
    return ["--also", "tp2", *earlier, "--out", str((ROOT / SFT / f"walker_{round_tag}_compare"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    p = sub.add_parser("d1")
    p.add_argument("--cases", nargs="+", default=D1_CASES)
    p.add_argument("--round", default=ROUND)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-revise", action="store_true", help="核验后不重写判错的答案")
    p.add_argument("--d2", action="store_true", help="再加上 D2 案卷（A 族找原片 / C 族核查声明，清单 walker_d2_cases.json）")
    p.add_argument("--variants", action="store_true", help="再加上环境变体（V9_CASES：max_evidence_cards=6）")
    p.add_argument("--skip-b", action="store_true", help="不跑 B 族 9 个（只跑 --d2 / --variants 加进来的案例）")
    p.add_argument("--only", nargs="*", default=None, help="只跑名字以这些前缀开头的案例（例如 fmnv_a_ groundlie360_v）")
    p.add_argument("--case-list", default=None, help="案例清单文件（experiments/sft_data 下，walker_d2_cases.py --list 写的），加到 --cases 后面")
    p.add_argument("--teacher", choices=["slots", "codex", "claude", "openrouter"], default=None,
                   help="语言槽教师：slots = 冻结文件的分槽表（默认，10-15 起）；claude = 本机 Claude CLI（Sonnet）；"
                        "openrouter = OpenRouter 上的单个模型（配 --teacher-model）；codex = 本机 Codex CLI")
    p.add_argument("--teacher-backup", default=None, help="分槽模式：qwen / strong / all——该后端整条 run 换成冻结文件里的备用模型")
    p.add_argument("--no-auto-backup", action="store_true",
                   help="Codex（strong）撞 usage limit 时不自动换备用 strong 重跑（默认自动换，manifest 记 teacher_role=backup）")
    p.add_argument("--teacher-model", default=None, help="--teacher openrouter 的模型，例如 qwen/qwen3.5-122b-a10b")
    p.add_argument("--jobs", type=int, default=1, help="同时跑几个 run（端点教师可以并行；strong=Codex 时 Codex 调用跨进程串行，建议 2）")
    p.add_argument("--revise-mode", choices=["support", "full"], default="support",
                   help="核验后重写答案：support = 只把证据不支持类意见给教师（默认，不用 GT）；full = 全部意见（含 GT 类，wk11/12 证明会教删正确内容）")
    p.add_argument("--second-judge", default=None, help="再用这个 OpenRouter 模型核一遍（与 Sonnet 对照核验一致性），例如 qwen/qwen3.5-122b-a10b；默认不跑")
    p.add_argument("--second-judge-sample", type=int, default=20, help="第二核验器只抽这么多个 run（按 run 名哈希定序，默认 20；0 = 全部）")
    p.add_argument("--arms", action="store_true", help="S5 对照：d0 首稿 / b 原方法 / g 证据门控 / n 择优 四版分别核验")
    p.add_argument("--no-status-rewrite", action="store_true",
                   help="核验后不把充分性层判 completed_without_finishing 的报告改为 INCOMPLETE（10-16 起默认改）")
    p.add_argument("--no-inloop-sufficiency", action="store_true",
                   help="不在循环内做充分性检查（10-16 r0+ 起默认做：S4 之后、GIA 之前核一次报告，判没答完就改 INCOMPLETE 并写可修缺口）")
    sr = sub.add_parser("status-rewrite", help="对已核验的一轮：充分性层判 completed_without_finishing 的报告改 INCOMPLETE，再核验")
    sr.add_argument("--round", default=ROUND)
    rv = sub.add_parser("revise", help="对已核验的一轮：判错的答案按核验意见重写一次，再核验 + 导出 + 对照")
    rv.add_argument("--round", default=ROUND)
    rv.add_argument("--cases", nargs="+", default=D1_CASES)
    one = sub.add_parser("one")
    one.add_argument("case_file")
    one.add_argument("run_name")
    one.add_argument("--round", default=ROUND)
    one.add_argument("--override", nargs="*", default=None, metavar="键=值")
    fz = sub.add_parser("freeze", help="r0 冻结预检：规格 / 工具层 / 核验器 hash 与 specs\\*_freeze.json 比对；--write 写回当前值")
    fz.add_argument("--round", default="wk16")
    fz.add_argument("--write", action="store_true", help="把当前工具层 hash 与核验器代码 / 传输层 hash 写回冻结文件（工具层定稿后用）")
    fz.add_argument("--teacher-backup", default=None, help="按备用教师核对（qwen / strong / all）")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--round", default=ROUND)
    cmp_.add_argument("--cases", nargs="+", default=D1_CASES)
    args = parser.parse_args()
    if args.job == "one":
        env_setup()
        return run_one_case(args.case_file, args.run_name, args.round, args.override)
    if args.job == "revise":
        if not os.environ.get("FOREN_ORACLE_SLOT") and not os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"):
            os.environ["FOREN_ORACLE_SLOT"] = "slots"      # 10-15：核验后重写走 S5 → strong 后端，与 d1 相同
        return revise_job(args)
    if args.job == "status-rewrite":
        log = Log(f"_walker_{args.round}_status改写日志.txt", f"walker {args.round}：充分性判定后 status 改写")
        return status_rewrite_round(args.round, log, env_setup()) or (1 if log.failed else 0)
    if args.job == "freeze":
        if not os.environ.get("FOREN_ORACLE_SLOT") and not os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT"):
            os.environ["FOREN_ORACLE_SLOT"] = "slots"      # 10-15：预检按默认教师（分槽表）核对，与 d1 相同
        if getattr(args, "teacher_backup", None):
            os.environ["FOREN_TEACHER_BACKUP"] = args.teacher_backup
        return freeze_job(args)
    if args.job == "compare":
        env = env_setup()
        return subprocess.call([PY, "scripts/walker_compare.py", "--round", args.round, "--cases", *args.cases,
                                *_compare_extra(args.round)], cwd=ROOT, env=env)
    if getattr(args, "skip_b", False):
        args.cases = [c for c in args.cases if c not in D1_CASES]
    if getattr(args, "d2", False):
        listing = ROOT / SFT / "walker_d2_cases.json"
        if not listing.is_file():
            print(f"没有 {listing}：先跑 python scripts\\walker_d2_cases.py 生成 D2 案卷。")
            return 1
        args.cases = list(args.cases) + [c for c in json.loads(listing.read_text(encoding="utf-8")) if c not in args.cases]
    if getattr(args, "case_list", None):
        listing = ROOT / SFT / args.case_list
        if not listing.is_file():
            print(f"没有 {listing}")
            return 1
        args.cases = list(args.cases) + [c for c in json.loads(listing.read_text(encoding="utf-8")) if c not in args.cases]
    if getattr(args, "variants", False):
        args.cases = list(args.cases) + [c for c in V9_CASES if c not in args.cases]
    if getattr(args, "only", None):
        args.cases = [c for c in args.cases if any(c.startswith(x) for x in args.only)]
    return d1(args)


if __name__ == "__main__":
    raise SystemExit(main())
