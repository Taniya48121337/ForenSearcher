"""r0 成本压缩（10-15，对话 5b）离线测试：strong 槽走 Codex CLI、核验器 Anthropic 直连 + 批量 + 缓存、冻结与打包核对。

不连外网、不调真的 Codex / Anthropic：Codex 用 tests 里写出的假 codex 脚本（按环境变量模拟 429 / usage limit），
Anthropic 用 127.0.0.1 上的假 Messages / Batches 服务。
- 分槽分发：strong type=codex → codex exec（-m / effort / --image / 图注进提示 / 只读），S0–S3 仍走端点；槽记录字段；
  429 / usage limit 按 30、60、120… 退避、总等待上限、用尽报 usage limit 并让 run 停下；跨进程串行锁；
  备用 strong（OpenRouter Sonnet 端点）不继承 codex 字段；未知后端类型报错；
- walker_run：codex 后端预检（成功 / usage limit）、核验 judge = api-batch、第二核验器抽样可复现；
- api_judge：请求体缓存断点位置（system + 共享前缀，问题在最后）、usage → 费用、直连 key 不拿 OpenRouter 的 sk-or-、
  Batches 三个接口；complete_verify_local 的 api-batch 步骤走批量编排器；
- 冻结文件：核验器 code_hash 不变、传输层 hash = 当前 api_judge.py；pack_r0 逐槽核对教师。
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments/sft_data", "scripts"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

FREEZE = ROOT / "specs" / "r0_freeze.json"

FAKE_CODEX = r'''#!/usr/bin/env python3
import json, os, pathlib, sys, time
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("codex-cli 9.9.9-fake"); sys.exit(0)
if args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT"); sys.exit(0)
state = pathlib.Path(os.environ["FAKE_CODEX_STATE"])
n = int(state.read_text()) if state.exists() else 0
state.write_text(str(n + 1))
prompt = sys.stdin.read()
with open(os.environ["FAKE_CODEX_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"n": n, "args": args, "prompt": prompt, "t0": time.time()}) + "\n")
time.sleep(float(os.environ.get("FAKE_CODEX_SLEEP", "0")))
with open(os.environ["FAKE_CODEX_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"n": n, "t1": time.time()}) + "\n")
if n < int(os.environ.get("FAKE_CODEX_FAIL429", "0")):
    print(json.dumps({"type": "error", "message": "unexpected status 429 Too Many Requests: You've hit your usage limit."}))
    print(json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again in 3 minutes."}}))
    sys.exit(1)
model = args[args.index("-m") + 1]
print(json.dumps({"type": "thread.started", "thread_id": "t"}))
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1200, "cached_input_tokens": 800, "output_tokens": 50}}))
print("model: " + model, file=sys.stderr)
pathlib.Path(args[args.index("-o") + 1]).write_text(os.environ.get("FAKE_CODEX_REPLY", '{"answer": "ok [E3]", "report": "r"}'))
'''


# ---------------------------------------------------------------------------- 假端点（Qwen 槽 / Anthropic）

class FakeServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.batches: dict[str, dict[str, Any]] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, status: int, payload: Any, ctype: str = "application/json") -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if "/messages/batches/" in self.path:
                    bid = self.path.rsplit("/", 2)[-2] if self.path.endswith("/results") else self.path.rsplit("/", 1)[-1]
                    batch = outer.batches[bid]
                    if self.path.endswith("/results"):
                        lines = [json.dumps({"custom_id": r["custom_id"], "result": {"type": "succeeded",
                                                                                     "message": outer.message(r["params"])}})
                                 for r in batch["requests"]]
                        self._send(200, "\n".join(lines).encode(), "application/x-jsonl")
                        return
                    self._send(200, {"id": bid, "processing_status": "ended", "results_url": None,
                                     "request_counts": {"succeeded": len(batch["requests"])}})
                    return
                self._send(200, {"data": [{"id": "fake-qwen"}], "id": "claude-sonnet-5-5"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append((self.path, body))
                if self.path.endswith("/messages/batches"):
                    bid = f"msgbatch_{len(outer.batches) + 1}"
                    outer.batches[bid] = {"requests": body["requests"]}
                    self._send(200, {"id": bid, "processing_status": "in_progress"})
                elif self.path.endswith("/messages"):
                    self._send(200, outer.message(body))
                else:
                    self._send(200, {"model": body.get("model"), "choices": [{"message": {"content": json.dumps(
                        {"rationale": "r", "query": "q", "decision": "FINISH"})}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    @staticmethod
    def message(body: dict[str, Any]) -> dict[str, Any]:
        cached = any("cache_control" in b for b in body["messages"][0]["content"])
        return {"model": body["model"], "type": "message", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps({"ok": True})}],
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 1000 if cached else 0,
                          "cache_creation_input_tokens": 0, "output_tokens": 20}}

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture()
def fake_codex(tmp_path, monkeypatch):
    path = tmp_path / "bin" / "codex"
    path.parent.mkdir()
    path.write_text(FAKE_CODEX, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    if os.name == "nt":             # Windows 起不了 shebang 脚本（WinError 193）：同一份 Python 桩，外面套 .cmd
        script = path.with_suffix(".py")
        path.rename(script)
        path = path.with_suffix(".cmd")
        path.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("FOREN_CODEX_BIN", str(path))
    monkeypatch.setenv("FAKE_CODEX_STATE", str(tmp_path / "codex.state"))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(tmp_path / "codex.log"))
    monkeypatch.setenv("FOREN_CODEX_LOCK", str(tmp_path / "codex.lock"))
    for name in ("FAKE_CODEX_FAIL429", "FAKE_CODEX_SLEEP"):
        monkeypatch.delenv(name, raising=False)
    return path


def codex_log(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "codex.log"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _freeze(tmp_path: Path, url: str) -> Path:
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    data["teacher"]["backends"]["qwen"].update({"endpoint": url, "model": "fake-qwen"})
    data["teacher"]["backup_backends"]["strong"]["endpoint"] = url
    path = tmp_path / "r0_freeze.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def oracle(monkeypatch, tmp_path, server, fake_codex):
    import oracle_generator as og

    for name in ("FOREN_ORACLE_SLOT_ENDPOINT", "FOREN_TEACHER_BACKUP", "FOREN_TEACHER_QWEN_MODEL", "FOREN_TEACHER_QWEN_ENDPOINT",
                 "FOREN_TEACHER_STRONG_MODEL", "FOREN_TEACHER_STRONG_ENDPOINT", "FOREN_ORACLE_RETRY_MAX_S", "FOREN_CODEX_SERIAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FOREN_ORACLE_SLOT", "slots")
    monkeypatch.setenv("FOREN_TEACHER_FREEZE", str(_freeze(tmp_path, server.url)))
    monkeypatch.setenv("FOREN_TEACHER_STRONG_API_KEY", "sk-or-test")
    sleeps: list[float] = []
    monkeypatch.setattr(og.time, "sleep", lambda s: sleeps.append(s))
    og._test_sleeps = sleeps
    og._CODEX_VERSION.clear()
    return og


def _call(gen: Any, tmp_path: Path, name: str, images: int = 0) -> tuple[str, list[dict[str, Any]], Path]:
    d = tmp_path / "requests" / f"request_{name}"
    (d / "actor").mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(images):
        p = d / "actor" / f"image_{i + 1:02d}.png"
        p.write_bytes(b"\x89PNG fake")
        paths.append(p)
    slots: list[dict[str, Any]] = []
    raw = gen._codex(d, d / "actor", name, "PROMPT " + name, paths, slots, {"mock": True})
    return raw, slots, d


# ---------------------------------------------------------------------------- 冻结文件

def test_freeze_teacher_and_verifier_sections():
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    teacher, verifier = data["teacher"], data["verifier"]
    assert teacher["backends"]["strong"]["type"] == "codex" and teacher["backends"]["strong"]["model"] == "gpt-6-astra"
    assert teacher["backup_backends"]["strong"]["type"] == "endpoint"
    assert teacher["backup_backends"]["strong"]["model"] == "anthropic/claude-sonnet-5.5"
    assert teacher["primary"] == "qwen/qwen3.5-397b-a17b+gpt-6-astra"
    # 10-15 对话 5（A5）：本机核验改回 Claude CLI 订阅；5b 的 Anthropic 直连 + 批量配置挪到 verifier.server 备用
    assert verifier["judge"] == "claude-cli" and verifier["model"] == "claude-sonnet-5-5" and verifier["model_alias"] == "sonnet"
    assert verifier["server"]["judge"] == "api" and verifier["server"]["mode"] == "batch"
    assert verifier["server"]["endpoint"] == "https://api.anthropic.com/v1"
    # A5 那条之后还会追加 changelog（A3、A10、对话 5' 换 code_hash……），只要求它在记录里
    assert any("r0 成本压缩" in c["change"] for c in data["changelog"]) and any("A5" in c["change"] for c in data["changelog"])


def test_verifier_hashes():
    import walker_run

    verifier = json.loads(FREEZE.read_text(encoding="utf-8"))["verifier"]
    # 10-17 r1：datagen/verify 改为 verifier@0.4.0，当前代码对 r1_freeze.json 草稿核；r0 的冻结值原样留在 r0_freeze.json
    r1 = json.loads((FREEZE.parent / "r1_freeze.json").read_text(encoding="utf-8"))["verifier"]
    assert walker_run.verifier_code_hash() == r1["code_hash"] and verifier["code_hash"] == r1["code_hash_r0"]
    assert walker_run.verifier_code_hash(("datagen/api_judge.py",)) == verifier["server"]["transport"]["hash"]


# ---------------------------------------------------------------------------- 分槽分发：Codex strong

def test_codex_strong_routing_and_record(oracle, server, tmp_path):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    assert gen.model_config["model"] == "fake-qwen+gpt-6-astra"
    assert gen.model_config["slots"]["S4"] == {"backend": "strong", "model": "gpt-6-astra", "role": "primary", "type": "codex"}
    assert gen.model_config["backends"]["strong"]["codex_version"] == "codex-cli 9.9.9-fake"
    json.dumps(gen.model_config)
    gen._caption_intro, gen._captions = "Captions:", ["frame at 3.0 s"]
    raw, slots, d = _call(gen, tmp_path, "S4_report", images=1)
    assert json.loads(raw)["answer"] == "ok [E3]"
    rec = slots[0]
    assert rec["backend"] == "codex" and rec["slot"] == "S4" and rec["slot_backend"] == "strong" and rec["backend_role"] == "primary"
    assert rec["model"] == "gpt-6-astra" and rec["model_served"] == "gpt-6-astra" and rec["codex_version"] == "codex-cli 9.9.9-fake"
    assert rec["usage"] == {"input_tokens": 1200, "cached_input_tokens": 800, "output_tokens": 50}
    assert rec["endpoint_host"] == "codex-cli" and rec["cost_usd"] == 0.0 and rec["effort"] == "medium"
    call = codex_log(tmp_path)[0]
    args = call["args"]
    assert args[:2] == ["exec", "--ephemeral"] and "read-only" in args and "--json" in args
    assert args[args.index("-m") + 1] == "gpt-6-astra" and 'model_reasoning_effort="medium"' in args
    assert args[args.index("--image") + 1].endswith("image_01.png") and args[-2:] == ["--", "-"]
    assert call["prompt"].startswith("PROMPT S4_report") and "Image 1: frame at 3.0 s" in call["prompt"]
    # S1 仍走 Qwen 端点；S5 的核验后重写也走 Codex
    server.calls.clear()
    _, slots, _ = _call(gen, tmp_path, "S1_query")
    assert slots[0]["backend"] == "endpoint" and server.calls[0][1]["model"] == "fake-qwen"
    _, slots, _ = _call(gen, tmp_path, "S5_answer_revise")
    assert slots[0]["backend"] == "codex" and slots[0]["slot"] == "S5"


def test_codex_usage_limit_backoff_then_ok(oracle, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_FAIL429", "2")
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    raw, slots, _ = _call(gen, tmp_path, "S5_answer")
    assert json.loads(raw)["answer"] == "ok [E3]"
    assert [a["rate_limited"] for a in slots[0]["attempts"]] == [True, True, False]
    assert oracle._test_sleeps == [30.0, 60.0] and gen.slot_failed is None


def test_codex_usage_limit_exhausted_stops_run(oracle, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_FAIL429", "99")
    monkeypatch.setenv("FOREN_ORACLE_RETRY_MAX_S", "900")
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    with pytest.raises(RuntimeError, match="usage limit"):
        _call(gen, tmp_path, "S4_report")
    assert oracle._test_sleeps == [30.0, 60.0, 120.0, 240.0, 450.0]     # 第 5 次 480 截到总预算：30+60+120+240+450 = 900
    assert len(codex_log(tmp_path)) // 2 == 6
    assert gen.slot_failed and "S4_report" in gen.slot_failed
    with pytest.raises(RuntimeError, match="earlier language slot failed"):
        gen._generate("sys", {"model_view": {}}, 9)


def test_codex_backoff_schedule_constant(oracle):
    assert oracle.CODEX_BACKOFF == (30.0, 60.0, 120.0, 240.0, 480.0)
    assert oracle.codex_rate_limited("unexpected status 429 Too Many Requests")
    assert oracle.codex_rate_limited("You've hit your usage limit. Try again in 3 minutes.")
    assert not oracle.codex_rate_limited("invalid image")


def test_codex_lock_serializes_processes(fake_codex, tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("FAKE_CODEX_SLEEP", "0.6")
    code = ("import sys; sys.path[:0] = {paths!r}; import oracle_generator as og, subprocess, os\n"
            "with og.codex_lock():\n"
            "    subprocess.run([os.environ['FOREN_CODEX_BIN'], 'exec', '-m', 'm', '-o', os.devnull], input='x', text=True)\n"
            ).format(paths=[str(ROOT / "src"), str(ROOT / "experiments" / "sft_data")])
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(2)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    rows = codex_log(tmp_path)
    spans = sorted([(r["t0"], next(x["t1"] for x in rows if x["n"] == r["n"] and "t1" in x)) for r in rows if "t0" in r])
    assert len(spans) == 2 and spans[0][1] <= spans[1][0]          # 第二个进程等第一个结束才开始


def test_backup_strong_is_endpoint_without_codex_fields(oracle, server, tmp_path, monkeypatch):
    monkeypatch.setenv("FOREN_TEACHER_BACKUP", "strong")
    table = oracle.load_slot_table()
    strong = table["backends"]["strong"]
    assert strong["type"] == "endpoint" and strong["role"] == "backup" and strong["model"] == "anthropic/claude-sonnet-5.5"
    assert strong["endpoint"] == server.url and "serial" not in {k for k, v in strong.items() if v is not None and k == "serial"}
    assert table["summary"] == "fake-qwen+anthropic/claude-sonnet-5.5"
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    server.calls.clear()
    _, slots, _ = _call(gen, tmp_path, "S5_answer")
    assert slots[0]["backend"] == "endpoint" and slots[0]["backend_role"] == "backup"
    assert server.calls[0][1]["model"] == "anthropic/claude-sonnet-5.5" and codex_log(tmp_path) == []


def test_unknown_backend_type_rejected(oracle, tmp_path):
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    data["teacher"]["backends"]["strong"]["type"] = "cli-proxy"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unknown type"):
        oracle.load_slot_table(path, backup="")


def test_endpoint_type_strong_still_works(oracle, server, tmp_path, monkeypatch):
    """将来换 CLIProxyAPI：只把冻结的 strong 改成 type=endpoint + endpoint / model。"""

    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    data["teacher"]["backends"]["qwen"].update({"endpoint": server.url, "model": "fake-qwen"})
    data["teacher"]["backends"]["strong"].update({"type": "endpoint", "endpoint": server.url, "model": "gpt-6-astra"})
    path = tmp_path / "proxy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("FOREN_TEACHER_FREEZE", str(path))
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    server.calls.clear()
    _, slots, _ = _call(gen, tmp_path, "S4_report")
    assert slots[0]["backend"] == "endpoint" and server.calls[0][1]["model"] == "gpt-6-astra"


# ---------------------------------------------------------------------------- walker_run

class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)


def test_walker_codex_preflight(oracle, tmp_path, monkeypatch):
    import walker_run

    cfg = oracle.load_slot_table()["backends"]["strong"]
    log = _Log()
    assert walker_run.codex_backend_preflight(log, cfg)
    assert any("codex-cli 9.9.9-fake" in x for x in log.lines) and any("Logged in using ChatGPT" in x for x in log.lines)
    args = codex_log(tmp_path)[0]["args"]
    assert args[args.index("-m") + 1] == "gpt-6-astra"
    monkeypatch.setenv("FAKE_CODEX_FAIL429", "99")
    log = _Log()
    assert not walker_run.codex_backend_preflight(log, cfg)
    assert any("usage limit" in x for x in log.lines)


def test_walker_teacher_check_and_judge(oracle, monkeypatch):
    import walker_run

    monkeypatch.setenv("FOREN_TEACHER_FREEZE", str(FREEZE))
    monkeypatch.delenv("FOREN_WALKER_JUDGE", raising=False)
    teacher = json.loads(FREEZE.read_text(encoding="utf-8"))["teacher"]
    check = walker_run.teacher_check(teacher)
    assert check["ok"] and check["role"] == "primary" and check["slots"]["S4"]["actual"] == "gpt-6-astra"
    assert walker_run.slot_teachers()["S5"]["endpoint_host"] == "codex-cli"
    monkeypatch.setenv("FOREN_TEACHER_BACKUP", "strong")
    check = walker_run.teacher_check(teacher)
    assert check["ok"] and check["role"] == "mixed" and check["slots"]["S5"]["role"] == "backup"
    assert walker_run.judge_kind() == "claude-cli" and walker_run.judge_label() == "claude-cli:sonnet"   # 10-15 对话 5
    monkeypatch.setenv("FOREN_WALKER_JUDGE", "api")
    assert walker_run.judge_label() == "api:claude-sonnet-5-5"


def test_second_judge_sample(tmp_path, monkeypatch):
    import walker_run

    runs = tmp_path / "runs"
    for i in range(30):
        d = runs / f"case{i:02d}_r0x_1"
        d.mkdir(parents=True)
        (d / "result.json").write_text("{}", encoding="utf-8")
    (runs / "caseZZ_r0x_1.failed-1").mkdir()
    monkeypatch.setattr(walker_run, "RUNS", runs)
    a = walker_run.second_judge_sample("r0x", 20)
    assert len(a) == 20 and a == walker_run.second_judge_sample("r0x", 20) and all(".failed-" not in r for r in a)
    assert len(walker_run.second_judge_sample("r0x", 0)) == 30


# ---------------------------------------------------------------------------- api_judge：直连 / 缓存 / 批量

def test_build_request_cache_layout(tmp_path):
    from foren_searcher.datagen.api_judge import USER_PREFIX, build_request

    img = tmp_path / "e1.png"
    img.write_bytes(b"\x89PNG" + b"x" * 50)
    payload = {"statement": "The clip shows X.", "evidence": [{"id": "E1", "text": "long shared evidence " * 60}], "task": "t"}
    body, meta = build_request("claude-sonnet-5-5", "SYSTEM " * 200, payload, [img], {"type": "object"}, effort="medium",
                               max_tokens=32000, cache_system="1h", cache_prefix="1h")
    content = body["messages"][0]["content"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert content[0]["text"].startswith(USER_PREFIX) and '"statement"' not in content[0]["text"]
    assert content[1]["type"] == "image" and content[1]["cache_control"]["ttl"] == "1h"
    assert content[-1]["text"].lstrip().startswith('"statement"') and "cache_control" not in content[-1]
    assert json.loads(content[0]["text"][len(USER_PREFIX):] + content[-1]["text"]) == payload
    assert body["output_config"]["effort"] == "medium" and body["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in body and "tools" not in body and meta["shareable"]


def test_usage_cost_and_key(monkeypatch):
    from foren_searcher.datagen import api_judge

    row = api_judge.usage_breakdown({"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
                                     "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000}, "output_tokens": 100_000},
                                    "claude-sonnet-5-5", batch=True)
    assert row["cost_usd"] == pytest.approx(0.5 * (2.0 + 0.10 + 4.0 + 1.0))
    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "sk-or-v1-openrouter")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-direct")
    assert api_judge._anthropic_key() == "sk-ant-direct"
    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "sk-ant-other")
    assert api_judge._anthropic_key() == "sk-ant-other"


def test_anthropic_judge_sync_and_batches(server, tmp_path, monkeypatch):
    import asyncio

    from foren_searcher.datagen.api_judge import AnthropicJudge, BatchClient, UsageLog

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("FOREN_JUDGE_API_KEY", raising=False)
    log = tmp_path / "usage.jsonl"
    judge = AnthropicJudge(model="anthropic/claude-sonnet-5.5", endpoint=server.url, usage_log=UsageLog(log), sleep=lambda s: None)
    assert judge.model == "claude-sonnet-5-5" and judge.name == "api:claude-sonnet-5-5"
    payload = {"statement": "s", "evidence": "shared " * 200}
    out = asyncio.run(judge.judge(system_prompt="证据核对员。只依据" + "x" * 400, payload=payload, images=[], schema=None))
    assert out == {"ok": True}
    path, body = server.calls[-1]
    assert path.endswith("/messages") and body["model"] == "claude-sonnet-5-5"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "pair" and rows[0]["mode"] == "sync" and rows[0]["cache_read"] == 1000 and rows[0]["cost_usd"] > 0
    client = BatchClient(judge)
    created = client.create([{"custom_id": "k1", "params": body}, {"custom_id": "k2", "params": body}])
    batch = client.retrieve(created["id"])
    assert batch["processing_status"] == "ended"
    results = client.results(batch)
    assert [r["custom_id"] for r in results] == ["k1", "k2"] and results[0]["result"]["type"] == "succeeded"
    judge.close()


def test_complete_verify_batch_steps():
    import complete_verify_local as cvl

    paths = cvl.output_paths("api-claude-sonnet-5-5", ["a"], specs=("hi@0.10.0", "gia@0.9.0"), round_tag="r0c5")
    steps = cvl.build_steps(["a"], ["--judge", "api", "--model", "claude-sonnet-5-5", "--endpoint", "https://api.anthropic.com/v1",
                                    "--batch-mode"], paths, ("hi@0.10.0", "gia@0.9.0"))
    for index, (_, argv) in enumerate(steps, 1):
        if index in cvl.JUDGE_STEPS:
            assert argv[:4] == ["-m", "foren_searcher.datagen.api_judge", "--batch", argv[3]]
            assert "--batch-mode" not in argv and argv[argv.index("--judge") + 1] == "api"


# ---------------------------------------------------------------------------- pack_r0：逐槽核对教师

def test_pack_r0_slot_teacher_check():
    import pack_r0

    teacher = json.loads(FREEZE.read_text(encoding="utf-8"))["teacher"]
    qwen = "qwen/qwen3.5-397b-a17b"
    primary = {**{k: qwen for k in ("S0", "S1", "S2", "S3")}, "S4": "gpt-6-astra", "S5": "gpt-6-astra"}
    prov = [{"run": "a", "slot_models": primary},
            {"run": "b", "slot_models": {**primary, "S4": "anthropic/claude-sonnet-5.5", "S5": "anthropic/claude-sonnet-5.5"}},
            {"run": "c", "slot_models": {**primary, **{k: "Qwen/Qwen3.5-397B-A17B-FP8" for k in ("S0", "S1", "S2", "S3")}}}]
    check = pack_r0.slot_teacher_check(prov, teacher)
    assert check["ok"] and check["runs"] == {"a": "primary", "b": "mixed", "c": "primary"}
    bad = pack_r0.slot_teacher_check(prov + [{"run": "d", "slot_models": {**primary, "S5": "openai/gpt-5"}},
                                             {"run": "e", "slot_models": None}], teacher)
    assert not bad["ok"] and bad["runs"]["d"] is None and bad["n_problems"] == 2
