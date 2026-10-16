"""对话 5a（10-15）离线测试：教师按槽分后端、核验器 --judge api、冻结预检与导出字段。

全部用本机假端点（127.0.0.1 上的 HTTP 服务），不连外网、不调 Codex / Claude CLI、不要 GPU：
- 分槽路由：S0–S3 → qwen 后端，S4 / S5（含 _retry / _revise）→ strong 后端；请求体按后端配置（温度 / 思考 / effort）；
  每次槽调用记 slot / slot_backend / backend_role / model / usage / cost；429 退避沿用端点逻辑；备用开关与环境变量覆盖；
- ApiJudge：与 ClaudeCliJudge 送出的文字逐字相同；system 原文、base64 图、strict json_schema、effort、不发温度与工具；
  429 / 5xx 退避、端点不认 schema 时降级；启动器把 "api" 接进 verify CLI 而不改 verify/ 里的文件；
- complete_verify_local --judge api 的判题步骤经启动器；walker_run 冻结预检按槽核对；export 的 slot_teacher / terminal_teacher。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "sft_data"))
sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------- 假端点

class FakeServer:
    """按顺序回放脚本里的回答；默认回一个合法 JSON。记录每个请求体。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.script: list[tuple[int, Any]] = []
        self.models = ["fake-qwen", "fake-strong", "fake-qwen-backup", "fake-strong-backup", "anthropic/claude-sonnet-5.5"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _send(self, status: int, payload: Any) -> None:
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._send(200, {"data": [{"id": m} for m in outer.models]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(body)
                if outer.script:
                    status, payload = outer.script.pop(0)
                else:
                    status, payload = 200, None
                if payload is None:
                    payload = {"model": body.get("model"), "provider": "FakeProvider",
                               "choices": [{"message": {"content": json.dumps({"ok": True, "answer": "ok [E3]", "rationale": "r",
                                                                                 "query": "q", "decision": "FINISH"})},
                                            "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 11, "completion_tokens": 7, "cost": 0.001,
                                         "completion_tokens_details": {"reasoning_tokens": 3}}}
                self._send(status, payload)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture()
def server():
    s = FakeServer()
    yield s
    s.close()


def _freeze(tmp_path: Path, url: str) -> Path:
    data = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))
    data["teacher"]["backends"]["qwen"].update({"endpoint": url, "model": "fake-qwen"})
    # 10-15 r0 成本压缩起冻结的 strong 是 type=codex；这里测的是端点类型的分槽路径（5a 的配置，也是将来换 CLIProxyAPI 的路径）
    data["teacher"]["backends"]["strong"] = {**data["teacher"]["previous_strong_5a"], "type": "endpoint", "endpoint": url,
                                             "model": "fake-strong"}
    data["teacher"]["backup_backends"]["qwen"]["model"] = "fake-qwen-backup"
    data["teacher"]["backup_backends"]["strong"].update({"endpoint": url, "model": "fake-strong-backup"})
    path = tmp_path / "r0_freeze.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def oracle(monkeypatch, tmp_path, server):
    import oracle_generator as og

    for name in ("FOREN_ORACLE_SLOT_ENDPOINT", "FOREN_TEACHER_BACKUP", "FOREN_TEACHER_QWEN_MODEL", "FOREN_TEACHER_QWEN_ENDPOINT",
                 "FOREN_TEACHER_STRONG_MODEL", "FOREN_TEACHER_STRONG_ENDPOINT", "FOREN_ORACLE_CASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FOREN_ORACLE_SLOT", "slots")
    monkeypatch.setenv("FOREN_TEACHER_FREEZE", str(_freeze(tmp_path, server.url)))
    monkeypatch.setattr(og.time, "sleep", lambda s: None)
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


# ---------------------------------------------------------------------------- 分槽路由

def test_slot_key_prefixes(oracle):
    assert [oracle.slot_key(n) for n in ("S0_rationale", "S1_query_retry", "S2_task", "S3_next", "S4_report_retry",
                                         "S5_answer_revise", "S5_answer_bon2")] == ["S0", "S1", "S2", "S3", "S4", "S5", "S5"]


def test_routing_and_records(oracle, server, tmp_path):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    assert gen.backend == "slots" and gen.codex is None
    assert gen.model_config["model"] == "fake-qwen+fake-strong"
    assert {k: v["model"] for k, v in gen.model_config["slots"].items()} == {
        "S0": "fake-qwen", "S1": "fake-qwen", "S2": "fake-qwen", "S3": "fake-qwen", "S4": "fake-strong", "S5": "fake-strong"}
    json.dumps(gen.model_config)                 # 能写进 transport.json（没有 client / key）
    expected = {"S0_rationale": "fake-qwen", "S1_query": "fake-qwen", "S1_query_retry": "fake-qwen", "S2_task": "fake-qwen",
                "S3_next": "fake-qwen", "S4_report": "fake-strong", "S4_report_retry": "fake-strong", "S5_answer": "fake-strong",
                "S5_answer_revise": "fake-strong"}
    for name, model in expected.items():
        server.calls.clear()
        _, slots, d = _call(gen, tmp_path, name, images=1)
        body = server.calls[0]
        assert body["model"] == model, name
        rec = slots[0]
        assert rec["model"] == model and rec["backend"] == "endpoint" and rec["slot"] == name[:2]
        assert rec["slot_backend"] == ("qwen" if model == "fake-qwen" else "strong") and rec["backend_role"] == "primary"
        assert rec["usage"] == {"input_tokens": 11, "output_tokens": 7} and rec["cost_usd"] == 0.001 and rec["reasoning_tokens"] == 3
        assert rec["endpoint_host"].startswith("127.0.0.1:")
        assert (d / f"slot_{name}_output.txt").is_file()
        content = body["messages"][0]["content"]
        assert content[0]["text"] == "PROMPT " + name and content[-1]["image_url"]["url"].startswith("data:image/png;base64,")
        if model == "fake-qwen":                 # qwen：温度 0.3；S0 关思考（vLLM 的 chat_template_kwargs），其余槽开
            assert body["temperature"] == 0.3
            assert ("chat_template_kwargs" in body) == (name == "S0_rationale")
        else:                                    # strong：不发温度，开思考（不发 enable_thinking=false）
            assert "temperature" not in body and "chat_template_kwargs" not in body
            assert body["max_tokens"] == 16384
    assert gen.slot_calls == len(expected)


def test_openrouter_bodies(oracle):
    frozen = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))
    frozen["teacher"]["backends"]["strong"] = frozen["teacher"]["previous_strong_5a"]     # 5a 的 OpenRouter strong（端点类型）
    path = Path(__import__("tempfile").mkdtemp()) / "r0_freeze.json"
    path.write_text(json.dumps(frozen, ensure_ascii=False), encoding="utf-8")
    table = oracle.load_slot_table(str(path), backup="")
    qwen, strong = table["backends"]["qwen"], table["backends"]["strong"]
    assert qwen["openrouter"] and strong["openrouter"] and qwen["trust_env"] and strong["trust_env"]
    b0 = oracle.endpoint_body(qwen, "S0_rationale", [{"type": "text", "text": "x"}])
    b1 = oracle.endpoint_body(qwen, "S1_query", [{"type": "text", "text": "x"}])
    b5 = oracle.endpoint_body(strong, "S5_answer", [{"type": "text", "text": "x"}])
    assert b0["reasoning"] == {"enabled": False} and b1["reasoning"] == {"enabled": True} and b1["temperature"] == 0.3
    assert b5["model"] == "openai/gpt-6-astra" and b5["reasoning"] == {"effort": "medium"} and "temperature" not in b5
    assert b5["response_format"] == {"type": "json_object"} and b5["usage"] == {"include": True}
    assert table["summary"] == "qwen/qwen3.5-397b-a17b+openai/gpt-6-astra"


def test_strong_429_backoff_then_ok(oracle, server, tmp_path):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    server.script = [(429, {"error": "rate"}), (503, {"error": "busy"})]
    raw, slots, _ = _call(gen, tmp_path, "S5_answer")
    assert json.loads(raw)["answer"] == "ok [E3]"
    assert [a.get("status") for a in slots[0]["attempts"]] == [429, 503, 200]
    assert gen.slot_failed is None


def test_slot_failure_latches(oracle, server, tmp_path, monkeypatch):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    for cfg in gen._backends.values():
        cfg["attempts"] = 2
    server.script = [(429, {"error": "rate"})] * 2
    with pytest.raises(RuntimeError):
        _call(gen, tmp_path, "S4_report")
    assert gen.slot_failed and "S4_report" in gen.slot_failed
    with pytest.raises(RuntimeError, match="earlier language slot failed"):
        gen._generate("sys", {"model_view": {}}, 9)


def test_backup_switch_and_env_override(oracle, server, tmp_path, monkeypatch):
    monkeypatch.setenv("FOREN_TEACHER_BACKUP", "strong")
    monkeypatch.setenv("FOREN_TEACHER_QWEN_MODEL", "Fake-Qwen-FP8")
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    assert gen.model_config["slots"]["S5"] == {"backend": "strong", "model": "fake-strong-backup", "role": "backup", "type": "endpoint"}
    assert gen.model_config["slots"]["S1"]["model"] == "Fake-Qwen-FP8"
    assert gen.model_config["backends"]["qwen"]["env_overrides"] == {"model": "FOREN_TEACHER_QWEN_MODEL"}
    server.calls.clear()
    _, slots, _ = _call(gen, tmp_path, "S5_answer_revise")
    assert server.calls[0]["model"] == "fake-strong-backup" and slots[0]["backend_role"] == "backup"
    assert oracle.model_matches("Fake-Qwen-FP8", "org/fake-qwen") and not oracle.model_matches("fake-qwen2", "fake-qwen")
    assert oracle.model_matches("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5")


def test_unknown_slot_name_is_an_error(oracle, tmp_path):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    with pytest.raises(RuntimeError, match="not in the frozen slot table"):
        _call(gen, tmp_path, "S9_something")


def test_legacy_single_endpoint_unchanged(oracle, server, tmp_path, monkeypatch):
    monkeypatch.setenv("FOREN_ORACLE_SLOT", "")
    monkeypatch.setenv("FOREN_ORACLE_SLOT_ENDPOINT", server.url)
    monkeypatch.setenv("FOREN_ORACLE_SLOT_MODEL", "one-model")
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    assert gen.backend == "endpoint"
    for name in ("S0_rationale", "S5_answer"):
        server.calls.clear()
        _call(gen, tmp_path, name)
        assert server.calls[0]["model"] == "one-model" and server.calls[0]["temperature"] == 0.3
        assert ("chat_template_kwargs" in server.calls[0]) == (name == "S0_rationale")


# ---------------------------------------------------------------------------- 核验器 --judge api

def _cli_text(payload: dict[str, Any], images: list[Path]) -> tuple[str, str]:
    """ClaudeCliJudge 会送出的 system 与 user 文字（截住 _run，不起进程）。"""

    from foren_searcher.datagen.verify.judges import ClaudeCliJudge

    judge = ClaudeCliJudge(model="sonnet", cli=["true"])
    seen: dict[str, Any] = {}

    def fake_run(system_prompt, text, imgs, schema):
        seen.update(system=system_prompt, text=text, images=imgs, schema=schema)
        return {"ok": True}, {}

    judge._run = fake_run
    asyncio.run(judge.judge(system_prompt="SYS 原文", payload=payload, images=images, schema={"type": "object"}))
    judge.close()
    return seen["system"], seen["text"]


def test_api_judge_matches_cli_text(server, tmp_path, monkeypatch):
    from foren_searcher.datagen.api_judge import ApiJudge

    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "k")
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG fake")
    payload = {"statement": "会议在周二举行。", "evidence": [{"id": "E1", "text": "周二"}]}
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    judge = ApiJudge(model="anthropic/claude-sonnet-5.5", endpoint=server.url, sleep=lambda s: None)
    answer = asyncio.run(judge.judge(system_prompt="SYS 原文", payload=payload, images=[img], schema=schema))
    assert answer["ok"] is True
    body = server.calls[-1]
    cli_system, cli_text = _cli_text(payload, [img])
    assert body["messages"][0] == {"role": "system", "content": cli_system}
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": cli_text}
    assert user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["response_format"]["json_schema"] == {"name": "verdict", "schema": schema, "strict": True}
    assert "tools" not in body and "temperature" not in body and body["max_tokens"] == 32000
    assert judge.name == "api:anthropic/claude-sonnet-5.5"


def test_api_judge_openrouter_body(monkeypatch):
    from foren_searcher.datagen.api_judge import ApiJudge

    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "k")
    monkeypatch.delenv("FOREN_JUDGE_API_PROVIDER", raising=False)
    judge = ApiJudge(endpoint="https://openrouter.ai/api/v1")
    body = judge.body("s", "t", [], None)
    assert body["model"] == "anthropic/claude-sonnet-5.5" and body["reasoning"] == {"effort": "medium"}
    assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": True} and "response_format" not in body
    assert "temperature" not in body and "tools" not in body
    direct = ApiJudge(endpoint="https://api.anthropic.com/v1", model="claude-sonnet-5-5")
    body = direct.body("s", "t", [], None)
    assert "reasoning" not in body and "reasoning_effort" not in body and "provider" not in body
    judge.close(), direct.close()


def test_api_judge_retry_and_schema_downgrade(server, monkeypatch):
    from foren_searcher.datagen.api_judge import ApiJudge
    from foren_searcher.datagen.verify.judges import LocalJudgeError

    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "k")
    waits: list[float] = []
    judge = ApiJudge(model="anthropic/claude-sonnet-5.5", endpoint=server.url, sleep=waits.append, attempts=6)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    server.script = [(429, {"error": "slow down"}), (400, {"error": "json_schema strict mode is not supported"}),
                     (402, {"error": {"message": "This request would exceed your available credits given your current in-flight "
                                                 "requests. Retry after in-flight requests complete."}}),
                     (502, {"error": "upstream"})]
    out = asyncio.run(judge.judge(system_prompt="s", payload={"a": 1}, images=[], schema=schema))
    assert out["ok"] is True and judge.schema_mode(schema) == "loose" and waits == [5.0, 20.0, 40.0]
    assert "strict" not in server.calls[-1]["response_format"]["json_schema"]
    assert judge.calls == 1 and judge.retries == 3 and judge.providers_used == {"FakeProvider": 1}
    other = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": False}
    assert judge.schema_mode(other) == "strict"          # 只降被拒的那个 schema
    server.script = [(401, {"error": "bad key"})]
    with pytest.raises(LocalJudgeError) as err:
        asyncio.run(judge.judge(system_prompt="s", payload={"a": 2}, images=[], schema=None))
    assert err.value.status_code == 401
    judge.close()


def test_api_judge_resolve(server, monkeypatch):
    from foren_searcher.datagen.api_judge import ApiJudge
    from foren_searcher.datagen.verify.judges import LocalJudgeUnavailable

    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "k")
    assert ApiJudge(model="anthropic/claude-sonnet-5.5", endpoint=server.url).resolve() == "api:anthropic/claude-sonnet-5.5"
    with pytest.raises(LocalJudgeUnavailable, match="没有模型"):
        ApiJudge(model="anthropic/claude-sonnet-9", endpoint=server.url).resolve()


def test_launcher_installs_api_kind(tmp_path, monkeypatch):
    import argparse

    from foren_searcher.datagen import api_judge
    from foren_searcher.datagen.verify import judges

    monkeypatch.setenv("FOREN_JUDGE_API_KEY", "k")
    api_judge.install()
    parser = argparse.ArgumentParser()
    judges.add_judge_arguments(parser, default="local", choices=("local", "openrouter", "claude-cli"))
    args = parser.parse_args(["--judge", "api", "--model", "anthropic/claude-sonnet-5.5", "--endpoint", "http://127.0.0.1:9/v1"])
    judge = judges.judge_from_args(args, cache_path=tmp_path / "verify_cache.jsonl")
    assert isinstance(judge, judges.CachedJudge) and judge.path.name == "verify_cache.api-claude-sonnet-5.5.jsonl"
    assert judge.name == "api:anthropic/claude-sonnet-5.5"
    # 其余 kind 原样走原来的 build_judge
    assert judges.build_judge("none") is None
    # 子进程：verify CLI 经启动器后 --judge 认 api（verify/ 里的文件一个字没改）
    proc = subprocess.run([sys.executable, "-m", "foren_searcher.datagen.api_judge", "foren_searcher.datagen.verify.reverify", "--help"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                          env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert proc.returncode == 0 and "api" in proc.stdout.split("--judge", 1)[1][:200], proc.stdout[-800:] + proc.stderr[-800:]


def test_verifier_code_hash_unchanged():
    import walker_run

    frozen = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))["verifier"]
    r1 = json.loads((ROOT / "specs" / "r1_freeze.json").read_text(encoding="utf-8"))["verifier"]       # 10-17：verifier@0.4.0
    assert walker_run.verifier_code_hash() == r1["code_hash"] and frozen["code_hash"] == r1["code_hash_r0"]
    assert frozen["judge"] == "claude-cli" and frozen["model"] == "claude-sonnet-5-5"          # 10-15 对话 5：改回 Claude CLI 订阅
    assert frozen["server"]["judge"] == "api" and frozen["server"]["mode"] == "batch"           # 5b 的 API 配置备用


def test_complete_verify_steps_use_launcher():
    import complete_verify_local as cvl

    paths = cvl.output_paths("api-claude-sonnet-5.5", ["a", "b"], specs=("hi@0.10.0", "gia@0.9.0"), round_tag="wk13api")
    steps = cvl.build_steps(["a", "b"], ["--judge", "api", "--model", "m", "--endpoint", "e"], paths, ("hi@0.10.0", "gia@0.9.0"))
    for index, (_, argv) in enumerate(steps, 1):
        if index in cvl.JUDGE_STEPS:
            assert argv[:3] == ["-m", "foren_searcher.datagen.api_judge", argv[2]] and argv[2].startswith("foren_searcher.datagen.verify.")
        else:
            assert argv[1] != "foren_searcher.datagen.api_judge"
    local = cvl.build_steps(["a"], ["--judge", "local", "--model", "m"], paths)
    assert local[0][1][:2] == ["-m", "foren_searcher.datagen.verify.reverify"]
    assert "verdicts_wk13api_judged.api-claude-sonnet-5.5" in paths["verdicts"]


# ---------------------------------------------------------------------------- walker_run 预检 / 导出字段

def test_walker_teacher_check(oracle, monkeypatch):
    import walker_run

    monkeypatch.setenv("FOREN_TEACHER_FREEZE", str(ROOT / "specs" / "r0_freeze.json"))
    teacher = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))["teacher"]
    check = walker_run.teacher_check(teacher)
    assert check["ok"] and check["role"] == "primary"
    assert walker_run.teacher_model() == "qwen/qwen3.5-397b-a17b+gpt-6-astra"            # 10-15：strong = Codex gpt-6-astra
    monkeypatch.setenv("FOREN_TEACHER_BACKUP", "strong")
    check = walker_run.teacher_check(teacher)
    assert check["ok"] and check["role"] == "mixed" and check["slots"]["S5"]["role"] == "backup"
    monkeypatch.setenv("FOREN_TEACHER_STRONG_MODEL", "openai/gpt-5")       # 不在冻结表里的模型 → 不通过
    assert not walker_run.teacher_check(teacher)["ok"]
    monkeypatch.setenv("FOREN_ORACLE_SLOT", "")
    assert not walker_run.teacher_check(teacher)["ok"]                      # 不是分槽模式（Codex / CLI）→ 不通过
    monkeypatch.delenv("FOREN_WALKER_JUDGE", raising=False)
    assert walker_run.judge_kind() == "claude-cli" and walker_run.judge_label() == "claude-cli:sonnet"


def test_export_slot_teacher_fields():
    from foren_searcher.datagen.export import slot_teachers

    class Log:
        teacher_model = "qwen/qwen3.5-397b-a17b+openai/gpt-6-astra"
        config = {"walker": {"slot_models": {"S0": "qwen/qwen3.5-397b-a17b", "S1": "qwen/qwen3.5-397b-a17b",
                                             "S2": "qwen/qwen3.5-397b-a17b", "S3": "qwen/qwen3.5-397b-a17b",
                                             "S4": "openai/gpt-6-astra", "S5": "anthropic/claude-sonnet-5.5"}}}

    assert slot_teachers(Log()) == {"slot_teacher": "qwen/qwen3.5-397b-a17b",
                                    "terminal_teacher": "openai/gpt-6-astra+anthropic/claude-sonnet-5.5"}

    class Old:
        teacher_model = "gpt-6-astra"
        config: dict[str, Any] = {"walker": {"teacher_model": "gpt-6-astra"}}

    assert slot_teachers(Old()) == {"slot_teacher": "gpt-6-astra", "terminal_teacher": "gpt-6-astra"}


def test_transport_schema_rewrites_type_arrays():
    from foren_searcher.datagen.api_judge import transport_schema, verifier_schemas

    schemas = verifier_schemas()
    assert {"PAIR_SCHEMA", "SUFFICIENCY_SCHEMA", "PLAN_SCHEMA"} <= set(schemas)
    hint = transport_schema(schemas["PAIR_SCHEMA"])["properties"]["contradicted_parts"]["items"]["properties"]["hint"]
    assert hint == {"anyOf": [{"type": "string", "enum": ["misread_fine_text", "sloppy_numeric"]}, {"type": "null"}]}
    by = transport_schema(schemas["SUFFICIENCY_SCHEMA"])["properties"]["requirements"]["items"]["properties"]["by"]
    assert by == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert transport_schema(schemas["COVERAGE_SCHEMA"]) == schemas["COVERAGE_SCHEMA"]      # 没有类型数组的原样


def test_teacher_retries_402_in_flight(oracle, server, tmp_path):
    gen = oracle.OracleGenerator(store=None, output_dir=tmp_path / "requests")
    server.script = [(402, {"error": {"message": "would exceed your available credits given your current in-flight requests"}})]
    raw, slots, _ = _call(gen, tmp_path, "S4_report")
    assert [a.get("status") for a in slots[0]["attempts"]] == [402, 200] and gen.slot_failed is None
    server.script = [(402, {"error": {"message": "Insufficient credits"}})]
    with pytest.raises(RuntimeError):
        _call(gen, tmp_path, "S5_answer")
