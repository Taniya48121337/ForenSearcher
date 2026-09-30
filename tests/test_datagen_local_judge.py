"""本地核验（OpenAICompatJudge）：请求格式、自动降级、错误分类、缓存按模型分文件、CLI 参数。全部离线（httpx.MockTransport）。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from foren_searcher.datagen.verify import judges
from foren_searcher.datagen.verify.anchored import _call_checked
from foren_searcher.datagen.verify.judges import (CachedJudge, LocalJudgeError, LocalJudgeUnavailable, OpenAICompatJudge,
                                                  add_judge_arguments, build_judge, judge_tag, local_cache_path,
                                                  resolve_local_model)

SCHEMA = {"type": "object", "properties": {"grounded": {"type": "boolean"}}, "required": ["grounded"]}


def _reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def _judge(handler, **kwargs) -> tuple[OpenAICompatJudge, list[dict]]:
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return handler(request, body)

    judge = OpenAICompatJudge(model="qwen3.5-9b", endpoint="http://local.test/v1", transport=httpx.MockTransport(wrapped), **kwargs)
    return judge, seen


def test_request_carries_schema_images_and_no_thinking(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8fake")
    judge, seen = _judge(lambda request, body: _reply('{"grounded": true}'))
    answer = asyncio.run(judge.judge(system_prompt="SYS", payload={"a": 1}, images=[image], schema=SCHEMA))
    assert answer == {"grounded": True}
    body = seen[0]
    assert body["model"] == "qwen3.5-9b" and body["temperature"] == 0.0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    user = body["messages"][1]["content"]
    assert user[0]["type"] == "text" and "待核数据" in user[0]["text"]
    assert user[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert judge.name == "local:qwen3.5-9b"


def test_thinking_on_sends_no_template_switch_and_strips_think_block() -> None:
    judge, seen = _judge(lambda request, body: _reply('<think>先想一想 {"x":1}</think>\n{"grounded": false}'), thinking=True)
    assert asyncio.run(judge.judge(system_prompt="S", payload={}, images=[])) == {"grounded": False}
    assert "chat_template_kwargs" not in seen[0]
    assert judge.name == "local:qwen3.5-9b+think"


def test_unsupported_optional_fields_are_dropped_once_and_remembered() -> None:
    def handler(request, body):
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format json_schema is not supported"})
        return _reply('{"grounded": true}')

    judge, seen = _judge(handler)
    assert asyncio.run(judge.judge(system_prompt="S", payload={}, images=[], schema=SCHEMA)) == {"grounded": True}
    assert asyncio.run(judge.judge(system_prompt="S", payload={"b": 2}, images=[], schema=SCHEMA)) == {"grounded": True}
    # 第一次：带 → 400 → 去掉重发；第二次直接不带
    assert ["response_format" in b for b in seen] == [True, False, False]
    assert all("chat_template_kwargs" in b for b in seen)      # 报错没点名它，不去掉


def test_unrelated_400_is_raised_without_dropping_schema() -> None:
    judge, seen = _judge(lambda request, body: httpx.Response(400, json={"error": "maximum context length exceeded"}))
    with pytest.raises(LocalJudgeError) as info:
        asyncio.run(judge.judge(system_prompt="S", payload={}, images=[], schema=SCHEMA))
    assert info.value.status_code == 400
    assert len(seen) == 1 and "response_format" in seen[0]


def test_server_error_is_retried_by_call_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("foren_searcher.datagen.verify.anchored._backoff", lambda failures: 0.0)
    calls = {"n": 0}

    def handler(request, body):
        calls["n"] += 1
        return httpx.Response(503, text="busy") if calls["n"] == 1 else _reply('{"grounded": true}')

    judge, _ = _judge(handler)
    raw, failure = asyncio.run(_call_checked(judge, "S", {}, [], SCHEMA))
    assert raw == {"grounded": True} and calls["n"] == 2


def test_resolve_model_uses_single_served_model_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(judges.LOCAL_MODEL_ENV, raising=False)
    one = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "gemma-local"}]}))
    assert resolve_local_model("http://x/v1", None, transport=one) == "gemma-local"
    with pytest.raises(LocalJudgeUnavailable, match="没有模型"):
        resolve_local_model("http://x/v1", "other", transport=one)


def test_unreachable_server_gives_clear_error() -> None:
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LocalJudgeUnavailable, match="连不上本地核验服务"):
        resolve_local_model("http://127.0.0.1:9/v1", "m", transport=httpx.MockTransport(refuse))


def test_local_cache_is_separate_file_per_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert judge_tag("local:Qwen/Qwen3.5-9B+think") == "local-qwen3.5-9b-think"
    assert judge_tag("openrouter:qwen/qwen3.8-27b") == "openrouter-qwen3.8-27b"
    base = tmp_path / "verify_cache.jsonl"
    assert local_cache_path(base, "local:qwen3.5-9b") == tmp_path / "verify_cache.local-qwen3.5-9b.jsonl"

    monkeypatch.setattr(judges, "resolve_local_model", lambda endpoint, model: "qwen3.5-9b")
    # 旧 OpenRouter 缓存里即使有同一个问题的回答，本地 judge 也不能拿来用
    key = CachedJudge.key_for("S", {}, [])
    base.write_text(json.dumps({"key": key, "answer": {"grounded": "from-openrouter"}}) + "\n", encoding="utf-8")
    judge = build_judge("local", cache_path=base, endpoint="http://local.test/v1")
    assert isinstance(judge, CachedJudge) and judge.path.name == "verify_cache.local-qwen3.5-9b.jsonl"
    assert key not in judge._cache


def test_cli_arguments_default_to_local() -> None:
    parser = argparse.ArgumentParser()
    add_judge_arguments(parser)
    args = parser.parse_args([])
    assert args.judge == "local" and args.think is False
    args = parser.parse_args(["--judge", "openrouter", "--model", "m", "--reasoning", "none"])
    assert args.judge == "openrouter" and args.reasoning == "none"


def test_reverify_accepts_explicit_run_dirs(tmp_path: Path) -> None:
    from foren_searcher.datagen.verify.reverify import expand_runs

    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "store.sqlite3").write_bytes(b"")
    (tmp_path / "not_a_run").mkdir()
    assert expand_runs([str(tmp_path)]) == [tmp_path / "a", tmp_path / "b"]
    assert expand_runs([str(tmp_path / "b")]) == [tmp_path / "b"]
    with pytest.raises(SystemExit):
        expand_runs([str(tmp_path / "missing")])
