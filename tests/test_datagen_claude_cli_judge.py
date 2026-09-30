"""ClaudeCliJudge（2026-09-30）：用假的 claude 可执行文件（python 脚本）测命令行参数、stream-json 输入、结构化输出、限流与失败处理。"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from foren_searcher.datagen.verify.judges import (ClaudeCliError, ClaudeCliJudge, ClaudeCliLimit, build_judge, judge_tag,
                                                  local_cache_path)

FAKE = r'''
import json, sys, os
args = sys.argv[1:]
log = os.environ["FAKE_LOG"]
stdin = sys.stdin.read()
open(log, "a", encoding="utf-8").write(json.dumps({"args": args, "stdin": stdin}, ensure_ascii=False) + "\n")
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "limit":
    print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "Claude usage limit reached"}))
    sys.exit(1)
if mode == "crash":
    sys.stderr.write("boom"); sys.exit(2)
schema = json.loads(args[args.index("--json-schema") + 1]) if "--json-schema" in args else None
out = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.01,
       "modelUsage": {"claude-haiku-4-5": {"outputTokens": 3}, "claude-sonnet-5-5": {"outputTokens": 40}}}
if schema is not None:
    out["structured_output"] = {"ok": True, "n_images": sum(1 for c in json.loads(stdin)["message"]["content"] if c["type"] == "image")}
    out["result"] = json.dumps(out["structured_output"])
else:
    out["result"] = "```json\n{\"plain\": 1}\n```"
print(json.dumps({"type": "system", "subtype": "init"}))
print(json.dumps(out))
'''


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    return [sys.executable, str(script)], log


def test_resolve_and_structured_output_with_images(fake, tmp_path):
    cli, log = fake
    judge = ClaudeCliJudge(model="sonnet", cli=cli)
    assert judge.resolve() == "claude-cli:claude-sonnet-5-5"
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out = asyncio.run(judge.judge(system_prompt="SYS", payload={"statement": "x"}, images=[img],
                                  schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}))
    assert out == {"ok": True, "n_images": 1}
    call = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()][-1]
    args = call["args"]
    for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--json-schema"):
        assert flag in args
    assert args[args.index("--tools") + 1] == "" and args[args.index("--setting-sources") + 1] == ""
    assert args[args.index("--model") + 1] == "claude-sonnet-5-5" and args[args.index("--system-prompt") + 1] == "SYS"
    msg = json.loads(call["stdin"])
    assert msg["message"]["content"][0]["text"].startswith("以下为待核数据")
    assert msg["message"]["content"][1]["source"]["media_type"] == "image/png"
    assert judge.calls == 2 and abs(judge.cost_usd - 0.02) < 1e-9
    judge.close()


def test_plain_text_json_is_parsed_without_schema(fake):
    judge = ClaudeCliJudge(cli=fake[0])
    assert asyncio.run(judge.judge(system_prompt="S", payload={}, images=[])) == {"plain": 1}


def test_limit_is_base_exception_and_sticky(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "limit")
    judge = ClaudeCliJudge(cli=fake[0])
    with pytest.raises(ClaudeCliLimit):
        asyncio.run(judge.judge(system_prompt="S", payload={}, images=[]))
    assert not issubclass(ClaudeCliLimit, Exception)      # anchored 的 except Exception 吞不掉，整步停下
    monkeypatch.setenv("FAKE_MODE", "ok")
    with pytest.raises(ClaudeCliLimit):                     # 一旦限流，后续调用不再起进程
        asyncio.run(judge.judge(system_prompt="S", payload={}, images=[]))


def test_crash_is_retryable_error(fake, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    judge = ClaudeCliJudge(cli=fake[0])
    with pytest.raises(ClaudeCliError) as info:
        asyncio.run(judge.judge(system_prompt="S", payload={}, images=[]))
    assert info.value.status_code is None and "boom" in str(info.value)


def test_build_judge_uses_model_tagged_cache(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("FOREN_CLAUDE_CLI", "unused")
    monkeypatch.setattr("foren_searcher.datagen.verify.judges.find_claude_cli", lambda explicit=None: fake[0])
    judge = build_judge("claude-cli", cache_path=tmp_path / "verify_cache.jsonl")
    assert judge.name == "claude-cli:claude-sonnet-5-5"
    assert judge.path == local_cache_path(tmp_path / "verify_cache.jsonl", "claude-cli:claude-sonnet-5-5")
    assert judge_tag(judge.name) == "claude-cli-claude-sonnet-5-5"


def test_find_cli_falls_back_to_common_install_locations(tmp_path, monkeypatch):
    from foren_searcher.datagen.verify import judges

    exe = tmp_path / ".local" / "bin" / "claude.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.delenv("FOREN_CLAUDE_CLI", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(judges.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert judges.find_claude_cli() == [str(exe)]
    exe.unlink()
    with pytest.raises(judges.LocalJudgeUnavailable) as info:
        judges.find_claude_cli()
    assert "claude.exe" in str(info.value)
