"""verifier@0.4.0（r1 预演之一，2026-10-17）：缓存分键 + 五处误报修正，每处一组测试。

金标条目编号见 claude/r1核验器修正_20261017.md（r0g2 的 F001–F015、r0a 的 F001/F002）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from foren_searcher.datagen.verify import judges
from foren_searcher.datagen.verify.judges import CachedJudge, CallableJudge
from foren_searcher.datagen.verify.verdict import VERIFIER_VERSION


def run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------- 0. 缓存键带核验器版本

def test_cache_key_namespaced_by_verifier_version(tmp_path: Path, monkeypatch: Any) -> None:
    assert VERIFIER_VERSION == "verifier@0.4.0"
    new = CachedJudge.key_for("S", {"a": 1}, [])
    monkeypatch.setattr(judges, "cache_namespace", lambda: "")
    old = CachedJudge.key_for("S", {"a": 1}, [])
    assert new != old                                    # 同一道题，0.3.x 的旧回答不命中
    # 旧缓存里有这道题的回答，0.4.0 的 CachedJudge 照样重新调用
    path = tmp_path / "cache.jsonl"
    path.write_text(json.dumps({"key": old, "answer": {"x": "old"}}) + "\n", encoding="utf-8")
    monkeypatch.undo()
    calls: list[int] = []

    async def gen(system_prompt: str, payload: dict[str, Any], images: list[Path]) -> dict[str, Any]:
        calls.append(1)
        return {"x": "new"}

    judge = CachedJudge(CallableJudge(gen), path)
    assert run(judge.judge(system_prompt="S", payload={"a": 1}, images=[])) == {"x": "new"} and calls == [1]
