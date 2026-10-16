"""测试会话的护栏（r1，10-17）：受 git 管理的核验缓存 experiments/sft_data/verify_cache*.jsonl 不许被测试改动。

10-16 / 10-17 在主目录跑核验相关测试时，这个文件被追加过假回答（r1 核验器修正报告五节、八节 7）。10-17 逐个文件
重跑与全量重跑都没再复现，没找到是哪一条；为免再发生，会话开始记下每个文件的大小与 sha1，结束时：只是在末尾追加
→ 截回原长度并报错；内容别处变了 → 只报错（不动文件，人工处理）。测试里要缓存的请用 tmp_path。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_SFT = Path(__file__).resolve().parents[1] / "experiments" / "sft_data"


def _snapshot() -> dict[Path, tuple[int, str]]:
    out = {}
    for path in sorted(_SFT.glob("verify_cache*.jsonl")):
        data = path.read_bytes()
        out[path] = (len(data), hashlib.sha1(data).hexdigest())
    return out


@pytest.fixture(scope="session", autouse=True)
def _tracked_verify_caches_untouched():
    before = _snapshot()
    yield
    problems = []
    for path, (size, digest) in before.items():
        if not path.is_file():
            problems.append(f"{path.name} 被删除")
            continue
        data = path.read_bytes()
        if hashlib.sha1(data).hexdigest() == digest:
            continue
        if len(data) > size and hashlib.sha1(data[:size]).hexdigest() == digest:
            path.write_bytes(data[:size])
            problems.append(f"{path.name} 被追加了 {len(data) - size} 字节（已截回原长度）")
        else:
            problems.append(f"{path.name} 内容被改动（未自动恢复）")
    for path in set(_SFT.glob("verify_cache*.jsonl")) - set(before):
        problems.append(f"测试新建了 {path.name}（请改用 tmp_path）")
    if problems:
        pytest.fail("测试改动了受管理的核验缓存：" + "；".join(problems), pytrace=False)
