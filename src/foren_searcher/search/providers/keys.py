"""以图搜图各引擎的 API key（10-12）：环境变量优先，其次 <项目根>/.local/<文件>（一行）。不打印 key。"""

from __future__ import annotations

import os
from pathlib import Path

# 引擎 → (环境变量, .local 下的文件名)
ENGINE_KEYS: dict[str, tuple[str, str]] = {
    "gcv": ("GOOGLE_VISION_API_KEY", "google_vision_key.txt"),
    "serpapi": ("SERPAPI_API_KEY", "serpapi_key.txt"),
    "serper": ("SERPER_API_KEY", "serper_key.txt"),
    "tineye": ("TINEYE_API_KEY", "tineye_key.txt"),
}


def first_key_line(text: str) -> str:
    """第一行既不是空行、也不以 # 开头的内容（模板文件里的说明写在 # 行）。"""

    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return ""


def find_key(name: str, root: str | Path | None = None) -> str | None:
    variable, filename = ENGINE_KEYS[name]
    key = os.environ.get(variable, "").strip()
    if not key and root is not None:
        path = Path(root) / ".local" / filename
        if path.is_file():
            key = first_key_line(path.read_text(encoding="utf-8-sig"))
    return key or None


def require_key(name: str, root: str | Path | None = None) -> str:
    key = find_key(name, root)
    if not key:
        variable, filename = ENGINE_KEYS[name]
        raise RuntimeError(f"以图搜图引擎 {name} 需要 API key：设环境变量 {variable}，或写进 .local/{filename}")
    return key


__all__ = ["ENGINE_KEYS", "find_key", "first_key_line", "require_key"]
