"""g1：给 codex_gate.py / claude_gate.py 做 Windows 启动器（.g1/bin/*.exe）。

格式同 pip / distlib 的控制台启动器：t64.exe + "#!<python>" 行 + 内含 __main__.py 的 zip。启动器把子进程放进
KILL_ON_JOB_CLOSE 的作业对象：walker 超时杀掉启动器时，闸门进程和它起的 codex / claude 一并结束（.cmd 垫片做不到，
还会经 cmd.exe 改坏带引号的参数）。生成物不进 git（.g1/ 在 .git/info/exclude）。
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pip._vendor.distlib as distlib

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / ".g1" / "bin"
HERE = Path(__file__).resolve().parent

STUB = """import sys
sys.path.insert(0, {here!r})
from {module} import main
sys.exit(main())
"""


def main() -> int:
    BIN.mkdir(parents=True, exist_ok=True)
    launcher = (Path(distlib.__file__).parent / "t64.exe").read_bytes()
    shebang = f'#!"{sys.executable}"\r\n'.encode("utf-8")
    for module in ("codex_gate", "claude_gate"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("__main__.py", STUB.format(here=str(HERE), module=module))
        target = BIN / f"{module}.exe"
        target.write_bytes(launcher + shebang + buf.getvalue())
        print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
