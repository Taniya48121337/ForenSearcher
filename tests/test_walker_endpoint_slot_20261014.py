"""walker 端点槽后端（10-14）：在子进程里跑 scripts/_test_walker_endpoint_slot.py（它会往 sys.modules 放 foren_searcher 桩）。"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_endpoint_slot_backend() -> None:
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "_test_walker_endpoint_slot.py")], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "latch stops the run" in proc.stdout
