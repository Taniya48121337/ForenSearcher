"""对抗探针集（S4 四条路径）的命令行入口：参数与输出见 src/foren_searcher/datagen/verify/adversarial.py 的模块说明与 docs/核验器量化工具.md。

    python scripts/adversarial_probes.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foren_searcher.datagen.verify.adversarial import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
