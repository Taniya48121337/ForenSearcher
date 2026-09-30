"""校准规则 R1/R2/R4 反例注入（假接受率 / 真接受率）的命令行入口：参数与输出见 src/foren_searcher/datagen/verify/calibrate_rule_audit.py 的模块说明与 docs/核验器量化工具.md。

    python scripts/calibrate_rule_audit.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foren_searcher.datagen.verify.calibrate_rule_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
