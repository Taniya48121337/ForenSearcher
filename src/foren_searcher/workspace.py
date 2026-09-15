"""兼容壳：原 workspace.py 已移到 hi/workspace.py（2026-09-15 拆包）。本模块把自己替换成新模块对象，旧 import 路径继续可用。"""

import sys as _sys

from . import hi as _pkg  # noqa: F401
from .hi.workspace import *  # noqa: F401,F403

_sys.modules[__name__] = _sys.modules["foren_searcher.hi.workspace"]
