"""INSPECT 后端数据契约和默认本地实现的公共导出。"""

from .base import *  # noqa: F401,F403
from .local import LocalInspectBackend

__all__ = ["LocalInspectBackend"]
