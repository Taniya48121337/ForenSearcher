"""候选来源打开器包：负责把 URL 转换成可提交的 OpenedMaterial。"""

from .dispatcher import SourceDispatcher
from .http import SafeHttpClient

__all__ = ["SafeHttpClient", "SourceDispatcher"]
