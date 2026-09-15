"""SEARCH 的运行时资源上限。

这些值约束网络等待、候选数量、并发数和下载大小，是实现层的默认保护参数；它们
不会被写成模型必须依赖的方法常量，所以部署者可以按环境调整。
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """集中保存 SEARCH 所有可配置预算，避免魔法数字散落在实现中。"""

    # 时间字段单位统一为秒。
    provider_timeout_s: float = 30.0
    source_timeout_s: float = 20.0
    dynamic_timeout_s: float = 45.0
    single_source_total_s: float = 60.0
    search_total_s: float = 180.0
    initial_candidates: int = 5
    maximum_candidates: int = 10
    target_sources: int = 5
    maximum_provider_requests: int = 2
    source_concurrency: int = 3
    dynamic_concurrency: int = 1
    maximum_redirects: int = 5
    maximum_document_bytes: int = 20 * 1024 * 1024
    maximum_image_bytes: int = 15 * 1024 * 1024
    maximum_html_bytes: int = 8 * 1024 * 1024
    user_agent: str = "ForenSearcher/0.1 (+research source acquisition)"

    def __post_init__(self) -> None:
        """在服务启动时一次性拒绝无穷、非正数和非法类型配置。"""

        # bool 在 Python 中也是 int，因此每组校验都要显式排除它。
        for name in (
            "provider_timeout_s", "source_timeout_s", "dynamic_timeout_s",
            "single_source_total_s", "search_total_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in (
            "initial_candidates", "maximum_candidates", "target_sources",
            "maximum_provider_requests", "source_concurrency", "dynamic_concurrency",
            "maximum_document_bytes", "maximum_image_bytes", "maximum_html_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.maximum_redirects, int) or isinstance(self.maximum_redirects, bool) or self.maximum_redirects < 0:
            raise ValueError("maximum_redirects must be a non-negative integer")
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("user_agent must be non-empty")
