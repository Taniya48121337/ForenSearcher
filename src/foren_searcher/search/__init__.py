"""SEARCH 子系统的稳定公共导出。

外部代码应从本模块导入请求、响应、策略、服务和工厂，避免依赖内部目录结构。
"""

from .contracts import (
    CandidateBatch,
    CandidateSource,
    OpenedMaterial,
    OpenedSource,
    OpenSourceRequest,
    SearchError,
    SearchFailure,
    SearchQueryRequest,
    SearchResponse,
    parse_search_request,
)
from .factory import (
    create_free_search_service,
    create_iqdb_reverse_image_search_service,
    create_search_service,
    create_searxng_search_service,
    create_serpapi_search_service,
    create_yandex_reverse_image_search_service,
)
from .service import SearchService
from .policy import SearchPolicy
from .providers.base import SearchProvider

__all__ = [
    "CandidateBatch", "CandidateSource", "OpenedMaterial", "OpenedSource",
    "OpenSourceRequest", "SearchError", "SearchFailure", "SearchPolicy",
    "SearchProvider", "SearchQueryRequest", "SearchResponse", "SearchService",
    "create_free_search_service", "create_iqdb_reverse_image_search_service",
    "create_search_service", "create_searxng_search_service", "create_serpapi_search_service",
    "create_yandex_reverse_image_search_service", "parse_search_request",
]
