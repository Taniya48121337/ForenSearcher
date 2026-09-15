"""SEARCH 服务装配工厂。

Service 只依赖抽象 Provider 和 Opener；本文件负责选择具体适配器并把它们接起来。
因此业务流程可独立测试，部署入口也不必重复创建 HTTP 客户端和关闭回调。
"""

from __future__ import annotations

from typing import Literal

from ..store.resolver import EvidenceResolver
from ..store import GlobalStore
from .openers.browser import PlaywrightRenderer
from .openers.dispatcher import SourceDispatcher
from .openers.document import DocumentSourceOpener
from .openers.http import SafeHttpClient
from .openers.image import ImageSourceOpener
from .openers.video import VideoSourceOpener
from .policy import SearchPolicy
from .providers.base import SearchProvider
from .providers.ddgs import DdgsProvider
from .providers.iqdb_reverse import IqdbReverseImageProvider
from .providers.routing import InputRoutingSearchProvider
from .providers.searxng import SearXngProvider
from .providers.serpapi import SerpApiProvider
from .providers.yandex_reverse import YandexReverseImageProvider
from .service import SearchService


def create_search_service(
    *, store: GlobalStore, provider: SearchProvider, policy: SearchPolicy | None = None,
    allow_private_network: bool = False, enable_dynamic_pages: bool = False,
    allow_image_uploads: bool = True, resolver: EvidenceResolver | None = None,
    close_provider: bool = True,
) -> SearchService:
    """用给定 Provider 装配完整的 HTTP、文档、图片和视频打开链。"""

    policy = policy or SearchPolicy()
    # 所有打开器共享一个带 SSRF 防护和大小限制的 HTTP 客户端。
    http = SafeHttpClient(policy, allow_private_network=allow_private_network)
    renderer = PlaywrightRenderer(http, policy) if enable_dynamic_pages else None
    document = DocumentSourceOpener(http, policy, renderer)
    dispatcher = SourceDispatcher(document, ImageSourceOpener(http, document, policy), VideoSourceOpener(http, policy))

    async def close_all() -> None:
        """先关闭 HTTP，再按所有权约定关闭 Provider。"""

        try:
            await http.aclose()
        finally:
            if close_provider:
                provider_close = getattr(provider, "aclose", None)
                if callable(provider_close):
                    await provider_close()

    return SearchService(store=store, provider=provider, opener=dispatcher, policy=policy,
        close_callback=close_all, allow_image_uploads=allow_image_uploads, resolver=resolver)


def create_serpapi_search_service(
    *, store: GlobalStore, api_key: str, policy: SearchPolicy | None = None,
    enable_dynamic_pages: bool = False, allow_image_uploads: bool = True,
) -> SearchService:
    """创建使用 SerpAPI 的付费/密钥型搜索服务。"""

    policy = policy or SearchPolicy()
    return create_search_service(store=store, provider=SerpApiProvider(api_key, policy.provider_timeout_s),
        policy=policy, enable_dynamic_pages=enable_dynamic_pages, allow_image_uploads=allow_image_uploads)


def create_searxng_search_service(
    *, store: GlobalStore, base_url: str, policy: SearchPolicy | None = None,
    language: str = "all", safe_search: int = 1, enable_dynamic_pages: bool = False,
) -> SearchService:
    """创建使用 SearXNG 的免费文本起源网页、图片和视频搜索服务。"""
    policy = policy or SearchPolicy()
    provider = SearXngProvider(
        base_url, policy.provider_timeout_s, language=language, safe_search=safe_search,
    )
    return create_search_service(
        store=store,
        provider=provider,
        policy=policy,
        enable_dynamic_pages=enable_dynamic_pages,
        allow_image_uploads=False,
    )


def create_yandex_reverse_image_search_service(
    *, store: GlobalStore, policy: SearchPolicy | None = None,
    enable_dynamic_pages: bool = False, allow_image_uploads: bool = True,
) -> SearchService:
    """创建 Yandex 反向图片搜索服务。"""

    policy = policy or SearchPolicy()
    return create_search_service(
        store=store,
        provider=YandexReverseImageProvider(policy.provider_timeout_s),
        policy=policy,
        enable_dynamic_pages=enable_dynamic_pages,
        allow_image_uploads=allow_image_uploads,
    )


def create_iqdb_reverse_image_search_service(
    *, store: GlobalStore, policy: SearchPolicy | None = None,
    index: Literal["artwork", "3d"] = "artwork", allow_image_uploads: bool = True,
) -> SearchService:
    """创建适合插画或 3D 图像的 IQDB 反向搜索服务。"""

    policy = policy or SearchPolicy()
    return create_search_service(
        store=store,
        provider=IqdbReverseImageProvider(policy.provider_timeout_s, index=index),
        policy=policy,
        allow_image_uploads=allow_image_uploads,
    )


def create_free_search_service(
    *, store: GlobalStore, searxng_base_url: str | None = None, policy: SearchPolicy | None = None,
    reverse_engine: Literal["yandex", "iqdb-artwork", "iqdb-3d", "none"] = "yandex",
    language: str = "all", safe_search: int = 1, enable_dynamic_pages: bool = False,
    allow_image_uploads: bool = True,
) -> SearchService:
    """组合免费文本搜索和反向图片后端；可指定自建 SearXNG。"""
    policy = policy or SearchPolicy()
    text_provider: SearchProvider
    if searxng_base_url:
        text_provider = SearXngProvider(searxng_base_url, policy.provider_timeout_s,
            language=language, safe_search=safe_search)
    else:
        text_provider = DdgsProvider(min(policy.provider_timeout_s, 12.0))
    if reverse_engine == "none":
        return create_search_service(store=store, provider=text_provider, policy=policy,
            enable_dynamic_pages=enable_dynamic_pages, allow_image_uploads=False)
    # 文本与图片输入由路由 Provider 分发，外层 Service 无需知道差别。
    if reverse_engine == "yandex":
        image_provider: SearchProvider = YandexReverseImageProvider(policy.provider_timeout_s)
    else:
        image_provider = IqdbReverseImageProvider(
            policy.provider_timeout_s, index="3d" if reverse_engine == "iqdb-3d" else "artwork",
        )
    return create_search_service(
        store=store,
        provider=InputRoutingSearchProvider(
            text_provider=text_provider,
            image_provider=image_provider,
        ),
        policy=policy,
        enable_dynamic_pages=enable_dynamic_pages,
        allow_image_uploads=allow_image_uploads,
    )
