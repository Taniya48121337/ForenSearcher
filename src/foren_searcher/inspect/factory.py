"""INSPECT 默认服务装配工厂。"""

from __future__ import annotations

from typing import Any

from ..store.resolver import EvidenceResolver
from ..store import GlobalStore
from .backends.local import LocalInspectBackend
from .acquisition import DirectMediaAcquirer
from .dispatcher import InspectDispatcher
from .policy import InspectPolicy
from .processors import create_processors
from .service import InspectService


def create_inspect_service(*, store: GlobalStore, policy: InspectPolicy | None = None, backend: Any | None = None, resolver: EvidenceResolver | None = None, acquirer: Any | None = None, enabled_operations: frozenset[str] | None = None) -> InspectService:
    """装配解析器、本地后端、线程分发器和可选视频获取器。"""

    policy = policy or InspectPolicy()
    # 只有工厂自行创建的资源才由 Service 负责关闭。
    owns_backend = backend is None
    owns_acquirer = acquirer is None
    backend = backend or LocalInspectBackend()
    resolver = resolver or EvidenceResolver(store)
    acquirer = acquirer or DirectMediaAcquirer()
    processors = create_processors(backend)
    dispatcher = InspectDispatcher(
        backend, processors, maximum_workers=policy.backend_workers
    )
    owned = (dispatcher,) + ((backend,) if owns_backend else ()) + ((acquirer,) if owns_acquirer else ())
    return InspectService(store=store, resolver=resolver, dispatcher=dispatcher, policy=policy, acquirer=acquirer, owned_resources=owned, enabled_operations=enabled_operations)
