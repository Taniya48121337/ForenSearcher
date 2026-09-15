"""兼容壳：原 orchestration.py 已拆入 gia/ 包（2026-09-15）。新代码请直接 import foren_searcher.gia。"""

from .gia._common import _CLAIM_REFERENCE, _EVIDENCE_LIKE_REFERENCE, _EVIDENCE_REFERENCE, _LOGGER, _error_payload, _record_runtime_event  # noqa: F401
from .gia.global_view import GlobalView, GlobalViewBuilder, bound_global_view  # noqa: F401
from .gia.rounds import RoundCoordinator, WorkspaceOutcome  # noqa: F401
from .gia.runner import GiaPolicyTimeout, GiaRunner, GiaTurnLimit  # noqa: F401
from .gia.runtime import InitialMedia, InvestigationLimit, InvestigationResult, InvestigationRuntime, RuntimeClosedError  # noqa: F401
