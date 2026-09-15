"""GIA 侧各模块共用的小工具：运行事件记录、异常打包、编号正则、日志。"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from ..store.global_store import GlobalStore

_EVIDENCE_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])E[1-9][0-9]*(?![A-Za-z0-9_])")

_EVIDENCE_LIKE_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])E[0-9]+(?![A-Za-z0-9_])")

_CLAIM_REFERENCE = re.compile(r"C(?:0|[1-9][0-9]*)\Z")

_LOGGER = logging.getLogger(__name__)



def _error_payload(exc: BaseException) -> dict[str, Any]:
    """把任意异常压缩成可持久化、最多 2000 字符的错误对象。"""

    message = str(exc)
    return {
        "type": type(exc).__name__,
        "message": message[:2000],
        **({"message_truncated": True} if len(message) > 2000 else {}),
    }


def _record_runtime_event(
    store: GlobalStore,
    *,
    actor_role: str,
    actor_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """尽力写运行轨迹；轨迹失败不能覆盖真正的调查结果或异常。"""

    try:
        store.record_event(
            actor_role=actor_role,
            actor_id=actor_id,
            event_type=event_type,
            payload=payload,
        )
    except Exception:
        # 数据主流程与审计附加记录分离，日志保留事件身份供运维修复。
        _LOGGER.exception(
            "could not persist a runtime trajectory event",
            extra={"actor_role": actor_role, "actor_id": actor_id, "event_type": event_type},
        )

