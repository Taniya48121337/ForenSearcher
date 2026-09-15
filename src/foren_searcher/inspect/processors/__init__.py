"""INSPECT 操作处理器：把稳定 operation 映射到可替换 Backend 能力。"""

from .base import BackendOperationProcessor, InspectProcessor, create_processors

__all__ = ["BackendOperationProcessor", "InspectProcessor", "create_processors"]
