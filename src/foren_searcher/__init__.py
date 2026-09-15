"""ForenSearcher 的公共导入入口。

使用者通常只需要写 ``from foren_searcher import ...``，不必记住每个类实际位于
哪个内部文件。本文件不执行业务逻辑，只把稳定的公共类型和工厂重新导出。

阅读顺序：先看契约数据类型，再看 Store/Memory，随后看编排 Runtime 和两个工具，
最后看可选的本地模型适配器。``__all__`` 是允许外部依赖的公开名称清单。
"""

# 第一组是跨模块传递的数据结构，以及把模型 JSON 转成这些结构的解析函数。
from .contracts import (
    Evidence,
    EvidenceCard,
    Finding,
    FindingDraft,
    GiaDecision,
    GiaFinishDecision,
    GiaInvestigateDecision,
    InvestigationTask,
    HiActionDecision,
    HiDecision,
    WorkspaceReport,
    WorkspaceReportDraft,
    WorkspaceTask,
    parse_hi_decision,
    parse_workspace_report,
    parse_gia_decision,
)
# 编号分配始终由 Harness 控制，模型不能自行生成真实主键。
from .ids import assign_workspace_tasks, format_workspace_id
# INSPECT 包负责处理已经进入 Store 的图片、音频、视频和文档。
from .inspect import (
    InspectCall,
    InspectPolicy,
    InspectResponse,
    InspectService,
    create_inspect_service,
    parse_inspect_call,
)
# Store 保存数据库记录和 Artifact；Memory 生成受限的模型可见视图。
from .store import ArtifactIntegrityError, GlobalStore, StoreHealth
from .memory import GlobalView, MemoryPolicy, RecordAccess
# orchestration 是顶层控制流：GIA 规划轮次，RoundCoordinator 运行 Workspace。
from .orchestration import (
    GiaRunner,
    GiaPolicyTimeout,
    GiaTurnLimit,
    InitialMedia,
    InvestigationResult,
    InvestigationRuntime,
    InvestigationLimit,
    RoundCoordinator,
    RuntimeClosedError,
    WorkspaceOutcome,
)
# Protocol 只规定模型适配器需要实现的方法，不绑定具体模型厂商。
from .policies import GiaPolicy, HiPolicy
# local_models 是可选 Transformers 实现；未安装模型依赖时核心框架仍可导入。
from .local_models import (
    LocalModelBusyError,
    LocalModelClosedError,
    LocalModelConfig,
    LocalModelContextError,
    LocalModelError,
    LocalModelOutputError,
    LocalModelMetrics,
    LocalModelUnavailableError,
    LocalPolicyBundle,
    TransformersGiaPolicy,
    TransformersHiPolicy,
    TransformersJsonGenerator,
    create_qwen_local_policies,
    download_inspect_demo_models,
    download_qwen_local_models,
    extract_json_object,
)
# workspace 包含单个 HI 的 Working Memory、Harness 和多轮执行循环。
from .workspace import (
    ToolObservation,
    WorkspaceHarness,
    WorkspaceHistoryEntry,
    WorkspaceMemory,
    WorkspaceRunner,
    WorkspaceTurnLimit,
)

# 下面只列“正式公共 API”。没有出现在这里的内部辅助函数可以随实现调整。
__all__ = [
    # 核心记录和解析契约。
    "Evidence",
    "EvidenceCard",
    "Finding",
    "FindingDraft",
    # 持久化与完整性检查。
    "ArtifactIntegrityError",
    "GlobalStore",
    # GIA/HI 决策及顶层调查编排。
    "GiaDecision",
    "GiaFinishDecision",
    "GiaInvestigateDecision",
    "GiaPolicy",
    "GiaRunner",
    "GiaPolicyTimeout",
    "GiaTurnLimit",
    "GlobalView",
    "HiActionDecision",
    "HiDecision",
    "HiPolicy",
    # 本地模型配置、异常、指标及策略适配器。
    "LocalModelConfig",
    "LocalModelBusyError",
    "LocalModelClosedError",
    "LocalModelContextError",
    "LocalModelError",
    "LocalModelOutputError",
    "LocalModelMetrics",
    "LocalModelUnavailableError",
    "LocalPolicyBundle",
    # 调查输入、输出和运行限制。
    "InitialMedia",
    "InvestigationTask",
    "InvestigationResult",
    "InvestigationRuntime",
    "InvestigationLimit",
    # INSPECT 的请求、响应、服务与工厂。
    "InspectCall",
    "InspectResponse",
    "InspectService",
    "InspectPolicy",
    "create_inspect_service",
    "parse_inspect_call",
    # 模型记忆视图与只读记录访问。
    "MemoryPolicy",
    "RecordAccess",
    "RoundCoordinator",
    "RuntimeClosedError",
    "StoreHealth",
    # 本地模型实现和 Workspace 执行组件。
    "ToolObservation",
    "TransformersGiaPolicy",
    "TransformersHiPolicy",
    "TransformersJsonGenerator",
    "WorkspaceReport",
    "WorkspaceReportDraft",
    "WorkspaceTask",
    "WorkspaceHarness",
    "WorkspaceHistoryEntry",
    "WorkspaceMemory",
    "WorkspaceOutcome",
    "WorkspaceRunner",
    "WorkspaceTurnLimit",
    # 编号、工厂、下载器和 JSON 解析函数。
    "assign_workspace_tasks",
    "format_workspace_id",
    "create_qwen_local_policies",
    "download_qwen_local_models",
    "download_inspect_demo_models",
    "extract_json_object",
    "parse_hi_decision",
    "parse_workspace_report",
    "parse_gia_decision",
]
