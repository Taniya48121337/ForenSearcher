"""共享证据库：SQLite 记录 + Artifact 文件目录、记录访问/卡片渲染、E→文件解析器。

GIA 与 HI 都通过这里读写证据；这是两个角色之间唯一的共享状态。
子模块：global_store（GlobalStore 本体）、records（RecordAccess、MemoryPolicy、卡片）、resolver（EvidenceResolver）。
"""

from .global_store import *  # noqa: F401,F403
from .records import *  # noqa: F401,F403
from .resolver import *  # noqa: F401,F403
