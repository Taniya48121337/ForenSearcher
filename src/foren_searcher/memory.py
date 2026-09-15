"""兼容壳：原 memory.py 已拆为 store/records.py（共享记录访问）、hi/memory.py（证据视图）、gia/global_view.py（全局视图）。

新代码请直接从这三个位置 import。
"""

from .store.records import *  # noqa: F401,F403
from .store.records import (  # noqa: F401
    _UNSEGMENTED_CHARACTERS, _UNSEGMENTED_TEXT, _WORD_CHARACTER, _candidate_card, _clip_text, _evidence_card,
    _evidence_index, _evidence_index_card, _evidence_page, _finding_card, _frame_coverage, _relevance, _report_card,
    _report_detail, _task_context, _tokens, _visual_observations, _workspace_action_observations, _workspace_task_card,
)
from .hi.memory import EvidenceViewBuilder  # noqa: F401
from .gia.global_view import GlobalView, GlobalViewBuilder, bound_global_view  # noqa: F401
