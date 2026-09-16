"""三层核验器（DESIGN.md 第 8 节的"审计员"）。

大白话：以前的评审员拿着演员当时看到的片段打 0–4 分，而且和教师是同一个人。现在换成三层、
逐条 Finding 核对、每条错误都从 errors.py 的 22 种里挑名字：

- 第一层 exec_checks：纯程序、不用模型。引用的 E 存不存在、看没看得见、报告空不空、数字和实体
  能不能在所引 E 的原文里找到、状态跟案卷要求对不对、来源有没有踩排除名单。能判死的直接判死
  （hard），拿不准的打标记（flag）交给第二层。
- 第二层 anchored：模型逐条 Finding × 逐条所引 E 重读原文（从 Store 取，不是演员看到的片段），
  给三态判定 supported / contradicted / not_found，并且必须给出原文引语；引语会被程序回查是否
  真的出现在原文里，查不到就按 not_found 处理。
- 第三层 sufficiency：模型看整份报告对照任务要求和必需证据清单，判断"该拿的拿齐没有"、
  COMPLETED / INCOMPLETE 报得对不对。

产出是 Verdict（verdict.py）：错误列表、处置（positive / negative / uncertain）、被污染的 Finding、
核验器与规格指纹。核验模型必须与教师分离（verifier ≠ teacher）；本包只定义 JsonJudge 协议，
不绑定具体模型。
"""

from .anchored import AnchoredResult, verify_anchored
from .context import DictLookup, EvidenceLookup, StoreLookup
from .exec_checks import exec_checks
from .sufficiency import verify_sufficiency
from .verdict import Disposition, JsonJudge, Verdict, verify_report

__all__ = ["AnchoredResult", "DictLookup", "Disposition", "EvidenceLookup", "JsonJudge", "StoreLookup", "Verdict",
           "exec_checks", "verify_anchored", "verify_report", "verify_sufficiency"]
