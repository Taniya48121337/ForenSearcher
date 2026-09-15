# Step 1 — Decompose the Novelty

Timestamp: 2026-08-26 00:32:49 +08:00

## Research problem

以 Harness-1 为运行底座，构建可处理异构视频核查任务并能迁移到新任务的统一 Deep Research Agent。

## Proposed novelty

把 Harness-1 的单一 Working Memory 扩展为动态 Working Memory Tree：根节点负责总任务；模型动态派生局部调查任务；每个节点是可独立进行多轮工具调用、证据整理和预算管理的 Harness 工作区；父到子的边只表示任务派生；证据通过全局共享池复用。

## Four atomic axes

- **Problem framing**：跨来源检索、片段定位、差异分析、篡改归因和事实核查等异构视频调查，并评价对未见任务的快速迁移。
- **Core mechanism**：动态局部任务树；节点是完整 Harness 工作区；单向任务派生边；全局共享证据池；模型决定何时及派生什么任务。
- **Key insight**：把需要隔离的调查上下文从一个扁平工作记忆中拆成可独立运行、可调度、可共享证据的工作区，避免把任务类型和证据关系写死。
- **Application domain**：异构视频核查，同时希望作为通用证据调查机制迁移到新核查任务。
