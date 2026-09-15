# Step 1 — Decompose the Novelty

Timestamp: 2026-08-16 Asia/Shanghai

## Research problem

在不能生成新视频、也无法进行大规模人工审核的条件下，如何利用已有 source–suspect 视频数据与隐藏标注，自动构造高质量 SFT agent 轨迹，使模型能够把多工具证据转化为完整、可验证的细粒度视频篡改声明，并学会修正错误证据、拒绝不支持的声明和避免过早停止。

## Proposed novelty

Claim-Causal Mutation-Tested Trajectory Compilation：为每条最终 manipulation claim 建立 action–observation–state–claim 依赖图，反向提取最小充分证据切片；对来源、区间、证据绑定、方向与停止状态施加 claim-conditioned 语义变异并重放后缀；用变异是否导致撤回、修正或重取证来检验证据的因果性，并将被杀死的变异自动编译为匹配的 repair SFT 轨迹。

## Four atomic axes

- **Problem framing**：固定的已有 source–suspect 视频与真实工具环境；输出不是单一答案，而是带时间区间、操作类型、实体、方向和 evidence IDs 的可验证 manipulation claims；评价包括声明完整性、证据绑定、错误恢复和停止行为。
- **Core mechanism**：claim-specific provenance graph + backward/dynamic slicing + semantic mutation testing + suffix replay + matched repair-trajectory compilation。
- **Key insight**：结果正确或整条工具轨迹有用，并不能证明每条声明真正依赖所引用证据；只有逐声明破坏关键依赖并观察系统撤回/修复，才能识别“装饰性证据”，且这些最小反事实天然形成高价值错误恢复监督。
- **Application domain**：来源溯源条件下的视频篡改调查与多模态工具 agent；机制可能推广到其他证据驱动的 agent 调查任务。
