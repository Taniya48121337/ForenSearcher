# UTQR-9 可执行评分说明

这份说明把统一 rubric 映射到可重复运行的证据，同时保留真人判断边界。它的目标不是让生成器轻易得到 100 分，而是先证明 92 分自动核心，再由独立真人决定最后 8 分与 H8。

## 分数结构

| 项目 | 权重 | 自动证据 |
|---|---:|---|
| C1 结果、绑定、终止正确 | 13 | 每步真实执行；最终只有一个 supported、committed、evidence-bound claim；无开放义务 |
| C2 身份、来源、角色、范围 | 12 | 最终 support ID 都存在且 active；无 student-visible 私有信息；完整终止 |
| C3 因果证据敏感性 | 12 | clean control、moved defect、remove fresh evidence 三组反事实全部通过 |
| C4 无关字段保持 | 10 | 只修复依赖字段，independent fields 保持不变 |
| C5 工具适配与信息增益 | 10 | 每步在调用前 eligible、调用后产生阶段对应状态增量、无完全相同工具参数重复 |
| C6 未公开缺陷诊断 | 10 | 先 challenge，再真实测量、局部 reopen、binding、commit、stop，顺序不可倒置 |
| C7 新证据吸收与局部恢复 | 12 | fresh evidence 被最终绑定消费，且完整恢复后才能 stop |
| C8 必要时显式修订 claim | 7 | `revise_claim_fields` 与 binding、commit 分步执行；不需要修订时最终状态仍须唯一一致 |
| C9 真人式自然推理 | 8 | 自动语言审计只提供 proxy；最终分必须由两位盲审者给出 |
| C10 SFT 目标卫生与学习价值 | 6 | 只监督最后一个已执行 assistant action；无 compiled answer 或 private intervention；每行有正学习权重 |

因此自动核心满分是 92，而不是 100。C9 未经真人评分时保持 `null`；H8 未完成时保持 `pending`。即使机器 proxy 高于 4/5，也不能把轨迹称为真人盲审通过。

## 真人六维 proxy 的上限

机器只能辅助检查 prospective reasoning、visible grounding、belief update、method fit 和 locality，单维上限设为 4.5/5；naturalness 上限为 4.0/5。它们用于定位待审轨迹，不计入 H8。

## 为什么不容易通过

评分器对七种不依赖具体案例的负向突变做了 falsification：提前终止、反事实不敏感、越界字段修改、不消费新证据、私有信息泄漏、空推理以及没有实质状态增量的动作。当前 7/7 都降低自动分并使 H1–H7 汇总失败；最小降分 1.111，最大降分 44.556。

## 当前扩展批

- VideoSham：16 个 episode，自动核心 mean/p10/min 均为 92/92。
- VCSL：18 个 episode，自动核心 mean/p10/min 均为 92/92。
- VidDiffBench：12 个 episode，自动核心 mean/p10/min 均为 92/92。
- 319/319 action 都在执行前 eligible、执行后产生阶段对应的实质状态增量。
- C9 与 H8 仍未完成。

机器可读逐 episode 证据在 `recovery_batch_candidate_v9_expanded_checkpoint_portfolio/utqr_episode_scorecards.json`，负向测试在同目录的 `utqr_falsification_report.json`。
