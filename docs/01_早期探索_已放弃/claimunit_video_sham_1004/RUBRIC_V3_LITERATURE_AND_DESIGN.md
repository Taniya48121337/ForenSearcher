# UTQR-3：统一可执行轨迹 Rubric 与 Decision-Sufficient SFT

更新时间：2026-08-20

## 1. 这次优化解决的不是“分数不够高”，而是“高分是否真的可训练”

UTQR-2 已经能拒绝错误答案、伪造 EvidenceAtom、不可执行动作、冗余步骤和错误 repair，但本轮全池审计发现两个旧 rubric 没有充分惩罚的问题：

1. teacher 实际决策使用压缩的当前 harness state，生产 SFT 却重复保存多个历史 state，形成训练/生成上下文错位；
2. 历史 VideoSham/VCSL/VidDiff 轨迹的本地 locator 可能残留在完整 transcript 中，尽管这些 locator 对决策没有语义价值。

因此 UTQR-3 增加 H7/Q9，并引入 `decision_sufficient_state_v1`。完整 transcript 继续作为本地审计证据；真正用于 SFT 的每一行只保留：

- 原始任务语义，但移除重复 tool registry；
- 唯一、权威的 pre-action harness state；
- 有序的历史工具名、参数与结果摘要；
- 最新 observation 的状态摘要；
- 完全不变的 assistant reasoning 与单一 tool-call target。

媒体/特征 locator 被归一化为 `source/suspect`、`query/reference` 或 `video_a/video_b`。这不是删证据：EvidenceAtom、数值测量、字段绑定、open obligations、eligible tools 和 repair mutation 都仍在权威 state 中。

## 2. 文献依据与我们真正借用的机制

| 工作 | 可核验机制 | 对本项目的影响 |
|---|---|---|
| [Harness-1](https://arxiv.org/abs/2606.02373) | harness 维护 compact evidence links、verification records、compressed/deduplicated observations 和 budget-aware context rendering | 不再让 SFT 模仿不断增长且重复的 transcript；policy 只学习语义决策 |
| [InfiAgent](https://arxiv.org/abs/2601.03204) | 每一步由 workspace state snapshot 与有限近期动作重建有界上下文 | compact row 使用一个当前 state 加小型 action history |
| [Dependency-Aware Trajectory Refinement](https://openreview.net/pdf?id=jcfAqlBva7)（投稿版本） | 把轨迹表示为 round-level dependency DAG，裁去未被下游消费的步骤；论文也显示过度合并会造成分布偏移 | 本项目继续用反事实 replay 判定 critical path；只删除重复 state，不合并具有不同 downstream consumer 的动作 |
| [APIGen](https://arxiv.org/abs/2406.18518) | format、真实 function execution、semantic verification 三层验证 | H2/H3/H7 分别覆盖可执行、证据语义、训练上下文保真 |
| [APIGen-MT](https://arxiv.org/abs/2504.03601) | blueprint-to-trajectory，两阶段生成；review committee 与迭代反馈验证多轮一致性 | 先产生可验证 state/action blueprint，再允许 teacher 填写自然语言决策，而不是自由生成整条“故事” |
| [ToolACE](https://arxiv.org/abs/2409.00920) | rule-based 与 model-based 双层 verification | 本项目以确定性 checker 为主，语义 judge 只适合风险采样，不能覆盖或替代执行证据 |
| [AgentTrek](https://arxiv.org/abs/2412.09605) | guided replay 在真实环境执行，并由 evaluator 筛选 | 对已有视频只做真实工具 replay，不让 teacher 虚构视频或 observation |
| [VideoSearcher](https://arxiv.org/abs/2607.02927) | 将 tool-invocation optimization 与 answer accuracy 分开 | Rubric 分开 Q1 outcome 与 Q3/Q4 tool policy；后续训练也应分别报告 action 与 answer 指标 |
| [Sequent-Prover](https://openreview.net/pdf/0a9e087289d6e4896987774b9757c730c736a6a9.pdf)（投稿版本） | 正确答案但 verifier support 弱的轨迹是 Silver，而非 Gold；tool observation 留在条件上下文但不计预测损失 | 本项目 Gold 必须同时 outcome 正确、EvidenceAtom 闭合、工具已执行；tool message 不进入 loss |
| [GLEAN](https://openreview.net/forum?id=l82IbMOAzp)（workshop 版本） | guideline-grounded step evidence、校准不确定性、对不确定案例主动追加 differential checks | 大批量时不对所有案例等成本 judge；按低置信、冲突、边界 margin 和新 route 做风险采样 |

## 3. UTQR-3 的九个质量维度

UTQR-3 保持 100 分，但增加一个不可被总分补偿的 H7：

| 维度 | 分值 | 人类审查问题 |
|---|---:|---|
| Q1 outcome/localization | 17 | 最终判断和定位真的对吗，而不是只覆盖到答案？ |
| Q2 evidence grounding | 16 | 每个字段能追到真实执行产生的 EvidenceAtom 吗？ |
| Q3 state-aware planning | 11 | 当时的状态真的允许并需要这个动作吗？ |
| Q4 causal utility | 12 | 删除该步后，合法终态是否会改变？工具 observation 是否真正改变后续？ |
| Q5 grounded rationale | 13 | reasoning 是否说清“观察到了什么—还缺什么—为何此动作补上”？ |
| Q6 typed repair | 11 | dropout、binding invalidation、decision reopen 是否采用不同且最小的恢复路径？ |
| Q7 uncertainty/conflict | 6 | 冲突、低置信和 abstain 是否被诚实处理？ |
| Q8 SFT hygiene | 7 | 每行是否只训练一个已执行 action，tool feedback 是否只作条件？ |
| Q9 decision-context fidelity | 7 | 压缩后是否保留同一 pre-action state 与 target，同时删除重复 state 和 private locator？ |

Gold 仍要求所有 hard gates 通过、总分至少 92、每个关键维度过最低线。Q9 低于 0.95 时不能靠 Q1 高分补偿。

## 4. 本轮实测

在当前三数据集 59 个 Gold case、600 个 per-action SFT 行上：

- 600/600 compact row 通过 exact target/state/action-order/eligibility/evidence-reference 检查；
- private locator、媒体文件名、URL、oracle/verifier 字段命中为 0；
- 模型可见字符从 17,791,569 降到 7,132,118，减少 59.9%；
- context ratio：均值 0.471，P50 0.416，P90 0.703；
- UTQR-3 重新评分：VideoSham 17/17、VCSL 21/21、VidDiffBench 21/21 Gold；
- 685/685 个 minimal-edit hard negative 被拒绝，覆盖 13 个家族；新增家族专门攻击 private locator、pre-state 不允许的 target action、以及尚不可见的 EvidenceAtom 引用。

这些结果证明“压缩没有改变当前已知决策”，但不证明下游模型一定更好。真正的大批量生产声明仍需要 SFT 消融：full transcript、naive truncation、decision-sufficient state 三组在相同 base model、相同 token budget 下比较 held-out task success、action validity、evidence grounding、repair success、tokens 和 stuck rate。

## 5. 可辩护的创新边界

单独的 state externalization、trajectory pruning、execution verification 或 tool-use SFT 都已有相关工作，因此不能把任何一个单独声称为新颖。

目前更可辩护的组合创新是：

> 面向异构视频证据任务的 **typed repair + EvidenceAtom provenance + counterfactual critical-path audit + privacy-normalized decision-state distillation**；每条 compact SFT 都带有从完整可执行 transcript 到唯一 pre-action state 和不变 target 的机器可验证保真证书。

针对性检索没有发现同时采用 `evidence_dropout / binding_invalidation / decision_reopen` 三类状态破坏、字段级 EvidenceAtom 绑定、逐步删除反事实和隐私 locator 归一化的同类轨迹生成工作。这个结论是截至 2026-08-20 的检索结果，不等于完成正式系统综述，也不能替代投稿前再次查新。

## 6. 尚未满足的量产证据

1. 还没有训练一个 student 来证明 compact SFT 优于 full transcript 或普通截断；
2. 每个 VideoSham teacher Gold 能力族只有 1–4 例；
3. 多次独立 rollout 的 case coverage 仍太低；
4. 现有 hard negatives 很强，但仍主要是已知 failure family；
5. 还需要风险分层的人类盲审，尤其检查“动作正确但 reasoning 解释错”的 Silver 轨迹。

因此当前可以称为“可执行、可审计、可直接用于受控 SFT 实验的数据版本”，不能称为“已经实证证明可无审查大批量生产”。

## 7. Student 消融已就绪但尚未长跑

`student_context_ablation_v2` 固定同一批 600 行、同一 472/64/64 split 和完全相同的 assistant target，只改变输入上下文：

1. `full_state_normalized_v1`：保留重复的完整状态历史，仅做 locator 隐私归一化；
2. `naive_token_tail_v1`：以 compact 行的 token 数为逐行上限，从左侧机械截断历史；
3. `decision_sufficient_state_v1`：唯一权威 pre-action state、动作/结果摘要和不变 target。

Qwen tokenizer 统计显示：full 输入 P50/P90/max 为 2,930/10,907/24,591 token，125/600 行超过 8,192；compact 为 2,774/3,665/6,836，全部在 8,192 内。naive-tail 有 383 行确实发生截断，这些行平均使用各自 compact 预算的 99.98%，因此不会因为少给 token 而形成明显不公平比较。三臂 private locator/oracle 命中均为 0。

本机 Qwen3-VL-8B-Instruct 4-bit LoRA 已完成 1 样本、1 优化步 smoke：assistant-target-only loss=3.354，约 21.8M 可训练参数，峰值显存 15.0 GiB。该数值只证明训练链路能运行，不属于方法效果证据。正式结论必须在相同更新数、优化器与 seed 下完成三臂训练，并同时报告 held-out target NLL、工具名准确率、参数 schema、闭环任务成功、证据 grounding、repair 成功率、stuck rate 和每个成功案例的推理 token。

> 2026-08-21 状态更新：用户要求停止所有实验。4k canary 中 full 与 naive 两臂各完成 12 步，compact 在第 6/12 步终止且没有最终 adapter；因此三臂实验无比较效力，前两臂 loss 也不得作为上下文优劣证据。权威停止记录见 `outputs/student_context_ablation_v2/EXPERIMENT_STOP_RECORD.json`。除非用户日后明确重新授权，不再运行训练或评测。
