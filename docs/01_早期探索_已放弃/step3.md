# Step 3 — Abstract-Level Triage

Timestamp: 2026-08-16 Asia/Shanghai

以下 Overlap score 为四轴初筛匹配数，0–4；“部分匹配”在本轮保守地按匹配计数，以避免高估新颖性。

## 1. OpenVisTool

- **Title**: OpenVisTool: An Open Recipe for Synthesizing Instructive Visual Tool-Use Trajectories
- **Date**: 2026-08
- **Problem framing**: 从教师生成的视觉工具轨迹构造可用于 SFT 的高质量监督。
- **Core mechanism**: 难度筛选、领域轨迹生成、结果正确性与整条工具痕迹相对无工具条件的 causal utility 验证。
- **Key insight**: 成功答案不能证明工具观察真正贡献了答案。
- **Application domain**: 图表、表格、GUI、视觉搜索、Web-to-HTML 等视觉工具使用。
- **Overlap score**: 3/4
- **Source**: https://arxiv.org/abs/2608.08557

## 2. CausalFlow

- **Title**: CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures
- **Date**: 2026-05
- **Problem framing**: 从失败 agent trace 中定位故障步骤并生成可训练的修复监督。
- **Core mechanism**: step-level intervention、后缀重执行、Causal Responsibility Score、最小编辑修复、对比监督对。
- **Key insight**: 只有执行反事实干预才能可靠定位失败因果步骤并把失败转化为监督。
- **Application domain**: 数学、代码、QA、医疗浏览。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2605.25338

## 3. From Faulty Memories to Corrected Actions

- **Title**: From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents
- **Date**: 2026-08
- **Problem framing**: 已诊断错误记忆后的状态、答案与轨迹恢复。
- **Core mechanism**: typed memory-to-action provenance graph、受影响子图、独立支持检查、选择性失效与重放。
- **Key insight**: 删除错误源不等于恢复已传播的下游状态；应保留独立支持的节点并只重算答案相关部分。
- **Application domain**: 记忆增强的购物、旅行、客服 agent。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2608.10502

## 4. Bridge Evidence

- **Title**: Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search
- **Date**: 2026-07
- **Problem framing**: 衡量多步搜索中单篇文档的真实因果效用。
- **Core mechanism**: 逐文档删除、从干预点重放后缀、CTU 评分。
- **Key insight**: 不直接支持答案的证据仍可能通过改变下一查询成为关键桥接证据。
- **Application domain**: HotpotQA agentic retrieval。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2607.15253

## 5. Causal Agent Replay

- **Title**: Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures
- **Date**: 2026-06
- **Problem framing**: 用执行干预而非日志判断定位 agent 失败的因果步骤。
- **Core mechanism**: agent SCM、五类 do-operation、同策略前向重放、对比估计与 Monte-Carlo Shapley。
- **Key insight**: 相关性日志归因不可靠；随机 agent 必须报告干预后的结果分布与不确定性。
- **Application domain**: 通用工具 agent 失败诊断。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2606.08275

## 6. STAMP

- **Title**: STAMP: Provenance-Guided Credit Assignment for Deep Search Agents
- **Date**: 2026-07
- **Problem framing**: 深度搜索 RL 中轨迹级奖励无法给产出证据的具体动作正确归因。
- **Core mechanism**: 训练时 evidence graph、引用支持验证、first-exposure provenance、sign-preserving advantage modulation。
- **Key insight**: 应分开判断“验证了什么证据”与“哪个动作首次暴露该证据”。
- **Application domain**: live-web deep search。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2607.11172

## 7. COVERT

- **Title**: Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning
- **Date**: 2026-04
- **Problem framing**: 为 agentic RL 构造可执行、自动判分的工具使用环境。
- **Core mechanism**: 可靠基础轨迹 + oracle-preserving 扰动；加入干扰工具、歧义查询、噪声或错误工具输出。
- **Key insight**: 分离可控扰动与保留的 oracle 元数据可规模化验证复杂工具行为。
- **Application domain**: 通用 function calling。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2604.09813

## 8. MetaRAG

- **Title**: MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems
- **Date**: 2025-11
- **Problem framing**: 无真值、黑盒 RAG 输出的细粒度幻觉检测。
- **Core mechanism**: 将答案拆成 atomic factoids，生成同义/反义 claim mutation，再对固定检索上下文做蕴含/矛盾验证。
- **Key insight**: 事实级蜕变关系可把黑盒幻觉检测转化为可执行测试。
- **Application domain**: 企业文本 RAG。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2509.09360

## 9. EviPath

- **Title**: From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents
- **Date**: 2025-09 / revised 2026-02
- **Problem framing**: 从已有答案与 supporting facts 构造 RAG-agent SFT 轨迹。
- **Core mechanism**: 反向溯因子任务规划、以金证据建立代理环境、生成完整 planner–executor 对话轨迹。
- **Key insight**: 现有数据集虽无过程监督，但答案和证据可以反向恢复依赖感知的行动路径。
- **Application domain**: 多跳开放域 QA。
- **Overlap score**: 2/4
- **Source**: https://openreview.net/forum?id=alNQNzaWhI

## 10. HardGen

- **Title**: From Failure to Mastery: Generating Hard Samples for Tool-use Agents
- **Date**: 2026-01 / revised 2026-06
- **Problem framing**: 自动生成复杂、可验证、针对模型弱点的工具轨迹。
- **Core mechanism**: 从失败 API 构建动态图，采样困难依赖，生成高级工具和困难问题，reasoner–verifier 闭环。
- **Key insight**: 困难度应来源于真实失败和隐式依赖，而不是随机拼接工具。
- **Application domain**: 通用工具调用。
- **Overlap score**: 1/4
- **Source**: https://openreview.net/forum?id=WLmL6YWM7t

## 11. InfCycle

- **Title**: InfCycle: Learning to Use Tools via Inference Compute and Cycle Consistency
- **Date**: 2024-09
- **Problem framing**: 无可靠外部 verifier 时提升工具轨迹合成质量。
- **Core mechanism**: 多阶段合成与 step-wise cycle consistency 验证。
- **Key insight**: 循环一致性可从合成轨迹内部提供过程监督。
- **Application domain**: 通用工具使用与推理。
- **Overlap score**: 1/4
- **Source**: https://openreview.net/forum?id=CD2wgg9RQD

## 12. From Agent Traces to Trust

- **Title**: From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents
- **Date**: 2026-06
- **Problem framing**: 统一综述 agent 的 evidence tracing 与 execution provenance。
- **Core mechanism**: provenance taxonomy，覆盖 claim–evidence、工具、记忆、动作、答案和恢复。
- **Key insight**: final accuracy 不能回答每条 claim 由什么证据产生；claim-level semantic provenance 与 recovery evaluation 仍是开放问题。
- **Application domain**: 通用 LLM agents。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2606.04990

## 13. From Fluent to Verifiable

- **Title**: From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents
- **Date**: 2026-02
- **Problem framing**: 把 deep-research agent 的 claim-level auditability 变成一等设计和评价目标。
- **Core mechanism**: semantic provenance graph、provenance coverage/soundness、冲突透明度与 audit effort。
- **Key insight**: 生成变便宜后，逐声明追溯与审计成本成为主要瓶颈。
- **Application domain**: deep research agents。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2602.13855

## 14. VidForensics-M1

- **Title**: VidForensics-M1: Meta-Detection Reinforcement Learning with Verifiable Temporal Grounding for AI-Generated Video Forensics
- **Date**: 2026-08
- **Problem framing**: 同时优化 AI 生成视频的检测标签和可验证时间证据。
- **Core mechanism**: 生成配对真假视频、时间定位监督、evidence-guided reward redistribution。
- **Key insight**: 文本理由易幻觉，而可控篡改区间提供更客观的验证信号。
- **Application domain**: AI-generated video detection。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2608.11201

## 15. VIGIL

- **Title**: VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection
- **Date**: 2026-03
- **Problem framing**: 提升 deepfake 检测的可解释性和跨生成器泛化。
- **Core mechanism**: plan-then-examine、部位级独立取证、分阶段证据注入与 part-aware RL reward。
- **Key insight**: 将证据生成和结论定位混在一起会产生幻觉，应先规划再用独立证据检查。
- **Application domain**: 人脸 deepfake detection。
- **Overlap score**: 2/4
- **Source**: https://arxiv.org/abs/2603.21526

## 16. Learning From Failure

- **Title**: Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents
- **Date**: 2024-02
- **Problem framing**: 避免只用成功轨迹训练 agent。
- **Core mechanism**: 将成功与失败轨迹的前缀/后缀以专门格式纳入微调。
- **Key insight**: 失败轨迹包含成功数据没有的决策边界信息。
- **Application domain**: 通用语言 agent。
- **Overlap score**: 1/4
- **Source**: https://arxiv.org/abs/2402.11651
