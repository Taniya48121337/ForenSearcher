# Step 6 — Comparison Against the Proposed Novelty

Timestamp: 2026-08-16 Asia/Shanghai

- **Proposed work**
  - Title: Claim-Causal Mutation-Tested Trajectory Compilation（待收缩）
  - Date: 2026-08 proposal
  - Source: —
  - Problem framing: 从已有 source–suspect 视频和真实工具输出自动构造可验证 SFT 调查轨迹。
  - Core mechanism: claim provenance graph、最小证据切片、证据语义变异、后缀重放、局部 repair supervision。
  - Key insight: 正确答案或整轨迹 causal utility 不能保证每条 claim 真正依赖其证据；应破坏依赖并要求选择性撤回/修复。
  - Application domain: source-grounded video manipulation investigation。

- **Prior work A**
  - Title: OpenVisTool
  - Date: 2026-08
  - Source: https://arxiv.org/abs/2608.08557
  - Problem framing: 构造高质量 visual tool-use SFT trajectories。
  - Core mechanism: whole-trace vs no-tool causal utility filtering。
  - Key insight: outcome-correct trajectories may contain decorative tool use。
  - Application domain: visual tool agents。
  - Axes matching: 3/4
  - Level: **Level 2 — High Overlap**（one axis differs；只在 claim-field intervention/repair mechanism 上明确不同）

- **Prior work B**
  - Title: CausalFlow
  - Date: 2026-05
  - Source: https://arxiv.org/abs/2605.25338
  - Problem framing: 从 failed traces 生成最小修复和训练监督。
  - Core mechanism: step intervention、suffix re-execution、minimal repair pairs。
  - Key insight: 因果干预把失败转成可靠监督。
  - Application domain: text/code/web agents。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work C**
  - Title: From Faulty Memories to Corrected Actions
  - Date: 2026-08
  - Source: https://arxiv.org/abs/2608.10502
  - Problem framing: 错误记忆传播后的状态与答案恢复。
  - Core mechanism: typed provenance graph、independent support、selective invalidation/replay。
  - Key insight: 删除根故障不足以修复已传播状态，应保留独立支持并局部重算。
  - Application domain: memory-augmented agents。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work D**
  - Title: Bridge Evidence
  - Date: 2026-07
  - Source: https://arxiv.org/abs/2607.15253
  - Problem framing: 衡量 agentic search 中单证据的因果效用。
  - Core mechanism: document omission + suffix replay + CTU。
  - Key insight: 静态相关性不能代表轨迹中的因果作用。
  - Application domain: multi-hop retrieval。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work E**
  - Title: Causal Agent Replay
  - Date: 2026-06
  - Source: https://arxiv.org/abs/2606.08275
  - Problem framing: 归因 agent 失败的因果步骤。
  - Core mechanism: SCM do-interventions、stochastic forward replay、Shapley。
  - Key insight: trace attribution must be interventional and distributional。
  - Application domain: generic agent debugging。
  - Axes matching: 1/4
  - Level: **Level 4 — Low Overlap**（three axes differ）

- **Prior work F**
  - Title: STAMP
  - Date: 2026-07
  - Source: https://arxiv.org/abs/2607.11172
  - Problem framing: 对证据生产动作做细粒度 RL credit assignment。
  - Core mechanism: training evidence graph、support verifier、first-exposure provenance、advantage modulation。
  - Key insight: 证据验证和产生证据的动作归因必须分开。
  - Application domain: deep search。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work G**
  - Title: COVERT
  - Date: 2026-04
  - Source: https://arxiv.org/abs/2604.09813
  - Problem framing: 构造可验证 tool-use RL environments。
  - Core mechanism: oracle-preserving controlled perturbations 与 capability-conditioned verifier。
  - Key insight: 受控扰动和保留 oracle 的分离使自动验证可扩展。
  - Application domain: generic function calling。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work H**
  - Title: MetaRAG
  - Date: 2025-11
  - Source: https://arxiv.org/abs/2509.09360
  - Problem framing: atomic factoid-level RAG hallucination detection。
  - Core mechanism: claim decomposition、synonym/antonym mutations、context entailment/contradiction oracle。
  - Key insight: factoid-level metamorphic relations expose unsupported claims。
  - Application domain: enterprise text RAG。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work I**
  - Title: EviPath
  - Date: 2025-09
  - Source: https://openreview.net/forum?id=alNQNzaWhI
  - Problem framing: 从已有答案与支持证据合成 agent SFT trajectories。
  - Core mechanism: abductive dependency-aware planning + proxy evidence environment。
  - Key insight: 静态标注可反向恢复 agent 过程监督。
  - Application domain: RAG multi-hop QA。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work J**
  - Title: HardGen
  - Date: 2026-01
  - Source: https://openreview.net/forum?id=WLmL6YWM7t
  - Problem framing: 生成针对模型弱点的困难工具轨迹。
  - Core mechanism: failure API graph + hard-query/tool synthesis + verifier loop。
  - Key insight: 真实失败比随机结构复杂度更能产生有效 hard examples。
  - Application domain: generic tool agents。
  - Axes matching: 1/4
  - Level: **Level 4 — Low Overlap**（three axes differ）

- **Prior work K**
  - Title: InfCycle
  - Date: 2024-09
  - Source: https://openreview.net/forum?id=CD2wgg9RQD
  - Problem framing: 无外部 step labels 的工具轨迹质量控制。
  - Core mechanism: step-wise cycle consistency。
  - Key insight: 循环一致性可作为合成轨迹 verifier。
  - Application domain: generic tool use。
  - Axes matching: 1/4
  - Level: **Level 4 — Low Overlap**（three axes differ）

- **Prior work L**
  - Title: From Agent Traces to Trust
  - Date: 2026-06
  - Source: https://arxiv.org/abs/2606.04990
  - Problem framing: evidence tracing 与 execution provenance 的统一研究议程。
  - Core mechanism: taxonomy/survey，不提供拟议算法。
  - Key insight: claim-level semantic provenance 与 recovery evaluation 仍是关键开放问题。
  - Application domain: generic LLM agents。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work M**
  - Title: From Fluent to Verifiable
  - Date: 2026-02
  - Source: https://arxiv.org/abs/2602.13855
  - Problem framing: deep-research claim-level auditability。
  - Core mechanism: semantic provenance graph 与 auditability metrics。
  - Key insight: 逐声明 evidence linkage 是可信研究 agent 的主要瓶颈。
  - Application domain: deep research。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work N**
  - Title: VidForensics-M1
  - Date: 2026-08
  - Source: https://arxiv.org/abs/2608.11201
  - Problem framing: 视频真伪标签与 supporting temporal evidence 的联合优化。
  - Core mechanism: synthetic paired videos + temporal grounding + evidence-guided reward redistribution。
  - Key insight: 时间定位比自由文本理由更客观可验证。
  - Application domain: video forensics。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ；且依赖生成新视频，与本项目限制不同）

- **Prior work O**
  - Title: VIGIL
  - Date: 2026-03
  - Source: https://arxiv.org/abs/2603.21526
  - Problem framing: 证据驱动、可解释的 deepfake detection。
  - Core mechanism: part-centric plan-then-examine 与独立 forensic evidence injection。
  - Key insight: evidence generation 与 conclusion localization 混合会产生幻觉。
  - Application domain: deepfake video/image forensics。
  - Axes matching: 2/4
  - Level: **Level 3 — Medium Overlap**（two axes differ）

- **Prior work P**
  - Title: Learning From Failure
  - Date: 2024-02
  - Source: https://arxiv.org/abs/2402.11651
  - Problem framing: 把失败轨迹纳入 agent fine-tuning。
  - Core mechanism: 成功/失败前后缀训练格式。
  - Key insight: 负例包含成功轨迹缺失的决策边界。
  - Application domain: generic agents。
  - Axes matching: 1/4
  - Level: **Level 4 — Low Overlap**（three axes differ）

## Overall result

最坏情况由 OpenVisTool 决定：3 个轴保守匹配，故当前宽表述为 **Level 2 — High Overlap**。这不等于方法没有价值，而是论文的新颖性不能写成“首个因果过滤视觉工具轨迹”“首个 claim mutation”“首个反事实 repair trajectory”或“首个 provenance graph selective replay”；这些都已有直接或组合式近邻。
