# Step 5 — Full-Paper Deep Dive

Timestamp: 2026-08-16 Asia/Shanghai

## Retrieval note

按 skill 要求尝试使用 bundled `fetch_paper.sh`，但当前 Windows 环境没有 `bash`，因此脚本无法启动。按 graceful fallback 规则，以下核验基于 arXiv 官方 HTML 正文和 OpenReview 官方 PDF 文本；不是只读摘要。每篇均检查了问题设置、方法和实验/限制。

## 1. OpenVisTool

- **Problem framing (verified)**：为 visual tool use 构造 cold-start SFT 数据；来自 chart、table、GUI grounding、visual search、web-to-HTML 五个领域。
- **Core mechanism (verified)**：先筛除 probe model 无工具也能稳定解答的问题；教师在真实工具执行中生成轨迹；用同一固定 probe 比较完整 tool trace 条件与 no-tool 条件的成功率差，只保留 outcome-valid 且 causal-utility 足够的轨迹。
- **Key insight (verified)**：成功教师可能依靠参数知识或原图粗线索回答，工具调用只是装饰；whole-trace counterfactual utility 比 judge-based plausibility 更可靠。
- **Application domain (verified)**：图像级视觉工具使用，不含视频取证、claim ledger 或 source–suspect 配对。
- **Venue**：arXiv preprint, v2, 2026-08-11。
- **Assumptions & scope**：以整条工具 calls+observations 为干预单位；比较的是 scalar final answer reliability；不定位单个 observation 对单个 claim field 的作用，不生成 revise/reject repair supervision。
- **Closest-passage evidence**：§3.1–3.4；作者明确把 causal utility 定义为 fixed probe 在 tool-trace-conditioned 与 no-tool 条件下的成功率提升。
- **Refined overlap**：Problem framing=match；Core mechanism=partial；Key insight=match；Application domain=partial/match（视觉 agent）。保守计 3 轴。
- **Source**：https://arxiv.org/html/2608.08557

## 2. CausalFlow

- **Problem framing (verified)**：输入失败的结构化 agent trace，输出导致失败的步骤、最小反事实修复以及可复用的监督对。
- **Core mechanism (verified)**：将 trace 视作前序依赖的 sequential chain；替换候选 step，重执行所有下游步骤；以 verifier 是否从失败翻转为成功定义 CRS；在成功替换中选 token 编辑最小者；输出 `(wrong step, corrected step)`。
- **Key insight (verified)**：日志归因和全局重写不能证明因果责任；step intervention + suffix execution 可以把失败变成 training-time preference/reward supervision。
- **Application domain (verified)**：GSM8K、MBPP、SealQA Hard、MedBrowseComp；不是视觉或视频。
- **Venue**：arXiv preprint, 2026-05-25。
- **Assumptions & scope**：从失败→成功；全局 task verifier 为主要 oracle；依赖模型是 sequential chain；大多数非代码任务使用 predictive re-execution；不从成功 claim 破坏单条 evidence edge，也不要求非目标 claim 字段保持不变。
- **Closest-passage evidence**：§3 定义 trace 与 intervention；§4.1 计算 CRS；§4.2 生成 minimal repair pairs；§5.3 说明 K=3 及 deterministic/predictive re-execution。
- **Refined overlap**：Problem framing=partial/differ；Core mechanism=match；Key insight=match；Application domain=differ。计 2 轴。
- **Source**：https://arxiv.org/html/2605.25338

## 3. MetaRAG

- **Problem framing (verified)**：在无金答案、黑盒企业 RAG 中检测输出对检索上下文的幻觉，并定位到 atomic factoid/span。
- **Core mechanism (verified)**：分解答案为 independently verifiable factoids；对每个 factoid 生成同义和反义变体；固定 retrieved context 为 oracle，要求同义变体被蕴含、反义变体被矛盾；聚合 fact-level penalties。
- **Key insight (verified)**：用蜕变关系可以在没有精确输出 oracle 时测试事实一致性。
- **Application domain (verified)**：文本 RAG hallucination detection；67 个企业回答，非 agent trajectory learning。
- **Venue**：Identity-Aware AI workshop at ECAI 2025；arXiv v2。
- **Assumptions & scope**：mutation 施加在 claim 文本而非 evidence/tool observation；不重放 agent 后缀；不构造 action–observation provenance；不产生训练轨迹，论文明确 mitigation/prevention 不在实验范围。
- **Closest-passage evidence**：§3.1–3.5，尤其 Algorithm 1 与 §3.3 的 synonym/antonym mutation。
- **Refined overlap**：Problem framing=partial；Core mechanism=partial/match（claim-level mutation testing）；Key insight=partial/match；Application domain=differ。保守计 2 轴。
- **Source**：https://arxiv.org/html/2509.09360

## 4. From Faulty Memories to Corrected Actions

- **Problem framing (verified)**：已知错误 memory 后，恢复答案、轨迹和 persistent state，同时保留无关的正确工作。
- **Core mechanism (verified)**：从 runtime provenance 建 typed memory-to-action graph；沿 propagation edges 求 affected set；用 independent trusted support 保留仍有独立依据的节点；失效 unsupported descendants；只对 answer-relevant affected subgraph 求 execution closure 并选择性重放。
- **Key insight (verified)**：source deletion 不等于 state recovery；正确的修复必须同时具备污染传播追踪、独立支持检查与局部重算。
- **Application domain (verified)**：shopping、travel、customer support memory agents 与 LongMemEval-V2 衍生压力集。
- **Venue**：arXiv preprint, 2026-08-11。
- **Assumptions & scope**：错误 memory 已由上游 detector 诊断；目标是 runtime recovery，不是 SFT 数据构造；不主动生成语义 mutants；最终关联对象主要是 answer node，而不是结构化多字段 claim ledger。
- **Closest-passage evidence**：§1 明确 diagnosis outside scope；§3.4 independent-support checking；§3.5 answer-relevant subgraph 与 execution closure；§3.6 selective replay。
- **Refined overlap**：Problem framing=differ；Core mechanism=match/partial；Key insight=match；Application domain=differ。计 2 轴。
- **Source**：https://arxiv.org/html/2608.10502

## 5. Bridge Evidence

- **Problem framing (verified)**：判断多步 agentic search 中一篇已读文档的因果价值，而不是其静态相关性。
- **Core mechanism (verified)**：对每个实际展示的文档逐一执行 omission intervention，固定已发生前缀，从删除点重放剩余轨迹；CTU 比较最终答案、下一查询检索质量和步数。
- **Key insight (verified)**：桥接证据可能不直接包含答案，却通过提供实体改变下一搜索；静态 relevance 与 causal utility 近乎独立。
- **Application domain (verified)**：Qwen2.5 ReAct + HotpotQA + Wikipedia retrieval。
- **Venue**：arXiv preprint, 2026-07。
- **Assumptions & scope**：仅单文档 omission；用于测量和分析，不合成 SFT；没有 claim field、evidence type mutation、selective invalidation 或 repair target。
- **Closest-passage evidence**：§1.1 说明逐文档删除；§2.5 定义从干预点重放；§2.6 定义 CTU。
- **Refined overlap**：Problem framing=differ；Core mechanism=partial/match；Key insight=match；Application domain=differ。计 2 轴。
- **Source**：https://arxiv.org/html/2607.15253

## 6. STAMP

- **Problem framing (verified)**：解决 deep-search GRPO 中 trajectory reward 与产生证据的 action 之间的 credit mismatch。
- **Core mechanism (verified)**：训练时 evidence graph 定义 target entities/relations；reference-based verifier 检查 cited documents 是否支持各 unit；将支持文档追溯到 first-exposure action；用 bounded sign-preserving modulation 调整该 step 的 advantage。
- **Key insight (verified)**：必须分开“轨迹验证了什么证据”和“哪个动作首次让证据可见”；结果正确轨迹中大量步骤并不产出证据，失败轨迹中也可能有有用证据步骤。
- **Application domain (verified)**：BrowseComp、BrowseComp-ZH、xbench-DS live-web deep search。
- **Venue**：arXiv preprint, 2026-07-13。
- **Assumptions & scope**：依赖已有 citations 和训练时 evidence graph；使用 support verification 与 first exposure，不执行 counterfactual mutation/replay；产出 RL step credit，不是 SFT repair traces。
- **Closest-passage evidence**：§2.2 evidence verification；§2.3 first-exposure provenance；§2.4 sign-preserving advantage modulation。
- **Refined overlap**：Problem framing=partial；Core mechanism=partial/match；Key insight=partial/match；Application domain=differ。计 2 轴。
- **Source**：https://arxiv.org/html/2607.11172

## 7. COVERT

- **Problem framing (verified)**：生成可执行、可验证的工具使用环境供 agentic RL，而不是仅做离线 SFT。
- **Core mechanism (verified)**：Stage I 通过格式、规则、语义三级验证生成可靠基础轨迹；Stage II 对 query/toolset/tool output 施加 capability-targeted augmentation，同时保留 oracle call/answer 或 capability label；exact 或 judge-assisted verifier 计算奖励。
- **Key insight (verified)**：只有显式区分“被扰动的环境组件”和“保留的 oracle 元数据”，才能自动验证歧义、干扰与错误工具结果下的行为。
- **Application domain (verified)**：BFCL v3、ACEBench function calling；mock tool environment。
- **Venue**：arXiv preprint, 2026-04-10。
- **Assumptions & scope**：扰动目标是一般 capability robustness；大多保留固定 oracle call/final answer；不从每个最终 claim 反向推导必要证据，也不测试“目标字段变化、非目标字段不变”。
- **Closest-passage evidence**：§3.1 multi-level validation；§3.2 oracle-preserving augmentation；Table 1；§4 capability-conditioned reward。
- **Refined overlap**：Problem framing=partial；Core mechanism=partial；Key insight=partial/match；Application domain=differ。保守计 2 轴。
- **Source**：https://arxiv.org/html/2604.09813

## Deep-dive correction to the original idea

原表述“claim-level mutation testing + suffix replay + repair supervision”不能作为一个整体无保留地声称新颖，因为：MetaRAG 已有 claim-level mutation；CausalFlow 已有 intervention/replay/minimal repair supervision；OpenVisTool 已有 visual trajectory causal filtering；Faulty-Memory work已有 typed dependency graph、independent support 与 selective replay。可守住的技术空隙必须进一步收缩为：**对 evidence 输入而非 claim 文本做干预，并以结构化 claim field 的目标敏感性与非目标不变性作为可执行 oracle，再将局部 obligation 状态变化编译成 SFT。**
