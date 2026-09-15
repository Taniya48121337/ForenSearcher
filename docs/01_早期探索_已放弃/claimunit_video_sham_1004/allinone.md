# 轨迹生成、因果评测与工具证据查新记录

## 检索设置

- 日期：2026-08-24
- 查询：`agent tool trajectory causal evaluation counterfactual evidence grounding training data verification`
- 年份：2024–2026
- 每来源上限：8
- 来源：OpenReview、DBLP、Crossref、arXiv、OpenAlex、Semantic Scholar
- 返回：24 条（Crossref 8、arXiv 8、Semantic Scholar 8）

检索异常：OpenReview 客户端未安装；OpenAlex 返回 504 Gateway Timeout；DBLP 返回 0 条。因而本次结果不等同于系统综述，重要方法另用论文官方页定向核验。

## 对本项目最重要的结论

1. 单轨迹成功不足以证明策略利用了工具证据；需要固定干预和向前重放。
2. 同结构证据替换、身份错配和 A/B 对调比随机扰动更能发现答案记忆与 donor copy。
3. Harness 应承担状态、约束和记账，策略仍要承担语义选择；把下一动作编译给模型会制造虚假 SFT。
4. 工具调用正确和最终答案正确应分开计分。
5. 多候选生成应配合可见证据评分、执行验证和修复；困难最小负例比大量容易模板更有训练价值。
6. 长轨迹的纯 LLM judge 不稳定，因此必须以机器因果探针为主、盲评只评人类式过程。

## 优先阅读

- [Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures](https://arxiv.org/abs/2606.08275)：对动作/状态做干预后向前重放，用于判断某一步是否真正导致失败或成功。
- [Seeing What Is Actually There: PriVE-Bench and PriVE-Tools](https://arxiv.org/abs/2607.16311)：成对反事实视觉证据，用于检验 VLM 是否真正受视觉证据控制。
- [Harness-1](https://arxiv.org/abs/2606.02373)：把长程状态和记账外置到 Harness，但保留策略的语义决策责任。
- [VideoSearcher](https://arxiv.org/abs/2607.02927)：多工具视频深度搜索，区分工具调用能力与答案质量。
- [Plan-RewardBench / Aligning Agents via Planning](https://arxiv.org/abs/2604.08178)：使用困难计划负例评测奖励模型，提示我们构造最小反事实坏轨迹。
- [AutoTraj](https://arxiv.org/abs/2601.23032)：多候选、修复和轨迹偏好构造，启发按需 best-of-N。

## 全部检索结果

### OpenReview（0）

客户端缺失：`openreview not installed; pip install openreview-py`。

### DBLP（0）

本查询未返回结果。

### Crossref（8）

1. Shengyao Sun. *Did It Happen? Counterfactual Evaluation of LLM Agent Recovery from Ambiguous Tool Outcomes*. [DOI](https://doi.org/10.21203/rs.3.rs-10730245/v1)
2. Jiuxiang You, Yi Yu, Zhenguo Yang. *Causal Dual-Interventional MedVQA via Textual Perturbation and Counterfactual Visual Verification*. [DOI](https://doi.org/10.20944/preprints202605.1802.v1)
3. Jeremy Meier, Michael Donner, Simon M. Laham. *Testing the Counterfactual Account: No Evidence that Counterfactual Relevance Explains the Effects of Norm Violations on Causal Judgements*. [DOI](https://doi.org/10.2139/ssrn.5175394)
4. Qiaoying Lu. *Two interventionist modes of counterfactual evaluation for causal strength*. Synthese, 2026. [DOI](https://doi.org/10.1007/s11229-026-05640-8)
5. Jisoo Kim, Minho Park, Hyunwoo Lee. *Counterfactual Policy Evaluation for Improving Coordination Quality in Multi-Agent Reinforcement Systems*. 2026. [DOI](https://doi.org/10.71465/ajbd3668)
6. Narendar Kumar, Vincent Hilaire, Stéphane Galland, et al. *Trustworthy Causal Reasoning in LLM-based Multi-Agent Smart Building Systems: A Multi-Level Evaluation Across Observational, Interventional, and Counterfactual Tasks*. Procedia Computer Science, 2026. [DOI](https://doi.org/10.1016/j.procs.2026.04.142)
7. Shanshan Luo, Zhi Geng. *Discussion on “Causal and Counterfactual Views of Missing Data Models”*. Statistica Sinica, 2026. [DOI](https://doi.org/10.5705/ss.202025.0152)
8. Shu Yang, Jae Kwang Kim. *Discussion on “Causal and Counterfactual Views of Missing Data Models”*. Statistica Sinica, 2026. [DOI](https://doi.org/10.5705/ss.202025.0165)

### arXiv（8）

1. Jaineet Shah. *Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures*. 2026. [arXiv](https://arxiv.org/abs/2606.08275)
2. Rui Wu, Lizheng Wang, Yongjun Li. *The Causal Round Trip: Generating Authentic Counterfactuals by Eliminating Information Loss*. 2025. [arXiv](https://arxiv.org/abs/2511.05236)
3. Datao Tang, Hao Wang, Yudeng Xin, et al. *TerraGen: A Unified Multi-Task Layout Generation Framework for Remote Sensing Data Augmentation*. 2025. [arXiv](https://arxiv.org/abs/2510.21391)
4. Kaiyue Yang, Yuyan Bu, Jingwei Yi, et al. *When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents*. 2026. [arXiv](https://arxiv.org/abs/2606.20023)
5. Datao Tang, Xiangyong Cao, Xuan Wu, et al. *AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation*. 2024. [arXiv](https://arxiv.org/abs/2411.15497)
6. Shulin Tian, Ziqi Huang, Fan Zhang, et al. *Open Evaluation Agent: Efficient and Promptable Evaluation of Visual Generative Models*. 2026. [arXiv](https://arxiv.org/abs/2608.09666)
7. Fan Zhang, Shulin Tian, Ziqi Huang, et al. *Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models*. 2024. [arXiv](https://arxiv.org/abs/2412.09645)
8. Jingyu Sun, Jiachen Tu, Yuyang Xue, et al. *Seeing What Is Actually There: PriVE-Bench and PriVE-Tools for Counterfactual Evaluation of Agentic Visual Evidence in VLMs*. 2026. [arXiv](https://arxiv.org/abs/2607.16311)

### OpenAlex（0）

接口错误：`504 Gateway Timeout`。

### Semantic Scholar（8）

1. F. Wang, Z. Guo, Z. Ye. *MedAgent: A Retrieval-Augmented Clinical Decision Support Agent with Verifiable Evidence Grounding for Evidence-Based Medicine*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/19037df859adc24a37e4ba3256657793e9c82afe)
2. Chenrui Shi, Yuwei Wu, Yang Liu, et al. *Interactive Reward Agent: GUI Task Evaluation via Environment-State Verification*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/2b8ba1228957040ba59c8274c2f07174f3a1f7d2)
3. Hang Zhang, Ruheng Wang, Yuelyu Ji, et al. *Scaling Medical Reasoning Verification via Tool-Integrated Reinforcement Learning*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/864449c94e308e7532c4b8432ded5fbefc870cc1)
4. Lang Mei, Xiaohan Yu, Chong Chen, et al. *SearchArt: Training Long-Horizon Search Agent with Scalable Synthetic and Verified Task*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/91cea1d58c91edb5897dc3659f7320eef2f9f6c5)
5. Wentao Shi, Yu Wang, Yuyang Zhao, et al. *AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation*. ACL, 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/5dc3997e0631d19608b84ecb2317dd9387515b47)
6. Zhuowen Liu, Bohan Cui, YinShang Guo, et al. *HERALD: Counterfactual Audits and Minimal Repairs for Proof-of-Retrieval Rewards*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/cdec73b1afeeee939f5b36c56a9f7ca8f70e91ae)
7. Lokanatha Reddy Gandikota, Venkata Raghavendra Swamy Gudipati, Sudheer Kumar Aluvala, et al. *Visual Dialogue System with Self-Verification Pipeline for Hallucination Mitigation*. Rev-AI, 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/ddc400bb0eb33b56b1222a8517d4fd502b8227d1)
8. Shuang Chen, Kaituo Feng, Hangting Chen, et al. *OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents*. 2026. [Semantic Scholar](https://www.semanticscholar.org/paper/d096fb2aa3c990c7db9f09eaf473c581e7ba3777)

## 与 UTQR-7 的对应关系

| 文献线索 | 落到本项目的实现 |
|---|---|
| 干预与向前重放 | semantic mask、schema swap、identity mismatch、删步固定后缀重放 |
| 成对反事实视觉证据 | A/B role swap、结构相同但语义不同的 EvidenceAtom |
| Harness 状态外置 | Harness 管状态与约束，policy view 不提供答案式下一动作 |
| 多工具视频搜索 | VideoSham/VCSL/VidDiff 使用统一接口但保留数据集专用测量工具 |
| 环境状态验证 | 所有正样本工具调用必须真实执行；盲评不代替机器验证 |
| 困难负例与最小修复 | unannounced repair、donor identity、partial/empty/conflicting result |
| 多候选和修复 | 首候选严格达标即停，否则最多补到 3 个；repair 轨迹提高权重 |

## 2026-08-24 实现与 canary 更新

完整自动化链已经在三个数据集各跑通一个真实案例：VideoSham `videosham-4055` 为 98.600 分，VCSL index 11 对应 pair 为 97.886 分，VidDiffBench `surgery_117` 为 99.200 分，三个因果包均通过 H1–H6 和双盲过程评审。完整项目回归为 431 passed、0 failed。

VidDiff 的关键修正是把证据失效恢复改为来源可追踪的局部回滚：失效的 action-refined EvidenceAtom 只重新开放 `compare_action_refined_symmetric`，保留仍有效的 uniform/global 证据；repair 因而从“10 步未完成”变为 9 步完整终止。A/B 探针也改为同时交换任务角色与所有活跃角色相关证据，而不是混合交换后的旧 atom 与未交换的新测量。

当前只能称为“跨数据集代表 canary 跑通”，不能称为 mass-production ready。VCSL 与 VidDiff 代表案例的生成选择包含 post-execution verifier/oracle-bounded 筛选，虽然 oracle 从未进入 teacher 上下文，但不能据此估计无偏成功率。下一步必须对每数据集 10 个不按 oracle 可达性筛选的 case-disjoint 样本报告全量通过与失败，再进入每数据集至少 30 个的批量放行审计。
