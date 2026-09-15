# ECSO 轨迹质量 Rubric：文献依据、评分规则与当前结果

## 1. 评测目标

评测单元不是孤立的最终回答，而是一对轨迹：

1. `clean_acquisition`：从可见媒体与中性 locator 获取证据，闭环 ClaimObligation；
2. `evidence_dropout_local_repair`：删除必要 EvidenceAtom 后，只重开受影响字段，重新获取证据并恢复声明。

Rubric 需要排除三种最危险的伪高质量样本：

- 最终答案碰巧正确，但中间过程跳过证据获取或验证；
- evidence ID 存在，但其类型不能支持对应 claim field；
- mutation 后清空整个状态再抄回答案，而不是目标敏感、非目标不变的局部修复。

因此采用两层判定：6 个 hard failure checks 负责拒绝不可用于训练的轨迹；12 个细粒度维度组成 100 分诊断分。只有所有 hard checks 通过、总分不低于 95，且每个维度至少达到 80%，才进入 `A_gold` SFT。

## 2. 定向文献检索

检索日期：2026-08-16。检索源限定为论文的官方页面，包括 ACL Anthology、ICLR Proceedings、NeurIPS/arXiv 和 OpenReview。关键词覆盖 `agent trajectory evaluation`、`process supervision`、`atomic factual support`、`citation correctness completeness`、`behavioral invariance`、`rubric sliding window`、`state-based evaluation` 与 `failure provenance DAG`。

这是一轮针对方法设计的 scoping review，不冒充 PRISMA 系统综述。纳入标准是论文必须贡献至少一个能直接转成 ECSO 可执行检查的评测概念；仅报告最终准确率、没有过程诊断的方法不作为主要依据。

| 论文 | 证据级别 | 采用的设计 | 在 ECSO 中的实现 |
|---|---|---|---|
| [CheckList](https://aclanthology.org/2020.acl-main.442/)（ACL 2020） | 同行评审 | Minimum Functionality、Invariance、Directional Expectation | clean validity、off-target invariance、target sensitivity |
| [FActScore](https://aclanthology.org/2023.emnlp-main.741/)（EMNLP 2023） | 同行评审 | 将长文本拆成可独立核验的 atomic facts | 将 claim 拆成 8 个可独立评分字段 |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/)（EMNLP 2023） | 同行评审 | citation correctness 与 completeness 分开评测 | Evidence binding 的正确性和覆盖率分开计分 |
| [Let's Verify Step by Step](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html)（ICLR 2024） | 同行评审 | process supervision 不应被 outcome supervision 替代 | 每一步状态转换单独验证，最终正确不覆盖过程错误 |
| [AgentBoard](https://arxiv.org/abs/2401.13178)（NeurIPS 2024 Oral） | 同行评审 | subgoal progress rate 和 grounding analysis | 每步记录 obligation progress，区分部分进展与成功 |
| [Prometheus 2](https://aclanthology.org/2024.emnlp-main.248/)（EMNLP 2024） | 同行评审 | user-defined evaluation criteria 与分档描述 | 每个 criterion 有固定定义、权重和满分条件；自然语言项可选 judge |
| [τ-bench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b126cc38b8638e07bef37e7b2bb72bf-Abstract-Conference.html)（ICLR 2025） | 同行评审 | 最终环境状态对 goal state；用 `pass^k` 测可靠性 | hidden oracle 只做 terminal check；批量阶段报告重复 rollout 可靠性 |
| [VitaBench](https://iclr.cc/virtual/2026/poster/10007108)（ICLR 2026） | 同行评审 | atomic rubric、跨长轨迹的 persistent criterion state | rubric state 不因窗口切分丢失；当前短轨迹直接保存 criterion state |
| [SkillCoach](https://arxiv.org/abs/2607.01874)（2026 preprint） | 预印本 | selection、following、composition、reflection；process 与 verifier 分离 | 证据选择、步骤执行、依赖顺序、停止前验证分别计分 |
| [AgentEval](https://arxiv.org/abs/2604.23581)（2026 preprint） | 预印本 | DAG 依赖与 root-cause attribution | FieldProvenanceGraph 和最早失败 criterion 诊断 |

两篇 2026 预印本只用于直接相关的工程启发，不作为唯一依据；核心结构同时由已同行评审的过程监督、行为测试、证据归因和状态评测工作支撑。

## 3. Rubric 结构

### 3.1 六个 hard failure checks

| ID | 检查 | 为什么必须拒收 |
|---|---|---|
| G1 | Oracle isolation | 轨迹看到编辑任务文本或 oracle 路径，会把标签泄漏训练成捷径 |
| G2 | Evidence reference and type integrity | 缺失/伪造 ID、字段不兼容证据或 distractor atom 都会训练错误归因 |
| G3 | No false closure | open obligation 或 unsupported field 被标成 verified，会破坏 ECSO 状态语义 |
| G4 | Terminal claim validity | 过程合理但最终 claim 错误，不能成为正向 SFT target |
| G5 | Mutation integrity | 直接改正确答案文本不是 evidence-side mutation，无法训练证据敏感性 |
| G6 | Repair non-leakage | repair 重用已删除证据等于绕过 reacquisition |

这些 checks 只放在训练数据入库边界，因为错误样本一旦进入 SFT/RL 数据会污染策略；普通单元测试无法判断语义证据是否支持字段。

### 3.2 100 分诊断分

| ID | 维度 | 分值 | 可执行定义 |
|---|---|---:|---|
| O1 | Terminal outcome correctness | 10 | claim type、scope、目标时长和 repair 终态与 hidden oracle 一致 |
| O2 | Atomic claim completeness | 8 | 8 个必需字段均存在且非空 |
| E1 | Field-evidence correctness and precision | 10 | 每个字段覆盖所需 evidence kinds，且不夹带不兼容/distractor atom；任一错误绑定该项计 0 |
| E2 | Field-evidence completeness | 10 | clean/repair 中每个 verified field 都有可见 evidence binding |
| E3 | Provenance-chain closure | 10 | 每个字段都有 action → evidence → field → claim/obligation 路径；任一断链该项计 0 |
| P1 | Step validity and order | 8 | acquisition → binding → propose → verify → stop 顺序合法 |
| P2 | Obligation lifecycle | 8 | clean 显式经历 open → evidence_ready → proposed → verified；repair 重新 open |
| P3 | Subgoal progress | 6 | clean 单调进展；dropout 后局部下降，再恢复到 1.0 |
| M1 | Target-field sensitivity | 8 | 被删除证据支持的字段全部且仅这些字段变 unsupported/reopened |
| M2 | Off-target invariance | 8 | source identity、scope 和 retained interval 的值、状态、绑定保持不变 |
| M3 | Local repair and recovery | 8 | 获取新证据、只恢复 target bindings、终态回到 clean goal state |
| S1 | Stop/reflection/efficiency/calibration | 6 | verify 后才 stop，无重复动作，高置信与对齐质量一致 |

完整机器可读定义见 `rubric/trajectory_quality_rubric.json`；执行器见 `src/trajectory_rubric.py`。

## 4. 当前轨迹为什么是多步骤而不是答案包装

Clean 轨迹包含 7 步：

```text
inspect_media
→ bind_comparison_scope
→ align_visual_timeline
→ measure_source_time_contraction
→ propose_claim
→ verify_claim
→ stop
```

Repair 轨迹也包含 7 步：

```text
inject_evidence_dropout
→ audit_field_support
→ reopen_affected_obligation
→ targeted_alignment_remeasurement
→ rebind_affected_fields
→ verify_repaired_claim
→ stop
```

其中 5 个 target fields 被重开；`source_id`、`comparison_scope_s`、`retained_source_interval_s` 三个 independently supported fields 原值和绑定均保持不变。这正对应 ECSO 的局部信用分配主张。

## 5. 当前评测结果与反例元评测

真实 VideoSham 1004 clean + repair bundle：

- hard checks：`6/6`；
- 诊断分：`100/100`；
- 等级：`A_gold`；
- SFT 入库：`accepted_for_sft=true`；
- LLM judge：未使用，核心分数全部来自结构化状态和真实工具证据。

为了避免 rubric 只会给高分，测试套件主动注入十种故障：

| 故障注入 | 预期检测 |
|---|---|
| 在 student SFT 中加入 oracle annotation | G1 reject |
| 把 field binding 换成伪 evidence ID | G2 reject |
| 给字段绑定一个真实但不兼容的 evidence | G2 reject + E1 failure |
| 给正确 binding 额外夹带一个 distractor evidence | G2 reject + E1 failure |
| 删除 repair evidence 到 target field 的 provenance edge | E3 failure |
| 有 open obligation 时 stop | G3 reject |
| mutation 直接改 claim text | G5 reject |
| repair 重新使用被删除的 evidence | G6 reject |
| dropout 后把所有字段全局清空 | M2 failure |
| 跳过 `verify_claim` | P1/P2/S1 failure |

上述反例均已被自动测试拒收。

## 6. 仍需做的人类校准

`100/100` 只证明这个确定性单例满足当前结构契约，不证明 rubric 已经在不同数据集、claim type 和教师模型上完成外部效度验证。批量生成前建议：

1. 抽取至少 50 对 clean/repair 轨迹，混入真实失败与合成故障；
2. 由两名不知道生成方法的标注者独立评分；
3. 对 hard checks 报 Cohen's κ，对连续维度报 ICC 或 Spearman；
4. 分析长度偏置、最终答案锚定和模型身份偏置；
5. rubric 修改只能在 calibration split 上完成，在 held-out validation split 验证后采用；
6. 批量阶段额外报告不同随机 rollout 的 `pass^k`、mutation operator 分层分数与 outcome/process 四象限。

在完成这一步之前，当前 rubric 可以作为严格的数据工程筛选器，但论文中应将其表述为“programmatic rubric validated on fault-injection tests”，而不是已经达到人类等价的通用评测器。
