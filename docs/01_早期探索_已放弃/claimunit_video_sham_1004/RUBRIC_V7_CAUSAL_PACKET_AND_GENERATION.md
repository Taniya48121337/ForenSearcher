# UTQR-7：统一的高质量工具轨迹生产方案

## 一句话说明

我们不再把“一条轨迹最后答对了”当作高质量。一个训练样本必须是一个**因果轨迹包**：同一基础轨迹还要在证据被遮蔽、替换、错配、A/B 对调、控制提示删除和无提示破坏时作出合理改变。只有整个包通过，基础轨迹中的已执行动作才可进入 SFT。

## 用什么数据

| 数据集 | 主要能力 | 本地工具证据 | 最终要学会的决策 |
|---|---|---|---|
| VideoSham | 多模态伪造/编辑比较 | 媒体身份、时间编辑、音频差异、空间区域差异 | 先判断应该测哪个模态，再形成并绑定差异结论 |
| VCSL | 视频拷贝片段定位 | 特征对身份、DP、局部对齐、时序稳定性、官方 SPD 等定位结果 | 在多个定位方法及冲突结果之间选择有证据的片段关系 |
| VidDiffBench | 两段视频的动作差异比较 | 全局时间线、运动、动作定向、对称残差与拒答结果 | 在 A/B 比较中确定方向与区间，证据不足时拒答 |

原始视频或特征只由本地工具读取。Teacher 看到的是逻辑媒体 ID、任务、当前 Harness 状态、工具定义和工具返回的 EvidenceAtom；它不看到本机路径、oracle、verifier 或人工编辑标签。

## 最终构造什么格式

训练数据仍是 Harness-1 风格的多轮原生工具调用，而不是一段总结文本：

```json
{
  "messages": [
    {"role": "system", "content": "在可执行工具环境中进行证据分析……"},
    {"role": "user", "content": "当前任务和调用前 Harness 状态"},
    {
      "role": "assistant",
      "content": "自然语言决策理由",
      "tool_calls": [{"name": "某个真实工具", "arguments": {}}]
    },
    {"role": "tool", "content": "真实执行结果和新的 Harness 状态"}
  ],
  "metadata": {
    "target_was_executed": true,
    "supervision_class": "evidence_interpretation",
    "training_weight": 1.0
  }
}
```

每个 SFT datum 只监督一个调用前状态下的 assistant 决策；被拒绝的调用、未执行的建议、完整轨迹回放和 verifier 信息不进入正样本 loss。完整轨迹与反事实轨迹保留在 sidecar 中供审计。

## 真人式思路如何写

每一步理由应自然包含四个成分，而不是背模板：

1. 刚出现的关键事实：哪条可见观察改变或保持了当前判断。
2. 当前信念与备选解释：至少保留一个仍合理的替代解释或不确定性。
3. 尚未解决的问题：现在到底缺哪一类信息。
4. 为什么选这个工具：它的结果如何区分不同可能，而不是预先假定工具会返回什么。

例如：

> EA-VIDDIFF-017 的全局运动只显示 B 在后半段更活跃，但这也可能来自镜头移动，而不一定是目标动作更完整。现在缺少的是动作区域内的定向证据；局部动作比较可以检验该差异是否仍集中在目标区域，若不能则应保留不确定而不是直接选择 B。

高质量不等于更长。正常为 2–4 句、约 35–140 个英文词或信息量相当的中文。理由不得出现“prompt 要求”“rubric 要求”“只有一个可用工具”“oracle”等过程伪迹。

## 生成流程

```text
数据集案例
  → 本地工具形成 EvidenceAtom
  → 删除答案式控制信息的 policy view
  → Teacher 提议一个动作
  → 严格理由评分；不达标才追加候选（最多 3 个）
  → Harness 真正执行工具
  → clean 轨迹 + repair 轨迹
  → 固定因果邻域的 8 类探针
  → 机器评分 + 两个隔离上下文的盲评
  → UTQR-7 通过后按步骤导出 SFT
```

### 1. 降控制视图

Teacher 不再看到编译好的下一动作、可直接复制的 binding、repair recipe、单值 target enum 或完整可物化答案。工具接口和真正状态仍然保留，所以不是删功能，而是防止策略从控制面抄答案。

### 2. 自适应多候选

先生成一个候选。若它已经同时做到证据引用、问题识别、替代解释、工具分辨性说明且无过程伪迹，就立即执行；否则才生成第二、第三个候选并选择可见证据最充分的一个。这保留 AutoTraj 式候选筛选的质量收益，又避免固定 best-of-3 的三倍成本。

### 3. 真执行和逐步因果审计

每个动作必须被 Harness 接受或执行。随后删除该动作并固定重放后缀；如果删掉它仍得到同一合法终局，这一步通常不应成为高权重 SFT。外部测量还会做观察 dropout，检查后续是否真正消费了工具结果。

### 4. 因果邻域

| 探针 | 改什么 | 高质量策略应怎样反应 |
|---|---|---|
| semantic mask | 保留证据对象但遮蔽语义值 | 重新测量或合理拒答，不复述原结论 |
| schema-compatible swap | 换成同结构但相反/不同的合法证据 | 依赖字段改变，身份与无关字段保持 |
| identity mismatch | 证据内容不变，任务/实体/范围身份错配 | 发现错配并核验或拒答，不复制 donor 身份 |
| A/B role swap | 固定规则交换两侧角色 | 方向和区间按规则变换，无关字段保持 |
| reduced control | 删除编译动作、单值目标和修复配方 | 仍能作出合法的语义决策 |
| structure only | 只保留结构，去掉证据语义 | 不能凭工具数量或 schema 猜出原答案 |
| unannounced repair | 注入 active-but-wrong、冲突、陈旧、部分或空结果 | 在 stop 前主动检查并局部修复/拒答 |
| step intervention | 删除单步并固定重放后缀 | 每个导出步骤都应改变信息或合法终局 |

这些探针借鉴了 [Causal Agent Replay](https://arxiv.org/abs/2606.08275) 的干预—向前重放思路和 [PriVE-Bench / PriVE-Tools](https://arxiv.org/abs/2607.16311) 的成对反事实视觉证据思想。状态外置继承自 [Harness-1](https://arxiv.org/abs/2606.02373)，多工具视频搜索结构参考 [VideoSearcher](https://arxiv.org/abs/2607.02927)。困难负例与过程判别参考 [Plan-RewardBench](https://arxiv.org/abs/2604.08178)，多候选与修复思想参考 [AutoTraj](https://arxiv.org/abs/2601.23032)。

## 统一评分：UTQR-7

| 维度 | 分值 | 要求 |
|---|---:|---|
| C1 结果与绑定安全 | 14 | 终局、隐藏结果、执行状态和绑定都正确 |
| C2 证据身份/来源/范围 | 14 | 所有字段有当前任务的合适证据，不能复制 donor 身份 |
| C3 因果证据敏感性 | 14 | 遮蔽或替换关键证据时，策略必须合理改变 |
| C4 无关不变性与角色等变性 | 12 | 无关字段保持，A/B 对调按规则变化 |
| C5 降控制自主性 | 12 | 没有答案式控制提示仍能作出语义决策 |
| C6 工具信息增益与经济性 | 10 | 每步被消费且不可无损删除，无重复空调用 |
| C7 不确定性与无提示修复 | 10 | 主动发现异常，局部修复或拒答 |
| C8 真人式决策过程 | 8 | 两个隔离盲评者在四维上平均至少 4/5 |
| C9 SFT 可迁移性与卫生 | 6 | 只训练真实执行的 action，不泄露私有状态 |

单个包必须总分至少 90，所有硬失败项通过，所有关键维度至少 0.85，任何维度不低于 0.75。两个盲评者之间任一维度相差超过 1 分时，不能未经复核直接通过。

这个标准故意不容易达到。此前 VidDiff 13 个可达案例映射到 UTQR-7 后为 **0/13 通过，31.3–53.4 分**；它们虽然有部分正确终局，却缺少身份错配、A/B 对调、无提示修复和删步因果证据。这种反向标定说明 rubric 没有给旧轨迹“送分”。

## 为什么这些硬失败是必要的

本项目已经观测到普通 schema、类型和终局测试无法覆盖的具体失败：

- 272 个状态中 226 个在控制面预暴露了部分或完整目标调用。
- 规则脚本无需证据即可完成 3/3 显式 repair。
- Student 对 10 个无提示证据异常的定向发现率为 0/10。
- shuffled evidence 会复制 donor 案例的结论。

因此只有这些与因果、跨案例身份和训练导出直接相关的边界被设为硬失败；普通版本、主键、类型和可执行性仍由原有代码与测试负责。

## 监督权重

| 类别 | 权重 |
|---|---:|
| 语义决策 | 1.0 |
| 证据解释/冲突裁决 | 1.0 |
| 不确定性与异常修复 | 1.2 |
| 单一可行动作的纯记账步骤 | 0.1 |
| 被拒绝或未执行动作 | 0.0 |

每个批次 control-only 比例不得超过 20%。这避免大量容易的 bind/commit/stop 模板压过真正的视频证据推理。

## 大批量生产前的放行顺序

1. 本地 dry-run：数据、工具和路径全部可用，不调用 Teacher。
2. 每个数据集 1 个已授权案例：跑完整因果包，先修协议错误。
3. 每个数据集 10 个案例：统计失败类型、工具效用和 pass@2。
4. 每个数据集至少 30 个 entity-disjoint 案例：每类探针通过率至少 85%，包通过率至少 90%，分数 p10 至少 90。
5. 用新实体留出集进行 student free-run；只有相对无 SFT 基线有真实增益，才称为可批量生产。

不得用 oracle 可达性来筛掉难例后再报告覆盖率；oracle 只允许在执行后验证结果。基础轨迹、探针、盲评和最终导出均缓存，失败重试只从缺失阶段继续，避免重复 Teacher 调用。

## 当前实现

- 评分规范：`rubric/unified_trajectory_quality_rubric_v7.json`
- 机器评分器：`src/utqr_v7_causal.py`
- 降控制与自适应候选：`src/causal_trajectory_generation.py`
- 八类探针执行器：`src/causal_probe_suite.py`
- 双盲过程评审：`src/blind_trajectory_judge.py`
- 三数据集入口：`src/videosham_batch.py`、`src/vcsl_batch.py`、`src/viddiff_batch.py`
- 已保存 VidDiff 轨迹的增量资格评测：`src/qualify_saved_viddiff_case.py`

截至 2026-08-24，完整回归为 **431 项全部通过**；本地预检 VideoSham 7/7、VCSL 12/12、VidDiff 5/5 均可进入 Teacher 执行。三个真实代表 canary 已通过完整 UTQR-7：

| 数据集 | 案例 | clean / repair 步数 | UTQR-7 | 双盲四维均值（前瞻/更新/方法/自然） |
|---|---|---:|---:|---|
| VideoSham | videosham-4055 | 6 / 5 | 98.600 | 4.0 / 4.5 / 4.0 / 4.0 |
| VCSL | index 11 对应 pair | 6 / 5 | 97.886 | 5.0 / 5.0 / 5.0 / 4.0 |
| VidDiffBench | surgery_117 | 12 / 9 | 99.200 | 4.0 / 5.0 / 5.0 / 4.0 |

三者 H1–H6 均通过。VidDiff 的 clean 真实执行了 `inspect_comparison_task → uniform timeline → global motion → action refined → claim → 三次 assessment → adjudication → bind → commit → stop`；证据失效后的 repair 只重跑失效的 action-refined 分支，再重做三次 assessment、adjudication、bind、commit 和 stop。它没有从头重跑三种测量。

### 工具是否应删除

- **不能按单个 canary 的未出现来删除。** VideoSham 的 temporal/audio/spatial 工具按案例模态条件开放；VCSL 每条固定路线只暴露一个真正可执行的 localizer；VidDiff 的 predicate 与 symmetry-residual 工具只在相应谓词或对称性冲突路线开放。
- `assess_evidence_candidate` 和 `adjudicate_evidence` 只在出现多个竞争 EvidenceAtom 时开放。VideoSham/VCSL 的简单 base 轨迹不需要它们，但因果探针中的冲突状态实际使用了它们；VidDiff base 也实际使用。
- VCSL 的 HV-SEGMENT、LOCAL-ALIGNMENT、TEMPORAL-NETWORK 当前属于离线路线/消融候选，尚没有本轮真实 UTQR-7 canary 证明其生产价值。它们不会在默认固定路线中一起暴露，因此暂不删除，也不计为生产主工具；后续 30-case telemetry 若仍为零选择或零不可替代调用，再降级或移除。
- 当前三条代表包的 `step_intervention.unnecessary_tool_calls=0`、`repeated_identical_calls=0`。因此没有证据支持删除已进入这些轨迹的工具。

### 当前放行结论

现在只完成了“大批量生产前放行顺序”的第 2 阶段：三个数据集各有一个完整真实 canary。**这证明格式、工具执行、局部修复、因果探针和盲评链已经跑通，不证明已经可以无审计地批量生产。** VCSL 与 VidDiff 的代表 case 是从 verifier/oracle 可达集合中选出的生成 canary；oracle 没有进入 teacher 上下文，但这种 case 选择不能用于报告无偏覆盖率。

下一阶段应在不按 oracle 可达性筛选的 case-disjoint 样本上运行每数据集 10 个 case，报告所有失败而不是只导出成功项；达到每数据集至少 30 个的批量标准后，才可标记 production-ready。当前不启动新的训练。
