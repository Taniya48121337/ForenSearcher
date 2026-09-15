# SFT-v2：可执行视频深搜轨迹实现说明

## 1. 现在训练的到底是什么

目标不是训练一个模型直接看视频后回答，而是训练一个类似 Harness-1 的策略模型：它读取当前可见状态，选择下一项工具，读取真实工具观测，再决定继续搜索、建立 claim、绑定证据、提交字段，或停止。

视频来自已有数据集，不生成新视频。自动化系统只负责：读取已有视频或官方特征、执行工具、把结果封装成 EvidenceAtom、让 LLM teacher 决策，并用隐藏标注在轨迹完成后验收最终答案。

当前适配的数据集是：

- VideoSham：偏时间篡改、删除区间和源—疑似视频对齐。
- VCSL：偏视频片段拷贝定位，使用官方特征做局部区间匹配。
- VidDiffBench：偏成对视频的动作/质量比较，需要根据多种视觉测量结果动态选择下一步。

## 2. 轨迹采用树状工具流程

```text
检查任务与媒体
  └─ 选择一个测量工具
       ├─ 证据充分且一致 → 建立 claim
       ├─ 置信度低/顺序不一致 → 换一种测量工具
       ├─ 多工具冲突 → 继续交叉验证
       └─ 工具耗尽 → 输出“证据不足/相等”或按可见聚合规则收敛
             └─ 绑定 EvidenceAtom
                  └─ 提交字段
                       └─ 全部高优先级义务关闭后 stop
```

VidDiff 的关键变化是：三个候选比较工具同时作为 label-blind action 暴露给 teacher。隐藏答案不能决定下一动作；只有已经返回的置信度、顺序一致性、工具间同意/冲突、剩余工具数能控制继续或停止。

## 3. 一条 SFT 样本的格式

采用 Harness-1 风格的标准多轮工具消息，每个 assistant 动作单独形成一个训练样本：

```json
{
  "messages": [
    {"role": "system", "content": "策略与证据约束"},
    {"role": "user", "content": "任务、工具和当前状态"},
    {
      "role": "assistant",
      "content": "上一观测说明了什么、还缺什么、为什么调用这个工具",
      "tool_calls": [{"function": {"name": "compare_uniform_timeline", "arguments": "..."}}]
    },
    {"role": "tool", "content": "真实工具观测与新的 harness state"}
  ],
  "metadata": {
    "source_turn_index": 2,
    "target_execution_status": "executed",
    "step_utility_verdict": "indispensable"
  }
}
```

训练 loss 只落在当前 assistant 的自然语言推理和工具调用上；tool observation 只作为上下文，不当成模型要生成的答案。被环境拒绝的动作可以保留为后续纠错上下文，但不会成为正例 target。

## 4. clean 与 repair 两类轨迹

每个 case 先生成 clean acquisition，再自动注入一种局部故障，生成 repair 轨迹。目前有三类故障：

1. `evidence_dropout`：证据本身消失，必须重开受影响字段、重新调用外部工具并绑定新证据。
2. `binding_invalidation`：证据仍存在，但 claim—evidence 边被破坏；应只重建绑定，不盲目重跑昂贵工具。
3. `decision_reopen`：证据和绑定都有效，但决策状态被重开；只需重新提交对应字段。

这三类故障分别训练“重新取证”“修复引用”“恢复决策”，避免所有 repair 都退化成同一种固定模板。

## 5. 如何确认每一步有意义

每一步都经过 recorded-observation causal replay：删除该动作，再用相同后续动作重放。如果无法得到相同的合法终态，该步才具有因果必要性。对于自适应搜索，低置信或冲突测量虽然可能不直接进入最终 claim，但它决定了后续分支，因此标记为 `branch_enabling_measurement`，而不是误删成“冗余步骤”。

每个 transition 另外导出 sidecar，包括：

- 当时可选动作集合；
- 实际动作和未判分替代动作；
- 义务、证据和终态的变化；
- grounded / consumed / causal / observation-sensitive 判定；
- branch-enabling 或 terminal-evidence 的决策角色；
- 每条 episode 内归一化的训练权重。

sidecar 不含 oracle，隐藏标注只在完整 rollout 结束后检查最终答案。

## 6. 真实 CLI canary 结果

样本：VidDiffBench `surgery_131`。

clean 工具序列：

```text
inspect_comparison_task
→ compare_action_refined_symmetric
→ compare_uniform_timeline
→ compare_global_motion_symmetric
→ create_candidate_claim
→ bind_evidence
→ set_field_decision
→ stop
```

repair 工具序列：

```text
set_field_decision(open)
→ compare_global_motion_symmetric
→ bind_evidence
→ set_field_decision(committed)
→ compare_uniform_timeline
→ compare_action_refined_symmetric
→ stop
```

结果：clean 8 个 SFT turn，repair 7 个 SFT turn，15/15 步通过因果审计；最终方向正确、工具结果重取、旧证据未复用、无 oracle 泄漏，整条 case 标记为 `safe_to_compile_sft=true`。

## 7. 批量运行建议

先对每个数据集各跑 10 个 case，并让三类 repair 轮换出现；只有 case 自身的全部校验通过才加入 accepted JSONL。VidDiff 使用 `--routing-mode adaptive`，不要再使用由 oracle 预先选好的 legacy 单分支作为正式训练数据。

CLI 适合 canary，但每一步都会启动完整 Codex 推理进程，其 token 统计不能直接代表后续 Responses API 的批处理成本。正式批量时切换 API backend，并保留相同的环境、rubric、sidecar 和 post-rollout oracle 验收。

## 8. 主要输出

- `sft_clean_accepted.jsonl`：clean 的逐动作 SFT 正例。
- `sft_repair_accepted.jsonl`：repair 的逐动作 SFT 正例。
- `transition_sidecar.jsonl`：逐步可选动作、状态变化和因果审计标签。
- `batch_report.json`：case 级正确性、工具利用率和是否可汇总训练。
- case 目录中的 `llm_teacher_*_full_trajectory.jsonl`：仅用于重放和审计，不直接训练。

## 9. VidDiff 10-case 实测结论

在完成 conflict policy、one-shot claim、失败聚合隔离、三类 repair contract 和分层动作树后，10 个 VidDiff case 的合并结果是：

- 3 个 case 全部通过并进入 SFT：`surgery_87`、`surgery_80`、`surgery_158`。
- 6 个 case 的轨迹过程与逐步审计通过，但最终方向与 oracle 不同，因此后验拒绝。
- 1 个早期 case 同时存在方向和过程审计失败，保留诊断但不进入训练。
- accepted 数据包含 clean 21 turns、repair 10 turns、transition sidecar 31 rows。
- accepted JSONL 的 oracle 泄漏扫描为 0，`safe_to_compile_accepted_sft=true`。

这说明轨迹框架已经能稳定保证工具执行、局部修复、终止条件和失败隔离；当前主要瓶颈已经转移到 VidDiff 多视觉测量冲突时的语义裁决准确率。

下一版不应退回“用 oracle 预选一个获胜工具”。更合适的方法是完整轨迹级 rejection sampling：对同一 case 生成多个完整、label-blind 的多工具 rollout，oracle 只在轨迹结束后作为 verifier；只保留同时通过答案和过程 rubric 的完整轨迹。这样可以利用多个合理的工具顺序与证据裁决路径，同时保证学生上下文中从未出现隐藏标签。

## 10. 完整轨迹级 Best-of-N

Best-of-N 已实现为 label-blind 路由组合，而不是把 oracle 答案放进 teacher prompt：

- `balanced`：当前阶段的候选测量工具全部可见，由 teacher 自由选择。
- `uniform_first`：先执行全时间线比较，再按观测决定是否继续。
- `global_first`：先执行全局对称比较。
- `action_first`：先执行动作细化对称比较。

每个 attempt 都生成独立的 clean、repair、step-utility report 和 validation report，保存在 `cases/<sample>/attempts/<index>_<variant>/`。只有第一条同时通过过程 rubric 与终局 oracle 验收的完整轨迹会被复制到 case 根目录并进入 aggregate SFT；其余 attempt 只用于诊断。

在 `surgery_28` canary 中：

1. `balanced` 完整轨迹选择 `compare_uniform_timeline`，过程正确但方向错误，被拒绝。
2. `global_first` 完整轨迹选择 `compare_global_motion_symmetric`，clean 6 turns、decision-reopen repair 2 turns，全部验证通过。
3. 系统在第 2 次成功后提前停止，没有执行剩余两个 variant。
4. accepted SFT 共 8 turns，oracle 泄漏扫描为 0。

这与“oracle 预选获胜工具”有本质区别：每个 route 在 rollout 期间只访问任务、工具观测和 harness state；隐藏标签只在完整轨迹结束后决定接受或拒绝。

## 11. 候选可达性筛选与新 10-case 结果

Best-of-N 之前新增了一项 verifier-only 可达性检查。它只在轨迹生成器之外判断：现有三个真实候选 EvidenceAtom 中，是否至少有一个支持参考方向。它不决定工具顺序，不向 teacher 暴露参考答案，也不默认启用；批量 canary 通过 `--skip-oracle-unreachable` 显式开启。

具体失败场景是：claim 字段只能绑定现有 EvidenceAtom 的值。如果三个候选都不含参考方向，无论 teacher 如何推理都不可能构造出既正确又有证据绑定的正轨迹。此时继续调用 CLI 只会浪费预算，或诱导系统伪造一个没有证据支持的答案。因此这类 case 标记为 `oracle_unreachable_candidate_set`，返回候选生成阶段补测量工具或补 EvidenceAtom，而不进入 SFT。

重新汇总同一组 10 个 VidDiff case 后：

- 7 个 case 完整通过并进入 accepted SFT；此前为 3 个。
- 2 个 case（`surgery_60`、`surgery_159`）的候选集合不可达，未启动 CLI rollout。
- 1 个 case（`surgery_116`）候选中存在正确方向，但三条完整 route 均只在 `oracle_direction_match` 上失败，说明剩余问题是冲突裁决或候选观测表达，而不是轨迹结构。
- accepted 数据共 clean 47 turns、repair 24 turns、transition sidecar 71 rows。
- accepted JSONL 的 oracle 泄漏扫描为 0，过程或多检查失败为 0，`safe_to_compile_accepted_sft=true`。

三条新增通过轨迹均由 `global_first` 开始：`surgery_57` 和 `surgery_155` 最终绑定全局运动测量，`surgery_131` 在全局观测不足后继续调用动作细化测量并绑定后者。这说明固定的首个探测动作不等于固定答案工具；后续分支仍由可见观测驱动。

当前应优先优化两层：先提高候选 EvidenceAtom 的正确方向召回率，解决不可达样本；再为“候选可达但裁决失败”的样本加入显式冲突表和可靠性特征。不要放宽 evidence binding，也不要把 oracle 放入路由器。
