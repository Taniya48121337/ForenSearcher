# UTQR-4：原生工具契约、Schema-valid SFT 与 Student Canary

更新时间：2026-08-21

## 1. 本轮结论

当前 600 条 SFT 已不再只是“文本里写了工具名”，而是把每个决策状态下真正可调用的函数定义交给 student，并让 teacher target 使用同一套外部参数格式。新的生产版本为：

- `outputs/production_sft_v7_external_contract`
- 59 个 Gold case；VideoSham 17、VCSL 21、VidDiffBench 21；
- 600 条 per-action SFT，472/64/64 train/validation/test；
- 600/600 target 通过自己可见的 JSON Schema；
- 600/600 context 通过状态、动作顺序、角色分离、工具可用性、Schema parity、历史参数契约和隐私检查；
- 192 条 target 做了必要的内部→外部接口归一化；自然语言 reasoning、工具选择和环境状态没有改变；
- student-visible 字符相对完整 transcript 减少 57.2%，private locator/oracle 命中为 0。

在 Qwen3-VL-8B-Instruct 的小型 QLoRA canary 中，正确 v4 数据训练 120 个平衡曝光后，在固定 9 条 held-out 生成上：

| 生成预算 | 可解析 | 工具名正确 | Schema-valid | pre-state eligible | 严格参数一致 | 执行语义一致 | 触顶 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 tokens | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 8/9 | 1/9 |
| 512 tokens | 9/9 | 9/9 | 9/9 | 9/9 | 8/9 | 9/9 | 0/9 |

唯一“严格参数不一致”是 `bind_evidence.binding_items` 的无语义顺序变化；字段到 EvidenceAtom 的映射完全相同。因此必须同时报告严格一致和执行语义一致，不能只取其中一个。

这些结果证明接口修复有效，但 9 条生成仍只是 canary，不足以宣称量产或闭环任务成功。

## 2. 发现的三个真实失败

### 2.1 历史动作长得像 target，模型复制历史 JSON

旧 compact context 的历史动作使用 `tool_name / arguments / call_id`，与待预测 target 的 envelope 相同。12-step 模型 9/9 都在复制历史 JSON，原生工具调用为 0。

修复：历史改为 `past_action.executed_tool / past_input`，环境结果放入 `environment_outcome`。这保留动作语义，但明确区分“已经发生的历史”和“现在要输出的动作”。

### 2.2 训练提示没有真正的工具函数定义

旧 SFT 只有 `eligible_tools` 名称，训练和评测调用 chat template 时没有传 `tools=`。Teacher 推理却收到完整工具名、描述、参数类型、required、enum 和 additionalProperties；这是训练—推理接口错位。

修复：从每条原始轨迹里的 `claim_schema + tool_registry + current eligible_tools` 恢复函数定义，转换为 Qwen 原生 function schema，并只暴露当前可执行的子集。每行包含：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "{task, current_harness_state, past_execution_history, ...}"},
    {
      "role": "assistant",
      "content": "基于可见状态的自然语言理由",
      "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}]
    }
  ],
  "tools": [
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
  ]
}
```

这里 reasoning 位于 `assistant.content`，不再重复塞入函数参数。

### 2.3 Teacher SFT 保存的是 harness 内部参数，不是模型看到的外部参数

最典型的是 `bind_evidence`：Teacher API schema 要求
`binding_items=[{field,support_ids}]`，执行前适配器把它转换成内部 `bindings={field:[ids]}`，但旧 SFT 保存了转换后的内部格式。另有 89 个 `stop` target 在正文已有 reasoning 的同时，又把 reasoning 重复保存在参数里。

这导致 192/600 target 不满足其可见 Schema。格式看起来可执行，实际却让 student 学习互相冲突的 API。

修复：SFT 只学习模型可见的外部参数；harness 内部转换只发生在执行边界。完整原始 transcript 继续保留，便于审计。

## 3. UTQR-4 的新增要求

UTQR-4 沿用原有 Q1–Q8，并把 H7/Q9 从“压缩上下文保真”升级为“决策上下文与原生工具契约保真”。每条正样本必须同时满足：

1. 权威 pre-action state 不变；
2. reasoning 与工具选择不变；
3. 历史动作顺序不变，且历史字段不与 target envelope 混淆；
4. `tools` 中的函数名与 `current_harness_state.eligible_tools` 精确一致；
5. 函数 schema 来自真实 teacher/executor 注册表，而非根据答案临时拼装；
6. target 工具当前可执行；
7. target arguments 通过同一条可见 JSON Schema；
8. 历史动作参数也使用外部工具契约；
9. 无本机路径、URL、媒体文件名、oracle 或 verifier 泄漏；
10. 完整 transcript 仍作为审计源，不以 compact 数据替代执行证据。

这不是为了增加形式门禁，而是针对已观测到的具体失败：低 NLL 的模型仍可能 0% 调用工具；普通的 target-name、state 和执行状态检查无法发现训练模板缺少 schema，也无法发现 `binding_items` 与 `bindings` 冲突。

## 4. Canary 的因果链

| 阶段 | 关键变化 | 训练曝光 | parse rate |
|---|---|---:|---:|
| v1 | 历史 envelope 冲突，无 schema | 12 | 0/9 |
| v2 | 历史角色分离，无 schema | 12 | 0/9 |
| v3 | 加原生 schema，但 192 个 target 仍是内部参数 | 12 | 6/9 |
| v4 | schema 与外部 target 完全一致 | 12 | 5/9（4 条触顶并循环） |
| v4 | 相同数据，曝光增至 120 | 120 | 8/9 @320；9/9 @512 |

因此最合理的解释是：角色分离是必要条件；真正的工具 schema 解决“如何调用”；外部参数统一解决监督冲突；足够的多工具曝光解决“停止解释并行动”。单看 teacher-forced NLL 会漏掉前两阶段的 0% 调用问题，所以必须保留自由生成评测。

## 5. 现在可以声称什么

可以声称：

- 已有视频数据可自动转成真实执行、自然语言 reasoning、原生函数调用、EvidenceAtom 绑定和 typed repair 的多轮 SFT；
- 三数据集的异构工具能在同一 `messages + tools` 契约中训练；
- 当前 600 条数据在确定性检查下具有 100% 的 target Schema validity 和 tool/state parity；
- 小型 held-out canary 表明接口修复能把原生工具调用从 0/9 提升到 9/9。

不能声称：

- 9 条生成代表稳定泛化；
- teacher target 一致等于闭环执行成功；
- 当前 600 条足以覆盖全部视频任务、长尾工具与 repair 状态；
- 512-token 预算就是最终部署最优值。

## 6. 下一步优先级

1. 在完整 64 条 validation 上生成，而不是只测 9 条；按数据集、工具、clean/repair 分层报告置信区间。
2. 把生成调用送入隔离的 harness checkpoint 做 shadow execution，报告 accepted/executed、state delta、EvidenceAtom 和下一状态，而不仅是字符串指标。
3. 单独统计 reasoning 重复率、动作前 token 数、调用完成 token 数和 token-limit hit，防止靠长预算掩盖策略循环。
4. 对工具名正确但参数语义不同的样本做人工盲审；严格一致与语义一致并列报告。
5. 完成 full / naive-tail / v4 三臂的同预算训练；当前结果只证明 v4 能工作，还没有完成上下文方法的正式优越性检验。
6. 扩大工具族与数据集后，先用 10-case canary 验证 schema/executor parity，再启动批量 teacher，不把前置检查替代真实执行与测量。

完整阶段汇总见 `outputs/student_external_contract_v1/native_tool_canary_summary.json`。

补充的 verifier-sidecar 等价审计为 9/9：生成动作分别在 VCSL、VidDiffBench、VideoSham 上各 3/3，与该状态下先前真实 accepted/executed、下游消费且因果必要的 transition 语义一致。报告位于 `outputs/student_external_contract_v1/generated_transition_equivalence_gen9.json`。该报告明确标注 `live_harness_reexecution_performed=false`，因此不能替代独立 checkpoint shadow execution。

UTQR-4 全池校准保留 59/59 Gold 正样本，并拒绝 862/862 个 minimal-edit hard negatives（16 个家族、三个数据集）；正式量产状态仍为 `quality_canaries_passed_scale_validation_pending`。
