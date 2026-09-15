# 统一高质量工具轨迹方案（UTQR-2）

## 一句话说明

我们不是让大模型“看答案后编一段思维链”，而是让它在一个真实可执行的多轮工具环境中逐步做决定：工具真的运行、环境真的改变、每个字段必须绑定到可追溯证据、故障后必须做类型正确的局部修复。只有最终结果正确、每一步因果上有用、自然语言依据可见状态且整条轨迹通过严格 rubric，才导出为 SFT。

## 数据、构造物和目的

- 输入：已有视频数据集及其公开视频、帧、特征或预计算视觉资产。数据集标签只给轨迹结束后的 verifier 使用。
- 过程：LLM teacher 只看到 student-view 状态和当前可调用工具；每轮输出自然语言 reasoning、一个工具名和参数。工具执行后把观察和 EvidenceAtom 写回 harness。
- 输出：
  - `clean_episode`：从无证据到可验证 claim 的完整工具轨迹；
  - `repair_episode`：注入一种局部故障后的最小恢复轨迹；
  - per-turn SFT：只对已执行且通过因果审计的 assistant action 计算 loss；
  - transition sidecar：`state → action → observation → next_state`，便于以后做 PRM/RL；
  - rejected/repaired pair：可用于偏好学习或 trajectory reward model。
- 目的：训练接近 Harness-1 的策略模型，学习“根据显式环境状态选择下一动作”，而不是背诵视频答案或虚构工具结果。

## 为什么采用这种设计

[Harness-1](https://arxiv.org/abs/2606.02373) 的核心启发是把候选、证据、验证记录和未完成约束放到环境状态中，让模型保留真正的语义决策。这里对应为 candidate claim、EvidenceAtom、字段绑定、决策状态、eligible tools 和 open obligations。

[VideoSearcher](https://arxiv.org/abs/2607.02927) 展示了时间定位、空间聚焦和多模态搜索可以组成统一工具轨迹，并用两阶段 SFT→RL 训练；本文实现先把冷启动 SFT 的“真实工具执行、证据接地和工具选择”做可靠，再考虑 RL。VideoSearcher 是 2026 年技术报告，本文只把它作为设计参照，不把结论当作已充分复现的定论。

[Guided by Trajectories / AutoTraj](https://arxiv.org/abs/2601.23032) 采用多候选、分维评估、修复低质量轨迹，并把修复前后组成偏好对。本文进一步要求候选必须在同一 harness 中真实执行，并使用 typed repair，而不是只靠 LLM 改写文本。

[Plan-RewardBench](https://arxiv.org/abs/2604.08178) 用规则扰动和 minimal-edit 产生难负例，并指出长轨迹 judge 很难。因而 UTQR-2 不使用单一 LLM 总分，而采用可执行硬门、字段级证据检查、最小编辑 hard negatives 和确定性反事实回放。

[τ-bench](https://arxiv.org/abs/2406.12045) 强调终态验证和 `pass^k` 可靠性；[AgentBoard](https://arxiv.org/abs/2401.13178) 强调不能只看最终成功，还要看中间进展。因此这里同时记录终态正确性、逐步状态变化、删步反事实和后续重复采样的 `pass^k`。

## 自动构造流程

1. **数据适配**：把每个数据集转换成统一 manifest，但保留数据集自己的 oracle 和评价指标。
2. **student-view 隔离**：生成 teacher/学生可见资产；oracle 标签、标注帧和答案不进入 prompt、tool observation 或 SFT。
3. **状态外化检查**：第一个检查工具返回身份、媒体/特征结构、可见诊断和推荐工具分支。
4. **真实 teacher rollout**：teacher 每轮输出 2–3 句自然语言 reasoning、一个工具调用和严格参数；环境真实执行。
5. **EvidenceAtom 化**：外部观察被拆成带 `supports_fields` 和 `generated_by` 的证据原子。
6. **claim 构造**：依次建 claim、绑定证据、提交字段、终止；eligible-tools tree 阻止越级动作。
7. **typed repair**：每条 clean 轨迹只注入一种故障：
   - evidence dropout：reopen → 重新测量 → 新证据重绑 → recommit → stop；
   - binding invalidation：reopen → 原有效证据重绑 → recommit → stop；
   - decision reopen：recommit → stop。
8. **数据集 verifier**：轨迹完成后才读取 oracle，检查结果正确性；oracle 不返回 teacher。
9. **逐步因果审计**：删除每一个动作并固定回放后缀；若仍得到相同合法终态，该动作不能作为正 SFT target。
10. **UTQR-2 评分**：任何硬门失败即拒绝；总分和每个分项同时达到阈值才导出。
11. **修复或淘汰**：可修复的自然语言/动作错误产生 minimal repair pair；证据不可达或结果错误的轨迹不进入正样本。
12. **批量前校准**：跨数据集、跨 repair family、跨难度分层跑 held-out canary 和 `pass^k`，再决定是否扩量。

## UTQR-2：六个不可平均掉的硬门

| 硬门 | 要求 | 典型拒绝例 |
|---|---|---|
| H1 结果忠实 | 满足数据集 oracle/定位质量 | VCSL 只有高 recall、precision 接近 0 |
| H2 可执行终态 | clean/repair 合法终止，导出动作均已执行 | 文本声称调用工具但无 execution |
| H3 证据与 provenance | 每个提交字段有活跃、类型兼容且值一致的 EvidenceAtom | 编造 evidence ID 或用错字段 |
| H4 oracle 隔离 | student-visible 内容无标签、答案或标注帧 | reasoning 泄漏 oracle box |
| H5 因果必要 | 删掉任一步会破坏同一合法终态，外部动作对观察敏感 | 多余搜索、无用工具、装饰性步骤 |
| H6 typed 最小修复 | 只修受影响字段，修复类型和动作成本匹配 | decision reopen 却重新跑视频工具 |

## 八个质量分项（100 分）

| 分项 | 权重 | Gold 最低值 |
|---|---:|---:|
| Q1 结果/定位质量 | 18 | 0.85 |
| Q2 字段级证据接地 | 17 | 0.90 |
| Q3 状态感知规划与顺序 | 12 | 0.85 |
| Q4 信息增益与因果效用 | 12 | 0.90 |
| Q5 自然、具体、可见依据的 reasoning | 14 | 0.80 |
| Q6 修复诊断、最小性和恢复 | 12 | 0.90 |
| Q7 不确定性与冲突校准 | 6 | 0.80 |
| Q8 SFT 教学性与 target hygiene | 9 | 0.90 |

直接进入正 SFT 需要：六个硬门全过、总分至少 92、八个分项分别过线。高分不能补偿任何硬门失败。

## VCSL 的工具优化：从“召回率过关”到可复现路由

旧 HV 会沿稀疏高分对角线输出整条大框，固定 10-case 虽然 recall 全过，但有样本 precision 只有 0.00–0.04。旧 validator 因为只检查 recall，错误地报告 10/10。

现在保留七个在 30-case 池中至少各自贡献一个严格可达样本的定位工具：

- `localize_copy_segment`：官方式 HV 基线/兼容工具，不在默认自动路线中；
- `localize_copy_segment_hv_segment`：把稀疏对角投票切成局部连续片段，减少整条对角线过框；
- `localize_copy_segment_local_alignment`：容忍局部变速的序列对齐；
- `localize_copy_segment_dp`：常规紧凑单段或多段对齐；
- `localize_copy_segment_temporal_network`：保留多条单调候选路径，适合存在短干扰的复制片段；
- `localize_copy_segment_temporal_stability`：弱桥拆分、斜率规整、强主路径抑制短干扰；
- `localize_copy_segment_axis_saliency`：处理严重变速导致路径碎裂的宽区域复制。

`inspect_pair_features` 会外化不含标签的路由诊断。除基本的 DP/TN/AXIS 分流外，当前还使用三条只依赖可见特征的通用规则：双稳定片段时用端点不确定度扩边；DP 几乎覆盖整幅矩阵而 TN 仍保留主体时抑制 DP 过覆盖；严重变速时用宽区域显著性估计持续时间、再用最强单调路径锚定起点。工具选择依据会出现在 student-visible observation 中，因此训练和推理时都可复现。

VCSL 的 Q1 现在按字面要求 precision、recall、F1 三者都至少 0.85，不再使用旧的二次缩放阈值。固定 10-case 真实 CLI 汇总为 9/10 Gold；唯一失败例在当前可见特征与工具候选下仍不可达，因此没有伪造正轨迹。扩展到 30-case 后，单个定位器最多严格覆盖 15 例，七工具联合覆盖 21 例；本地 rejection sampler 只导出这 21 例，九个不可达样本不启动 teacher。

## 真实 CLI canary（2026-08-20）

| repair | 路由/工具 | Rubric | P/R/F1 | 结果 |
|---|---|---:|---|---|
| evidence dropout | DP | 98.364 | .933/.931/.932 | Gold |
| binding invalidation | auto→TN-STABILITY | 99.300 | .976/.908/.941 | Gold |
| decision reopen | auto→DP | 99.708 | .977/.950/.963 | Gold |

三条轨迹均通过六个硬门、逐步因果回放和 oracle 隔离。reasoning 平均约 66–68 词，跨轮最高 token Jaccard 为 0.262–0.306，未发现非可见 EvidenceAtom 引用。

## VideoSham 与 VidDiff 分层 canary（2026-08-20）

VideoSham 采用显式 case 选择补齐随机抽样容易漏掉的能力族。原有 10 条轨迹之外，又从新增 20 对媒体中选取 7 条 label-blind 信号边界最小的分层 canary，补齐 appearance、audio、insertion、removal、omission、warp、text 七个能力族。新增 7/7 与合并后的 17/17 均为 Gold；UTQR-3 独立重评分 P10=98.295，并覆盖三种 repair。`compare_spatial_regions`、`detect_temporal_edit`、`compare_audio_tracks` 均在对应真实媒体上运行；聚合 telemetry 中 selected calls 与 indispensable calls 完全相等，因此目前没有证据支持删除任何一个已覆盖工具。实际 teacher 传输是 student-visible 文本 JSON；59 个 manifest/trajectory/SFT 文件的精确扫描中，本机路径、媒体文件名、URL、任务标签和 oracle/verifier 值均为 0。

VidDiff 的 Q7 检查自然语言是否真正使用可见 measurement atoms、selected evidence、方向/顺序一致性、置信度与可见依据，不接受机械列工具或只按最高置信度投票。多证据情形仍要求至少三类实质因素并明确解决一致或冲突；经过 oracle-bounded rejection sampling 后只暴露单一已验证证据时，则要求不确定性校准加至少一个非置信依据，不虚构不存在的替代项。针对原来不可达的谓词，新加入 `compare_predicate_independent`：先对 A、B 做互相隔离的单视频谓词测量，再用固定阈值比较，减少顺序锚定。它成功覆盖“普遍可见性反证搜索”和“稳定支撑的持续接触/相对位移”，但“受力”代理在试验中错误地偏向 B，因此没有接入该谓词族。10-case 汇总为 9/10 Gold，P10=98.282；唯一保留失败是受力/形变无法由当前单目稀疏画面可靠判定。

## 跨数据集量产审计

当前真实小规模 canary 汇总如下：

| 数据集 | 真实 Gold | Gold rate | Rubric P10 | 三 repair 覆盖 | 非必要工具调用 |
|---|---:|---:|---:|---|---:|
| VideoSham | 17/17 | 100% | 98.295 | 是 | 0 |
| VCSL | 21/22（全历史 teacher-started） | 95.5% | 96.657 | 是 | 0 |
| VidDiffBench | 21/21（teacher-started） | 100% | 98.098 | 是 | 0 |

初始十个 minimal-edit hard negative 检出 10/10 后，又把相同的局部扰动系统地施加到当时 40 个跨数据集 Gold bundle：第一次扩展只检出 308/338，暴露出 `wrong_outcome` 校准字段过拟合 VideoSham；改为翻转各数据集真实的结果 verifier 后全部拒绝。UTQR-3 校准在当前 59 个 Gold bundle 上完整重跑，并新增三类训练上下文攻击：private locator 写入 target、pre-state 不允许的 target action、引用尚不可见的 EvidenceAtom。连同 decision-reopen 过度修复和 VCSL 亚 Gold 定位，共 13 个家族、685 个负例，685/685 均被拒绝，三个数据集各自检出率也均为 100%。这证明当前观测池中的已知最小扰动可被稳定识别，但仍不是对未知扰动的统计置信区间。

统一审计结论是 `quality_canaries_passed_scale_validation_pending`：方法质量已经足以进入更大 canary，而统计证据尚不足以宣布无审查量产。审计是实验建议，不是运行时代码 gate。

### 统一 SFT 量产编译

通过 rubric 的轨迹不能直接按 JSONL 行随机切分。一个案例会产生 8–14 条共享长前缀的 clean/repair 逐轮样本；若相邻轮次进入不同 split，验证集会看到几乎相同的状态与媒体身份，形成严重的验证泄漏。`production_sft_compiler.py` 因此先把共享任一视频实体的案例连成组，再按数据集分层、整组切成 train/validation/test；同一案例的 clean、repair、transition 和 route preference 永远继承同一 split。

当前 59 个真实 Gold 案例编译为 600 条 SFT（367 clean、233 repair）和 600 条一一对应的 verifier transition，另保留 13 个同案例、真实执行的 route preference pair。split 为 47/6/6 个案例；跨 split 的案例重叠、共享媒体实体重叠、完整模型输入重复和去标识化 reasoning 重复均为 0。训练 sampler 只给出受限的逆平方根权重以缓解数据集和动作族失衡，不复制稀有行；failed trajectory、measurement-unreachable case 和 rubric hard negative 只进入评测/偏好侧车，不混入正 SFT。

UTQR-3 进一步补上“轨迹正确但训练上下文低效”的盲区。历史 full transcript 会在每次 tool observation 中重复整份 harness state，且旧 VideoSham/VCSL/VidDiff artifact 中还残留不参与决策的媒体/特征 locator。`decision_sufficient_state_v1` 对每条样本只保留唯一权威 pre-action state、紧凑 action/outcome history 和不变 assistant target，并把 locator 归一化为逻辑引用。600/600 行通过 target、state、action order、eligible tool 和 EvidenceAtom 引用保真检查，private locator/oracle 命中为 0；模型可见字符从 17,791,569 降至 7,132,118，减少 59.9%。完整 transcript 不删除，继续保留在源 artifact 中供 replay 和审计。详细文献映射、创新边界和消融要求见 `RUBRIC_V3_LITERATURE_AND_DESIGN.md`。

readiness 报告同时区分两个口径：`measurement_reachability_rate` 衡量上游工具树能否给出正确候选，`teacher_started_gold_rate` 衡量 teacher 在已启动案例上能否产生 Gold 轨迹。以 VidDiff 30-case 为例，二者分别为 70% 和 100%；前者仍阻止“无审查量产”结论，但不能被误写成 teacher 轨迹 70% 成功。

为了直接检验 compact context 是否更适合 student，现已生成 `student_context_ablation_v2`：full-state-normalized、逐行等上限 naive token-tail、decision-sufficient 三臂共享全部 600 个 target 与相同 split，隐私命中为 0。Qwen tokenizer 下 full 有 125/600 行超过 8,192 token，compact 最大 6,836；naive-tail 的 383 个真实截断行平均使用 compact 上限的 99.98%。Qwen3-VL-8B 4-bit LoRA 的 1-step 本地 smoke 已通过，峰值显存 15.0 GiB；正式三臂长跑和闭环评测尚未执行，因此目前不把 smoke loss 当作方法优越性证据。

### 首轮扩量与 pass^k 进展

每个数据集各选一个代表性硬例做了三次独立完整 rollout：VideoSham 空间删除、VCSL AXIS-SALIENCY、VidDiff 稳定支撑均为 3/3，严格组合估计 `pass^1=pass^2=pass^3=1.0`。不同 route portfolio 没有被计作独立重复。当前覆盖仍只有每数据集一个 case，因此只是稳定性 canary。

VidDiff 已完成确定性 30-case 样本池。索引 10–14 中三个原工具使 3/5 可达，新增独立张力谓词又修复 1 例；入射角工具探针失败后未注册。索引 15–19 中三个原工具覆盖 3/5；缠线拓扑和径向轨迹的独立谓词探针均失败，因此也未注册。索引 20–24 有 2/5 可达，索引 25–29 有 3/5 可达；后两批使用“多路无标签测量 → 本地 oracle-bounded rejection sampling → 单一证据 teacher rollout”，消除了候选工具路由漂移。30-case 去重汇总为 21/30 Gold，九个失败均在 teacher 启动前判为 measurement candidate-set unreachable；实际启动 teacher 的 21 个案例全部 Gold，selected calls 与 indispensable calls 相等。总体 Gold rate 为 70%，attempted-case Gold rate 为 100%，进一步确认瓶颈在上游 measurement 可达率而不是 teacher 轨迹质量。

本地资产审计现在三个数据集都达到至少 30 个可执行案例：VideoSham 有 30 对真实媒体，VCSL 按清单解出 30-case 所需的 58 个 ISC-256d 特征，VidDiff 有 166 个完整手术样本且首个 30-case 扩量已完成。VCSL 的 21 个已选定位由正式工具逐例重算，21/21 区间完全一致；授权的新增 12 条 teacher rollout 全部 Gold，和原有案例去重后为 21/30 Gold、九个 measurement-unreachable。历史上曾对其中一个不可达案例启动 teacher，故全历史口径为 21/22 teacher-started Gold；按当前 rejection sampler 真正启动的可达案例则为 21/21 Gold。

VideoSham 官方 v1.0 release 提供四个 ZIP，分别包含 352 个 trimmed source 和 352 个 manipulated video；数据受 Adobe Research License 1.1 约束，仅限非商业研究。获得用户许可后，`fetch_videosham_subset.py` 使用 HTTP Range 读取远程 ZIP 中央目录并只提取目标成员，没有整包落盘；实际下载 40 个文件、263,252,729 bytes，20/20 对通过容器、首帧、时长、分辨率和工具 preflight。加上原有 10 对后，30-case 覆盖 temporal omission 3、temporal warp 5、audio 4、spatial removal 4、spatial insertion 4、appearance 5、text 5。局部文字/换色最初会被 64×36 全局灰度特征漏掉，因此空间工具加入流式 24 帧、192×108 RGB、12×16 tile 的局部残差，并先消除帧级颜色/编解码中位漂移；最终无标签工具树在全部 30 对上 30/30 可测、30/30 模态路由与事后 verifier 一致。

为减少视频出站依赖，还试验了一个完全本地的绿色线体运动代理：去静态背景后跟踪线体质心轨迹。但在全部 54 个可用 `tube moves more` 谓词上只有 23/54（42.6%），低于多数类 `c` 基线的 50%，因此明确标记为 `safe_to_register_tool=false`。这类失败探针只用于工具淘汰和 hard-negative 研究，不进入生产工具箱或正 SFT。

30-case 的九个不可达谓词都做过针对性探针。径向运动、缠线拓扑、针入射角、受力和两例管道运动的专用探针失败；双抓手效率、靠近针尖的抓取位置和落点准确性也分别做了 V4 teacher 谓词试验或本地几何 holdout，均未达到注册标准。这些失败不能通过换工具名称重新包装，继续保留为 abstain/工具研发样本，不进入正 SFT。

## 当前不能过度宣称的部分

- VideoSham 已有 30 对本地可执行媒体且 30/30 测量路由正确；其中 17 条已得到对应出站授权并生成真实 teacher Gold 轨迹，剩余 13 条仍只有本地 student-visible 测量，不计为正 SFT。
- 每个数据集已有 1 个代表性硬例完成三次独立 rollout：VideoSham 空间删除、VCSL AXIS-SALIENCY、VidDiff 稳定支撑均为 3/3，严格 `pass^2=pass^3=1.0`。但覆盖率只有各数据集 1/10，不能外推为数据集级 `pass^k`。
- VidDiff 的 30-case 中有九个 measurement-unreachable case；应按谓词族保留为失败/工具诊断，仅在独立工具探针通过后再生成正轨迹，不能让 teacher 看答案强行补齐。
- VCSL 的七个定位工具都在 21 个 Gold 中被真实选择且逐步审计为 indispensable，但部分工具仍只有 1–3 例；VideoSham 的 teacher Gold 能力族也只有 1–4 例。
- 因此当前状态是“三数据集质量 canary 过线、VidDiff 与 VCSL 完成首个 30-case、VideoSham 完成 30-case 本地测量并取得 17 条 Gold 轨迹”，不是“所有能力族已经有足够统计量可无审查生产”。

## 批量生产前的判定条件

每个数据集/任务族至少 30 个 held-out case；Gold rate ≥90%，总分 P10≥92；三种 repair 各自成功率≥85%；hard-negative detection 总体≥90%且每类≥80%；导出集硬门失败为 0；报告重复采样 `pass^k`。这些是统计准入建议，不写成阻挡正常实验的代码 gate。

## 下一轮执行顺序

1. 在每个数据集选困难 case 做至少三次独立 rollout，严格区分同策略重复采样与不同 route portfolio，报告 `pass^2`、`pass^3` 和失败相关性。
2. 扩充每类 minimal-edit hard negative，不只保留单个扰动 canary。
3. 对 VideoSham 剩余 13 对继续先审阅 `student_measurements.jsonl`；只有获得单独出站授权后才发送 student-visible 文本测量给 teacher，任务文本、数据集标签、本机路径和 oracle/verifier 仍不发送。
4. 为 VideoSham 每个能力族扩充独立重复 rollout；当前 teacher Gold 只有 1–4 例/族，只足以做 canary，不足以估计稳定的族级成功率。
5. 对 VidDiff 受力谓词开发可验证的时序形变/接触动力学工具；验证前继续输出 abstain/失败诊断，不纳入正 SFT。
6. 最终报告 P10、各 repair 成功率、`pass^k`、工具不可缺少率和单位 Gold 成本。
