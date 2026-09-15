# 多数据集 CLI Canary 报告（2026-08-18）

## 结论

当前满足“素材/测量可取得、工具能执行、输出可由隐藏标注后验验证”的训练数据集有三套：VideoSham、VCSL 与 VidDiffBench。前两套均按固定随机种子 `20260818` 跑完 10 个 case 的真实 Codex CLI teacher rollout；VidDiffBench 先在随机 10 例上执行三种 label-blind 测量分支，再由隐藏标签做后验拒绝采样，保留 8 例并完成真实 CLI clean/repair rollout。

VidDiffBench 没有任何单一测量分支达到 8/10，因此这些分支不能作为无监督在线判断器直接注册；但 best-of-tool-tree 的 8 条真实正确测量可安全用于有标签数据集的离线 SFT 构造，另外 2 例保持拒收。

| 数据集 | 10-case 阶段 | 结果 | 是否进入 SFT |
|---|---:|---:|---:|
| VideoSham | 完整 CLI clean + repair | 10/10 passed | 是，110 条逐轮样本 |
| VCSL | 完整 CLI clean + repair | 10/10 passed | 是，110 条逐轮样本 |
| VidDiffBench / JIGSAWS | 三分支 canary + verifier 拒绝采样 + CLI | 8/10 accepted，8/8 rollout passed | 是，88 条逐轮样本 |

## VideoSham

- 随机样本覆盖空间删除/插入、时间编辑和音频替换。
- 10 个 clean episode 各 6 个有效动作，10 个 repair episode 各 5 个有效动作。
- 60 条 clean SFT + 50 条 repair SFT。
- 10/10 通过；0 个预检阻塞；0 个 CLI 启动后失败。
- 工具选择次数与“删除该步后轨迹是否仍成立”的不可删除次数完全一致：
  - `inspect_media`: 10 / 10
  - `compare_spatial_regions`: 8 / 8
  - `detect_temporal_edit`: 6 / 6
  - `compare_audio_tracks`: 6 / 6
  - `create_candidate_claim`: 10 / 10
  - `bind_evidence`: 20 / 20
  - `set_field_decision`: 30 / 30
  - `stop`: 20 / 20
- 已实现但未使用工具：无。

报告：`outputs/batch/videosham_seed_20260818_n10/batch_report.json`

## VCSL

- 使用作者发布的 ISC 256 维逐帧特征；工具执行时不读取复制区间标注。
- 先对 10 个随机正样本执行无 LLM 的测量 canary：10/10 区间召回超过 0.5，平均 precision 0.6670、recall 0.9907、F1 0.7206。
- 测量通过后才启动 CLI teacher；10 个完整 clean + repair episode 全部通过。
- 60 条 clean SFT + 50 条 repair SFT。
- 工具选择次数与不可删除次数完全一致：
  - `inspect_pair_features`: 10 / 10
  - `localize_copy_segment`: 20 / 20（clean 与 repair 各一次）
  - `create_candidate_claim`: 10 / 10
  - `bind_evidence`: 20 / 20
  - `set_field_decision`: 30 / 30
  - `stop`: 20 / 20
- 已实现但未使用工具：无；被选择但可删除的工具步骤：无。

测量报告：`outputs/probes/vcsl_isc_hv_seed_20260818_n10/probe_report.json`

CLI 报告：`outputs/batch/vcsl_seed_20260818_n10/batch_report.json`

## VidDiffBench

- 使用同一批 10 对真实 JIGSAWS 手术视频，三个 measurement canary 均不向工具暴露 `differences_gt` 或 `retrieval_frames`：
  - v1 均匀时间线单次比较：2/10。
  - v2 全局时间线 + 自动高运动局部 + A/B、B/A 对称复核：5/10。
  - v3 使用 v2 两次盲检索区间做高密度细查：4/10，说明错误关注区间会被级联放大，因此回退。
- 三种真实测量的正确结果并集覆盖 8/10。离线 compiler 在所有候选执行完成后才读取隐藏标签，保留一条真实正确 observation，拒收 `surgery_159` 与 `surgery_60`。
- 学生导出无 oracle/GT/retrieval-frame 字段；8 个 accepted case 的 Harness CLI clean/repair 全部通过，共 48 条 clean SFT + 40 条 repair SFT。
- accepted 工具选择次数与不可删除次数一致：
  - `inspect_comparison_task`: 8 / 8
  - `compare_uniform_timeline`: 4 / 4
  - `compare_global_motion_symmetric`: 10 / 10
  - `compare_action_refined_symmetric`: 2 / 2
  - `create_candidate_claim`: 8 / 8
  - `bind_evidence`: 16 / 16
  - `set_field_decision`: 24 / 24
  - `stop`: 16 / 16
- 重要边界：这是**有标签数据集的离线 rejection sampling**，不能把三种视觉分支包装成已经达到 8/10 的自主在线工具。

v1/v2/v3 报告：`outputs/probes/viddiff*_surgery_seed_20260818_n10/probe_report.json`

选择审计：`outputs/selection/viddiff_best_of_tree_seed_20260818_n10/selection_audit.json`

CLI 报告：`outputs/batch/viddiff_verified_seed_20260818_accepted_8/batch_report.json`

## 其余候选数据集为什么没有启动 CLI

这些不是被遗忘，而是尚未同时满足“可取得输入 + 可执行工具 + 可验证输出”：

- VCDB：官方当前下载页面不可用，不能构造可复现的本地输入。
- QVHighlights：公开视频包体量约 135GB，当前还没有通过 canary 的 query-conditioned moment localizer。
- AVID：当前可取得仓库主要是标注，缺少已核验的媒体闭环。
- VideoDR：样本依赖开放网页视频 URL 与搜索/浏览环境，当前离线 Harness 尚无可复现 adapter。
- EVID-Bench：按项目设计保留为最终评测集，不进入 SFT。

不应为这些数据集先让 CLI 生成“纸面轨迹”；应先把相应外部工具做成有真实 observation 的环境，再各跑 10 个 measurement canary。

## 工具保留、修改与删除规则

单个 case 没用到某个工具，不等于该工具无用。树状 registry 本来就应只暴露当前数据集和当前状态需要的分支。

1. **保留**：在至少一种已覆盖能力/数据集中被选中，且反事实删除后无法得到同样合法终局。
2. **修改并暂不注册**：工具可以执行，但测量准确率不足，或输出没有被后续证据绑定消费。VidDiff 第一版属于此类。
3. **合并或删除**：在代表性数据集上有充分暴露机会却始终零使用，且没有独特 observation；或者每次被调用都可删除而不影响结果。
4. **不奖励调用次数**：只有“选择次数 = 不可删除次数”的动作进入 accepted SFT。

当前结论是：VideoSham 的 8 个工具和 VCSL 的 6 个工具全部保留；VidDiff 的三个测量分支只保留在“隐藏标签后验验证”的离线构造树中，不能当作无验证在线判断器。accepted 轨迹中的每个工具都通过不可删除审计，没有依据删除；两例全分支失败样本则保留为后续工具改进集。

## 验证

- 自动化测试：47/47 passed。
- VCSL accepted SFT 中未出现 `oracle_boxes` 或官方 label 文件名。
- VidDiff accepted SFT 中未出现 oracle、GT、`differences_gt` 或 `retrieval_frames`。
- 三套正式 CLI 批次合计产生 308 条逐轮 SFT 样本。
