# Step 4 — High-Potential Candidates

Timestamp: 2026-08-16 Asia/Shanghai

选出 7 篇进行正文级核验：

1. **OpenVisTool** — 同为视觉工具轨迹构造，且直接以 causal utility 判断 SFT 数据质量；是问题设置和核心洞察上最危险的单篇近邻。
2. **CausalFlow** — 已覆盖 step intervention、suffix re-execution、minimal repair 和 learning-ready contrastive supervision；直接威胁“变异生成修复轨迹”主张。
3. **MetaRAG** — 已覆盖 atomic factoid decomposition、controlled mutations 与 claim-level verification；直接威胁“逐 claim 变异测试”主张。
4. **From Faulty Memories to Corrected Actions** — 已覆盖 typed provenance graph、独立支持检查、选择性 invalidation 和局部 replay；直接威胁“依赖图 + 反向切片 + 局部修复”主张。
5. **Bridge Evidence** — 已覆盖单证据删除和从干预点重放；直接威胁“删除证据来测试因果性”主张。
6. **STAMP** — 已覆盖训练时 evidence graph、claim/relation 支持验证和 evidence-to-action provenance credit；威胁“证据图用于训练信号”主张。
7. **COVERT** — 已覆盖 oracle-preserving 受控扰动、错误工具输出与自动可验证训练环境；威胁“对工具观察施加可控噪声”主张。

EviPath 虽与“从已有答案和证据反向合成轨迹”高度相关，但其方法不做反事实干预或局部失效，因此在 7 篇上限下作为结构化语料中的背景近邻保留，不进入最危险正文组。VidForensics-M1 与 VIGIL 属于同领域近邻，但机制分别偏向时间 grounding 奖励和部位级检测，未覆盖轨迹因果测试。
