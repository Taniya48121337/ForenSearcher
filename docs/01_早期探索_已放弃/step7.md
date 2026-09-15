# Step 7 — Delta and Verdict

Timestamp: 2026-08-16 Asia/Shanghai

## Verdict

**Level 2 — High Overlap**

OpenVisTool 与本项目在 visual tool trajectory synthesis、因果证据质量和多模态应用三个轴上高度相近；CausalFlow 又覆盖 intervention + suffix replay + minimal repair supervision；MetaRAG 覆盖 atomic claim mutation；Faulty-Memory work 覆盖 typed provenance、independent support 与 selective replay。因此原始宽主张只剩下“结构化视频取证 claim field 的选择性因果契约”这一条清晰差异轴，必须以它为核心重构方法和论文标题。

## Delta

> Unlike OpenVisTool, which tests whether an entire visual tool trace improves a scalar final answer over a no-tool baseline, the proposed work intervenes on individual source, interval, direction, entity, and evidence dependencies and requires targeted invalidation of the affected claim field while independently supported fields remain invariant, yielding automatically verified local revise/reject/reacquire supervision rather than trajectory filtering alone.

## Recommended reframing

建议将方法从笼统的 **Claim-Causal Mutation-Tested Trajectory Compilation** 收缩并改名为：

**ClaimUnitTest: Field-Selective Evidence Intervention for Video-Forensic Trajectory Compilation**

新的 load-bearing mechanism 是一个 evidence–claim intervention matrix：

- 行是结构化 claim fields：source identity、operation、entity/event、start/end interval、direction、evidence status。
- 列是可干预的 evidence atoms 或 provenance edges。
- 每次只变异一个 evidence dependency 并重放后缀。
- **Target sensitivity**：被该证据支持的字段必须 revise / reject / reopen / abstain。
- **Off-target invariance**：具有独立支持的其他字段必须保持不变。
- **Sufficiency**：保留的最小 evidence slice 单独重放仍能恢复目标字段。
- 将实际状态变化直接编译为局部 obligation-transition SFT，而不是让教师自由写一段“反思”。

这一定义避开了四个已被覆盖的宽主张：whole-trace causal utility、generic step repair、static factoid mutation、generic dependency rollback。其具体可测收益是更高的局部修正率、更低的无关字段坍塌率、更少的 premature stop，以及更好的 claim–evidence completeness。
