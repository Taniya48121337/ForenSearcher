# Step 2 — Search and Deduplicate

Timestamp: 2026-08-16 Asia/Shanghai

## Queries

1. **Original-Problem**: `video forensic agent evidence trajectory synthesis source suspect manipulation claim`
2. **Broad-Domain**: `agent tool-use trajectory synthesis verification causal utility`
3. **Method-Signature**: `claim evidence mutation counterfactual replay repair supervision`

补充精确查询：

- `"claim-conditioned" "mutation testing" agent trajectory`
- `"evidence mutation" LLM agent trajectory repair`
- `"claim field" counterfactual evidence intervention LLM`
- `"metamorphic testing" LLM agent evidence claims`
- `"selective invalidation" claim evidence provenance agent`

## Search execution notes

- 按要求尝试使用 paper-search 跨 arXiv、Semantic Scholar、OpenAlex、OpenReview、Crossref 与 DBLP 检索。
- 首次沙箱运行中，各外部 API 因 Windows socket 权限失败；批准外网后，OpenReview 仍报告 `openreview not installed. pip install openreview-py`，OpenAlex 出现 `504 Server Error: Gateway Timeout`，其余接口出现反复限流，CLI 最终被中止。上述零结果不被视作负面查新证据。
- 随后使用公开网页检索并读取 arXiv HTML / OpenReview PDF 原文，覆盖相同三组问题，并对标题和 arXiv ID 去重。
- 精确短语检索没有发现题名或摘要直接声称 `claim-conditioned evidence-edge mutation + field-selective invalidation + repair-trajectory compilation` 的论文；但发现多个分别覆盖其组成部分的强近邻工作。

## Deduplicated corpus

1. OpenVisTool: An Open Recipe for Synthesizing Instructive Visual Tool-Use Trajectories — arXiv:2608.08557
2. CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures — arXiv:2605.25338
3. From Faulty Memories to Corrected Actions: Dependency-Guided Rollback Repair for Memory-Augmented Agents — arXiv:2608.10502
4. Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search — arXiv:2607.15253
5. Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures — arXiv:2606.08275
6. STAMP: Provenance-Guided Credit Assignment for Deep Search Agents — arXiv:2607.11172
7. Controllable and Verifiable Tool-Use Data Synthesis for Agentic Reinforcement Learning (COVERT) — arXiv:2604.09813
8. MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems — arXiv:2509.09360
9. From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training RAG Agents (EviPath) — OpenReview / NeurIPS 2025 Efficient Reasoning Workshop
10. From Failure to Mastery: Generating Hard Samples for Tool-use Agents (HardGen) — OpenReview ACL ARR 2026
11. InfCycle: Learning to Use Tools via Inference Compute and Cycle Consistency — OpenReview ICLR 2025 withdrawn submission
12. From Agent Traces to Trust: Evidence Tracing and Execution Provenance in LLM Agents — arXiv:2606.04990
13. From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents — arXiv:2602.13855
14. VidForensics-M1: Meta-Detection Reinforcement Learning with Verifiable Temporal Grounding for AI-Generated Video Forensics — arXiv:2608.11201
15. VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection — arXiv:2603.21526
16. Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents — arXiv:2402.11651

## Model-recall augmentation

没有加入未经在线核实的 model-recall 条目。程序切片、变异测试与蜕变测试作为跨领域基础机制在分析中使用，但不伪造具体论文元数据。
