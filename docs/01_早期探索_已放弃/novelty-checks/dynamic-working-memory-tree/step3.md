# Step 3 — Abstract-Level Triage

Timestamp: 2026-08-26 00:32:49 +08:00

## Structured papers

### 1. ToE

- Date: 2026-06
- Problem framing: automated claim verification with heterogeneous web evidence.
- Core mechanism: dynamically expanding claim/subclaim argument tree; per-node retrieval and evidence evaluation; bottom-up score aggregation.
- Key insight: low-reliability claims should be recursively decomposed and verified.
- Application domain: textual fact-checking.
- Overlap score: 3/4.
- Source: https://arxiv.org/abs/2606.27736

### 2. Trification

- Date: 2025-11
- Problem framing: complex claim verification requiring complete coverage and logical dependencies.
- Core mechanism: tree planner produces a dynamically modifiable action DAG with SEARCH, REFINE, THINK and JUDGE nodes.
- Key insight: explicitly structured dependencies permit parallel execution and recovery after evidence failure.
- Application domain: textual fact-checking on HOVER and FEVEROUS.
- Overlap score: 3/4.
- Source: https://arxiv.org/abs/2512.00267

### 3. LoCal

- Date: 2025
- Problem framing: complex claims requiring multi-hop evidence and logical/causal consistency.
- Core mechanism: decomposing agent, specialized reasoning agents and evaluating agents; iterative correction.
- Key insight: decomposition results require logical and counterfactual checking.
- Application domain: textual fact-checking.
- Overlap score: 2/4.
- Source: https://doi.org/10.1145/3696410.3714748

### 4. DEFAME

- Date: 2025
- Problem framing: open-domain text-image claim verification with retrieved multimodal evidence.
- Core mechanism: one dynamic six-stage MLLM pipeline with an evolving fact-check report and multimodal tools.
- Key insight: dynamically plan tools and search depth rather than use a static classifier.
- Application domain: multimodal fact-checking.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2412.10510

### 5. ProgramFC

- Date: 2023-07
- Problem framing: complex textual claims requiring multi-step evidence reasoning.
- Core mechanism: LLM generates a sequential reasoning program and delegates steps to QUESTION, VERIFY and PREDICT handlers.
- Key insight: explicit programs improve explainability and few-shot data efficiency.
- Application domain: HOVER and FEVEROUS textual fact-checking.
- Overlap score: 2/4.
- Source: https://aclanthology.org/2023.acl-long.386/

### 6. MERMAID

- Date: 2026-01
- Problem framing: veracity assessment with dynamic retrieval and repeated evidence use.
- Core mechanism: multi-agent Reason–Action loop plus persistent evidence memory and cross-claim reuse.
- Key insight: retrieval should be coupled to reasoning and remembered rather than repeated independently.
- Application domain: fact-checking and claim verification.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2601.22361

### 7. SUCEA

- Date: 2025-06
- Problem framing: adversarial claims that are difficult to retrieve evidence for directly.
- Core mechanism: claim segmentation, iterative evidence retrieval with claim editing, then evidence aggregation.
- Key insight: evidence-guided rewriting improves retrievability of decomposed claims.
- Application domain: textual adversarial fact-checking.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2506.04583

### 8. RAMA

- Date: 2026
- Problem framing: ambiguous multimodal misinformation requiring web evidence.
- Core mechanism: query formulation, cross-source evidence aggregation and multimodal multi-agent ensemble.
- Key insight: authoritative web retrieval plus complementary agents improves trustworthiness.
- Application domain: multimodal fact-checking.
- Overlap score: 1/4.
- Source: https://arxiv.org/abs/2507.09174

### 9. Decomposition Dilemmas

- Date: 2025-04
- Problem framing: evaluates when claim decomposition helps or hurts downstream fact-checking.
- Core mechanism: empirical comparison and taxonomy of decomposition errors.
- Key insight: decomposition introduces a trade-off between coverage and noise.
- Application domain: fact-checking and factuality evaluation.
- Overlap score: 1/4.
- Source: https://aclanthology.org/2025.naacl-long.320/

### 10. GEAR

- Date: 2019-07
- Problem framing: aggregate multiple retrieved evidence sentences for FEVER verification.
- Core mechanism: fully connected evidence graph plus neural aggregation.
- Key insight: evidence pieces should communicate rather than be concatenated independently.
- Application domain: FEVER fact verification.
- Overlap score: 1/4.
- Source: https://aclanthology.org/P19-1085/

### 11. Reasoning Over Semantic-Level Graph for Fact Checking

- Date: 2020-07
- Problem framing: reason over multiple pieces of textual evidence.
- Core mechanism: semantic-role graph with graph convolution and graph attention.
- Key insight: semantic structure between words and evidence improves verification.
- Application domain: FEVER fact checking.
- Overlap score: 1/4.
- Source: https://aclanthology.org/2020.acl-main.549/

### 12. DTCA

- Date: 2020-07
- Problem framing: transparent evidence selection for explainable claim verification.
- Core mechanism: learned decision tree selects credible comments, followed by co-attention.
- Key insight: make evidence discovery interpretable and focus it on false claim parts.
- Application domain: RumourEval and PHEME.
- Overlap score: 1/4.
- Source: https://aclanthology.org/2020.acl-main.97/

### 13. TreeSeeker

- Date: 2026-06
- Problem framing: multi-step deep web search under competing search directions and limited budget.
- Core mechanism: sub-goal trees, branch-local TreeMem, and TreeSearch decisions to explore, exploit or prune/return.
- Key insight: separate branch-local evidence and failure state so the controller can compare investigation directions.
- Application domain: deep-search QA.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2606.11662

### 14. ReTree

- Date: 2026-08
- Problem framing: long-horizon search with growing context and retrieval-induced error cascades.
- Core mechanism: tree-structured working memory whose nodes store task-state summaries, local evidence and revision history; contradictions repair an ancestor and prune descendants.
- Key insight: evidence provenance and dependent-state repair require structured memory rather than a flat report.
- Application domain: multi-hop QA and search agents.
- Overlap score: 2/4.
- Source: https://arxiv.org/abs/2608.10676
