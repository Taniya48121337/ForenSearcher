# Step 6 — Comparison

Timestamp: 2026-08-26 00:32:49 +08:00

- **Proposed work**
  - Title: Dynamic Working Memory Tree for transferable heterogeneous video verification
  - Date: proposed
  - Source: current project
  - Problem framing: one Deep Research Agent for multiple video-verification tasks and unseen-task transfer.
  - Core mechanism: each model-generated local task owns a full Harness workspace; derivation edges are untyped; evidence is globally shared.
  - Key insight: isolate investigation contexts at the workspace level while preserving reusable evidence and generic task generation.
  - Application domain: heterogeneous video verification.

- **ToE**
  - Title: ToE
  - Date: 2026-06
  - Source: https://arxiv.org/abs/2606.27736
  - Problem framing: automated fact-checking.
  - Core mechanism: dynamic claim/subclaim tree with per-node evidence retrieval and evaluation.
  - Key insight: recursively decompose uncertain claims and aggregate scores bottom-up.
  - Application domain: textual claims.
  - Axes matching: 3/4; **Level 2 — High Overlap (one axis differs)**.

- **Trification**
  - Title: Trification
  - Date: 2025-11
  - Source: https://arxiv.org/abs/2512.00267
  - Problem framing: automated fact-checking.
  - Core mechanism: dynamic verification-action DAG with dependencies and replacement subgraphs.
  - Key insight: structure and dynamically repair the verification plan.
  - Application domain: textual complex claims.
  - Axes matching: 3/4; **Level 2 — High Overlap (one axis differs)**.

- **TreeSeeker**
  - Title: TreeSeeker
  - Date: 2026-06
  - Source: https://arxiv.org/abs/2606.11662
  - Problem framing: deep search under competing directions.
  - Core mechanism: sub-goal trees, branch-local memory, explore/exploit/prune scheduling.
  - Key insight: keep evidence and failure state attached to the branch that produced them.
  - Application domain: web-search QA.
  - Axes matching: 2/4; **Level 3 — Medium Overlap (two axes differ)**.

- **ReTree**
  - Title: Self-Correcting Long-Horizon Search Agents via Tree-Structured Memory
  - Date: 2026-08
  - Source: https://arxiv.org/abs/2608.10676
  - Problem framing: long-horizon search memory and repair.
  - Core mechanism: tree-structured working memory over per-step states.
  - Key insight: evidence lineage allows bounded context and structural repair.
  - Application domain: QA/search.
  - Axes matching: 2/4; **Level 3 — Medium Overlap (two axes differ)**.

- **MERMAID**
  - Title: MERMAID
  - Date: 2026-01
  - Source: https://arxiv.org/abs/2601.22361
  - Problem framing: veracity assessment with iterative retrieval.
  - Core mechanism: persistent shared evidence memory, no task tree.
  - Key insight: reuse evidence across claims and couple retrieval with reasoning.
  - Application domain: fact-checking.
  - Axes matching: 2/4; **Level 3 — Medium Overlap (two axes differ)**.

- **LoCal**
  - Title: LoCal
  - Date: 2025
  - Source: https://doi.org/10.1145/3696410.3714748
  - Problem framing: complex fact-checking.
  - Core mechanism: model-generated subtasks delegated to specialized agents, then iteratively evaluated.
  - Key insight: logical and counterfactual evaluation repairs decomposition errors.
  - Application domain: textual fact-checking.
  - Axes matching: 2/4; **Level 3 — Medium Overlap (two axes differ)**.

- **DEFAME**
  - Title: DEFAME
  - Date: 2025
  - Source: https://arxiv.org/abs/2412.10510
  - Problem framing: multimodal open-domain fact-checking.
  - Core mechanism: dynamic six-stage tool pipeline and one evolving report.
  - Key insight: dynamically acquire evidence until the report supports a verdict.
  - Application domain: text-image fact-checking.
  - Axes matching: 2/4; **Level 3 — Medium Overlap (two axes differ)**.

Overall minimum level: **Level 2 — High Overlap**.
