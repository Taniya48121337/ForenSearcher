# Targeted Paper Search — Dynamic Trees in Fact-Checking

Timestamp: 2026-08-26 00:32:49 +08:00

## Search errors surfaced

- OpenReview connector: `openreview not installed. pip install openreview-py`.
- Initial sandboxed database calls: Windows socket permission failure.
- Approved searches: intermittent API rate limiting.
- Initial console rendering: GBK UnicodeEncodeError; corrected by UTF-8 execution.

## Semantic Scholar / arXiv highlights

| Title | Date | Venue | Relevance |
|---|---:|---|---|
| ToE | 2026 | arXiv preprint | Dynamic claim/subclaim fact-checking tree |
| Trification | 2025 | arXiv preprint | Dynamic verification-action dependency graph |
| DEFAME | 2025 | ICML | Dynamic multimodal tool pipeline; no tree |
| MERMAID | 2026 | arXiv preprint | Persistent evidence memory and cross-claim reuse |
| SUCEA | 2025 | arXiv preprint | Claim decomposition and iterative retrieval |
| RAMA | 2026 | WWW Companion | Retrieval-augmented multimodal multi-agent verification |
| TreeSeeker | 2026 | arXiv preprint | Sub-goal trees and branch-local search memory |
| ReTree | 2026 | arXiv preprint | Tree-structured working memory over search states |

## OpenAlex / ACL / Crossref highlights

| Title | Date | Venue | Relevance |
|---|---:|---|---|
| ProgramFC | 2023 | ACL | Sequential subtask program and fixed handlers |
| LoCal | 2025 | WWW | Decomposition, specialized agents and iterative evaluators |
| Decomposition Dilemmas | 2025 | NAACL | Shows decomposition can introduce harmful noise |
| GEAR | 2019 | ACL | Evidence graph neural aggregation |
| Reasoning Over Semantic-Level Graph | 2020 | ACL | Semantic evidence graph reasoning |
| DTCA | 2020 | ACL | Learned decision tree for evidence selection |

## Overview

Three complementary queries covering fact-checking trees, hierarchical agents and working-memory trees were searched over 2018–2026. The retrieved literature separates into evidence graphs, claim-decomposition systems, dynamic task/action graphs, multimodal tool pipelines, persistent evidence memory, and generic deep-search tree memory.

## Trends

- 2019–2021: graph neural networks primarily organize evidence representations.
- 2023–2025: LLM systems decompose claims into programs or agent subtasks.
- 2025–2026: dynamic verification graphs, persistent evidence memory and branch-controlled search become explicit.
- 2026: both fact-checking-specific dynamic trees and general tree-structured agent memory appear, making a bare tree claim difficult to defend.

## Key themes

1. Evidence graphs: GEAR, Semantic-Level Graph, DTCA.
2. Claim decomposition: ProgramFC, LoCal, SUCEA, Decomposition Dilemmas.
3. Dynamic fact-checking trees: ToE, Trification.
4. Dynamic multimodal verification: DEFAME, RAMA.
5. Agent memory and search trees: MERMAID, TreeSeeker, ReTree.

## Keywords frequency in the relevant set

| Keyword | Count |
|---|---:|
| evidence | 14 |
| fact-checking / verification | 12 |
| decomposition / subtask | 7 |
| tree / graph | 7 |
| memory / reuse | 4 |

## Reading path

1. ProgramFC — foundational LLM subtask delegation in fact-checking.
2. LoCal — iterative multi-agent decomposition and evaluation.
3. DEFAME — dynamic multimodal evidence acquisition without a tree.
4. Trification and ToE — the closest fact-checking tree competitors.
5. TreeSeeker and ReTree — the closest mechanism competitors for branch-local and tree-structured search memory.
