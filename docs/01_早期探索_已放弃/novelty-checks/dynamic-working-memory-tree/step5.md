# Step 5 — Full-Paper Deep Dive

Timestamp: 2026-08-26 00:32:49 +08:00

The bundled `fetch_paper.sh` could not be used because Bash is unavailable in the Windows environment. Primary-source arXiv HTML/PDF, ACL Anthology PDF and official proceedings pages were used as the documented fallback.

## ToE — verified

- Problem: claim verification with heterogeneous web retrieval; evaluated on LIAR, PolitiFact, Check-COVID and AdvFact.
- Mechanism: node is a claim or subclaim; selected nodes run multi-source retrieval and receive veracity/reliability scores; low reliability triggers 2–4 child subclaims; scores propagate bottom-up; queue priority uses parent uncertainty and node importance.
- Scope: tree semantics are argument decomposition and verdict aggregation; maximum depth is fixed at 5; retrieval action space has eight predefined source categories.
- Closest evidence: Method §III and Tree Management §III-D, especially arXiv HTML lines 75–83 and 171–242.
- Refined overlap: framing match; mechanism partial match; key insight match; domain partial match.

## Trification — verified

- Problem: complex claim verification over HOVER and FEVEROUS.
- Mechanism: initial tree planner produces a DAG; nodes are fixed SEARCH, REFINE, THINK and JUDGE actions; edges are execution dependencies; parent outputs propagate to children; failed THINK/JUDGE nodes generate a replacement subtree.
- Scope: nodes are single typed actions, not persistent multi-round workspaces; the structure is a DAG and can have multiple parents.
- Closest evidence: §3.1–3.4, especially arXiv HTML lines 92–125 and 139–221.
- Refined overlap: framing match; mechanism partial match; key insight match; domain partial match.

## TreeSeeker — verified

- Problem: deep web search with several plausible directions and limited budget.
- Mechanism: root question is decomposed into subgoals; each subgoal has a search tree; TreeMem stores branch-local evidence, uncertainty, progress and failure cues; deeper nodes contain recent tool calls and observations; TreeSearch chooses EXPLORE, EXPLOIT or PRUNE.
- Scope: branch-local state is a scheduling object, not a full autonomous Harness workspace; goals and search-path trees are separate levels.
- Closest evidence: PDF §3.1–3.3, especially pages 2–4.
- Refined overlap: framing partial; mechanism match; key insight match; domain differs.

## ReTree — verified

- Problem: bounded context and evidence correction in long-horizon search.
- Mechanism: explicitly named tree-structured working memory; each node stores summary, local evidence, revision history and parent pointer; each compatible retrieval step creates a child state; contradictions repair the evidence-introducing ancestor and prune descendants.
- Scope: one node per search-state update, not one node per local task workspace; the tree serves state lineage and repair, not task decomposition.
- Closest evidence: §1 and §4.1–4.3, especially arXiv HTML lines 59–63 and 129–152.
- Refined overlap: framing partial; mechanism match; key insight partial; domain differs.

## MERMAID — verified at abstract/full-text preview level

- Problem: veracity assessment with redundant retrieval across claims.
- Mechanism: persistent evidence memory inside a multi-agent iterative retrieval/reasoning loop, supporting cross-claim evidence reuse.
- Scope: no task tree and no local workspace node abstraction.
- Closest evidence: arXiv abstract and full-text preview.
- Refined overlap: framing match; mechanism partial; key insight partial; domain match.

## LoCal — verified

- Problem: complex multi-hop fact-checking with logical and causal errors.
- Mechanism: decomposer emits a sequence of fact-verification or QA subtasks plus a deduction function; specialized agents solve them; two evaluators trigger iteration.
- Scope: sequence and agent roles rather than a dynamic tree of persistent workspaces.
- Closest evidence: WWW 2025 official record and full-paper method §3.2.
- Refined overlap: framing match; mechanism partial; key insight partial; domain match.

## DEFAME — verified

- Problem: open-domain text-image fact-checking with textual and visual evidence.
- Mechanism: single evolving fact-check report and six-stage Plan–Execute–Summarize–Develop–Predict–Justify loop; NEI triggers another retrieval round.
- Scope: dynamic depth and tools, but one report/working state and no tree.
- Closest evidence: ICML paper §3, especially arXiv HTML lines 127–163.
- Refined overlap: framing partial; mechanism partial; key insight partial; domain match.
