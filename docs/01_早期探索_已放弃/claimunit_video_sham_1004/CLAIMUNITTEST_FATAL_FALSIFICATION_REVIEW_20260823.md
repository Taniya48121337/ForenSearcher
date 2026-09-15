# ClaimUnitTest fatal-falsification review (2026-08-23)

## Bottom line

The strongest potentially fatal issue is **control-plane sufficiency**: the current
student-visible harness may already reveal enough state, eligible actions, exact
argument affordances, and claim-domain constraints for a policy to pass without
using the semantic content of video-derived evidence.

If a structure-only policy performs close to the full policy, the evidence-grounded
learning interpretation is false. The surviving contribution would be a reliable
stateful harness and trajectory compiler, not learned field-selective causal
reasoning from video evidence.

This is not hypothetical in the current 272-turn UTQR-6 export:

- 165/272 turns expose exactly one eligible tool;
- 90/272 turns contain at least one compiled binding, decision, or restoration
  action in `state_action_affordances`;
- 117/272 turns expose a singleton enum for at least one target field, covering
  19 cases;
- the compression contract says compiled actions are authoritative and should be
  copied exactly;
- examples explicitly state that a named tool is the only eligible action.

These features are legitimate engineering aids, but they confound the scientific
claim that the student learned to use evidence content causally.

## Minimal decisive experiment

Run the same sealed case set under four views, without retraining the evaluator:

1. **Full**: current state, evidence content, schemas, and affordances.
2. **Structure-only**: keep IDs, evidence kinds, active/dropped status, obligations,
   eligible tools, and compiled affordances; replace measurement values and
   rationales with opaque placeholders.
3. **Evidence-only**: retain real observations and public schemas, but remove
   singleton target enums, exact compiled actions, restoration hints, and
   single-choice tool routing whenever more than one action is semantically valid.
4. **Shuffled evidence**: permute measurement contents across cases while keeping
   the same control-plane structure. Correct behavior should change or abstain.

Measure full closed-loop terminal success, field correctness, tool selection,
evidence binding, sensitivity, invariance, abstention, and unnecessary tool cost.
Do not use teacher-forced NLL as the decisive endpoint.

### Falsification rule

- If `structure-only` is close to `full`, ClaimUnitTest has not shown evidence
  grounding; it has shown contract following.
- If shuffled evidence does not cause appropriate claim changes or abstention,
  the policy is not consuming evidence semantics.
- If removing compiled affordances destroys performance while retaining evidence
  content, the main capability resides in the harness, not the learned policy.

## Additional threats that invalidate current empirical conclusions

### Adaptive validation overfitting

The current six case-disjoint validation cases have been repeatedly inspected and
used to alter training curricula, identifier semantics, token budgets, and the
evaluator. They are no longer a final test set. Case disjointness is insufficient;
the final test must also be development-process disjoint and opened only after the
method and scoring rules are frozen.

### Announced repair versus error discovery

Evidence-dropout repair often exposes `dropped_evidence_ids`, unsupported fields,
`must_reopen_before_rebinding`, eligible restoration tools, or complete repair
actions. Passing these cases proves response to an announced state change, not the
ability to detect that evidence is wrong. A decisive transfer test must include
unannounced wrong intervals, source swaps, contradictory observations, stale but
active evidence, and natural tool failures.

### Circular mutation oracle

When the same hand-authored evidence-to-field contract defines the mutation,
renders the visible state, and scores the expected field change, sensitivity and
invariance can become a consistency test of that contract. Independent
counterfactual worlds produced by real re-execution, plus held-out dependency
topologies, are needed before using strong causal language.

### Verifier-conditioned selection

The VidDiff external panel has 13/30 end-to-end safe coverage, although conditional
trajectory validity is 13/13. Oracle/verifier fields may be absent from SFT rows,
but label-dependent selection still changes which cases and paths enter training.
Claims must therefore use full-panel coverage and separately report conditional
trajectory quality.

## Claims that survive even if the fatal test fails

- The project implements a real stateful video-tool harness.
- It can compile executable clean and repair traces.
- It has strong programmatic auditing, quarantine, replay, and leakage controls.
- Exact executable affordances improve interface reliability.

The claims that would not survive are that current 100/100 trajectories demonstrate
semantic video-evidence reasoning, that field-selective causal dependence has been
learned, or that the SFT corpus teaches general repair behavior.

## Review process note

The local `research-review` skill was used for its falsification framework. Its
configured external Codex reviewer tools and referenced routing/tracing files were
not available in this environment, so this record is an internal evidence-backed
review rather than an independent external review.
