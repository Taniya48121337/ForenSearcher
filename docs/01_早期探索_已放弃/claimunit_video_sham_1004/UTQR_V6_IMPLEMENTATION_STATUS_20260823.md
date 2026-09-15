# UTQR-6 trajectory generation status (2026-08-23)

## What is now working

- A unified UTQR-6 rubric separates executable machine checks, blind human-style review,
  and batch-production evidence.
- The current cross-dataset panel contains 40 episodes and 288 executed native-tool turns.
- All 40 episodes score 100/100 under the strict closed-loop audit and pass the current
  independent semantic/privacy re-audit.
- The export contains 272 diversity-limited SFT turns after capping exact repeated reasoning
  per tool; the complete unfiltered package contains 288 valid turns.
- The complete test suite passes: 370 tests, with three non-failing dependency warnings.

## VidDiff external holdout

The frozen, label-blind three-route protocol was evaluated on 30 disjoint cases with forward
and A/B-swapped comparisons. Measurement reachability and trajectory quality are reported
separately:

- measurement reachable: 13/30 (43.33%);
- semantically valid trajectory conditional on reachability: 13/13 (100%);
- end-to-end safe coverage: 13/30 (43.33%);
- quarantined because no frozen route is safely reachable: 17/30.

For the explicitly named five cases, `surgery_117`, `surgery_166`, and `surgery_78` are
reachable; `surgery_20` and `surgery_73` remain quarantined. No oracle/verifier field is
present in student SFT rows.

## Generator improvements validated in real rollouts

1. Downstream tool schemas compile exact student-visible invariants, preventing plausible but
   invalid paraphrases in executable arguments.
2. Evidence candidates are assessed by their own evidence ID, rather than being confused with
   the most recently created atom.
3. Natural assessment language may directly retain/reject a named candidate; it no longer has
   to repeat a checker-like “decision gap” phrase on every turn.
4. A narrow typed causal-edge repair handles one known reversal pattern for order-disagreement
   abstention. It changes no tool, value, interval, label, or evidence ID, records the repair,
   and must pass the same semantic checker afterward.
5. Accepted-prefix resume and stitch repair only the failed suffix. This preserved expensive
   measurements while producing complete 100/100 episodes for `surgery_15`, `surgery_166`,
   and the `surgery_27` repair trajectory.

## What is not yet proven

UTQR-6 currently reports `machine_qualified=true`, but:

- two independent blind judgments per episode have not been completed;
- there are not yet 30 independent trajectory cases for every dataset;
- every dataset does not yet have at least 10 clean and 10 repair episodes;
- SFT training gain on a case-disjoint evaluation has not yet been demonstrated.

Therefore `current_batch_gold=false` and `mass_production_ready=false`. The package is a
machine-qualified research canary, not yet a production-scale Gold corpus.

## Primary artifacts

- `rubric/unified_trajectory_quality_rubric_v6.json`
- `outputs/student_external_contract_v1/utqr_v6_current40/panel_manifest_v5.json`
- `outputs/student_external_contract_v1/utqr_v6_current40/semantic_audit_v5.json`
- `outputs/student_external_contract_v1/utqr_v6_current40/current40_sft_v5_diverse.jsonl`
- `outputs/student_external_contract_v1/utqr_v6_current40/utqr_v6_hybrid_v5.json`
- `outputs/student_external_contract_v1/utqr_v6_current40/viddiff_holdout30_trajectory_coverage_v1.json`
- `outputs/student_external_contract_v1/utqr_v6_current40/blind_review_v5/utqr_v6_judge_packet.jsonl`
