# V6 executable-affordance experiment

## Outcome

The current best candidate is `v10_bind_then_viddiff_route8_lr1e5` together with the native
state-constrained tool contract. It preserves the original 7/7 repaired state actions and 6/6
cross-dataset external-tool sentinels, and fixes the newly discovered VidDiff conflict-repair
double-bind failure. This remains an expanded canary result, not a production-readiness claim.

## Why the change was needed

The 120-exposure v4 adapter looked perfect on a nine-row canary but reached only 63/64 parseable,
62/64 schema-valid, and 53/64 teacher-semantic-equivalent calls on the full fixed validation split.
Executing generated state actions from their public pre-action snapshots found 37/44 high-quality
state actions. Four apparent teacher mismatches were actually valid alternative claim IDs; seven
were genuine no-op, malformed, incomplete, or incompatible actions.

One full v4 training epoch repaired four of those seven. V5 then externalized two facts the harness
could already derive from student-visible state:

- active EvidenceAtom IDs compatible with each unsupported claim field and its current value;
- meaningful field-decision transitions, such as `open -> committed` after visible support.

This uses no oracle label, verifier result, private locator, or raw feature. V5 repaired six of the
seven failures. The remaining action understood that six fields must be committed but emitted the
wrong JSON keys.

V6 compiles every meaningful decision transition into a complete native call with the exact visible
tool argument names. All 162 decision targets in the 600-row corpus are exactly present among these
public-state-derived executable actions. A 16-step, low-learning-rate continuation from V5 learned
the interface without the binding regression caused by a larger 64-step continuation.

## Fixed-canary comparison

| Candidate | Previously failing state actions executed successfully | External-tool contract sentinels | Important observation |
|---|---:|---:|---|
| 120-exposure v4 | 0/7 | not isolated here | Initial failures discovered by full validation |
| full-epoch v4 | 4/7 | 6/6 | More exposure helps but does not solve conflicting evidence |
| full-epoch v5 | 6/7 | 6/6 | Visible affordances fix evidence selection; one malformed decision call remains |
| v6 + 64 decision steps | 6/7 | 6/6 | Fixes decision syntax but regresses one bind action |
| v6 + 16 decision steps | 7/7 | 6/6 | Earlier best targeted candidate |
| v10 + native state constraints | 7/7 | 6/6 | Current best after conflict-route expansion |

For the final row set, all 13 calls were parseable, tool-name correct, schema-valid, and pre-state
eligible. Twelve exactly matched the teacher's semantic arguments. The thirteenth bound one extra
active EvidenceAtom that independently supported the same claim value; the harness accepted the
action, all bindable fields were covered, and the state changed. Its explanation is less specific
than the teacher's, so it remains a human-review item.

## What was actually executed

Internal state tools (`create_candidate_claim`, `bind_evidence`, `set_field_decision`, and `stop`)
were executed against harnesses reconstructed from each row's public pre-action snapshot. Passing
requires an accepted/executed result, an actual state change, and grounded creation or maximal
binding coverage.

External media and feature tools were checked only for parseability, exact visible JSON schema, and
pre-state eligibility in this experiment. They were not independently re-executed, so `6/6` must
not be described as live external-tool success.

## Disjoint 24-row validation panel

A second panel selected 24 validation rows after explicitly excluding the seven optimized failures
and six external sentinels; the overlap audit was 0/24. All calls parsed, validated against their
visible schema, and were pre-state eligible. Tool choice matched the teacher on 23/24 and teacher
argument semantics matched on 21/24. All 13 state actions passed public-prestate harness execution.
The three state argument mismatches were alternative claim IDs with grounded field values.

The one external tool-choice mismatch was informative rather than cosmetic. At a VidDiff repair
checkpoint, the teacher selected `compare_global_motion_symmetric`, while the student selected
`compare_uniform_timeline`. Recorded label-blind measurement replay showed:

1. the generated uniform branch executed but did not support any of the four fixed repair values;
2. the search remained open and exposed global/action-refined as the next tools;
3. global produced compatible evidence, but the resulting `a/b` disagreement required the remaining
   action-refined branch;
4. after two additional measurements, the route closed and exposed a valid `bind_evidence` state.

The action is therefore executable and recoverable, but less efficient than the teacher route by two
measurement calls. The harness now prevents a high-confidence measurement from closing the difference
goal when active visible evidence still cannot support the current repair claim. This compatibility
override uses current claim values and active visible EvidenceAtoms only.

## Cross-dataset external replay and target-hidden reasoning audit

Representative generated external calls were re-executed rather than checked only as JSON contracts:

- VideoSham 4189 was decoded from the local source/suspect videos. The generated inspection and temporal
  measurement executed, and dropout repair reproduced the same semantic measurement.
- Two VCSL cases were recomputed from the official ISC features. TN-STABILITY produced a non-empty box;
  DP closed after its clean measurement, reopened after evidence dropout, and reproduced the same box.
- The VidDiff surgery 155 branch remains explicitly a replay of recorded label-blind measurement outputs,
  not raw-video reprocessing.

All three replay audits passed. A separate target-hidden developer audit then scored the 24 generated
steps on action contract, evidence citation integrity, state assertions, explicit gap grounding, utility,
route minimality, and natural-language quality. It accepted 23/24. The rejected item was the recoverable
but two-measurement-longer VidDiff repair branch above. This audit did not read the teacher action, but it
is reproducible developer review rather than independent human review.

## True student free-running episodes

The local student was also run as the actual policy, not against independent teacher-prepared snapshots.
Each generated action was executed, and the resulting harness state became the next model input. Three
clean episodes all reached an allowed `stop` with zero rejected actions:

| Dataset | Turns | Student-generated tool path |
|---|---:|---|
| VideoSham | 6 | inspect, temporal measure, create, bind, commit, stop |
| VCSL | 6 | inspect features, TN-STABILITY, create, bind, commit, stop |
| VidDiffBench | 8 | inspect, action-refined, uniform, global, create, bind, commit, stop |

The VideoSham and VCSL prompts fit within 4096 tokens. Four late VidDiff prompts did not. A second VidDiff
run with a 6144-token context preserved the complete prompt, took the same eight-step route, and again
finished with zero rejections. The closed-loop default is therefore 6144 rather than relying on left
truncation.

## Repair-route restoration hint

The low-efficiency VidDiff repair exposed a simple state-visible signal: after evidence dropout, the open
claim still records `measurement_method=compare_global_motion_symmetric`, and that tool is currently
eligible. `repair_restoration_candidates` now exposes this same-method remeasurement as a restoration
candidate while still requiring the new observation to execute and pass compatibility checks. It does not
preview an uncalled measurement output.

Only 9/600 rows satisfy this condition. All 600 original targets remain aligned with their visible
affordances. Without any additional training, the existing adapter changed the failing surgery 155 action
from uniform to global with exact arguments. Checkpoint replay then reached a bindable state in one
measurement with zero extra recovery measurements, versus two extra measurements before the hint.

## Full free-running repair episodes

The same three datasets were then started from public post-dropout states rather than teacher-prepared
per-turn snapshots. Each student had to reopen only the invalidated slice, execute a fresh measurement,
bind its newly created evidence ID, recommit the repaired fields, and pass terminal validation. All three
episodes completed in the minimal five actions with zero rejected actions, prompt truncations, or generation
limit hits:

| Dataset | Student-generated repair path |
|---|---|
| VideoSham | reopen, temporal remeasure, rebind, recommit, stop |
| VCSL | reopen, DP re-localize, rebind, recommit, stop |
| VidDiffBench | reopen, global-motion remeasure, rebind, recommit, stop |

The VidDiff episode explicitly cited `repair_restoration_candidates`, generated a fresh `...-002` evidence
atom, used that exact ID in the next binding call, and preserved the independently supported identity fields.

## Strict unified closed-loop rubric and calibration

A target-hidden 100-point rubric now audits both clean and repair free-running episodes across ten dimensions:
terminal integrity, execution integrity, evidence-reference integrity, cross-turn entity consistency, evidence
consumption, mutation locality, dependency ordering, route efficiency, human-like gap reasoning, and context
integrity. Acceptance requires every critical dimension and at least 98/100.

All six executed episodes scored 100/100. To ensure this was not an easy rubric, ten plausible hard-negative
families were injected: open-obligation termination, hidden rejected action, hallucinated evidence citation,
claim-ID drift, unused measurement evidence, repair overreach, bind/commit inversion, redundant measurement,
generic reasoning, and prompt truncation. The rubric rejected 10/10; the generic-reasoning and truncation cases
scored 95 but still failed the 98-point threshold.

## Expanded capability panel and native state constraints

The next six free-running episodes added VideoSham spatial comparison, VCSL DP localization,
two decision-reopen repairs, and VidDiff clean/repair conflict routes. The first candidate reached
terminal success on all six, but the strict audit rejected the surgery-70 repair at 67/100: it first
bound the newest global atom to four fields, received `partially_accepted`, then repaired the mistake
with a second bind. This is the concrete failure that terminal-only evaluation had hidden.

Diagnosis found two representation problems. Four states exposed complete bind calls before
`bind_evidence` was actually eligible, and bind-eligible states simultaneously exposed a low-level
candidate table and a unique complete call. The corrected policy contract now:

- exposes binding choices only when `bind_evidence` is currently eligible;
- compiles all bindable fields into a minimum-cardinality complete native call;
- removes the redundant low-level candidate table from the policy view when that call exists;
- writes the visible complete action into the native JSON schema as exact argument constraints.

These constraints are derived only from the current public claim values, active EvidenceAtoms,
schema compatibility, field status, and eligible tools. They do not use teacher targets, oracle
labels, verifier outputs, or uncalled measurements. Across 600 rows, 103 binding states and 162
decision states now receive native state-derived argument constraints; all 162 decision targets
remain exact and no original target-affordance alignment fails.

The v10 continuation starts from the bind-specialized v7 adapter and uses only 30 VidDiff measurement
transitions for eight low-learning-rate steps, preserving binding composition while correcting route
timing. The final expanded panel completed 6/6 episodes with zero rejected or partially accepted
actions. The strict target-hidden audit accepted 6/6 at 100/100: clean paths used 6, 6, and 8 actions;
decision-reopen repairs used the minimal 2 and 2 actions; the conflict evidence-dropout repair used
7 actions with one bind. The original 13-row failure/sentinel regression was also 13/13 exact on tool,
schema, arguments, semantics, and pre-state eligibility; all 7 state actions passed real Harness
replay and all 6 external sentinels retained their contract checks.

## Main artifacts

- `outputs/student_external_contract_v1/shadow_execution_validation64.json`: original full-split
  state-action audit.
- `outputs/student_external_contract_v1/decision_sufficient_state_v5_action_affordances/manifest.json`:
  600-row visible-affordance alignment audit.
- `outputs/student_external_contract_v1/decision_sufficient_state_v6_executable_affordances/manifest.json`:
  162/162 executable-decision target alignment.
- `outputs/student_external_contract_v1/eval_v6_repair16_original_failures7_plus_external6.json`:
  final targeted generation report.
- `outputs/student_external_contract_v1/shadow_execution_v6_repair16_original_failures7_plus_external6.json`:
  final state execution report.
- `outputs/student_external_contract_v1/eval_v6_repair16_novel_validation24.json` and
  `shadow_execution_v6_repair16_novel_validation24.json`: disjoint broader panel.
- `outputs/student_external_contract_v1/viddiff_surgery155_generated_branch_checkpoint_replay.json`:
  generated external branch and complete recovery replay.
- `outputs/student_external_contract_v1/cross_dataset_generated_external_checkpoint_replay.json`:
  raw VideoSham/VCSL replay and recorded-measurement VidDiff audit boundary.
- `outputs/student_external_contract_v1/target_hidden_reasoning_quality_audit_novel24.json`:
  seven-dimension target-hidden reasoning audit.
- `outputs/student_external_contract_v1/student_closed_loop_canary_three_datasets.json` and
  `student_closed_loop_viddiff_6144.json`: true free-running episode reports.
- `outputs/student_external_contract_v1/student_closed_loop_canary_three_datasets_restoration_hint_6144.json`:
  final three-dataset regression with zero rejected actions, zero prompt truncations, and zero generation
  token-limit hits.
- `outputs/student_external_contract_v1/eval_v6_restoration_hint_viddiff_repair2.json` and
  `viddiff_surgery155_restoration_hint_checkpoint_replay.json`: corrected repair choice and direct replay.
- `outputs/student_external_contract_v1/student_closed_loop_repair_three_datasets_6144.json`:
  three complete student-generated repair episodes.
- `outputs/student_external_contract_v1/student_closed_loop_unified_quality_audit.json`:
  strict 100-point clean/repair closed-loop audit.
- `outputs/student_external_contract_v1/student_closed_loop_rubric_hard_negative_calibration.json`:
  10/10 hard-negative rejection calibration.
- `outputs/student_external_contract_v1/student_closed_loop_v10_native_constrained_expanded_six_episode_panel.json`:
  final six-case expanded free-running panel.
- `outputs/student_external_contract_v1/student_closed_loop_v10_native_constrained_expanded_quality_audit.json`:
  6/6 strict 100-point acceptance for the expanded panel.
- `outputs/student_external_contract_v1/eval_v10_native_constrained_original_failures7_plus_external6.json` and
  `shadow_execution_v10_native_constrained_original_failures7_plus_external6.json`: exact 13-row
  regression and 7/7 state-action execution replay.

## Remaining work before batch production

1. Run independent human review on the target-hidden packet; the current semantic review is automated
   developer audit and must not be reported as human evaluation.
2. Expand free-running coverage beyond the current 12 accepted episodes, especially the remaining
   VCSL localizer families, VideoSham audio/temporal families, and more VidDiff conflict topologies.
3. Measure throughput with a compact conditional affordance representation. V5/V6 add state tokens,
   even though they did not increase generated-answer length.

The final code/artifact regression passed 124 tests; the two emitted warnings are existing SWIG type
deprecation warnings rather than test failures.
