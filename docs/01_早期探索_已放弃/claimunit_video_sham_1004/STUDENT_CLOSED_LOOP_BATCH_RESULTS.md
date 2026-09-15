# Student closed-loop batch results

## What is now executable

`src/run_student_closed_loop_batch.py` runs the local student policy in a real,
multi-turn harness. Every next prompt is rendered from the state produced by the
previous executed tool call. The runner supports checkpoints, resume, clean and
repair episodes, dynamic native tool schemas, strict trajectory auditing, and
teacher-free external seed files.

Only trajectories that pass every critical closed-loop check and score at least
98/100 are eligible for SFT export. The export is then checked again for:

- exactly one native tool call per assistant turn;
- arguments valid under the visible native schema;
- target tool visible as eligible in the pre-action state;
- pre-action history length equal to `turn_index - 1`;
- non-empty natural-language reasoning;
- no teacher, oracle, or verifier flag;
- unique row IDs and accepted whole-trajectory provenance.

A complete audit export is retained. A second diversity-balanced view caps an
exactly repeated `(target tool, assistant reasoning)` pair at two rows, without
deleting or modifying the complete audit source.

## Fully held-out V6 test batch

Output: `outputs/student_external_contract_v1/student_batch_test_v10_native`

- 6 independent cases: 2 VideoSham, 2 VCSL, 2 VidDiffBench;
- clean and repair episode for every case: 12 episodes total;
- 12/12 accepted, minimum and mean score 100/100;
- 68 executed student actions, zero rejected actions;
- 68 complete SFT rows, 68/68 export validation passed;
- no rows removed by the diversity cap;
- no teacher action, oracle, or verifier used during generation.

Measured local throughput:

| Dataset | Episodes | Actions | Seconds/episode | Seconds/action |
|---|---:|---:|---:|---:|
| VCSL | 4 | 22 | 82.4 | 15.0 |
| VideoSham | 4 | 20 | 89.1 | 17.8 |
| VidDiffBench | 4 | 26 | 146.4 | 22.5 |

VidDiffBench is the current throughput bottleneck because its routes use more
comparison tools and longer evidence states.

## Teacher-free VideoSham external holdout

Seed output: `outputs/student_external_contract_v1/videosham_external_holdout_v1`

Batch output:
`outputs/student_external_contract_v1/student_batch_videosham_external_holdout_v1`

The 13 cases were absent from every V6 train, validation, and test split. Seeds
contain only a neutral task, logical media identity, comparison scope, empty
pre-action state, system policy, and native inspect schema. They do not contain
an assistant target, original edit instruction, URL, local path, oracle, or
verifier value.

- 13/13 environments reconstructed and matched the visible first-step tool;
- 13/13 free-run trajectories accepted at 100/100;
- 78/78 executed actions, zero rejected actions;
- 3 temporal routes and 10 spatial routes selected from live inspection state;
- 78 complete SFT rows, all validated;
- 66 rows in the optional diversity-balanced training view;
- mean 105.6 seconds/episode and 17.6 seconds/action.

## Official SPD as a complementary VCSL tool

The released VCSL ISC SPD model is now exposed as the executable
`localize_copy_segment_spd_official` tool. It is lazy-loaded only when the
`spd_official` route is selected and receives only the released ISC similarity
map. Oracle boxes are used after execution by the verifier, never by routing,
inference, the student seed, or the exported SFT row.

On the same frozen 30-pair VCSL pool, the prior label-blind portfolio reached
21/30 cases. Official SPD alone reaches 19/30 and strictly recovers two prior
misses, so the complementary union reaches 23/30 (76.7%). The accepted boxes
for the two new cases score 0.986 and 0.982 F1 respectively under the hidden
verifier.

Corrected teacher-free seed output:
`outputs/student_external_contract_v1/vcsl_spd_external_holdout_v2`

Accepted batch output:
`outputs/student_external_contract_v1/student_batch_vcsl_spd_external_holdout_v2`

- both cases were absent from every V6 train, validation, and test split;
- 2/2 environment preflights and 2/2 free-run trajectories passed;
- both trajectories independently produced the exact six-step chain
  `inspect -> SPD -> create -> bind -> commit -> stop`;
- 12/12 actions executed with zero rejection and both episodes scored 100/100;
- all 12 SFT rows passed native-schema, pre-action-history, eligibility, and
  no-teacher/no-oracle validation;
- the three-box case preserved all three measured intervals in its claim.

Combining the V6 test, external VideoSham, and corrected VCSL SPD batches yields
27 accepted free-run episodes, 158 executed student actions, and 158 complete
validated SFT rows. Independent case coverage is now VideoSham 15, VCSL 4, and
VidDiffBench 2. The optional balanced view contains 146 rows.

## Independent stochastic repeat panel

The batch runner now retains deterministic greedy decoding as its default, but
also supports explicit low-temperature sampling with a recorded seed. Three
clean held-out cases were selected, one per dataset, and independently rerun
with seeds 101, 202, and 303 at temperature 0.2 and top-p 0.95.

- 9/9 complete trajectories passed at 100/100;
- 60/60 executed actions and 60/60 per-run SFT rows passed validation;
- all three cases have strict `pass^1 = pass^2 = pass^3 = 1.0`;
- each case preserved the same indispensable tool sequence across all seeds;
- after adding a rollout identifier, all 60 row IDs are unique and the merged
  repeat panel passes the full SFT validator;
- 19 of 20 action positions have three textually distinct reasoning variants;
  the remaining position has two;
- the diversity cap removes no rows.

Artifacts:

- `outputs/student_external_contract_v1/student_passk_repeat_panel_v1/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/student_passk_videosham1007_v1/pass_k_report.json`
- `outputs/student_external_contract_v1/student_passk_vcsl_localalign_v1/pass_k_report.json`
- `outputs/student_external_contract_v1/student_passk_viddiff26_v1/pass_k_report.json`

Sampling is useful for stability auditing and a small number of reasoning
variants, but is not the default production mode. Across the nine rollouts it
was materially slower, especially for the eight-step VidDiff route (240--303
seconds per episode), while greedy generation remains the throughput-oriented
path.

A second three-case panel targets repair rather than clean acquisition. Seeds
404, 505, and 606 independently reran one repair case per dataset:

- 9/9 repair trajectories passed at 100/100;
- 39/39 recovery actions and SFT rows passed validation;
- all three repair cases have `pass^1 = pass^2 = pass^3 = 1.0`;
- the minimal repair sequence was identical across seeds for every case;
- 12 of 13 action positions have three distinct reasoning variants and one has
  two;
- rollout-aware row IDs are emitted directly by the sampled batch runner, so
  the 39-row cross-run panel has zero ID collisions and needs no post-hoc fix.

Artifact:
`outputs/student_external_contract_v1/student_repair_passk_repeat_panel_v1/repeat_panel_audit.json`

Five genuinely external VideoSham cases were then sampled twice with seeds 707
and 808. The panel covers two temporal-edit and three spatial-comparison routes:

- 10/10 trajectories and 60/60 actions/SFT rows passed at 100/100;
- every case has strict `pass^1 = pass^2 = 1.0`;
- all 30 action positions have two distinct reasoning variants;
- every case retained one stable six-step indispensable tool sequence;
- sampled row IDs were unique at generation time and the merged 60-row panel
  passed full and diversity-balanced validation without removals;
- mean throughput was 105.0 seconds/episode for seed 707 and 126.2 for seed
  808, demonstrating quality stability but measurable sampling latency variance.

Artifact:
`outputs/student_external_contract_v1/student_external_videosham_passk_repeat_panel_v1/repeat_panel_audit.json`

A second external VideoSham panel adds one temporal route with a non-zero scope
start and four spatial routes. Seeds 909 and 1001 again produced 10/10 accepted
trajectories, 60/60 validated SFT rows, stable six-step tool sequences, two
reasoning variants at every action position, and `pass^2 = 1.0` for all five
cases.

This panel exposed a throughput tail that mean latency hid: `videosham-4025`
took 544 seconds as the third case in a long-lived worker, but 227 seconds in a
fresh three-case shard; `4059` and `4061` completed in roughly 140--154 seconds
when moved to a fresh two-case shard. The long 354-token binding action was
schema-required, so reducing the generation cap to 256 would have truncated a
valid tool call. The production response is therefore small resumable worker
shards plus p50/p95/max and per-turn generation telemetry, not weaker reasoning
or removal of slow cases.

Artifacts:

- `outputs/student_external_contract_v1/student_external_videosham_panel2_head_repeat_v1/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/student_external_videosham_panel2_tail_repeat_v1/repeat_panel_audit.json`

The final repeat-coverage panel closes the remaining five evaluated cases. Two
independent seeds, 1102 and 1203, reran VideoSham `4073`, `4127`, and `4214`;
seeds 1304 and 1405 reran both official-SPD VCSL external holdouts. All 10
trajectories and all 60 SFT actions passed at 100/100, every case has
`pass^2 = 1.0`, and all 30 action positions have two distinct reasoning
variants while retaining a stable indispensable tool sequence. The VideoSham
seed-1203 shard also validates the new latency telemetry: episode p50 was 140.4
seconds, p95/max was 304.2 seconds, and the longest valid action generated 355
tokens. The two VCSL shards averaged 124.7--128.3 seconds per episode.

Artifacts:

- `outputs/student_external_contract_v1/student_external_videosham_final_repeat_v2/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/student_external_vcsl_spd_final_repeat_v2/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/production_readiness_current.json`

## Latest semantic remediation pass (2026-08-22)

The earlier aggregate 40/40 result was rescored from self-contained SFT turns
with stricter natural-language fact checks. That audit found 14 trajectories
whose execution structure was valid but whose reasoning contained a real or
ambiguous semantic defect. The offline auditor now hydrates compact batch
reports from native-tool SFT rows and can independently rescore only the
candidate-accepted members of multiple reports.

Thirteen of the 14 trajectories were regenerated and independently accepted at
100/100. The repaired set contains two VCSL routes (DP and official SPD), two
standard VidDiff clean cases, and nine external-holdout VidDiff clean cases.
The nine-case external set combines six deterministic successes with three
bounded low-temperature successes and contains 70 accepted SFT turns. All nine
independently rescore to 100/100.

The remediation exposed and fixed general generation/audit defects rather than
case-specific answers: current tool eligibility is checked before argument
schema validity; invalid native arguments receive exact schema feedback;
negated “rather than rerunning” language is not treated as a rerun claim;
qualified uniqueness such as “the only measurement that became stable” is not
confused with “only one measurement exists”; unobserved scalar fields cannot be
invented; active atoms cannot be described as dropped; and a just-executed
measurement must be bound or explicitly cited with a visible reason for
exclusion.

The isolated `viddiff-surgery_26` repair was then rerun with the existing v10
adapter and the corrected harness constraints. It completed the required
reopen, three complementary measurements, evidence binding, recommitment, and
stop in seven actions. Both the online report and independent audit scored it
100/100, exporting seven SFT turns. The merged 40-episode panel is therefore
40/40 accepted with 265 usable SFT turns: VideoSham 17/17, VCSL 6/6, and
VidDiffBench 17/17. This closes the current panel but does not by itself remove
the broader scale-validation boundary.

A case-disjoint incremental adapter was also trained on the 13 replacement
trajectories to test whether learning would reduce retries. It was rejected:
the target repair fell to 27/100 and failed to terminate after repeatedly
calling first-use measurements reruns. The root cause is supervision mismatch,
not insufficient training steps: 54/60 increment rows were VidDiff clean
acquisition turns, and the increment contained no independent VidDiff
evidence-dropout remeasurement trajectory. The v10 adapter remains the default.
Future adapter training must first build a repair-balanced set with multiple
independent evidence-dropout cases; clean semantic corrections cannot stand in
for repair-policy supervision.

Artifacts:

- `outputs/student_external_contract_v1/student_all40_latest_semantic_rescore_v1.json`
- `outputs/student_external_contract_v1/student_latest_semantic_repairs_13_composite_audit_v2.json`
- `outputs/student_external_contract_v1/student_viddiff_semantic_repair_strict9_composite_audit_v1.json`
- `outputs/student_external_contract_v1/semantic_repair_target26_baseline_v10_current/independent_semantic_audit.json`
- `outputs/student_external_contract_v1/semantic_repair_target26_candidate_v1/batch_report.json`
- `outputs/student_external_contract_v1/current_semantic_remediation_status_v2.json`

## Counterfactual repair-state expansion (2026-08-22)

The failed semantic-only adapter update revealed a supervision mismatch: clean
trajectory corrections do not teach evidence-dropout recovery. A local
counterfactual compiler now converts an independently accepted clean terminal
state into a real repair seed by removing exactly one active measurement
EvidenceAtom, invalidating only its claim bindings, reopening the affected
goals, and preserving media, claim values, and independently supported identity
fields. It generates no video and uses no teacher action, oracle, or verifier.

Four case-disjoint VidDiff seeds (`surgery_108`, `117`, `16`, and `84`) passed
4/4 real-environment preflights. The protected `surgery_26` case was excluded.
Initial rollouts exposed two general harness issues. First, rerun language must
be interpreted relative to the visible clean snapshot in a repair episode, not
only the current suffix history. Second, accumulating full rationales from
several measurements can truncate the bind turn. The policy view now keeps the
latest rationale in full and summarizes earlier rationale text while retaining
every decision-critical structured field and the full auditable state.

After those changes, 3/4 repair cases independently scored 100/100 and produced
19 validated SFT turns: `108` used seven actions in 488 seconds, `117` used
seven in 340 seconds, and `84` used five in 114 seconds. Their maximum prompts
were 5,603, 5,479, and 4,177 tokens, all below the 6,144 limit. `surgery_16`
remains quarantined: after global motion it selected the now-ineligible same
tool for all eight retries instead of choosing either remaining route.

A second, correctly stratified repair-only adapter update trained on the 19
accepted turns with `surgery_16` and `26` protected. It was also rejected: the
held-out `16` case remained at 27/100 and became slower. The remaining defect is
therefore action-name perseveration, not missing repair examples alone. A
hierarchical constrained-action decoder was subsequently tested: it fixed the
held-out `16` route but failed the protected `26` evidence-consumption
regression, so it was not promoted. The complete result appears in the later
decoder section.

Artifacts:

- `outputs/student_external_contract_v1/viddiff_counterfactual_repair_seeds_v1/manifest.json`
- `outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot108_v3/independent_semantic_audit.json`
- `outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot117_v2/independent_semantic_audit.json`
- `outputs/student_external_contract_v1/viddiff_counterfactual_repair_pair16_84_v1/independent_semantic_audit.json`
- `outputs/student_external_contract_v1/repair_policy_increment_v1/decision_sufficient_state_v6_executable_affordances/manifest.json`
- `outputs/student_external_contract_v1/counterfactual_repair_experiment_status_v1.json`

## External VidDiff symmetry-abstention expansion

Four cases recovered by the label-blind symmetry-residual abstention route were
run twice through the real local student loop with independent base seeds 1506
and 1607. All 8/8 trajectories produced the full nine-action sequence (inspect,
three visual comparisons, derived abstention, create, bind, commit, stop),
terminated legally, scored 100/100 under the latest offline rescore, and passed
the exported SFT validator. Each run contributes 36 accepted native-tool rows.

Failed checkpoints are retained under each batch's `quarantine/` directory and
are excluded from accepted exports. They exposed inaccurate meta-action
wording, reasoning--argument misalignment, historical EvidenceAtom fact drift,
`null`/false confusion, retry-language leakage, first-use tools described as
reruns, support/binding contradictions, and incomplete task/meta binding scope.
Across the two final runs, 40 of 112 candidate drafts were rejected before
execution; the remaining 72 actions form the accepted repeat panel.

The repeat panel has two reasoning variants at all 36 action positions, 72
unique rekeyed row IDs, stable executable tool sequences, and full validator
success. Every one of the four cases has strict `pass^1 = pass^2 = 1.0` across
the two independent stochastic rollouts.

The cross-dataset advisory audit now contains 40/40 accepted student episodes
and 264/264 validated actions. Held-out breadth is VideoSham 15, VCSL 4, and
VidDiffBench 15, leaving honest shortfalls of 15, 26, and 15 (56 total). VidDiff
measurement reachability is 13/30. The overall status remains
`quality_canaries_passed_scale_validation_pending`; the new four cases improve
breadth and repeat coverage but do not satisfy 30 held-out cases per dataset or
repeats for every evaluated case. Repeated rollouts now cover 25/34 evaluated
student cases (73.53%).

Artifacts:

- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1506_v1/batch_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1607_v1/batch_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_repeat_panel_v1/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/student_cross_dataset_policy_diversity_current.json`
- `outputs/student_external_contract_v1/production_readiness_current.json`

## VidDiff complementary-tool boundary

The attempted original-frame coarse-to-fine branch did not meet the admission
criterion. Grounding-DINO crops improved small-object visibility, but the local
8B VLM's free geometry estimates achieved only 50.0% on all 56 pairs for the
needle-tip predicate and only 23.5% directional precision. A stricter
point-overlay and horizontal-flip consistency method correctly rejected every
tested point annotation after visual inspection exposed fixture, shadow, and
collapsed-point errors. No new VidDiff tool or SFT trajectory was registered.

The full negative-result record is in
`VIDDIFF_COMPLEMENTARY_TOOL_EXPERIMENTS.md`.

## Failures found and corrected

1. The first SFT exporter serialized a mutable history list after executing the
   target action. This leaked the current action into its own pre-action prompt.
   The exporter now freezes the user payload before execution. A regression
   validator rejects any row whose history length differs from `turn_index - 1`.
2. The original human-reasoning check rejected two complete stop explanations
   solely because they had 44 words rather than 45. Non-terminal decisions still
   require 45–120 words; stop explanations allow 25–120 words, and every turn
   must contain 2–4 sentences, name the decision gap, and avoid hidden evaluation
   information. All 10 closed-loop hard-negative families remain rejected.
3. Exact repeated boundary reasoning is reported explicitly. It is a soft
   diversity signal, not a correctness gate. The complete audit set and optional
   diversity-balanced training view are both preserved.

## Honest production-readiness boundary

`outputs/student_external_contract_v1/production_readiness_current.json` remains
`quality_canaries_passed_scale_validation_pending`.

The remaining problem is measurement and held-out case reachability, not the
closed-loop policy on reachable cases:

- VideoSham now has 15 independent student free-run cases, so it needs 15 new
  disjoint held-out cases.
- VCSL has 4 student held-out cases and needs 26 new disjoint cases; its current
  local 58-feature cache covers only the original 30-pair measurement pool.
- VidDiffBench has 15 student held-out cases and needs 15 new disjoint cases.
- The total honest shortfall is therefore 56 new cases. Re-splitting old
  train/validation cases or counting repeat rollouts as new cases is forbidden.
- VCSL now reaches 23/30 cases after adding official SPD as a complementary
  route; 7/30 remain measurement-unreachable under the strict verifier.
- VidDiffBench reaches 13/30 cases after adding symmetry-residual abstention;
  17/30 remain measurement-unreachable under the current strict policy.
- Independent stochastic repeats cover 25/34 evaluated student cases (73.53%).
  Every measured case has at least two distinct recorded seeds and strict
  full-trajectory `pass^2 = 1.0`; nine evaluated cases still need a second run.

The next implementation priority is true held-out breadth rather than further
threshold tuning: each dataset still needs at least 30 independently evaluated
student cases, and the remaining nine evaluated cases need independent repeats.
The 17 VidDiff and seven VCSL measurement-unreachable cases remain quarantined
until a genuinely different local measurement model exists. New routes should
only be exported after full-family verifier evaluation and the same
target-hidden closed-loop audit.

## Prepared scale expansion

A new VidDiffBench 30-case candidate panel has been prepared entirely locally
from the 166 available surgery samples. It has zero case overlap with the prior
30-case pool and contains 60 locally generated mosaics. The dry-run report marks
all 30 cases `ready` and records `cli_requested=false`; therefore no frame or
mosaic from this new panel has been sent to a teacher service. The compact output
directory suffix fixes a concrete Windows path-length failure while the exact 30
indices remain in the JSON report.

Artifact:
`outputs/probes/viddiff_external_holdout30_dryrun_v2/viddiff_surgery_seed_20260821_n30_subset30_range0-34/probe_report.json`

Executing the three label-blind visual comparison routes on this panel requires
new explicit authorization to send these 60 mosaics and their neutral comparison
instructions to the Codex teacher. Oracle directions remain local and are used
only after all candidate measurements return. VideoSham has no additional local
pairs: the apparent 40-file count includes ten macOS `._` resource-fork files,
leaving the same 30 real pairs already exhausted by the old pool.

The outbound boundary is now enforced at runtime, immediately before each CLI
vision call. It rejects local paths, URLs, oracle/ground-truth fields,
verifier results, and metrics recursively, and all 30 prepared cases pass this
preflight. The first five cases (`surgery_20`, `117`, `73`, `166`, and `78`)
also have local global-timeline and motion-strip assets ready for the symmetric
route. All three probes support explicit `--resume`: only completed `passed` or
`failed` measurements are reused for a CLI run, while dry-run `ready` rows and
`execution_failed` rows are executed again. A replay of the 30-case dry run
reported 30 resumed and zero executed cases; the five-case route-2 replay
reported five resumed and zero executed cases.

Pilot route-2 artifact:
`outputs/probes/viddiff_external_holdout30_pilot5_v2_dryrun/viddiff_v2_surgery_seed_20260821_n5_idx0-1-2-3-5/probe_report.json`

The downstream compiler is also ready. `prepare_viddiff_student_holdout_seeds.py`
takes the post-execution selection audit plus the student-visible adaptive
candidate sets, excludes every case already present in any V6 split, and emits
only a neutral two-message initial state whose sole eligible action is
`inspect_comparison_task`. It embeds no teacher target, raw image, URL, local
path, oracle, or verifier value. A synthetic unseen case passes the real harness
preflight, while a candidate containing `video_path` is rejected.

Recompiling the old 10-case VidDiff pool produced eight safe accepted candidate
sets with zero private/oracle leak tokens. The holdout compiler then emitted
zero new seeds because all eight were correctly recognized as existing V6
cases. This is the intended anti-duplication behavior: old training cases cannot
increase held-out breadth. The runtime candidate loader now recursively scrubs
legacy artifacts as well, so historical `video_path` and `candidate_report`
fields cannot enter new student states.

Artifacts:

- `outputs/selection/viddiff_privacy_recompile_v1/viddiff_best_of_tree_seed_20260818_n10/candidate_measurements.student.jsonl`
- `outputs/student_external_contract_v1/viddiff_external_holdout_privacy_control_v1/manifest.json`

## VidDiff external holdout execution (2026-08-21; supersedes the prepared-only note)

The authorized 30-case panel was executed in six disjoint five-case shards with
seed `20260821`. Every case completed the uniform timeline route, the symmetric
global-motion route, and the follow-up action-refined route. The two symmetric
routes each ran forward and with A/B swapped. All 90 case-route executions
produced interval-valid measurements and none had an execution failure. No raw
video, URL, local path, oracle/GT field, verifier result, or metric was sent to
the teacher.

The first local selection pass exposed a real quality bug: it recorded
`order_consistent` but did not require it for admission. That permissive policy
would have accepted 23/30 cases. It is retained only as a diagnostic artifact
and is not an SFT source. The corrected policy requires interval validity,
oracle agreement after all teacher calls finish, and A/B-swap consistency for
every symmetric route. Under that policy, 9/30 cases are measurement-reachable
and 21/30 remain quarantined. The selected final evidence routes are seven
uniform-timeline cases and two action-refined cases. Global motion selects no
final atom under the strict rule, but remains a useful upstream localization
route for the action-refined strip; it must not be described as an independent
answer tool.

The nine admitted seeds are teacher-free two-message initial states. They have
zero overlap with V6, contain no teacher action, oracle/verifier value, raw
asset, URL, or local path, and restore the real harness with the expected sole
initial action `inspect_comparison_task` in 9/9 preflights.

The first local student batch found one substantive failure: `surgery_108`
terminated correctly but scored 90 because the first measurement EvidenceAtom
was neither cited later nor bound. The failed episode was not exported. The
student compression contract now requires every created measurement atom to be
consumed later, including explicit citation when an inconclusive atom motivates
another measurement. A one-case rerun reached 100, and a fresh full nine-case
regression then achieved 9/9 accepted episodes, 70/70 SFT actions, minimum score
100, and zero failure dimensions. Diversity balancing retained 63/70 rows. One
case (`surgery_84`) naturally stopped after a six-action route; the other eight
used eight actions, so the new consumption rule did not force uniform maximal
search.

At that earlier strict-9 checkpoint, advisory readiness remained
`quality_canaries_passed_scale_validation_pending`. The independently evaluated
student held-out counts are now VideoSham 15, VCSL 4, and VidDiffBench 11, so the
honest shortfalls are 15, 26, and 19 respectively (60 total). VidDiff's new
strict measurement reachability is 9/30, not the discarded permissive 23/30.

Primary artifacts:

- `outputs/selection/viddiff_external_holdout30_strict_v2/selection_audit.json`
- `outputs/student_external_contract_v1/viddiff_external_holdout30_strict_v2/manifest.json`
- `outputs/student_external_contract_v1/student_external_viddiff_strict9_consumption_contract_v3/batch_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_strict9_consumption_contract_v3/accepted_student_sft.jsonl`
- `outputs/student_external_contract_v1/production_readiness_current.json`

## Hierarchical constrained-action decoder result (2026-08-22)

The proposed decoder was implemented as an opt-in experiment. It externalizes a
compact current execution frontier at the tail of the student-visible state,
constrains logits only while generating the native tool name, and leaves the
model responsible for reasoning, branch choice among multiple eligible tools,
arguments, evidence arbitration, decisions, and stopping. Rejected drafts can
be regenerated with a pinned action only for surface alignment defects; any
evidence or state-fact error unlocks the action and requires replanning. A
second tail-visible frontier lists newly created measurement atoms that must be
bound or explicitly discussed as weaker.

This fixed the intended hard case: `surgery_16` rose from 27 to an independently
audited 100/100, terminated in six actions, and produced six valid native-tool
SFT turns. However, the predeclared protected `surgery_26` regression failed.
The current v10 adapter completes that case at 100/100 in seven actions, whereas
the final constrained candidate stopped at 27/100 after five actions. It chose
a schema-valid binding from the uniform-timeline atom but repeatedly omitted the
just-created global-motion atom and the visible reason for rejecting it. All
candidate output was quarantined and contributed zero SFT rows.

Therefore the decoder is not promoted and cross-dataset VideoSham/VCSL trials
were intentionally skipped. The result is useful but narrow: executable-action
constraints solve illegal route perseveration, not evidence arbitration or
consumption. The existing v10 model remains the current default. The complete
suite passes 230 tests.

Artifact:
`outputs/student_external_contract_v1/constrained_action_decoder_status_v1.json`

## Explicit evidence-adjudication experiment (2026-08-22)

The protected `surgery_26` regression showed that prompt-level warnings and a
legal-tool decoder cannot prevent silent omission when several active
EvidenceAtoms compete for the same reopened fields. An opt-in
`adjudicate_evidence` state action was therefore added. It appears only when a
real conflict exists, requires an exact selected/rejected partition of every
competitor, validates every rejection against student-visible facts, and
restricts the following binding to the selected evidence. Single-evidence
episodes keep their original action surface.

UTQR-5 now scores the same distinction at export time. H3 rejects incomplete
partitions or bindings that bypass the recorded selection; Q5 requires every
alternative to be unambiguously identified and explained; Q7 forbids silent
omission and newest/majority/confidence-only shortcuts. Exact IDs remain
mandatory in the native action, while natural language may use a unique visible
measurement method. This is closer to a human comparison narrative without
weakening structural traceability.

The harness replay passed on the real `surgery_26` pre-bind state, and all
structural/rubric tests passed. The current student did not pass the protected
free-generation regression. Across five bounded variants it reached the new
adjudication state but failed to name and diagnose both rejected alternatives
within four drafts; every run remained nonterminal at 27/100 and exported zero
SFT rows. Two label-blind, case-disjoint three-competitor warm-up rows were
compiled from training cases `surgery_65` and `surgery_158`, executed by the
harness, and used for a four-step adapter update. That adapter still failed the
unseen `surgery_26` adjudication and is not promoted.

The evidence supports a precise conclusion: the new state action and rubric are
executable, but the current training corpus is too sparse to teach high-quality
multi-evidence adjudication language. The existing v10 adapter remains the
default. `surgery_16` and VideoSham/VCSL candidate trials were skipped after the
predeclared protected failure. The complete local suite passes 239 tests.

Next, collect a case-disjoint training panel with multiple 3- and 5-competitor
states, different selected methods, abstention and non-abstention outcomes, and
all rejection-defect families. Re-train only after that panel passes the exact
partition, reasoning coverage, and bind-consistency audits; then re-run
`surgery_26` followed by `surgery_16` before any cross-dataset expansion.

Artifact:
`outputs/student_external_contract_v1/evidence_adjudication_experiment_status_v1.json`

## Progressive per-candidate assessment: one protected pass, one remaining route failure (2026-08-22)

The overloaded single-turn adjudication was replaced by a real state action,
`assess_evidence_candidate`. The harness now exposes exactly one
`current_candidate_id` per turn, records a retain/reject ledger with visible defect
codes, unlocks `adjudicate_evidence` only after every competitor is assessed, and
allows binding only from the selected partition. Null order consistency remains
unobserved rather than false. A 52-row, 13-trajectory, protected-case-disjoint
curriculum contains 39 atomic assessments and 13 final adjudications; all local
execution, privacy, partition, bind-consistency, duplicate-reasoning, and SFT checks
passed.

The v2 queued adapter completed protected `surgery_26` at 100/100 in the exact
11-action route and exported 11 validated SFT turns. It nevertheless required 15
generation attempts and 1,706.394 seconds, so semantic quality passed while
throughput failed. The second protected case, `surgery_16`, remained quarantined at
27/100. A tail-visible identity guard reduced its failed run from 402.711 to 183.624
seconds and separated the active replacement `-002` atom from the dropped `-001`
tombstone, but the model still called `compare_action_refined_symmetric` while its
reasoning incorrectly planned direct binding.

A new 56-row balanced curriculum therefore mixes the 52 progressive rows with four
already accepted, protected-case-disjoint replacement-route turns from
`surgery_108` and `surgery_117`. It covers action-refined, uniform, and global
remaining routes and explicitly excludes `surgery_16` and `surgery_26`. The v3
replacement-lineage adapter trained for 56 steps with zero truncation and remains
unpromoted pending the required protected order: `26`, then `16`, then and only then
authorized new holdouts. The existing v10 adapter remains the default.

Artifact:
`outputs/student_external_contract_v1/progressive_evidence_assessment_experiment_status_v3.json`

## Diverse adjudication warm-up: protected negative result (2026-08-22)

The two-row warm-up was expanded to 14 label-blind rows from 13 case-disjoint
VidDiffBench training cases. The panel covers all three selected measurement
methods and four visible rejection-defect families. Every compiled action executed,
every selected atom bound successfully, every rejected-atom counterfactual bind was
blocked for the intended reason, no prompt was truncated, and the complete local
suite passed 242 tests. These rows remain auxiliary single-turn warm-ups rather than
complete production trajectories.

A 14-step adapter update still failed the predeclared `surgery_26` regression. It
generated the exact structured selected/rejected partition, but all four drafts
described the two rejected methods only as generic alternatives and falsely treated
the selected atom's unobserved order consistency as a negative observation. The run
remained nonterminal at 27/100 and exported zero SFT rows. The candidate was not
promoted, and `surgery_16`, both newly authorized external holdout panels, and all
cross-dataset expansion were not run.

This isolates the remaining problem more sharply: a single overloaded adjudication
turn can learn the native action while still collapsing the human comparison prose.
The next implementation should externalize one stateful assessment per competing
EvidenceAtom and require the final adjudication to consume the complete visible
assessment ledger. That makes each intermediate tool meaningful and converts a
three-way prose burden into auditable one-candidate decisions without relaxing the
final rubric.

Artifacts:
`outputs/student_external_contract_v1/evidence_adjudication_warmup_v3/`
and
`outputs/student_external_contract_v1/evidence_adjudication_experiment_status_v2.json`
