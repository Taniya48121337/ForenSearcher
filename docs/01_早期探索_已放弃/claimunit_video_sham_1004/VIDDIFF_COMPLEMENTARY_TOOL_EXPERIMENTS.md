# VidDiff complementary-tool experiments

## Frozen failure pool

The current 30-case VidDiff pool has 21 verifier-reachable cases and 9 cases
for which none of the existing recorded comparison tools returns the verifier
direction. The 9-case pool contains five `c` cases and four directional cases.
Oracle directions are never used during a measurement; they are read only after
each candidate has produced a fixed result.

## Rejected candidates

### Green thread/tube motion proxy

The original color-motion prototype happened to recover the two unreachable
`tube moves more` cases. Re-evaluation on all 54 pairs with that statement gave
42.6% accuracy, below the 50.0% majority baseline. Visual inspection showed that
it followed green suture pixels rather than the yellow task tube. It is not
eligible for registration.

Artifact:
`outputs/student_external_contract_v1/viddiff_tube_motion_recheck_v1.json`

### Yellow tube centerline motion

A corrected object-centric measurement segmented the central yellow tube and
measured translation, bending, centroid range, and silhouette change. The
development result was 44.4% and the held-out result was 37.0%; grasper
occlusion still dominated the apparent tube silhouette, and both target cases
were reversed. It is not eligible for registration.

Artifact:
`outputs/student_external_contract_v1/viddiff_yellow_tube_motion_v1.json`

### Local Qwen3-VL multi-view scorer

The first local-only scorer measured each video independently from uniform and
targeted mosaics, then required view stability, observability, and a fixed
support margin. It predicted `c` for all 9 cases: 5/9, exactly the majority
baseline, with zero directional recall.

Splitting the ultra-wide strips back into independent tiles fixed a real input
resolution problem. Treating a view with observability below 45 as missing
evidence rather than counterevidence then recovered `surgery_123` without
regressing any of the five `c` cases. The result rose to 6/9.

Artifact:
`outputs/student_external_contract_v1/viddiff_local_qwen3vl_observable_tiles_v3.json`

### Full-predicate unary ranking and pairwise consistency

The tile-based unary scorer was evaluated on all 56 pairs for “The needle is
grasped closer to the tip.” Each of 20 unique videos was scored once and reused
across pairs. Accuracy was 64.3% versus a 62.5% majority baseline, but
directional precision was only 3/5 and non-`c` recall was 14.3%, which is not
safe enough for a training-data tool.

Requiring unary direction, forward pairwise direction, and mapped swapped-order
direction to agree removed both false directions, but also removed every true
direction. The final result became all `c`, with 62.5% accuracy and zero
directional recall. It is not eligible for registration.

Artifacts:

- `outputs/student_external_contract_v1/viddiff_qwen_unary_tip_grasp_v1.json`
- `outputs/student_external_contract_v1/viddiff_qwen_tip_grasp_pair_consistency_v1.json`

### Grounding-DINO crops plus free-form VLM geometry

The coarse-to-fine hypothesis was implemented with original 640x480 frames,
Grounding-DINO needle proposals, spatially fixed false-positive suppression,
3.5x enlarged crops, and a Qwen3-VL per-crop estimate of the jaw contact's
position along the needle arc. The estimator had to see the needle, sharp tip,
active jaws, and contact point, and two crops were required for a video-level
measurement.

On `surgery_123`, the continuous measurement had the correct ordering (A 30.0%
from the tip versus B 37.5%) but did not clear the fixed 12-point margin. On the
full 56-pair predicate family, the method reached only 50.0% accuracy versus the
62.5% majority baseline. It raised non-`c` recall to 19.1%, but directional
precision was only 23.5% (4/17). A diagnostic threshold sweep did not rescue
the estimator: every margin from 5 through 15 had directional precision below
25%, and larger margins removed the true directions before the false ones.

Visual inspection found that original-frame crops fixed the earlier tiny-object
resolution problem and static suppression removed repeated fixture detections.
The remaining failure is semantic calibration: the 8B VLM marked nearly every
crop measurable and emitted confident arc percentages even when the sharp tip,
tail, or exact contact was ambiguous. Free numerical VLM geometry is therefore
not eligible for registration.

Artifacts:

- `outputs/student_external_contract_v1/viddiff_grounded_tip_crop_surgery123_v1.json`
- `outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_surgery123_v1.json`
- `outputs/student_external_contract_v1/viddiff_grounded_tip_geometry_full_v1.json`

## Surviving method hypothesis

The next candidate should keep original-frame retrieval and static-background
suppression, but stop asking a general VLM for an unconstrained percentage.
The most defensible remaining design is discrete, reference-anchored evidence:

1. retrieve only frames where a jaw--needle contact is visible;
2. create explicit visual anchors for tip-third, middle-third, and tail-third;
3. classify each crop into an ordinal bin or `unobservable`, with crop-order and
   frame-order consistency checks;
4. compare videos only when multiple independent frames produce a stable ordinal
   separation;
5. validate once on the full predicate family, reporting directional precision
   and non-`c` recall rather than tuning a margin on `surgery_123`.

If ordinal anchoring also fails, this predicate should remain unreachable for
automatic trajectory generation rather than contributing noisy SFT supervision.

### Point-grounded overlay with flip equivariance

As a stricter alternative to free numerical estimates, the localizer was asked
to externalize three normalized points on each crop: sharp tip `T`, suture end
`S`, and active jaw contact `C`. The tool horizontally flipped every crop,
localized it again, mapped the points back, rejected field/rationale
contradictions, and computed the contact ratio only when all three points were
stable. The original and annotated crops were retained as human-auditable
evidence.

The multi-image protocol first failed because the 8B model confused crop IDs
with point labels. A corrected single-image protocol produced parseable points,
but zero of 16 crops passed. This was not excessive filtering: visual inspection
of the overlays showed points on fixture pegs, shadows, or grasper bodies;
several annotations collapsed `T`, `S`, and `C` to the same location; and the
model often changed observability after a horizontal flip. The strict
equivariance rejection therefore prevented false precision as intended.

Artifact:
`outputs/student_external_contract_v1/viddiff_tip_point_overlay_surgery123_v1.json`

The result closes the current VidDiff needle-tip branch. Neither margin tuning
nor weaker consistency rules are justified by the evidence. This predicate
remains unavailable to automatic SFT trajectory generation with the current
local models; no candidate from this branch was added to the production tool
registry.

No VidDiff tool from this round has been added to the production registry. The
VCSL SPD integration and all existing VideoSham/VCSL/VidDiff routes remain
unchanged.

## External holdout multi-route result (2026-08-21)

A disjoint 30-case VidDiff panel was evaluated with the existing three-route
label-blind tree. Uniform timeline was correct on 7/30 cases. The symmetric
global-motion route was direction-correct on 14/30 but order-consistent on only
13/30; the symmetric action-refined route was direction-correct on 12/30 and
order-consistent on 13/30. All route outputs had valid intervals and no route
execution failed.

An audit found that direction correctness alone was too permissive for the
symmetric routes. Requiring mapped A/B-swap consistency reduced final reachable
coverage from the diagnostic 23/30 to the defensible 9/30. The strict selected
set contains seven uniform-timeline atoms and two action-refined atoms. Global
motion has no strict final selections, but it is still the label-blind source of
the local windows used by the action-refined route; therefore it is retained as
an upstream search/localization action rather than advertised as a final
comparison answerer.

This result changes the next research priority. The main VidDiff bottleneck is
not trajectory formatting: all nine reachable cases can produce full
target-hidden student trajectories at 100/100 after enforcing explicit evidence
consumption. The bottleneck is visual measurement robustness under A/B swap.
Future tools should be judged primarily by strict incremental reachability over
the 21 currently unreachable cases, not by raw direction accuracy before the
swap-consistency check.

Artifacts:

- `outputs/selection/viddiff_external_holdout30_strict_v2/selection_audit.json`
- `outputs/student_external_contract_v1/student_external_viddiff_strict9_consumption_contract_v3/batch_report.json`

## Symmetry-residual abstention and strict-13 result (2026-08-21)

A fourth action, `compare_symmetry_residual_abstention`, was added as a
label-blind derived-evidence tool rather than another visual direction scorer.
It is eligible only after both symmetric source routes have executed. For the
predeclared geometry, count/frequency, and object-motion predicate families, it
returns only role-invariant `c` when both source routes independently collapse
to abstention under mapped A/B-swap disagreement. It cannot emit `a` or `b`,
does not inspect new pixels, and is unavailable for force/tension and
visibility/support statements.

The fixed rule recovered 2/2 development cases and, without modification,
4/4 previously unreachable external-holdout cases with zero false additions:
`surgery_127`, `surgery_15`, `surgery_166`, and `surgery_78`. Strict VidDiff
measurement reachability therefore rises from 9/30 to 13/30 (43.33%); 17/30
remain quarantined. This is a precision-first selective-evidence result, not a
claim of conformal risk control. A future calibration split must traverse the
same acquisition policy before any statistical guarantee is reported.

Repeated real-student rollouts exposed semantic failures that terminal success
alone missed: meta evidence described as new sensing, source and meta facts
being conflated, `null` order consistency being narrated as false, first-use
tools being called reruns, identity fields being attributed to a measurement
atom, and evidence support being confused with binding. These drafts are now
rejected before tool execution. Retry feedback contains only natural-language
corrections, and accepted reasoning is checked against every named visible
EvidenceAtom, not only the latest one.

Two independent sampled four-case batches (base seeds 1506 and 1607) each
achieved 4/4 at 100/100 and 36/36 accepted native-tool SFT actions. Across both
runs, 112 generation attempts produced 72 accepted actions; 40 flawed drafts
were rejected before execution. The repeat panel contains 72 unique rekeyed
rows, two reasoning variants at every one of the 36 action positions, no row-ID
collisions, stable executable tool sequences, and strict `pass^1 = pass^2 =
1.0` for all four cases. The two batch means were 624.4 and 518.5 seconds per
episode. The current cross-dataset diversity advisory covers 28 accepted
episodes, six measurement routes, five predicate families, dominant-route share
0.3571, and no warning.

Artifacts:

- `outputs/selection/viddiff_external_holdout30_symmetry_abstention_v1/selection_audit.json`
- `outputs/selection/viddiff_external_holdout30_symmetry_abstention_v1/symmetry_abstention_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1506_v1/batch_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1607_v1/batch_report.json`
- `outputs/student_external_contract_v1/student_external_viddiff_symmetry4_repeat_panel_v1/repeat_panel_audit.json`
- `outputs/student_external_contract_v1/student_cross_dataset_policy_diversity_current.json`
- `literature-review/ZOTERO_UPLOAD_PLAN.json`

## Latest-rubric trajectory remediation (2026-08-22)

A self-contained rescore of the current student corpus showed that terminal
success had hidden several language-policy errors: first-use measurements were
called reruns, selected methods were not named, structured A/B facts were
misstated, supported fields were called unsupported, and unobserved
order-consistency values were inferred. These are trajectory-quality failures,
not measurement-verifier failures.

The generator now rejects these drafts before execution and returns a concrete
student-visible revision requirement. It also prioritizes current tool
eligibility over argument-schema validation, so an attempted premature bind is
correctly redirected to the remaining measurement instead of being mislabeled
as malformed JSON. Exact schema feedback is used only when the selected native
tool is actually eligible.

For the strict nine-case external panel, the deterministic pass accepted six
cases and isolated three. A single bounded low-temperature pass repaired the
three difficult cases. The composite set independently audits at 9/9,
100/100, with 70 native-tool SFT turns. This does not increase strict
measurement reachability beyond the already admitted nine cases; it improves
the reliability of trajectories on those reachable cases.

The repair view of `surgery_26` was rerun with the unchanged v10 adapter after
adding the generic evidence-consumption rule. It completed reopen, three
complementary measurements, rebinding, recommitment, and stop in seven actions.
The online report and independent audit both scored it 100/100, and all seven
native-tool turns passed SFT validation. The improvement is attributed to the
harness feedback change, because it occurred before any new adapter training.

A subsequent case-disjoint adapter update was a negative result. Although its
60-row increment contained accepted semantic corrections, 54 rows were clean
VidDiff acquisition turns and none provided an independent evidence-dropout
remeasurement example. On the same protected repair, the new adapter scored
27/100, failed to terminate, and repeatedly described first-use tools as
reruns. It is therefore not promoted. The experiment shows that semantic
quality repair and harness-state repair are distinct supervision strata; the
next training set must balance repair operators and include multiple independent
VidDiff evidence-dropout trajectories before another adapter update.

Artifacts:

- `outputs/student_external_contract_v1/student_viddiff_semantic_repair_strict9_composite_audit_v1.json`
- `outputs/student_external_contract_v1/semantic_repair_target26_baseline_v10_current/independent_semantic_audit.json`
- `outputs/student_external_contract_v1/semantic_repair_target26_candidate_v1/batch_report.json`
- `outputs/student_external_contract_v1/current_semantic_remediation_status_v2.json`

## Counterfactual evidence-dropout repair pilot (2026-08-22)

Accepted clean VidDiff terminal states can now be mutated locally into new
repair tasks without generating video or revealing a label. The compiler drops
one supporting measurement atom, invalidates only its four comparative-field
bindings, and preserves identity evidence and claim values. Four seeds passed
environment preflight; three generated full, independently audited 100/100
repair trajectories, contributing 19 native-tool SFT turns.

The pilot also produced an actionable hard case. `surgery_16` repeatedly emits
an already executed and therefore ineligible global-motion action even when the
visible action tree contains only uniform and action-refined branches. A small
repair-only adapter did not correct this behavior, so it was not promoted. The
next route-policy improvement should constrain only the tool-name branch to the
visible eligible set while leaving the model responsible for branch choice,
reasoning, arguments, evidence arbitration, and stopping.

Artifact:
`outputs/student_external_contract_v1/counterfactual_repair_experiment_status_v1.json`

## Constrained action frontier: local win, protected regression failure

An opt-in hierarchical decoder combined a tail-visible eligible-action
frontier, tool-name-only constrained decoding, surface-only action locking, and
a student-visible evidence-consumption frontier. It repaired the original
`surgery_16` ineligible-tool loop and produced a six-action, independently
audited 100/100 trajectory.

The candidate nevertheless failed the protected `surgery_26` repair that v10
already solves at 100/100. After executing uniform, action-refined, and global
measurements, it selected a valid uniform-timeline binding but did not cite the
latest global-motion EvidenceAtom or explain its visible defect. Four bounded
revisions repeated the omission, so the episode was quarantined at 27/100 with
zero exported SFT turns. Because this is an evidence-consumption regression,
not a surface formatting error, the candidate is rejected rather than relaxed.
No VideoSham/VCSL expansion was run after the protected failure.

Artifact:
`outputs/student_external_contract_v1/constrained_action_decoder_status_v1.json`

## Explicit evidence adjudication: executable mechanism, insufficient supervision

An opt-in `adjudicate_evidence` action now externalizes the comparison that was
previously left inside free-form reasoning. When active measurements disagree
over the same open claim fields, the harness exposes one exact action that
selects compatible support and rejects every alternative with validated visible
defect codes. Binding cannot proceed first or use rejected evidence. This
addresses the protected failure in which the latest global-motion atom was
silently ignored even though the final binding itself was schema-valid.

The three VidDiff routes now also have distinct model-visible purposes: uniform
timeline supplies coarse whole-video coverage, global motion tests role-swap
stability under motion emphasis, and action-refined strips inspect brief local
predicate-relevant behavior. This repaired a real ambiguity in the previous
tool schemas, which had described all three routes with the same generic text.

The mechanism passed synthetic tests and a replay of the real `surgery_26`
conflict state. Free generation remained the limiting factor. The current model
could emit the exact structured partition but repeatedly summarized the two
rejections as "the conflicting alternatives" instead of identifying and
diagnosing each one. A two-case, case-disjoint warm-up adapter reduced factual
errors but did not generalize; the final protected run remained 27/100,
nonterminal, with zero exported rows. No candidate was promoted and no
cross-dataset run followed.

The next experiment should increase supervision breadth rather than add more
single-case retry rules. A useful panel must cross selected route, competitor
count, direction versus abstention, order consistency, and defect family. The
promotion order remains `surgery_26` then `surgery_16`, followed only then by
VideoSham/VCSL repair regressions.

Artifact:
`outputs/student_external_contract_v1/evidence_adjudication_experiment_status_v1.json`

## Queued evidence assessment and replacement-lineage follow-up

The new harness externalizes evidence arbitration as three auditable stages:
measurement, one queued `assess_evidence_candidate` action per active competitor,
and a final ledger-consistent `adjudicate_evidence` action. This avoids asking one
language turn to compare every candidate while still requiring exact IDs, direction,
order consistency (including unobserved null), retain/reject disposition, and a
visible defect for each rejection.

Protected `surgery_26` now passes the full 11-action repair at 100/100 with 11 valid
SFT rows, but local-model latency remains non-production at 1,706.394 seconds.
Protected `surgery_16` still fails before arbitration because a correct
action-refined tool call is paired with reasoning that plans direct binding. A
student-visible identity guard now separates active replacement IDs from dropped
tombstones and cut the failed diagnostic run to 183.624 seconds, but did not by
itself solve tool/reasoning alignment.

The next candidate mixes 52 queued assessment/adjudication rows with four accepted
replacement-route rows from disjoint `surgery_108` and `surgery_117` states. It is
saved but not promoted. No newly authorized holdout was used or transferred; the
promotion order remains protected `26`, protected `16`, then authorized holdouts.

## Diverse warm-up still exposes a single-turn language bottleneck

Fourteen audited adjudication warm-ups from 13 disjoint training cases learned the
exact native partition but did not solve the protected free-generation case. On
`surgery_26`, all four adjudication drafts collapsed two individually structured
rejections into the phrase "two conflicting alternatives" and incorrectly inferred
that the selected `order_consistent=null` atom had no order inconsistency. The run
scored 27/100, remained nonterminal, and exported no SFT.

The next method change is therefore structural rather than another prompt warning:
record one executable, student-visible assessment for each competing atom, then let
the final adjudication consume that complete ledger. The assessment action has a
real state effect, reduces each natural-language turn to one evidence item, and
preserves the exact final partition and bind restriction. No newly authorized
external holdout case was used after the protected failure.
