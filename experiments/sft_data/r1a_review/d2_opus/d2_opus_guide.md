# D2 trajectory read-through guide — "what would this trajectory teach the student that we do NOT want taught?"

## What you are reading
A rendered trajectory of a forensic media-verification agent run. Two roles:
- **GIA** (coordinator): plans tasks per round (INVESTIGATE with T1..Tn), later FINISH with an answer against a case file of requirements.
- **HI workspaces** (W1-1, W2-1 …): investigate one task with tools: SEARCH (text / image→ reverse image search / dates), READ_RECORD (open a page or evidence record), INSPECT (frames, OCR, ASR, compare_media on video), then REPORT (findings with statements, cited evidence ids, limitations; status COMPLETED / INCOMPLETE).

The run is **program-driven**: a rule (e.g. `C4_search`, `C5_read_page`, `G1_round1`) chooses the action; a teacher LLM fills the language slots: S0 rationale, S1 queries, S2 task text, S3 "next/sufficient" choice, S4 report, S5 answer. The student (Qwen 27B) will be fine-tuned on **everything shown under "Decision"** (actions + arguments + rationale + reports + plans + answers), given only what is shown under "State"/"Observation". Observations are truncated in this rendering (…[+N chars]); if you cannot judge because of truncation, say so instead of guessing.

## What to flag
Flag anything in a Decision that, if imitated, is a **bad habit**. Categories (use these exact tags):
- `wasted_step` — action adds nothing (re-reading, redundant search with same intent, reading an obviously irrelevant page, inspecting frames without a purpose).
- `missing_step` — an obvious necessary step was never taken (e.g. no reverse image search in a provenance task, no date lookup, never opened the one page that mattered, never compared A vs B) and the report/answer then says "could not be determined".
- `rationale_mismatch` — rationale does not match the action/arguments, or justifies with facts not visible in State/Observation, or cites an E-id not visible, or is boilerplate ("gather more evidence").
- `hidden_knowledge` — decision text uses information the student could not have seen (names, dates, locations, numbers appearing nowhere in prior observations). Be strict: quote the token.
- `query_quality` — search query leaks the expected answer, is too generic, repeats a failed query verbatim, or is in the wrong language.
- `overclaim` — report/answer asserts more than the cited evidence shows (page date treated as image date; "earliest copy"; attribution drift: source says X → stated as fact).
- `underclaim` — INCOMPLETE or "cannot be determined" when the visible evidence actually answers the task; or limitations that are false/boilerplate.
- `citation` — cites evidence that does not support the sentence, cites wrong id, or key claims uncited.
- `numeric` — timestamps/durations/percentages/dates copied wrong from tool output.
- `plan_quality` — GIA task text copies the overall task verbatim (no decomposition), smuggles unproven premise, assigns a 2nd-round task that re-investigates a resolved point, or FINISHes while a must-answer point is open.
- `tool_failure_habit` — many `page=failed`; the trajectory "teaches" giving up or reporting negatives that only reflect tool failure, without saying so.
- `format` — non-English text in a Decision (except quoting evidence), malformed arguments, wrong ids.
- `other`.

Severity: `major` = imitating this would make the student measurably worse (wrong verdicts, hallucination, leaks, skipping key work); `minor` = cosmetic/inefficient.

Also record `good_patterns`: 1–4 things this trajectory teaches well (so we know what to keep).

## Output
Write ONLY a JSON object to the output path given:
```
{"run": "...", "family": "A|B|C|E|L", "steps_read": N, "overall": "clean|minor_issues|problematic",
 "issues": [{"step": "e.g. W1-1 turn 3 (request_0004)", "category": "...", "severity": "major|minor",
             "what": "1–2 sentences, quote the offending text", "why_bad_to_teach": "1 sentence", "ideal": "what the step should have been, 1 sentence"}],
 "good_patterns": ["..."],
 "cannot_judge": ["things truncation prevented you from checking"],
 "summary": "2–4 sentences"}
```
Be concrete and quote. Prefer 3–10 well-founded issues over a long list of nitpicks. Do not invent issues to fill the list; "clean" is an acceptable overall.


## r1-specific (this batch is from the r1 pipeline)
New program rules you will see: `E2` reverse image search with image output, `E6`/`L6` compare_media of a page's picture vs the given image, `E6a`/`L6a` fetching a page's main image, `E7`/`L3b`/`C9v` "review comparison image / reference photos / full transcript before reporting", `L3` reference-photo comparison, `L4b` mandatory text search for the place, `C2v` video_structure scene frames, `C2w` frames per shot, in-loop sufficiency check (a COMPLETED report downgraded to INCOMPLETE with a stated gap), and round-2 tasks whose Scope starts with "first fetch the main image of <page> and compare it". Judge these on the same standard: does the rationale match the action, is the comparison conclusion honest about what the comparison image shows, does round 2 actually do the Scope's first item first, does the report say "candidate" for pages whose picture was not compared. Also flag (category `other`) any place where a Decision reads like a template filled in mechanically (identical rationale wording across steps).
