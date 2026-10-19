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



## Batch-specific (g2a_review; Codex teacher batches g1 / g2)
These runs come from the r1 pipeline (same program rules as r1: E2/E6/L6/C2v/C2w/C9v, in-loop sufficiency check, round-2 Scope items), but ALL teacher slots S0–S5 were written by Codex (gpt-6-astra) instead of Qwen for S0–S3. Read with the same standard, and look especially at:
- **Rationales (S0)**: do they cite content not yet read (hidden knowledge), are they template-like / identical across steps, do they assert "matches" BEFORE any compare_media has run (flag as `rationale_mismatch`, quote it).
- **Search queries (S1)**: taken from visible cues (frames, OCR, transcript) or copied from the claim? Do they carry spellings / names / dates that appear nowhere in prior observations (`hidden_knowledge`, quote the token)?
- **Page selection (S3)** and **round-2 task text (S2)**: does the Scope target the stated gap; is the Scope's first item actually done first?
- **A family**: was the candidate compared (compare_media) before being called the original; are removed-interval times copied correctly from the tool; is an upload / archive date stated as "earliest source"?
- **B family**: region OCR text reproduced faithfully; media-attribute (audio track) differences reported with evidence.
- **In-loop downgrade**: when a COMPLETED report is downgraded to INCOMPLETE in-loop, is the stated gap real, and does the next step address it?

### Batch README (Chinese, why each run was picked)
# g2a D2 通读样本（4 条，g2 = 轮次 r1g2，60 runs，S0–S5 Codex，冻结口径 specs/g1_freeze.json）

渲染：`python scripts/render_trajectories.py --runs <下列 run> --out experiments/sft_data/g2a_review`（修过的渲染器，同 g1a）。
每个补量格 1 条；通读重点：A 族 S1 的检索词是不是取自画面 / 转写线索（而不是声明或模型自己的知识）、找到的候选是否先比对再下结论、
日期是"这一份上传"的日期还是被说成最早出处；B 族区域 OCR 的文字是否照录。

| 文件 | 补量格 | 为什么挑它 | 结果（核验后） |
|---|---|---|---|
| [fmnv_a_0Uc3vwNlDdY_r1g2_1](fmnv_a_0Uc3vwNlDdY_r1g2_1.md) | A · SEARCH text→video | 冒烟第 1 条：文字查询直接找到原片，删掉区间报 B 43.1–59.0 s（GT 43.9–59.6） | W1 COMPLETED；HI accepted 8 / held 1 / rejected 1 |
| [fmnv_a_4aIK1tjpGuk_r1g2_1](fmnv_a_4aIK1tjpGuk_r1g2_1.md) | A · SEARCH text_image→video | 第 1 条文字查询没对上，第 2 条文字 + 一帧搜到原片；删掉 B 58.5–79.0 s（GT 58.6–78.6） | W1 COMPLETED；accepted 6 / held 2 / rejected 1 |
| [fmnv_a_fsdDrrt_maI_r1g2_1](fmnv_a_fsdDrrt_maI_r1g2_1.md) | A · compare_media 视频模式 + 原片日期 | 找到原片后整段比对、查上传日期与最早存档；它的 F1 在金标 g2a（F001，核验器标错） | W1 COMPLETED；accepted 6 / held 2 / rejected 1 |
| [videosham_4002_r1g2_1](videosham_4002_r1g2_1.md) | B · INSPECT ocr(image)（M3c 区域 OCR） | 文字篡改："John Conor / 39 Aug 2045" 对 "CONOR HEWETT / SEPTEMBER 4, 2021"，比对报横条差异框 → 两侧放大图各 OCR 一次 | W1 COMPLETED；accepted 8 |

