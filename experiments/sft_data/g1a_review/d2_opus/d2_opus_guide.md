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



## Batch-specific (g1a_review; Codex teacher batches g1 / g2)
These runs come from the r1 pipeline (same program rules as r1: E2/E6/L6/C2v/C2w/C9v, in-loop sufficiency check, round-2 Scope items), but ALL teacher slots S0–S5 were written by Codex (gpt-6-astra) instead of Qwen for S0–S3. Read with the same standard, and look especially at:
- **Rationales (S0)**: do they cite content not yet read (hidden knowledge), are they template-like / identical across steps, do they assert "matches" BEFORE any compare_media has run (flag as `rationale_mismatch`, quote it).
- **Search queries (S1)**: taken from visible cues (frames, OCR, transcript) or copied from the claim? Do they carry spellings / names / dates that appear nowhere in prior observations (`hidden_knowledge`, quote the token)?
- **Page selection (S3)** and **round-2 task text (S2)**: does the Scope target the stated gap; is the Scope's first item actually done first?
- **A family**: was the candidate compared (compare_media) before being called the original; are removed-interval times copied correctly from the tool; is an upload / archive date stated as "earliest source"?
- **B family**: region OCR text reproduced faithfully; media-attribute (audio track) differences reported with evidence.
- **In-loop downgrade**: when a COMPLETED report is downgraded to INCOMPLETE in-loop, is the stated gap real, and does the next step address it?

### Batch README (Chinese, why each run was picked)
# g1a D2 通读样本（6 条，g1 = 轮次 r1g1，147 runs，S0–S5 全走 Codex）

渲染：`python scripts/render_trajectories.py --runs <下列 run> --out experiments/sft_data/g1a_review`（10-18 修过的渲染器：INSPECT asr / ocr
印分段转写与 OCR 文字，GIA 印报告的 supporting_records，都按 rendered_payload）。

通读重点：S0–S3 由 Qwen 换成 Codex 后，**动作理由**（是否引了还没读到的内容、是否模板化、比对前是否就说"匹配"）、**检索词**（照抄声明
还是取自画面 / 转写线索；有没有带进模型自己知道、材料里没有的拼写）、**选页**、**第 2 轮任务文本**（Scope 是否针对缺口）。

| 文件 | 族 | 为什么挑它 | 结果（核验后） |
|---|---|---|---|
| [fmnv_a_3elCxlGqUHM_r1g1_1](fmnv_a_3elCxlGqUHM_r1g1_1.md) | A | 冒烟起就跟着的案例：找到原片，删掉区间报 30.2–41.6 s（GT 30.6–41.5，IoU ≈ 0.95） | W1 COMPLETED → accepted |
| [fmnv_a_1jLoMoqV8oE_r1g1_1](fmnv_a_1jLoMoqV8oE_r1g1_1.md) | A | 没找到原片、如实 INCOMPLETE——A 族这类报告 20 份全被核验判 negative 进了 rejected，看是报告的问题还是核验的问题 | W1 INCOMPLETE → rejected（negative_report_until_recall_calibrated） |
| [videosham_2004_r1g1_1](videosham_2004_r1g1_1.md) | B | VideoSham "静音换吉他配乐"：画面一致、音轨不同；循环内被改 INCOMPLETE（媒体属性 / 成对帧没引） | W1 COMPLETED → 循环内 INCOMPLETE |
| [fmnv_c_Aq7rDQx9jns_r1g1_1](fmnv_c_Aq7rDQx9jns_r1g1_1.md) | C | Codex 检索词取自帧里的字幕（"TRANSPORTER 2"），与 r1 Qwen 版照抄声明对照 | W1 COMPLETED → accepted |
| [fmnv_c_eLD1XA2UcwQ_r1g1_1](fmnv_c_eLD1XA2UcwQ_r1g1_1.md) | C | FMNV CD 类假标题与画面"大体相符"（数据集标注本身的问题）；检索从帧里的 "310fire via Storyful" 署名起查 | W1 COMPLETED → accepted |
| [groundlie360_v32327700_r1g1_1](groundlie360_v32327700_r1g1_1.md) | C | **第 2 轮**：W1 被循环内检查改 INCOMPLETE（整段视频说了什么没写、楼梯上的人没确认是 Harris）→ GIA 派第 2 轮 → W2 仍 INCOMPLETE；答案在金标 g1a（A001） | W1 / W2 INCOMPLETE；GIA FINISH |

