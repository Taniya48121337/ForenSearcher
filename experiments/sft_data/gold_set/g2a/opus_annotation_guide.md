# Gold-set annotation guide (ForenSearcher r1a verifier gold set)

You are an independent human-quality annotator producing a *gold label* for ONE item. The labels are later used to measure the precision of an automatic verifier, so be strict, literal, and evidence-bound. You are NOT shown the verifier's opinion and must not guess it.

Background: a forensic-verification agent system. An "investigator" workspace receives a task, gathers evidence cards (web pages, OCR text, image frames, tool outputs), and writes Findings (statement + limitations + cited evidence). A "coordinator" plans tasks per round and finally writes an answer against a case-file of requirements.

## Core rules (apply to every kind)
1. Judge ONLY from what is in the item: the task text, the statement/answer/plan, and the cited evidence (text + images). Do NOT use outside world knowledge to fill gaps. If a claim could be true in reality but the cited evidence does not show it, it is unsupported.
2. Read images when provided (use the Read tool on the image path). If an image is referenced but unreadable, say so in the note; if that is the only evidence for the claim, use `cannot_tell`.
3. Be literal about numbers, units, dates, timestamps, names, and quantifiers ("all", "every", "the only", "first", "original").
4. "Attribution drift": the evidence reports that a SOURCE SAYS X; the statement asserts X as fact without attribution. That is an error (attribution_drift), even if X is plausible.
5. "Scope overreach": the statement generalises beyond what the evidence shows (e.g. evidence shows one frame matches, statement says the whole clip matches; evidence shows a page exists, statement says it is the earliest/original).
6. A statement that is explicitly hedged and whose limitations honestly cover its gaps is NOT an error for those gaps.
7. Reporting that something was NOT found / could not be determined is fine if the evidence shows the attempts; it is `unsupported_assertion` only if the statement claims a negative stronger than the attempts justify (e.g. "no earlier copy exists online" after one search).

## Kind: finding
Question: Is this Finding's statement supported by the evidence it cites?
`main` ∈
- `full` — every factual claim in the statement is directly supported by the cited evidence (text or image), with no overreach.
- `partial` — the core is supported but some part goes beyond the evidence, or a secondary detail is wrong.
- `none` — the main claim is not supported or contradicted by the evidence.
- `cannot_tell` — the evidence needed cannot be inspected (e.g. image missing/unreadable, cited evidence absent) so you cannot judge.
`errors` (multi-select, from this list only): `unsupported_assertion`, `scope_overreach`, `wrong_citation`, `attribution_drift`, `sloppy_numeric`, `misread_fine_text`, `other`.
`span`: quote verbatim the exact fragment(s) of the statement that go beyond the evidence, separated by ｜ (empty string if none).
Severity hint for your note: say whether each error is major (would mislead the final verdict) or minor (cosmetic / does not change the conclusion).

## Kind: answer
Question: Is this final answer acceptable overall?
`main` ∈ `acceptable`, `minor` (small issues, still acceptable), `unacceptable`, `cannot_tell`.
`req`: for every requirement id in `requirements`, `"yes"` if the answer covers it with support, `"no"` otherwise. Requirements with `must_cite: false` are optional — mark coverage anyway, but missing an optional one is not by itself unacceptable.
`errors` from: `unsupported_assertion`, `scope_overreach`, `wrong_citation`, `omitted_required_fact`, `premature_finish`, `wrong_label`, `other`.
- `omitted_required_fact`: a must-cite requirement is not answered.
- `premature_finish`: the answer concludes while the cited findings show the question is still open / evidence insufficient.
- `wrong_label`: the verdict direction (e.g. "manipulated"/"authentic", "matches"/"does not match", "earliest source is X") contradicts the findings/evidence.
Judge the answer against the Findings it cites and the directly cited evidence, not against external truth. If the Findings themselves look wrong but the answer faithfully reports them, note it, and judge the answer mainly on faithfulness + coverage + hedging.
Unacceptable = a must-cite requirement unanswered, a wrong verdict direction, or a central claim unsupported. Minor = wording/precision issues or an optional requirement missed.

## Kind: plan
Question: Is this round's set of proposed tasks reasonable overall?
`main` ∈ `good`, `minor` (basically reasonable, small issues), `bad`, `cannot_tell`.
`tasks`: for each proposed task T1..Tn (in order), `"yes"` if it targets an unresolved gap given `reports_so_far` (round 1: any gap of the overall task counts), `"no"` if it re-investigates something already resolved or is unrelated to any gap, `"unsure"`.
`errors` from: `task_not_gap_driven`, `overlapping_sibling_tasks`, `premise_smuggling`, `dependent_sibling_tasks`, `redundant_reinvestigation`, `other`.
- `premise_smuggling`: a task text states as fact something not yet established by reports_so_far ("verify whether claim X is true" is NOT smuggling; "since the video was filmed in Paris, find…" when nothing established Paris IS smuggling).
- `overlapping_sibling_tasks`: two tasks in the same round would do substantially the same work.
- `dependent_sibling_tasks`: a task needs the result of another task in the same round (tasks in one round run in parallel).
Also check: is anything obviously necessary for the overall task missing from the plan? Mention in note (does not have its own error type; use `other` if it makes the plan bad).

## Output
Write ONLY a JSON object (no prose around it) to the output path you were given, shaped like:
```
{"id": "...", "kind": "...", "run": "...", "main": "...", "errors": [...], "span": "...", "note": "...", "req": {...}, "tasks": {...}, "seconds": 0}
```
- `span` only for finding (else omit). `req` only for answer (else omit). `tasks` only for plan (else omit).
- `note`: 1–5 sentences, in English, concrete: which claim, which evidence id, why. For each error also say major/minor. If `main` is full/acceptable/good, say briefly what you checked.
- Do not add keys other than these.



## g2a-specific checks (batch guide, Chinese, authoritative for family-specific points)
The batch guide below is in Chinese. Follow its family-specific checks. Key points restated in English:
- All S0–S5 teacher slots in this batch were written by Codex (gpt-6-astra); judge the text exactly as for any other batch.
- A family (find the original video): when the statement says "the original is X", do the cited compare_media correspondences / frames actually prove it is the same footage? Do the removed-interval times (both sides) match the compare_media mapping? If no original was found, has the statement asserted edits beyond what the given material itself shows? A date lookup gives THIS upload's platform date / earliest archive — it must not be stated as "the earliest source" / "original publication".
- B family (VideoSham A-vs-B): are the difference positions / times exactly what compare_media lists or the frames show? Tool measurements quoted verbatim with attribution to the tool are NOT an error (decided 10-16). Region OCR: does the statement reproduce the recognised text and its position faithfully?
- C family (video claim check): frame conclusions — is it really visible in the cited frames; is "not seen" overstated as "not there"? Full-transcript summaries — cover the whole transcript, nothing added. External sources — attribution ("according to …") consistent with the cited page.
- Answers: GroundLie360 Fake posts often end "cannot be confirmed from the obtained evidence" because fact-check sites were blocked by a guardrail; that is acceptable if the attempts are shown. A-family "original not found, cannot determine" written honestly is acceptable. Never judge against ground truth.
- Known verifier false-positive patterns from r1a (write "verifier-FP-pattern: …" in the note if you see them): attribution_drift flagged although the statement attributes; sloppy_numeric flagged on a verbatim tool measurement with attribution; compare_media grading bands not recognised (≤2% same picture, ≤35% same with local differences).

### Original batch guide (Chinese)
# g2a 金标标注指南（2026-10-19，补量格专项）

15 条：**10 条 Finding**（F001–F010）+ **5 条答案**（A001–A005），全部来自 g2（轮次 `r1g2`，60 个 run：FMNV-A 40 + VideoSham 20，
S0–S5 全走 Codex，冻结口径 `specs/g1_freeze.json`；verifier@0.4.0，`verdicts_r1g2_judged…60runs-20b3224e.jsonl` / `verdicts_units_r1g2_judged…`）。
格式与 r1a / g1a 相同：`items.json`（证据正文上限 8000 字）、`strata.json`（**标注时不要打开**）、`media/`、`标注.html`。
复算：`python scripts/_gold_eval.py --batch experiments/sft_data/gold_set/g2a --labels labels_<标注人>.json`。

## 这批查什么

g2 是按教学矩阵的缺口格定向补的（A 族的 SEARCH text→video / text_image→video、原片日期、compare_media 视频模式；B 族的区域 OCR），
所以 Finding 按"所引证据是哪种新动作"抽，看这些新动作写出的结论是否被证据支持。

## 怎么抽的（标注人可以知道，但不要据此猜某一条是否被标错）

`_gold_sample.py --findings 400 --answers 60 --plans 0 --seed 20261019g2 --round-prefix r1g2 --only-cases walker_r1g2_done_cases.json`
抽出 g2 全部 191 条 Finding + 60 条答案作草稿（`claude/_g2_work/g2a_draft/`），再用 `claude/_g2_work/g2a_subset.py` 按所引证据的
acquired_by 归类取：A_compare（A 族、引 compare_media 记录）3、A_frames（引剪点旁的帧）2、A_dates（引原片的日期查询）2、B_ocr（引放大
区域的 OCR，VideoSham 文字篡改）2（g2 里只有 2 个 run 可取）+ B_compare 1 补足；类内核验器标 major/critical 的先取（10 条里 7 条标错），
每 run 至多 1 条。答案：A 族 3（negative 2、positive 1）、B 族 2（各 1）。

## Finding：判"所引证据是否支持这条陈述"

`main`：full / partial / none / cannot_tell，口径同 r1a / g1a 指南。按类多看一眼：

1. **A_compare**：compare_media 报的对应段 / 删掉的区间，陈述照抄得对不对（两侧时间、"B x–y s 在 A z s 处被略去"）；"找到的就是原片"有没有
   超出比对能证明的（找到的可能是另一份转载）。
2. **A_frames**：剪点两侧的帧是否真显示了陈述说的过渡；"帧里没看到"有没有写成"没有"。
3. **A_dates**：日期查询给的是这一份上传的平台日期 / 最早存档，陈述有没有把它说成"最早出处"。
4. **B_ocr / B_compare**：放大区域 OCR 认出的字、对照图里的差异框，陈述的文字与位置是否一致；工具测量值照抄并归因给工具不算错。

## 答案：判"按所引 Finding / E，答案能不能接受"

`main`：acceptable / minor / unacceptable，口径同前：每句有支撑、回答了每条要求、结论不过头；**不要拿 GT 判**（没找到原片如实写
"无法确定"可以接受——g1 里 A 族这类报告 20/20 被核验判 negative，这批答案的 negative 2 条正是要看的）。

## 交回

导出的 `labels_<标注人>.json` 放回本目录。新发现的核验器误报模式只记录。


## Note field
If you see a pattern that an automatic verifier would plausibly get wrong (e.g. tool measurement copied verbatim; ±1 day date rounding; "no fragment" claims), add "verifier-FP-pattern: ..." at the end of the note.
