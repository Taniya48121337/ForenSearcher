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



## g1a-specific checks (batch guide, Chinese, authoritative for family-specific points)
The batch guide below is in Chinese. Follow its family-specific checks. Key points restated in English:
- All S0–S5 teacher slots in this batch were written by Codex (gpt-6-astra); judge the text exactly as for any other batch.
- A family (find the original video): when the statement says "the original is X", do the cited compare_media correspondences / frames actually prove it is the same footage? Do the removed-interval times (both sides) match the compare_media mapping? If no original was found, has the statement asserted edits beyond what the given material itself shows? A date lookup gives THIS upload's platform date / earliest archive — it must not be stated as "the earliest source" / "original publication".
- B family (VideoSham A-vs-B): are the difference positions / times exactly what compare_media lists or the frames show? Tool measurements quoted verbatim with attribution to the tool are NOT an error (decided 10-16). Region OCR: does the statement reproduce the recognised text and its position faithfully?
- C family (video claim check): frame conclusions — is it really visible in the cited frames; is "not seen" overstated as "not there"? Full-transcript summaries — cover the whole transcript, nothing added. External sources — attribution ("according to …") consistent with the cited page.
- Answers: GroundLie360 Fake posts often end "cannot be confirmed from the obtained evidence" because fact-check sites were blocked by a guardrail; that is acceptable if the attempts are shown. A-family "original not found, cannot determine" written honestly is acceptable. Never judge against ground truth.
- Known verifier false-positive patterns from r1a (write "verifier-FP-pattern: …" in the note if you see them): attribution_drift flagged although the statement attributes; sloppy_numeric flagged on a verbatim tool measurement with attribution; compare_media grading bands not recognised (≤2% same picture, ≤35% same with local differences).

### Original batch guide (Chinese)
# g1a 金标标注指南（2026-10-18，Codex 教师专项）

20 条：**14 条 Finding**（F001–F014）+ **6 条答案**（A001–A006），全部来自 g1（轮次 `r1g1`，147 个 run，S0–S5 全走 Codex
`gpt-6-astra`，`specs/g1_teacher.json`；verifier@0.4.0，`verdicts_r1g1_judged…147runs-9460dc9a.jsonl` / `verdicts_units_r1g1_judged…`）。
格式与 r1a 相同：`items.json`（要看的内容，证据正文上限 8000 字）、`strata.json`（抽样层与核验器判定，**标注时不要打开**）、`media/`、
`标注.html`（双击打开，导出 `labels_<标注人>.json`）。
复算：`python scripts/_gold_eval.py --batch experiments/sft_data/gold_set/g1a --labels labels_<标注人>.json`。

## 这批查什么

g1 与 r1 的唯一差别是 **S0–S3（动作理由、检索词、第 2 轮任务文本、选页）由 Qwen 换成了 Codex**；S4 报告 / S5 答案两边都是
gpt-6-astra。所以这批的目的：看换教师后，报告 / 答案的 Finding 是否仍被所引证据支持，核验器在 A / B / C 族上的标错是否靠得住。

## 怎么抽的（标注人可以知道，但不要据此猜某一条是否被标错）

- `_gold_sample.py --findings 14 --answers 6 --plans 0 --seed 20261018g1 --oversample-flagged --answers-balanced --round-prefix r1g1
  --only-cases walker_r1g1_done_cases.json` → 草稿 25 条 Finding（g1 全部核验器标 major/critical 的 Finding）+ 6 答案；
  再按族收窄到 14 条（`claude/_g1_work/g1a_subset.py`：B 1——g1 里 B 族被标错的只有这 1 条、A 4、C 9 = GroundLie360 4 + FMNV-C 5，
  每个 run 至多 1 条，按草稿顺序取）。**14 条全是核验器标错的**，比例不代表总体。
- 答案：核验器答案层 negative 3 条、positive 3 条（A 族 2、C 族 3、B 族 1）。

## Finding：判"所引证据是否支持这条陈述"

`main`：**full** / **partial** / **none** / **cannot_tell**，有问题勾错误类型、`span` 抄片段、`note` 一句理由——口径同 r1a 指南。
按族要多看一眼：

1. **A 族（找原片）**：陈述说"原片是 X"时，所引的 compare_media 对应段 / 帧真能证明是同一段视频吗？删掉的区间（两侧时间）与
   compare_media 的映射一致吗？没找到原片时，有没有把"材料本身看得出的"以外的改动写成结论？
2. **B 族（VideoSham 两段对比）**：陈述里的差异位置、时间是否就是 compare_media 列出 / 帧里看得到的；工具测量值照抄并归因给工具
   不算错（10-16 已定）。
3. **C 族（视频声明核查）**：帧结论——帧里真看得到吗，"没看到"有没有写成"没有"；整段转写概括——是否覆盖全段、有没有转写里没有的话；
   外部来源——陈述的归因（according to …）是否与所引页面一致。

r1a 的已知核验器误报模式（有归因仍标 attribution_drift、B 族照抄测量值标 sloppy_numeric、compare_media 分档不认）遇到时在 note 里写
"误报模式：…"。

## 答案：判"按所引 Finding / E，答案能不能接受"

`main`：**acceptable** / **minor** / **unacceptable**，口径同 r1a：每句有支撑、回答了每条要求、结论不过头；**不要拿 GT 判**。
GroundLie360 的 Fake 帖子答案多是"按现有证据无法证实"（Snopes 被检索护栏挡住）——按证据如实写"无法证实"是可以接受的。

## 交回

导出的 `labels_<标注人>.json` 放回本目录。新发现的核验器误报模式只记录，本轮不修核验器。


## Note field
If you see a pattern that an automatic verifier would plausibly get wrong (e.g. tool measurement copied verbatim; ±1 day date rounding; "no fragment" claims), add "verifier-FP-pattern: ..." at the end of the note.
