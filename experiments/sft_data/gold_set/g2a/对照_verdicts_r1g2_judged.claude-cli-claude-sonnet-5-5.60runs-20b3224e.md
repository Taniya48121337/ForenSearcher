# 金标对照：g2a × verdicts_r1g2_judged.claude-cli-claude-sonnet-5-5.60runs-20b3224e.jsonl / verdicts_units_r1g2_judged.claude-cli-claude-sonnet-5-5.60runs-20b3224e.jsonl

## 标注人 opus

| 单元 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 | 召回 | 一致率 | κ | 无法判断 | 用时中位数（分钟） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finding | 10 | 5 | 2 | 0 | 3 | 0.714 | 1.0 | 0.8 | 0.6 | 0 | 0.0 |
| answer | 5 | 1 | 2 | 0 | 2 | 0.333 | 1.0 | 0.6 | 0.286 | 0 | 0.0 |
| plan | 0 | 0 | 0 | 0 | 0 | None | None | None | None | 0 | — |

（上表答案 / 计划按 disposition；Finding 按原始判定的 major/critical。下面分层见各小节。）

### Finding · major/critical 分层（清单 D1）

| 口径 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 [95%] | 召回 [95%] | κ | 未标错里人工认为有错 |
|---|---|---|---|---|---|---|---|---|---|
| 原始判定 · major/critical | 10 | 5 | 2 | 0 | 3 | 0.714 [0.36, 0.92] | 1.0 [0.57, 1.00] | 0.6 | 0 / 3 |
| 校准后（export 用）· major/critical | 10 | 5 | 2 | 0 | 3 | 0.714 [0.36, 0.92] | 1.0 [0.57, 1.00] | 0.6 | 0 / 3 |
| 原始判定 · 含 minor | 10 | 5 | 2 | 0 | 3 | 0.714 [0.36, 0.92] | 1.0 [0.57, 1.00] | 0.6 | 0 / 3 |
| 校准后 · 含 minor | 10 | 5 | 2 | 0 | 3 | 0.714 [0.36, 0.92] | 1.0 [0.57, 1.00] | 0.6 | 0 / 3 |

"含 minor" 两行是假设 export 把 minor 也 held 时的效果，供 D1 收紧参考。标错条数少时精确率区间很宽，要和召回、最后一列一起看。

| 核验器标错类型（原始 major/critical） | 标错 | 人工认为有问题 | 人工认为没问题 | 条目（✗ = 被校准撤销） |
|---|---|---|---|---|
| sloppy_numeric | 1 | 1 | 0 | F010（partial） |
| unsupported_assertion | 6 | 4 | 2 | F001（partial）、F002（full）、F003（partial）、F004（full）、F006（partial）、F007（partial） |

| 人工勾选的错误类型 | 人工有问题 | 核验器标出（校准后 major/critical） | 只在 minor 层标出 | 完全未标 | 没标出的条目 |
|---|---|---|---|---|---|
| unsupported_assertion | 4 | 4 | 0 | 0 | — |
| sloppy_numeric | 1 | 1 | 0 | 0 | — |

误报（人工认为没问题、核验器标了 major/critical）2 条：

| 条目 | run | 工作区 / Finding | 原始类型 | 校准后类型 | 人工 | 备注 |
|---|---|---|---|---|---|---|
| F002 | fmnv_a_6C0lUMKMj0w_r1g2_1 | W1-1 F3 | unsupported_assertion | unsupported_assertion | full | Checked each claim against the cited evidence. The first sentence only lists what was cited: E43 (compare_media result), E48 (date lookup for the candidate YouTube URL) and E11 (ASR on Material A), and all three are present. In E11, processed_ranges is [[0.0, 71.78004166666666]], and the segment from 14.0 to 20.56 s begins 'It was so well put together', so both match exactly. The statement draws no conclusion about the original or its dates beyond this, so it has no overreach (it does very little for the task, but every claim is supported). verifier-FP-pattern: the long float 71.78004166666666 is copied verbatim from the tool, which should not be flagged as sloppy_numeric. |
| F004 | fmnv_a_h3uASpEE75Q_r1g2_1 | W1-1 F3 | unsupported_assertion | unsupported_assertion | full | The statement makes only provenance claims: that E39 is the whole-video compare_media result (it is, comparing A=E1 with B=E37), and that frame records exist for A at 11.545 s (E4) and 34.635 s (E6) and for B at 16.884 s (E40, actual time_s 16.884) and 50.617 s (E41, time_s 50.617). All four times match the cited frame records exactly, and the images are readable: a THV11 studio, Trump at a podium, and two Denver7 anchors. The statement claims no correspondence between the frames, and its limitations say outright that sampling times do not establish correspondence or coverage, so nothing overreaches. The finding is nearly content-free, though: it does not report E39's actual conclusion (B is probably not the source, only 9% matched, sound similarity 0.22), but that is an omission, not an unsupported claim. verifier-FP-pattern: a requested_time vs actual time_s difference (E40 requested 16.869, actual 16.884) could be flagged as sloppy_numeric, but the statement uses the actual time_s. |

### 答案 / 计划 · major/critical 层（与 export B1 的 held 口径一致）

| 口径 | n | 人工有问题且核验器标出 | 核验器误报 | 核验器漏报 | 都认为没问题 | 精确率 [95%] | 召回 [95%] | κ | 未标错里人工认为有错 |
|---|---|---|---|---|---|---|---|---|---|
| answer | 5 | 1 | 2 | 0 | 2 | 0.333 [0.06, 0.79] | 1.0 [0.21, 1.00] | 0.286 | 0 / 2 |

答案按校准后的错误；计划的错误是从核验日志恢复的，日志里没有严重度，按错误类型默认值算（task_not_gap_driven / premise_smuggling = major，unnecessary_round = minor）。

| 条目 | run | 人工 | 人工错误类型 | 核验器 disposition | 核验器 major/critical | 备注 |
|---|---|---|---|---|---|---|
| A001 | fmnv_a_F51nH6oLGfY_r1g2_1 | acceptable | — | negative | premature_finish | Checked each sentence against E41/E46: the candidate URL and the 2025-01-06 upload/earliest-archive dates match E46 and are presented only as this candidate's dates, not as the original publication; the 27% consistent-match figure, the sound difference A 47.0-90.0 s vs B 0.2-43.2 s, envelope correlation 0.16 and 0/85 windows are all verbatim from E41. REQ1 and REQ2 are not established (no source confirmed, no edit inventory), but the answer states this honestly as undetermined with the supporting measurement, which the batch guide treats as acceptable for A-family negatives; REQ3 (what could not be determined) is covered well, including the explicit caveat that the video is not shown to be unedited. Small precision point (not an error): E41 also lists two partial corresponding spans (A 16.2-19.2 s = B 42.6-45.1 s; A 25.2-40.2 s = B 51.5-66.0 s) that the answer omits, so 'paired timestamps could not be determined' slightly understates the partial overlap, though these spans do not establish the candidate as the source. verifier-FP-pattern: honest 'original not found / cannot determine' answer with must-cite requirements unmet may be flagged as omitted_required_fact or premature_finish; tool measurements quoted verbatim with attribution. |
| A002 | fmnv_a_0zC0K0YZEzg_r1g2_1 | acceptable | — | negative | premature_finish | Checked every line against E25: 95% match, B 2-130 s range, the four removed intervals (B 26.6-42.6 @A27.8, 51.8-65.0 @A37.8, 70.9-79.4 @A44.2, 97.5-118.6 @A63.2), reversal A 1.8-5.2=B 1.9-4.5, freeze A 8.8-11.2 (B 10.5), and sound difference A 28.0-29.5 vs B 42.6-44.1 are all reproduced exactly. E38 gives only this upload's platform date (2025-01-17), and the answer correctly calls it a 'matching source copy' with 'earliest-original status unverified', so it does not overreach to 'original publication'. Limitations (0.5 s precision, reversal/freeze not visually confirmed, sound cause unknown, added footage/alterations unverified) cover REQ3. Tiny nit, not an error: A 5.2-8.8 s and 11.2-16.2 s are not in E25's corresponding_spans and the answer does not mention them, but the tool itself does not list them as differences either. verifier-FP-pattern: compare_media measurements (incl. 'B 10.5-10.5 s', spectral-based sound difference) copied verbatim with attribution to the tool. |
| A003 | fmnv_a_9ACukGwm6O8_r1g2_1 | minor | unsupported_assertion | positive | — | Checked against E26 and E33. The URL, the KRGV channel and the 2024-09-25 upload date match E33, and the answer correctly calls the earliest upload unverified rather than calling it the original. The 100% sample match, the removed span (B 31.3-43.1 s, 11.8 s, skipped at A 31.8 s), all four sound-difference intervals on both sides, the three matching face tracks and the 0.5 s boundary limitation are all copied accurately from E26. One minor error: the video title 'Cards Against Humanity suing SpaceX for allegedly trespassing on Boca Chica property' appears in neither E33 nor E26, so it is unsupported. This does not change the verdict, because REQ1 is already met by the URL and channel. The 'no added or reordered footage' claim is attributed to the comparison ('The comparison lists...'), and the content of the sound changes is honestly listed as not established. verifier-FP-pattern: the sound-difference spectral similarities are quoted as the tool's own intervals and should not be flagged as sloppy_numeric. |
| A004 | videosham_0099_r1g2_1 | unacceptable | omitted_required_fact | negative | premature_finish | Each sentence checks out against its citation: the 94% match, the B 0-8 s location and the 12/12 sound windows match E21, and the transcript text, 4.68-6.16 s timing and 0-6.2 speech / 6.2-8.1 other-sound labels match E32. The hedges about B's source identity and unsampled regions are honest. REQ1 (must-cite: does A show the described screen flicker, confirmed by the picture, with positions on both sides) is not answered (major). No frame or region evidence is cited at all. The answer also says the locations 'cannot be determined', yet E21's corresponding_spans show an unmatched gap at A 3.3-3.8 s vs B 3.0-4.0 s, and the answer never reports it. REQ2 is covered by E21/E32, and REQ3/REQ4 are covered only as honest 'cannot determine' statements, since no difference was confirmed visually. verifier-FP-pattern: a hedged 'cannot be determined' answer with all sentences supported may pass sentence-level checks, but the investigation stopped without any visual inspection. |
| A005 | videosham_0109_r1g2_1 | acceptable | — | positive | — | Checked images: E21 (A/B at 5.2 s) shows three foreground people in B with a central man in black shirt and gray/blue shorts inside the red box, and pavement in A at the same spot; E25 (B, 0.000 s crop) shows a person in black shirt and gray shorts, E28 (A, 0.000 s crop) shows empty pavement, so the visual difference, its location and the three-vs-two impact are supported (REQ1, REQ3). The 100% sample match, A/B [0.000, 5.680) s pixel-difference segment and 10/10 sound windows are copied faithfully from E24, and the ASR results (A no segments, B 'You' 0-2 s) match E31/E32 and are hedged as not establishing an audible difference (REQ2). The answer does not presume B's source identity or editing direction and lists unverified items (unsampled intervals, audible content, faces) (REQ4). Only cosmetic awkwardness: the sentence that the measured interval 'does not establish the person's absence through continuous visual inspection' is oddly phrased but not wrong. verifier-FP-pattern: tool measurements (100% samples, 10/10 windows, frame-exact interval) quoted verbatim with attribution to the comparison. |

### D1 判定（Finding，major/critical）

- 核验器原始判定：精确率 0.714（5 / 7，95% [0.36, 0.92]）< 0.85 → 收紧 export；召回 1.0（5 / 5）；未标错里人工认为有错 0 / 3
- 校准后（export 实际用的）：精确率 0.714（5 / 7，95% [0.36, 0.92]）< 0.85 → 收紧 export；召回 1.0（5 / 5）；未标错里人工认为有错 0 / 3

双方都认为有问题的 Finding 里，人工勾选的错误类型与核验器同类 5 次、异类 0 次。
