# 教学矩阵（accepted 行 / 案例；Q = Qwen 版 S0–S3 行，X = Codex 版行；⚠ 缺口：行 < 15 或案例 < 5；— 该族 walker 不走）

| 族 | accepted 行 | 案例 |
|---|---|---|
| A | 131 | 29 |
| B | 315 | 50 |
| C | 1585 | 136 |
| E | 423 | 35 |
| E（不可回答） | 209 | 15 |
| L | 121 | 8 |
| L（不可回答） | 68 | 4 |

| 组 | 列 | A | B | C | E | E（不可回答） | L | L（不可回答） |
|---|---|---|---|---|---|---|---|---|
| SEARCH | 文字搜 → 网页 | — | — | 235/134（Q39 X196） | ⚠ 13/11（Q13 X0） | 15/12（Q15 X0） | ⚠ 7/7（Q7 X0） | ⚠ 5/4（Q5 X0） |
| SEARCH | 文字搜 → 图片（参考图） | — | — | — | — | — | ⚠ 11/8（Q11 X0） | ⚠ 6/4（Q6 X0） |
| SEARCH | 文字搜 → 视频 | ⚠ 13/9（Q0 X13） | — | — | — | — | — | — |
| SEARCH | 以图搜 | — | — | — | 35/35（Q35 X0） | 15/15（Q15 X0） | ⚠ 8/8（Q8 X0） | ⚠ 4/4（Q4 X0） |
| SEARCH | 图 + 文字搜（text_image） | ⚠ 7/7（Q0 X7） | — | — | — | — | — | — |
| SEARCH | 查页面日期（include dates） | ⚠ 8/8（Q0 X8） | — | 18/15（Q10 X8） | 99/35（Q99 X0） | 40/15（Q40 X0） | — | — |
| SEARCH | 打开页面（source → text） | — | — | 27/13（Q0 X27） | ⚠ 8/4（Q8 X0） | ⚠ 7/5（Q7 X0） | ⚠ 11/5（Q11 X0） | ⚠ 9/3（Q9 X0） |
| SEARCH | 取页内主图（source → image） | — | — | — | ⚠ 0/0（Q0 X0） | ⚠ 4/4（Q4 X0） | ⚠ 6/4（Q6 X0） | ⚠ 4/3（Q4 X0） |
| SEARCH | 打开视频（source → video） | — | — | ⚠ 7/7（Q1 X6） | — | — | — | — |
| READ_RECORD | 读页面正文 | — | — | 422/136（Q123 X299） | 71/35（Q71 X0） | 52/15（Q52 X0） | 27/7（Q27 X0） | ⚠ 11/4（Q11 X0） |
| READ_RECORD | 续读（offset > 0） | — | — | ⚠ 7/7（Q3 X4） | — | — | — | — |
| READ_RECORD | 读工具记录（OCR / 结构等） | — | ⚠ 1/1（Q0 X1） | 15/12（Q1 X14） | 15/10（Q15 X0） | ⚠ 8/6（Q8 X0） | ⚠ 7/3（Q7 X0） | ⚠ 7/4（Q7 X0） |
| READ_RECORD | 看参考图（L） | — | — | — | — | — | 17/8（Q17 X0） | ⚠ 9/4（Q9 X0） |
| READ_RECORD | 读整段转写 | — | — | — | — | — | — | — |
| READ_RECORD | 对照图回看 | — | ⚠ 1/1（Q1 X0） | — | 27/27（Q27 X0） | ⚠ 10/10（Q10 X0） | — | — |
| READ_RECORD | 交报告前回看帧（C9v） | — | — | 131/131（Q33 X98） | — | — | — | — |
| INSPECT | OCR 图片 | ⚠ 1/1（Q0 X1） | ⚠ 12/5（Q6 X6） | — | 33/33（Q33 X0） | ⚠ 14/14（Q14 X0） | ⚠ 7/7（Q7 X0） | ⚠ 4/4（Q4 X0） |
| INSPECT | OCR 视频 | 23/23（Q0 X23） | — | 126/126（Q33 X93） | — | — | — | — |
| INSPECT | ASR | 28/28（Q0 X28） | 56/28（Q31 X25） | 132/132（Q34 X98） | — | — | — | — |
| INSPECT | get_frames times_s | 28/11（Q0 X28） | 75/28（Q55 X20） | 107/107（Q25 X82） | — | — | — | — |
| INSPECT | get_frames + box 放大 | ⚠ 0/0（Q0 X0） | 30/13（Q17 X13） | — | — | — | — | — |
| INSPECT | get_frames count（均匀） | — | ⚠ 1/1（Q1 X0） | — | — | — | — | — |
| INSPECT | get_frames range_s | — | — | — | — | — | — | — |
| INSPECT | video_structure | — | ⚠ 1/1（Q1 X0） | 124/124（Q33 X91） | — | — | — | — |
| INSPECT | compare_media 图片模式 | — | — | — | 67/35（Q67 X0） | 21/14（Q21 X0） | ⚠ 11/8（Q11 X0） | ⚠ 6/4（Q6 X0） |
| INSPECT | compare_media 视频模式 | ⚠ 11/11（Q0 X11） | 52/45（Q34 X18） | ⚠ 3/3（Q1 X2） | — | — | — | — |
| INSPECT | compare_media range_a_s / region | — | ⚠ 5/5（Q3 X2） | — | — | — | — | — |
| INSPECT | locate | — | ⚠ 11/4（Q7 X4） | — | — | — | — | — |
| QUERY_MEMORY | QUERY_MEMORY | — | ⚠ 1/1（Q1 X0） | — | — | — | — | — |
| REPORT | 报告 COMPLETED | ⚠ 5/5（Q0 X5） | 19/18（Q12 X7） | 103/103（Q32 X71） | 24/24（Q24 X0） | ⚠ 0/0（Q0 X0） | ⚠ 2/2（Q2 X0） | ⚠ 1/1（Q1 X0） |
| REPORT | 报告 INCOMPLETE（walker / S4 自己判） | ⚠ 0/0（Q0 X0） | ⚠ 3/3（Q2 X1） | ⚠ 8/7（Q1 X7） | ⚠ 4/4（Q4 X0） | ⚠ 10/10（Q10 X0） | ⚠ 2/2（Q2 X0） | ⚠ 0/0（Q0 X0） |
| REPORT | 报告循环内改写为 INCOMPLETE | ⚠ 0/0（Q0 X0） | ⚠ 7/7（Q5 X2） | ⚠ 9/9（Q1 X8） | ⚠ 2/2（Q2 X0） | ⚠ 2/2（Q2 X0） | ⚠ 1/1（Q1 X0） | ⚠ 0/0（Q0 X0） |
| REPORT | 报告事后 status 改写 | ⚠ 0/0（Q0 X0） | — | ⚠ 0/0（Q0 X0） | — | — | — | — |
| GIA | GIA 第 1 轮派工 | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） | ⚠ 0/0（Q0 X0） |
| GIA | GIA 第 2 轮按缺口派工 | — | ⚠ 4/4（Q2 X2） | 18/18（Q2 X16） | ⚠ 2/2（Q2 X0） | ⚠ 8/8（Q8 X0） | ⚠ 1/1（Q1 X0） | ⚠ 2/2（Q2 X0） |
| GIA | 第 2 轮 Scope 第一项是取图比对 | — | — | — | — | — | ⚠ 1/1（Q1 X0） | ⚠ 1/1（Q1 X0） |
| GIA | GIA FINISH | ⚠ 7/7（Q0 X7） | 41/39（Q28 X13） | 93/93（Q28 X65） | 23/23（Q23 X0） | ⚠ 3/3（Q3 X0） | ⚠ 3/3（Q3 X0） | ⚠ 0/0（Q0 X0） |
| GIA | 第 2 轮后 FINISH | — | ⚠ 4/4（Q2 X2） | — | ⚠ 1/1（Q1 X0） | ⚠ 2/2（Q2 X0） | ⚠ 1/1（Q1 X0） | ⚠ 0/0（Q0 X0） |
| 习惯 | "候选"措辞（candidate） | ⚠ 1/1（Q0 X1） | ⚠ 5/4（Q3 X2） | ⚠ 14/10（Q6 X8） | ⚠ 11/10（Q11 X0） | ⚠ 13/11（Q13 X0） | ⚠ 5/3（Q5 X0） | ⚠ 1/1（Q1 X0） |
| 习惯 | 页面日期与图片日期分开 | — | — | — | 48/27（Q48 X0） | ⚠ 5/4（Q5 X0） | — | — |
| 习惯 | 归因保留（according to / reports …） | ⚠ 9/7（Q0 X9） | 56/34（Q37 X19） | 204/119（Q59 X145） | 47/29（Q47 X0） | ⚠ 12/10（Q12 X0） | ⚠ 7/4（Q7 X0） | ⚠ 1/1（Q1 X0） |
| 习惯 | 如实 INCOMPLETE（写明缺口） | ⚠ 0/0（Q0 X0） | ⚠ 10/10（Q7 X3） | 17/11（Q2 X15） | ⚠ 6/6（Q6 X0） | ⚠ 12/11（Q12 X0） | ⚠ 3/3（Q3 X0） | ⚠ 0/0（Q0 X0） |
| S4 结论 | 比对结论（compare_media / 参考图） | ⚠ 5/5（Q0 X5） | 28/27（Q18 X10） | ⚠ 4/4（Q0 X4） | 30/30（Q30 X0） | ⚠ 11/11（Q11 X0） | ⚠ 5/4（Q5 X0） | ⚠ 1/1（Q1 X0） |
| S4 结论 | 帧结论（画面里看到什么） | ⚠ 5/5（Q0 X5） | 23/22（Q13 X10） | 110/110（Q31 X79） | — | — | — | — |
| S4 结论 | 整段转写概括 | — | — | 101/101（Q30 X71） | — | — | — | — |
| S4 结论 | 区间（两侧时间对应） | ⚠ 5/5（Q0 X5） | 22/21（Q14 X8） | — | — | — | — | — |

工具目录里有、accepted 一次都没出现的形态（13）：
- `SEARCH input_type=image output_type=text`（三个桶里都没有：walker 从来不走）
- `SEARCH input_type=image output_type=video`（三个桶里都没有：walker 从来不走）
- `SEARCH input_type=text_image output_type=text`（三个桶里都没有：walker 从来不走）
- `SEARCH input_type=text_image output_type=image`（三个桶里都没有：walker 从来不走）
- `INSPECT get_frames · range_s`（三个桶里都没有：walker 从来不走）
- `INSPECT get_frames · pages`（三个桶里都没有：walker 从来不走）
- `INSPECT get_frames · page_range`（三个桶里都没有：walker 从来不走）
- `INSPECT ocr · range_s`（三个桶里都没有：walker 从来不走）
- `INSPECT asr · language`（三个桶里都没有：walker 从来不走）
- `INSPECT asr · range_s`（三个桶里都没有：walker 从来不走）
- `INSPECT compare_media · region`（三个桶里都没有：walker 从来不走）
- `INSPECT compare_media · min_area`（三个桶里都没有：walker 从来不走）
- `QUERY_MEMORY · cursor`（三个桶里都没有：walker 从来不走）

## 坏习惯（程序口径）

- 理由模板化：3/2852 = 0.1%；只算 S0 教师写的 0/1644 = 0.0%；分教师 {"codex": {"rows": 1428, "templated": 0, "s0_rows": 721, "s0_templated": 0}, "qwen": {"rows": 1424, "templated": 3, "s0_rows": 923, "s0_templated": 0}}；编号 / 数字换占位符后 3 行 = 0.1%（{"codex": 0, "qwen": 3}）
- 比对前就说匹配：296 行（12.1%）；断言式 92（3.8%）；分包 {"r1": {"flagged": 244, "assertive": 65}, "g1": {"flagged": 52, "assertive": 27}}
- GIA 答案有可见输入里没有的实体：6/170 = 3.5%；分包 {"g1": {"answers": 85, "with_miss": 1}, "r1": {"answers": 85, "with_miss": 5}}
