# cases_draft：自动起草的案卷（待人工审核）

生成（2026-09-24，12 个 run → 11 份案卷 + 1 份 `.from_`）：

```
PYTHONPATH=src python -m foren_searcher.datagen.draft_case --out experiments/sft_data/cases_draft --runs \
  experiments/sft_data/runs/{true_17838107_r1,true_20205316753904_validation,true_20361105_visibility_validation,\
true_24171541709824_r1,true_39132300_validation_r3,true_4124903_r1,true_4124903_r2,videosham_0069_r1,\
videosham_1007_validation,videosham_1028_r2,videosham_2010_r1,videosham_4126_split_intervals_r7}
```

规则见 `src/foren_searcher/datagen/draft_case.py` 模块 docstring。要点：

- 任务分两套：**声明核查**（true）按 外部记载 / 媒体内容 / 来源日期 三类起草；**媒体对照**（videosham）按
  差异位置 / 画面区域 / 声音 / 对含义的影响 / 未能确定的内容 五类起草，不套"外部可靠来源"模板，
  source_policy=provided_media，task_kind 取 `video_diff`（找可见区间的 4126 取 `temporal_locate`）。
- 基线 report 判定先过 `verify.calibrate.recalibrate` 再算干净 Finding；答案所引 E 没有一条被干净 Finding 引用时
  **来源降级**为"答案引用的 E"，每一项的来源级别写在审核单 `trace.<key>.origin`
  （`clean∩answer` / `clean_finding_only` / `answer_only` / `run_only`）。
- 引用括号认 （E1）/(E1)/[E1]/【E1】，相邻括号 `[E3][E6]` 合并成一句。
- `date_in_range` 保守：日期须在答案首段结论里、且能在所引文本来源正文里找到，才写 ±1 天；否则不写并在 notes 说明。

文件：

- `<dataset>_<id>.json`：案卷草稿，`Case.load` 可加载；审核通过后挪进 `../cases/`。
- `<name>.review.md`：人读的审核单（generated_from / generated_at / needs_review / task_family / requirement_origin /
  notes 逐条 / 证据 trace，末尾附原 JSON 块，`draft_case.read_review()` 可读回）。`Case` 不收多余字段，所以 draft 块放这里；
  因为是 .md，本目录可以直接 `load_cases()`——**`.from_` 文件除外**（见下）。
- 同一案例多个 run：草稿默认叫 `<dataset>_<id>.json`；若它已存在且审核单的 generated_from 是另一个 run，新草稿写成
  `<dataset>_<id>.from_<run>.json`（审核单同名 `.review.md`），审核单第一条提示"已有另一 run 的草稿"。这时两份 case_id 相同，
  `load_cases()` 会报重复——审核时二选一或合并，删掉另一份。同一 run 重跑原地覆盖。

## 12 个 run 的起草结果（2026-09-24）

requirement 来源：task = 取自任务分句；claim_template = 中性模板 + 声明原文（首轮任务只复述了总任务，抽不出分句）。
required_evidence 列为 文本（url_pattern）/ 派生（acquired_by）。所有草稿都没有 `label_expected`（仓库里没有数据集标注）。

| 草稿 | run | task_kind | requirements | required_evidence 文本/派生 | checks | 主要审核提示 |
|---|---|---|---|---|---|---|
| true_17838107 | true_17838107_r1 | claim_check | 2（task） | 3 / 1（frames） | — | recalibrate 撤销 W1-1 F2 的污染；REQ1 文字缺主语（"这种影响"指声明），可补 |
| true_20205316753904 | true_20205316753904_validation | claim_check | 2（claim_template） | 0 / 2（frames、crops） | — | **来源降级**：唯一的 F1 校准后仍污染，证据取答案引用的 E3/E6/E10/E11；run 没做过 SEARCH，**没有文本来源项**；REQ1（外部记载）答案无回应，审核时决定删掉还是保留 |
| true_20361105 | true_20361105_visibility_validation | claim_check | 3（task） | 3 / 1（frames） | — | 答案没引帧，frames 取干净 Finding F1 的 E4/E6（clean_finding_only）；REQ2（视频内容）答案无回应；recalibrate 撤销 W1-2 F5 |
| true_24171541709824 | true_24171541709824_r1 | claim_check | 2（task） | 4 / 0 | — | 1863-07-03 是历史背景，不在首段结论，**不写 date_in_range**；REQ1 很长（T1 全部分句），可精简；run 没抽帧 |
| true_39132300 | true_39132300_validation_r3 | claim_check | 2（claim_template） | 2 / 0 | — | 首轮任务只复述总任务，要求是声明绑定模板，需人工改写；REQ2（媒体）答案无回应，run 也没看媒体，多半该删 |
| true_4124903 | true_4124903_r1 | claim_check | 3（task） | 4 / 1（frames） | entity 耶拿桥；date 2016-07-13..15 | recalibrate 撤销 F3/F5/F7 后 E24 入选，prefpolice 模式放宽到 `7537*`（与手写一致） |
| true_4124903.from_true_4124903_r2 | true_4124903_r2 | claim_check | 3（task） | 3 / 1（frames） | entity 耶拿桥；date 2016-07-13..15 | 与上一份 case_id 相同，二选一；E6 只被有问题的 Finding 引用，未入选 |
| videosham_0069 | videosham_0069_r1 | video_diff | 4（task） | 0 / 2（frames、media_profile） | — | 首轮任务 2 个含时间的分句没写进要求；recalibrate 撤销 W1-1 F1–F3 |
| videosham_1007 | videosham_1007_validation | video_diff | 4（task） | 0 / 2（alignment、frames） | — | 答案里"右缘/右下"句归到 region，但任务没有画面区域要求；E48 未入选 |
| videosham_1028 | videosham_1028_r2 | video_diff | 4（task） | 0 / 2（alignment、frames） | — | 答案的音频句没有对应要求（本 run 首轮任务没提音频）；E18 未入选 |
| videosham_2010 | videosham_2010_r1 | video_diff | 4（task） | 0 / 3（alignment、frames、crops） | — | REQ1（差异位置）答案无单独回应句（位置写在含义句里）；E15/E16 未入选 |
| videosham_4126 | videosham_4126_split_intervals_r7 | temporal_locate | 2（task） | 0 / 1（frames） | — | task_kind 也可取 video_diff，审核时定；首轮任务是录制的续做计划，14 个含时间分句没写进要求；答案所引多数帧只经过污染 Finding，未入选 |

验收（对照 2026-09-24 要求）：

- required_evidence 全部非空（12/12）。true 类"≥1 文本来源 + 帧（若抽过帧）"：6 份达标，true_20205316753904 不达标——
  run 根本没做 SEARCH/open，没有可用的文本来源，按"不引入 run 里没有的事实"不编；videosham 类 5/5 至少 1 条派生证据。
- requirements 不是中性模板：11 份案卷里 9 份全部取自任务分句；true_20205316753904、true_39132300 是声明绑定模板（validation run
  的首轮任务逐字复述总任务，没有更具体的文字可取）。没有一份用纯中性模板。
- 草稿目录去掉 `.from_` 文件后 `load_cases()` 通过（11 个 case_id）。

## true_4124903 草稿 vs 手写案卷 `cases/true_4124903.json`

| 字段 | 结果 | 说明 |
|---|---|---|
| case_id / dataset / source_id / split / task_kind / family_key / task_text / claims / source_policy | 对 | 与手写一致 |
| media | 对 | `I0 → ../runs/true_4124903_r1/artifacts/6ddac….mp4`；mp4 本身没入库，路径按 store 登记写（notes 里有提示） |
| REQ1 外部来源对火情原因/是否恐袭的记载 | 对 | 取自 T2 分句，与手写几乎同文 |
| REQ2 视频画面能否支持恐袭说法 | 对 | 取自 T1 分句；多带了一句"是否对应…场景" |
| REQ3 来源与事件语境，must_cite=False | 基本对 | 文字是 T1 的"来源与事件语境"，没有"日期"字样，审核时可改 |
| required_evidence `prefpolice_tweets` | 对 | recalibrate 后 E24 所在的 F5 不再污染，E21+E24 → `*prefpolice/status/7537*`，同手写 |
| `snopes` | 对，must_cite 不同 | pattern 同手写；草稿 must_cite=True（手写 False，靠 any_one_of 表达可选） |
| `frames` | 对 | `acquired_by=INSPECT/get_frames`，supports REQ2 |
| `youtube_upload` | 对 | pattern、must_cite=False、supports REQ3 均同手写 |
| `nydailynews`（E22） | 多 | 手写没有；干净 F6 引用、答案也引用，放进了 REQ1 的 any_one_of 组 |
| required_evidence_rule | 基本对 | `any_one_of:nydailynews,prefpolice_tweets,snopes; all:frames`（手写同结构，少 nydailynews） |
| checks `label_expected` | 漏 | 仓库里没有 TRUE 数据集标注；按规则不从答案反推。给 `--labels` 就会写 |
| checks `entity_in_evidence 耶拿桥` | 对 | 旁证是答案判定里已核实的"耶拿桥上"片段 |
| checks `date_in_range 2016-07-13..15` | 对 | 日期在答案首段结论里，且见于 E11（Snopes，"14 July 2016"）与 E16（YouTube 元数据 20160714）正文 |
