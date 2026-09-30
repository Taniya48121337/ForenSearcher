# 审查结果：来源 D 离线换题第二批（2026-09-29）

审查人：独立审稿，不是起草者。审查标准沿用 `review_20260928_labels_offline.md`。所有结论都附文件证据。
- 源状态：`uploads/.../runs/<run>/requests/<req>/request.json` 的 global_view。
- 证据原文：各 run 的 `store.sqlite3`，复制后读取。
- 污染判定：`verdicts_r0_judged.jsonl` 的 tainted_findings，加 `verdicts_units_judged.jsonl` 中 answer 单元的 `extra.inherits_from`。
- 改后的 task、expected_decision、rationale 全文见同目录的 `D第二批审查.json`。本文件只写理由和要点。

## 总览

| 文件 | 通过 | 修改 | 删除 | 新增 | 改后 F / I |
|---|---|---|---|---|---|
| true_4124903_r2 | v1 v2 | v3（改判 I）v4 | — | — | 2 / 2 |
| true_17838107_r1 | v2 v4 | v1 v3 | — | — | 2 / 2 |
| true_20205316753904_validation | v2 | v1 v3 | — | — | 1 / 2 |
| true_20361105_visibility_validation | v1 v3 v4 | v2 | — | — | 2 / 2 |
| true_39132300_validation_r3 | v1 | v2 v3 v4 | — | — | 2 / 2 |
| videosham_0069_r1 | v1 v4 | v2 v3 | — | — | 2 / 2 |
| videosham_1007_validation | v2 v3 | v1 v4 | — | — | 2 / 2 |
| videosham_4126_split_intervals_r7 | v2 v4 | v3 | v1 | — | 1 / 2 |
| **合计 31 题** | **15** | **15** | **1** | **0** | **14 / 16** |

改后每个文件仍同时有 FINISH 和 INVESTIGATE，所以没有新增题目。

题面没有“依据已取得的材料”这类捷径措辞；检索全部 task，只有 4124903 v2 的“能否据此”，它指的是具体的平台信息，可以接受。上一轮的主要问题已经消除。这一批的问题集中在另外三类：
1. 标签依赖 GIA 看不到的系统知识；
2. 所需证据不在离线视图里；
3. 题面有预设、缺先行词，或把调查后才知道的事实写进题面。

### 最重要的问题

1. **有两道 FINISH 题依赖 GIA 看不到的能力限制。**
   - **4124903_r2 v3**：“取不到 YouTube 视频帧”来自代码（`search/openers/video.py` 的 `skip_download=True`）。状态里只有结果描述，没有能力声明。而且现在就有能改变结论的补查：封面只和 A 的 2 帧比过。改为 INVESTIGATE 的窄题。
   - **20205 v1**：同类问题。“文件里有没有拍摄元数据”这一问，GIA 看不到 inspect_operations，规则 6 的措辞读起来反而像还有检查可做。改成问“已有属性记录能说明什么”。
   - 对照：0069 v1、1007 v3、20361105 v2 的状态里都有 status_reason 明写“现有动作不支持试听、ASR”，符合上一轮 1028 v3 的标准，保留 FINISH。
2. **4126 v1 在离线流程中产不出 FINISH 样本。**
   - E9 的内容不在视图里：只有卡片，没有 content；supporting_records 和任何 Finding 都没有它的时间映射。GIA 必须先 READ_RECORD。
   - `export.offline_gate` 对记忆动作返回 held（`offline_memory_action`）。
   - rationale 还把预置材料 E9 说成“W1-1 取得”。store 中它的 discovered_by 是 `provided_saved_media`。
   - 处理：删除。
3. **题面缺陷与隐性污染。**
   - 39132300 v2、v3、v4 都写“这款面具”，新题里没有先行词。GIA 和 HI 都看不到原任务。
   - 20205 v3 的例子“手握扶手”，正是 r0 核验模型对 E10 的判读，等于把答案写进题面；“除头发外”则预设了被判污染的 F1 的说法。
   - 4126 v3 的例子“纹理重复、残影、边缘不连续”预设 A 经过删除编辑。
   - 17838107 v3 预设画面中有进食；0069 v2 预设对齐给出了对应；1007 v1 照搬 F10 的结论句。
   - 0069 v3 的 rationale 引用 F10 当作可靠依据。F10 在形式上干净，但它的核心说法“1.80、2.68 秒抬头”，r0 在同一 verdict 里对 F11 判为与 E42、E16 相反。

### 流程层面的提醒（不针对单题）

- `run_variant` 调用 `decide(..., history=())`，源请求的 memory_history 被丢弃，但 turn_index 和 remaining_turns 仍沿用源值。受影响的有两处：
  - 17838107 的 E33 第[68]段（FMCSA 15–30 分钟）；
  - 20361105 的 E24 在 08:48 附近的字幕（“threat pregnant”）。

  这两处在离线视图中都看不到原文，只能经 F6、F11 转述。答案本身仍然成立，但 GIA 如果按规则“先核对原证据”去读原文，就会被记为 held。建议统计 D run 中 held 的比例。如果比例高，就给 FINISH 题优先选“关键原文已在 supporting_records 里”的状态。
- `_sources/` 下的状态摘要是起草者整理的，审查时一律回到 request.json 核对。4126 的 original_materials 在摘要里容易被忽略，E9 就在这里。

---

## 各题

### true_4124903_r2 / request_0037

**源状态核对**
- W1-1 为 COMPLETED（F1–F3），W1-2 为 COMPLETED（F4–F6），这两个在 history_index 里。
- W2-1 为 INCOMPLETE（F7–F10），在 latest_reports 里。
- runtime_constraints：round 3，remaining_turns=2，can_create_round=true。

**污染**
- r0：F3（minor）、F5（wrong_citation）、F7、F8（minor）。
- units 在 recalibrate 后全部为 clean。

**store 记录**
- W2-1 对两个 YouTube URL 执行了 `output_type=video` 的 open，只得到元数据和封面（E24–E27）。
- 对 A 只取了 10 帧。

**各题结论**
- **v1 通过。** 编号和说法都属实。E21 原文“sin relación con el presunto ataque terrorista”；F5 已排除；E15 可作旁证。
- **v2 通过。** E24 在 supporting_records 中可见。第二问由“上传日期≠拍摄日期”决定。
- **v3 修改，改判 INVESTIGATE。** 理由见第 6 节第 1 问。改后的题问的是：A 中有没有与封面图对应的画面。
- **v4 修改（只改 rationale）。** rationale 建议读 Dailymotion 页 E12、E13、E14、E16，但这几页只抽到导航文字（“Skip to player / Watch fullscreen / Font”），E14 只多一行标题。“E24 的频道字段”是“5,555,554 vues”，这是播放量，不是上传者。INVESTIGATE 仍然成立，可以通过检索当日的转发和上传取得证据。

### true_17838107_r1 / request_0027

**源状态核对**
- W1-1 为 COMPLETED（F1–F3），W1-2 为 COMPLETED（F4–F6）。
- remaining_turns=5。

**污染**
- r0：F2（wrong_citation，E20 中手持的是检测仪，不是面包）。
- units 全部为 clean。

**各题结论**
- **v1 修改（只改 rationale）。** “执法检测通常用什么办法”：FMCSA（E33）是美国交通部对商业驾驶员的检测规程，不是警方执法；西澳（E29）是转述。直接回答这一问的是 E27 第[7]段：“law enforcement officers are required, in most jurisdiction, to repeat any tests … after a 15- to 20-minute waiting period”，原 rationale 没有引。E33 第[68]段在可见的前 6000 字之外。
- **v2 通过。**
- **v3 修改。** 题面预设画面里有进食，改为“能否看到进食”。rationale 要补上 E27 第[4]、[8]段：Snopes 称视频里咬一口后直接复测，等待约 1 秒。这是现成的外部描述，教师可能据此直接 FINISH，必须说明报道不能代替画面核验，而且 A 是否就是 Pilbara 那段视频没有确认。
- **v4 通过。** E32 原文只有“Europe PMC requires Javascript…”。

### true_20205316753904_validation / request_0005

**源状态核对**
- 只有 W1-1（COMPLETED，F1）。
- store 中 search_executions 为 0；inspect 只有 get_frames（count=8）和一次 crop。

**污染**
- F1 在 r0 与 units 都是 tainted。

**各题结论**
- **v1 修改。** 理由见总览第 1 条。改后问的是：E2 这份媒体属性记录能说明什么。E2 的标签本身就写着“本记录未核验日期、地点或作者等来源元数据，不能据此判断这些信息不存在”。
- **v2 通过。**
- **v3 修改。** 去掉“除头发外”和“手握扶手”等例子。rationale 补一句：GIA 能直接看到 E3–E8、E10、E11 这些附图，但没有干净的 Finding 描述支撑点，细部需要裁剪。

### true_20361105_visibility_validation / request_0050

**源状态核对**
- W1-1 为 INCOMPLETE（F1–F3），W1-2 为 INCOMPLETE（F4–F6），在 history_index 里。
- W2-1 为 INCOMPLETE（F7–F9），W2-2 为 INCOMPLETE（F10、F11）。

**污染**
- r0：F5（把 threaten 归给 E11，另把“We're gonna”写成“We're going to”）。
- 其余 Finding 都干净。

**各题结论**
- **v1 通过。**
- **v2 修改（只改 rationale），维持 FINISH。** 理由见第 6 节第 2 问。
- **v3 通过。** 检索 store 中各报道全文，没有 Wilson 事后回应的内容。
- **v4 通过。** 补充一点：GIA 能直接看到 E3–E8，但小字需要 OCR，帧间隔约 5 秒，INVESTIGATE 成立。

### true_39132300_validation_r3 / request_0008

**源状态核对**
- 只有 W1-1（COMPLETED，F1）。
- store 中 inspect_executions 为空。
- E2 全文、E5 前 6000 字在 supporting_records 中。

**污染**
- F1 两处都干净。

**已入库但没被 Finding 使用的网页里有关键内容**
- E3（It's Fun Tyme!）：#9206 系列 1992 年上市，头卡印 Fantastic Faces；上市时叫 'Weeping' ghost；电影后先叫 Scream Stalker，2000 年才正式用 Ghost Face。
- E4：Fun World 1992 年的目录页（只有图注）；1996 年注册并命名 GhostFace；Alterian 的诉讼。
- E7：Scream 3 时注册。

**各题结论**
- **v1 通过。**
- **v2 修改。** “这款面具”没有先行词。
- **v3 修改。** 三处要改：
  1. 补上先行词；
  2. 原题“找出能直接证明…1996 年以前已经上市销售的当年资料”把待证命题当作已知，改成问“有没有”；
  3. rationale 写明 E3、E4 已有 1992 年上市和目录的转述，但仍是二手资料，目录页图片也没有取到；本 run 可以用 ddgs 检索，包括图片输出。
- **v4 修改。** 原题问“上映之前还是之后”，但来源之间对定名年份有冲突（1996 / 1999 / 2000），标签不稳。改成问“上映前是否以 Ghostface 出售、当时叫什么、这个名字随什么出现”。在这几点上所有来源一致，FINISH 稳定；rationale 写明年份分歧。

### videosham_0069_r1 / request_0045

**源状态核对**
- history_index：W1-1 为 INCOMPLETE（F1–F4），W1-2 为 INCOMPLETE（F5、F6）。
- latest_reports：W2-1 为 INCOMPLETE（F7–F9），W2-2 为 INCOMPLETE（F10–F12）。
- E5 全文在 supporting_records 中。

**污染**
- r0：F1、F2、F3（minor），F5，F7、F8、F9，F11。
- units 中 F2、F6、F7 为 clean。

**各题结论**
- **v1 通过。** W1-2 的 status_reason 写明“现有动作也不支持独立播放、提取或转写”。store 中 W1-2 做过 mode=audio 的对齐，结果为 failed。
- **v2 修改。** 题面“给出了哪些时间对应”，而 E5 的 matches 为 []。改为“是否给出了时间对应”。
- **v3 修改（只改 rationale）。** 不能用 F10 作 B 侧的依据（见总览第 3 条）。
- **v4 通过。**

### videosham_1007_validation / request_0045

**源状态核对**
- history_index：W1-1（F1–F3）、W1-2（F4、F5）。
- latest_reports：W2-1（F6–F8）、W2-2（F9–F12）。

**污染**
- r0：F1、F3、F5、F7、F8、F11。
- units 中 F1–F11 全部为 clean。

**各题结论**
- **v1 修改。** 题面照搬 F10 的“增加了…语境”。rationale 补上限制：B 的尾段还有 E41–E44 这几帧没有被描述。
- **v2 通过。**
- **v3 通过。** W1-2 的 status_reason 写明“所列动作也不提供试听或声音内容识别功能”。
- **v4 修改（只改 rationale）。** B 的 0.000、0.601、1.168 秒帧（E34、E35、E18）和 A 的 0.000、0.720 秒帧（E14、E15）已经在 current_materials 里，只是没有任何 Finding 判读过。INVESTIGATE 仍然成立。

### videosham_4126_split_intervals_r7 / request_0062

**源状态核对**
- history_index：W1-1（F1–F3）、W1-2（F4–F7）。
- latest_reports：W2-1（F8、F9）、W2-2（F10–F12）、W2-3（F13、F14），全部 INCOMPLETE。
- original_materials 为 E1–E17，都是预置材料（discovered_by=provided_saved_media），只给卡片。

**污染**
- r0：F1–F14 全部污染。
- units 中只有 F4、F12 为 clean。
- r0 还多处指出，报告把穿绿衣的滑板者当成了“黑衣人物”。

**各题结论**
- **v1 删除。** 理由见总览第 2 条。
- **v2 通过。** store 中唯一的对齐记录 E9 是 visual 模式。
- **v3 修改。** 去掉修补痕迹类的例子，也去掉与场景不符的“建筑、天空”。
- **v4 通过。** E3、E4 的标签和 content 都可见，不需要读取记录就能作答。

**起草者提到“4126 v1、v4 几乎只能依据原始工具记录”，分开看：**
- v4 依据的 E3、E4 在视图里可见，可以保留；
- v1 依据的 E9 在视图里没有内容，这是问题所在。

仅凭原始工具记录作答本身没有问题，前提是这份记录的内容在视图里。

---

## 6. 起草者最没把握的两题

### 6.1 4124903_r2 v3：状态里能不能看出取不到远程视频帧？

**不能明确看出。** 状态里相关的文字只有三处：

| 位置 | 原文 |
|---|---|
| W2-1 status_reason | “候选公开视频仅取得封面和元数据，尚未建立与材料 A 的视频帧及片段时序对应” |
| F8 限制 | “未取得候选视频的可比较帧” |
| W1-1 F3 限制 | “仅查看了候选视频的元数据和封面，未取得其完整视频” |

这三处都是对这一次结果的描述，不是“现有动作不能下载视频”这样的能力声明。GIA system prompt 还写着“Workspace 能实际检索网页和处理图片、视频、音频、PDF”。“取不到”只能从代码里知道：store 显示 W2-1 以 `output_type=video` 打开 URL 后只得到 E24–E27，而 `openers/video.py` 设了 `skip_download=True`。

更关键的是，确实存在能改变结论的补查：
- F8 只把封面 E19、E27 与 A 的 0 秒和 2.1 秒两帧比过，结论是“相容”；
- A 的 25.1 秒里只看了 10 个时刻；
- 烟雾形态随时间变化，如果 A 中某帧与封面逐处对应，就能把“相容”升级为“A 至少包含封面画面”。

所以原题标 FINISH 站不住。原题“是否为同一段画面”即使补查也只能部分回答，标签不稳，因此改成针对封面的窄题，标 INVESTIGATE。

如果想保留“不追取不到的证据”这类 FINISH 样本，应另选状态里有明确能力句的源。

### 6.2 20361105 v2：frighten 还是 threaten？

**现有证据已经足以作答，应 FINISH。** 答案是：原话用词无法确定。

**各方记录**

| 版本 | 来源 |
|---|---|
| frighten | E11 第[7]段；E16（Snopes）第[11]段转引；E19 第[5]段；E22 第[5]段；E25（Fox）第[4]段；E18 第[4]段 |
| threaten | E20（Free Beacon）第[2]段；E16 第[4]段 |
| 无法定词 | E24（CBS Miami 自动字幕）在 08:48 附近识别为“Congress threat pregnant members of Congress” |

**为什么再查也定不了**
- 原声在状态中明确取不到：W2-1 的 status_reason 写明“当前可用动作不支持音轨播放或 ASR，官方 Facebook 视频获取失败”；W2-2 也写明未取得原声或可靠逐字转录。两轮补查都失败了。
- 再检索只能拿到更多二手转写。这些报道多为相互转引，次数多少不能代替原声。
- 即使给定视频 A 带有字幕条，那也是上传者做的转写，同样不能定词。

**需要改的地方**
- FINISH 标签和题面都正确，只改 rationale。
- rationale 要写进其他几篇报道的记录，说明为什么“多数写 frighten”不能用来定词。
- 离线视图看不到 E24 的这一段，只能经 F11 的限制转述。

**不必改的地方**

同文件的 v4（画面文字，INVESTIGATE）与 v2 不冲突：v4 是为了找台标、查来源，不是为了定词。
