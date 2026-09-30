# 审查结果：VideoSham label_expected 草案 + 来源 D 离线换题（2026-09-28）

审查人：独立审稿（非草稿作者）。结论都有文件证据支撑，路径相对 `experiments/sft_data/`。

## 总览

| 部分 | 通过 | 修改 | 删除 | 新增 |
|---|---|---|---|---|
| A. label_expected（5 条） | 0 | 3（0069、1028、2010） | 2（1007、4126） | — |
| B. 离线换题（14 题） | 3（2417 v3/v4、4124903 v4） | 11 | 0 | 1（1028 v4） |

修改后离线题共 15 道，FINISH 9 道、INVESTIGATE 6 道。每个文件都同时有两类。

### 最重要的问题

1. **FINISH 题带有捷径措辞。** 8 道 FINISH 中有 6 道在题面写了“依据已取得的材料”或“仅依据当前（已取得的）材料”，6 道 INVESTIGATE 一道都没有。4124903 v2 和 2010 v2 的 rationale 直接写“题目限定当前材料，不需要再取证”，也就是说决定由措辞给定，不由证据给定，学生只要学会看措辞就能答对。修改稿已全部去掉这类措辞：能去的直接删，2010 v2 改问“现有对齐结果能否建立对应”。
2. **1028 v3 的 INVESTIGATE 无法兑现。** `runs/videosham_1028_r2/run_config.json` 的 inspect_operations 没有 asr 和 audio_qa，W2-2 的 status_reason 也写明“现有动作不提供试听或ASR”，再派 HI 拿不到言语证据，所以改为 FINISH（回答无法核验）。4124903 v3 的 rationale 建议“取 YouTube 视频做帧对齐”，但 `search/openers/video.py` 用的是 `skip_download=True`，系统不下载视频文件；这题保留 INVESTIGATE，rationale 改成用封面 E17 比对并检索更早的上传。
3. **rationale 引用编号错误 5 处，另有 2 处依赖污染 Finding。** 编号错误：2417 v2 写 W1-2 F2，应为 W1-2 F8；4124903 v1 写 W1-2 F1，应为 W1-2 F5；4124903 v2 写 W1-2 F3，应为 W1-2 F7；2010 v3 写 W3-1 F3，该工作区不存在，应为 W2-1 F9；2010 v1 没写编号。依赖污染 Finding：1028 v2 依赖 W1-2 F3，2010 v1 若按 history_index 会用到 W1-1 F1，两条都被判 tainted。
4. **label 草案：2 条没有区分力，2 条漏判真实答案。** 4126 的“黑衣”就写在案卷 task_text 里，错误答案和方向相反的探针都会命中。1007 的“删减”只在真实答案的否定句里命中，是假覆盖。0069 和 2010 的草案词表都没命中真实答案，因为答案写的是“画面重复出现”和“绘制的…道路/插画”。另外，2010 的“红色汽车”A、B 两侧都有，错误答案也会命中。

污染状态的判定依据：`verdicts_units_judged.jsonl` 中该 request 的 answer 单元的 `extra.inherits_from`（recalibrate 之后的状态），并与 `verdicts_r0_judged.jsonl` 的 `tainted_findings` 对照。

---

## 任务 A：VideoSham label_expected

### 实测方法

videosham run 的 `requests/*/` 下没有 response.json（`runs/` 里只有 1028_r2、2010_r1 两个 run，各只有一个 request.json）。GIA 的 FINISH 答案改从两处取：① `verdicts_units_judged.jsonl` 中 5 个 videosham answer 单元的 `extra.anchored.sentences`，拼接后即为答案全文；② 离线 D 导出 `export/r0_12runs-…/gia_*.jsonl` 里 4 条 videosham FINISH（题目已换，仅作参考）。另外按审核要求补了否定句和错误答案探针。匹配规则与 `label_expected_impl.py.txt` 一致：`f.lower() in answer.lower()`。

| 案例 | 样本 | 期望 | 草案命中 | 建议命中 |
|---|---|---|---|---|
| videosham:0069 | 真实run答案 videosham_0069_r1/request_0060 | — | 无 | 重复出现、画面重复 |
| videosham:0069 | 探针-否定 | 不应命中 | 慢放、重复帧 | 重复帧、慢放 |
| videosham:0069 | 探针-正确(中文口语) | 应命中 | 无 | 重复出现、画面重复 |
| videosham:0069 | 探针-正确(英文) | 应命中 | slow | duplicat、slow |
| videosham:0069 | 探针-错误答案 | 不应命中 | 无 | 无 |
| videosham:1007 | 真实run答案 videosham_1007_validation/request_0072 | — | 删减 | （删除） |
| videosham:1007 | 探针-否定(GIA常见限制句) | 不应命中 | 删减 | （删除） |
| videosham:1007 | 探针-错误答案 | 不应命中 | 缩短、剪掉 | （删除） |
| videosham:1007 | 探针-正确 | 应命中 | 丢帧 | （删除） |
| videosham:1028 | 真实run答案 videosham_1028_r2/request_0021 | — | lorem ipsum、Lorem ipsum、Lorem Ipsum | lorem ipsum、lorem、占位文字 |
| videosham:1028 | 离线D答案 videosham_1028_r2__D_v1（换题，仅作参考） | — | lorem ipsum、Lorem ipsum、Lorem Ipsum | lorem ipsum、lorem、占位文字 |
| videosham:1028 | 离线D答案 videosham_1028_r2__D_v2（换题，仅作参考） | — | lorem ipsum、Lorem ipsum、Lorem Ipsum | lorem ipsum、lorem、占位文字 |
| videosham:1028 | 探针-错误答案(看错文字) | 不应命中 | 无 | 无 |
| videosham:1028 | 探针-占位文字 | 应命中 | 无 | 占位文字 |
| videosham:1028 | 探针-OCR变体 | 应命中 | 无 | lorem |
| videosham:2010 | 真实run答案 videosham_2010_r1/request_0019 | — | 无 | 绘制的、插画 |
| videosham:2010 | 离线D答案 videosham_2010_r1__D_v1（换题，仅作参考） | — | 卡通 | 卡通、插画 |
| videosham:2010 | 离线D答案 videosham_2010_r1__D_v2（换题，仅作参考） | — | 无 | 无 |
| videosham:2010 | 探针-错误答案 | 不应命中 | 红色汽车 | 无 |
| videosham:2010 | 探针-否定 | 不应命中 | 卡通道路、卡通 | 卡通、插画 |
| videosham:2010 | 探针-正确(无“卡通”) | 应命中 | 无 | 绘制的 |
| videosham:4126 | 真实run答案 videosham_4126_split_intervals_r7/request_0107 | — | 黑衣、黑衣人 | （删除） |
| videosham:4126 | 探针-错误答案 | 不应命中 | 黑衣、黑衣人 | （删除） |
| videosham:4126 | 探针-方向相反 | 不应命中 | 黑衣、黑衣人 | （删除） |

真实答案本身不一定正确：0069 答案把重复定位在 A 的 2.76–3.20 秒，与官方的 00:06–00:10 对不上，但给定 B 只有 4.04 秒，官方时间轴与片段不一定一致。表中只用它们检验措辞覆盖，不当作真值。

### videosham:0069：修改

过窄：真实答案（verdicts_units_judged.jsonl 中 videosham_0069_r1/request_0060 的 answer 单元）写的是“局部画面重复”“相同低头画面在A中相隔0.36秒重复出现”，草案 6 个词一个都没命中，会把基本答对的答案记成 omitted；补“重复出现/画面重复/复制帧/duplicat”后命中。不加“变速/删减/定格”，因为 GIA 的限制句里常出现这些词的否定用法（同一答案 A8“不能据此认定连续定格…整段固定倍速”、A18“无法确定完整删减、重排或变速方式”）。剩余风险：否定句“未发现重复帧/慢放”仍会命中，这一点子串匹配无法消除；slow 可能命中英文里描述正常动作的 slowly。另外，草案 official 字段把 Manipulation 缩写成“Frame duplication/Slowing down”，原文是“Frame duplication/Reversed Frames/Frame dropping/Speeding up/Slowing down”（这是类目清单，不是本例的具体操作），入库时应照抄原文。官方区间 00:06–00:10 与给定片段对不上（答案 A13：A 长 10.20 秒、B 长 4.04 秒），位置交给 REQ1 判，不写进 label。

### videosham:1007：删除

子串匹配表达不了官方描述。官方描述是“A 丢掉 0:00–0:15 的帧，压成约 10 秒”，这是“B 中有、A 中缺失”的关系判断，而且案卷 REQ4 要求不预设制作先后，谨慎的正确答案往往不会直接写“删帧”。实测：真实答案（videosham_1007_validation/request_0072）只在否定句 A7“不能据此判断…或把B这些采样时间之间的整段认定为精确删减区间”里命中“删减”，是假覆盖；错误答案探针“A 比 B 缩短了约7秒，因为 B 片尾多了一张宣传卡被剪掉”命中“缩短/剪掉”。“删减/变速”是 GIA 限制句里的常用词（0069 答案 A18、2010 答案 A12 都有），过宽；只留“删帧/丢帧”又会漏掉“1.32→1.36 秒直接跳到划船者”这类正确的描述性写法。按原则 4（宁可少写），删除，由 REQ1（“A 相对 B 是否存在丢帧/缩短”）和充分性层判。

### videosham:1028：修改

方向正确：lorem ipsum 是画面上可直接看到的文字，不预设编辑方向，也不出现在 task_text 里（cases/videosham_1028.json）。实测真实答案和两条离线 D 答案都命中，“看错文字”的错误答案探针不命中。修改两处：①实现用 f.lower() in answer.lower()，所以 aliases “Lorem ipsum”“Lorem Ipsum”是空操作，应删掉；②OCR 或转写容易出现变体（双空格、lorem-ipsum、lpsum 等），加宽松形式“lorem”（除此以外几乎不会出现），再加“占位文字”（真实答案 A4 用了这个说法）。已知不足：把文字归到错误的一侧（写成“B 显示 lorem ipsum”）子串无法区分，属低概率残余。

### videosham:2010：修改

草案既过宽又过窄。过宽：红色汽车在 A、B 中都有（官方说从 B 这段 0:38–0:41 抠出，B 为 unedited），说出“红色汽车”不代表发现了差异。实测错误答案探针“两段视频都是红色汽车在人群中驶过的实景，未发现差异”命中“红色汽车”。过窄：真实答案（videosham_2010_r1/request_0019）写的是“写实红色敞篷车与绘制的山、云、栅栏和道路组合”“插画道路”，草案 5 个词都没命中（“红色敞篷车”不含“红色汽车/红色车”）。改为只核能区分两者的“卡通/绘制/插画”背景：修改后真实答案命中“绘制的/插画”，离线 D_v1 答案命中“卡通”，错误答案探针不命中。残余风险：否定句“未见卡通道路”仍会命中。

### videosham:4126：删除

标签没有区分力。案卷 task_text 本身就写着“黑衣人物在 B 可见而在 A 对应位置不可见”（cases/videosham_4126.json），任何复述题面的答案都会命中“黑衣”。实测真实答案、错误答案探针“黑衣人物在 A 与 B 中均可见，未发现差异区间”、方向相反探针三条全部命中。四个 alias 里有三个（穿黑衣、黑衣男子、黑衣人）本身就包含“黑衣”，是冗余的。官方要求的是“在 A 中缺失及其区间”，子串匹配表达不了，交给 REQ1/REQ2 和 interval 类检查判。

### 建议的最终 JSON（与草案同格式；另存为 `review/videosham_label_proposals.reviewed.json`）

```json
{
 "_说明": "审查后建议（2026-09-28）。null 表示建议删除该条 label_expected，由 requirements 与充分性层判。",
 "videosham:0069": {
  "official": "Duplicate frames to make it seem like the rider is going very slow (00:06 - 00:10)；Frame duplication/Reversed Frames/Frame dropping/Speeding up/Slowing down",
  "check": {
   "kind": "label_expected",
   "params": {
    "label": "重复帧",
    "aliases": [
     "帧重复",
     "重复出现",
     "画面重复",
     "复制帧",
     "慢放",
     "放慢",
     "慢动作",
     "duplicat",
     "slow"
    ]
   },
   "note": "官方：A 在 00:06–00:10 复制帧，使骑手显得很慢。只核“答案点出画面重复或放慢”；位置与方向由 REQ1/REQ4 判。否定句会误命中，属已知残余风险。"
  }
 },
 "videosham:1007": {
  "official": "Drop frames from 0:00 to 0:15 to make it into a 10 second video；Frame duplication/Reversed Frames/Frame dropping/Speeding up/Slowing down",
  "check": null
 },
 "videosham:1028": {
  "official": "0:00 to 0:06 change the text in the blue box to lorem ipsum；Text Replaced/Text Added",
  "check": {
   "kind": "label_expected",
   "params": {
    "label": "lorem ipsum",
    "aliases": [
     "lorem",
     "占位文字"
    ]
   },
   "note": "官方：A 在 0:00–0:06 把蓝框文字换成 lorem ipsum。匹配不区分大小写，大小写变体不必列。"
  }
 },
 "videosham:2010": {
  "official": "This is the same video as 2009, cut out the red car from 0:38-0:41, and put it on a cartoon road (you can draw something simple yourself)；Adding an entity/ Adding a subject",
  "check": {
   "kind": "label_expected",
   "params": {
    "label": "卡通",
    "aliases": [
     "绘制的",
     "插画",
     "手绘",
     "动画风格",
     "cartoon"
    ]
   },
   "note": "官方：A 在 0:38–0:41 把红色汽车抠出，放到卡通道路上。只核“答案点出 A 的背景是绘制/卡通的”；红色汽车两侧都有，不作为标签。"
  }
 },
 "videosham:4126": {
  "official": "Remove the guy in black from the frame 00: 01-00: 07；Removing an entity/Removing a subject",
  "check": null
 }
}
```

---

## 任务 B：来源 D 离线换题

### 各源状态核对摘要

- **true_24171541709824_r1 / request_0022**：latest_reports 有 W1-1（COMPLETED，F1–F6）和 W1-2（COMPLETED，F7–F8）。F2 为 tainted，其余 clean。
- **true_4124903_r1 / request_0022**：W1-1（COMPLETED，F1–F4），W1-2（COMPLETED，F5–F7）。r0 中 F3、F5、F7 为 tainted；recalibrate 后 inherits 全部为 clean。run_config 有 ddgs 搜索，inspect 为 get_frames、get_pages、crop、ocr、align_media。
- **videosham_1028_r2 / request_0021**：history_index 有 W1-1（F1）、W1-2（F2–F4）；latest_reports 有 W2-1（COMPLETED，F5、F6）和 W2-2（**INCOMPLETE**，F7、F8）。clean：F1、F2、F4、F7、F8；tainted：F3、F5、F6。inspect 没有 asr/audio_qa。
- **videosham_2010_r1 / request_0019**：history_index 有 W1-1（F1–F3）、W1-2（F4–F6）；latest_reports 有 W2-1（COMPLETED，F7–F9）。**没有 W3-1。** tainted：F1、F2、F8；clean：F3–F7、F9。F4 的限制写明“当前接口未提供可直接听辨的音频或转写”。

### 比例

草案 8 FINISH / 6 INVESTIGATE，每个文件都两类兼有，比例本身合理。真正的问题在措辞与决定的相关性（见总览第 1 条）。修改后为 9 / 6（1028 v3 改为 FINISH，新增 v4 INVESTIGATE）。另外，1028 v1↔v4、2010 v2↔v3、2417 v1/v2↔v3/v4 形成了同一状态、同一话题下的对照，符合 SFT 方法文档 §4 的要求：换的总任务必须让正确决策真的不同。

### true_24171541709824_r1

#### v1：修改（expected_decision = FINISH）

- 理由：rationale 属实：W1-1 为 COMPLETED，F5 引 E9，E9 第[11]段原文写 2003 年弗州官员（含州长 Mark Warner）要求归还，Gov. Tim Pawlenty 回应“They're not getting it… rightfully ours”；units 核验 inherits 中 F5 为 clean。FINISH 正确。问题在题面的“依据已取得的材料”：这类限定语出现在 8 道 FINISH 中的 6 道，6 道 INVESTIGATE 一道都没有，构成捷径线索。去掉后 FINISH 仍然成立。rationale 还应补上限制说明。
- task：2003年弗吉尼亚州要求明尼苏达归还第28步兵团战旗时，明尼苏达州州长作何回应？给出出处。
- rationale：W1-1（COMPLETED）F5（干净）引 E9（2017 年 Twin Cities《先锋报》回顾）：2003 年弗州官员（含州长 Mark Warner）要求归还，明州州长 Tim Pawlenty 拒绝，称战旗理应属于明尼苏达。答案须带上 F5 的限制：依据为事后新闻回顾，未取得 2003 年的双方函件或同期官方记录。现有材料足以回答，不需要再查。

#### v2：修改（expected_decision = FINISH）

- 理由：rationale 引错了 Finding。W1-2 只有 F7、F8；F2 属于 W1-1（1960/1961 年请求），而且被 r0 判为 tainted（《先锋报》一处 unsupported_assertion）。实际可用的是 W1-2 F8（E2、E5：馆藏、战争部登记后回到明尼苏达、MinnPost 2023 称永久馆藏）和 F7（E2、E3：1863-07-03 Marshall Sherman 缴获）。W1-2 为 COMPLETED，r0 判 positive。FINISH 正确，E2 原文有“currently part of the Minnesota Historical Society collections”。去掉“依据已取得的材料”这一线索；“第28步兵团”是调查后才确认的身份（F7），题面改用原声明里的“邦联战旗”，减少泄露。
- task：这面被争议的邦联战旗目前由哪个机构保管？它是怎样从战场缴获一路进入该机构馆藏的？给出出处。
- rationale：W1-2（COMPLETED）F7（E2、E3）：弗吉尼亚第28步兵团战旗，1863-07-03 葛底斯堡皮克特冲锋中由明尼苏达第1团列兵 Marshall Sherman 缴获。F8（E2、E5）：曾交美国战争部登记，19 世纪末回到明尼苏达，现为明尼苏达历史学会馆藏，2023 年 MinnPost 称其属于永久馆藏。限制取 F8：馆藏状态以所读资料的记载时间为限，未作实时保管确认。W1-1 F2 被判污染，不作依据。

#### v3：通过（expected_decision = INVESTIGATE）

- 理由：rationale 属实：W1-1 F4（E6、E9）写明决议“已由规则委员会提交参议院全院”（E6 原文“forwarded Monday to the full Senate by the Rules Committee”），限制写明“未取得决议原文和最终表决记录”。缺的是关键证据，且可以取得：case source_policy 为 unrestricted，run 有 ddgs 搜索，当前材料中已有 E11“2000 Uncodified Acts”、E13“LIS Help”等弗州立法站点线索，再派 HI 查 LIS 2000 年会期记录可行。题中的 2000 年决议是跟进式前提，不泄露答案（是否通过）。
- task：找到2000年弗吉尼亚州参议院那项请求明尼苏达归还战旗的决议的正式文本或议会表决记录，确认它最终是否获得通过。
- rationale：W1-1 F4 只说决议已由规则委员会提交参议院全院（E6），没有表决结果；需要查议会记录。

#### v4：通过（expected_decision = INVESTIGATE）

- 理由：属实：已有材料时间最晚的是 E5（2023 年 MinnPost）与 E9（2017），没有 2023 年之后的请求记录。source_policy 无 date_cutoff，可检索。题面不泄露答案。与 v1、v2 构成同一状态下的 FINISH↔INVESTIGATE 对照。
- task：核查2023年之后，弗吉尼亚州官方是否再次正式要求归还这面战旗，以及明尼苏达方面如何回应。
- rationale：现有报告最晚到 2017 年回顾报道与 2023 年 MinnPost 的馆藏说明，没有 2023 年之后的请求记录；需要新检索。

### true_4124903_r1

#### v1：修改（expected_decision = FINISH）

- 理由：rationale 引错了 Finding。W1-2 只有 F5、F6、F7，没有 F1；警方通报在 W1-2 F5（E24、E21），W1-1 F3 也有（E21）。W1-1 F1 是帧描述。内容本身属实：E24“Suite à un incident technique un camion avec des artifices est en feu pont d'Iéna”，E21“Aucun incendie à la #TourEiffel…Il est éteint”。FINISH 正确。去掉“依据已取得的材料”线索。另外提示“已扑灭”要引 E21，不能引 E24（原答案 units 核验记录了 E24 wrong_citation：“Intervention des pompiers”）。
- task：巴黎警方如何说明2016年7月14日埃菲尔铁塔附近的火情？给出出处。
- rationale：W1-2（COMPLETED）F5 引 E24（21:51 通报：技术事故致载烟花卡车在耶拿桥起火）与 E21（22:33 通报：铁塔没有起火，意外卡车火灾已扑灭）；W1-1 F3 也记载了同一通报（E21，Franceinfo E19）。F6 的 E22、E11 只是转述同一通报。限制：通报没有说明技术事故的具体机制。可直接回答。

#### v2：修改（expected_decision = FINISH）

- 理由：rationale 引错了 Finding。W1-2 F3 不存在（F3 在 W1-1，讲来源与语境）；讲帧的是 W1-2 F7（E3、E5、E7、E10）。W1-1 F1、F4 引用无误。题面“仅依据当前已取得的材料”把决定写死在题目里，rationale 自己也写了“题目限定当前材料，不需要再取证”，FINISH 因此不言自明。这里去掉限定后 FINISH 依然成立：已看 8 个采样帧（E3–E10），再多抽帧也改变不了“画面本身不能证明恐袭”。另外这道题与原任务的 REQ2 和 W1-1 子任务几乎同文，训练价值偏低，保留但不宜再加同类题。
- task：给定视频 A 的画面本身能否证明巴黎发生了恐怖袭击？说明画面能支持和不能支持的内容。
- rationale：W1-1 F1（E3、E4、E5、E7、E8、E9）与 W1-2 F7（E3、E5、E7、E10）：夜间亮灯的铁塔附近有大量烟雾，较低处可见明火，不能据此认定塔身着火；W1-1 F4：火焰和烟雾不足以确定恐袭性质。限制：离散采样帧，画面不能证明拍摄日期、燃烧物或起火原因。补抽帧不会改变结论，可直接 FINISH。

#### v3：修改（expected_decision = INVESTIGATE）

- 理由：expected_decision 正确，rationale 事实也对：W1-1 F2 的限制写明只是来源候选，尚未核验与 A 的画面对应，原拍摄者和最早发布者未知。但 rationale 建议的“取该视频做帧对齐”在本系统里做不到：src/foren_searcher/search/openers/video.py 用 yt-dlp 时设了 skip_download=True，只取元数据、封面和字幕；OpenSourceRequest.include 只允许 ('subtitles',)（search/contracts.py:124），拿不到视频文件，也就对不了帧。能再派 HI 取得的是：拿封面 E17 与 A 的帧裁剪比对、检索更早的上传或转发、比较时长（A 为 25.100 秒，E2；YouTube 为 25 秒，E16）。题面应允许部分核实。
- task：确定给定视频 A 的原始拍摄者与最早发布时间，并尽可能核实它与 YouTube 视频 cv_F3UkbdbI 是否为同一段画面；说明无法核实的部分。
- rationale：W1-1 F2 写明 cv_F3UkbdbI（E16，Rutnet，上传日期 20160714）只是来源候选，尚未核验与 A 的画面对应，原拍摄者和最早发布者未知。可以再派 HI：用封面 E17 与 A 的帧（E3–E10）做裁剪比对，检索更早的上传或转发来源。系统不下载 YouTube 视频文件，做不了整段帧对齐；时长接近（25.1 秒 vs 25 秒）只是线索。

#### v4：通过（expected_decision = INVESTIGATE）

- 理由：属实：W1-1 F3 限制写“没有在该通报中详细说明起火机制”，W1-2 F5 限制写“警方简短通报没有说明技术事故的具体机制”。缺的是关键内容，可以检索（unrestricted，ddgs），当前材料中还有 E12–E15、E23 等未被 Finding 使用的报道可以读取。离线 D 教师在这题给了 FINISH，把“技术事故”当成了原因，说明这是一道有价值的难例。
- task：查明2016年7月14日耶拿桥烟花卡车起火的具体技术原因，并给出官方调查或权威报道依据。
- rationale：W1-1 F3 与 W1-2 F5 的限制都写明警方通报没有说明起火机制；现有报告回答不了具体原因，需要新的检索。

### videosham_1028_r2

#### v1：修改（expected_decision = FINISH）

- 理由：rationale 属实：history_index 中 W1-2 F4 为四个时刻（E5/E7、E10/E12、E11/E13、E6/E8，即 0、1.6、3.2、4.8 秒）的标题对照，W1-1 F1 为 1.6、3.2 秒。这两条在 r0 分别被判 minor 和 wrong_citation，recalibrate 后 units inherits 为 clean（见 cases_draft/videosham_1028.review.md 注 2、3）。FINISH 正确。题面的问题：“比较…左下角标题区…说明这一差异”，既指明了差异位置，又预设差异存在，有轻度泄露；而且题面几乎照抄 F4 的结论。改为不预设差异的问法。
- task：视频 A 与 B 画面上的标题文字是否相同？如不同，分别写了什么，对观众理解画面主题有何影响？
- rationale：W1-2 F4（干净）：四个已检查时刻 A 为单行“lorem ipsum”，B 为两行“A conversation / about CRPS”，B 提供了 CRPS 谈话这一主题提示；W1-1 F1 在 1.6、3.2 秒给出同样结论。限制取 F4、F1：标题不证明实际谈话内容或诊断，不能确定修改方向，未查看的区间不当作一致。可直接 FINISH。

#### v2：修改（expected_decision = FINISH）

- 理由：rationale 部分依据是污染 Finding：“视觉对齐同秒对应的记录”是 W1-2 F3，r0 与 units inherits 都判 tainted（“0–4.84秒”“实际成对帧与该候选一致”unsupported_assertion）。单靠 F2（干净，四组成对帧）就足以支持 FINISH。题面“（人物、拐杖、背景建筑）”是 F2 的调查结果，属于泄露；“仅依据当前材料”是捷径线索。两处都去掉后 FINISH 仍然成立。
- task：视频 A 与 B 的主体画面是否来自同一镜头？说明依据的时刻、可对应的画面要素和这一判断的范围。
- rationale：W1-2 F2（干净）：0、1.6、3.2、4.8 秒四组成对帧中，人物姿态、拐杖位置、左侧木建筑和树木小路相互对应（E5/E7、E10/E12、E11/E13、E6/E8）；W1-1 F1 旁证。限制：只限已观察的帧，不证明整段相同，也不确认 B 的来源身份。W1-2 F3（对齐同秒对应）被判污染，不作依据。

#### v3：修改（expected_decision = FINISH）

- 理由：expected_decision 错误。缺的证据再派 HI 也取不到：runs/videosham_1028_r2/run_config.json 的 inspect_operations 只有 get_frames、get_pages、crop、ocr、align_media，没有 asr 和 audio_qa；W2-2 的 status_reason 明写“现有动作不提供试听或ASR”，GIA 状态里可以看到这一句。按“INVESTIGATE 须能再派 HI 取得”的标准，应改为 FINISH，答“无法核验”并说明原因。这类样本教 GIA 不追取不到的证据，本身有价值。rationale 对状态的描述（W2-2 INCOMPLETE，F7 只有对齐候选）是准确的。前提：离线环境必须沿用源 run 的 inspect 配置；如果离线环境开了 asr，这题就应当是 INVESTIGATE。为了保留本文件的 INVESTIGATE 样本，另外新增 v4。
- task：确认视频 A、B 的音轨中是否有可辨认的言语；如果有，言语内容是否与 B 的标题“A conversation about CRPS”相对应。
- rationale：W2-2 为 INCOMPLETE：F7（E20）只有音频对齐候选（0–4.84 秒，score 不代表内容一致）；F8（E12、E13）说明标题不能证明音轨里有相关谈话。status_reason 写明现有动作不提供试听或 ASR，本环境再派 HI 也取不到言语证据。应直接 FINISH：言语是否存在和内容均无法核验，说明原因，不能根据标题推定。

#### v4：新增（expected_decision = INVESTIGATE）

- 理由：替代原 v3 的 INVESTIGATE 名额，与 v1 构成同一话题下的 FINISH↔INVESTIGATE 对照。缺口真实存在：W1-1 F1 的限制写“不能据此确定差异的完整起止时间或未查看区间是否始终如此”；visual_observation_summary 中 A（E1）的 unconfirmed_ranges_s 为 [0.04,1.6]、[1.64,3.2]、[3.24,4.8]。get_frames 在 inspect_operations 里，可以取得。这也对应官方描述中的 0:00–0:06 区间。
- task：确定 A、B 标题文字差异在各自视频中的起止时间：差异是贯穿整段，还是只在部分时段出现？给出边界两侧的帧。
- rationale：现有标题对照只有 0、1.6、3.2、4.8 秒四个时刻（W1-2 F4、W1-1 F1），F1 限制明言不能确定完整起止时间；A 在 0.04–1.6、1.64–3.2、3.24–4.8 秒没有观察过。需要再派 HI 用 get_frames 加密抽帧，确认差异边界。

### videosham_2010_r1

#### v1：修改（expected_decision = FINISH）

- 理由：FINISH 正确，但 rationale 没有写出 Finding 编号，也没有区分污染状态。history_index 里 W1-1 F1 被 r0 与 units 判为 tainted（“汽车经过人群”unsupported_assertion），可用的是 W1-2 F5（干净，E14、E17、E11、E13）。题面“这种差异如何改变场景表达”几乎照抄 F5 原句“可确认的背景差异改变了场景表达”，并预设差异存在；“背景环境”也直接点出了差异维度。改为中性问法。
- task：视频 A 与 B 中车辆所处的场景环境是否一致？如不一致，分别是什么样的环境，对画面所表达的场合有何影响？
- rationale：W1-2（COMPLETED）F5（干净）：A 在 0.48、2.88 秒为写实红色敞篷车配绘制的道路、围栏、山和云，B 在 1.00、2.96 秒为车辆和人群聚集的实景，场景表达从插画道路变为现场环境。限制取 F5：采样时刻不是已确认的一一对应帧，不能确认 B 是 A 的来源。W1-1 F1 被判污染（“汽车经过人群”），不作依据。

#### v2：修改（expected_decision = FINISH）

- 理由：rationale 属实：W1-2 F6（E5、E18）、W1-1 F3（E3、E4、E5）、W2-1 F9 都写明视觉和音频对齐均未返回候选，units inherits 三条都是 clean。问题出在题面：“仅依据当前材料”把 FINISH 写死了（rationale 自己也说“题目限定当前材料”），和 v3 的差别只剩这个限定语，模型学到的是看措辞而不是看证据。改为明确问“现有对齐结果”能否建立时间对应。这是一个关于已有测量的问题，FINISH 由证据决定，同时保留与 v3 的对照。
- task：现有的视觉和音频对齐结果能否建立视频 A 与 B 之间可靠的时间对应关系？说明原因以及这一结论的边界。
- rationale：W1-1 F3（E3、E4、E5）与 W1-2 F6（E5、E18）：视觉对齐（SIFT/FLANN）和音频对齐都没有返回候选，无法建立时间映射；W2-1 F9 同样说明。限制：对齐未命中不等于两者没有关联，也不能证明存在删减。问题针对已有测量结果，可以直接回答。

#### v3：修改（expected_decision = INVESTIGATE）

- 理由：rationale 引用的 Finding 不存在：state 里没有 W3-1，只有 W1-1、W1-2 和 latest_reports 中的 W2-1。“无法确认A中哪一时刻与B的2.96秒严格对应”出自 W2-1 F9（E5、E18、E17、E13）。题面“取自 B 的哪一个时间段”预设 A 由 B 加工而来，违背审核原则第 3 条（不预设候选 B 的来源身份），需要改写。INVESTIGATE 正确，而且可以取得：B 只在 0、0.48、1、1.48、2、2.48、2.96 秒采过样，HI 可以用 get_frames 在 2.5–3.0 秒一带加密抽帧，再用 crop 比对车身姿态和轮毂；本 run 的 HI 已用过 crop（E19、E23）。
- task：确定 A 中红色敞篷车的画面与 B 中哪一时刻的车辆画面相对应，给出两侧时间与依据；若无法对应，说明已排除的范围。
- rationale：W2-1 F9 写明无法确认 A 中哪一时刻与 B 的 2.96 秒对应，W1-1 F3、W1-2 F6 写明两种对齐都没有候选；B 只有 7 个稀疏采样。需要再派 HI 在 B 的 2.5–3.0 秒一带加密抽帧，裁剪比对车身角度、轮毂和乘员。W2-1 F8（视角变化）被判污染，不作依据。

---

## 附：其他观察（不影响结论）

- 离线 D 日志（`_来源D离线换题日志.txt`）：教师在 1028 v1、v2 和 2417 v2 给了 INVESTIGATE，在 4124903 v4 给了 FINISH，与期望不一致。前三题正是题面预设差异或带“现在”一类时间词的题；4124903 v4 则说明这是有效的难例。
- 1028 的原始 FINISH 被 units 核验判为 premature_finish（REQ2 音频无 Finding）。但 REQ2 的文字允许“无法核验的说明”，而本环境本来就没有 ASR。建议核验层对“环境不提供该操作”的情形不判 premature_finish，否则会与 1028 v3 改为 FINISH 的结论相冲突。