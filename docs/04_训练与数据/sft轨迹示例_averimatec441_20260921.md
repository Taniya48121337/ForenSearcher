# SFT 轨迹示例：averimatec 441（按 v2 方法逐行构造）

案例来源：`experiments/sft_data/runs/averimatec_441_q0_r1`，2026-09-12 由教师 gpt-6-astra 实跑，1 个工作区、6 次 HI 调用、1 份 COMPLETED 报告。选它是因为：一张图、一个问题、纯文本证据、六步就结束，能把 v2 的每个环节都走一遍而不被细节淹没。

**先说一个必须诚实交代的事实。** 这个 run 是 `mode: hi`、`gia_output_source: scripted_dispatcher`——GIA 不是模型，是脚本：第一轮把任务原文照抄派成一个工作区，第二轮输出一句占位答案"本次局部任务执行结束；结果保存在工作区报告中。"所以 HI 的 6 行是真实教师轨迹（来源 A），GIA 的 2 行在这个 run 里**不存在可用的教师输出**，下面 GIA 两行是按 v2 来源 B（同状态再生：拿这个 run 当时的全局视图快照，让教师重出一步）的形态写的，`origin: B`、`teacher_model` 标为待定。这正是 v2 说"GIA 数据必须靠 B/D 补"的具体样子。

全文结构：0 案卷 → 1 调查时间线 → 2 HI 六行 → 3 GIA 两行 → 4 verdicts → 5 manifest → 6 这个例子暴露的问题。user 字段按 `compact_8k` 档位的样式手写示意（实际由 `state_render.render_hi_payload / render_gia_payload` 现场生成，数据文件里只存 `run/workspace/turn` 定位，不存渲染文本）。

---

## 0. 案卷 `cases/averimatec_441_q0.json`

没有案卷，§2.4 缺口评判和 hindsight 都没法算，所以先写它。数据集原始标注只有问题和答案（"Hyderabad, India"），`requirements[]` 和 `required_evidence` 是按 v2 §7 的规矩从教师首次成功 run 经核验后的证据链补出来的（`derived_from_run` 标明来源，不是人拍脑袋）。

```json
{
  "case_id": "averimatec:441:q0",
  "dataset": "averimatec",
  "family_key": "averimatec:441",
  "split": "dev",
  "task": "完成以下局部调查问题，查找直接证据并说明适用范围：Where is the grave in the image located?\n背景声明：A photo shows that Pakistani parents locked their daughter's grave to avoid her corpse falling prey to necrophilia\n声明日期（任务元数据）：2023-04-26",
  "media": [{"ref": "I0", "kind": "image", "path": "images/averimatec_441_image_0.jpg", "width": 368, "height": 388}],
  "source_policy": {"exclude_domains": [], "allow_fact_checkers": true},
  "checks": [
    {"kind": "label_expected", "unit": "answer", "value": "Hyderabad, India", "match": "entity"},
    {"kind": "entity_in_evidence", "unit": "report", "entities": ["Hyderabad"], "any_cited": true},
    {"kind": "status_expected", "unit": "report", "value": "COMPLETED"}
  ],
  "requirements": [
    {"id": "REQ1", "must_cite": true,  "text": "给出坟墓所在的国家与城市（含具体墓地或地标）"},
    {"id": "REQ2", "must_cite": true,  "text": "说明定位依据（地理定位、实地核查或原发布者说明），并注明适用范围"},
    {"id": "REQ3", "must_cite": false, "text": "所附照片与来源中被核查的照片是否为同一张"}
  ],
  "required_evidence": [
    {"key": "altnews_locate", "kind": "page", "url_pattern": "altnews.in/*padlocked-grave*", "supports": ["REQ1", "REQ2"], "derived_from_run": "averimatec_441_q0_r1"},
    {"key": "afp_locate",     "kind": "page", "url_pattern": "factcheck.afp.com/doc.afp.com.33EA28V", "supports": ["REQ1", "REQ2"], "derived_from_run": "averimatec_441_q0_r1"}
  ],
  "required_evidence_rule": "any_one_of"
}
```

---

## 1. 调查时间线（trajectory.jsonl 33 个事件压成一张表）

| 事件 | 角色 | turn | 输出 | 观察 |
|---|---|---|---|---|
| 2–3 | GIA | r1/t1 | INVESTIGATE，1 个任务（脚本照抄） | 开 W1-1 |
| 5–8 | HI W1-1 | 1 | SEARCH text_image（以图搜图 + 关键词） | **ACCESS_RESTRICTED**：本部署不允许把调查图片发给搜索商，0 结果 |
| 9–12 | HI | 2 | SEARCH text（关键词） | 9 个候选 R1–R9，自动打开 5 页 → E2 (ETV Bharat)、E3 (AFP)、E4 (NDTV)、E5 (Logical Indian)、E6 (Alt News)；R5/R6/R7 打开失败 |
| 13–16 | HI | 3 | READ_RECORD E6 offset 0 | Alt News 第 1 页（[1]–[18]：传播链） |
| 17–20 | HI | 4 | READ_RECORD E6 offset 6000 | 第 2 页（[19]–[36]：**[29] 地址、[31] 委托居民到场、[35] 清真寺宣礼员证实**） |
| 21–24 | HI | 5 | READ_RECORD E3 offset 0 | AFP 全文（**[21] 街景定位、[22] 记者 5 月 2 日到访**） |
| 25–28 | HI | 6 | 报告 COMPLETED，F1 引 [E6, E3] | 入库 |
| 31–32 | GIA | r2/t1 | FINISH（脚本占位句） | 结束 |

有用证据集合 U（v2 §2.2）= 案卷 required_evidence {E6, E3} ∪ 被判 supported 的 F1 所引 {E6, E3} = **{E3, E6}**。E2、E4、E5 是同话题的其他核查报道，没被引用，不在 U 里，但也不是噪声。

---

## 2. HI 六行（`hi_accepted.jsonl`）

共同字段先写一次，每行只列变化的部分。

```
system:  specs/hi@0.1.0/student_prompt.md   （hash 见 metadata；运行时那 ~2K 常量说明按重构记录第四节的遗留项仍在 user 侧）
images:  ["runs/averimatec_441_q0_r1/requests/request_000N/actor/image_01.png"]   （六步都只附 E1 这一张，368×388）
```

### HI 行 1 —— turn 1，SEARCH text_image（撞上部署限制）

**user（compact_8k 示意）**
```
[任务 W1-1] 完成以下局部调查问题，查找直接证据并说明适用范围：Where is the grave in the image located?
背景声明：A photo shows that Pakistani parents locked their daughter's grave to avoid her corpse falling prey to necrophilia
声明日期：2023-04-26
[材料] C0 任务卡 | E1 image "材料 image_0" src=I0 by=HARNESS/ingest（像素已附）
[候选来源] 无
[证据视图] E1
[焦点] 无   [已看正文] 无
[历史] 无
[最新观察] 无
[约束] turn 1/12 · 直接视觉输入 · 图片上限 8
[可用动作] SEARCH(input_type: text|image|text_image|source, query, output_type) · INSPECT(targets[E], operation: get_frames|get_pages|crop|ocr|align_media, params) · QUERY_MEMORY(query, record_types?) · READ_RECORD(record_id, offset?)
```

**assistant（教师真实输出 + 拟加的 rationale）**
```json
{"rationale": "任务问的是 E1 中坟墓的位置，画面本身没有文字地标（绿色铁栅栏、乌尔都/阿拉伯文墓碑、红花），最直接的办法是拿 E1 以图搜图并附上声明关键词。",
 "action": "SEARCH",
 "arguments": {"input_type": "text_image", "query": {"evidence_id": "E1", "text": "Where is this grave with a green locked metal grille located? Viral April 2023 claim Pakistani parents locked daughter's grave necrophilia fact check"}, "output_type": "text"},
 "focus_evidence_ids": ["E1"]}
```

**metadata**
```json
{"role": "hi", "unit": "step", "origin": "A", "run": "averimatec_441_q0_r1", "workspace": "W1-1", "turn": 1,
 "case_id": "averimatec:441:q0", "dataset": "averimatec", "case_family": "averimatec:441",
 "teacher_model": "gpt-6-astra", "spec": "hi@0.1.0", "spec_hash": "sha256:…", "state_renderer": "compact_8k",
 "verifier": "verifier@0.2.0", "judge_model": "qwen/qwen3.8-27b",
 "verdict": "positive", "error_types": [], "env_errors": ["env_tool_failure"],
 "tainted": false, "clue_gain": 0, "sample_weight": 1.0,
 "input_tokens_est": 3900, "supervised_tokens_est": 140, "exported_at": "2026-09-21"}
```

**为什么入选**：程序检查——动作合法、E1 可见、非重复。工具随后报 ACCESS_RESTRICTED 是环境级错误（`env_tool_failure`），记在 metadata 里、不算 HI 的错（v2 §2.5）。clue_gain 0，权重 1。**但这一行是本例最值得讨论的一行**，见第 6 节。

### HI 行 2 —— turn 2，SEARCH text（拿到全部有用证据）

**user 变化**：`[焦点] E1 · [已看正文] E1 · [历史] t1 SEARCH text_image → ACCESS_RESTRICTED（0 结果） · [最新观察] SEARCH 错误：stage=upload code=ACCESS_RESTRICTED "this deployment does not allow sending investigation images to a search provider"`

**assistant**
```json
{"rationale": "以图搜图被部署禁止，改用纯文本：声明时间（2023 年 4 月）、画面特征（绿色栅栏、挂锁）和事实核查关键词组合搜索。",
 "action": "SEARCH",
 "arguments": {"input_type": "text", "query": "April 2023 fact check green grille locked grave Pakistani parents daughter necrophilia grave location", "output_type": "text"},
 "focus_evidence_ids": ["E1"]}
```

**metadata 变化**：`"turn": 2, "error_types": [], "clue_gain": 2, "sample_weight": 3.0`

**核验**：程序——合法、与 t1 不重复（input_type 不同、查询不同）、且 t1 失败后**没有原样重发**（不是 ignored_feedback，是改法重试）。hindsight——这一步注册了 E2–E6，其中 E3、E6 ∈ U，`clue_gain = 2`（acquired 分量）。权重 1+2=3，封顶。

### HI 行 3 —— turn 3，READ_RECORD E6 第 1 页

**user 变化**：候选来源 R1–R9 一行卡（含 page=ready(E2…E6) / failed）；证据视图 E1–E6 一行卡；`[最新观察] SEARCH 9 个候选，5 页已开：E2 ETV Bharat / E3 AFP / E4 NDTV / E5 Logical Indian / E6 Alt News；R5 dawn / R6 newschecker / R7 propakistani 打开失败`

**assistant**
```json
{"rationale": "五页都是核查报道；标题里唯一直接写出城市的是 E6（Alt News，'is from Hyderabad'）和 E3（AFP，'from India'），先读 E6 全文找具体地址和定位方法。",
 "action": "READ_RECORD", "arguments": {"record_id": "E6", "offset": 0}, "focus_evidence_ids": ["E1"]}
```

**metadata 变化**：`"turn": 3, "clue_gain": 1, "sample_weight": 2.0`

**核验**：E6 ∈ U，此步首次读取其正文（observed 分量 +1）。第 1 页 [1]–[18] 只讲传播链，锚定层的引语都不在这一页，所以 quote-page 分量 0。

### HI 行 4 —— turn 4，READ_RECORD E6 第 2 页

**user 变化**：`[已看正文] E1, E6 · [历史] … t3 READ_RECORD E6 (0→6000) · [最新观察] E6 第 1 页正文（按 8K 档截断至摘要 + "…[已截断；READ_RECORD offset=6000 可续读]"）`

**assistant**
```json
{"rationale": "第 1 页到 [18] 仍在讲媒体转载，地址和核查方法应在后文，续读。",
 "action": "READ_RECORD", "arguments": {"record_id": "E6", "offset": 6000}, "focus_evidence_ids": ["E1"]}
```

**metadata 变化**：`"turn": 4, "clue_gain": 1, "sample_weight": 2.0`

**核验**：程序——与 t3 不重复（offset 不同）。hindsight——锚定层为 F1 摘出的引语 [29]、[31]、[35] 全部落在这一页（quote-page 分量 +1）。这就是 v2 §2.2 给 READ_RECORD 续页设 quote-page 分量的原因：不加它，这一步会被算成 0 收益的"翻页"。

### HI 行 5 —— turn 5，READ_RECORD E3

**user 变化**：`[最新观察] E6 第 2 页：[29] 墓地在 Masjid E Salar Mulk 对面、Darab Jung Colony, Madannapet, Hyderabad；[31] Alt News 委托当地社工 Abdul Jaleel 到场拍照；[35] 宣礼员证实地址…`

**assistant**
```json
{"rationale": "Alt News 已给出地址和实地核对；再读 AFP（E3）看是否有独立的定位方法，两家独立来源互证后再写报告。",
 "action": "READ_RECORD", "arguments": {"record_id": "E3", "offset": 0}, "focus_evidence_ids": ["E1"]}
```

**metadata 变化**：`"turn": 5, "clue_gain": 2, "sample_weight": 3.0`

**核验**：E3 ∈ U 首次读取（+1），引语 [21]、[22] 在本页（+1）。

### HI 行 6 —— turn 6，工作区报告

**user 变化**：`[已看正文] E1, E6, E3 · [最新观察] E3 全文：[21] AFP 用 Google 街景从 Masjid-e-Salar Mulk 定位墓地；[22] AFP 记者 5 月 2 日到访并拍照；[26] 视频中宣礼员与社工 Abdul Jaleel 对话…`

**assistant（教师真实输出）**
```json
{"rationale": "两家独立核查来源都把坟墓定位到海得拉巴同一墓地，一家实地委托核对、一家街景定位加记者到访，足以回答位置问题；未做像素比对，写入限制。",
 "task_id": "W1-1", "status": "COMPLETED",
 "findings": [{"statement": "图中坟墓位于印度海得拉巴（Hyderabad）Madannapet 地区的 Darab Jung Colony，Masjid-e-Salar Mulk 清真寺对面的墓地内，并非巴基斯坦。Alt News 报道了具体地址，并委托当地居民到场拍摄核对；AFP 也通过街景定位，且其记者于2023年5月2日到访墓地、拍摄该坟墓。",
               "evidence_ids": ["E6", "E3"],
               "limitations": ["定位依据为已读取的实地核查报道；本次未直接查看报道中的街景及现场照片。结论仅针对所附照片中的坟墓。"]}],
 "selected_evidence_ids": ["E6", "E3"]}
```

**metadata 变化**：`"unit": "report", "turn": 6, "verdict": "positive", "error_types": ["scope_overreach"], "tainted": false, "clue_gain": null, "sample_weight": 2.0`

**核验（三层，见第 4 节 verdicts）**：第一层无 hard 错；第二层 F1 拆四部分全部 supported；第三层 REQ1、REQ2 covered，状态应为 COMPLETED；一条 minor `scope_overreach`（"图中坟墓"与来源中被核查的照片同一性未经像素比对）不影响处置。

---

## 3. GIA 两行（`gia_accepted.jsonl`，origin B）

```
system:  specs/gia@0.1.0/student_prompt.md
images:  第 1 轮 ["…/E1.png"]（GIA 原件像素，2026-09-21 改动后第一轮就能看到）；第 2 轮 ["…/E1.png"]（原件组仍在，报告已不夹带 HI 过程图）
```

### GIA 行 1 —— 第 1 轮，INVESTIGATE

**user（compact_8k 示意）**
```
[总任务] 完成以下局部调查问题，查找直接证据并说明适用范围：Where is the grave in the image located?
背景声明：A photo shows that Pakistani parents locked their daughter's grave to avoid her corpse falling prey to necrophilia
声明日期：2023-04-26
[原始材料] C0 任务卡 | E1 image "材料 image_0" src=I0（像素已附）
[本轮新证据] 无   [最新报告] 无   [历史索引] 无
[约束] 第 1 轮 · turn 1/8 · 可开新轮
[可用动作] INVESTIGATE(tasks[{task, materials[]}]) · FINISH(answer) · QUERY_MEMORY · READ_RECORD
```

**assistant（教师同状态再生；脚本原输出不可用）**
```json
{"rationale": "问题只有一个：E1 中坟墓在哪。画面可见绿色铁栅栏加挂锁、阿拉伯字母墓碑、红色花瓣，没有可读地名；一个工作区做以图搜图与核查报道检索即可，定位依据和照片同一性一并要求，避免只拿到转述。",
 "decision": "INVESTIGATE",
 "tasks": [{"task": "确定所附照片 E1 中带绿色铁栅栏和挂锁的坟墓的具体地理位置（国家、城市、墓地或地标）。优先寻找做过地理定位或实地核查的来源（事实核查机构、原发布者），而不是只转述说法的媒体报道。报告中说明定位依据、适用范围，并注明来源核查的照片与 E1 是否为同一张。",
            "materials": ["C0", "E1"]}]}
```

对照：脚本原输出把总任务原文照抄成任务。两者差在三处——写进了画面特征（GIA 看了像素）、区分"定位/实地核查"与"转述"、把照片同一性写进任务。这三处分别对应第 0 轮错误分布里的 completed_without_finishing、attribution_drift、scope_overreach。

**metadata**
```json
{"role": "gia", "unit": "plan", "origin": "B", "run": "averimatec_441_q0_r1", "round": 1, "turn": 1,
 "state_ref": {"run": "averimatec_441_q0_r1", "event_id": 2, "snapshot": "perturb.snapshot@…"},
 "contrast_of": null, "teacher_model": "<待定：Qwen3.5-122B-A10B 或 API 教师>",
 "spec": "gia@0.1.0", "spec_hash": "sha256:…", "state_renderer": "compact_8k",
 "verifier": "verifier@0.2.0", "judge_model": "qwen/qwen3.8-27b",
 "verdict": "positive", "error_types": [], "round_yield": 2, "sample_weight": 1.0}
```

**核验（§2.4）**：程序——1 个任务、可解析、materials 存在、无兄弟依赖；前提走私抽取：任务文本陈述为事实的命题只有"E1 中有绿色铁栅栏和挂锁"，这是画面可直接验证的描述而非结论，不记错。缺口评判——已覆盖要求为空（第 1 轮），任务对准 REQ1、REQ2、REQ3，must_cite 项无遗漏。round_yield（只记不筛）——本轮报告经核验后覆盖 REQ1、REQ2，=2。

### GIA 行 2 —— 第 2 轮，FINISH

**user（compact_8k 示意）**
```
[总任务] （同上）
[原始材料] C0 | E1 image（像素已附）
[本轮新证据] E2 ETV Bharat 核查页 | E3 AFP 核查页 | E4 NDTV | E5 Logical Indian | E6 Alt News 核查页
[最新报告] W1-1 COMPLETED · F1 "图中坟墓位于印度海得拉巴 Madannapet 的 Darab Jung Colony，Masjid-e-Salar Mulk 清真寺对面…AFP 记者 2023-05-02 到访" ← [E6, E3] · 限制：未直接查看街景与现场照片；结论仅针对所附照片
  支撑摘录 E6 [29][31][35]（8K 档只留卡片行，16K 档留 1200 字摘录）· E3 [21][22]
  观察摘要：HI 看过 E1；读过 E6 两页、E3 一页
[历史索引] r1 INVESTIGATE 1 任务
[约束] 第 2 轮 · turn 1/8
```

**assistant（教师同状态再生）**
```json
{"rationale": "唯一要求项（位置）已由 W1-1 用两家独立来源回答，一家实地核对、一家街景定位，无未覆盖要求；照片同一性 HI 已如实写成限制，不需要再开一轮，直接作答并把限制带上。",
 "decision": "FINISH",
 "answer": "所附照片中的坟墓位于印度海得拉巴（Hyderabad）Madannapet 的 Darab Jung Colony、Masjid-e-Salar Mulk 清真寺对面的墓地内，不在巴基斯坦 [F1: E6, E3]。定位依据有两条相互独立的核查：Alt News 给出该地址并委托当地社工到场拍摄核对，清真寺宣礼员亦证实地址 [E6]；AFP 用 Google 街景从该清真寺定位墓地，其记者于 2023 年 5 月 2 日到访并拍摄 [E3]。适用范围：本结论针对两家机构核查的那张流传照片；本次未对所附照片与报道中的照片做像素比对，同一性依据是来源对同一张流传图片的描述 [E3, E6]。"}
```

**metadata 变化**：`"unit": "answer", "round": 2, "state_ref.event_id": 31, "verdict": "positive", "error_types": [], "tainted_input": false, "recovered_from_tainted": false, "sample_weight": 1.0`

**核验（§2.3 链式三层）**：第一层——F1、E6、E3 都在视图内；案卷 `label_expected` "Hyderabad, India" 实体匹配通过；F1 不在 tainted_findings 里，`tainted_input=false`。第二层——答案三句拆断言，跳过 F1 直接对 E：句 1 由 E6 [29] + E3 [21] 支持；句 2 前半由 E6 [29][31][35] 支持，后半由 E3 [21][22] 支持；句 3 是限定语，不需支持。第三层——REQ1、REQ2 covered，无 must_cite 项缺失，不是 premature_finish。

**脚本原输出为什么必须排除**：`"answer": "本次局部任务执行结束；结果保存在工作区报告中。"` 第一层就挂——无引用、无案卷标签匹配（`label_expected` 不命中）。而且它的 `teacher_model` 是 `scripted_dispatcher`，根本不是策略输出。现有 corpus 里 GIA 的 37 行如果有来自这类 hi-mode run 的，要查一遍。

---

## 4. `verdicts.jsonl`（本 run 的全部条目，8 行 + 1 行脚本对照）

```jsonl
{"unit":"step","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":1,"errors":[],"env_errors":[{"type":"env_tool_failure","detail":"SEARCH upload ACCESS_RESTRICTED"}],"checks":{"legal":true,"visible":["E1"],"duplicate_of":null,"ignored_feedback":false},"clue_gain":{"acquired":0,"observed":0,"quote_page":0,"total":0},"sampled_judge":null,"verifier":"verifier@0.2.0"}
{"unit":"step","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":2,"errors":[],"checks":{"legal":true,"duplicate_of":null,"ignored_feedback":false,"retry_after_failure":"changed_method"},"clue_gain":{"acquired":2,"acquired_ids":["E3","E6"],"observed":0,"quote_page":0,"total":2},"sampled_judge":null}
{"unit":"step","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":3,"errors":[],"checks":{"legal":true,"visible":["E6"],"duplicate_of":null},"clue_gain":{"acquired":0,"observed":1,"observed_ids":["E6"],"quote_page":0,"total":1},"sampled_judge":null}
{"unit":"step","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":4,"errors":[],"checks":{"legal":true,"duplicate_of":null},"clue_gain":{"acquired":0,"observed":0,"quote_page":1,"quote_pages":[{"evidence_id":"E6","offset":6000,"quotes":["[29]","[31]","[35]"]}],"total":1},"sampled_judge":null}
{"unit":"step","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":5,"errors":[],"checks":{"legal":true,"duplicate_of":null},"clue_gain":{"acquired":0,"observed":1,"observed_ids":["E3"],"quote_page":1,"quote_pages":[{"evidence_id":"E3","offset":0,"quotes":["[21]","[22]"]}],"total":2},"sampled_judge":null}
{"unit":"report","role":"hi","run":"averimatec_441_q0_r1","workspace":"W1-1","turn":6,"report_id":"W1-1","disposition":"positive","errors":[{"type":"scope_overreach","severity":"minor","finding_id":"F1","span":"图中坟墓","detail":"E1 与来源核查的流传照片同一性未经像素比对；限制里已声明","verifier_method":"llm"}],"tainted_findings":[],"layers_run":["exec","anchored","sufficiency"],
 "exec":{"hard":[],"flags":[],"visible_union":["E1","E2","E3","E4","E5","E6"],"cited_exist":true,"status_nonempty":true,"date_check":{"2023-05-02":"in_range"}},
 "anchored":{"F1":{"parts":[
   {"text":"位于印度海得拉巴 Madannapet 的 Darab Jung Colony，Masjid-e-Salar Mulk 清真寺对面的墓地，并非巴基斯坦","support":[{"evidence_id":"E6","quote":"The cemetery is located opposite Masjid E Salar Mulk, a mosque in Darab Jung Colony, Madannapet, Hyderabad","offset":6000,"para":29,"method":"reread_store"},{"evidence_id":"E3","quote":"from a mosque in Hyderabad called Masjid-e-Salar Mulk, which is opposite the site","para":21,"method":"reread_store"}],"contradict":[]},
   {"text":"Alt News 报道了具体地址，并委托当地居民到场拍摄核对","support":[{"evidence_id":"E6","quote":"Alt News contacted a social worker named Abdul Jaleel who is a resident of Hyderabad. On being requested by us, he visited the spot and provided us with photographs of the grave","para":31,"method":"reread_store"}],"contradict":[]},
   {"text":"AFP 也通过街景定位","support":[{"evidence_id":"E3","quote":"AFP geolocated the graveyard using Google Street View","para":21,"method":"reread_store"}],"contradict":[]},
   {"text":"其记者于2023年5月2日到访墓地、拍摄该坟墓","support":[{"evidence_id":"E3","quote":"An AFP journalist in Hyderabad visited the cemetery on May 2 and took a photo of the grave","para":22,"method":"reread_store"},{"evidence_id":"E3","quote":"Published on May 5, 2023","para":3,"note":"年份来自发布日期"}],"contradict":[]}],
   "coverage":{"unsupported_parts":[],"hint":"scope: 图中坟墓 ⇔ 来源照片 同一性"}}},
 "sufficiency":{"REQ1":"covered","REQ2":"covered","REQ3":"partially_covered_by_limitation","answerable":true,"status_should_be":"COMPLETED"},
 "judge_model":"qwen/qwen3.8-27b","spec_hash":"sha256:…","state_renderer":"compact_8k"}
{"unit":"plan","role":"gia","run":"averimatec_441_q0_r1","round":1,"origin":"B","disposition":"positive","errors":[],"checks":{"parsable":true,"materials_exist":true,"sibling_dependency":false,"premises_asserted":["E1 中有绿色铁栅栏和挂锁"],"premises_unverified":[]},"gap":{"covered_before":[],"task_targets":{"T1":["REQ1","REQ2","REQ3"]},"must_cite_untargeted":[],"unnecessary_round":false},"round_yield":2}
{"unit":"answer","role":"gia","run":"averimatec_441_q0_r1","round":2,"origin":"B","disposition":"positive","errors":[],"tainted_input":false,"recovered_from_tainted":false,"exec":{"cited_exist":["F1","E6","E3"],"label_expected":"Hyderabad, India","label_match":true},"anchored":{"sentences":3,"supported":2,"qualifier_only":1,"unsupported_parts":[]},"sufficiency":{"must_cite_uncovered":[],"premature_finish":false}}
{"unit":"answer","role":"gia","run":"averimatec_441_q0_r1","round":2,"origin":"A","teacher_model":"scripted_dispatcher","disposition":"negative","excluded_reason":"not_a_policy_output","errors":[{"type":"unsupported_assertion","severity":"critical","detail":"答案无任何引用且不含位置信息"}],"exec":{"cited_exist":[],"label_match":false}}
```

---

## 5. `manifest.json`（只含这一个 run 时的样子）

```json
{"export_id": "20260921_hi@0.1.0_gia@0.1.0", "spec": {"hi": "hi@0.1.0", "gia": "gia@0.1.0"}, "spec_hash": {"hi": "sha256:…", "gia": "sha256:…"},
 "state_renderer": "compact_8k", "verifier": "verifier@0.2.0", "judge_model": "qwen/qwen3.8-27b",
 "verdicts_file": "verdicts_r0_judged.jsonl", "cases_dir": "cases/",
 "files": {"hi_accepted": "hi_accepted.jsonl", "hi_rejected": "hi_rejected.jsonl", "gia_accepted": "gia_accepted.jsonl", "gia_rejected": "gia_rejected.jsonl"},
 "counts": {"hi_accepted": 6, "hi_rejected": 0, "gia_accepted": 2, "gia_rejected": 0},
 "by_unit": {"hi": {"step": 5, "report": 1}, "gia": {"plan": 1, "answer": 1}},
 "by_origin": {"hi": {"A": 6}, "gia": {"B": 2}},
 "by_action": {"hi": {"SEARCH": 2, "READ_RECORD": 3, "COMPLETED": 1}, "gia": {"INVESTIGATE": 1, "FINISH": 1}},
 "by_case_family": {"averimatec:441": {"hi": 6, "gia": 2}},
 "excluded": [{"run": "averimatec_441_q0_r1", "unit": "answer", "reason": "not_a_policy_output"}, {"run": "averimatec_441_q0_r1", "unit": "plan", "reason": "not_a_policy_output"}],
 "sample_weight_sum": {"hi": 13.0, "gia": 2.0},
 "env_errors": {"env_tool_failure": 1}}
```

HI 六行权重合计 13：三步高收益步（t2、t5 各 3；t3、t4 各 2）加报告 2 加一步 0 收益的 1。这个案子学生从中学到的"信号密度"分布正是我们想要的——两次 SEARCH 里改法重试的那次和读到关键段落的那两次最重。

---

## 6. 这一个例子暴露的四件事

**第一，HI 行 1 该不该教。** 教师第一步选以图搜图，在这个部署里是被禁止的；按 v2 规则它是合法动作、环境错误、权重 1，进 accepted。但学生学到的是"先试一个在这里必然失败的动作"。更好的处理不在数据端，而在规格端：`harness_params` 或 `available_actions` 应把"image / text_image 输入被禁用"写进 HI 看到的动作表，这样教师根本不会出这一步。这就是 v2 §10 说的 diagnose → revise：`env_tool_failure(ACCESS_RESTRICTED)` 在 51 个 run 里若反复出现，它就是第一个修订目标（改 harness_params，升 patch）。修订后用来源 B 在 t1 状态重生成，得到的对照对 `{bad: 行1, good: 直接文本搜索}` 才是有价值的。

**第二，脚本 GIA 的 run 一行 GIA 数据都产不出。** 51 个 run 里 `mode: hi` 的有多少，决定了来源 A 的 GIA 行到底有几条可用；现有 corpus 里 GIA 37 行的 origin 要按 `gia_output_source` 复核。GIA 数据从一开始就得靠 B（同状态再生）和 D（离线换题）——本例的 GIA 两行就是 B 的形态。

**第三，scope_overreach 这条 minor 是真问题，不是核验器误报。** E1 与核查报道里的照片是不是同一张，HI 没有验证，只在限制里声明。以图搜图被禁后，HI 其实可以 SEARCH 打开 Alt News 页面的图片再 INSPECT align_media 比对，但它没做。案卷 REQ3 设为非 must_cite 是对的（不影响处置），但这条错误在 diagnose 里要计数：如果它在图片类案例里普遍出现，第二轮修订目标就是 teacher_instruction 里加"图片同一性优先用像素比对确认"。

**第四，READ_RECORD 续页的 clue_gain 必须靠引语页码。** 本例 E6 的关键段落全在第 2 页；没有 quote-page 分量，t4 会被算成 0 收益、t3 反而有 1，学生会学到"读第一页就够"。锚定层输出引语时要带 offset / 段落号，这是对 `verify/anchored.py` 的一个具体要求。
