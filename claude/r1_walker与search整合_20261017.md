# r1 预演之二：walker 与 search 整合——E/L/C 真去比对、循环内充分性检查、新搜索策略（2026-10-17）

负责范围：walker（`datagen/walker.py`）、`experiments/sft_data/oracle_generator.py`、`scripts/walker_run.py`、search 层的以图搜合并、
`status_rewrite.walker_gaps`、导出的 hidden_context 检查器、消息渲染里的一句 notice、规格 hi@0.11.0、冻结草稿。
**核验器 `datagen/verify/`、`errors.py` 一个字没动**（verifier code_hash 仍 `sha256:c66504d4…2f96`）。不建卷、不全量重跑。

## 一、结论（冒烟 r1s，8 个案例）

| 验收项 | 目标 | 结果 | |
|---|---|---|---|
| E 可回答 accepted 里的 COMPLETED 报告 | ≥ 2/3（r0：E 0） | **3/3**（e133、e120、e92；都没被改写，循环内检查与事后核验都判 COMPLETED） | ✅ |
| L 可回答 accepted 里的 COMPLETED 报告 | ≥ 1/2（r0：L 0） | **0/2**（两遍都没做到；l623 第 1 轮 W1 INCOMPLETE 报告第 1 遍进了 accepted、第 2 遍 held） | ❌ 见四节 3 |
| C 画面类声明 accepted 里的 COMPLETED 报告 | 2/2 | **2/2**（groundlie360_v1959500、fmnv_c_FEbcNwkD7EU，0.4.0 新改写的 8 份 C 里的两份） | ✅ |
| 事后 status_rewrite 改写 | ≈ 0 | **0**（两遍都是 0） | ✅ |
| 第 2 轮派工 | （r0：3 行） | **2**（l101、l623，两遍各 2；都是循环内检查 / walker 写的可修缺口触发的 G2_round2_search_gap） | |
| 循环内检查与事后核验同一 key | | 没被改写的 5 份 5/5 key 相同（离线按事后核验同一函数重算），循环内的 13 条并进主缓存后事后核验命中 | ✅ |
| Serper | ≤ 40 次 Lens | 冒烟 **2 次 Lens = 6 点**（e120、e92；e133 回放了开发冒烟时录下的结果）；开发冒烟另 1 次 = 3 点；余额 2431 → **2422** | ✅ |
| OpenRouter | | 第 1 遍 8 个 run **$0.56**，第 2 遍重跑 L 两条 **$0.21**，合计 **约 $0.77**（S0–S3 Qwen 397B；S4 / S5 走 Codex 订阅、核验走 Claude CLI 订阅，不计费） | |
| 冻结预检（FOREN_FREEZE_STRICT=1 对 r1_freeze 草稿） | | 两遍都 ok、failed [] | ✅ |

一句话：核验器 0.4.0 要的三类动作（网页上的图与给定图比对、地点与参考照片比对、画面类声明看帧）现在由 walker 程序化地做，
并作为证据记录交给 S4；E 3/3、C 2/2 在**没有任何改写**的情况下以 COMPLETED 进了 accepted，循环内检查与事后核验对这些报告的
判定一致（同一 cache key，事后直接命中）。**L 族没达标**：第 1 遍查出两处 walker 没做的动作（参考图取了无关图、页面上的照片没比对），
修了（`a69b812`）再跑一遍，动作都做出来了，但两条都没找到"载有这张照片、说明拍于哪里"的来源——卡在检索召回和回合预算，
不是核验器不认（四节 3）。

## 二、第 0 步：收拢分支

- 开工时没有别的会话在跑（`list_sessions` 全部 isRunning=false，进程表里没有 walker / 核验进程）。
- r0+ 的 walker / `status_rewrite.py` / export 改动**早已提交**（103285d、2640e29、eeb9cc5、c4136e8、3064bf4 都在 `r1-image-search` 上），
  主目录里没有未提交的 r0+ 改动；工作区里那些 M / D 文件（cli_generator.py 的分叉试点、旧导出、cases/ 的旧改动等）是更早的会话留下的，没碰。
- `r1` 是从 `r1-image-search` 的 HEAD（0716917）拉出的，`git merge r1-image-search` = Already up to date（快进关系）。
  `git worktree remove .wt_r1` 后在主目录 `git checkout r1`。主目录里 `scripts/_gold_eval.py`（未提交改动）与
  `claude/r1核验器修正_20261017.md`（未跟踪）与 `r1` 上的版本逐字节相同，先备份到草稿目录再让 checkout 覆盖。
- 本对话 11 个提交（含本报告）都在 `r1`，**没 push**。

## 三、改了什么（每项单独提交，分支 `r1`）

| 提交 | 项 | 改动 |
|---|---|---|
| `9705402` | 1 | 循环内充分性检查用 0.4.0 的 `sufficiency_payload` + `lookup=StoreLookup(self.store)`，与事后核验同一 payload、同一 key；**第 2 轮沿用第 1 轮的族**（见下） |
| `b59ea2e` | 2 | E 族：以图搜取页面上的图、读过的页面 E6 比对、没比对 / 没对上只写候选、E7 补看对照图、哈希 no match 只降权；search 层第二家引擎缩略图 |
| `f4331c2` | 3 | L 族：S4 必须写参考图比对结论（required_findings 进 S4 合同校验） |
| `c548321` | 4 | C 族：video_structure + 按镜头取帧；交报告前整段转写 + 场景帧放回附图；画面类 / 说话类声明的 S4 合同项 |
| `fc21206` | 5 | r1 以图搜策略：Lens 只给 E、图床只给 E 的案例图、litterbox 关、L / C 只 Yandex |
| `6d9ed31` | 6 | 规格 hi@0.11.0（teacher_instruction 末尾两句） |
| `cf5aa09` | 7 | harness_event.notice 英文；hidden_context 放行 1/fps 与 INSPECT 采样参数；测试 |
| `f09861a` | 10 | `r1_freeze.json` 草稿填规格 / spec_hash / checks / teacher / walker 策略，重算工具层 hash；`specs_for` 认 r1* |
| `d7ed5f7` | 8 | 测试会话护栏（受管理的 verify_cache 不许被测试改动） |
| `a69b812` | 3 补 | 冒烟第 1 遍查出的 L 族两处：参考图按查询词排序去掉无关图；以图搜页面上的图与给定照片比对（L6） |

### 第 1 项：循环内充分性检查（接到 0.4.0）

- `oracle_generator._after_report`：payload 改由 `verify.sufficiency.sufficiency_payload(stored, task, case, requirement_ids, lookup)` 生成
  （`lookup = StoreLookup(self.store)`，证据索引带所引 E 的标题 / 取得方式 / 网址），`verify_sufficiency` 同样传 lookup；记日志的
  key 也按这份 payload 算。**验证**：冒烟里 5 份没被改写的 COMPLETED 报告，按事后核验同一函数、同一范围重算的 key 与循环内记的
  key 5/5 逐字相同（`claude/_r1s_work/inloop_key_check.py`）；被循环内改写的 2 份 key 不同（存进 store 的已是 INCOMPLETE 报告），
  符合预期。循环内的 13 条判官回答在核验前并进主缓存，事后核验按 plan 定范围的那一步直接命中这 5 份的充分性判定。
- 判 `completed_without_finishing`（major / critical）→ status 改 INCOMPLETE、missing（`build_reason` 过滤后）+ walker 的可修缺口进
  status_reason → GIA 的 `_search_gap` 认这些说法、在轮次预算内派第 2 轮；第 2 轮报告同样核（r0+ 已有，这次接上 0.4.0）。
  事后 `status_rewrite` 保留作兜底。
- **第 2 轮沿用第 1 轮的族**（新修）：r0h e133 的第 2 轮任务 "…as claimed. Scope: open and read <网址>" 被 `task_family` 判成
  B 族（`\bclaim\b` 不认 claimed），走到 G9 直接交 INCOMPLETE——GIA 派的第 2 轮白派了。OracleGenerator 按 W1 工作区的任务记
  `run_family`，之后的工作区放进 memory，`decide_hi` / `_report` 用它；`status_rewrite.walker_gaps` 重放时同样带上。

### 第 2 项：E 族"网页上的图是不是同一张"

walker 规则（`_decide_image_origin`）：

- **E2 以图搜改 `output_type=image`**：每个结果另带这个网页上的那张图（引擎报的原图；取不到退到缩略图）+ 页面 + 缩略图。
  （`output_type=text` 时 Yandex 结果没有 image_url，页面上的图根本拿不到；页面抽取也不抽图。）
- **(a) E6_compare_page_image**：读过、带图、还没比过的页面 → `INSPECT compare_media(E1, 那张图)`（图片模式：SIFT 配准后逐像素差异 +
  "A | B | 对齐后的 B"三栏对照图），有日期的按日期先比，每个工作区 ≤ 3 次。结果分级：差异面积 ≤ 2% same、≤ 35% edited
  （同一张有差异）、更大或配准失败且 > 10% different。比对记录与对照图都是证据 E。**E7_view_comparison_before_report**（同 L3b）：
  要交报告时对照图不在附图里就补看一次（E1 + 页面那张图 + 对照图作焦点），下一轮直接交报告（这一轮也不插 QUERY_MEMORY）。
- **(b) 没比对不许写"最早来源"**：COMPLETED 只给比对过且对上（same / edited）的有日期页面；有日期页面都没比对 → INCOMPLETE
  "仍是候选"；比对了但都不是同一张 → INCOMPLETE "compare_media 没显示是同一张"。S4 说明逐页写比对结果，并写明"没比对 / 没对上
  的页面只能写 candidate 并说明原因"；`required_findings` 要求每条比对记录都被某个 Finding 引用并写结论（进 S4 合同校验，不合格
  重写一次）。
- **(c) `verified_match=different` 只降权**：日期查询与读页的排序里排后，不排除。
- **(d) 缩略图第二家引擎**（search 层 `multi_reverse._second_engine_thumbs`）：base 引擎缩略图没判 near-identical 时，再试同一网页
  其他引擎的缩略图，取更好的判定，extra 记 `verified_engine_fallback`。
- 实测佐证 (c)：冒烟 e133 的 10 个缩略图哈希**全判 no match**，而页面上的原图与 E1 配准成功（52 个内点）、差异 11%（两边裁法
  不同 + 重压缩）——哈希只能降权，不能拿来排除。

### 第 3 项：L 族

- 接 r0+ 的结论（eeb9cc5：r0g 四份可回答 L 报告没有一条写比对结果；L3b 改成交报告前最后一步）。r0h 的核验仍判 l101 / l274
  "A feature-by-feature comparison of E1 with the reference photos" 缺：参考图回到了附图，S4 却没写结论。L3（READ_RECORD 参考图、
  给定照片作焦点）本来就是真动作，这次补的是"结论交给 S4、S4 必须写"：S4 说明 + `required_findings`——参考图在附图里 → 一条
  Finding 写 match / no match / cannot tell（及原因），列出对上 / 对不上的特征，引 E1 与参考图；不在附图里 → 写本报告无法确认比对
  及原因；没取到参考图 → 写无法比对及原因。第 2 轮跟进同样要求。
- **冒烟第 1 遍又查出两处 walker 没做的动作**（`a69b812`，四节）：参考图取了无关图；以图搜 / 文字搜到的页面上的照片没与给定照片比。

### 第 4 项：C 族

- `claim_kinds`：程序按声明文字判 visual（画面里的动作 / 事件 / 物体）与 speech（有人说了什么），可两者都是；没有说话类的词时
  按画面类（REQ1 本来就问"画面、屏幕文字、语音"）。
- **C2v_scene_frames**：OCR 之后 `INSPECT video_structure`（镜头切分 + 每个镜头的中间帧图）；没有这个操作时 get_frames 均匀 6 帧。
  **C2w_frames_per_shot**：镜头少或长时按镜头取帧（长镜头约每 30 s 一张，≤ 6 张）——r1dev v1959500 整段 166 s 只有 1 个镜头、
  1 张中间帧，看不出茶里有没有"虫子"。
- **C9v_review_before_report**（同 L3b）：交报告前 READ_RECORD 完整转写（有语音时），场景帧作焦点放回附图，下一轮直接交报告。
- S4 说明 + `required_findings`：画面类 → 一条 Finding 写帧里有没有声明说的那件事（看得到什么、哪一帧、看不到什么），引帧；
  有转写 → 一条 Finding 概括整段视频说了什么（不只第一句），引转写记录。这就是 0.4.0 第 2 处补的三类缺口里"没看帧"与"只引转写
  第一句"的验收口径。

### 第 5 项：搜索策略（`walker_run.r1_search_policy`，只对 r1* 轮次，r0 行为不变）

| 族 | 以图搜引擎 | 图床 | 备注 |
|---|---|---|---|
| E | lens,yandex | imgbb（只在 run 子进程里设），`FOREN_LENS_IMAGE_FALLBACK=off` | |
| L / C | yandex | 关 | L 族默认不用 Lens（10-16 实测 L 4 张前 10 召回 Lens / Yandex 都是 0） |
| B | 关 | — | 给定媒体不检索 |

`FOREN_REVERSE_NORMALIZE=1`（walker 要从摘要里读缩略图自核结论）；Lens 次数上限仍由 `FOREN_LENS_MAX_SEARCHES` 控制（冒烟设 40，
不写进代码）。`.local` 的 key 一个都没入库。run_config.walker 记 `search_policy` 与 `policy_version`。

### 第 6 项：规格 hi@0.11.0

teacher_instruction 末尾加两句："A near-identical hash does not show that the copy is unedited: compare that page's picture with the given
image using compare_media to find local differences. A hash mismatch can come from cropping, an added border or a local edit, and does not
show that the page lacks the picture."（前面一句引子说明以图搜结果会给缩略图比对结论。）student_prompt / rubric / harness_params 与
0.10.0 逐字相同——student_prompt 必须与运行时 `_hi_turn_prompt` 逐字一致（导出记 `runtime_system_matches_spec`），改它就得改运行时
提示；两句是给教师槽（S4）的写作约束，与 0.10.0 只改 teacher_instruction 的做法相同。**新 spec_hash
`sha256:c981099064da5e91b13b5ffbb7f06b6d3b23a662994c1d3907f98d8736a7b504`**；gia@0.9.0 不动（`a43f17ad…b033`）。注意九节 3。

### 第 7 项：工具层与检查器

- 重复操作合并时插的 `harness_event.notice`（`local_models._generation_messages`）：英文规格用 `prompts_en.MERGED_OCCURRENCES_NOTICE`，
  中文规格不变。
- `export.hidden_context`：等于 1 / 学生可见帧率（±1%）的数放行；INSPECT 动作参数（times_s、range_s 等采样点）不按隐藏词查。

### 第 8 项：小修

- **测试往受管理的缓存写假回答**：35 个核验相关测试文件逐个单跑、再全量跑一遍（2285 passed / 10 failed），每次前后对比全部
  `verify_cache*.jsonl`（先备份）：**都没再改动，复现不了**，没找到是哪一条。`datagen/verify` 本对话不改，所以加了
  `tests/conftest.py` 会话级护栏：开始记大小与 sha1，结束时只在末尾追加 → 截回原长度并报错；别处变了 / 新建 → 报错。用一条临时测试
  验证过（追加后被截回，与备份逐字节相同）。
- `status_rewrite.build_reason` 的 GT 匹配：r0+ 已按《r0数据包定稿》二节改完（2640e29：只去掉与 GT 字段实际匹配的条目、句首大写词
  不当专名、中文条目退到 text_en），不再动。
- 全量测试 10 条失败**在本对话改动前同样失败**：`test_datagen_units_export`、`test_datagen_sufficiency_scope` ×2、
  `test_teacher_student_input…hidden_context[hold]` 用本对话之前的 `export.py` / `status_rewrite.py` 跑同样失败（用的是工作区里有旧改动的
  案卷 / 判定文件）；其余是 0.4.0 报告列过的计时 / 环境 / 真实媒体类（judge_progress、inspect_real_cases、inspect_production_media、
  web_trial、workspace_time_budget、draft_case）。本对话新增测试 `tests/test_r1_walker_20261017.py` 16 条全过。

## 四、冒烟 r1s

命令（两遍相同；第 2 遍前把 L 两条第 1 遍的 run 改名为 `*.failed-pass1-20261017`，d1 跳过已完成的 6 条、只重跑 L）：

```
FOREN_FREEZE_STRICT=1 FOREN_LENS_MAX_SEARCHES=40 python scripts\walker_run.py d1 --round r1s --jobs 2 --no-auto-backup ^
  --cases averimatec_e133 averimatec_e120 averimatec_e92 averimatec_l623 averimatec_l101 groundlie360_v1959500 fmnv_c_FEbcNwkD7EU videosham_4126
```

带循环内检查（默认开）、核验 0.4.0（claude-cli sonnet）、答案按证据重写、事后 status 改写、导出
（`export/r1s_8runs-6a86be40_hi@0.11.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5`）。统计：`claude/_r1s_work/analyze_r1s.py`。

### 1. 每个案例的动作序列（新规则加粗）

| run | 动作序列（W1 工作区；→ GIA） | 新规则触发 | 报告 → 循环内 → 桶 |
|---|---|---|---|
| e133 | OCR → 以图搜(image) → 日期 ×3 → 读最早有日期页 → **E6 比对** → 读页 → **E6 比对** → **E7 补看对照图** → 报告 | E6×2、E7×1 | COMPLETED → 不改 → accepted |
| e120 | OCR → 以图搜(image) → 日期 ×2 → 读页 → **E6** → 读页 → **E6** → **E7** → 报告 | E6×2、E7×1 | COMPLETED → 不改 → accepted |
| e92 | OCR → 以图搜(image) → 日期 ×2 → 读页 → **E6** → 文字搜 → 日期 → 文字搜 → **E7** → 报告 | E6×1、E7×1 | COMPLETED → 不改 → accepted |
| l623（第 2 遍） | OCR → 参考图(text→image) → **L3**(相关图) → 以图搜(image) → 读页 → **L6 比对** → 读页 → **L6** → 读页 → **L3b** → 报告 → G2 派第 2 轮（读 1 页 + 参考图 + **L3**） | L3×2、L6×2、L3b×1 | W1 COMPLETED → **循环内改 INCOMPLETE** → held；W2 INCOMPLETE → rejected |
| l101（第 2 遍） | OCR → 参考图 → **L3** → 以图搜(image) → 读页 → **L6** → 文字搜 → 读页 ×2 → **L3b** → 报告 → G2 派第 2 轮（读 3 页 + 参考图 + **L3**） | L3×2、L6×1、L3b×1 | W1 INCOMPLETE（walker 判）→ held；W2 INCOMPLETE → held |
| v1959500 | ASR → OCR → **C2v video_structure**（1 个 166 s 镜头）→ **C2w 按镜头取 6 帧** → 文字搜 → 读页 ×2 → 日期 → **C9v 整段转写 + 6 帧** → 报告 | C2v、C2w、C9v | COMPLETED → 不改 → accepted |
| FEbcNwkD7EU | ASR → OCR → **C2v**（2 个镜头）→ **C2w 2 帧** → 文字搜 → 读页 ×2 → 打开链接视频 → compare_media → **C9v** → 报告 | C2v、C2w、C9v | COMPLETED → 不改 → accepted |
| videosham_4126（B） | compare_media → 裁帧 B → 裁帧 A → 报告 | —（B 族未改） | INCOMPLETE（walker 判，区间边界）→ accepted |

- S4 合同校验：新加的 required_findings（E 比对结论 / L 比对结论 / C 帧与整段转写）**一次都没触发重写**——S4 第一次就写了这些
  Finding（例：e133 "Material A crops the left side, changes the framing and adds an upper watermark"；l623 第 1 遍"参考图是字母
  练习纸、字母 U，没有能认建筑的特征"）。触发重写的两次是旧规则（4126 只引采样帧；l623 第 2 遍）。
- 缩略图哈希：e133 的 10 个缩略图全判 no match，而页面原图与 E1 配准成功（52 内点、差异 11%）——(c) 降权不排除是对的。

### 2. 循环内检查、第 2 轮、事后改写

| | 第 1 遍 | 第 2 遍（L 重跑后，全轮） |
|---|---|---|
| 报告份数 / 核了（COMPLETED） | 10 / 7 | 10 / 6 |
| 循环内改 INCOMPLETE | 2（l623 W1：参考图没真比、BuzzFeed 的照片没确认是同一张；l101 W2：没读到载有这张图的来源） | 1（l623 W1：参考图比对没有结论、载有这张图的页面没说拍于哪里） |
| 第 2 轮派工（G2_round2_search_gap） | 2 | 2 |
| 事后 status_rewrite 改写 | **0** | **0** |
| 判官新调用 / 缓存命中（循环内） | 13 / 0 | 12 条并进主缓存，新增 1 |

### 3. L 族为什么没达标（walker 没做动作，还是核验器不认）

- **第 1 遍：walker 没做的动作 2 处**（已修 `a69b812`）：
  1. l623 的参考图查询 "U.S. Capitol building interior" 前 3 张是拉链零件、字母 U 剪贴画，"State Capitol Buildings Interior" 排在后面；
     L3 拿无关图比，S4 如实写"参考图里没有能认建筑的特征"，核验判"没有与国会大厦参考照片比对"。→ `ranked_refs`。
  2. l623 W1 交的是 COMPLETED（BuzzFeed 写明照片拍于 Kansas），核验判"没确认 BuzzFeed 文章里的照片就是给定这张"——与 E 族同一类缺口。
     → L4 以图搜改 image 输出、读过的以图搜页面做 L6 比对。
- **第 2 遍：动作都做了，答案没找到。** l623：参考图换成真正的国会大厦内景，S4 写"比对没有结论"（实际是堪萨斯州议会大厦，与
  美国国会大厦内景确实"不像也不能排除"）；L6 比了两个以图搜页面上的照片，但这两页都没说拍于哪里；第 1 遍靠 S3 选的文字搜索
  找到的 BuzzFeed，第 2 遍 S3 没再选文字搜（回合被 L6 比对用掉 2 个），于是没有来源说拍摄地 → 循环内改 INCOMPLETE → 第 2 轮
  读到的 lookphotos 页是安全验证页。l101：GT 是 Kolkata，以图搜与文字搜两遍都没碰到说加尔各答的页面。
- **结论**：核验器的判定与报告内容相符（没找到来源说拍于哪里，就不该 COMPLETED），不是核验器不认；瓶颈是 (a) 检索召回——
  L 族以图搜在 10-16 实测里前 10 召回就是 0/4，(b) 12 回合里 OCR / 参考图 / L3 / 以图搜 / 读页 / 比对 / L3b / 报告已占满，
  (c) **文字搜到的页面（BuzzFeed）拿不到页面上的照片**：页面打开器不抽页内图片，`SEARCH source=R output_type=image` 对文字结果
  只有缩略图（多半也没有），所以"这篇文章里的照片是不是同一张"walker 根本没有工具去比（八节 1）。

### 4. 花费

- Serper：冒烟 Lens 2 次（e120、e92 各 1 次，每次 3 点）= 6 点；e133 回放开发冒烟（r1dev，1 次 = 3 点）录下的结果，0 次。
  L / C 族按策略不用 Lens。余额 2431 → 2422（余额接口读数）。`FOREN_LENS_MAX_SEARCHES=40` 远没到。
- OpenRouter（S0–S3，Qwen 397B @ DeepInfra）：第 1 遍 $0.56、第 2 遍 L 两条 $0.21，合计约 **$0.77**。
- Codex（S4 / S5 + 答案重写）、Claude CLI（循环内检查 + 核验）：订阅，不另计费；Claude CLI 报的 API 折算见日志（订阅登录不实际扣费），
  限额命中 0。

## 五、怎么读这些数

1. **E / C 的 COMPLETED 是"做了动作"换来的，不是放宽**：核验器没动，E 族 0.4.0 判"没确认网页上的图就是给定那张"的缺口，
   这次报告里每条用到的页面都有一条 compare_media 记录和一句结论；C 族报告引了场景帧和整段转写。循环内检查（同一判官、同一
   payload）对这 5 份都判 COMPLETED，事后核验命中同一条缓存——所以事后改写为 0 不是因为兜底没跑，而是没有要改的。
2. **样本只有 8 个**：E 3 个、C 2 个都选的是可回答案例；E 的结论是"不晚于 X 日的最早副本"，与 GT 的最早来源不一定同一个
   （e133 找到 2021-08-26 的 163.com，GT 是 2021-08-12 的 Hindustan Times）——核验器的充分性层按设计不拿 GT，这类差别只在答案层
   的 GT 检查里出现。全量重跑（第 3 个对话）才能看 E 可回答 15 个里的比例。
3. **答案层**：答案 positive 第 1 遍 4/8 → 重写后 7/8；第 2 遍 L 两条的答案仍 negative（没有来源说拍摄地）。
4. **第 2 轮派工 2 行**都来自 L 族，E / C 一次都没派——它们第 1 轮就答完了。r0 的"第 2 轮派工 3 行"主要是缺口派不出去；
   这次派出去的两次都按 Scope 读了页、补了参考图，只是没找到答案。

## 六、冻结草稿 `specs/r1_freeze.json`

| 项 | 值 |
|---|---|
| status | draft（tag 不打） |
| specs | hi@0.11.0 / gia@0.9.0 |
| spec_hash | hi `sha256:c9810990…b504`，gia `sha256:a43f17ad…b033` |
| tool_layer.hash | `sha256:04590a14dbed932b11b42a650ef0ea64470201ff2207877538c9e27e2d3a4e20`（search 层第二家引擎缩略图） |
| verifier.code_hash | `sha256:c66504d4…2f96`（未变），transport `7a9be8f2…bb3bd` |
| checks / teacher | 照抄 r0（同一分槽表与主 / 备用模型；oracle_generator 的槽表仍从 r0_freeze 读） |
| walker | `policy_version = walker-r1@2026-10-17`，规则清单、以图搜策略、inloop_sufficiency=true |

`walker_run.py freeze --round r1s`：ok、failed []。`FOREN_FREEZE_STRICT=1` 的冒烟两遍都按这份草稿过了预检。walker.py 不在工具层 hash
的路径里（hi/workspace.py、inspect、search、state_render.py），所以 `a69b812` 之后 hash 不变；walker 版本靠 policy_version 记。

## 七、产物

- 代码：分支 `r1` 本对话 11 个提交（三节 + 本报告），没 push。
- 冒烟：`experiments/sft_data/runs/*_r1s_1`（第 1 遍的 L 两条留档为 `*_r1s_1.failed-pass1-20261017`）、
  `verdicts_r1s_judged.*`、`export/r1s_*`、`walker_r1s_inloop.json`、`walker_r1s_revise.json`、`walker_r1s_status_rewrite.json`、
  `walker_r1s_compare.md`、日志 `_walker_r1s日志.txt`。
- 开发冒烟（mock 槽、真检索）：`runs/*_r1dev_1`、`*_r1dev2_1`（e133 / v1959500 / l623；测试的重放用例读它们）。
- 统计脚本：`claude/_r1s_work/analyze_r1s.py`（→ `r1s_summary.json`，第 1 遍另存 `r1s_summary.pass1.json`）、`inloop_key_check.py`。
- 规格：`specs/hi@0.11.0/`、生成脚本 `scripts/_make_hi_0_11_0.py`。

## 八、没做完 / 要负责人定

1. **L 族 COMPLETED 0/2**（目标 ≥ 1/2）：动作齐了，卡在召回与工具。要负责人定的方向：(a) 工具层让页面打开器抽页内主图
   （og:image / 正文第一张大图）作为一种材料——E / L 族才能比"文字搜到的文章里的照片"（BuzzFeed 这类），这会改工具层 hash；
   (b) L 族回合上限从 12 放宽（harness 参数，改规格）；(c) L 族文字搜索（"照片 + 地点"）提前到以图搜之后必做一次，而不是交给 S3 选。
   我倾向 (a)+(c)。
2. **合并与推送**：分支 `r1` 比 `r1-image-search` 多 28 个提交（0.4.0 那次 17 个 + 本对话 11 个），在主目录、没 push；主目录现在停在 `r1`（不是
   `r1-image-search`）。`.wt_r1` 已删。
3. **hi@0.11.0 的版本号**：只改了 teacher_instruction，`datagen.specs.validate_bump` 的规则要求升 patch（hi@0.10.1）。按要求发成
   0.11.0、changelog 里写明了；如果要合规则，改名成 0.10.1 只需重跑 `scripts/_make_hi_0_11_0.py`（改版本号）并更新 r1_freeze。
   这两句没进 student_prompt（学生看不到）；要学生也学到，得同时改运行时 `_hi_turn_prompt`（升 minor 才名副其实）。
4. **E 族"最早"与 GT 的差别**（五节 2）：e133 报的是不晚于 2021-08-26，GT 是 2021-08-12；以图搜结果里没有 Hindustan Times。
   要不要在 E 族加"按日期再搜一次更早的副本"（文字 + 图），全量后再看。
5. **测试写缓存**没复现、没找到元凶（三节第 8 项）；现在有 conftest 护栏兜底。核验器那边（CachedJudge 写死默认路径）要不要改成
   测试下拒写，等下次允许动核验器时再定。
6. **r1_freeze 的 teacher 照抄 r0**、oracle_generator 的槽表仍从 `specs/r0_freeze.json` 读（`FOREN_TEACHER_FREEZE` 没改默认）；打
   `r1-frozen` 时要不要把槽表也切到 r1_freeze，第 3 个对话定。
7. walker.py 不在工具层 hash 的路径里，walker 的版本只靠 `walker.policy_version` 记；全量前如果 walker 再改，记得同步改这个版本号。
8. `test_r1_walker_20261017.py` 里的重放用例读本机 run 目录（r1dev / r1s / r0h，git 不管），换机器会跳过（同 test_r0plus 的做法）。

### 八节的负责人决定（10-17）

1 → 取 (a)+(c)，(b) 回合上限不放宽（九节）。2 → `r1` 分支 push 一份作备份，main 不动（已 push，见九节末）。3 → hi 版本号保持 0.11.0，
不改名。4 → E 族"再搜更早副本"等全量后看答案层 GT 检查再定。5 → 等允许动核验器时再改。6 → 第 3 个对话打 `r1-frozen` 时把槽表
切到 r1_freeze。7、8 → 照旧。

## 九、补：页内主图抽取 + L 族必做文字搜（负责人定 (a)+(c)），冒烟 r1s2

### 1. 改动（分支 `r1`）

| 提交 | 改动 |
|---|---|
| `342fc06` | **工具层**：`ImageSourceOpener` 对没有引擎原图的网页（`SEARCH {"source": R, "output_type": "image"}`）抽页内主图——og:image（含 secure_url / url）→ twitter:image → link rel=image_src → article / main / body 里第一批 `<img>`（跳过 data:、svg、logo / icon / 头像、声明宽高 < 200），取第一张能下载、能完整解码、≥ 200×150 的图；材料 kind=image，content 记图片网址 / 所在页面 / 抽取方式，label "main image of the page (<方式>)"，acquired_by `SEARCH/open/page_main_image:<方式>+Pillow`。按需触发，文字搜索的结果形态与回放 key 不变 |
| `15c97f9` | **walker**：读过但没有图的页面（文字搜到的）先 `E6a / L6a_open_page_image` 取页内主图，下一轮 E6 / L6 比；`_search_pages` 把之后取到的图补进先前的页面条目；**L 族比对 ≤ 2 次**（打开算进名额，还剩 ≥ 4 回合才打开）；**L4b_place_text_search**：以图搜之后必做一次文字搜（S1 新 purpose `place_text`：图里的文字照原文与原文字、以图搜结果页标题、地点线索），不交给 S3，它的第 1 个结果页必读、以图搜结果页必读减到 1 个。策略版本 walker-r1@2026-10-17b，工具层 hash 重算 |
| `a2c9c15` | **walker 修**：`HiState` 也从 SEARCH 结果材料登记卡片——卡片清单满（12）之后，第 2 轮打开再读的页面进不了 S4 的可引清单（r1s2 第 1 遍 l208 W2-1 读到写着 Peru 的 geologyscience 页却"按 evidence-ID 限制不能引"）。17c |
| `4fcfd17` | **工具层**：页内主图网址取不到时再试互联网档案馆原样存档（`web.archive.org/web/2id_/<网址>`）；**walker**：特意取了页内主图的页面排在比对最前（r1s2 第 2 遍 l623 的比对名额先给了以图搜副本）。17d，工具层 hash `sha256:fb74a40b…698f` |

`r1_freeze.json`：tool_layer.hash `sha256:fb74a40bdf985954328252e9d7e9bcc51d906a7f7870c67572d806cc434c698f`，`walker.policy_version =
walker-r1@2026-10-17d`（规则清单逐版追加），verifier code_hash 仍 `c66504d4…`；`freeze --round r1s2` ok。测试 `test_r1_walker_20261017.py`
19 条（新增页内主图候选顺序 / 过滤、补进页面条目、L6a → L6 计入上限）、`test_r0plus` 的 l208 重放预先记 L4b 已做；两文件 31 条全过。

### 2. 冒烟 r1s2（L 4 个可回答 + e92）

命令同四节（`--round r1s2`，`FOREN_FREEZE_STRICT=1`，`--jobs 2`，`FOREN_LENS_MAX_SEARCHES=40`）。跑了三遍：第 1 遍（17b）L 0/4，
查出卡片登记的 bug（`a2c9c15`）；第 2 遍（17c）L 1/4，查出 l623 的图片主机一时解析失败 + 比对优先级（`4fcfd17`）；第 3 遍（17d）
重跑 l623 / l101 / l274（l208、e92 沿用第 2 遍）。前两遍的 L run 留档为 `*_r1s2_1.failed-pass1-20261017` / `.failed-pass2-20261017`。

| 案例 | 最终（第 3 遍）动作要点 | 报告 → 循环内 → 事后 → 桶 | |
|---|---|---|---|
| **l208**（GT：Peru） | OCR → 参考图 → L3 → 以图搜(image) → **L4b**"landslide near Cusco Peru 2018 aerial photo" → 读 SERC 页 → **L6a** 取页内主图（first_large_img）→ **L6** 比对 **0.0%（同一张）** → 读页 → L3b → 报告 | W1 COMPLETED（Cusco, Peru，引 SERC 页与比对记录）→ 不改 → 不改 → **accepted** | ✅ |
| l623（GT：Kansas） | … → L4b → S3 再搜一条 → 读 BuzzFeed（Kansas / Topeka）→ **L6a 取图失败（NETWORK_ERROR）** → 读页 → L3b → 报告 → 第 2 轮读 3 页 | W1 COMPLETED → **循环内改 INCOMPLETE**（"BuzzFeed 的照片没确认是同一张"）→ accepted（INCOMPLETE）；W2 COMPLETED → 循环内不改 → **事后改写** → held | ❌ |
| l101（GT：Kolkata） | … → L4b"slums on railway tracks India…" → 读页 → **L6** 以图搜副本 0.0% 同一张（博客页，没写拍摄地）→ 读页 ×2 → L3b → 报告 → 第 2 轮 | W1 INCOMPLETE（walker 判：没有来源说拍于哪里）→ held；W2 COMPLETED → 不改 → held（答案层 GT 检查：结论里没有 Kolkata） | ❌ |
| l274（GT：Kremlin） | … → L4b"Vladimir Putin golden doors Kremlin Grand Kremlin Palace…" → 读 Daily Mail 页 → **L6a** 取到 og:image → **L6 比对 75% 不同**（是另一张克里姆林宫照片）→ 读页 → L3b → 报告 → 第 2 轮 | W1 INCOMPLETE → rejected；W2 INCOMPLETE → held | ❌ |
| e92（E，回归） | 与 r1s 同：以图搜(image) → 日期 → 读页 → **E6** ×2（20% / 25%，同一张有差异）→ E7 → 报告 | COMPLETED → 不改 → accepted | ✅ E 没坏 |

**L 族 accepted 里的 COMPLETED：1/4（l208）**，目标 ≥ 2/4 **没达到**。e92 照旧 COMPLETED。第 2 轮派工 3（l623、l101、l274）；
事后 status_rewrite 改写 1（l623 W2）；答案 positive 2/5 → 重写后 3/5。

### 3. 页内主图抽取：命中几次、比对结果

| | 第 1 遍 | 第 2 遍 | 第 3 遍 | 最终 4+1 个案例 |
|---|---|---|---|---|
| 发起取页内主图（E6a / L6a） | 3 | 4 | 2（另 l208 沿用第 2 遍 1 次） | l208 1、l274 1、l623 1、l101 0、e92 0 |
| 取到（ready） | 1（l274 og:image） | 2（l208 first_large_img、l274 og:image） | 1（l274 og:image） | **2**：l208、l274 |
| 没取到 | l101 absent、l623 EXTRACT_FAILED | l101 absent、l623 NETWORK_ERROR | l623 NETWORK_ERROR（档案馆存档也没取到） | l623 |
| 取到的图的比对结果 | l274：75% 不同 | l208：**0.0% 同一张**；l274：75% 不同 | l274：75% 不同 | l208 同一张（→ COMPLETED）；l274 不同（Daily Mail 那张是克里姆林宫的另一张照片，判"不是同一张"是对的） |

另外以图搜（image 输出）副本的比对：l208 / l101 / l623 各有 1 个副本 0–1.2% 同一张——这些页面（博客、图集）载有这张照片，但都没写拍摄地。
e92 的两次 E6 都是以图搜副本（20%、25%）；e92 没读文字搜到的页面，不需要取页内主图。

### 4. 没到 2/4：按"检索召回 / 回合 / 工具"归因

| 案例 | 归因 | 说明 |
|---|---|---|
| l623 | **工具**（主）+ 检索召回 | 三遍里两遍文字搜都找到了 BuzzFeed（写明 Kansas / Topeka），但它的图片主机 img.buzzfeed.com（BuzzFeed News 2023 年停刊）在冒烟时两次连不上（EXTRACT_FAILED / NETWORK_ERROR），档案馆存档也没取到；单独试时又能取到——这台机器到该主机的网络时好时坏。照片取不到就比不了，核验器按规则判"没确认是同一张"。第 2 轮再读的页（patriots.win、imgflip、dailydot）不写拍摄地 |
| l101 | **检索召回** | 文字搜与以图搜都没碰到写"Kolkata"的页面；以图搜副本（博客）是同一张但没写地点；第 2 轮的 COMPLETED 报告给的地点与 GT 不符（答案层 GT 检查判 omitted Kolkata） |
| l274 | **检索召回** | L4b 的查询词对（Kremlin / Grand Kremlin Palace），但搜到的页（Daily Mail）用的是另一张照片；没有一页"载有这张照片并写明拍于克里姆林宫"，第 2 轮的"克里姆林宫"只靠与参考图的外观相似，循环内检查改 INCOMPLETE |
| （回合） | 次要 | 12 回合下 L 族：OCR、参考图、L3、以图搜、L4b、读 2–3 页、取图 + 比对、L3b、报告，基本用满；l623 第 3 遍 S3 又加了一条文字搜，取图那一步已排在第 9 回合，比对来不及（取图失败也没给比对留余地） |

## 十、要负责人定（第九节新增）

1. **L 族 1/4**：剩下 3 条里 2 条是检索召回（l101、l274：网上没找到"载有这张照片 + 写明地点"的页），1 条是工具（l623：页内主图主机
   连不上）。可选：(i) l623 这类给图片下载加重试 / 换出口（代理）——工具层；(ii) 第 2 轮 Scope 里写明"取 <页面> 的页内主图并比对"，
   让第 2 轮专门补这一步（walker + GIA，不改回合上限）；(iii) 接受 L 族可回答案例的 COMPLETED 率就在 1/4 左右，全量时看 15 个可回答的整体。
   我倾向 (ii)：l623 第 1 轮已经找到 BuzzFeed，只差比对，第 2 轮补比对最便宜。
2. **第 2 轮报告的循环内检查与事后核验范围不同**：循环内第 2 轮只能按任务文字定范围（没有 plan 评判），事后按 plan 定范围，两边判定
   可能不一致——r1s2 l623 W2 循环内判 COMPLETED、事后改写 INCOMPLETE（事后改写 1 份就是它）。要不要让循环内第 2 轮也跑 plan 评判（多
   1 次判官调用）？
3. 页内主图取不到时档案馆存档用的是 `web/2id_/`（最近一次存档）；要不要按页面日期取存档时间点，全量前定。

本节花费：OpenRouter 三遍合计约 **$1.81**（第 1 遍 $0.76、第 2 遍 L 四条 $0.64、第 3 遍三条 $0.41）；Serper **0 点**（e92 回放 r1s 的录制，
L 族按策略不用 Lens），余额仍 2422。

产物：`runs/*_r1s2_1`（及两遍留档）、`verdicts_r1s2_judged.*`、`export/r1s2_*`、`walker_r1s2_*.json`、日志 `_walker_r1s2日志.txt`；
统计 `claude/_r1s_work/r1s2_summary*.json`、`r1s2_page_images*.json`、`page_image_stats.py`（未入库）；开发冒烟 `runs/*_r1dev3_1`。

### 十节的负责人决定（10-17）

1 → 取 (ii)（十一节）。2 → 要：第 2 轮的循环内检查也跑 plan 评判（十一节）。3 → 全量前不动，页内主图的档案馆存档仍用 `web/2id_/`
（最近一次存档），留给第 3 个对话定（十二节）。

## 十一、L 族最后一轮补：(ii) 第 2 轮先取页内主图并比对；循环内第 2 轮 plan 评判；冒烟 r1s3

### 1. 改动（`931ca82`，walker-r1@2026-10-17e，工具层 hash `sha256:49bb6f7e…ba16`）

- **walker 写可修缺口**：`uncompared_pages`——读过、但照片没与给定照片比上的页面（特意取过页内主图的在前，其次文字搜到的页；以图搜副本
  本来带图，不算）。L 族报告里读到的页面没有一张比上、又有这种页面时，报告 INCOMPLETE，status_reason 写成具体动作
  "the picture on <页面> was not compared with the given picture: fetch the main image of <页面> and compare it with the given picture"；
  循环内检查改写 COMPLETED 时（`inloop_actions`）这一条排在最前。S4 保留这条缺口原文。
- **GIA 第 2 轮 Scope**：`_search_gap` 认这一条（kind=page_image），Scope 第一项写 "first fetch the main image of <页面> and compare it
  with the given picture"，其后才是 "open and read …"。
- **第 2 轮必须先做**：`_decide_search_followup` 先 `SEARCH {"source": <页面>, "output_type": "image"}` 取页内主图 → `compare_media` →
  读这一页，再做 Scope 里别的事；这一页的照片没比上（取不到 / 没比 / 不是同一张）报告就 INCOMPLETE 并写明原因；比对结论进 S4 合同项。
- **工具层**：网址来源其实是网页时改取页内主图；页内主图下载网络 / 超时 / 限流类失败按 1、2、4 秒退避共试 3 次（4xx 不重试），
  档案馆存档仍是最后一招。
- **循环内第 2 轮 plan 评判**：`_inloop_scope` 对第 2 轮也跑 `judge_plan`；covered_before 按事后核验同一算法——之前各轮的报告按事后
  参数跑 `verify_report`（同一判官、同一缓存）→ `recalibrate` → 没被污染的 Finding 所引必需证据支持的要求；existing_reports 取 store
  里本轮之前的报告全文；可见 E 与 `reverify._visible_ids` 同一口径。开跑前离线核对 r1s2 三条：covered_before 与事后 units 文件
  逐项相同，第 2 轮 plan 评判的 key 都已在事后缓存里（`claude/_r1s_work/round2_scope_check.py`）。
- 测试：页内主图缺口 → Scope 第一项、in-loop 动作顺序、Scope 解析、下载重试（4xx 不重试）；`test_r1_walker_20261017.py` 22 条 +
  `test_r0plus` 12 条 + search 两个文件，共 90 条全过。

### 2. 冒烟 r1s3（重跑 l623、l101、l274；l208、e92 沿用 r1s2）

| 案例 | 第 1 轮 | 第 2 轮（Scope 第一项 = 取页内主图并比对） | 循环内 / 事后 / 桶 | 验收 |
|---|---|---|---|---|
| **l623** | L4b 找到 BuzzFeed（Kansas / Topeka）；L6a 取图 NETWORK_ERROR（3 次重试 + 存档都没取到）；报告 COMPLETED → 循环内改 INCOMPLETE，缺口 = 取 BuzzFeed 页内主图并比对 | **S2f_open_page_image** 取到 og:image → **S2f_compare** 0.50%（同一张，右缘 x 0.94–1.00 有局部差异）→ 读 BuzzFeed 页 → 再读 3 页 → 报告 **COMPLETED**（"照片与 BuzzFeed 的 Capitol Meemaw 是同一张，BuzzFeed 写明在 Kansas，声明的国会大厦地点不对"） | W2：循环内 judge_targets REQ1–2、判 COMPLETED；事后 positive、status_should_be COMPLETED，**key 相同**；**accepted** | ✅ |
| l101 | 读页、L6 比以图搜副本（同一张，博客页不写地点）；报告 INCOMPLETE，缺口 = 取 travelchardham / jagran 两页的页内主图并比对 | 两页都取到 first_large_img → 比对 **80% / 68% 不同**（不是同一张）→ 读两页 → 再读 2 页 → 报告 **INCOMPLETE**（"两页的照片都不是这张"） | W1 / W2 held（答案层 GT 检查：没有 Kolkata） | ✅ 按 Scope 比了、如实 INCOMPLETE |
| l274 | L4b 找到 Kremlin 线索；Daily Mail og:image 比对 75% 不同；报告 INCOMPLETE，缺口 = 取 Moscow Times 页内主图并比对 | 取到 og:image → 比对 **78% 不同** → 读页 → 再读 2 页 → 报告 **INCOMPLETE**（"Moscow Times 的主图不是这张"） | W2 accepted（INCOMPLETE）；W1 rejected | ✅ 按 Scope 比了、如实 INCOMPLETE |

- **事后 status_rewrite 改写 0**（r1s3）。第 2 轮报告里经循环内检查的只有 l623 W2（COMPLETED）：循环内与事后同一范围（plan 评判
  judge_targets REQ1–2）、同一 key（`inloop_key_check.py`：e2fee720… = e2fee720…），判定相同。l623 W1 的 key 不同是因为它被循环内改写了
  （存进 store 的是 INCOMPLETE 报告），符合预期。l101 / l274 的第 2 轮报告本来就是 INCOMPLETE，循环内不核。
- 第 2 轮派工 3（三条都派了）；答案 positive 0/3 → 重写后 1/3（l623）。
- **L 族可回答 accepted 里的 COMPLETED：l208（r1s2）+ l623（r1s3）= 2/4**，达到 r1s2 时定的 ≥ 2/4。
- 页内主图（r1s3）：第 1 轮 L6a 2 次（l274 取到 og:image、不是同一张；l623 网络失败），第 2 轮 S2f 4 次全取到（l623 og:image 同一张；
  l101 first_large_img ×2、l274 og:image，都不是同一张）。
- 花费：OpenRouter 约 **$0.44**；Serper 0 点（余额仍 2422）。

### 3. l101 / l274 的归因：引擎召回不到

两条在三轮冒烟（r1s2 三遍 + r1s3）里，以图搜（Yandex）与文字搜都没有一页"载有这张照片并写明拍摄地"：l101（GT Kolkata）以图搜到的
副本只有不写地点的博客，文字搜到的铁路贫民窟 / 海尔德瓦尼车站页的照片都不是这张；l274（GT Kremlin）文字搜到的克里姆林宫 / 普京宫殿
报道用的是别的照片。walker 该做的动作都做了（参考图比对、以图搜、必做文字搜、页内主图比对、第 2 轮按 Scope 补比对），报告如实
INCOMPLETE——**归"引擎召回不到"**，不是 walker 或核验器的问题。

**建议第 3 个对话建卷时把"可回答"定义为：GT 可回答，且引擎前 N 条召回（以图搜 + 文字搜）里有载图页**（载有这张照片、并写明
GT 那一点的页面）。按这个定义 l101 / l274 当前算不可回答，可回答 L 族就是 l208、l623（2/2）。

## 十二、要负责人定（第 3 个对话）

1. **"可回答"的新定义**（十一节 3）：GT 可回答且引擎前 N 召回里有载图页；N 取多少、以图搜和文字搜各算几条，建卷时定。
2. **页内主图的档案馆存档时间点**：现在用 `web/2id_/`（最近一次存档），按负责人 10-17 定全量前不动；是否改成按页面日期取存档，留给
   第 3 个对话。
3. 打 `r1-frozen` 时把 oracle_generator 的槽表切到 r1_freeze（负责人已定，第 3 个对话做）；E 族"再搜更早副本"全量后看答案层 GT 检查再定。
