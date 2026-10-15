# walker 轨迹生产：服务器（8×H20 96GB）

一个入口：`server/16_walker_prod.sh`。只跑生产，不跑自进化循环。循环那套布局在 `11_loop_layout.sh`，按 A100 写的，不受影响。

## 0. 前提

- 已按 `README_服务器跑baseline.md` 完成：第 1 步建环境、第 3 步数据，`server/env.sh` 里的 `FS_PROXY` / `FS_HF_ENDPOINT` 也已填好。
- 检索要出外网：DuckDuckGo、以图搜图、网页、yt-dlp。B 族给定媒体不检索，不受网络影响。

## 1. 起教师

```bash
bash server/16_walker_prod.sh up          # 默认 397b：Qwen3.5-397B-A17B-FP8，8 卡 TP8，显存 80%；工具和 GPU 7 共用
bash server/16_walker_prod.sh up 122b     # 不推荐（见下）：Qwen3.5-122B-A10B-FP8，GPU 0-3，TP4
bash server/16_walker_prod.sh status
```

- **为什么是 397B（10-15，wk13 / wk14 对照，Sonnet 核验）：**
  - 397B 每一列都不差于 122B：报告 positive 7 对 4，HI accepted 207 对 120，墙钟中位数 403 s 对 823 s，30 个 run 全部跑成（122B 有 3 个没跑成）。
  - 但在 18 个检索类案例上，报告严格通过只有 2，Codex（wk12）是 8。按 `specs/r0_freeze.json` 的判定规则，Qwen 不能当主教师，只能作备用。所以服务器产出的数据记作 397B 教师，和 r0 的 Codex 数据分批，不混在一起。
  - 27B（wk15）没跑。
- **教师权重：** 第一次 `up` 时下载到 `.models/llm/`。H20 是 Hopper 架构，FP8 直接算。
- **核验器不在服务器上。** 冻结的核验器是 Sonnet（Claude CLI），服务器上没有。Qwen 122B 当核验器和 Sonnet 的入选一致性 κ≈0，见 `walker_wk13_judges.md`。所以服务器只产轨迹，核验、重写、导出回本机做（第 5 步）。确实要在服务器上核验时，设 `FS_PROD_VERIFIER_ENDPOINT` / `FS_PROD_VERIFIER_NAME`；这样产出的数据不算 r0 口径。

## 2. 冒烟一条

```bash
bash server/16_walker_prod.sh smoke       # videosham_4126，不核验，2–5 分钟
```

## 3. 建案卷（放量）

```bash
bash server/16_walker_prod.sh cases ws1 --pool all --n-a 60 --n-c-fake 80 --n-c-real 40 --n-e 60 --n-g360-fake 60 --n-g360-real 30
```

- 清单写到 `experiments/sft_data/walker_ws1_cases.json`。
- A / C 族的候选扩到 FMNV 全集：真新闻按编号一分为二，一半做找原片，一半做真标题核查；CD 类全部做假标题核查。已下载的视频排在前面。
- EVID test 里出现过的视频编号会跳过，防止泄漏。
- 要下载原片：yt-dlp 走 `FS_PROXY`。

## 4. 生产一轮（只产轨迹）

```bash
FS_PROD_JOBS=8 bash server/16_walker_prod.sh run ws1
bash server/16_walker_prod.sh bundle ws1      # → walker_ws1_bundle.tgz，含 runs/*_ws1_1、cases/、清单、run_manifests/ws1.jsonl
```

- **一轮包含：** 9 个 B 族案例，加上清单里的案例，各跑一条轨迹，`FS_PROD_JOBS` 条并行。默认带 `--no-verify`。
- **断点续跑：** 中途断了重跑同一条命令，已完成的 run 会跳过。
- **自动停止：** 教师连续两次失败（429 / 402 / 槽失败）就停。

## 5. 回本机核验、重写、导出

1. 把 `walker_ws1_bundle.tgz` 解到本机 ForenSearcher 根目录。
2. 本机运行：

   ```
   python scripts\walker_run.py d1 --round ws1 --case-list walker_ws1_cases.json --teacher openrouter --teacher-model qwen/qwen3.5-397b-a17b
   ```

   这条命令会：
   - 跳过已完成的 run；
   - 用 Sonnet 核验；
   - 只按证据重写答案。重写调 OpenRouter 上的同一个 397B，和服务器教师是同一个模型，费用很低；
   - 再核验一次，然后导出。

   半截的 run 会在本机用 OpenRouter 重跑。
3. 第一次放量前，先拿 2–3 个 run 走一遍上面这条路，确认案卷里的媒体路径在本机也找得到。

## 6. 打包训练集

```bash
bash server/16_walker_prod.sh pack p1 ws1 ws2     # 在本机导出之后做（本机没有 bash 时直接 python scripts\pack_sft.py）
python training/sft_lora.py --config <把 export_dir 指到 experiments/sft_data/sft_pack/p1 的配置>
```

`sft_pack/p1/{hi,gia}_accepted.jsonl` 和 `images/` 就是训练输入，`manifest.json` 里有行数、权重、族 / 数据集 / 教师分布和 token 分位数。

## 进训练集的规则（10-14 晚起）

- **写答案这一步不用 GT。** 首稿就是答案。核验后只把"所引证据不支持 / 引错 / 说过头"这类意见交给教师重写；GT 类意见（"应有的结论 'misleading'"等）不给，wk11 / wk12 证明它会教教师删掉正确内容来躲检查。
- **报告 / 答案严格入选。** positive 而且没有 major / critical 错误才收；带错的记 held，原因写错误类型。
- **教师输入 = 学生输入。**
  - 渲染超预算时，会先压缩证据卡片清单。以前长视频的 GIA 状态压不下，教师看的是原始视图，导出时整行丢掉。
  - 附图前放说明，和学生看到的排法一致。
  - walker 给教师的提示不再抄写学生看不到的转写原文。
- **GIA 第 1 轮照抄总任务的派工行默认不进**（10-15 质量门 C3，`plan_copies_task` 记 held；export `--plan-copy downweight` 时改为 ×0.3 进）。
- **HI 动作的 rationale 由教师按当步状态写（S0 槽）。** 不合格的（过长、引了状态里没有的编号）退回模板。
