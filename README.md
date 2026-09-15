# ForenSearcher

一个面向"视频 / 多模态事实核查"的调查型 AI 系统。它不是看一眼就猜真假，而是像人工调查员那样：把一条说法拆成几个能查的小问题，去网上找来源、下载对比、抽帧、听音频，把找到的证据一条条记下来，最后给出每句话都能追溯到证据的结论。

系统里只有两个需要训练的模型角色：**GIA**（项目经理，决定这一轮查什么、证据够不够）和 **HI**（调查员，在一个独立的工作区里用工具查一个具体问题）。完整的设计说明见 [DESIGN.md](DESIGN.md)，项目是怎么一步步走到今天的见 [docs/README.md](docs/README.md)。

> 当前状态（2026-09-15）：框架代码能跑通真实任务；SFT 训练数据构造试点已完成 826 行候选；**尚未做过任何一次模型训练，也没有任何对比实验结果。** 详见 `docs/05_项目审查/`。

---

## 目录结构

```
ForenSearcher/
├── README.md            ← 你正在读的这个：是什么、怎么装、怎么跑
├── DESIGN.md            ← 系统设计定案 + 22 条"不要再引入"的设计 + 已定决策 + 自我发展循环主线
├── pyproject.toml       ← Python 包定义与依赖
├── docs/                ← 所有文档，按项目发展阶段编号排列（见 docs/README.md）
├── configs/             ← 同一份代码、两份配置：local.toml（5090 调试）/ server.toml（8×A100 训练）
├── src/foren_searcher/  ← 全部源代码（2026-09-15 按角色拆包）
│   ├── contracts.py       数据结构：Evidence / Finding / Report / GIA、HI 的决策格式
│   ├── config.py          读取 configs/*.toml（模型名、路径、运行限制、LoRA 超参）
│   ├── gia/               GIA 一侧：global_view（全局视图）、runner（一次决策）、rounds（一轮并行 Workspace）、runtime（整个调查）
│   ├── hi/                HI 一侧：workspace（单个工作区的循环与工具执行）、memory（证据视图）
│   ├── store/             两个角色唯一共享的东西：global_store（SQLite + 文件）、records（记录读取与卡片）、resolver（E→文件）
│   ├── search/            SEARCH 工具（搜索引擎、打开网页/PDF/视频、反向搜图）
│   ├── inspect/           INSPECT 工具（抽帧、OCR、ASR、裁剪、画面/音频对齐）
│   ├── openrouter.py      通过 OpenRouter 调用云端模型的适配器
│   ├── local_models.py    本地 Transformers 模型适配器（Qwen3 纯文本时代的代码，待按 Qwen3.5 视觉模型重写）
│   ├── harness1.py        【已弃用】接私有分叉 Harness-1 的适配层，仅兼容旧参数，之后删除
│   ├── memory.py / orchestration.py / workspace.py / evidence_resolver.py  兼容壳，只转发到上面三个包
│   ├── demo.py            命令行入口：check / run
│   └── web_demo.py, web/  演示网站（与研究主线无关，可选）
├── tests/               ← 约 90 个测试文件
├── experiments/sft_data/← SFT 训练数据构造试点（脚本、rubric、试验文档；大体积语料不进 git）
├── external/            ← 外部数据集（VCSL、VidDiffBench），不进 git
├── scripts/             ← Windows 下的快捷启动脚本（.cmd）
└── _trash_20260915/     ← 整理时移走的运行残留，确认无用后整个删掉即可（不进 git）
```

---

## 安装

需要 Python 3.12。在项目根目录打开终端：

```
pip install -e ".[test]"
```

可选的功能组（按需安装）：

```
pip install -e ".[web]"            # 演示网站
pip install -e ".[free-search]"    # 免费搜索后端（ddgs）
pip install -e ".[local-models]"   # 本地 Transformers 模型
```

不再需要 Harness-1。`--harness-root` / `FOREN_HARNESS_ROOT` 仍被接受，但只会启用一个已弃用的兼容适配层并给出警告；不给就直接用本项目的 `WorkspaceHarness`。

## 选择配置

本地和服务器用同一份代码、不同的配置文件。默认读取 `configs/local.toml`；到服务器上：

```
export FOREN_CONFIG=configs/server.toml        # Linux
set FOREN_CONFIG=configs\server.toml           # Windows
```

配置里有四段：`[models]`（学生/教师模型名、权重目录）、`[paths]`（输出、数据集位置）、`[runtime]`（轮数、步数、超时）、`[training]`（LoRA 秩、批大小、序列长度）。字段写错会在启动时报出具体是哪一项。

## 配置模型

当前主线用 OpenRouter 上的云端模型。先设置 API Key（会保存到 `.local/openrouter.json`，这个目录已被 git 忽略，不会泄露）：

```
python -m foren_searcher.openrouter_settings
```

## 跑一次调查

```
# 先检查环境是否齐全
python -m foren_searcher.demo check --model-backend openrouter --search ddgs

# 跑一个任务
python -m foren_searcher.demo run \
    --task "这段视频是否拍摄于 2026 年成都某事故？" \
    --media path/to/video.mp4 \
    --output-dir outputs/my-first-run \
```

结果写在 `--output-dir` 下：`result.json`（最终答案与证据引用）、`report.md`（可读报告）、`trajectory.jsonl`（每一步的完整轨迹）。

## 跑测试

```
pytest tests
```

## 演示网站（可选）

```
python -m foren_searcher.web_demo --open
```

浏览器打开 `http://127.0.0.1:8765`。

---

## 接下来该做什么

2026-09-15 六个待决问题已全部拍板，并确立了"证据锚定的自我发展循环"为主线（DESIGN.md 第 7、8、9 节；讨论理由见 `docs/05_项目审查/设计讨论记录_20260915.md`）。按顺序：

1. ~~代码拆包为 `foren_searcher/hi/`、`gia/`、`store/`；`harness1.py` 归档，`demo.py` 去掉 `--harness-root` 硬依赖；路径和模型名全部配置化~~（2026-09-15 已完成：1418 项测试通过，与拆包前基线一致）。
2. 本地用 Qwen3.5-4B 跑通"构造数据 → 训一步 → 存 LoRA → 加载推理 → 打分"链路。
3. 服务器（8×A100）上用 Qwen3.5-9B 做第一次真实 SFT，在留出集上看有没有可测差异。这是从"数据构造试验"跨到"研究"的门槛。
4. 移植 Harness-1 的四个机制进 `WorkspaceHarness`（重要性分级精选集、内容指纹去重、句子级压缩、离线核验器），每个都要证明确实触发并在留出集上有用。
5. 按 `docs/06_数据集候选表_20260915.md` 每个能力维度挑一个训练集，先写适配器说明再下载。

在第 3 步有结果之前，不要再往 `experiments/sft_data` 加新案例、新 rubric 版本或新诊断脚本。

## 旧版 README

之前 68KB 的 README 记录了大量实现细节和历次修复说明，已移到 `docs/03_框架实现/README_旧版_详细技术说明.md`，需要查具体行为时再去翻。
