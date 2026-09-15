# ForenSearcher

一个面向"视频 / 多模态事实核查"的调查型 AI 系统。它不是看一眼就猜真假，而是像人工调查员那样：把一条说法拆成几个能查的小问题，去网上找来源、下载对比、抽帧、听音频，把找到的证据一条条记下来，最后给出每句话都能追溯到证据的结论。

系统里只有两个需要训练的模型角色：**GIA**（项目经理，决定这一轮查什么、证据够不够）和 **HI**（调查员，在一个独立的工作区里用工具查一个具体问题）。完整的设计说明见 [DESIGN.md](DESIGN.md)，项目是怎么一步步走到今天的见 [docs/README.md](docs/README.md)。

> 当前状态（2026-09-15）：框架代码能跑通真实任务；SFT 训练数据构造试点已完成 826 行候选；**尚未做过任何一次模型训练，也没有任何对比实验结果。** 详见 `docs/05_项目审查/`。

---

## 目录结构

```
ForenSearcher/
├── README.md            ← 你正在读的这个：是什么、怎么装、怎么跑
├── DESIGN.md            ← 系统设计定案 + 22 条"不要再引入"的设计 + 6 个待决定问题
├── pyproject.toml       ← Python 包定义与依赖
├── docs/                ← 所有文档，按项目发展阶段编号排列（见 docs/README.md）
├── src/foren_searcher/  ← 全部源代码
│   ├── contracts.py       数据结构：Evidence / Finding / Report / GIA、HI 的决策格式
│   ├── orchestration.py   主循环：GIA 派任务 → 多个 HI 并行调查 → GIA 综合
│   ├── workspace.py       单个 HI 工作区内部的循环
│   ├── memory.py          HI 看到的"局部工作记忆"怎么组织
│   ├── store.py           持久化：SQLite + 文件目录，所有证据都存这里
│   ├── search/            SEARCH 工具（搜索引擎、打开网页/PDF/视频、反向搜图）
│   ├── inspect/           INSPECT 工具（抽帧、OCR、ASR、裁剪、画面/音频对齐）
│   ├── openrouter.py      通过 OpenRouter 调用云端模型的适配器（当前主线）
│   ├── local_models.py    本地 Transformers 模型适配器（实验性，未来可能重写）
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

运行调查还需要一个外部目录：**Harness-1** 仓库（论文 arXiv 2606.02373 的运行框架）。用 `--harness-root <路径>` 或环境变量 `FOREN_HARNESS_ROOT` 指向它。这个依赖是否保留是待决定问题之一（见 DESIGN.md 第 7 节）。

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
    --harness-root <Harness-1 路径>
```

结果写在 `--output-dir` 下：`result.json`（最终答案与证据引用）、`report.md`（可读报告）、`trajectory.jsonl`（每一步的完整轨迹）。

## 跑测试

```
pytest tests
```

## 演示网站（可选）

```
python -m foren_searcher.web_demo --harness-root <Harness-1 路径> --open
```

浏览器打开 `http://127.0.0.1:8765`。

---

## 接下来该做什么

按优先级：

1. 对 DESIGN.md 第 7 节的六个待决定问题逐一拍板（模型口径、HI 是否多模态、Harness-1 去留、消融清单、新任务适应、新颖性定位）。
2. 把 `experiments/sft_data` 里仍用旧 rubric（r2）评审的 27 个 run 统一用 r6 重评；对 2–3 个短案例做 3 次重复采样，确认分差不是噪声；冻结一个真正未见的保留集。
3. 在开发集（340 行）上做一次学生模型的 LoRA SFT，在保留集上看有没有可测差异。这是从"数据构造试验"跨到"研究"的门槛。

在第 3 步有结果之前，不要再往 `experiments/sft_data` 加新案例、新 rubric 版本或新诊断脚本。

## 旧版 README

之前 68KB 的 README 记录了大量实现细节和历次修复说明，已移到 `docs/03_框架实现/README_旧版_详细技术说明.md`，需要查具体行为时再去翻。
