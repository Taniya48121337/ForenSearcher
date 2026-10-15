# 在服务器上跑 baseline（8×A100 80G，Ubuntu，纯命令行）

本文档对应的被测系统与本机 EVID 全量 evidC27G 相同：
- 模型：Qwen3.8-27B，关思考；
- DETECT：v2.4；
- 规格 hi@0.7.0 / gia@0.8.0，3 轮，compact_16k，eval 兜底，屏蔽核查网站；
- 评委：Claude Sonnet。

和本机唯一的区别是模型的服务方式：本机用 Ollama Q4，服务器用 vLLM bf16。服务器的数字是一套新的 baseline，与本机数字只能大致对照。

## 要跑的四组

| 组 | 数据 | DETECT | 轮次名 | 样本数 |
|---|---|---|---|---|
| evid_base | EVID-Bench 131–222 | 关 | srvF27 | 91 |
| evid_detect | EVID-Bench 131–222 | 开 | srvG27 | 91 |
| g360 | GroundLie360 | 开 | gsrv27 | `FS_BENCH_N`（默认 200） |
| fmnv | FMNV | 开 | fsrv27 | `FS_BENCH_N`（默认 200） |

- 可选两组 `g360_base`、`fmnv_base`：同样的样本，不开 DETECT。
- GroundLie360 和 FMNV 的 200 个样本一半真、一半假，假样本平均分到各个类型。
- 时间估计：16 个 run 并行，单个 run 10–15 分钟。四组共约 580 个 run，6–9 小时。

## GPU 分配

默认如下，可在 `server/env.sh` 里改：

| GPU | 用途 |
|---|---|
| 0–5 | vLLM：每张卡一个 27B bf16 副本（数据并行） |
| 6–7 | 智能体的工具模型：Whisper 语音转写、CLIP 按文字找帧、GroundingDINO 图内定位、FaceNet 人脸比对；16 个并行 run 共用 |

---

## 准备：本机打包，传到服务器

**在 Windows 本机**双击 `scripts\_pack_server.cmd`。

窗口最后会显示类似内容：

```
packed 6xx files (... MB before compression): ...
FaceNet weights: included (...)
archive: E:\Codex\workspaces\ChatGPT\ForenSearcher\dist\ForenSearcher_server_20261012_1530.tar.gz  (1xx MB)
sha256:  ...
```

压缩包里有代码、规格、配置、测试、服务器脚本、EVID 案卷和清单、GroundLie360 / FMNV 标注、FaceNet 权重。不含视频，也不含大模型权重，这两样由服务器自己下载。

**在本机 PowerShell** 里上传，把用户名、地址、文件名换成你的：

```powershell
scp E:\Codex\workspaces\ChatGPT\ForenSearcher\dist\ForenSearcher_server_20261012_1530.tar.gz 用户名@服务器地址:~/
```

**在服务器上**解包：

```bash
cd ~ && tar -xzf ForenSearcher_server_*.tar.gz && cd ForenSearcher
sha256sum ~/ForenSearcher_server_*.tar.gz   # 和本机窗口里的 sha256 对一下
ls server/                                   # 应看到 00_check_system.sh … 08_export_for_scoring.sh
```

**改配置**：

```bash
nano server/env.sh
```

只需看"network"一段：
- 服务器上网要走代理时，填 `FS_PROXY`，例如 `http://10.0.0.5:7890`；
- 连不上 huggingface.co 时，填 `FS_HF_ENDPOINT=https://hf-mirror.com`；
- pip 慢时，填 `FS_PIP_INDEX`，例如清华源。

其他设置一般不用动：GPU 分配、并行数 `FS_JOBS=16`、每个数据集样本数 `FS_BENCH_N=200`。

> 下面所有命令都在 `~/ForenSearcher` 目录里执行；第 N 步对应脚本 `server/0N_*.sh`。长时间的步骤建议在 `tmux` 里跑：用 `tmux new -s fs` 开会话，`Ctrl-b d` 挂起，`tmux attach -t fs` 回来。

---

## 第 0 步：检查服务器（只读，约 1 分钟）

```bash
bash server/00_check_system.sh
```

它会逐项打印 `[ OK ]` / `[WARN]` / `[FAIL]`：
- 系统、CPU、内存、磁盘（建议 ≥ 400 GB 空闲）；
- 8 张 GPU 和驱动支持的 CUDA 版本（vLLM 需要 ≥ 12.4）；
- python 是否 ≥ 3.11，venv、ffmpeg、node、claude CLI 是否在；
- 网络连通性：PyPI、Hugging Face、GitHub、YouTube、DuckDuckGo、Wikipedia。

最后一行是 `Step 0 done: nothing blocking` 才往下走。

常见 FAIL 的处理：
- `FS_PYTHON_BIN ... < 3.11`：
  - `sudo apt install python3.11 python3.11-venv`，或用 conda 装一个 3.11 / 3.12；
  - 然后在 env.sh 里设 `FS_PYTHON_BIN=python3.11`，或填 conda 环境里 python 的完整路径。
- `YouTube NOT reachable`：
  - 必须解决（设 `FS_PROXY`）。EVID 和 FMNV 都要到 YouTube 上找原片、下原片，否则分数会明显变低。
- `Hugging Face NOT reachable`：设 `FS_HF_ENDPOINT=https://hf-mirror.com`。

## 第 1 步：装 Python 环境（约 15–25 分钟）

```bash
bash server/01_setup_env.sh
```

会建两个环境：
- `.venv-agent`：智能体、工具模型、评分；
- `.venv-vllm`：vLLM 服务，单独放是因为它自带固定版本的 torch。

完成时会列出每个包的版本，并显示 `torch CUDA build 12.x, GPUs visible 8`，最后一行是 `Step 1 done`。

pip 失败时会打印日志最后 25 行，完整日志在 `server_logs/pip_agent.log` 和 `pip_vllm.log`。多数是网络问题，设好 `FS_PIP_INDEX` 或 `FS_PROXY` 后重跑即可。已装好的包会跳过。

## 第 2 步：下载模型（约 55 GB；网速 50 MB/s 时约 20 分钟）

```bash
bash server/02_download_models.sh
```

每个模型下完会打印一行，例如：

```
[ OK ] LLM: 27 files, 55.6 GB, weight files: 12  (1100 s)
[ OK ] ASR: 6 files, 0.49 GB ...
[ OK ] frame search: ...
[ OK ] region search: ...
[ OK ] already present / copied from the package: .../20180402-114759-vggface2.pt
```

中断了直接重跑，会接着下。

## 第 3 步：下载数据集

### 3a. EVID-Bench（约 10 GB）

这个数据集要先在网页上同意条款，只需做一次：
1. 用你的 Hugging Face 账号打开 https://huggingface.co/datasets/Kirito-Lab/EVID-Bench ，点 "Agree and access"。
2. 在 https://huggingface.co/settings/tokens 新建一个 Read token。
3. 在服务器上登录：

   ```bash
   source server/env.sh
   $FS_ENV_AGENT/bin/hf auth login          # 粘贴 token；没有 hf 命令就用：$FS_ENV_AGENT/bin/huggingface-cli login
   # 或者：export HF_TOKEN=hf_xxxxxxxx
   ```

然后执行：

```bash
bash server/03_download_data.sh
```

EVID 部分成功的输出：

```
[ OK ] manifest ids: package 222, Hugging Face 222, same=True
[ OK ] EVID videos present 222/222
```

### 3b. GroundLie360 与 FMNV 的视频

这两个集的视频只放在网盘上，命令行没法直接下。两种办法，任选一种。

**办法一（推荐）：在本机浏览器下载，再传到服务器**

1. 本机下载：
   - GroundLie360：[OneDrive](https://1drv.ms/f/c/4900809cf598ca24/IgBhYxreDDVKS5aASa3ypGoaAe52eP_P2ORzX0yOEh62nCo?e=C7wzgZ) 或 [百度网盘](https://pan.baidu.com/s/18gVXt0GSg1--SyNm6flpdw?pwd=vmj3)（提取码 vmj3）；
   - FMNV：[百度网盘](https://pan.baidu.com/s/1o-xqR5lyzkEEyU60tkTd9g?pwd=2i94)（提取码 2i94）。
2. 在服务器上建目录：

   ```bash
   mkdir -p plan-harness-1/datasets/GroundLie360/videos plan-harness-1/datasets/FMNV/videos
   ```

3. 在本机 PowerShell 上传（路径换成你下载的位置）。下载得到的是 zip 就直接传 zip：

   ```powershell
   scp -r D:\下载\GroundLie360\* 用户名@服务器地址:~/ForenSearcher/plan-harness-1/datasets/GroundLie360/videos/
   scp -r D:\下载\FMNV\*         用户名@服务器地址:~/ForenSearcher/plan-harness-1/datasets/FMNV/videos/
   ```

4. 回到服务器，有 zip 就解压。文件放在 `videos/` 下的任何子目录都行，程序按"文件名 = 视频编号"去找：

   ```bash
   cd plan-harness-1/datasets/GroundLie360/videos && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; cd ~/ForenSearcher
   cd plan-harness-1/datasets/FMNV/videos         && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done; cd ~/ForenSearcher
   ```

**办法二：在服务器上用 BaiduPCS-Go 下百度网盘**

需要百度账号的 BDUSS（浏览器登录 pan.baidu.com 后，在开发者工具 → Application → Cookies 里找 BDUSS）。

```bash
cd ~ && wget https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/v3.9.7/BaiduPCS-Go-v3.9.7-linux-amd64.zip
unzip BaiduPCS-Go-v3.9.7-linux-amd64.zip && cd BaiduPCS-Go-v3.9.7-linux-amd64
./BaiduPCS-Go login -bduss=你的BDUSS
./BaiduPCS-Go transfer https://pan.baidu.com/s/1o-xqR5lyzkEEyU60tkTd9g 2i94     # FMNV 转存到自己的网盘
./BaiduPCS-Go transfer https://pan.baidu.com/s/18gVXt0GSg1--SyNm6flpdw vmj3     # GroundLie360
./BaiduPCS-Go ls /                                                             # 看转存后的文件夹名
./BaiduPCS-Go download "/转存后的FMNV文件夹" --saveto ~/ForenSearcher/plan-harness-1/datasets/FMNV/videos
./BaiduPCS-Go download "/转存后的GroundLie360文件夹" --saveto ~/ForenSearcher/plan-harness-1/datasets/GroundLie360/videos
```

版本号或命令有出入时，以 `./BaiduPCS-Go --help` 为准。下完同样解压 zip。

**检查**：放好后重跑一次

```bash
bash server/03_download_data.sh
```

应看到：

```
[ OK ] GroundLie360: 20xx video files, 20xx/2044 match an annotation (Real ..., Fake ...)
[ OK ] GroundLie360: decodable 20/20 checked
[ OK ] FMNV: 23xx video files, 23xx/2393 match an annotation ...
```

- `match` 数为 0：文件名不是视频编号。把 `unmatched examples` 那几个文件名发给我，我改匹配规则。
- 只下了一部分也能跑，程序只从已有的视频里抽样。

## 第 4 步：启动模型服务（约 5–10 分钟）

```bash
bash server/04_start_vllm.sh
```

它在后台启动 vLLM，每分钟报一次加载进度。成功时显示：

```
[ OK ] server is up after 4xx s
[ OK ] answer '{"ok": true}' in 0.8 s; no thinking text
```

- 看加载日志：`tail -f server_logs/vllm.log`
- 停止服务：`bash server/04_stop_vllm.sh`
- 服务器重启后，要重新执行这一步。

启动失败时，脚本会打印日志最后 40 行和常见处理方法：
- vLLM 太旧：在 `.venv-vllm` 里 `pip install -U "vllm>=0.17"`。
- 显存不够：把 env.sh 里的 `FS_MAX_MODEL_LEN` 改成 36864。
- 不支持数据并行：把 `FS_VLLM_GPUS` 设成 `0`，只起一个副本，会慢很多，同时把 `FS_JOBS` 降到 4。

## 第 5 步：开跑前的全面测试（约 30–40 分钟）

```bash
bash server/05_preflight.sh
```

每个模块都会真实跑一次，每项打印一行 `[PASS]` / `[WARN]` / `[FAIL]`，附耗时和修复提示：

| 部分 | 测什么 |
|---|---|
| env | python 包与版本；工具 GPU 能否计算、各有多少空闲显存；磁盘；ffmpeg |
| files | 代码、规格、配置齐全；EVID 222 个视频 + 222 个案卷；GroundLie360 / FMNV 匹配到的视频数 |
| tools | INSPECT 三个模型操作已启用；Whisper 在 GPU 上转写一段测试音频；CLIP 认出"红色圆"；GroundingDINO 框出"红色圆"；FaceNet 在 GPU 上加载 |
| llm | 服务列出模型；关思考的 JSON 回答；看图答颜色；一次 8 张图；JSON-schema 约束输出；16 个请求并发的速度 |
| net | 智能体自己的 HTTP 客户端抓网页；YouTube 搜索；YouTube 下载（完整版和 360p 纯画面兜底版各一次）；DuckDuckGo 文字 / 视频搜索；核查网站在屏蔽名单里 |
| detect | 在合成视频对上跑 compare_source（GPU）；用模型服务给一张图写描述 |
| judge | Claude CLI 评委回一道小题 |
| tests | 单元测试：DETECT、数据集接入、下载 |
| e2e | 同时跑 3 个完整调查：EVID video_002（开发段样本，不在测试的 131–222 里）、GroundLie360 1 个、FMNV 1 个。检查是否 FINISH、用了哪些动作、工具模型是否都开着、DETECT 调用和图片描述次数、检索快照是否记录、答案末尾的固定行能否解析；最后用 Sonnet 给 EVID 那个样本打分，确认评分链路通 |

最后打印汇总，并写报告 `server_logs/preflight_report.md`：

```
==== summary: PASS 3x, WARN x, FAIL 0  -> server_logs/preflight_report.md
Ready for step 6 (bash server/06_run_baselines.sh).
```

- **FAIL 必须修好再往下走。** 修完可以只重测那一部分，例如 `bash server/05_preflight.sh --only net`。部分名：env files tools llm net detect judge tests e2e。
- 只想先快速看一遍、不跑完整调查：`bash server/05_preflight.sh --quick`，约 5 分钟。
- 异常的完整堆栈在 `server_logs/preflight_tracebacks.log`。

**评委（judge）这一项**需要 Claude Code CLI：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && source ~/.bashrc
nvm install 20
npm install -g @anthropic-ai/claude-code
claude          # 按提示登录（会给一个链接，在本机浏览器打开并粘回验证码）；或者 export ANTHROPIC_API_KEY=sk-ant-...
claude -p "reply ok" --model sonnet   # 回 ok 即可
```

服务器装不了也没关系：baseline 照样跑，分数到本机打（见第 6 步末尾）。

## 第 6 步：跑 baseline

```bash
tmux new -s fs
bash server/06_run_baselines.sh
```

- 挂起：`Ctrl-b d`；回来：`tmux attach -t fs`。
- 四组依次跑，每组跑完自动重跑一次没完成的样本，再用 Sonnet 评分。
- 屏幕每分钟输出一行进度：

  ```
  [2026-10-13 02:14:00] evid_base: 37/91 done, 52 min elapsed, ETA 76 min | vLLM up | free disk 812 GB
  ```

**中途查看**（另开一个终端）：

```bash
bash server/07_status.sh                      # 每组完成数 + GPU 占用
tail -f server_logs/06_run_baselines.log      # 进度
tail -f server_logs/evid_base_srvF27.runlog   # 某一组的详细输出
```

**中断后续跑**：再执行一次 `bash server/06_run_baselines.sh`，已完成的 run 会跳过。

**只跑某几组**：

```bash
bash server/06_run_baselines.sh --arms g360 fmnv
bash server/06_run_baselines.sh --arms g360_base fmnv_base   # 不开 DETECT 的对照
```

**结果**：
- 总表：`server_logs/baselines_summary.md`，随时可用 `bash server/06_run_baselines.sh --summary-only` 重新生成。内容包括：
  - EVID 两组的点级、视频级准确率，没读过的样本（干净子集），AI 生成类；
  - GroundLie360 / FMNV 的 Acc、Fake F1、macro-F1、类型 F1。
- 逐样本明细：
  - `experiments/evid/scores/srvF27.sonnet.subset.md`（以及 srvG27）；
  - `experiments/bench/g360/scores/gsrv27.md`、`experiments/bench/fmnv/scores/fsrv27.md`。

**服务器上不能用 Claude 评委时**：

```bash
bash server/06_run_baselines.sh --no-score
bash server/08_export_for_scoring.sh
```

第二条命令把答案、配置和轨迹打成一个小包（不含视频），并打印到本机打分的命令。

---

## 出问题时把这些发给我

- 当前步骤的日志 `server_logs/0X_*.log`；
- 第 5 步的 `server_logs/preflight_report.md`，有异常时连同 `preflight_tracebacks.log`；
- 第 6 步的 `server_logs/06_run_baselines.log`，以及出问题那一组的 `*.runlog` 末尾 100 行。
