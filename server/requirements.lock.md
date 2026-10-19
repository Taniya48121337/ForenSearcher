# 服务器环境锁定（Vast 1×H200，2026-10-19）

`pip freeze` 于 2026-10-19 在 E2 基线所在的 Vast 实例上导出。换机器重建时照这两份装，再按下面三节补上 pip 装不出来的东西。

| 文件 | 环境 | 用途 |
|---|---|---|
| `requirements.lock.txt` | `/venv/main`（镜像 vastai/pytorch_cuda-12.8.1-auto 自带，Python 3.12.15） | agent / 工具 / 评分 / 训练（`training/sft_lora.py`）。项目本身以 `pip install -e ".[free-search,asr,semantic,ocr,test]"` 装（freeze 时用 `--exclude-editable` 去掉了那一行） |
| `requirements.vllm.lock.txt` | `/workspace/venvs/vllm`（uv venv，Python 3.12） | 只跑 vLLM 学生端点 |

## 1. 关键版本与为什么

- `/venv/main`：**torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / triton 3.6.0 是镜像自带的，不重装**；装其他包时用约束文件钉住
  （`/venv/main/constraints-torch.txt`：上面三行 + `opencv-python<5`）。transformers **5.3.0**、peft **0.18.1** 与本机训练 / 预扫一致。
- `flash-linear-attention 0.5.2`（含 `fla-core`）与 `causal_conv1d 1.7.0`：Qwen3.5/3.8 线性注意力层的快路径（`modeling_qwen3_5.is_fast_path_available = True`）。
  causal-conv1d 没有 torch 2.11 的现成 wheel，要用 nvcc 12.8 从源码编译（`MAX_JOBS=32`，约 4 分钟）。
- opencv：rapidocr-onnxruntime 会拉进 opencv-python 5.x 盖掉 cv2；项目约束 headless < 5，所以 `opencv-python` 与 `opencv-python-headless` 都钉 **4.14.0.94**。
- vLLM 环境：vllm **0.32.0** 要 torch **2.13.0**（PyPI 默认构建是 CUDA 13.0，`nvidia-cuda-runtime 13.0.96`）、transformers 5.19.0。
  **CUDA 12.8 的 vLLM 不存在**（0.25–0.32 只发 cu129 / cu130；PyTorch 官方 cu128 只到 2.11）；驱动 595.99 支持到 CUDA 13.2，能跑。
  训练端 transformers 5.3.0 与推理端 5.19.0 对 sft0 的 20 行（HI 10 + GIA 10，最多 8 张图）编码逐字节相同（`server/23_tokenizer_parity.py`）。

## 2. site-packages 里的手改（换机器会丢，必须重做）

**rapidocr 的 onnxruntime 线程数**（2026-10-19 16:40，E2 基线中途，负责人定）：
`/venv/main/lib/python3.12/site-packages/rapidocr_onnxruntime/config.yaml` 第 13–14 行

```
    intra_op_num_threads: &intra_nums -1    →    intra_op_num_threads: &intra_nums 2
    inter_op_num_threads: &inter_nums -1    →    inter_op_num_threads: &inter_nums 1
```

原文件备份在服务器 `/workspace/backup/rapidocr_config.yaml.orig`。重做：

```bash
d=$(python -c "import rapidocr_onnxruntime,os;print(os.path.dirname(rapidocr_onnxruntime.__file__))")
sed -i 's/intra_op_num_threads: &intra_nums -1/intra_op_num_threads: \&intra_nums 2/; s/inter_op_num_threads: &inter_nums -1/inter_op_num_threads: \&inter_nums 1/' "$d/config.yaml"
```

为什么改这里：项目代码 `inspect/backends/local.py` 用 `RapidOCR()` 不传参数，线程数只能从包里的 config.yaml 取（-1 = 全部核）；
12 个并发评测进程各开满 128 核的 onnxruntime 线程池，CPU 争抢让单 run 从约 11 分钟拖到 40–50 分钟。只影响速度、不影响输出，不进冻结口径；
`experiments/eval/r0/eval_config.json` 的 `harness.thread_caps` 与 changelog 记着，训前训后同一组值。

同一组线程上限里另外三项是环境变量，在评测启动脚本里设（不改 site-packages）：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`
（torch intra-op 随 OMP = 4，faster-whisper / ctranslate2 同）。

## 3. 不在 pip 里的东西

- 工具模型：`/workspace/models/inspect/{faster-whisper-small,clip-vit-base-patch32,grounding-dino-tiny}`（`server/02_download_models.sh`），
  FaceNet `20180402-114759-vggface2.pt` 在 torch hub 默认目录 `/root/.cache/torch/hub/checkpoints/`。
- Node 22.23.3（`/opt/nvm`）；Claude Code 2.1.303（`claude.ai/install.sh`，`/usr/local/bin/claude` 链接，订阅登录；评委 / dev_eval 核验器用）。
- ffmpeg 6.1.1、git-lfs 3.4.1、tmux 3.4、rsync 3.2.7（镜像自带）。
