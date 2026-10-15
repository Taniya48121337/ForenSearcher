# 自进化循环：8×A100 80GB 的 GPU 分配与脚本（2026-10-12）

口径：DESIGN.md 与 configs/server.toml。学生 Qwen3.5-9B，GIA 和 HI 各挂一套 LoRA；教师 Qwen3.5-122B-A10B FP8；训练用核验器选与教师不同家族的本地模型；verifier_ref 再换一个家族。

先跑完 baseline（步骤 0–8），再起循环。baseline 占满 8 张卡：vLLM 在 0–5，工具在 6–7。`04_stop_vllm.sh` 会 `pkill -f "vllm serve"`，把循环的服务也一起停掉，所以两者不要同时跑。

## 1. 原则：按阶段换布局，不按角色固定分卡

一轮里的步骤基本是串行的：采样 → 核验 → 内环 → 用 spec@n+1 造数据 → 训 HI → 门 B → 训 GIA → 门 B → dev_eval。固定分卡的话，训练时教师的 4 张卡闲着，采样时训练卡闲着。所以只用两套布局，每轮切换两次。

| GPU | 布局 A：推理（采样、桥接、核验、门 A、门 B、dev_eval） | 布局 B：外环训练窗口 |
|---|---|---|
| 0–3 | 教师 122B FP8，张量并行 4，端口 8000 | 教师停掉，每卡跑一个 LoRA 训练任务 |
| 4–5 | 学生 9B，两个数据并行副本，端口 8002；一个底座挂所有臂、所有轮的 HI / GIA LoRA | 不变（LoRA 保持加载，门 B 随训随测） |
| 6 | 训练用核验器，端口 8001 | 不变 |
| 7 | 工具（Whisper、CLIP、GroundingDINO、FaceNet、DETECT）；verifier_ref 够小时也放这里，端口 8003 | 不变 |

与 10-12 对话里的初稿相比有一处改动：训练只用 0–3，不用 0–5。这样学生不用重启，已加载的 LoRA 也不会丢，训完一个就能马上过门 B。如果 E1 的臂乘种子超过 4 个，`13_train_queue.sh` 会排队：9B 的 LoRA 在几百行数据上训练时间不长，分两批跑可以接受。

显存粗算：
- 122B FP8：权重约 125GB，4 卡平分每卡约 31GB，剩约 40GB 给 KV。
- 9B bf16：权重约 19GB，剩约 50GB 给 KV。
- 核验器：≤32B 稠密 bf16 单卡能放下。
- 学生只给 2 张卡，是因为调查的瓶颈在搜索、下载、抽帧：本机 baseline 每个 run 要 12–15 分钟。并发量靠 `FS_JOBS` 调。第一轮看 vLLM 日志里的 KV 占用和排队数，学生确实排队时，再考虑把 6 号卡也切给学生（核验器挪到 7 号）。

## 2. 一轮的命令顺序

```bash
bash server/10_loop_setup.sh                       # 只做一次：装 peft，下权重，检查分卡
bash server/11_loop_layout.sh A                    # 教师 / 学生 / 核验器 / (verifier_ref)
ACTOR=teacher bash server/14_run_cases.sh --round r0 --limit 1    # 先 1 个案例
# ① 采样：学生每案例 4 条 + 教师每案例 1 条（算教师率），跑完自动核验
HI_MODEL=hi_r0 GIA_MODEL=gia_r0 bash server/14_run_cases.sh --round r1s --samples 4
ACTOR=teacher bash server/14_run_cases.sh --round r1t
# ② 诊断 / 路由（CPU）  ③ 内环子步（门 A 的执行者是学生，教师只做提议者）  ④ 用 spec@n+1 造外环数据（自采 + 教师桥接）
bash server/11_loop_layout.sh B                    # 停教师，空出 0–3
bash server/13_train_queue.sh jobs/r1_hi.txt       # 各臂 × 各种子并行
bash server/12_add_lora.sh hi_r1_full_s1 outputs/lora/r1/full/hi_s1/final     # 不重启加载，门 B-HI
# ⑤ HI@n+1 跑出真实报告，搭 GIA 离线状态（只用学生，教师可以不开）
bash server/13_train_queue.sh jobs/r1_gia.txt ; bash server/12_add_lora.sh gia_r1_full_s1 ...   # 门 B-GIA
bash server/11_loop_layout.sh A                    # dev_eval，以及下一轮
```

教师每轮停启两次，122B 每次加载估计 5–10 分钟，比让 4 张卡空几个小时划算。

## 3. 脚本

| 文件 | 作用 |
|---|---|
| `loop_env.sh` | 分卡、端口、模型名、LoRA 上限的唯一出处（先 source env.sh 再覆盖）；`FOREN_CONFIG=configs/loop_server.toml` |
| `10_loop_setup.sh` | 检查分卡不重叠、训练槽 ⊆ 教师卡；把 peft 装进 .venv-agent；下载教师、学生、核验器权重；确认 transformers 认得学生架构 |
| `11_loop_layout.sh A\|B\|status\|stop\|restart-student` | 起停服务。发现端口被别的服务（例如 baseline 的 vLLM）占着就停下报错；教师卡上还有训练任务时不起教师 |
| `12_add_lora.sh <名字> <目录>` | 登记到 `server_logs/loras.tsv`，再通过 `/v1/load_lora_adapter` 不重启加载。学生重启时按登记表用 `--lora-modules` 全部挂上 |
| `13_train_queue.sh <任务文件>` | 每行一组 `sft_lora.py` 参数，按 `FS_TRAIN_SLOTS` 每卡一个任务并行，多出来的排队；卡上有占用就拒绝开跑；汇总写 `server_logs/train/queue_*.tsv` |
| `14_run_cases.sh` | `ACTOR=teacher\|student`；学生按角色选 LoRA（`HI_MODEL` / `GIA_MODEL`）；agent 进程只看到工具卡；跑完用 6 号卡的核验器核验 |

配套代码改动：
- `training/sft_lora.py`：默认开梯度检查点。`device_map=auto` 且进程看得到多张卡时直接退出（否则 HF 会把模型按层摊到所有卡上串行跑）。新增 `--seed`；`train_summary.json` 记 `seed`、`cuda_visible_devices` 和显存峰值。
- `datagen/teacher.py` 与 `run_cases.py`：`--model` 可以写成 `HI=hi_r1,GIA=gia_r1`，同一个学生服务按角色发不同的 LoRA 名。只给一个名字时，行为和原来一样。
- `scripts/_pack_server.py`：打包加上 `training/`。

训练配置在服务器上要把 `"model"` 改成 `.models/llm/Qwen3.5-9B`（`FS_MODEL_DIR` 下的目录名），`lora_r` 不超过 `FS_LORA_RANK=32`。

## 4. 开跑前要实测的几件事（容器里没有 GPU，都没测过）

1. **122B FP8 能不能在 A100 上起来。** A100 没有 FP8 计算单元，vLLM 只能把 FP8 权重按"只量化权重"的方式跑：显存按 FP8 算，速度不会比 BF16 快。MoE 加混合注意力的组合能不能走这条路，要先起一次看日志。
2. **vLLM 加载的 LoRA 和训练时是否一致。** 用同一个 LoRA，在一条导出行上比较 HF + PEFT 与 vLLM 的贪心输出。要特别留意 Qwen3.5 线性注意力层的投影名是否被 vLLM 接受。
3. **9B 在 32K 上单卡训练的显存峰值。** 看 `train_summary.json` 的 `peak_gpu_mem_gb`。
4. **16 路并发下工具卡的显存峰值。** 这决定 verifier_ref 能不能和工具放在同一张卡上（`FS_VREF_MEM=0.45` 时约给工具留 40GB）。

## 5. 前提变了怎么改

- **学生改用 27B**（EVID baseline 现在用的是 Qwen3.8-27B）：推理仍是一卡一个副本，布局 A 不变。训练 32K 一个任务要 2 张卡：`FS_TRAIN_SLOTS="0,1;2,3"`，训练配置写 `"allow_multi_gpu": true`。一次只能跑 2 个任务。
- **122B FP8 在 A100 上起不来**：备选是 Int4 版（AWQ / GPTQ，约 65GB），`FS_TEACHER_GPUS=0,1`，可以多腾出 2 张卡给训练。但这等于换了教师，要写进 DESIGN。
- **教师继续用 Codex**（walker 计划 §10 第 3 条还没定；09-24 的决定是正式第 0 轮用本地 122B）：0–3 号卡全部空出来，学生 2–3 张、核验 1–2 张、工具 1 张，训练窗口能同时跑 6–7 个任务。
- **E1 四臂不要每臂分卡**：四臂共用同一套服务，学生底座挂多份 LoRA。预算按 `budget.jsonl` 里每臂的 token、请求数和训练 GPU 时记账。
