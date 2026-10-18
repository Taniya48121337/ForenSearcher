# g1a D2 通读样本（6 条，g1 = 轮次 r1g1，147 runs，S0–S5 全走 Codex）

渲染：`python scripts/render_trajectories.py --runs <下列 run> --out experiments/sft_data/g1a_review`（10-18 修过的渲染器：INSPECT asr / ocr
印分段转写与 OCR 文字，GIA 印报告的 supporting_records，都按 rendered_payload）。

通读重点：S0–S3 由 Qwen 换成 Codex 后，**动作理由**（是否引了还没读到的内容、是否模板化、比对前是否就说"匹配"）、**检索词**（照抄声明
还是取自画面 / 转写线索；有没有带进模型自己知道、材料里没有的拼写）、**选页**、**第 2 轮任务文本**（Scope 是否针对缺口）。

| 文件 | 族 | 为什么挑它 | 结果（核验后） |
|---|---|---|---|
| [fmnv_a_3elCxlGqUHM_r1g1_1](fmnv_a_3elCxlGqUHM_r1g1_1.md) | A | 冒烟起就跟着的案例：找到原片，删掉区间报 30.2–41.6 s（GT 30.6–41.5，IoU ≈ 0.95） | W1 COMPLETED → accepted |
| [fmnv_a_1jLoMoqV8oE_r1g1_1](fmnv_a_1jLoMoqV8oE_r1g1_1.md) | A | 没找到原片、如实 INCOMPLETE——A 族这类报告 20 份全被核验判 negative 进了 rejected，看是报告的问题还是核验的问题 | W1 INCOMPLETE → rejected（negative_report_until_recall_calibrated） |
| [videosham_2004_r1g1_1](videosham_2004_r1g1_1.md) | B | VideoSham "静音换吉他配乐"：画面一致、音轨不同；循环内被改 INCOMPLETE（媒体属性 / 成对帧没引） | W1 COMPLETED → 循环内 INCOMPLETE |
| [fmnv_c_Aq7rDQx9jns_r1g1_1](fmnv_c_Aq7rDQx9jns_r1g1_1.md) | C | Codex 检索词取自帧里的字幕（"TRANSPORTER 2"），与 r1 Qwen 版照抄声明对照 | W1 COMPLETED → accepted |
| [fmnv_c_eLD1XA2UcwQ_r1g1_1](fmnv_c_eLD1XA2UcwQ_r1g1_1.md) | C | FMNV CD 类假标题与画面"大体相符"（数据集标注本身的问题）；检索从帧里的 "310fire via Storyful" 署名起查 | W1 COMPLETED → accepted |
| [groundlie360_v32327700_r1g1_1](groundlie360_v32327700_r1g1_1.md) | C | **第 2 轮**：W1 被循环内检查改 INCOMPLETE（整段视频说了什么没写、楼梯上的人没确认是 Harris）→ GIA 派第 2 轮 → W2 仍 INCOMPLETE；答案在金标 g1a（A001） | W1 / W2 INCOMPLETE；GIA FINISH |
