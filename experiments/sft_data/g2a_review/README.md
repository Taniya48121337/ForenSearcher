# g2a D2 通读样本（4 条，g2 = 轮次 r1g2，60 runs，S0–S5 Codex，冻结口径 specs/g1_freeze.json）

渲染：`python scripts/render_trajectories.py --runs <下列 run> --out experiments/sft_data/g2a_review`（修过的渲染器，同 g1a）。
每个补量格 1 条；通读重点：A 族 S1 的检索词是不是取自画面 / 转写线索（而不是声明或模型自己的知识）、找到的候选是否先比对再下结论、
日期是"这一份上传"的日期还是被说成最早出处；B 族区域 OCR 的文字是否照录。

| 文件 | 补量格 | 为什么挑它 | 结果（核验后） |
|---|---|---|---|
| [fmnv_a_0Uc3vwNlDdY_r1g2_1](fmnv_a_0Uc3vwNlDdY_r1g2_1.md) | A · SEARCH text→video | 冒烟第 1 条：文字查询直接找到原片，删掉区间报 B 43.1–59.0 s（GT 43.9–59.6） | W1 COMPLETED；HI accepted 8 / held 1 / rejected 1 |
| [fmnv_a_4aIK1tjpGuk_r1g2_1](fmnv_a_4aIK1tjpGuk_r1g2_1.md) | A · SEARCH text_image→video | 第 1 条文字查询没对上，第 2 条文字 + 一帧搜到原片；删掉 B 58.5–79.0 s（GT 58.6–78.6） | W1 COMPLETED；accepted 6 / held 2 / rejected 1 |
| [fmnv_a_fsdDrrt_maI_r1g2_1](fmnv_a_fsdDrrt_maI_r1g2_1.md) | A · compare_media 视频模式 + 原片日期 | 找到原片后整段比对、查上传日期与最早存档；它的 F1 在金标 g2a（F001，核验器标错） | W1 COMPLETED；accepted 6 / held 2 / rejected 1 |
| [videosham_4002_r1g2_1](videosham_4002_r1g2_1.md) | B · INSPECT ocr(image)（M3c 区域 OCR） | 文字篡改："John Conor / 39 Aug 2045" 对 "CONOR HEWETT / SEPTEMBER 4, 2021"，比对报横条差异框 → 两侧放大图各 OCR 一次 | W1 COMPLETED；accepted 8 |
