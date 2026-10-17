# r1a D2 通读样本（8 条，r1 全量 134 runs）

渲染：`python scripts/render_trajectories.py --runs <下列 run> --out experiments/sft_data/r1a_review`

| 文件 | 族 | 为什么挑它 | 结果 |
|---|---|---|---|
| [averimatec_e133_r1_1](averimatec_e133_r1_1.md) | E | 冒烟起就跟着的案例：以图搜（Lens 缓存命中）→ 日期 → 读页 → E6 比对 → E7 回看对照图 | W1 COMPLETED → accepted |
| [averimatec_e9_r1_1](averimatec_e9_r1_1.md) | E | r0 判不可回答、新定义下可回答（Lens 召回载图页） | W1 COMPLETED → accepted |
| [averimatec_e85_r1_1](averimatec_e85_r1_1.md) | E | 保留的不可回答案例（"如实 INCOMPLETE"教学），派了第 2 轮 | W1 / W2 INCOMPLETE |
| [averimatec_l208_r1_1](averimatec_l208_r1_1.md) | L | 指定；L4b 文字搜 → L6a 页内主图 → L6 比对 | W1 COMPLETED → accepted |
| [averimatec_l623_r1_1](averimatec_l623_r1_1.md) | L | 指定；第 2 轮按 Scope 第一项取 BuzzFeed 页内主图并比对 | W1 INCOMPLETE（循环内改）→ W2 COMPLETED |
| [fmnv_c_RXR0B1SxiU0_r1_1](fmnv_c_RXR0B1SxiU0_r1_1.md) | C | 画面类声明：C2v 场景帧 → C2w 按镜头取帧 → C9v 整段转写 + 帧；它的 F3 在金标 r1a（F005） | COMPLETED |
| [groundlie360_v1959500_r1_1](groundlie360_v1959500_r1_1.md) | C | 画面类声明（茶里有没有"虫子"），单镜头长视频按 30 s 取帧 | COMPLETED |
| [averimatec_e122_r1_1](averimatec_e122_r1_1.md) | E | 第 2 轮：可回答 E 派了第 2 轮仍 INCOMPLETE（r1 全量 35 次第 2 轮里只有 l623 一次变成 COMPLETED） | W1 / W2 INCOMPLETE |
