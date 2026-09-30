# 对照：r0 → r0b

## 教师侧行为（只看 run，不依赖核验器）

| 指标 | r0 | r0b |
|---|---|---|
| run 数 | 12 | 11 |
| 每 run 教师调用 | 37.4 | 18.1 |
| 每 run 输入 token | 1453971 | 629476 |
| GIA 给出 FINISH 的比例 | 1 | 1 |
| HI 动作步数 | 343 | 134 |
| SEARCH 占步数 | 0.122 | 0.112 |
| READ_RECORD 占步数 | 0.574 | 0.351 |
| INSPECT 占步数 | 0.268 | 0.478 |
| 报告 COMPLETED | 18 | 13 |
| 报告 INCOMPLETE | 23 | 12 |
| 最长同动作连击（最大） | 11 | 7 |
| 最长同动作连击（中位） | 3.5 | 3 |
| 带 rationale 的输出占比 | 0 | 0.759 |
| 轨迹里出现的核查站点 URL 数 | 10 | 0 |
| 实际打开的核查站点页面数 | 18 | 0 |
| 护栏拦下的核查站点候选/打开 | 0 | 9 |
| 护栏按日期拦下 | 0 | 0 |

## 数据侧（导出；两边核验器：local-gemma3-12b-16k-latest / local-gemma3-12b-16k-latest）

| 指标 | r0 | r0b |
|---|---|---|
| 行数 | hi_accepted 178; hi_held 165; hi_rejected 45; gia_accepted 22; gia_held 34; gia_rejected 3 | hi_accepted 105; hi_held 31; hi_rejected 23; gia_accepted 15; gia_held 21; gia_rejected 4 |
| HI 权重和 | 468 | 239.75 |
| GIA 权重和 | 22 | 15 |
| GIA accepted 答案数 | 1 | 0 |
| HI 权重份额（按动作） | INSPECT 0.5; READ_RECORD 0.404; SEARCH 0.088; COMPLETED 0.006; INCOMPLETE 0.002 | INSPECT 0.515; READ_RECORD 0.356; SEARCH 0.104; COMPLETED 0.025 |
| HI 监督 token 中位 | 91 | 132 |
| accepted 报告状态 | None | hi COMPLETED 2 |
| 核查站点沾染 | None | rows_in_context 0; rows_target_held 0; accepted_in_context 0 |
| 带 rationale 的 accepted 行 | None | hi 102; gia 5 |
| 权重调整次数 | None | memory_action_gain0 1; repetitive_run_middle 27 |

说明：教师侧指标不依赖核验器，可直接比；数据侧指标只有两边用同一个核验器时才可比。
READ_RECORD 占比高本身不一定是坏事（读页面是取证的一部分），要看 SEARCH 是否上来、连击是否消失。
