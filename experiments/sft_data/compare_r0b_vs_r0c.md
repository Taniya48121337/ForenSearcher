# 对照：r0b → r0c

## 教师侧行为（只看 run，不依赖核验器）

| 指标 | r0b | r0c |
|---|---|---|
| run 数 | 11 | 11 |
| 每 run 教师调用 | 18.1 | 1 |
| 每 run 输入 token | 629476 | 0 |
| GIA 给出 FINISH 的比例 | 1 | 0 |
| HI 动作步数 | 134 | 0 |
| SEARCH 占步数 | 0.112 | 0 |
| READ_RECORD 占步数 | 0.351 | 0 |
| INSPECT 占步数 | 0.478 | 0 |
| 报告 COMPLETED | 13 | 0 |
| 报告 INCOMPLETE | 12 | 0 |
| 最长同动作连击（最大） | 7 | 0 |
| 最长同动作连击（中位） | 3 | 0 |
| 带 rationale 的输出占比 | 0.759 | 0 |
| 轨迹里出现的核查站点 URL 数 | 0 | 0 |
| 实际打开的核查站点页面数 | 0 | 0 |
| 护栏拦下的核查站点候选/打开 | 9 | 0 |
| 护栏按日期拦下 | 0 | 0 |
| 声明类 run 数 | 6 | 6 |
| 声明类：每 run 检索查询数 | 2 | 0 |
| 声明类：带核查字样的查询占比 | 0.083 | 0 |
| 声明类：每 run 打开页面数 | 7.17 | 0 |
| 声明类：每 run 打开的非核查域名数 | 6.83 | 0 |
| 声明类：每 run 实际读过的非核查域名数 | 3 | 0 |
| 声明类：每 run 报告引用的非核查域名数 | 2.33 | 0 |
| 声明类：读过或引用的核查类页面（域名） | 1 | 0 |

## 数据侧（导出；两边核验器：claude-cli-claude-sonnet-5-5 / —）

| 指标 | r0b | r0c |
|---|---|---|
| 行数 | hi_accepted 123; hi_held 19; hi_rejected 17; gia_accepted 18; gia_held 19; gia_rejected 3 | — |
| HI 权重和 | 268.75 | — |
| GIA 权重和 | 18 | — |
| GIA accepted 答案数 | 3 | — |
| HI 权重份额（按动作） | INSPECT 0.48; READ_RECORD 0.351; SEARCH 0.104; INCOMPLETE 0.037; COMPLETED 0.019; QUERY_MEMORY 0.009 | — |
| HI 监督 token 中位 | 135 | — |
| accepted 报告状态 | hi INCOMPLETE 6; COMPLETED 2 | — |
| 核查站点沾染 | rows_in_context 0; rows_target_held 0; accepted_in_context 0 | — |
| 带 rationale 的 accepted 行 | hi 114; gia 5 | — |
| 权重调整次数 | memory_action_gain0 8; repetitive_run_middle 24 | — |

说明：教师侧指标不依赖核验器，可直接比；数据侧指标只有两边用同一个核验器时才可比。
READ_RECORD 占比高本身不一定是坏事（读页面是取证的一部分）；声明类案例看"每 run 查询数、实际读过和引用的独立域名数"，
不看 SEARCH 占步数。VideoSham 是给定媒体比对，本来不搜索。
