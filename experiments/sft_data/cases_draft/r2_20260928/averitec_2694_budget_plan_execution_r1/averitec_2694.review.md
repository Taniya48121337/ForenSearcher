# 审核单：averitec_2694.json

- generated_from: `averitec_2694_budget_plan_execution_r1`
- generated_at: 2026-09-28
- needs_review: true
- task_family: claim
- requirement_origin: REQ2=task, REQ1=neutral

## notes

1. 数据集 averitec 的 task_kind 猜不出，暂填 claim_check
2. REQ2 是中性模板（任务文字里抽不到 external 类分句），需人工改写
3. W2-2：recalibrate 撤销了 F9 的污染（verifier@0.1.0 原判误报）
4. W2-1：recalibrate 撤销了 F14 的污染（verifier@0.1.0 原判误报）
5. W3-2：recalibrate 撤销了 F24 的污染（verifier@0.1.0 原判误报）
6. units 判定里没有首轮 judge_targets，supports 只按答案句归类
7. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
8. date_in_range 未写：答案日期 2017-07-31 不在首段结论里，可能是历史背景而不是声明所指事件的日期（拿不准不写）

## 证据来源（trace）

- `wits`：E1、E10（clean∩answer；工作区 W1-1、W1-2、W2-1、W2-2、W3-1、W3-2）
- `sabeex`：E13（clean∩answer；工作区 W2-2、W3-1、W3-2）
- `spu`：E12（clean∩answer；工作区 W2-2）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "averitec_2694_budget_plan_execution_r1",
    "generated_at": "2026-09-28",
    "notes": [
      "数据集 averitec 的 task_kind 猜不出，暂填 claim_check",
      "REQ2 是中性模板（任务文字里抽不到 external 类分句），需人工改写",
      "W2-2：recalibrate 撤销了 F9 的污染（verifier@0.1.0 原判误报）",
      "W2-1：recalibrate 撤销了 F14 的污染（verifier@0.1.0 原判误报）",
      "W3-2：recalibrate 撤销了 F24 的污染（verifier@0.1.0 原判误报）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "date_in_range 未写：答案日期 2017-07-31 不在首段结论里，可能是历史背景而不是声明所指事件的日期（拿不准不写）"
    ],
    "needs_review": true,
    "task_family": "claim",
    "requirement_origin": {
      "REQ2": "task",
      "REQ1": "neutral"
    },
    "task_categories": {
      "T1": "",
      "T2": "provenance"
    },
    "trace": {
      "wits": {
        "evidence_ids": [
          "E1",
          "E10"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-1",
          "W1-2",
          "W2-1",
          "W2-2",
          "W3-1",
          "W3-2"
        ],
        "answer_categories": [
          "external",
          "provenance"
        ],
        "judge_categories": []
      },
      "sabeex": {
        "evidence_ids": [
          "E13"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W2-2",
          "W3-1",
          "W3-2"
        ],
        "answer_categories": [
          "external",
          "provenance"
        ],
        "judge_categories": []
      },
      "spu": {
        "evidence_ids": [
          "E12"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W2-2"
        ],
        "answer_categories": [
          "external"
        ],
        "judge_categories": []
      }
    }
  }
}
```
