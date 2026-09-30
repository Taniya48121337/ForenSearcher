# 审核单：averitec_2694.json

- generated_from: `averitec_2694_budget_plan_execution_12k_r1`
- generated_at: 2026-09-28
- needs_review: true
- task_family: claim
- requirement_origin: REQ1=task, REQ2=task

## notes

1. 数据集 averitec 的 task_kind 猜不出，暂填 claim_check
2. W3-1：recalibrate 撤销了 F23、F24 的污染（verifier@0.1.0 原判误报）
3. units 判定里没有首轮 judge_targets，supports 只按答案句归类
4. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
5. date_in_range 未写：答案日期 2017-07-31 不在首段结论里，可能是历史背景而不是声明所指事件的日期（拿不准不写）

## 证据来源（trace）

- `wits`：E5、E1（clean∩answer；工作区 W1-1、W1-2、W2-1、W2-2、W3-1）
- `spu`：E6（clean∩answer；工作区 W2-2）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "averitec_2694_budget_plan_execution_12k_r1",
    "generated_at": "2026-09-28",
    "notes": [
      "数据集 averitec 的 task_kind 猜不出，暂填 claim_check",
      "W3-1：recalibrate 撤销了 F23、F24 的污染（verifier@0.1.0 原判误报）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "date_in_range 未写：答案日期 2017-07-31 不在首段结论里，可能是历史背景而不是声明所指事件的日期（拿不准不写）"
    ],
    "needs_review": true,
    "task_family": "claim",
    "requirement_origin": {
      "REQ1": "task",
      "REQ2": "task"
    },
    "task_categories": {
      "T1": "external,provenance",
      "T2": "provenance"
    },
    "trace": {
      "wits": {
        "evidence_ids": [
          "E5",
          "E1"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-1",
          "W1-2",
          "W2-1",
          "W2-2",
          "W3-1"
        ],
        "answer_categories": [
          "external",
          "provenance"
        ],
        "judge_categories": []
      },
      "spu": {
        "evidence_ids": [
          "E6"
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
