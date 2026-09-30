# 审核单：qvhighlights_9167.json

- generated_from: `qvhighlights_9167_r2`
- generated_at: 2026-09-28
- needs_review: true
- task_family: claim
- requirement_origin: REQ1=neutral, REQ2=neutral

## notes

1. 数据集 qvhighlights 的 task_kind 猜不出，暂填 claim_check
2. run 没有 FINISH 答案：required_evidence / checks 无从起草
3. run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写
4. REQ1 是中性模板（任务文字里抽不到 media 类分句），需人工改写
5. REQ2 是中性模板（任务文字里抽不到 external 类分句），需人工改写
6. REQ1（media）在最终答案里没有带引用的句子回应
7. REQ2（external）在最终答案里没有带引用的句子回应
8. W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）
9. run 里没有可用的文本来源（没有带 source_url 的 E 被答案或干净 Finding 引用，run 也没有做过 SEARCH/open），required_evidence 缺文本来源项，审核时补或确认本案不需要
10. 答案没有引用帧，取干净 Finding 引用的 E19、E22、E24、E16、E17、E25、E36、E26（答案未引用，审核时确认是否必需）
11. units 判定里没有首轮 judge_targets，supports 只按答案句归类
12. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推

## 证据来源（trace）

- `frames`：E19、E22、E24、E16、E17、E25、E36、E26（clean_finding_only；工作区 W1-1）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "qvhighlights_9167_r2",
    "generated_at": "2026-09-28",
    "notes": [
      "数据集 qvhighlights 的 task_kind 猜不出，暂填 claim_check",
      "run 没有 FINISH 答案：required_evidence / checks 无从起草",
      "run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写",
      "REQ1 是中性模板（任务文字里抽不到 media 类分句），需人工改写",
      "REQ2 是中性模板（任务文字里抽不到 external 类分句），需人工改写",
      "REQ1（media）在最终答案里没有带引用的句子回应",
      "REQ2（external）在最终答案里没有带引用的句子回应",
      "W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）",
      "run 里没有可用的文本来源（没有带 source_url 的 E 被答案或干净 Finding 引用，run 也没有做过 SEARCH/open），required_evidence 缺文本来源项，审核时补或确认本案不需要",
      "答案没有引用帧，取干净 Finding 引用的 E19、E22、E24、E16、E17、E25、E36、E26（答案未引用，审核时确认是否必需）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推"
    ],
    "needs_review": true,
    "task_family": "claim",
    "requirement_origin": {
      "REQ1": "neutral",
      "REQ2": "neutral"
    },
    "task_categories": {},
    "trace": {
      "frames": {
        "evidence_ids": [
          "E19",
          "E22",
          "E24",
          "E16",
          "E17",
          "E25",
          "E36",
          "E26"
        ],
        "origin": "clean_finding_only",
        "workspaces": [
          "W1-1"
        ],
        "answer_categories": [
          "media"
        ],
        "judge_categories": []
      }
    }
  }
}
```
