# 审核单：averitec_2872.json

- generated_from: `averitec_2872_q0_binding_validation`
- generated_at: 2026-09-28
- needs_review: true
- task_family: claim
- requirement_origin: REQ1=neutral, REQ2=neutral

## notes

1. 数据集 averitec 的 task_kind 猜不出，暂填 claim_check
2. run 没有 FINISH 答案：required_evidence / checks 无从起草
3. run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写
4. REQ1 是中性模板（任务文字里抽不到 external 类分句），需人工改写
5. REQ2 是中性模板（任务文字里抽不到 media 类分句），需人工改写
6. REQ1（external）在最终答案里没有带引用的句子回应
7. REQ2（media）在最终答案里没有带引用的句子回应
8. run 里没有可用的文本来源（没有带 source_url 的 E 被答案或干净 Finding 引用），required_evidence 缺文本来源项，审核时补或确认本案不需要
9. units 判定里没有首轮 judge_targets，supports 只按答案句归类
10. required_evidence 为空：答案与干净 Finding 都没有可用的 E
11. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推

## 证据来源（trace）


## 原 JSON

```json
{
  "draft": {
    "generated_from": "averitec_2872_q0_binding_validation",
    "generated_at": "2026-09-28",
    "notes": [
      "数据集 averitec 的 task_kind 猜不出，暂填 claim_check",
      "run 没有 FINISH 答案：required_evidence / checks 无从起草",
      "run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写",
      "REQ1 是中性模板（任务文字里抽不到 external 类分句），需人工改写",
      "REQ2 是中性模板（任务文字里抽不到 media 类分句），需人工改写",
      "REQ1（external）在最终答案里没有带引用的句子回应",
      "REQ2（media）在最终答案里没有带引用的句子回应",
      "run 里没有可用的文本来源（没有带 source_url 的 E 被答案或干净 Finding 引用），required_evidence 缺文本来源项，审核时补或确认本案不需要",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "required_evidence 为空：答案与干净 Finding 都没有可用的 E",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推"
    ],
    "needs_review": true,
    "task_family": "claim",
    "requirement_origin": {
      "REQ1": "neutral",
      "REQ2": "neutral"
    },
    "task_categories": {},
    "trace": {}
  }
}
```
