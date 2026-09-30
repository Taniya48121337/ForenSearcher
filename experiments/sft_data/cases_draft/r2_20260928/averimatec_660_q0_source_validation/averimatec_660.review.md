# 审核单：averimatec_660.json

- generated_from: `averimatec_660_q0_source_validation`
- generated_at: 2026-09-28
- needs_review: true
- task_family: claim
- requirement_origin: REQ1=neutral, REQ2=neutral

## notes

1. run 没有 FINISH 答案：required_evidence / checks 无从起草
2. run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写
3. REQ1 是中性模板（任务文字里抽不到 external 类分句），需人工改写
4. REQ2 是中性模板（任务文字里抽不到 media 类分句），需人工改写
5. REQ1（external）在最终答案里没有带引用的句子回应
6. REQ2（media）在最终答案里没有带引用的句子回应
7. 答案没有引用文本来源，取干净 Finding 引用的 E9、E8、E3（答案未引用，审核时确认是否必需）
8. units 判定里没有首轮 judge_targets，supports 只按答案句归类
9. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推

## 证据来源（trace）

- `youtube_upload`：E9、E8（clean_finding_only；工作区 W1-1）
- `blackheritageradio`：E3（clean_finding_only；工作区 W1-1）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "averimatec_660_q0_source_validation",
    "generated_at": "2026-09-28",
    "notes": [
      "run 没有 FINISH 答案：required_evidence / checks 无从起草",
      "run 里没有 GIA INVESTIGATE 决策，要求全部用模板，需人工改写",
      "REQ1 是中性模板（任务文字里抽不到 external 类分句），需人工改写",
      "REQ2 是中性模板（任务文字里抽不到 media 类分句），需人工改写",
      "REQ1（external）在最终答案里没有带引用的句子回应",
      "REQ2（media）在最终答案里没有带引用的句子回应",
      "答案没有引用文本来源，取干净 Finding 引用的 E9、E8、E3（答案未引用，审核时确认是否必需）",
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
      "youtube_upload": {
        "evidence_ids": [
          "E9",
          "E8"
        ],
        "origin": "clean_finding_only",
        "workspaces": [
          "W1-1"
        ],
        "answer_categories": [
          "external"
        ],
        "judge_categories": []
      },
      "blackheritageradio": {
        "evidence_ids": [
          "E3"
        ],
        "origin": "clean_finding_only",
        "workspaces": [
          "W1-1"
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
