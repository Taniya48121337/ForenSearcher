# 审核单：videosham_1028.json

- generated_from: `videosham_1028_r2`
- generated_at: 2026-09-24
- needs_review: true
- task_family: media
- requirement_origin: REQ1=task, REQ2=task, REQ3=task, REQ4=task

## notes

1. 答案句（audio）没有对应的任务要求：音频对齐返回了覆盖0–4.84秒的候选对应，但不能据此认定声音或言语完全一致。…（E20）
2. W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）
3. W1-2：recalibrate 撤销了 F2、F4 的污染（verifier@0.1.0 原判误报）
4. 答案引用但没有被干净 Finding 引用、未入选的 E：E18（判定为有问题的 Finding 才引用它们，审核时看要不要补）
5. units 判定里没有首轮 judge_targets，supports 只按答案句归类
6. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
7. source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认
8. media I0（E1）的文件 7d159b6029ad4904a775b93bc02e7469.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入
9. media I1（E2）的文件 8aceba6d1d10483ab137a143961b1e0c.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入

## 证据来源（trace）

- `alignment`：E5、E7、E20（clean∩answer；工作区 W1-2、W2-2）
- `frames`：E10、E12（clean∩answer；工作区 W1-1、W1-2、W2-2）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "videosham_1028_r2",
    "generated_at": "2026-09-24",
    "notes": [
      "答案句（audio）没有对应的任务要求：音频对齐返回了覆盖0–4.84秒的候选对应，但不能据此认定声音或言语完全一致。…（E20）",
      "W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）",
      "W1-2：recalibrate 撤销了 F2、F4 的污染（verifier@0.1.0 原判误报）",
      "答案引用但没有被干净 Finding 引用、未入选的 E：E18（判定为有问题的 Finding 才引用它们，审核时看要不要补）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认",
      "media I0（E1）的文件 7d159b6029ad4904a775b93bc02e7469.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入",
      "media I1（E2）的文件 8aceba6d1d10483ab137a143961b1e0c.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入"
    ],
    "needs_review": true,
    "task_family": "media",
    "requirement_origin": {
      "REQ1": "task",
      "REQ2": "task",
      "REQ3": "task",
      "REQ4": "task"
    },
    "task_categories": {
      "T1": "location,meaning,uncertain",
      "T2": "location,meaning,uncertain"
    },
    "trace": {
      "alignment": {
        "evidence_ids": [
          "E5",
          "E7",
          "E20"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-2",
          "W2-2"
        ],
        "answer_categories": [
          "audio",
          "location",
          "meaning",
          "region"
        ],
        "judge_categories": []
      },
      "frames": {
        "evidence_ids": [
          "E10",
          "E12"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-1",
          "W1-2",
          "W2-2"
        ],
        "answer_categories": [
          "location",
          "meaning",
          "region"
        ],
        "judge_categories": []
      }
    }
  }
}
```
