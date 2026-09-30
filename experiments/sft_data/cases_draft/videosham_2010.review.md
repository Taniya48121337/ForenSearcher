# 审核单：videosham_2010.json

- generated_from: `videosham_2010_r1`
- generated_at: 2026-09-24
- needs_review: true
- task_family: media
- requirement_origin: REQ1=task, REQ2=task, REQ3=task, REQ4=task

## notes

1. REQ1（location）在最终答案里没有带引用的句子回应
2. 答案句（region）没有对应的任务要求：2. 运动与构图：A上述样本中的车辆位置、侧面构图和乘员姿态基本一致，2.48与…（E14、E15、E16、E17、E11、E13）
3. W1-2：recalibrate 撤销了 F4、F6 的污染（verifier@0.1.0 原判误报）
4. 答案引用但没有被干净 Finding 引用、未入选的 E：E15、E16（判定为有问题的 Finding 才引用它们，审核时看要不要补）
5. units 判定里没有首轮 judge_targets，supports 只按答案句归类
6. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
7. source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认
8. media I0（E1）的文件 800da724d84d43afb39e1494af9c98b7.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入
9. media I1（E2）的文件 9689df738e2a4c1181a5f3410ef832de.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入

## 证据来源（trace）

- `frames`：E14、E17、E11、E13（clean∩answer；工作区 W1-2、W2-1）
- `crops`：E23、E19（clean∩answer；工作区 W2-1）
- `alignment`：E5、E18（clean∩answer；工作区 W1-1、W1-2、W2-1）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "videosham_2010_r1",
    "generated_at": "2026-09-24",
    "notes": [
      "REQ1（location）在最终答案里没有带引用的句子回应",
      "答案句（region）没有对应的任务要求：2. 运动与构图：A上述样本中的车辆位置、侧面构图和乘员姿态基本一致，2.48与…（E14、E15、E16、E17、E11、E13）",
      "W1-2：recalibrate 撤销了 F4、F6 的污染（verifier@0.1.0 原判误报）",
      "答案引用但没有被干净 Finding 引用、未入选的 E：E15、E16（判定为有问题的 Finding 才引用它们，审核时看要不要补）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认",
      "media I0（E1）的文件 800da724d84d43afb39e1494af9c98b7.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入",
      "media I1（E2）的文件 9689df738e2a4c1181a5f3410ef832de.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入"
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
      "T2": "audio,location,meaning,uncertain"
    },
    "trace": {
      "frames": {
        "evidence_ids": [
          "E14",
          "E17",
          "E11",
          "E13"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-2",
          "W2-1"
        ],
        "answer_categories": [
          "meaning",
          "region"
        ],
        "judge_categories": []
      },
      "crops": {
        "evidence_ids": [
          "E23",
          "E19"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W2-1"
        ],
        "answer_categories": [
          "meaning"
        ],
        "judge_categories": []
      },
      "alignment": {
        "evidence_ids": [
          "E5",
          "E18"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-1",
          "W1-2",
          "W2-1"
        ],
        "answer_categories": [
          "audio"
        ],
        "judge_categories": []
      }
    }
  }
}
```
