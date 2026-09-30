# 审核单：videosham_1007.json

- generated_from: `videosham_1007_validation`
- generated_at: 2026-09-24
- needs_review: true
- task_family: media
- requirement_origin: REQ1=task, REQ2=task, REQ3=task, REQ4=task

## notes

1. 答案句（region）没有对应的任务要求：可确认两者在风景转向划船者的局部衔接上存在差异；现有证据不足以确定具体编辑方式或…（E24、E5）
2. 答案句（region）没有对应的任务要求：B在1.635秒呈现相近山湖景色，3.403秒可见右下方船桨和右缘部分人物，4.…（E19、E48、E29）
3. W1-1：recalibrate 撤销了 F1、F3 的污染（verifier@0.1.0 原判误报）
4. W1-2：recalibrate 撤销了 F5 的污染（verifier@0.1.0 原判误报）
5. W2-1：recalibrate 撤销了 F7、F8 的污染（verifier@0.1.0 原判误报）
6. W2-2：recalibrate 撤销了 F11 的污染（verifier@0.1.0 原判误报）
7. 答案引用但没有被干净 Finding 引用、未入选的 E：E48（判定为有问题的 Finding 才引用它们，审核时看要不要补）
8. units 判定里没有首轮 judge_targets，supports 只按答案句归类
9. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
10. source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认
11. media I0（E1）的文件 dafc9dbdaabe4d8a99a6814623e29a74.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入
12. media I1（E2）的文件 691eb0cd3e444a0795f41f667c65cb77.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入

## 证据来源（trace）

- `frames`：E24、E19、E29、E38（clean∩answer；工作区 W1-1、W2-1、W2-2、W3-2）
- `alignment`：E5、E8、E49、E33（clean∩answer；工作区 W1-2、W2-1、W2-2、W3-2）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "videosham_1007_validation",
    "generated_at": "2026-09-24",
    "notes": [
      "答案句（region）没有对应的任务要求：可确认两者在风景转向划船者的局部衔接上存在差异；现有证据不足以确定具体编辑方式或…（E24、E5）",
      "答案句（region）没有对应的任务要求：B在1.635秒呈现相近山湖景色，3.403秒可见右下方船桨和右缘部分人物，4.…（E19、E48、E29）",
      "W1-1：recalibrate 撤销了 F1、F3 的污染（verifier@0.1.0 原判误报）",
      "W1-2：recalibrate 撤销了 F5 的污染（verifier@0.1.0 原判误报）",
      "W2-1：recalibrate 撤销了 F7、F8 的污染（verifier@0.1.0 原判误报）",
      "W2-2：recalibrate 撤销了 F11 的污染（verifier@0.1.0 原判误报）",
      "答案引用但没有被干净 Finding 引用、未入选的 E：E48（判定为有问题的 Finding 才引用它们，审核时看要不要补）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认",
      "media I0（E1）的文件 dafc9dbdaabe4d8a99a6814623e29a74.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入",
      "media I1（E2）的文件 691eb0cd3e444a0795f41f667c65cb77.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入"
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
          "E24",
          "E19",
          "E29",
          "E38"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-1",
          "W2-1",
          "W2-2",
          "W3-2"
        ],
        "answer_categories": [
          "meaning",
          "region"
        ],
        "judge_categories": []
      },
      "alignment": {
        "evidence_ids": [
          "E5",
          "E8",
          "E49",
          "E33"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-2",
          "W2-1",
          "W2-2",
          "W3-2"
        ],
        "answer_categories": [
          "audio",
          "location",
          "region"
        ],
        "judge_categories": []
      }
    }
  }
}
```
