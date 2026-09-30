# 审核单：videosham_0069.json

- generated_from: `videosham_0069_r1`
- generated_at: 2026-09-24
- needs_review: true
- task_family: media
- requirement_origin: REQ1=task, REQ2=task, REQ3=task, REQ4=task

## notes

1. 首轮 GIA 任务里 2 个分句含时间/编号（上一轮观察），没有写进 requirements
2. W1-1：recalibrate 撤销了 F1、F2、F3 的污染（verifier@0.1.0 原判误报）
3. W2-1：recalibrate 撤销了 F7 的污染（verifier@0.1.0 原判误报）
4. W3-1：recalibrate 撤销了 F13 的污染（verifier@0.1.0 原判误报）
5. units 判定里没有首轮 judge_targets，supports 只按答案句归类
6. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
7. source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认
8. media I0（E1）的文件 819b84d032d04724adf88ce41158c664.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入
9. media I1（E2）的文件 3f6e46d0d50046c8a427f35ebb5e1444.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入

## 证据来源（trace）

- `frames`：E67、E59、E68、E69、E65、E37、E63、E24（clean∩answer；工作区 W3-1）
- `media_profile`：E3、E4（clean∩answer；工作区 W1-1、W1-2、W2-1）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "videosham_0069_r1",
    "generated_at": "2026-09-24",
    "notes": [
      "首轮 GIA 任务里 2 个分句含时间/编号（上一轮观察），没有写进 requirements",
      "W1-1：recalibrate 撤销了 F1、F2、F3 的污染（verifier@0.1.0 原判误报）",
      "W2-1：recalibrate 撤销了 F7 的污染（verifier@0.1.0 原判误报）",
      "W3-1：recalibrate 撤销了 F13 的污染（verifier@0.1.0 原判误报）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认",
      "media I0（E1）的文件 819b84d032d04724adf88ce41158c664.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入",
      "media I1（E2）的文件 3f6e46d0d50046c8a427f35ebb5e1444.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入"
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
          "E67",
          "E59",
          "E68",
          "E69",
          "E65",
          "E37",
          "E63",
          "E24"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W3-1"
        ],
        "answer_categories": [
          "location",
          "meaning"
        ],
        "judge_categories": []
      },
      "media_profile": {
        "evidence_ids": [
          "E3",
          "E4"
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
