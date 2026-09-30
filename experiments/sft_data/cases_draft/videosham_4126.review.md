# 审核单：videosham_4126.json

- generated_from: `videosham_4126_split_intervals_r7`
- generated_at: 2026-09-24
- needs_review: true
- task_family: media
- requirement_origin: REQ1=task, REQ2=task

## notes

1. task_kind 取 temporal_locate：任务是找 A/B 差异（人物可见性）的时间区间；TaskKind 没有"差异区间定位"专值，video_diff 也说得通，审核时二选一
2. 首轮 GIA 任务里 14 个分句含时间/编号（上一轮观察），没有写进 requirements
3. 答案句（region）没有对应的任务要求：3. 后段出现边界：2.233333秒两侧均未见目标，该帧持续至2.266667…（E40、E66、E33、E26、E90）
4. W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）
5. W1-2：recalibrate 撤销了 F4 的污染（verifier@0.1.0 原判误报）
6. W2-2：recalibrate 撤销了 F12 的污染（verifier@0.1.0 原判误报）
7. W3-2：recalibrate 撤销了 F17 的污染（verifier@0.1.0 原判误报）
8. 答案引用但没有被干净 Finding 引用、未入选的 E：E46、E51、E79、E52、E10、E14、E61、E84、E55、E85、E62、E70、E63、E71、E90、E91、E99、E12、E16、E74、E82、E13、E17、E42、E68、E108、E110、E109、E111、E8、E6（判定为有问题的 Finding 才引用它们，审核时看要不要补）
9. units 判定里没有首轮 judge_targets，supports 只按答案句归类
10. 没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推
11. source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认
12. media I0（E1）的文件 fcc76c4a59fe45d3abba373be8ea440f.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入
13. media I1（E2）的文件 92c513797f7842a48b77a58ea7246527.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入

## 证据来源（trace）

- `frames`：E40、E66、E33、E26、E72、E80、E34、E27、E47、E57、E73、E81、E165、E176、E167、E177、E168、E178、E170、E179、E75、E83（clean∩answer；工作区 W1-2、W2-2、W3-3）

## 原 JSON

```json
{
  "draft": {
    "generated_from": "videosham_4126_split_intervals_r7",
    "generated_at": "2026-09-24",
    "notes": [
      "task_kind 取 temporal_locate：任务是找 A/B 差异（人物可见性）的时间区间；TaskKind 没有\"差异区间定位\"专值，video_diff 也说得通，审核时二选一",
      "首轮 GIA 任务里 14 个分句含时间/编号（上一轮观察），没有写进 requirements",
      "答案句（region）没有对应的任务要求：3. 后段出现边界：2.233333秒两侧均未见目标，该帧持续至2.266667…（E40、E66、E33、E26、E90）",
      "W1-1：recalibrate 撤销了 F1 的污染（verifier@0.1.0 原判误报）",
      "W1-2：recalibrate 撤销了 F4 的污染（verifier@0.1.0 原判误报）",
      "W2-2：recalibrate 撤销了 F12 的污染（verifier@0.1.0 原判误报）",
      "W3-2：recalibrate 撤销了 F17 的污染（verifier@0.1.0 原判误报）",
      "答案引用但没有被干净 Finding 引用、未入选的 E：E46、E51、E79、E52、E10、E14、E61、E84、E55、E85、E62、E70、E63、E71、E90、E91、E99、E12、E16、E74、E82、E13、E17、E42、E68、E108、E110、E109、E111、E8、E6（判定为有问题的 Finding 才引用它们，审核时看要不要补）",
      "units 判定里没有首轮 judge_targets，supports 只按答案句归类",
      "没找到数据集标注（run_config 无 label 字段，未给 --labels），label_expected 未写；不从答案反推",
      "source_policy 取 provided_media（run 未配搜索 / evidence_source_policy=provided_media），审核时确认",
      "media I0（E1）的文件 fcc76c4a59fe45d3abba373be8ea440f.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入",
      "media I1（E2）的文件 92c513797f7842a48b77a58ea7246527.mp4 不在 run/artifacts 里（未入库），路径按 store 登记写入"
    ],
    "needs_review": true,
    "task_family": "media",
    "requirement_origin": {
      "REQ1": "task",
      "REQ2": "task"
    },
    "task_categories": {
      "T1": "location",
      "T2": "location",
      "T3": "location,uncertain"
    },
    "trace": {
      "frames": {
        "evidence_ids": [
          "E40",
          "E66",
          "E33",
          "E26",
          "E72",
          "E80",
          "E34",
          "E27",
          "E47",
          "E57",
          "E73",
          "E81",
          "E165",
          "E176",
          "E167",
          "E177",
          "E168",
          "E178",
          "E170",
          "E179",
          "E75",
          "E83"
        ],
        "origin": "clean∩answer",
        "workspaces": [
          "W1-2",
          "W2-2",
          "W3-3"
        ],
        "answer_categories": [
          "location",
          "region"
        ],
        "judge_categories": []
      }
    }
  }
}
```
