# Lens（Serper）与 Yandex 以图搜图召回对比（2026-10-16）

同一批 20 张查询图（字节相同）。Yandex 为第一轮结果，Lens 为第二轮；合并为离线按 multi_reverse 逻辑合并。
名次是统一写法下的最终名次（核对判 near-identical 的排最前）。

| 图 | 族 | Yandex 名次 | Lens 名次 | 合并名次 | 真来源核对判定（Yandex / Lens） | 判定对不对 | 经图床 |
|---|---|---|---|---|---|---|---|
| e107 ※ | E | 2 | 1 | 1 | different / different | Yandex 错（同一张照片被加框贴旗，判 no match）；Lens 错（同上） | imgbb |
| e297 | E | 2 | – | 2 | same / – | Yandex 对（同一张照片，near-identical） | imgbb |
| e260 | E | 3 | 2 | 5 | same / different | Yandex 对（同一张照片，near-identical）；Lens 对（页面缩略图是同一事件的另一张照片，no match） | imgbb |
| e505 | E | 1 | – | 2 | same / – | Yandex 对（同一张照片，near-identical） | imgbb |
| e248 | E | – | – | – | – / – | – | imgbb |
| e193 ※ | E | 26 | – | 51 | unchecked / – | – | imgbb |
| e372 | E | – | 2 | 3 | – / different | Lens 对（页面缩略图是同一现场的另一张照片，no match） | imgbb |
| e403 | E | – | 1 | 1 | – / same | Lens 对（同一张照片，near-identical） | imgbb |
| e459 | E | – | – | – | – / – | – | imgbb |
| e122 ※ | E | 20 | – | 20 | unchecked / – | – | imgbb |
| l341 | L | – | – | – | – / – | – | imgbb |
| l274 ※ | L | – | – | – | – / – | – | imgbb |
| l623 | L | – | – | – | – / – | – | imgbb |
| l497 | L | – | – | – | – / – | – | imgbb |
| l703 | L | – | – | – | – / – | – | imgbb |
| evid_009 | EVID | – | – | – | – / – | – | imgbb |
| evid_022 | EVID | – | – | – | – / – | – | imgbb |
| evid_042 | EVID | – | – | – | – / – | – | imgbb |
| evid_076 | EVID | – | – | – | – / – | – | imgbb |
| evid_103 | EVID | – | 4 | 7 | – / different | Lens 对（页面缩略图是同一场抗议的另一张照片，no match） | imgbb |

※ 不计入召回，见下文"单列"。

## 前 10 召回（不含 ※ 的几张）

| 族 | 张数 | Yandex 单独 | Lens 单独 | 两者合并 |
|---|---|---|---|---|
| E | 7 | 3/7 | 3/7 | 5/7 |
| L | 4 | 0/4 | 0/4 | 0/4 |
| EVID | 5 | 0/5 | 1/5 | 1/5 |

- Lens 前 10 有、Yandex 前 10 没有的真来源：3 个（e372, e403, evid_103）
- Yandex 前 10 有、Lens 前 10 没有的真来源：2 个（e297, e505）

## 单列（不计入召回）

- **e107**：查询图本身被改过（单边黑框 + 贴旗），核对漏判。Yandex 2，Lens 1，合并 1；真来源核对 Yandex different / Lens different。
- **e193**：真来源在前 10 之外（Yandex 第 26）。Yandex 26，Lens –，合并 51；真来源核对 Yandex unchecked / Lens –。
- **e122**：真来源在前 10 之外（Yandex 第 20）。Yandex 20，Lens –，合并 20；真来源核对 Yandex unchecked / Lens –。
- **l274**：Yandex 结果里有篡改副本被判 near-identical（第一轮第 8 名）。Yandex –，Lens –，合并 –；真来源核对 Yandex – / Lens –。

## Lens 结果里的日期

Serper 版 Lens 的 date 字段；约略年份按取到结果那天（date_basis）换算。"网址年份"是结果网址里带的年份（有的话），用来粗查换算对不对；E 族另列 GT 最早出处日期。

| 图 | 名次 | 结果网站 | Lens 原文 | 换算 | 网址年份 | 换算对不对 | GT 最早出处 |
|---|---|---|---|---|---|---|---|
| （Lens 结果里没有日期） | | | | | | | |

Lens 结果共 457 条，带日期的 0 条。

## 额度与图床

- Serper Lens 请求 20 次（硬上限 20，lens/ledger.json），每次扣 [3] 个额度，合计 60；余额 2491 → 2431（余额接口约 20 秒后才更新，第一次的读数是事后补记的）。
- 图床上传 20 次，用的是 imgbb（imgbb 10 分钟后删除；litterbox 备用没有用上）。
- 返回 0 条结果、照样扣了额度的：e297, e122, l341（原因未查：可能是 Google 当时取不到 imgbb 的图，也可能 Lens 确实没找到；e297 的同一张图 Yandex 找到了 3 条 near-identical）。
- 硬盘缓存命中 0 次（这批图以前没用 Lens 搜过）。

## 观察

- Serper 版 Lens 的结果多是"同一场景的其他照片"（visual matches），不是同一张图：真来源页面的缩略图常是该页的另一张照片（e372、e260、evid_103），核对判 no match 对缩略图是对的，但不说明页面里没有查询图。
- 合并时同一网页的核对取 base 引擎（Lens 在前）的缩略图：e260 的真来源 Yandex 缩略图判 near-identical、Lens 缩略图判 no match，合并后按 Lens 判，名次从 Yandex 的 3 / Lens 的 2 落到 5。可以考虑 base 对不上时再试其他引擎的缩略图。
- 第一轮 Yandex 只存了核对后重排的顺序，离线合并把它当作 Yandex 原始名次。

