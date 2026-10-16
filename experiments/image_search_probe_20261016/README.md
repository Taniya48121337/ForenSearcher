# 以图搜图召回实测（2026-10-16）

多引擎以图搜图统一写法（`search/providers/reverse_normalize.py`，开关 `FOREN_REVERSE_NORMALIZE=1`，默认关）与
缩略图自核（`search/providers/thumb_verify.py`）上线后的两次实测：

1. **Yandex 单引擎**（10-16 第一轮，不花额度）：本目录 `yandex/`。
2. **Google Lens（经 Serper）对比**（10-16 第二轮，Serper 硬上限 20 次）：`lens/` 与 `lens_vs_yandex.md`。

## 样本（30 张，第二轮 Lens 用其中 20 张）

| 组 | 张数 | 来源 | 真来源（GT） |
|---|---|---|---|
| E 族（图片最早出处） | 10 | `experiments/sft_data/media/d3/averimatec_eNNN.jpg`：e107 e297 e260 e505 e248 e193 e372 e403 e459 e122 | `experiments/sft_data/unanswerable_r0b.json` 的 `gt_norm`（页面级网址） |
| L 族（拍摄地） | 5 | 同上：l341 l274 l623 l497 l703 | 同上（l497 的 gt_norm 丢了查询参数，按 blogId=iamryo、logNo=150165640240 匹配） |
| EVID 帧 | 15 | `plan-harness-1/datasets/EVID-Bench/videos/video_NNN.mp4` 38% 处的一帧，存于 `query_frames/`：002 009 022 042 049 056 069 076 083 096 103 110 117（单一原片）、015 063（多原片） | `evid_bench.json` 的 `source`（YouTube 编号） |

- 同一组近似重复只留一张（e84/e92/l498、e133/e411/e522 等）；E/L 图都是公开数据集 AVerImaTeC 的题图。
- 查询图：E/L 把 d3 原图重新编码成 JPEG q92（长边 ≤1280），EVID 用 OpenCV 取帧后同样编码；两轮用的是同一批字节。
- 输出类型：E/L 按 `text`（找网页），EVID 按 `video`。

## 方法

- `recall_yandex.py`（第一轮原样存档）：每张图一个 `MultiReverseImageProvider([yandex], normalize=True)`，
  前 10 条下载缩略图（SafeHttpClient）比 dHash / pHash，判 same / similar / different / unchecked；same 排最前。
- 命中判定：结果网址去掉协议、`www.` / `m.`、末尾 `/` 后以 GT 开头（EVID 按 YouTube 编号）。
- 同时记"只命中同一站点、不是那一页"的名次，以及有没有命中案卷排除的原核查文章（第一轮 0 次）。
- 第一轮 30 次 Yandex 搜索全部成功（每次间隔 4 秒，合计约 137 秒），约 280 张 Yandex 缩略图全部下载成功。

## 第一轮结论（Yandex 单引擎）

| 组 | 真来源进前 10 | 真来源名次 |
|---|---|---|
| E 族 | **4/10** | e505 第 1、e107 第 2、e297 第 2、e260 第 3；e193 第 26、e122 第 20（前 10 外） |
| L 族 | **0/5** | — |
| EVID 帧 | **0/15** | — |

逐图见 `yandex/table.md`，原始结果 `yandex/results.json`。

### 核对准确度（逐张人工看过所有非 different 判定，见 `yandex/sheet_same.jpg`、`yandex/sheet_similar.jpg`）

- 真来源在前 10 的 4 张：3 张判 same（对），**e107 判 different（漏判）**——查询图被加了单边黑框、贴了泰国国旗
  （`yandex/e107_pair.jpg`），哈希对不上。
- 判 **same** 的 29 条：28 条确是同一张照片；**1 条是篡改副本**（l274 第 8 名：同一张照片，手里被贴了东西，
  `yandex/l274_r8.jpg`）——感知哈希看不出局部编辑。
- 判 **similar** 的 24 条：18 条其实是同一张照片的裁切 / 加框 / 加字版本，6 条是同一场景的不同画面（l341、evid_069、evid_015）。
- 因此给 agent 的话改成只说测量结果："near-identical" / "partial match, not near-identical" / "no match"
  （第二轮又给 no match 补上"裁切、加框、局部编辑都可能让哈希对不上"），`extra.verified_match` 的取值不变。

### 阈值来源

`scripts/_calibrate_thumb_hash.py`（只读本机图片）：160 张图做正例（缩放 / 重压缩 / 裁 3–6% / 加黑边），
不同案例的图 23000 对、同视频不同帧 1998 对做负例。same = dHash ≤ 0.25 且 pHash ≤ 0.22；
similar = dHash ≤ 0.36 且 pHash ≤ 0.32。负例里没有一对判 same。

## 第二轮：Google Lens（经 Serper）对比（2026-10-16）

- 样本：上面的 E 10 张、L 5 张，加 EVID 5 帧（042、076、022、009、103：原片都还在 YouTube 上（oEmbed 200）；挑单一原片、
  帧上没有或只有很少的二次剪辑字幕 / 叠加物的——Lens 认的是原片画面）。查询图字节与第一轮相同（`lens/query/`）。
- 方法：`recall_lens.py`。Serper 硬上限 20 次（`GoogleLensProvider(max_searches=…)` + `lens/ledger.json` 跨进程累计）；
  本机图经 imgbb 图床（只在该进程里设 `FOREN_LENS_IMAGE_HOST=imgbb`）；Yandex 不重搜，用第一轮结果。
  先跑 5 张（e107 e248 e372 l274 l703），确认图床成功、有结果后再跑其余 15 张。
- 额度：20 次请求，每次扣 **3** 个 Serper 额度（余额 2491 → 2431，共 60）；图床 imgbb 上传 20 次，全部成功。
- 结果（不含 e107、e193、e122、l274 四张单列的）：

| 族 | Yandex 单独 | Lens 单独 | 两者合并 |
|---|---|---|---|
| E（7 张） | 3/7 | 3/7 | 5/7 |
| L（4 张） | 0/4 | 0/4 | 0/4 |
| EVID（5 帧） | 0/5 | 1/5 | 1/5 |

  Lens 有、Yandex 没有：e372、e403、evid_103；Yandex 有、Lens 没有：e297、e505。Lens 结果 457 条里没有一条带日期。
  3 次搜索返回 0 条但照样扣额度（e297、e122、l341）。详见 `lens_vs_yandex.md`。

## 文件

- `recall_yandex.py`：第一轮脚本（存档）。
- `yandex/results.json`：每张图的全部结果（网址、判定、哈希距离、是否 GT / 同站点 / 排除来源）与核对统计。
- `yandex/table.md`：逐图对照表。
- `yandex/sheet_same.jpg`、`yandex/sheet_similar.jpg`：判 same / similar 的查询图与缩略图并排。
- `yandex/e107_pair.jpg`、`yandex/l274_r8.jpg`：漏判与误判的两例。
- `query_frames/`：15 张 EVID 帧（E/L 图在 `experiments/sft_data/media/d3/`）。
- `recall_lens.py`：第二轮脚本；`lens/results.json`（每张图的 Lens 原始顺序 `lens_raw`、统一写法后的结果与核对、额度）、
  `lens/ledger.json`（每次请求与余额）、`lens/query/`（20 张查询图）、`lens/gt_thumbs.jpg`（真来源缩略图与查询图并排）。
- `compare_lens_yandex.py` → `lens_vs_yandex.md`：逐图对照、离线合并名次、召回汇总、日期、额度。
