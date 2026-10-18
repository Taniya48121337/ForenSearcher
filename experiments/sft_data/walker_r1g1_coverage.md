# walker r1g1 工具覆盖

147 个 run。计数按实际动作（response.json）。

| 工具 / 变体 | 合理用途（触发条件） | 自然出现的案例族 | 本轮次数 | 最少 | 状态 | 用到的案例（前 4 个） |
|---|---|---|---|---|---|---|
| SEARCH text→video | 找原片：用 OCR / 语音里的可识别线索搜视频 | A | 39 | 3 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM |
| SEARCH text_image→video | 找原片：文字查询没对上时，文字 + 一帧同时搜（文字搜索与以图搜图合并） | A | 24 | 1 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_4vbQKcShRXs, fmnv_a_65Xcqdl0HuE |
| SEARCH image→video | 找原片：文字与图文都没对上时只用帧搜 | A | 0 | 0 | 条件未出现 |  |
| SEARCH image→text | 图片最早出处 / 照片拍摄地：反向图片搜索找载有同一张图的网页 | E L | 0 | 4 | 不够 |  |
| SEARCH text→text | 核查声明 / 图片出处：搜事件本身的报道 | C E L | 210 | 5 | 够 | fmnv_c_0NqD96FRgOo, fmnv_c_1484605287315214338, fmnv_c_1485920619115909122, fmnv_c_1491485799170420737 |
| SEARCH text→image | 照片拍摄地：声明点名了地点 / 地标时，取该地点的参考照片与给定照片当面比对 | L | 0 | 3 | 不够 |  |
| SEARCH open dates | 要回答谁先发布 / 某来源是否早于声明：查页面声明日期、上传日期、最早存档 | E A C | 17 | 5 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM, fmnv_a_47uxQL9Ggdc |
| SEARCH open video | 读到的页面给出声明所说视频的链接时打开（hi@0.10.0 起自动带原语言字幕，原 include=subtitles 并入） | C | 7 | 0 | 够 | fmnv_c_1896358842776408410, fmnv_c_Gnv1XWciX8o, fmnv_c_fO5oQqZiXxw, groundlie360_v2276004 |
| SEARCH open URL/R | 打开一个已知网页来源（非视频、非日期） | — | 30 | 0 | 够 | fmnv_c_1485920619115909122, fmnv_c_1892942120954003622, fmnv_c_f6LMYqXwK7w, fmnv_c_imhMnsZI1Xo |
| INSPECT get_frames count | 看不到原片对应时两侧均匀取帧；以图找帧前取样 | B | 0 | 1 | 不够 |  |
| INSPECT get_frames times_s | 在比对报出的边界 / 对应时刻两侧取帧核看 | B A | 133 | 8 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM, fmnv_a_47uxQL9Ggdc |
| INSPECT get_frames times_s+box | 同一时刻画面不同的小区域：两侧放大 | B A | 17 | 2 | 够 | fmnv_a__KneSpxzvFg, videosham_4008, videosham_4025, videosham_4055 |
| INSPECT get_frames pages | 读到的 PDF 没有文字层（扫描件）：渲染页面再看 | C E | 0 | 0 | 条件未出现 |  |
| INSPECT ocr(image) | 放大后的横条区域（字幕、标题、标牌）读字；给定图片读水印 / 字幕 / 标牌 | B E L | 7 | 6 | 够 | fmnv_a__KneSpxzvFg, videosham_4008, videosham_4025, videosham_4055 |
| INSPECT ocr(video) | 找原片 / 核查声明：整段材料的画面文字（台标、字幕、名牌） | A C | 130 | 3 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM |
| INSPECT asr | 材料有音轨且任务涉及说了什么，或需要语音里的线索 | A B C | 155 | 5 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM |
| INSPECT locate(text) | 按描述在单段视频里找片段 | B(qv) | 0 | 1 | 不够 |  |
| INSPECT locate(image) | 对应类任务整段比对没对上：拿 A 的帧去 B 里找 | B(vcsl) | 4 | 1 | 够 | vcsl_pair_g1_02 |
| INSPECT video_structure | 单段视频按描述定位时先看镜头表，按切点取边界帧 | B(qv) | 100 | 1 | 够 | fmnv_c_0NqD96FRgOo, fmnv_c_1484605287315214338, fmnv_c_1485920619115909122, fmnv_c_1491485799170420737 |
| INSPECT compare_media | 两段视频（给定的 B / 找到的原片 / 页面链接的视频）整段比对；hi@0.10.0 起画面对不上且两边都有声音时自动只比声音（原 mode=audio 并入） | A B C | 30 | 6 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM, fmnv_a_47uxQL9Ggdc |
| INSPECT compare_media range | 等长短视频要逐帧精确边界时限定 A 全长再比 | B | 2 | 1 | 够 | vcsl_pair_g1_02, vcsl_pair_g1_03 |
| READ_RECORD text | 检索自动打开的页面、打开视频带的字幕：读正文 | C E L | 313 | 5 | 够 | fmnv_c_0NqD96FRgOo, fmnv_c_1484605287315214338, fmnv_c_1485920619115909122, fmnv_c_1491485799170420737 |
| READ_RECORD offset>0 | 正文没读完（next_offset 非空）且要求项还没覆盖 | C | 4 | 0 | 够 | fmnv_c_2Go7J9p4oeo, groundlie360_v21694609, groundlie360_v2256709, groundlie360_v46443403 |
| READ_RECORD focus | 比对图 / 帧没附上时带 focus 回看；参考照片与给定照片并排看 | B L | 101 | 3 | 够 | fmnv_c_0NqD96FRgOo, fmnv_c_1484605287315214338, fmnv_c_1485920619115909122, fmnv_c_1491485799170420737 |
| QUERY_MEMORY (HI) | 交报告前要引用的 E 已掉出视图（只剩编号）：按标签查回来；V9 变体（max_evidence_cards=6 + max_history_entries=4）里自然出现 | B(v9) | 0 | 1 | 不够 |  |
| GIA INVESTIGATE round1 | 每个案例第 1 轮 | 全部 | 147 | 1 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM |
| GIA INVESTIGATE round2 | 报告留下能靠更多查看补上的缺口（B 族时间窗 / 音轨） | B | 19 | 1 | 够 | fmnv_c_1485920619115909122, fmnv_c_1892942120954003622, fmnv_c_f6LMYqXwK7w, fmnv_c_imhMnsZI1Xo |
| GIA READ_RECORD | 报告引用的图不在本次附图里 | 全部 | 91 | 2 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_3elCxlGqUHM, fmnv_a_CqPYUMkicks, fmnv_a_HhGABZH0LxM |
| GIA QUERY_MEMORY | 第 2 轮之后第 1 轮报告不在视图里 | B(第 2 轮) | 19 | 0 | 够 | fmnv_c_1485920619115909122, fmnv_c_1892942120954003622, fmnv_c_f6LMYqXwK7w, fmnv_c_imhMnsZI1Xo |
| GIA FINISH | 每个案例结束 | 全部 | 147 | 1 | 够 | fmnv_a_1_FXVjf46T8, fmnv_a_1jLoMoqV8oE, fmnv_a_3Oz9RaXz5wc, fmnv_a_3elCxlGqUHM |

不够的：SEARCH image→text、SEARCH text→image、INSPECT get_frames count、INSPECT locate(text)、QUERY_MEMORY (HI)。补法只能是加案例或环境变体，不改 walker 去硬用。
