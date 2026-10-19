# walker r1g2 工具覆盖

60 个 run。计数按实际动作（response.json）。

| 工具 / 变体 | 合理用途（触发条件） | 自然出现的案例族 | 本轮次数 | 最少 | 状态 | 用到的案例（前 4 个） |
|---|---|---|---|---|---|---|
| SEARCH text→video | 找原片：用 OCR / 语音里的可识别线索搜视频 | A | 53 | 3 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| SEARCH text_image→video | 找原片：文字查询没对上时，文字 + 一帧同时搜（文字搜索与以图搜图合并） | A | 25 | 1 | 够 | fmnv_a_4aIK1tjpGuk, fmnv_a_6C0lUMKMj0w, fmnv_a_6VKUdVYw_j8, fmnv_a_8SN_JjTXSRY |
| SEARCH image→video | 找原片：文字与图文都没对上时只用帧搜 | A | 0 | 0 | 条件未出现 |  |
| SEARCH image→text | 图片最早出处 / 照片拍摄地：反向图片搜索找载有同一张图的网页（r1 起 E/L 走 image→image；条件触发） | E L | 0 | 0 | 条件未出现 |  |
| SEARCH text→text | 核查声明 / 图片出处：搜事件本身的报道 | C E L | 0 | 5 | 不够 |  |
| SEARCH text→image | 照片拍摄地：声明点名了地点 / 地标时，取该地点的参考照片与给定照片当面比对 | L | 0 | 3 | 不够 |  |
| SEARCH open dates | 要回答谁先发布 / 某来源是否早于声明：查页面声明日期、上传日期、最早存档 | E A C | 27 | 5 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| SEARCH open video | 读到的页面给出声明所说视频的链接时打开（hi@0.10.0 起自动带原语言字幕，原 include=subtitles 并入） | C | 0 | 0 | 条件未出现 |  |
| SEARCH open URL/R | 打开一个已知网页来源（非视频、非日期） | — | 0 | 0 | 条件未出现 |  |
| INSPECT get_frames count | 看不到原片对应时两侧均匀取帧；以图找帧前取样 | B | 7 | 1 | 够 | fmnv_a_6C0lUMKMj0w, fmnv_a_BMljmBWBhek, fmnv_a_BQBoqdkM4hw, fmnv_a_F51nH6oLGfY |
| INSPECT get_frames times_s | 在比对报出的边界 / 对应时刻两侧取帧核看 | B A | 73 | 8 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| INSPECT get_frames times_s+box | 同一时刻画面不同的小区域：两侧放大 | B A | 27 | 2 | 够 | videosham_0072, videosham_0090, videosham_0109, videosham_1021 |
| INSPECT get_frames pages | 读到的 PDF 没有文字层（扫描件）：渲染页面再看 | C E | 0 | 0 | 条件未出现 |  |
| INSPECT ocr(image) | 放大后的横条区域（字幕、标题、标牌）读字；给定图片读水印 / 字幕 / 标牌 | B E L | 6 | 6 | 够 | videosham_1056, videosham_4002, videosham_4140 |
| INSPECT ocr(video) | 找原片 / 核查声明：整段材料的画面文字（台标、字幕、名牌） | A C | 40 | 3 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| INSPECT asr | 材料有音轨且任务涉及说了什么，或需要语音里的线索 | A B C | 76 | 5 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| INSPECT locate(text) | 按描述在单段视频里找片段 | B(qv) | 0 | 1 | 不够 |  |
| INSPECT locate(image) | 对应类任务整段比对没对上：拿 A 的帧去 B 里找 | B(vcsl) | 0 | 1 | 不够 |  |
| INSPECT video_structure | 单段视频按描述定位时先看镜头表，按切点取边界帧 | B(qv) | 0 | 1 | 不够 |  |
| INSPECT compare_media | 两段视频（给定的 B / 找到的原片 / 页面链接的视频）整段比对；hi@0.10.0 起画面对不上且两边都有声音时自动只比声音（原 mode=audio 并入） | A B C | 47 | 6 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| INSPECT compare_media range | 等长短视频要逐帧精确边界时限定 A 全长再比 | B | 3 | 1 | 够 | videosham_0090, videosham_0099, videosham_4066 |
| READ_RECORD text | 检索自动打开的页面、打开视频带的字幕：读正文 | C E L | 0 | 5 | 不够 |  |
| READ_RECORD offset>0 | 正文没读完（next_offset 非空）且要求项还没覆盖 | C | 0 | 0 | 条件未出现 |  |
| READ_RECORD focus | 比对图 / 帧没附上时带 focus 回看；参考照片与给定照片并排看 | B L | 0 | 3 | 不够 |  |
| QUERY_MEMORY (HI) | 交报告前要引用的 E 已掉出视图（只剩编号）：按标签查回来；V9 变体（max_evidence_cards=6 + max_history_entries=4）里自然出现 | B(v9) | 0 | 1 | 不够 |  |
| GIA INVESTIGATE round1 | 每个案例第 1 轮 | 全部 | 60 | 1 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |
| GIA INVESTIGATE round2 | 报告留下能靠更多查看补上的缺口（B 族时间窗 / 音轨） | B | 0 | 1 | 不够 |  |
| GIA READ_RECORD | 报告引用的图不在本次附图里 | 全部 | 30 | 2 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_5A6U5EjH048, fmnv_a_A5dqLhDAnfQ |
| GIA QUERY_MEMORY | 第 2 轮之后第 1 轮报告不在视图里 | B(第 2 轮) | 0 | 0 | 条件未出现 |  |
| GIA FINISH | 每个案例结束 | 全部 | 60 | 1 | 够 | fmnv_a_0Uc3vwNlDdY, fmnv_a_0zC0K0YZEzg, fmnv_a_4aIK1tjpGuk, fmnv_a_5A6U5EjH048 |

不够的：SEARCH text→text、SEARCH text→image、INSPECT locate(text)、INSPECT locate(image)、INSPECT video_structure、READ_RECORD text、READ_RECORD focus、QUERY_MEMORY (HI)、GIA INVESTIGATE round2。补法只能是加案例或环境变体，不改 walker 去硬用。
