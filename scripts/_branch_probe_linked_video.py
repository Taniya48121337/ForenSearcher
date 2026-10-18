"""分支可达探测：C8 页内视频（C 族 C8_open_linked_video → SEARCH open video（source=URL, output_type=video）
→ C8b_read_subtitles → C8c_compare_linked_video）。

触发条件（walker._linked_video_step）：C 族已读过的页面正文里出现视频链接（YouTube watch / shorts、youtu.be、Vimeo、
Dailymotion，= walker._VIDEO_LINK），且还剩 ≥ 4 轮——walker 打开它一次（自动带原语言字幕），字幕取到就读，再与材料视频
比对一次。探测：C 族首条查询（替身 = 声明前 12 词）前 N 条结果打开后正文里有没有这类链接；AVeriTeC 另看 GT 来源页。
这同时是覆盖表里"SEARCH output_type=video"在 C 族的那一支（A 族那一支见 _branch_probe_text_video / text_image_video）。
走 walker 同一条 search 层与页面打开器（ddgs 文字搜，免费），不打开视频、不发图，Serper 0 点；与另两个 C 探测共用扫描缓存。

    python scripts/_branch_probe_linked_video.py --n 10
输出 claude/_longtail_work/probe_linked_video.json
"""

from _branch_probe_common import c_probe_main

if __name__ == "__main__":
    raise SystemExit(c_probe_main("video_links", "probe_linked_video.json", __doc__, 5,
                                  "条件 = 读到的页面正文含视频链接；C8c 还要材料是视频（AVeriTeC 文字声明只到 C8 / C8b）"))
