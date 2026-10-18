"""分支可达探测：READ_RECORD offset>0（C 族 C5_read_more，正文没读完时续读）。

触发条件（walker._decide_claim → S3_next 槽）：读过的页面 next_offset 非空（正文长于一页 = specs.read_page_chars 6000 字），
S3 槽给出 offsets 清单，**教师在"读 / 搜 / 查日期 / 交报告"里选了续读**才出这一行。所以探测只能确认"条件会出现"
（前 N 条结果里有长于 6000 字的页面），选不选是教师的事：r0g C 族 36 run 续读 24 次，r1 同样 36 个案例只有 3 次
（r1 加了 C2v / C2w / C9v 程序化看帧，S3 轮次从 246 降到 178），g1（Codex 教师）100 个 C run 4 次。
走 walker 同一条 search 层与页面打开器（ddgs 文字搜，免费），Serper 0 点；与另两个 C 探测共用扫描缓存。

    python scripts/_branch_probe_read_offset.py --n 10
输出 claude/_longtail_work/probe_read_offset.json
"""

from _branch_probe_common import c_probe_main

if __name__ == "__main__":
    raise SystemExit(c_probe_main("long", "probe_read_offset.json", __doc__, 5,
                                  "条件 = 页面正文 > 6000 字；行要 S3 教师选续读，条件出现不等于行出现"))
