"""案卷 × 检索护栏对照（2026-09-29）：每个案卷的必需证据组在"核查站点黑名单 + 来源排除"下还能不能取到，
以及若按声明日期截断，各证据项从 URL 能看出的发布日期在声明日期之前还是之后。只读，不改案卷。

    python scripts/_guard_case_audit.py [--cases experiments/sft_data/cases] [--out 报告.md]

判定：
- 被拦：url_pattern（把 * 换成占位）命中 FACT_CHECK_DOMAINS，或命中案卷 source_policy.excluded_keys。
- 组不可达：组内含 must_cite=True 的项，且组内所有项都被拦（acquired_by 类证据恒可达）。可选组（全部 must_cite=False）只列出不算问题。
- 日期：只从 URL 猜——/YYYY/MM/、-YYYY-MM-DD、DDMMYY 段、推特 status 号（snowflake）；猜不到记"?"。声明日期取自题面"（声明日期：…）"。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.cases import load_cases  # noqa: E402
from foren_searcher.search.retrieval_guard import fact_check_match  # noqa: E402

CLAIM_DATE = re.compile(r"声明日期：(\d{4}-\d{2}-\d{2})")


def url_date(pattern: str) -> date | None:
    m = re.search(r"status/(\d{15,20})", pattern)
    if m:
        ms = (int(m.group(1)) >> 22) + 1288834974657
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
    m = re.search(r"(?<!\d)(20\d{2}|19\d{2})[/-](\d{1,2})(?:[/-](\d{1,2}))?(?!\d)", pattern)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
        except ValueError:
            return None
    m = re.search(r"/(\d{2})(\d{2})(\d{2})/", pattern)          # deccanchronicle 式 DDMMYY
    if m:
        try:
            return date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def blocked(pattern: str | None, excluded: tuple[str, ...]) -> str | None:
    if not pattern:
        return None
    probe = pattern.replace("[?]", "?").replace("*", "x").lstrip("x")
    hit = fact_check_match(probe)
    if hit:
        return f"黑名单 {hit}"
    key = probe.lower().removeprefix("www.")
    for ex in excluded:
        if key.startswith(ex.lower().rstrip("/")):
            return "排除原始核查"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(ROOT / "experiments" / "sft_data" / "cases"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    lines = ["| 案卷 | 声明日期 | 组 | 项（must / 被拦 / URL 日期） | 结论 |", "|---|---|---|---|---|"]
    problems = 0
    for cid, case in sorted(cases.items()):
        if not case.required_evidence:
            continue
        m = CLAIM_DATE.search(case.task_text)
        cdate = date.fromisoformat(m.group(1)) if m else None
        req = {r.key: r for r in case.required_evidence}
        for mode, keys in case.evidence_groups():
            cells, reachable, before = [], [], []
            for k in keys:
                r = req[k]
                why = None if r.acquired_by else blocked(r.url_pattern, case.source_policy.excluded_keys)
                d = url_date(r.url_pattern or "") if r.url_pattern else None
                rel = "?" if d is None or cdate is None else ("前" if d <= cdate else "后")
                cells.append(f"{k}（{'M' if r.must_cite else 'o'}{' / ' + why if why else ''}{' / ' + d.isoformat() + rel if d else ''}）")
                if not why:
                    reachable.append(k)
                    if r.acquired_by or rel in ("前", "?"):
                        before.append(k)
            must = any(req[k].must_cite for k in keys)
            if not must:
                verdict = "可选组"
            elif not reachable:
                verdict = "**不可达**"
                problems += 1
            elif cdate and not before:
                verdict = "可达；按声明日期截断则全部在后"
            else:
                verdict = "可达"
            lines.append(f"| {cid} | {cdate or ''} | {mode} | {'；'.join(cells)} | {verdict} |")
    text = "\n".join(lines) + f"\n\n不可达的必需组：{problems}\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
