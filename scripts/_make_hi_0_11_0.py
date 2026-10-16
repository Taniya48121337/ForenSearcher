"""r1（10-17）：hi@0.11.0 = hi@0.10.0 + teacher_instruction 末尾两句（以图搜的缩略图哈希怎么读）。gia@0.9.0 不动。

    python scripts\\_make_hi_0_11_0.py          # 写 specs\\hi@0.11.0\\，打印新 hash

student_prompt / rubric / harness_params 与 0.10.0 逐字相同（student_prompt 要与运行时 _hi_turn_prompt 逐字一致，改了它
就得改运行时提示；这两句是给教师槽的写作约束，与 0.10.0 只改 teacher_instruction 的做法相同）。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.specs import Spec  # noqa: E402

ADDED = ("\n\nReverse image search results may say how the page's thumbnail compares with the query image. A near-identical "
         "hash does not show that the copy is unedited: compare that page's picture with the given image using compare_media "
         "to find local differences. A hash mismatch can come from cropping, an added border or a local edit, and does not "
         "show that the page lacks the picture.")


def main() -> int:
    parent = Spec.load(ROOT / "specs" / "hi@0.10.0")
    child = replace(
        parent, version="0.11.0", parent="0.10.0", teacher_instruction=parent.teacher_instruction + ADDED,
        changelog=tuple(parent.changelog) + ({
            "targets": ["unsupported_assertion", "omitted_required_fact", "completed_without_finishing"],
            "rationale": "r1（10-17，claude/r1_walker与search整合_20261017.md 六节）：teacher_instruction 末尾加两句——缩略图哈希"
                         "近同不等于没篡改，用 compare_media 查局部差异；哈希不匹配可能是裁切 / 加框 / 局部改动，不等于页面没有这张图"
                         "（10-16 实测：e107 真来源加框贴旗被判 no match；判 same 的里有局部改过的副本）。student_prompt / rubric / "
                         "harness_params 同 0.10.0。版本号按负责人要求升 minor（validate_bump 对只改教师指令的版本要求 patch，见报告）。",
        },),
        notes=parent.notes + " 0.11.0：只改 teacher_instruction（末尾两句，以图搜缩略图哈希的读法），其余同 0.10.0。",
    )
    out = child.save(ROOT / "specs")
    print(out, child.hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
