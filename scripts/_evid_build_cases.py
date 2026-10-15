"""EVID-Bench → ForenSearcher 案卷（2026-10-04）。

    python scripts/_evid_build_cases.py                      # 222 个案卷写到 experiments/evid/cases/
    python scripts/_evid_build_cases.py --check              # 只检查：标注、视频文件、已有案卷是否一致，不写文件

为什么单独放 experiments/evid/：EVID-Bench 是锁定的 test 集，不能混进 experiments/sft_data/cases/（那里的划分、
splits_manifest、金标抽样都按 dev 案卷算）。run_cases 用 --cases experiments/evid/cases --split test 指过来即可。

案卷只写演员能看到的东西：任务文字 + 视频。真值（groundtruth、source YouTube ID）不进案卷正文，评分时
scripts/_evid_score.py 直接读 evid_bench.json；案卷里只留 reference_path 指过去（演员不可见）。

任务文字按 EVID 官方任务描述写（可疑视频 → 找原始素材 → 比较 → 3–5 个伪造点），不加任何针对具体样本或
具体伪造类型的提示；输出格式只要求编号列表，方便评分时切分成点。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.cases import Case, Media, SourcePolicy  # noqa: E402

DEFAULT_DATASET = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench"
# 看过轨迹、并据此改过规格的样本（cases.py：test 案卷 used_for_revision 非空 → 自动降级为 dev_train，不再计入 test）。
# 10-05：冒烟三个样本的检索轨迹促成 hi@0.5.0 / gia@0.6.0（查询不写内部编号、出处类任务先找原件）。
USED_FOR_REVISION: dict[str, tuple[str, ...]] = {
    "video_001": ("hi@0.5.0", "gia@0.6.0"),
    "video_027": ("hi@0.5.0", "gia@0.6.0"),
    "video_151": ("hi@0.5.0", "gia@0.6.0"),
}
DEFAULT_OUT = ROOT / "experiments" / "evid" / "cases"

# 10-05 改（evidb5 起）：旧版第一类就举"换脸 / 加物体 / 插入合成片段"等例子，8B 把它们套到剪辑视频上——全量 94 个剪辑类
# 样本里 64 个答案声称 AI 生成或换脸；另有 46 个答"未发现篡改 / 无法确认"，一半的点不写误导效果（官方评分的主维度）。
# 现在：只列三大类、不举具体手法；说明材料确实被改过、不要答"没有篡改"；每点写清原片 / 改动 / 误导效果；英文作答。
TASK_TEXT = (
    "Material A is a video that has been manipulated to mislead viewers. The manipulation may come from editing "
    "footage of a single source, from splicing footage of several sources into one story, or from AI generation or "
    "alteration. Do not assume which kind it is until you have evidence.\n\n"
    "Investigate Material A. Try to find the original source footage online and compare it with Material A. Whether "
    "or not the original is found, examine the whole of Material A (its scenes and their order, captions, narration "
    "and audio) and work out what false impression it is built to create.\n\n"
    "Final answer, in English: a numbered list of 3 to 5 forgery points (\"1. ...\", \"2. ...\"). Each point is one "
    "self-contained paragraph that states (a) what the original footage shows or most likely shows, (b) what was "
    "changed in Material A and how, and (c) the false impression or misleading conclusion this creates for viewers. "
    "Each point should cover a different aspect, and the last point should summarize the overall manipulation and the "
    "misleading message it serves. Only describe manipulations supported by what you observed in Material A or in "
    "sources you found; do not list a kind of manipulation just because it is possible. Material A has been "
    "manipulated, so do not answer that nothing was changed: if the original cannot be found, give the most likely "
    "forgery points based on Material A itself and say which points could not be verified against a source."
)


def build(entry: dict, *, dataset_root: Path, out_dir: Path, manifest_path: Path,
          used_for_revision: tuple[str, ...] = ()) -> Case:
    sample_id = str(entry["id"])
    video = dataset_root / "videos" / f"{sample_id}.mp4"
    rel_video = Path(os.path.relpath(video, out_dir)).as_posix()
    sources = sorted(str(s) for s in entry.get("source") or [])
    task = entry.get("task") or {}
    return Case(
        case_id=f"evid:{sample_id}",
        dataset="evid",
        source_id=sample_id,
        split="test",
        task_kind="provenance",
        task_text=TASK_TEXT,
        # 同一份原始素材做出的多个伪造视频算一个 family（防止以后有人把其中一个挪进 dev）
        family_key="evid_src:" + "+".join(sources) if sources else f"evid:{sample_id}",
        media=(Media(ref="V0", path=rel_video, media_type="video/mp4"),),
        source_policy=SourcePolicy(mode="unrestricted"),
        reference_path=Path(os.path.relpath(manifest_path, out_dir)).as_posix(),
        used_for_revision=used_for_revision,
        notes=f"EVID-Bench {task.get('category')}/{task.get('type')}，topic={entry.get('topic')}；"
              f"GT {len(entry.get('groundtruth') or [])} 点。真值只在 reference_path，案卷正文不含。",
    )


def _existing_revisions(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    try:
        return tuple(json.loads(path.read_text(encoding="utf-8")).get("used_for_revision") or ())
    except ValueError:
        return ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EVID-Bench → ForenSearcher 案卷（split=test）")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET), help="含 evid_bench.json 与 videos/ 的目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="案卷输出目录")
    parser.add_argument("--check", action="store_true", help="只检查，不写文件")
    args = parser.parse_args(argv)

    dataset_root, out_dir = Path(args.dataset_root).resolve(), Path(args.out).resolve()
    manifest_path = dataset_root / "evid_bench.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_video, written, changed, same = [], 0, 0, 0
    for entry in entries:
        target = out_dir / f"evid_{entry['id']}.json"
        used = _existing_revisions(target) + USED_FOR_REVISION.get(str(entry["id"]), ())
        case = build(entry, dataset_root=dataset_root, out_dir=out_dir, manifest_path=manifest_path,
                     used_for_revision=tuple(dict.fromkeys(used)))      # 已有标记只增不减
        if not (dataset_root / "videos" / f"{entry['id']}.mp4").is_file():
            missing_video.append(entry["id"])
        text = json.dumps(case.to_dict(), ensure_ascii=False, indent=2)
        if target.is_file():
            if target.read_text(encoding="utf-8") == text:
                same += 1
                continue
            changed += 1
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            written += 1
    degraded = sorted(e["id"] for e in entries if _existing_revisions(out_dir / f"evid_{e['id']}.json")
                      or USED_FOR_REVISION.get(str(e["id"])))
    print(f"降级出 test（used_for_revision 非空）：{len(degraded)} 个 {degraded}；test 剩 {len(entries) - len(degraded)} 个")
    print(f"EVID-Bench 标注 {len(entries)} 条；视频缺失 {len(missing_video)}" + (f"：{missing_video[:10]}" if missing_video else ""))
    if args.check:
        print(f"已有案卷：一致 {same}，不一致 {changed}，未生成 {len(entries) - same - changed}（--check 不写文件）")
    else:
        print(f"案卷目录 {out_dir}：新写 / 覆盖 {written}（其中覆盖 {changed}），未变 {same}")
    return 1 if missing_video else 0


if __name__ == "__main__":
    raise SystemExit(main())
