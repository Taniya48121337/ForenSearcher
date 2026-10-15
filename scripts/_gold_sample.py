"""人工金标集第一步（本机跑）：分层抽样 + 生成离线标注页。

    python scripts/_gold_sample.py [--findings 60] [--answers 12] [--plans 8] [--seed 20260928]
                                   [--report-verdicts experiments/sft_data/verdicts_r0_judged.jsonl]
                                   [--unit-verdicts experiments/sft_data/verdicts_units_judged.jsonl]
                                   [--out experiments/sft_data/gold_set/v1]

产物（--out 目录）：
  items.json        抽中的条目（人要看的内容：任务、Finding / 答案 / 计划、所引证据原文与图片路径）
  strata.json       每条的抽样层与核验器当时的判定（**标注页不读这个文件**，标注时看不到核验器结论）
  media/            所引图片的缩略副本（最长边 960px）
  标注.html          双击打开即可标注；进度存在浏览器本地；"导出"下载 labels_<标注人>.json

抽样：
  - Finding 级（报告单元的核心判断：结论是否被所引 E 支持）：按"核验器是否标了 major/critical 错误"分两层，各一半；
    层内按数据集轮转、每个 run 最多 3 条，避免集中在长 run 上。
  - 答案单元：units 判定里的 answer 全取（不足 --answers 时全取）。
  - 计划单元：plan 按 positive / negative 分层，negative 全取，其余按 seed 抽满 --plans。
  抽样只依赖 seed 与判定文件，可复现。

--plan-tasks（2026-09-29，v2 批次）：按"任务"抽，专核计划单元的两类任务级错误 premise_smuggling / task_not_gap_driven。
    python scripts/_gold_sample.py --plan-tasks [--controls N] --out experiments/sft_data/gold_set/v2_plan_tasks
  - 被核验器标了这两类错误的任务全取（按 run + request + T 编号去重）；
  - 对照：同一批计划单元里没被标这两类的任务，按 seed 抽 N 个（默认与被标数相同），混在一起按 seed 打乱，标注页看不出哪些被标过；
  - 每条问三件事：题面有没有把没证实的说法写成事实（并照抄片段）、是否针对此前报告没解决的缺口、这个任务该不该派。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SFT = ROOT / "experiments" / "sft_data"

from foren_searcher.datagen.cases import load_cases  # noqa: E402
from foren_searcher.datagen.runlog import load_run  # noqa: E402

TEXT_LIMIT = 2500
_REF = re.compile(r"\b([EF]\d+)\b")


def h(seed: str, *parts: Any) -> str:
    return hashlib.sha1("|".join([seed, *map(str, parts)]).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_dir(name: str) -> Path | None:
    for base in (SFT / "runs", SFT / "runs_offline"):
        if (base / name / "store.sqlite3").is_file():
            return base / name
    return None


class Media:
    def __init__(self, out: Path) -> None:
        self.dir = out / "media"
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image  # noqa: F401
            self.pil = True
        except ImportError:
            self.pil = False

    def copy(self, src: Path, name: str) -> str | None:
        if not src or not Path(src).is_file():
            return None
        target = self.dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".jpg")
        if not target.exists():
            if self.pil:
                from PIL import Image
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((960, 960))
                    im.save(target, "JPEG", quality=82)
            else:
                shutil.copy2(src, target)
        return f"media/{target.name}"


def evidence_card(log: Any, eid: str, media: Media, run: str) -> dict[str, Any]:
    try:
        t = log.lookup.get(eid)
    except Exception as exc:  # noqa: BLE001
        return {"id": eid, "missing": f"{type(exc).__name__}: {exc}"}
    content = t.content or ""
    card = {"id": eid, "label": t.label, "source_url": t.source_url, "acquired_by": t.acquired_by,
            "provenance": getattr(t, "provenance", None), "modalities": list(t.modalities),
            "text": content[:TEXT_LIMIT] + (f"\n…（原文共 {len(content)} 字，只显示前 {TEXT_LIMIT} 字）" if len(content) > TEXT_LIMIT else "")}
    if t.image_path:
        card["image"] = media.copy(t.image_path, f"{run}_{eid}")
    return card


def finding_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        subj = row["subject"]
        flagged: dict[str, list[str]] = defaultdict(list)
        for err in row.get("errors") or []:
            fid = (err.get("extra") or {}).get("finding")
            if fid and err.get("severity") in ("major", "critical"):
                flagged[fid].append(err["type"])
        fids = {c["finding_id"] for c in row.get("coverage") or []} | {j["finding_id"] for j in row.get("judgments") or []}
        for fid in sorted(fids, key=lambda x: int(x[1:])):
            out.append({"run": subj["run"], "dataset": subj.get("dataset") or subj["run"].split("_")[0],
                        "workspace": subj.get("workspace_id"), "finding": fid, "case_id": subj.get("case_id"),
                        "verifier_flagged": bool(flagged.get(fid)), "verifier_errors": flagged.get(fid, []),
                        "report_disposition": row.get("disposition"), "verifier": row.get("verifier"),
                        "judge_model": row.get("judge_model")})
    return out


def stratified(cands: list[dict[str, Any]], n: int, seed: str, per_run: int = 3) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in sorted(cands, key=lambda c: h(seed, c["run"], c["workspace"], c["finding"])):
        by_ds[c["dataset"]].append(c)
    picked, used = [], defaultdict(int)
    while len(picked) < n and any(by_ds.values()):
        for ds in sorted(by_ds):
            while by_ds[ds]:
                c = by_ds[ds].pop(0)
                if used[c["run"]] < per_run:
                    picked.append(c)
                    used[c["run"]] += 1
                    break
            if len(picked) >= n:
                break
    return picked


PLAN_TASK_TYPES = ("premise_smuggling", "task_not_gap_driven")


def write_batch(out: Path, meta: dict[str, Any], items: list[dict[str, Any]], strata: list[dict[str, Any]]) -> None:
    (out / "items.json").write_text(json.dumps({"meta": meta, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "strata.json").write_text(json.dumps({"meta": meta, "strata": strata}, ensure_ascii=False, indent=1), encoding="utf-8")
    template = (ROOT / "scripts" / "gold_annotate_template.html").read_text(encoding="utf-8")
    payload = json.dumps({"meta": meta, "items": items}, ensure_ascii=False).replace("</", "<\\/")
    (out / "标注.html").write_text(template.replace("/*__ITEMS__*/null", payload), encoding="utf-8")


def reports_seen_of(log: Any, view: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for rep in view.get("history_index") or []:
        if rep.get("record_type") == "report":
            fs = []
            for fid in rep.get("finding_ids") or []:
                try:
                    f = log.store.finding(fid)
                    fs.append({"id": fid, "statement": f.statement, "limitations": list(f.limitations)})
                except KeyError:
                    pass
            out.append({"task_id": rep.get("task_id"), "status": rep.get("status"), "task": rep.get("task"), "findings": fs})
    return out


def plan_task_flags(unit: dict[str, Any]) -> dict[str, list[str]]:
    """计划单元判定里，每个任务（T1…）被标的 PLAN_TASK_TYPES 错误。"""

    flags: dict[str, list[str]] = defaultdict(list)
    for err in unit.get("errors") or []:
        task = (err.get("extra") or {}).get("task")
        if err.get("type") in PLAN_TASK_TYPES and task:
            flags[str(task)].append(err["type"])
    return flags


def plan_tasks_batch(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if (out / "items.json").exists():
        raise SystemExit(f"{out} 已有 items.json，换一个 --out（不覆盖已开始标注的批次）")
    cases = load_cases(SFT / "cases")
    units = [u for u in load_jsonl(Path(args.unit_verdicts)) if u.get("unit") == "plan" and run_dir(u["subject"]["run"])]
    logs: dict[str, Any] = {}
    flagged, controls = [], []
    try:
        for u in units:
            subj = u["subject"]
            if subj["run"] not in logs:
                logs[subj["run"]] = load_run(run_dir(subj["run"]))
            log = logs[subj["run"]]
            d = next((d for d in log.decisions if d.request_id == subj.get("request") and d.kind == "plan"), None)
            if d is None:
                continue
            flags = plan_task_flags(u)
            tasks = d.output.get("tasks") or []
            for k, t in enumerate(tasks):
                label = f"T{k + 1}"
                row = {"unit": u, "decision": d, "log": log, "label": label, "task": t, "types": flags.get(label, [])}
                (flagged if row["types"] else controls).append(row)
        n_ctrl = len(flagged) if args.controls is None else args.controls
        controls = sorted(controls, key=lambda r: h(args.seed, r["unit"]["subject"]["run"], r["unit"]["subject"].get("request"), r["label"]))[:n_ctrl]
        rows = sorted(flagged + controls, key=lambda r: h(args.seed, "mix", r["unit"]["subject"]["run"], r["unit"]["subject"].get("request"), r["label"]))
        items, strata = [], []
        for i, r in enumerate(rows):
            subj, view = r["unit"]["subject"], r["decision"].view
            case = cases.get(subj.get("case_id") or "")
            tasks = r["decision"].output.get("tasks") or []
            item_id = f"T{i + 1:03d}"
            items.append({"id": item_id, "kind": "plan_task", "run": subj["run"], "case_id": subj.get("case_id"),
                          "request": subj.get("request"), "task_label": r["label"], "task": view.get("task"),
                          "round": (view.get("runtime_constraints") or {}).get("round_index"),
                          "requirements": [{"id": q.id, "text": q.text, "must_cite": q.must_cite} for q in case.requirements] if case else [],
                          "reports_so_far": reports_seen_of(r["log"], view),
                          "target": {"task": r["task"].get("task"), "materials": r["task"].get("materials")},
                          "siblings": [{"label": f"T{k + 1}", "task": t.get("task")} for k, t in enumerate(tasks) if f"T{k + 1}" != r["label"]]})
            strata.append({"id": item_id, "kind": "plan_task", "run": subj["run"], "request": subj.get("request"), "task_label": r["label"],
                           "stratum": "flagged" if r["types"] else "control", "verifier_types": r["types"],
                           "verifier_disposition": r["unit"].get("disposition"), "verifier": r["unit"].get("verifier"),
                           "judge_model": r["unit"].get("judge_model")})
    finally:
        for log in logs.values():
            log.close()
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter(t for s in strata for t in s["verifier_types"])
    meta = {"batch": out.name, "seed": args.seed, "mode": "plan_tasks", "unit_verdicts": Path(args.unit_verdicts).name,
            "counts": {"plan_task": len(items)}}
    write_batch(out, meta, items, strata)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"被标任务 {len(flagged)}（{dict(counts)}）+ 对照 {len(controls)}；标注页：{out / '标注.html'}")
    return 0


HARD_MUST = ("verbatim_but_unsupported", "repair_failed", "rubber_stamp")     # 方案 §2：全收
HARD_FILL = ("disagree", "unstable", "gold_conflict", "repair_fixed")       # 按模态分层补满


def from_hard_batch(args: argparse.Namespace) -> int:
    """v1' 定向批次（2026-10-02 方案 §2）：从 calibrate_precision 的 hard_items 抽。

    构成（总数 = --hard-total + --controls，默认 65 + 15）：
      1. 全部 verbatim_but_unsupported / repair_failed / rubber_stamp；
      2. 全部 firm_error（核验器对该单元记了 major/critical 错误）——这是"核验器说错"的总体，全标就是普查，Finding 精确率直接可算；
      3. 其余疑难原因（disagree / unstable / gold_conflict / repair_fixed）按 图片 / 文本 轮转补到 --hard-total；
      4. 视频帧单元不足 --min-frames 时，用含视频帧的疑难件 / 对照替换文本对照；
      5. 对照：四层全通过的单元按 seed 抽 --controls 条；全部按 seed 打乱，ID 一律 H001…，标注页看不出来源层。
    单元 = 报告 Finding 或答案句（答案句的条目带 context：整段答案，只帮助理解指代）；rubber_stamp 条目展示的是**替换后的**证据。
    核验结论与原因只进 strata.json。
    """

    sys.path.insert(0, str(ROOT / "src"))
    from foren_searcher.datagen.verify.calibrate_precision import open_run  # noqa: E402

    out = Path(args.out)
    if (out / "items.json").exists():
        raise SystemExit(f"{out} 已有 items.json，换一个 --out（不覆盖已开始标注的批次）")
    # 可以给多个 precision 文件（同一批 run、不同核验器版本）：疑难件取并集、对照取"各版都全通过"的交集，
    # 每条记下各版核验器的判定（strata.verifier_flags），标一次就能同时评几个版本。
    precs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.from_hard]
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for prec in precs:
        tag = (prec.get("meta") or {}).get("verifier") or "?"
        for x in prec["hard_items"]:
            k = (x["unit"], x["kind"])
            if k not in merged:
                merged[k] = {**x, "reasons": [], "details": {}}
            m = merged[k]
            m["reasons"] += [f"{r}@{tag}" for r in x["reasons"] if f"{r}@{tag}" not in m["reasons"]]
            for r, d in (x.get("details") or {}).items():
                m["details"][f"{r}@{tag}"] = d
            m["video_frames"] = m["video_frames"] or x["video_frames"]
    hard = list(merged.values())
    for x in hard:
        x["base_reasons"] = sorted({r.split("@")[0] for r in x["reasons"]})
    prec = precs[0]
    units_meta = {u["unit"]: u for p in precs for u in p.get("units") or []}
    common_ctrl = set(precs[0].get("controls") or [])
    for p in precs[1:]:
        common_ctrl &= set(p.get("controls") or [])
    prec = {**prec, "controls": [k for k in prec.get("controls") or [] if k in common_ctrl]}
    versions = []
    for p in precs:
        meta_p = p.get("meta") or {}
        versions.append((meta_p.get("verifier") or "?",
                         {(r["subject"]["run"], r["subject"].get("workspace_id")): r for r in load_jsonl(SFT / meta_p["report_verdicts"])},
                         {(u["subject"]["run"], u["subject"].get("request")): u for u in load_jsonl(SFT / meta_p["unit_verdicts"]) if u.get("unit") == "answer"}))

    def order(x: dict[str, Any]) -> str:
        return h(args.seed, x["unit"], x["kind"])

    must = [x for x in hard if set(x["base_reasons"]) & set(HARD_MUST) or "firm_error" in x["base_reasons"]]
    rest = sorted((x for x in hard if x not in must), key=order)
    by_mod: dict[str, list[dict[str, Any]]] = {"image": [x for x in rest if x["image"]], "text": [x for x in rest if not x["image"]]}
    picked = sorted(must, key=order)
    while len(picked) < args.hard_total and any(by_mod.values()):
        for m in ("image", "text"):
            if by_mod[m] and len(picked) < args.hard_total:
                picked.append(by_mod[m].pop(0))
    hard_keys = {x["unit"] for x in hard}
    ctrl_pool = sorted((k for k in prec.get("controls") or [] if k not in hard_keys), key=lambda k: h(args.seed, "ctrl", k))
    frames_ctrl = [k for k in ctrl_pool if (units_meta.get(k) or {}).get("video_frames")]
    other_ctrl = [k for k in ctrl_pool if k not in frames_ctrl]
    frames_now = sum(1 for x in picked if x["video_frames"])
    need_frames = max(0, args.min_frames - frames_now)
    ctrls = frames_ctrl[:min(need_frames, args.controls)]
    ctrls += [k for k in (other_ctrl + frames_ctrl[len(ctrls):]) if k not in ctrls][: args.controls - len(ctrls)]

    out.mkdir(parents=True, exist_ok=True)
    media = Media(out)
    reports = {(r["subject"]["run"], r["subject"].get("workspace_id")): r for r in load_jsonl(Path(args.report_verdicts))}
    answers = {(u["subject"]["run"], u["subject"].get("request")): u for u in load_jsonl(Path(args.unit_verdicts)) if u.get("unit") == "answer"}
    runs: dict[str, Any] = {}

    def ctx(run: str) -> Any:
        if run not in runs:
            runs[run] = open_run(run_dir(run))
        return runs[run]

    def card(c: Any, eid: str, *, override: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            t = c.lookup.get(eid)
        except KeyError as exc:
            return {"id": eid, "missing": f"KeyError: {exc}"}
        content = t.content or ""
        out_card = {"id": eid, "label": t.label, "source_url": t.source_url, "acquired_by": t.acquired_by,
                    "provenance": t.provenance or None, "modalities": list(t.modalities),
                    "text": content[:TEXT_LIMIT] + (f"\n…（原文共 {len(content)} 字，只显示前 {TEXT_LIMIT} 字）" if len(content) > TEXT_LIMIT else "")
                    if t.is_text else (t.media_note or "")}
        if t.image_path:
            out_card["image"] = media.copy(t.image_path, f"{c.name}_{eid}")
        if override:
            if override.get("donor_image"):
                out_card["image"] = media.copy(Path(override["donor_image"]), f"swap_{Path(override['donor_image']).stem}")
                out_card["text"] = ""
                out_card["provenance"] = None
            if override.get("donor_text"):
                out_card["text"] = override["donor_text"][:TEXT_LIMIT]
                out_card["label"], out_card["source_url"] = "（来源）", None
        return out_card

    def build(entry: dict[str, Any] | None, unit_key: str, item_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        run, scope_part, fid = unit_key.split("|")
        kind, scope = scope_part.split(":", 1)
        c = ctx(run)
        if kind == "report":
            row = reports[(run, scope)]
            f = c.store.finding(fid)
            try:
                task = str(c.store.workspace_task_record(scope).get("task") or "")
            except KeyError:
                task = ""
            statement, limitations, context = f.statement, list(f.limitations), None
            judged = [j["evidence_id"] for j in row.get("judgments") or [] if j["finding_id"] == fid]
        else:
            row = answers[(run, scope)]
            sents = ((row.get("extra") or {}).get("anchored") or {}).get("sentences") or []
            s = next(x for x in sents if x["id"] == fid)
            statement, limitations, task = s["text"], [], str(c.config.get("task") or "")
            context = "".join(x["text"] for x in sents)
            judged = [j["evidence_id"] for j in row.get("judgments") or [] if j["finding_id"] == fid]
        swap = entry is not None and entry["kind"] == "swap"
        if swap:
            d = next(v for k, v in entry["details"].items() if k.split("@")[0] == "rubber_stamp")[0]
            evidence = [card(c, d["evidence_id"], override={"donor_image": d.get("donor_image"), "donor_text": d.get("donor_text_head")})]
        else:
            evidence = [card(c, e) for e in judged]
        item = {"id": item_id, "kind": "finding", "run": run, "case_id": c.case_id, "workspace": scope if kind == "report" else None,
                "finding_id": fid, "task": task, "statement": statement, "limitations": limitations, "evidence": evidence}
        if context:
            item["context"] = context
        def firm_of(r: dict[str, Any]) -> list[str]:
            return [e["type"] for e in r.get("errors") or [] if e.get("severity") in ("major", "critical")
                    and ((e.get("extra") or {}).get("finding") == fid or (e.get("extra") or {}).get("sentence") == fid)]

        firm = firm_of(row)
        flags = {}
        for tag, reps, ans in versions:
            r = reps.get((run, scope)) if kind == "report" else ans.get((run, scope))
            if r is not None:
                flags[tag] = {"flagged": False if swap else bool(firm_of(r)), "errors": [] if swap else firm_of(r)}
        stratum = {"id": item_id, "kind": "finding", "unit": unit_key, "unit_kind": kind, "run": run, "dataset": c.dataset,
                   "workspace": scope if kind == "report" else None, "request": scope if kind == "answer" else None,
                   "finding": fid, "case_id": c.case_id, "stratum": "swap" if swap else ("hard" if entry else "control"),
                   "reasons": entry["reasons"] if entry else [], "details": entry["details"] if entry else {},
                   # swap 条目：核验器在替换后的证据上判了 supported（= 没标错）；人工判"不支持"即证实盖章
                   "verifier_flagged": False if swap else bool(firm), "verifier_errors": [] if swap else firm,
                   "video_frames": bool(entry["video_frames"]) if entry else bool((units_meta.get(unit_key) or {}).get("video_frames")),
                   "verifier_flags": flags, "verifier": row.get("verifier"), "judge_model": row.get("judge_model")}
        return item, stratum

    rows = [(x, x["unit"]) for x in picked] + [(None, k) for k in ctrls]
    rows = sorted(rows, key=lambda r: h(args.seed, "mix", r[1], r[0]["kind"] if r[0] else "ctrl"))
    items, strata = [], []
    try:
        for i, (entry, key) in enumerate(rows):
            item, stratum = build(entry, key, f"H{i + 1:03d}")
            items.append(item)
            strata.append(stratum)
    finally:
        for c in runs.values():
            c.close()
    reasons = Counter(r for st_ in strata for r in st_["reasons"])
    meta = {"batch": out.name, "seed": args.seed, "mode": "from_hard", "precision_files": [Path(p).name for p in args.from_hard],
            "report_verdicts": Path(args.report_verdicts).name, "unit_verdicts": Path(args.unit_verdicts).name,
            "counts": {"finding": len(items)}}
    write_batch(out, meta, items, strata)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"疑难 {sum(s['stratum'] != 'control' for s in strata)}（原因 {dict(reasons)}）+ 对照 {sum(s['stratum'] == 'control' for s in strata)}；"
          f"核验器标错 {sum(s['verifier_flagged'] for s in strata)}；含视频帧 {sum(s['video_frames'] for s in strata)}；标注页：{out / '标注.html'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--findings", type=int, default=60)
    parser.add_argument("--answers", type=int, default=12)
    parser.add_argument("--plans", type=int, default=8)
    parser.add_argument("--seed", default="20260928")
    parser.add_argument("--report-verdicts", default=str(SFT / "verdicts_r0_judged.jsonl"))
    parser.add_argument("--unit-verdicts", default=str(SFT / "verdicts_units_judged.jsonl"))
    parser.add_argument("--out", default=str(SFT / "gold_set" / "v1"))
    parser.add_argument("--plan-tasks", action="store_true", help="v2：按任务抽，专核 premise_smuggling / task_not_gap_driven")
    parser.add_argument("--controls", type=int, default=None, help="--plan-tasks 的对照任务数（默认与被标任务数相同）；--from-hard 默认 15")
    parser.add_argument("--from-hard", nargs="+", default=None,
                        help="v1'：从 calibrate_precision 的 precision_*.json 的 hard_items 抽定向批次；可给多个版本（并集，各版判定都记进 strata）")
    parser.add_argument("--hard-total", type=int, default=65)
    parser.add_argument("--min-frames", type=int, default=20, help="--from-hard：视频帧单元至少这么多条")
    parser.add_argument("--oversample-flagged", action="store_true",
                        help="核验器标了 major/critical 的 Finding 全取，不足 --findings 再按原分层补（10-15 对话 5''）")
    parser.add_argument("--only-cases", default=None, help="只抽这份案卷清单里的案例（experiments/sft_data 下的 JSON 列表）")
    args = parser.parse_args()
    if args.plan_tasks:
        return plan_tasks_batch(args)
    if args.from_hard:
        args.controls = 15 if args.controls is None else args.controls
        return from_hard_batch(args)
    out = Path(args.out)
    if (out / "items.json").exists():
        raise SystemExit(f"{out} 已有 items.json，换一个 --out（不覆盖已开始标注的批次）")
    out.mkdir(parents=True, exist_ok=True)
    media = Media(out)
    cases = load_cases(SFT / "cases")
    reports = load_jsonl(Path(args.report_verdicts))
    units = load_jsonl(Path(args.unit_verdicts))

    cands = [c for c in finding_candidates(reports) if run_dir(c["run"])]
    if args.only_cases:
        # 10-15 对话 5''：只抽这些案卷的 run（r0g 补金标只抽新案例）；清单是 walker_d2_cases.py 写的案卷名列表
        names = set(json.loads((SFT / args.only_cases).read_text(encoding="utf-8")))
        keep = lambda run: any(run.startswith(n + "_") and run[len(n) + 1:].split("_")[0].startswith("r0") for n in names)  # noqa: E731
        cands = [c for c in cands if keep(c["run"])]
        units = [u for u in units if keep(u["subject"]["run"])]
    if args.oversample_flagged:
        # 10-15 对话 5''（追加 10）：核验器标了 major/critical 的 Finding 全取（r0a 只有 3 条，精确率测不出来），不足再按原分层补
        flagged = [c for c in cands if c["verifier_flagged"]]
        picks = stratified(flagged, len(flagged), args.seed, per_run=10 ** 6)
        picks += stratified([c for c in cands if not c["verifier_flagged"]], max(0, args.findings - len(picks)), args.seed)
    else:
        half = args.findings // 2
        picks = stratified([c for c in cands if c["verifier_flagged"]], half, args.seed)
        picks += stratified([c for c in cands if not c["verifier_flagged"]], args.findings - len(picks), args.seed)
    answers = [u for u in units if u.get("unit") == "answer"]
    answers = sorted(answers, key=lambda u: h(args.seed, u["subject"]["run"], u["subject"].get("request")))[: args.answers]
    plans = [u for u in units if u.get("unit") == "plan"]
    neg = [u for u in plans if u.get("disposition") != "positive"]
    pos = sorted((u for u in plans if u.get("disposition") == "positive"), key=lambda u: h(args.seed, u["subject"]["run"], u["subject"].get("request")))
    plans = (neg + pos)[: max(args.plans, len(neg))]

    logs: dict[str, Any] = {}

    def log_of(run: str) -> Any:
        if run not in logs:
            logs[run] = load_run(run_dir(run))
        return logs[run]

    items, strata = [], []
    try:
        for i, c in enumerate(picks):
            log = log_of(c["run"])
            f = log.store.finding(c["finding"])
            try:
                task = str(log.store.workspace_task_record(f.workspace_id).get("task") or "")
            except KeyError:
                task = ""
            item_id = f"F{i + 1:03d}"
            items.append({"id": item_id, "kind": "finding", "run": c["run"], "case_id": c["case_id"],
                          "workspace": f.workspace_id, "finding_id": f.finding_id, "task": task,
                          "statement": f.statement, "limitations": list(f.limitations),
                          "evidence": [evidence_card(log, e, media, c["run"]) for e in f.evidence_ids]})
            strata.append({"id": item_id, **c})
        for i, u in enumerate(answers):
            subj = u["subject"]
            log = log_of(subj["run"])
            d = next(d for d in log.decisions if d.request_id == subj.get("request") and d.kind == "answer")
            answer = str(d.output.get("answer") or "")
            refs = sorted(set(_REF.findall(answer)), key=lambda x: (x[0], int(x[1:])))
            f_cards, e_ids = [], []
            for r in refs:
                if r.startswith("F"):
                    try:
                        f = log.store.finding(r)
                        f_cards.append({"id": r, "workspace": f.workspace_id, "statement": f.statement,
                                        "limitations": list(f.limitations), "evidence_ids": list(f.evidence_ids)})
                    except KeyError:
                        f_cards.append({"id": r, "missing": "store 里没有这个 Finding"})
                else:
                    e_ids.append(r)
            case = cases.get(subj.get("case_id") or "")
            item_id = f"A{i + 1:03d}"
            items.append({"id": item_id, "kind": "answer", "run": subj["run"], "case_id": subj.get("case_id"),
                          "request": subj.get("request"), "task": d.view.get("task"), "answer": answer,
                          "requirements": [{"id": r.id, "text": r.text, "must_cite": r.must_cite} for r in case.requirements] if case else [],
                          "findings": f_cards, "evidence": [evidence_card(log, e, media, subj["run"]) for e in e_ids]})
            strata.append({"id": item_id, "kind": "answer", "run": subj["run"], "request": subj.get("request"),
                           "verifier_disposition": u.get("disposition"), "verifier_errors": [e["type"] for e in u.get("errors") or []],
                           "verifier": u.get("verifier"), "judge_model": u.get("judge_model")})
        for i, u in enumerate(plans):
            subj = u["subject"]
            log = log_of(subj["run"])
            d = next(d for d in log.decisions if d.request_id == subj.get("request") and d.kind == "plan")
            view = d.view
            reports_seen = []
            for rep in view.get("history_index") or []:
                if rep.get("record_type") == "report":
                    fs = []
                    for fid in rep.get("finding_ids") or []:
                        try:
                            f = log.store.finding(fid)
                            fs.append({"id": fid, "statement": f.statement, "limitations": list(f.limitations)})
                        except KeyError:
                            pass
                    reports_seen.append({"task_id": rep.get("task_id"), "status": rep.get("status"), "task": rep.get("task"), "findings": fs})
            item_id = f"P{i + 1:03d}"
            items.append({"id": item_id, "kind": "plan", "run": subj["run"], "case_id": subj.get("case_id"),
                          "request": subj.get("request"), "task": view.get("task"),
                          "round": (view.get("runtime_constraints") or {}).get("round_index"),
                          "reports_so_far": reports_seen,
                          "proposed_tasks": [{"task": t.get("task"), "materials": t.get("materials")} for t in d.output.get("tasks") or []]})
            strata.append({"id": item_id, "kind": "plan", "run": subj["run"], "request": subj.get("request"),
                           "verifier_disposition": u.get("disposition"), "verifier_errors": [e["type"] for e in u.get("errors") or []],
                           "verifier": u.get("verifier"), "judge_model": u.get("judge_model")})
    finally:
        for log in logs.values():
            log.close()

    meta = {"batch": out.name, "seed": args.seed, "report_verdicts": Path(args.report_verdicts).name,
            "unit_verdicts": Path(args.unit_verdicts).name,
            "counts": {k: sum(1 for it in items if it["kind"] == k) for k in ("finding", "answer", "plan")}}
    write_batch(out, meta, items, strata)
    flagged = sum(1 for s in strata if s.get("verifier_flagged"))
    print(json.dumps(meta, ensure_ascii=False))
    print(f"Finding 层：核验器标错 {flagged} / 未标错 {meta['counts']['finding'] - flagged}；图片 {len(list(media.dir.glob('*.jpg')))} 张")
    print(f"标注页：{out / '标注.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
