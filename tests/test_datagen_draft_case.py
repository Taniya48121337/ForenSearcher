"""案卷自动起草（datagen/draft_case.py）的测试：纯函数 + 对 true_4124903_r1 的真实 run 起草。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foren_searcher.datagen.cases import Case, RequiredEvidence, load_cases
import datetime as dt

from foren_searcher.datagen.draft_case import (answer_segments, classify, classify_media, classify_media_answer, date_check,
                                               date_in_text, draft_case, is_media_task, main, media_task_kind, read_review,
                                               site_key, split_clauses, url_key_and_pattern, _subsplit_media)
from foren_searcher.datagen.runlog import load_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "sft_data" / "runs"
RUN = RUNS / "true_4124903_r1"
VERDICTS = ROOT / "experiments" / "sft_data" / "verdicts_r0_judged.jsonl"
UNITS = ROOT / "experiments" / "sft_data" / "verdicts_units_judged.jsonl"
CASES = ROOT / "experiments" / "sft_data" / "cases"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not VERDICTS.is_file() or not UNITS.is_file(),
                               reason="试点 run / 判定文件未随仓库入库")


# ------------------------------------------------------------------ 纯函数

def test_classify_ignores_quoted_claim_text():
    assert classify("判断其是否对应“法国国庆日期间埃菲尔铁塔附近出现火焰和浓烟”的场景") == "media"   # 引号里的"日期"不算
    assert classify("重点查证外部可靠来源对相关火情原因及是否涉及恐怖袭击的记载") == "external"
    assert classify("核查材料 A 视频的来源与事件语境") == "provenance"
    assert classify("给定视频画面确实显示铁塔附近有烟雾，但画面本身不能确定起火原因") == "media"
    assert classify("随便说点什么") is None


def test_split_clauses_keeps_quotes_whole():
    assert split_clauses("核查声明“甲，乙”，以及丙。注意丁") == ["核查声明“甲，乙”", "以及丙", "注意丁"]


def test_answer_segments_by_citation_brackets():
    segs = answer_segments("结论：不实。警方当日通报起火。（E24、E21）视频画面显示烟雾。（E3）上传日期为某日。(E16)")
    assert [s.evidence_ids for s in segs] == [["E24", "E21"], ["E3"], ["E16"]]
    assert [s.category for s in segs] == ["external", "media", "provenance"]


def test_url_patterns_match_their_sources():
    key, pattern = url_key_and_pattern(["https://x.com/prefpolice/status/753719036245073920"])
    assert key == "prefpolice_tweets"
    req = RequiredEvidence(key=key, description="d", url_pattern=pattern)
    assert req.matches(source_url="https://twitter.com/prefpolice/status/753719036245073920", acquired_by=None)
    _, pattern = url_key_and_pattern(["https://x.com/prefpolice/status/753719036245073920", "https://x.com/prefpolice/status/753708496382812160"])
    assert pattern == "*prefpolice/status/7537*"
    key, pattern = url_key_and_pattern(["https://www.youtube.com/watch?v=cv_F3UkbdbI&t=3"])
    assert (key, pattern) == ("youtube_upload", "youtube.com/watch[?]v=cv_F3UkbdbI*")
    assert RequiredEvidence(key=key, description="d", url_pattern=pattern).matches(
        source_url="https://www.youtube.com/watch?v=cv_F3UkbdbI", acquired_by=None)
    _, pattern = url_key_and_pattern(["https://www.snopes.com/fact-check/eiffel-tower-terrorist-attack/?utm=1"])
    assert pattern == "snopes.com/fact-check/eiffel-tower-terrorist-attack*"
    assert site_key("https://news.bbc.co.uk/a/b") == "bbc"


# ------------------------------------------------------------------ 真实 run

@pytest.fixture(scope="module")
def drafted(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Case]:
    out = tmp_path_factory.mktemp("cases_draft")
    path = draft_case(RUN, verdicts_path=VERDICTS, units_path=UNITS, out_dir=out, cases_dir=CASES, today="2026-09-24")
    return path, Case.load(path)


@needs_run
def test_draft_loads_and_has_sidecar(drafted: tuple[Path, Case]):
    path, case = drafted
    assert path.name == "true_4124903.json"
    assert case.case_id == "true:4124903" and case.split == "dev_train" and case.task_kind == "claim_check"
    assert case.claims and case.claims[0] in case.task_text
    assert [m.ref for m in case.media] == ["I0"]
    review = path.with_name("true_4124903.review.md")
    assert review.read_text(encoding="utf-8").startswith("# 审核单：true_4124903.json")
    sidecar = read_review(review)
    assert sidecar["needs_review"] is True and sidecar["generated_from"] == "true_4124903_r1"
    assert sidecar["generated_at"] == "2026-09-24" and sidecar["notes"]
    assert any("label_expected" in n for n in sidecar["notes"])       # 仓库里没有数据集标注：不写并说明
    assert not any(c.kind == "label_expected" for c in case.checks)


@needs_run
def test_draft_requirements_count(drafted: tuple[Path, Case]):
    _, case = drafted
    assert 2 <= len(case.requirements) <= 4
    assert [r.id for r in case.requirements] == [f"REQ{i + 1}" for i in range(len(case.requirements))]
    assert case.requirements[-1].must_cite is False                    # 来源/日期类放最后、非必引


@needs_run
def test_draft_required_evidence_matches_run_evidence(drafted: tuple[Path, Case]):
    _, case = drafted
    sidecar = read_review(_sidecar(drafted))["trace"]
    log = load_run(RUN)
    try:
        for item in case.required_evidence:
            assert item.url_pattern or item.acquired_by
            ids = sidecar[item.key]["evidence_ids"]
            assert ids
            for eid in ids:
                text = log.lookup.get(eid)
                assert item.matches(source_url=text.source_url, acquired_by=log.store.evidence(eid).acquired_by, label=text.label), (item.key, eid)
    finally:
        log.close()
    assert case.evidence_groups()                                      # 规则能解析


@needs_run
def test_draft_overlaps_hand_written_case(drafted: tuple[Path, Case]):
    _, case = drafted
    hand = Case.load(CASES / "true_4124903.json")
    hand_keys = {r.key for r in hand.required_evidence}
    draft_keys = {r.key for r in case.required_evidence}
    assert len(hand_keys & draft_keys) * 2 >= len(hand_keys)
    checks = {c.kind: c.params for c in case.checks}
    assert checks.get("date_in_range") == {"min": "2016-07-13", "max": "2016-07-15"}


@needs_run
def test_cli_writes_draft(tmp_path: Path):
    assert main(["--runs", str(RUN), "--out", str(tmp_path), "--verdicts", str(VERDICTS), "--units", str(UNITS),
                 "--cases", str(CASES)]) == 0
    assert Case.load(tmp_path / "true_4124903.json").case_id == "true:4124903"


@needs_run
def test_draft_dir_loadable_and_other_run_not_overwritten(tmp_path: Path):
    first = draft_case(RUN, verdicts_path=VERDICTS, units_path=UNITS, out_dir=tmp_path, cases_dir=CASES)
    assert set(load_cases(tmp_path)) == {"true:4124903"}                # 审核单是 .md，目录可直接 load_cases
    again = draft_case(RUN, verdicts_path=VERDICTS, units_path=UNITS, out_dir=tmp_path, cases_dir=CASES)
    assert again == first and len(list(tmp_path.glob("*.json"))) == 1   # 同一 run 重跑：原地覆盖
    # 伪造"已有另一 run 的草稿"：把审核单里的 generated_from 改掉
    review = tmp_path / "true_4124903.review.md"
    review.write_text(review.read_text(encoding="utf-8").replace('"generated_from": "true_4124903_r1"', '"generated_from": "true_4124903_r2"'),
                      encoding="utf-8")
    before = (tmp_path / "true_4124903.json").read_text(encoding="utf-8")
    third = draft_case(RUN, verdicts_path=VERDICTS, units_path=UNITS, out_dir=tmp_path, cases_dir=CASES)
    assert third.name == "true_4124903.from_true_4124903_r1.json"
    assert (tmp_path / "true_4124903.json").read_text(encoding="utf-8") == before
    assert Case.load(third).case_id == "true:4124903"
    notes = read_review(tmp_path / "true_4124903.from_true_4124903_r1.review.md")["notes"]
    assert "已有另一 run 的草稿" in notes[0] and "true_4124903_r2" in notes[0]


@needs_run
def test_labels_file_adds_label_expected(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"true:4124903": "不实"}, ensure_ascii=False), encoding="utf-8")
    case = Case.load(draft_case(RUN, verdicts_path=VERDICTS, units_path=UNITS, out_dir=tmp_path / "out", labels_path=labels, cases_dir=CASES))
    assert {"kind": "label_expected", "label": "不实"} in [{"kind": c.kind, **c.params} for c in case.checks]
    assert case.reference_path == str(labels)


# ------------------------------------------------------------------ 引用括号 / 媒体对照 / 日期（纯函数）

def test_answer_segments_square_brackets_and_adjacent_merge():
    segs = answer_segments("画面基本支持该描述：女子悬空。[E3][E6][E10] 但连接处被遮挡。[E11] 报道称属实。[E29、E21]")
    assert [s.evidence_ids for s in segs] == [["E3", "E6", "E10"], ["E11"], ["E29", "E21"]]
    assert answer_segments("甲。【E1】")[0].evidence_ids == ["E1"]


def test_media_task_detection_and_kind():
    assert is_media_task("videosham", "比较给定视频 A 与候选视频 B", [])
    assert not is_media_task("true", "核查以下声明：X", ["X"])
    assert is_media_task("other", "比较给定视频 A 与候选视频 B 的差异", [])
    assert media_task_kind("确定黑衣人物在 B 可见而在 A 对应位置不可见的全部时间区间。核查每段区间的起止边界") == "temporal_locate"
    assert media_task_kind("比较给定视频 A 与候选视频 B，调查两者可由材料确认的差异") == "video_diff"


def test_media_classifier_and_subsplit():
    assert classify_media("调查两者可由材料确认的差异及其对画面含义的影响") == "meaning"
    assert classify_media("说明画面中的区域") == "region"
    assert classify_media("比较两者的可用音频") == "audio"
    assert classify_media("不要把未检查的区域或模态视为没有差异") == "uncertain"
    assert _subsplit_media("说明差异所在位置和未能确定的内容") == ["说明差异所在位置", "说明未能确定的内容"]
    assert _subsplit_media("说明定位精度及仍不确定的范围") == ["说明定位精度及仍不确定的范围"]   # 同类不拆
    # 答案句几乎都带"不能"，归类时不落到 uncertain
    assert classify_media_answer("左下角标题区不同，标题本身不能证明谈话内容") == "region"


def test_date_in_text_formats():
    day = dt.date(2016, 7, 14)
    for text in ("on July 14, 2016 police said", "14 July 2016", "le 14 juillet 2016", "upload 20160714", "2016-07-14",
                 "2016年7月14日", "Jul. 14, 2016"):
        assert date_in_text(day, text), text
    assert not date_in_text(day, "July 14, 2017")


def test_date_check_is_conservative():
    sources = {"E2": "captured on July 3, 1863 at Gettysburg", "E9": "The 2003 request"}
    notes: list[str] = []
    # 日期只在背景段（不在首段结论）：不写
    answer = "该声明基本属实：官员多次要求返还。[E6]\n\n该旗1863年7月3日在葛底斯堡被缴获。[E2]"
    assert date_check(answer, sources, notes) is None and "不在首段结论" in notes[-1]
    # 在首段但来源正文里没有：不写
    notes.clear()
    assert date_check("结论：2016年7月14日的火情不是恐袭。（E1）", {"E1": "no date here"}, notes) is None
    assert "找不到" in notes[-1]
    # 首段 + 来源正文都有：写 ±1 天
    check = date_check("结论：2016年7月14日的火情不是恐袭。（E1）", {"E1": "Paris, 14 July 2016 — fire"}, [])
    assert check is not None and check.params == {"min": "2016-07-13", "max": "2016-07-15"} and "E1" in check.note
    assert date_check("没有日期。", sources, []) is None


# ------------------------------------------------------------------ 其它 run：降级 / videosham / 日期

def _bridge_run(p: Path) -> bool:
    """验收只针对 GIA 由模型驱动的完整调查 run（bridge run）。试点期的单工作区 run（averimatec_423_q0_* 等，
    gia_output_source=scripted_dispatcher、没有 GIA 首轮任务）起草不出要求文字，按设计走 _build_cases_r2 从标注生成。"""

    if not (p / "store.sqlite3").is_file():
        return False
    try:
        config = json.loads((p / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return config.get("gia_output_source") not in (None, "scripted_dispatcher") and config.get("origin", "A") == "A"


def _verdict_runs() -> set[str]:
    """验收对象限于判定文件里有报告判定的 run（09-24 那批 bridge run）；本地 8B 教师的成本实测 run、之后的对照 run 不在此列。"""

    if not VERDICTS.is_file():
        return set()
    return {json.loads(line)["subject"]["run"] for line in VERDICTS.read_text(encoding="utf-8").splitlines() if line.strip()}


ALL_RUNS = sorted(p.name for p in RUNS.glob("*") if _bridge_run(p) and p.name in _verdict_runs()) if RUNS.is_dir() else []


def _draft_run(name: str, out: Path) -> tuple[Case, dict]:
    path = draft_case(RUNS / name, verdicts_path=VERDICTS, units_path=UNITS, out_dir=out, cases_dir=CASES, today="2026-09-24")
    return Case.load(path), read_review(path.with_name(path.name[:-5] + ".review.md"))


def _needs(name: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(not (RUNS / name / "store.sqlite3").is_file() or not VERDICTS.is_file(),
                              reason=f"{name} / 判定文件未随仓库入库")


@_needs("true_20205316753904_validation")
def test_degrades_to_answer_evidence_without_clean_findings(tmp_path: Path):
    # W1-1 唯一的 F1 在 recalibrate 后仍污染 → 没有干净 Finding → 取答案引用的 E 并注明来源降级
    case, draft = _draft_run("true_20205316753904_validation", tmp_path)
    assert case.required_evidence
    assert {item["origin"] for item in draft["trace"].values()} == {"answer_only"}
    assert draft["notes"][0].startswith("来源降级")
    assert {r.acquired_by for r in case.required_evidence} == {"INSPECT/get_frames", "INSPECT/crop"}
    assert any("缺文本来源项" in n for n in draft["notes"])          # run 没做过 SEARCH/open：不编文本来源，只提示
    assert set(draft["requirement_origin"].values()) == {"claim_template"}
    assert all(case.claims[0] in r.text for r in case.requirements)


@needs_run
def test_recalibrated_verdicts_widen_clean_set(drafted: tuple[Path, Case]):
    # 原判 F5 污染（E24 只被它引用）；recalibrate 撤销后 E24 入选，prefpolice 的模式放宽到两条推文的公共前缀
    _, case = drafted
    item = next(r for r in case.required_evidence if r.key == "prefpolice_tweets")
    assert item.url_pattern == "*prefpolice/status/7537*"


@_needs("videosham_1028_r2")
def test_videosham_uses_media_template(tmp_path: Path):
    case, draft = _draft_run("videosham_1028_r2", tmp_path)
    assert case.task_kind == "video_diff" and case.source_policy.mode == "provided_media" and not case.claims
    assert draft["task_family"] == "media"
    texts = [r.text for r in case.requirements]
    assert not any("外部" in t or "可靠来源" in t for t in texts)
    assert any("位置" in t for t in texts) and any("区域" in t for t in texts) and any("含义" in t for t in texts)
    assert case.requirements[-1].must_cite is False and "未能确定" in case.requirements[-1].text
    assert case.required_evidence and all(r.acquired_by and not r.url_pattern for r in case.required_evidence)
    assert not case.checks                                              # 无标注、不写日期
    for t in texts:                                                     # 要求文字来自任务，不含答案里的观察
        assert "lorem" not in t and "CRPS" not in t and not any(ch.isdigit() for ch in t)


@_needs("videosham_4126_split_intervals_r7")
def test_videosham_interval_task(tmp_path: Path):
    case, draft = _draft_run("videosham_4126_split_intervals_r7", tmp_path)
    assert case.task_kind == "temporal_locate"
    assert any("temporal_locate" in n for n in draft["notes"])
    assert "起止边界" in case.requirements[0].text
    assert any(r.acquired_by == "INSPECT/get_frames" for r in case.required_evidence)


@_needs("true_24171541709824_r1")
def test_historical_date_not_checked(tmp_path: Path):
    case, draft = _draft_run("true_24171541709824_r1", tmp_path)
    assert not any(c.kind == "date_in_range" for c in case.checks)
    assert any("1863-07-03" in n and "date_in_range 未写" in n for n in draft["notes"])


@pytest.mark.skipif(not ALL_RUNS or not VERDICTS.is_file(), reason="run 未随仓库入库")
def test_all_runs_have_required_evidence(tmp_path: Path):
    """验收：每个 run 的 required_evidence 非空；claim 类有文本来源（run 做过 SEARCH 时）+ 帧（run 抽过帧时）；media 类有派生证据。"""

    for name in ALL_RUNS:
        case, draft = _draft_run(name, tmp_path / name)
        assert case.required_evidence, name
        assert "neutral" not in draft["requirement_origin"].values(), name
        log = load_run(RUNS / name)
        try:
            kinds = {ev.acquired_by or "" for ev in log.store.evidence_records()}
        finally:
            log.close()
        if draft["task_family"] == "media":
            assert any(r.acquired_by for r in case.required_evidence), name
        else:
            if any(k.startswith("SEARCH/open") for k in kinds):
                assert any(r.url_pattern for r in case.required_evidence), name
            if "INSPECT/get_frames" in kinds:
                assert any(r.acquired_by == "INSPECT/get_frames" for r in case.required_evidence), name
        assert case.evidence_groups() is not None


def _sidecar(drafted: tuple[Path, Case]) -> Path:
    return drafted[0].with_name("true_4124903.review.md")
