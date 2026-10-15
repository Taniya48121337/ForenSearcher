"""造数链路改英文（10-07）：案卷英文任务文字、run_cases 按规格语言取任务、核验器程序规则中英双语、r0f 用英文规格。"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from foren_searcher.datagen.cases import Case, CaseError, load_cases
from foren_searcher.datagen.verify import lang
from foren_searcher.datagen.verify.anchored import back_reference_support, is_limitation_part, is_task_given, is_verdict_sentence
from foren_searcher.datagen.verify.answer_anchored import split_answer
from foren_searcher.datagen.verify.exec_checks import dates_in, fact_check_errors
from foren_searcher.datagen.verify.inference import new_tokens, quoted_mention
from foren_searcher.datagen.verify.scope import targets_by_text
from foren_searcher.datagen.verify.units import _HEDGE, _SIBLING_DEP, _overlap_ratio, label_hit

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments" / "sft_data" / "cases"
CJK = re.compile(r"[一-鿿]")


# ---------------------------------------------------------------- 案卷

def test_every_case_has_english_task_and_requirements() -> None:
    cases = load_cases(CASES)
    assert len(cases) >= 33
    for case in cases.values():
        en = case.task_for("en")
        assert en and not CJK.search(en), case.case_id
        assert case.task_for("zh") == case.task_text
        for req in case.requirements:
            assert req.text_en and not CJK.search(req.text_en), (case.case_id, req.id)
        for check in case.checks:
            if check.kind == "label_expected":
                forms = [check.params["label"], *check.params.get("aliases", [])]
                assert any(not CJK.search(f) for f in forms), case.case_id


def test_claims_and_dates_survive_translation() -> None:
    cases = load_cases(CASES)
    for case in cases.values():
        if case.claims:
            assert case.claims[0].strip().rstrip(".") in case.task_for("en"), case.case_id
        date = re.search(r"声明日期：(\d{4}-\d{2}-\d{2})", case.task_text)
        if date:
            assert f"(claim date: {date.group(1)})" in case.task_for("en"), case.case_id


def test_task_for_english_without_translation_is_an_error() -> None:
    case = Case(case_id="x:1", dataset="x", source_id="1", split="dev_train", task_kind="claim_check", task_text="核查", family_key="x:1")
    assert case.task_for("zh") == "核查"
    with pytest.raises(CaseError):
        case.task_for("en")


def test_run_one_gives_the_model_the_english_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import foren_searcher.gia as gia_package
    from foren_searcher.datagen import run_cases

    captured: dict = {}

    @dataclass
    class FakeResult:
        decision: dict

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            await captured["inspect_service"].aclose()

        async def run_new(self, **kwargs):
            captured["run_new"] = kwargs
            return FakeResult({"verdict": "x"})

    class FakeTeacher:
        generation_observer = None

        def __init__(self) -> None:
            from foren_searcher.inspect.policy import InspectPolicy

            self.image_policy = InspectPolicy()

    monkeypatch.setattr(gia_package, "InvestigationRuntime", FakeRuntime)
    monkeypatch.setattr(run_cases, "make_teacher", lambda *a, **k: FakeTeacher())
    cases = tmp_path / "cases"
    (cases / "media").mkdir(parents=True)
    (cases / "media" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    base = {"dataset": "videosham", "split": "dev_train", "task_kind": "video_diff", "task_text": "比较给定视频",
            "media": [{"ref": "I0", "path": "media/a.png", "media_type": "image/png"}], "source_policy": {"mode": "provided_media"}}
    good = cases / "videosham_1.json"
    good.write_text(json.dumps({**base, "case_id": "videosham:1", "source_id": "1", "family_key": "videosham:1",
                                "task_text_en": "Compare the given video A"}, ensure_ascii=False), encoding="utf-8")
    bad = cases / "videosham_2.json"
    bad.write_text(json.dumps({**base, "case_id": "videosham:2", "source_id": "2", "family_key": "videosham:2"},
                              ensure_ascii=False), encoding="utf-8")
    specs = tmp_path / "specs"
    for name in ("hi@0.7.0", "gia@0.8.0"):
        shutil.copytree(ROOT / "specs" / name, specs / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOREN_CONFIG", raising=False)
    common = dict(runs_dir=tmp_path / "runs", round_tag="r0f", sample=1, teacher="endpoint", endpoint="http://t/v1", model="m",
                  thinking=True, specs_root=specs, hi_spec="hi@0.7.0", gia_spec="gia@0.8.0", search_mode="record",
                  snapshot_root=tmp_path / "snap", tool_cache="off")
    asyncio.run(run_cases.run_one(good, "videosham_1_r0f_1", **common))
    call = captured["run_new"]
    assert call["task"] == "Compare the given video A" and call["claim_materials"] == {"C0": {"task": "Compare the given video A"}}
    assert [m.label for m in call["media"]] == ["Material A"]
    config = json.loads((tmp_path / "runs" / "videosham_1_r0f_1" / "run_config.json").read_text(encoding="utf-8"))
    assert config["task"] == "Compare the given video A" and config["task_language"] == "en"
    with pytest.raises(CaseError):
        asyncio.run(run_cases.run_one(bad, "videosham_2_r0f_1", **common))
    assert not (tmp_path / "runs" / "videosham_2_r0f_1").exists()          # 在建 run 目录之前就报错


# ---------------------------------------------------------------- 核验器程序规则

def test_english_answer_is_split_into_sentences() -> None:
    answer = ("False. The fire near the Eiffel Tower on July 14, 2016 was caused by fireworks [E3]. "
              "Paris police said there was no attack (E4, E5). The U.S. embassy issued no alert [E6].")
    sentences = split_answer(answer)
    assert len(sentences) == 4                       # 以前整段算一句（只按中文句末切）
    assert [s.evidence_ids for s in sentences] == [["E3"], ["E3"], ["E4", "E5"], ["E6"]]     # 无引用的 "False." 回溯到下一组
    assert sentences[2].text.startswith("Paris police said")
    chinese = split_answer("该声明不实。铁塔附近起火是烟花引起的[E3]。警方称没有袭击（E4）。")
    assert [s.evidence_ids for s in chinese] == [["E3"], ["E3"], ["E4"]]


def test_limitation_parts_in_english() -> None:
    assert is_limitation_part("The original upload could not be obtained")
    assert is_limitation_part("whether the clip was edited remains unverified")
    assert not is_limitation_part("could not confirm what Reuters reported on 14 July")       # 夹带专名 / 数字
    assert not is_limitation_part("the video shows a fire near the tower")
    assert is_limitation_part("未能取得原始视频") and not is_limitation_part("未能取得 Reuters 的报道")


def test_task_given_backref_and_verdict_in_english() -> None:
    task = "Fact-check the following claim ... (claim date: 2020-07-02)"
    assert is_task_given("The claim was published on July 2, 2020", task)
    assert not is_task_given("The story first appeared on July 2, 2020 on Foolish Humor", task)
    others = {"F1": "Foolish Humor published the story on its satire site", "F2": "The site labels its content as satire"}
    assert back_reference_support("As established above, the satire site published the story", others)
    assert not back_reference_support("As established above, Budweiser fired the employee in 2008", others)
    assert is_verdict_sentence("The claim is false") and is_verdict_sentence("Conclusion: misleading")
    assert not is_verdict_sentence("The video was uploaded in 2019")


def test_dates_intervals_and_entities_in_english() -> None:
    from foren_searcher.datagen.cases import Check
    from foren_searcher.datagen.verify.exec_checks import exec_checks

    assert dates_in("on 14 July 2016 and July 15, 2016") == ["2016-07-14", "2016-07-15"]
    assert lang.intervals_in("The segment runs from 84.0 s to 118.5 s") == [(84.0, 118.5)]
    case = load_cases(CASES)["true:4124903"]
    check = next(c for c in case.checks if c.kind == "entity_in_evidence")
    errors = fact_check_errors(check, "The fireworks were launched from the Pont d'Iéna [E3]", "fireworks fired from the pont d'Iéna")
    assert errors == []
    bare = Check.from_dict({"kind": "entity_in_evidence", "params": {"text": "耶拿桥", "aliases": ["pont d'Iéna"]}})
    assert fact_check_errors(bare, "launched from the Pont d'Iéna [E3]", "pont d'Iéna")      # 没有 statement_aliases：照旧 omitted
    assert exec_checks is not None


def test_labels_hedges_and_sibling_dependencies_in_english() -> None:
    assert label_hit("False", "Verdict: False. The claim is fabricated")
    assert not label_hit("True", "The claim is not true") and not label_hit("True", "This is untrue")
    assert not label_hit("False", "The claim is mostly false") and label_hit("Mostly False", "The claim is mostly false")
    assert label_hit("Partly true", "Rating: partly true") and not label_hit("属实", "部分属实")
    assert _HEDGE.search("Whether he said it remains unverified") and _HEDGE.search("尚未核实")
    assert _SIBLING_DEP.search("Using the results from the other task, compare the frames")
    assert not _SIBLING_DEP.search("Find the earliest upload of the clip and compare it with Material A")


def test_inference_tokens_and_quoted_mentions_in_english() -> None:
    context = "The pillar in the image stands in Bharatpur. The Delhi iron pillar dates from the Gupta period."
    assert new_tokens("the claim conflates two different pillars", context) == []
    assert new_tokens("the claim conflates the pillar with one in Agra", context) == ["Agra"]
    assert quoted_mention("the ram was sacrificed", 'The report rejected the explanation that "the ram was sacrificed".')


def test_overlap_and_scope_fallback_in_english() -> None:
    assert _overlap_ratio("Budweiser fired the employee", "The satire site published the story") < 0.3
    assert _overlap_ratio("the satire site published the story", "Foolish Humor, a satire site, published the story") >= 0.6
    case = load_cases(CASES)["averitec:392"]
    assert targets_by_text(case, "Find which website or post first published the Budweiser story") == {"REQ1"}
    assert "REQ1" in targets_by_text(case, "这则故事最早发布在哪个网站或帖子")


def test_launcher_r0f_uses_english_specs() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_launch_en", ROOT / "scripts" / "_launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    assert launch.R0F_SPECS == ["--hi-spec", "hi@0.7.0", "--gia-spec", "gia@0.8.0"]
    hi_prompt = (ROOT / "specs" / "hi@0.7.0" / "student_prompt.md").read_text(encoding="utf-8")
    assert launch.R0F_MARKS["hi_role"] in hi_prompt and launch.R0F_MARKS["hi_prompt"] in hi_prompt
