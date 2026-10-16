"""10-17 r1 预演之二：walker 与 search 整合——E 族比对网页上的图、L 族比对结论、C 族看帧与整段转写、第 2 轮沿用族、
缩略图第二家引擎、hidden_context 放行 1/fps 与 INSPECT 采样参数、重复操作合并的英文 notice。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments" / "sft_data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from foren_searcher.datagen import walker as w  # noqa: E402


# ---------------------------------------------------------------------------- E 族

def test_thumb_verdict_from_snippet():
    assert w.thumb_verdict("Thumbnail compared with the query image: near-identical. Found by Yandex") == "same"
    assert w.thumb_verdict("Thumbnail compared with the query image: partial match, not near-identical.") == "similar"
    assert w.thumb_verdict("Thumbnail compared with the query image: no match (cropping, an added border …)") == "different"
    assert w.thumb_verdict("Found by Google Lens (match type not reported).") is None


def _cmp_obs(area, applied=True, status="ready"):
    return {"operation": "compare_media", "targets": ["E1", "E8"], "status": status, "result_evidence_id": "E26",
            "created": ["E26", "E25"], "params": {},
            "result": {"summary_lines": [f"The two images differ in pixels: changed area {area * 100:.2f}%"],
                       "frame_diff": {"image_comparison": {"changed_area": area},
                                      "registration": {"requested": True, "applied": applied}}}}


def test_compare_class_thresholds():
    assert w._compare_class(_cmp_obs(0.01))[0] == "same"
    assert w._compare_class(_cmp_obs(0.11))[0] == "edited"        # r1dev e133：同一张、两边裁法不同 + 重压缩
    assert w._compare_class(_cmp_obs(0.50))[0] == "different"
    assert w._compare_class(_cmp_obs(0.20, applied=False))[0] == "different"   # 特征点对不上、差异又大
    assert w._compare_class({"status": "failed", "result": {}})[0] == "failed"


def test_image_compares_maps_other_image_to_record_and_composite():
    state = SimpleNamespace(observations=[_cmp_obs(0.11)], cards={})
    got = w.image_compares(state, "E1")
    assert got["E8"]["class"] == "edited" and got["E8"]["record"] == "E26" and got["E8"]["composite"] == ["E25"]


def test_image_of_prefers_page_copy_then_thumbnail():
    result = {"materials": [{"kind": "image", "status": "failed", "evidence": []},
                            {"kind": "page", "status": "ready", "evidence": [{"evidence_id": "E9", "modalities": ["text"]}]},
                            {"kind": "thumbnail", "status": "ready", "evidence": [{"evidence_id": "E10", "modalities": ["image"]}]}]}
    assert w._image_of(result) == ("E10", "thumbnail")
    # 截断的结果行也认得 image= / thumbnail=
    rows = w._results_from_items(["[R2] https://x.org/a; image=ready(E8), page=ready(E9), thumbnail=ready(E10)"])
    assert w._page_of(rows[0]) == "E9" and w._image_of(rows[0]) == ("E8", "image")


def test_compare_required_and_notes_call_uncompared_pages_candidates():
    pages = [{"evidence_id": "E9", "candidate": "R2", "image": "E8", "image_kind": "image"},
             {"evidence_id": "E13", "candidate": "R4", "image": None}]
    compares = {"E8": {"class": "edited", "area": 0.11, "record": "E26", "composite": ["E25"], "summary": ""}}
    notes = " ".join(w._image_match_notes(SimpleNamespace(), "E1", pages, {"E9", "E13"}, compares))
    assert "E9 (R2): its picture E8 was compared with E1 by compare_media (record E26" in notes
    assert "E13 (R4): its picture was not compared with E1 (no copy of the picture was obtained from it)" in notes
    assert "is a candidate only" in notes and "does not show that the page lacks the picture" in notes
    req = w._compare_required("E1", pages, {"E9", "E13"}, compares)
    assert req and req[0]["ids"] == ["E26"]


def _replay(run: str, upto: str, workspace: str = "W1-1", memory: dict | None = None):
    rdir = ROOT / "experiments" / "sft_data" / "runs" / run / "requests"
    if not rdir.is_dir():
        pytest.skip(f"{run} not present")
    memory = {} if memory is None else memory
    got = None
    for rd in sorted(rdir.glob("request_*"), key=lambda p: int(p.name.split("_")[-1])):
        req = json.loads((rd / "request.json").read_text(encoding="utf-8"))
        if req["payload"].get("workspace_id") != workspace:
            continue
        got = w.decide_hi(req["payload"], memory, req.get("rendered_payload"))
        if rd.name == upto:
            break
    return got, memory


def test_e_dev_run_compares_read_page_then_reviews_then_completes():
    # r1dev e133：读完 R2 的页面 E9（2021-08-26）→ 比对它上面那张图 E8；再比 R4 的图 E12；交报告前补看对照图；报告 COMPLETED
    got, _ = _replay("averimatec_e133_r1dev_1", "request_0008")
    assert got.rule == "E6_compare_page_image" and got.output["arguments"]["targets"] == ["E1", "E8"]
    got, _ = _replay("averimatec_e133_r1dev_1", "request_0011")
    assert got.rule == "E7_view_comparison_before_report"
    got, _ = _replay("averimatec_e133_r1dev_1", "request_0012")
    assert got.kind == "report" and got.slot["status"] == "COMPLETED"
    assert any("compare_media" in n for n in got.slot["notes"]) and got.slot["required_findings"]


# ---------------------------------------------------------------------------- 第 2 轮沿用第 1 轮的族

def test_round2_keeps_round1_family():
    # r0h e133 W2-1："…as claimed. Scope: open and read <网址>"——任务文字判成 B 族、走到 G9；带上第 1 轮的族就按 Scope 去读
    got, _ = _replay("averimatec_e133_r0h_1", "request_0014", workspace="W2-1", memory={"run_family": "E"})
    assert got.rule == "S2f_open_page"
    got, _ = _replay("averimatec_e133_r0h_1", "request_0014", workspace="W2-1", memory={})
    assert got.rule == "G9_unsupported_family"


# ---------------------------------------------------------------------------- C 族

def test_claim_kinds():
    assert w.claim_kinds("Worms in Lipton Lemon Green tea!!! Ewwwww. Share now!") == {"visual"}
    assert "speech" in w.claim_kinds("Kasich says women should get out of their kitchens")
    assert w.claim_kinds("Video shows Biden saying he lost") == {"visual", "speech"}


def test_c_dev_run_samples_frames_per_shot_for_a_single_long_shot():
    # r1dev v1959500：video_structure 只切出 1 个 166 s 的镜头（1 张中间帧）→ 按镜头补取帧，约每 30 s 一张，≤ 6 张
    got, _ = _replay("groundlie360_v1959500_r1dev_1", "request_0005")
    assert got.rule == "C2w_frames_per_shot"
    times = got.output["arguments"]["request"]["params"]["times_s"]
    assert len(times) == 6 and times[0] < 20 and times[-1] > 140


def test_claim_media_notes_require_frames_and_whole_transcript():
    state = SimpleNamespace(observations=[{"operation": "asr", "status": "ready", "targets": ["E1"], "result_evidence_id": "E11"},
                                          {"operation": "video_structure", "status": "ready", "targets": ["E1"],
                                           "result": {"images": [{"evidence_id": "E13"}]}, "created": ["E13", "E14"],
                                           "result_evidence_id": "E14", "params": {}}],
                            attached=["E13"], cards={})
    notes, required = w._claim_media_notes(state, "E1", "Worms in Lipton Lemon Green tea!!!")
    text = " ".join(notes)
    assert "scene frames E13" in text and "something visible in the footage" in text and "not only its opening sentence" in text
    assert [r["ids"] for r in required] == [["E13"], ["E11"]]


def test_required_findings_contract():
    from oracle_generator import required_finding_problems  # type: ignore

    required = [{"ids": ["E13"], "words": "show|visible", "message": "frames"},
                {"ids": ["E11"], "words": "say|says", "message": "transcript"}]
    report = {"findings": [{"statement": "Frame E13 shows tea with small dark pieces.", "evidence_ids": ["E13"]}]}
    assert required_finding_problems(report, required) == ["transcript"]
    report["findings"].append({"statement": "The speaker says the pieces are lemon bits.", "evidence_ids": ["E11"]})
    assert required_finding_problems(report, required) == []


# ---------------------------------------------------------------------------- L 族

def test_l_report_requires_comparison_verdict():
    got, _ = _replay("averimatec_l623_r1dev_1", "request_0012")
    assert got.kind == "report"
    req = got.slot["required_findings"]
    assert req and req[0]["message"].startswith("no finding")      # 比对结论（或无法比对及原因）必须写成 Finding


def test_l_reference_photos_ranked_by_query_words():
    # r1s l623：查询 "U.S. Capitol building interior" 的前 3 张是拉链零件、字母 U 剪贴画；相关的 "State Capitol Buildings Interior" 在后面
    rdir = ROOT / "experiments" / "sft_data" / "runs" / "averimatec_l623_r1s_1" / "requests" / "request_0004"
    if not rdir.is_dir():
        pytest.skip("r1s l623 not present")
    req = json.loads((rdir / "request.json").read_text(encoding="utf-8"))
    refs = w.ranked_refs(w._search_obs(w.HiState(req["payload"], {})))
    assert [r["evidence_id"] for r in refs] == ["E15", "E18"]


# ---------------------------------------------------------------------------- search：第二家引擎的缩略图

def test_second_engine_thumbnail_used_when_first_does_not_match():
    from foren_searcher.search.providers.multi_reverse import MultiReverseImageProvider

    class Verifier:
        top_n = 10
        stats: dict = {}

        async def verify(self, data, urls, total_timeout_s=None):
            return [{"verified_match": "same" if u and "yandex" in u else "different", "thumbnail_url": u} for u in urls]

    provider = MultiReverseImageProvider([("lens", object()), ("yandex", object())], normalize=True, verifier=Verifier())
    lens = SimpleNamespace(thumbnail_url="https://lens/t.jpg", image_url=None)
    yandex = SimpleNamespace(thumbnail_url="https://yandex/t.jpg", image_url=None)
    top = [{"base": lens, "per_engine": {"lens": lens, "yandex": yandex}}]
    checks = [{"verified_match": "different", "thumbnail_url": "https://lens/t.jpg"}]
    img = ROOT / "tests" / "__r1_tmp_query.bin"
    img.write_bytes(b"x")
    try:
        asyncio.run(provider._second_engine_thumbs(top, ["https://lens/t.jpg"], checks, img, started=0.0))
    finally:
        img.unlink()
    assert checks[0]["verified_match"] == "same" and checks[0]["verified_engine_fallback"] == "https://yandex/t.jpg"


# ---------------------------------------------------------------------------- 工具层 / 导出检查器

def test_hidden_context_releases_frame_step_and_inspect_params():
    from foren_searcher.datagen.export import hidden_context

    messages = [{"role": "user", "content": json.dumps({"profile": "640x352, 24.000 fps"})}, {"role": "assistant", "content": ""}]
    payload = {"x": "0.0417 and 12.375 and 99.125"}
    out = {"action": "INSPECT", "arguments": {"targets": ["E1"], "request": {"operation": "get_frames",
                                                                                "params": {"times_s": [12.375]}}},
           "rationale": "step 0.0417 s; also 99.125"}
    assert hidden_context(messages, payload, out) == ["99.125"]


def test_merged_occurrence_notice_is_english(monkeypatch):
    from foren_searcher import prompts_en

    assert prompts_en.MERGED_OCCURRENCES_NOTICE.startswith("This entry merges several executed operations")
    assert all(ord(c) < 128 for c in prompts_en.MERGED_OCCURRENCES_NOTICE)
