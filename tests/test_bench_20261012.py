"""GroundLie360 / FMNV adapters (10-12): annotation loading, sampling, task text, answer parsing, metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _bench_common as B  # noqa: E402

G360_HEADER = ("video_id,video_date,title,debunking_videoid,original_videoid,evidence_videoid,first_level_annotation,"
               "second_level_annotation,title_third_level_annotation,platform,false_speech_temporal,false_speech_content,"
               "video_transcript,transcript_concrete,false_speech_location,categories,tags,video_scene_segmentation,"
               "temporal_edit_location\n")


def _g360(tmp_path: Path) -> Path:
    root = tmp_path / "GroundLie360"
    root.mkdir()
    rows = [
        '183204,20171018.0,"This is sick. ""48 Hour Challenge"" Game",[],[],[],Fake,"[\'false_title\', \'false_speech\']",'
        '"[{\'start\': 13, \'end\': 33, \'value\': \' ""48 Hour Challenge""\', \'entity\': \'false description\'}]",twitter,'
        '"[[0, 10.17536]]",[],t,[],[],[],[],"[[  1 184]\n [185 222]]","[0, 1]"',
        '1958010,20160101.0,Kasich speaks,[],[1958017],[],Real,[],[],youtube,[],[],t,[],[],[],[],"[[1 50]]",[0]',
    ]
    (root / "data.csv").write_text(G360_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_g360_loading_and_task_text(tmp_path):
    samples = {s.sample_id: s for s in B.load_g360(_g360(tmp_path))}
    fake = samples["183204"]
    assert fake.label == "Fake" and fake.types == ("false_title", "false_speech") and fake.date == "2017-10-18"
    assert fake.gt["scene_frames"] == [[1, 184], [185, 222]] and fake.gt["edited_scenes"] == [0, 1]
    assert fake.gt["false_speech_s"] == [[0, 10.17536]] and "48 Hour Challenge" in fake.gt["title_spans"][0]
    text = B.task_text(fake)
    assert "posted on Twitter on 2017-10-18" in text and "VERDICT: Real or Fake" in text and "Fake" not in text.split("\n")[0]
    assert samples["1958010"].label == "Real" and samples["1958010"].types == ()


def test_fmnv_loading(tmp_path):
    root = tmp_path / "FMNV"
    root.mkdir()
    (root / "data.json").write_text(json.dumps([
        {"video_id": "abc", "class": "fv", "contextual_class": "", "label": "false", "subject": "x", "title": "T1"},
        {"video_id": "def", "class": "", "contextual_class": "", "label": "true", "subject": "y", "title": "T2"}]))
    s = {x.sample_id: x for x in B.load_fmnv(root)}
    assert s["abc"].label == "Fake" and s["abc"].types == ("CE",) and s["def"].label == "Real"
    assert "TYPE: CD, CE, SV, CA, or none" in B.task_text(s["abc"])


def test_pick_is_balanced_nested_and_needs_videos():
    samples = [B.Sample("g360", f"f{i}", "Fake", (t,), "") for i, t in enumerate(B.G360_TYPES * 3)]
    samples += [B.Sample("g360", f"r{i}", "Real", (), "") for i in range(20)]
    avail = {s.sample_id for s in samples} - {"r0"}
    small, big = B.pick("g360", samples, avail, 12), B.pick("g360", samples, avail, 20)
    assert len(small) == 12 and sum(s.label == "Fake" for s in small) == 6
    assert len({s.types[0] for s in small if s.label == "Fake"}) == 6          # one per type
    assert {s.sample_id for s in small if s.label == "Fake"} <= {s.sample_id for s in big}
    assert "r0" not in {s.sample_id for s in big}


def test_parse_answer_lines():
    answer = ("1. The clip is real footage.\n\n**VERDICT: Fake**\nTYPES: false_title, CGI\n"
              "FALSE_TITLE_WORDS: \"48 Hour Challenge\"\nFALSE_SPEECH_TIME: 0-10.2; 30 - 35.5\nEDITED_TIME: none")
    p = B.parse_answer("g360", answer)
    assert p["parsed"] and p["verdict"] == "Fake" and p["types"] == ["false_title", "CGI"]
    assert p["title_words"] == "48 Hour Challenge" and p["false_speech_s"] == [[0, 10.2], [30, 35.5]] and p["edited_s"] == []
    q = B.parse_answer("fmnv", "...\nVERDICT: Real\nTYPE: none")
    assert q["verdict"] == "Real" and q["type"] is None
    assert B.parse_answer("fmnv", "VERDICT: Fake\nTYPE: SV (synthetic voiceover)")["type"] == "SV"
    assert not B.parse_answer("fmnv", "I think it is fake.")["parsed"]


def test_metrics():
    rows = [{"label": "Fake", "pred_verdict": "Fake"}, {"label": "Fake", "pred_verdict": None},
            {"label": "Real", "pred_verdict": "Real"}, {"label": "Real", "pred_verdict": "Fake"}]
    m = B.binary_metrics(rows)
    assert m["acc"] == 0.5 and m["parsed"] == 0.75 and abs(m["fake_p"] - 0.5) < 1e-9 and abs(m["fake_r"] - 0.5) < 1e-9
    assert B.token_f1("48 Hour Challenge", [' "48 Hour Challenge"']) == 1.0
    assert B.span_f1([[0, 10]], [[0, 10]]) == 1.0 and B.span_f1([], [[0, 1]]) == 0.0 and B.span_f1([[0, 1]], []) is None
    t = B.type_metrics([{"g": ["CGI"], "p": ["CGI"]}, {"g": [], "p": ["CGI"]}], B.G360_TYPES, "g", "p")
    assert abs(t["per_type"]["CGI"]["f1"] - 2 / 3) < 1e-9 and t["macro_f1"] == t["per_type"]["CGI"]["f1"]
