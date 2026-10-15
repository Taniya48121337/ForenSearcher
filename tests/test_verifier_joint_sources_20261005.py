"""多来源合并复核（候选，默认关闭；10-05）。场景取自 r0e videosham:1028 A10：两侧转写各在一条 E 里。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.errors import ErrorType, Severity, error
from foren_searcher.datagen.verify import anchored as A
from foren_searcher.datagen.verify import joint_sources as J
from foren_searcher.datagen.verify.context import DictLookup, EvidenceText


class FakeJudge:
    def __init__(self, reply: dict[str, Any] | None) -> None:
        self.reply, self.calls = reply, []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list, schema: dict | None = None) -> Any:
        self.calls.append({"system": system_prompt, "payload": payload, "images": list(images)})
        if self.reply is None:
            raise ValueError("bad output")
        return self.reply


@pytest.fixture(autouse=True)
def _reset():
    saved = dict(J.JOINT_SOURCES)
    J.JOINT_SOURCES["enabled"] = True
    yield
    J.JOINT_SOURCES.update(saved)


ASR_A = "材料A ASR：0.00–1.60 s “大家好欢迎收看”；1.60–3.20 s “今天的天气预报”；3.20–4.84 s “明天多云转晴”。"
ASR_B = "材料B ASR：0.00–1.60 s “大家好欢迎收看”；1.60–3.20 s “今天的天气预报”；3.20–4.84 s “明天多云转晴”。"
ALIGN = "align_media：音频对齐 offset 0.000 s，score 0.998，覆盖 0–4.84 s。"
LOOKUP = DictLookup({"E13": EvidenceText("E13", ASR_A, derived_from=("E1",)), "E14": EvidenceText("E14", ASR_B, derived_from=("E2",)),
                     "E15": EvidenceText("E15", ALIGN, derived_from=("E1", "E2")), "E3": EvidenceText("E3", "单条网页正文，与转写无关的内容")})
STATEMENT = "两侧全段自动转写及分段一致，音频对齐支持同时间对应（E13、E14、E15）。"
SPAN = "两侧全段自动转写及分段一致"


def _err(span: str = SPAN, eids: tuple[str, ...] = ("E13", "E14", "E15"), **kw: Any):
    return error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=span, method="model",
                 severity=kw.pop("severity", Severity.CRITICAL), evidence_ids=list(eids), extra={"finding": "A10", **kw})


def _reply(supported: bool, quotes: list[tuple[str, str]], span: str = SPAN) -> dict[str, Any]:
    return {"spans": [{"span": span, "supported": supported, "note": "对照两侧转写",
                       "quotes": [{"evidence_id": e, "quote": q} for e, q in quotes]}]}


def run(errors, judge, statements=None):
    return asyncio.run(J.apply_joint_check(errors, statements or {"A10": STATEMENT}, LOOKUP, judge))


def test_disabled_is_noop() -> None:
    J.JOINT_SOURCES["enabled"] = False
    judge = FakeJudge(_reply(True, [("E13", "大家好欢迎收看"), ("E14", "大家好欢迎收看")]))
    errs = [_err()]
    assert run(errs, judge) == errs and judge.calls == []


def test_two_sources_compared_downgrades_with_trace() -> None:
    judge = FakeJudge(_reply(True, [("E13", "1.60–3.20 s “今天的天气预报”"), ("E14", "1.60–3.20 s “今天的天气预报”")]))
    (e,) = run([_err()], judge)
    assert e.severity is Severity.MINOR and e.extra["program_rule"] == "joint_sources"
    assert e.extra["original_severity"] == "critical" and e.extra["joint"]["used"] == ["E13", "E14"]
    assert e.detail.startswith("（多来源合并复核：E13+E14")
    call = judge.calls[0]
    assert call["system"] == J.JOINT_SYSTEM_PROMPT and call["payload"]["spans"] == [SPAN]
    assert [s["evidence_id"] for s in call["payload"]["sources"]] == ["E13", "E14", "E15"]
    assert ASR_A in call["payload"]["sources"][0]["content"]          # 给的是来源原文，不只是支持片段


def test_guards_keep_original_verdict() -> None:
    # 只用到一条来源
    (e,) = run([_err()], FakeJudge(_reply(True, [("E13", "大家好欢迎收看"), ("E13", "明天多云转晴")])))
    assert e.severity is Severity.CRITICAL and e.extra["joint"]["status"] == "rejected_quotes"
    # 有一条文本引语是编的
    (e,) = run([_err()], FakeJudge(_reply(True, [("E13", "大家好欢迎收看"), ("E14", "两侧完全一致无差别")])))
    assert e.severity is Severity.CRITICAL and "查不到" in e.extra["joint"]["why"]
    # 模型说不能合并得出
    (e,) = run([_err()], FakeJudge(_reply(False, [])))
    assert e.severity is Severity.CRITICAL and e.extra["joint"]["status"] == "not_joint"
    # 调用失败：原样保留
    errs = [_err()]
    assert run(errs, FakeJudge(None)) == errs


def test_candidate_selection() -> None:
    judge = FakeJudge(_reply(True, [("E13", "大家好欢迎收看"), ("E14", "大家好欢迎收看")]))
    single = _err(eids=("E13",))                                   # 只引一条：不复核
    minor = _err(severity=Severity.MINOR)
    ruled = _err(program_rule="back_reference")
    contra = error(ErrorType.WRONG_CITATION, "E13 与该部分相反：…", span=SPAN, method="model", evidence_ids=["E13", "E14"],
                   extra={"finding": "A10"})
    errs = [single, minor, ruled, contra]
    assert run(errs, judge) == errs and judge.calls == []
    # 引了两条但其中一条在本 run 里查不到 → 可用来源不足两条，不问
    assert run([_err(eids=("E13", "E99"))], judge)[0].severity is Severity.CRITICAL and judge.calls == []


def test_one_call_per_statement_and_images_attached(tmp_path: Path) -> None:
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    lookup = DictLookup({**LOOKUP.records, "E20": EvidenceText("E20", "", modalities=("image",), image_path=img)})
    reply = {"spans": [{"span": SPAN, "supported": True, "quotes": [{"evidence_id": "E13", "quote": "大家好欢迎收看"},
                                                                     {"evidence_id": "E20", "quote": "画面左上角字幕"}]},
                       {"span": "音频对齐支持同时间对应", "supported": False, "quotes": []}]}
    judge = FakeJudge(reply)
    errs = [_err(eids=("E13", "E20")), _err(span="音频对齐支持同时间对应", eids=("E13", "E20"))]
    out = asyncio.run(J.apply_joint_check(errs, {"A10": STATEMENT}, lookup, judge))
    assert len(judge.calls) == 1 and judge.calls[0]["images"] == [img]
    assert out[0].severity is Severity.MINOR and out[1].severity is Severity.CRITICAL


def test_wired_into_verify_anchored_and_version_label() -> None:
    from foren_searcher.datagen.verify import inference as I
    from foren_searcher.datagen.verify import semantic_coverage as S
    saved = (dict(A.PROGRAM_RULES), dict(I.INFERENCE_CHECK), dict(S.SEMANTIC_COVERAGE))
    try:
        A.PROGRAM_RULES.update({k: True for k in A.PROGRAM_RULES}); I.INFERENCE_CHECK["enabled"] = True
        S.SEMANTIC_COVERAGE["enabled"] = True
        assert A.feature_version() == "verifier@0.3.6+joint"
        J.JOINT_SOURCES["enabled"] = False
        assert A.feature_version() == "verifier@0.3.6"
    finally:
        A.PROGRAM_RULES.update(saved[0]); I.INFERENCE_CHECK.update(saved[1]); S.SEMANTIC_COVERAGE.update(saved[2])


def test_default_off(monkeypatch) -> None:
    import importlib
    monkeypatch.delenv("FOREN_VERIFIER_JOINT", raising=False)
    mod = importlib.reload(J)
    try:
        assert mod.JOINT_SOURCES["enabled"] is False
        monkeypatch.setenv("FOREN_VERIFIER_JOINT", "1")
        assert importlib.reload(J).JOINT_SOURCES["enabled"] is True
    finally:
        monkeypatch.delenv("FOREN_VERIFIER_JOINT", raising=False)
        importlib.reload(J)
