"""教师输入 = 学生输入（10-05）：教师按学生同一档位看状态；导出把"用了学生看不到的内容"的行降为 held。

审计依据：r0e 的 35 条 HI 报告监督里 17 条写了学生输入（compact_16k 渲染后）里没有、只在原始全量视图里有的数字或专名。
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen import export as X
from foren_searcher.datagen import run_cases
from foren_searcher.datagen.teacher import render_for_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_20361105_r0e_1"
needs_run = pytest.mark.skipif(not (RUN / "requests").is_dir(), reason="r0e run 未随仓库入库")


def _request(name: str) -> dict[str, Any]:
    return json.loads((RUN / "requests" / name / "request.json").read_text(encoding="utf-8"))


def _with_renderer(payload: dict[str, Any], renderer: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload))
    view = payload.get("model_view") if payload.get("workspace_id") else payload.get("global_view")
    view.setdefault("runtime_constraints", {})["state_renderer"] = renderer
    return payload


def test_render_for_model_leaves_legacy_untouched() -> None:
    payload = {"workspace_id": "W1-1", "model_view": {"runtime_constraints": {"state_renderer": "legacy"}, "task": "t"}}
    shown, info = render_for_model(payload)
    assert shown is payload and info is None


@needs_run
def test_render_for_model_compacts_and_keeps_repair_feedback() -> None:
    payload = _with_renderer(_request("request_0008")["payload"], "compact_16k")
    payload["validation_error"] = "bad citation E99"
    shown, info = render_for_model(payload)
    assert info and info["renderer"] == "compact_16k" and info["rendered_chars"] < info["raw_chars"]
    assert shown["validation_error"] == "bad citation E99"
    # renderer 参数优先于 payload 里写的档位（bridge 用它对齐导出档位）
    legacy = _with_renderer(_request("request_0008")["payload"], "legacy")
    _, forced = render_for_model(legacy, "compact_16k")
    assert forced and forced["renderer"] == "compact_16k"


class _Done(Exception):
    pass


def _codex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, list[str]]:
    sys.path.insert(0, str(ROOT / "experiments" / "sft_data"))
    import cli_generator as C  # type: ignore

    prompts: list[str] = []

    def fake_run(command: list[str], *, input: str, **_: Any) -> subprocess.CompletedProcess:
        prompts.append(input)
        out = Path(command[command.index("-o") + 1])
        out.write_text('{"decision":"FINISH","answer":"x"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(C.shutil, "which", lambda _: "codex")
    monkeypatch.setattr(C.subprocess, "run", fake_run)
    generator = C.CliGenerator(store=None, output_dir=tmp_path / "requests")
    monkeypatch.setattr(generator, "_image_message", lambda view: (None, []))
    return generator, prompts


@needs_run
def test_codex_teacher_sees_the_rendered_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator, prompts = _codex(monkeypatch, tmp_path)
    payload = _with_renderer(_request("request_0008")["payload"], "compact_16k")
    generator._generate("SYS", payload, 1)
    saved = json.loads((tmp_path / "requests" / "request_0001" / "request.json").read_text(encoding="utf-8"))
    rendered, info = render_for_model(payload)
    # 落盘：原始 payload（导出据它重渲染）+ 实际发给教师的 rendered_payload
    assert saved["payload"] == payload and saved["rendered_payload"] == rendered and saved["state_render"] == info
    sent = json.loads(prompts[0][prompts[0].index("{"):])
    assert sent["payload"] == rendered and "rendered_payload" not in sent and "state_render" not in sent


@needs_run
def test_codex_teacher_on_legacy_spec_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator, prompts = _codex(monkeypatch, tmp_path)
    payload = _with_renderer(_request("request_0008")["payload"], "legacy")
    generator._generate("SYS", payload, 1)
    saved = json.loads((tmp_path / "requests" / "request_0001" / "request.json").read_text(encoding="utf-8"))
    assert "rendered_payload" not in saved
    assert json.loads(prompts[0][prompts[0].index("{"):])["payload"] == payload


def test_endpoint_teacher_renders_in_teacher_mode_too() -> None:
    # 以前只有 student 模式 / 显式 --state-renderer 才渲染；教师模式也要看学生那份
    assert "render_inputs = True" in inspect.getsource(run_cases.run_one)


def test_hidden_context_counts_only_tokens_the_raw_view_had() -> None:
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "E3 [image] frame 1.500s ← E1"},
                {"role": "assistant", "content": "{}"}]
    raw = {"model_view": {"evidence_view": ["E3 frame 1.500s", "E9 page: Reuters 2019-03-04 duration 14.25"]}}
    output = {"status": "COMPLETED", "findings": [{"statement": "Reuters reported it on 2019-03-04; E3 at 1.500s; Nowhere 777",
                                                  "evidence_ids": ["E3", "E9"]}]}
    # Nowhere / 777 是输出自己写的（原始视图里也没有），不算"学生看不到"
    assert X.hidden_context(messages, raw, output) == ["Reuters", "2019", "E9"]


def test_hidden_context_regex_examples() -> None:
    messages = [{"role": "user", "content": "nothing here"}, {"role": "assistant", "content": "{}"}]
    raw = {"x": "IPCC 1971 3.7 28.320 E12 Caddy"}
    out = {"statement": "IPCC says 3.7 since 1971; offset 28.320 in E12 per Caddy; 12 frames"}
    assert X.hidden_context(messages, raw, out) == ["IPCC", "3.7", "1971", "28.320", "E12", "Caddy"]
    assert X.hidden_context(messages, raw, out, limit=2) == ["IPCC", "3.7"]


@needs_run
@pytest.mark.parametrize("mode", ["hold", "off"])
def test_export_holds_rows_that_used_hidden_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str) -> None:
    from foren_searcher.datagen.cases import load_cases
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.specs import Spec
    from foren_searcher.datagen.verify.verdict import read_verdicts

    log = load_run(RUN)
    try:
        specs = {"hi": Spec.load(ROOT / "specs" / "hi@0.5.0"), "gia": Spec.load(ROOT / "specs" / "gia@0.6.0")}
        verdicts = [v for v in read_verdicts(Path(__file__).parent / "fixtures" / "bridge_20261005" / "verdicts_reports.jsonl")
                    if v.subject.get("run") == log.name]
        case = load_cases(ROOT / "experiments" / "sft_data" / "cases").get(log.case_id or "")
        baseline, _ = X.export_run(log, case, verdicts, specs, renderer="compact_16k", out_dir=tmp_path / "a",
                                   image_root=tmp_path / "a" / "images", hidden_context_mode=mode)
        assert all(r["metadata"]["hidden_context"] == [] for r in baseline)   # 这个 run 本身没有
        monkeypatch.setattr(X, "hidden_context", lambda messages, payload, output, limit=8: ["Reuters", "2019"])
        # 10-15 B2：本 run 证据里出现过的词会被放行；这里测的是"扣下"本身，关掉放行（放行另见 test_export_gate_20261015）
        monkeypatch.setattr(X, "release_by_evidence", lambda tokens, decision, messages, corpus: {})
        rows, _ = X.export_run(log, case, verdicts, specs, renderer="compact_16k", out_dir=tmp_path / "b",
                               image_root=tmp_path / "b" / "images", hidden_context_mode=mode)
    finally:
        log.close()
    before = {r["metadata"]["request"]: r["metadata"]["bucket"] for r in baseline}
    assert any(b == "accepted" for b in before.values())
    for row in rows:
        m = row["metadata"]
        assert m["hidden_context"] == ["Reuters", "2019"]
        if mode == "hold" and before[m["request"]] == "accepted":
            assert m["bucket"] == "held" and m["bucket_reason"] == "hidden_context:Reuters,2019" and m["sample_weight"] == 0.0
        else:
            assert m["bucket"] == before[m["request"]]
    manifest = X.write_outputs(rows, [], out_dir=tmp_path / "b", specs=specs, renderer="compact_16k",
                               verdicts_source="v", cases_dir=str(ROOT / "experiments" / "sft_data" / "cases"), cases={})
    assert manifest["hidden_context"]["rows"] == len(rows)
    assert manifest["hidden_context"]["held"] == (sum(b == "accepted" for b in before.values()) if mode == "hold" else 0)


def _launch():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_launch_r0f", ROOT / "scripts" / "_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(root: Path, *, specs: tuple[str, str], features: dict, rendered: bool, english: bool = True) -> Path:
    run = root / "case_r0f_1"
    (run / "requests" / "request_0001").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({
        "search": "ddgs_text", "video_search": {"mode": "ytdlp_then_ddgs"}, "query_guard": {"enabled": True},
        "teacher_specs": {"hi": f"hi@{specs[0]}#abc", "gia": f"gia@{specs[1]}#def"},
        "task": "Fact-check the following claim" if english else "核查以下声明", "task_language": "en" if english else "zh",
        "runtime_guards": {"input_features": features}}, ensure_ascii=False), encoding="utf-8")
    system = ("You are HI, the multimodal investigation agent inside an independent Workspace. "
              "Search engines cannot see this investigation's E/R/F/W/C IDs" if english else "你是多模态调查代理 HI。搜索引擎看不到本次调查的材料")
    request = {"system_prompt": system,
               "payload": {"available_actions": [{"name": "SEARCH", "description": "Image-input search is not available"}]}}
    if rendered:
        request["state_render"] = {"renderer": "compact_16k", "raw_chars": 10, "rendered_chars": 5}
    (run / "requests" / "request_0001" / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    return run


def test_r0f_launcher_checks_specs_features_and_teacher_rendering(tmp_path: Path) -> None:
    # 10-07：r0f 改为英文规格 hi@0.7.0 / gia@0.8.0（= hi@0.6.0 / gia@0.7.0 + language=en）
    launch = _launch()
    assert launch.JOBS["r0f"] is launch.round_r0f and launch.R0F_SPECS == ["--hi-spec", "hi@0.7.0", "--gia-spec", "gia@0.8.0"]
    assert (ROOT / "scripts" / "_r0f.cmd").read_text(encoding="utf-8").strip().splitlines()[-2] == "python scripts\\_launch.py r0f"
    on = {"ingest_keyframes": 8, "language": "en"}
    good = launch.r0f_signature(_fake_run(tmp_path / "a", specs=("0.7.0", "0.8.0"), features=on, rendered=True))
    assert good["ok"], good
    raw = launch.r0f_signature(_fake_run(tmp_path / "b", specs=("0.7.0", "0.8.0"), features=on, rendered=False))
    assert not raw["ok"] and "原始全量视图" in raw["why"]
    old = launch.r0f_signature(_fake_run(tmp_path / "c", specs=("0.6.0", "0.7.0"), features={"ingest_keyframes": 8},
                                         rendered=True, english=False))
    assert not old["ok"] and "规格不是" in old["why"] and "不是英文规格" in old["why"] and "英文 HI 请求" in old["why"]
