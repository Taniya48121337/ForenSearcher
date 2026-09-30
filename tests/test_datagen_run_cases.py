"""批量跑调查的离线部分：计划、媒体标签、来源排除、服务器教师的落盘格式与重试、execution_links、成本统计。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from foren_searcher.datagen.links import LinkError, link_events, write_execution_links
from foren_searcher.datagen.run_cases import KeyExclusion, media_inputs, plan, run_name_for, run_stats
from foren_searcher.datagen.cases import Case
from foren_searcher.datagen.teacher import EndpointTeacher, RoleInstructions, TeacherTransportError


def _case(root: Path, source_id: str, *, split: str = "dev_train", media: int = 1) -> Path:
    (root / "media").mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(media):
        (root / "media" / f"{source_id}_{i}.mp4").write_bytes(b"x")
        items.append({"ref": f"I{i}", "path": f"media/{source_id}_{i}.mp4", "media_type": "video/mp4"})
    value = {"case_id": f"videosham:{source_id}", "dataset": "videosham", "source_id": source_id, "split": split,
             "task_kind": "video_diff", "task_text": "比较 A 与 B", "family_key": f"videosham:{source_id}", "media": items,
             "source_policy": {"mode": "provided_media"}}
    path = root / f"videosham_{source_id}.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_plan_filters_split_ids_and_marks_status(tmp_path: Path) -> None:
    cases, runs = tmp_path / "cases", tmp_path / "runs"
    _case(cases, "0001")
    _case(cases, "0002", split="test")
    items = plan(cases, runs_dir=runs, round_tag="r0", samples=2)
    assert [p.run_name for p in items] == ["videosham_0001_r0_1", "videosham_0001_r0_2"]
    assert all(p.status == "todo" for p in items)
    # 给 ids 时不看划分；case_id 与文件名都认
    assert [p.case_id for p in plan(cases, runs_dir=runs, round_tag="r0", ids=["videosham:0002"])] == ["videosham:0002"]
    assert [p.case_id for p in plan(cases, runs_dir=runs, round_tag="r0", ids=["videosham_0002"])] == ["videosham:0002"]
    with pytest.raises(SystemExit):
        plan(cases, runs_dir=runs, round_tag="r0", ids=["videosham:9999"])
    done = runs / "videosham_0001_r0_1"
    done.mkdir(parents=True)
    (done / "result.json").write_text("{}")
    (done / "execution_links.json").write_text("[]")
    (runs / "videosham_0001_r0_2").mkdir()
    assert [p.status for p in plan(cases, runs_dir=runs, round_tag="r0", samples=2)] == ["done", "incomplete"]


def test_media_are_labelled_in_order_like_pilot_runs(tmp_path: Path) -> None:
    path = _case(tmp_path, "0003", media=2)
    media = media_inputs(Case.load(path), path)
    assert [(m.source_ref, m.label, m.original_filename, m.modalities) for m in media] == [
        ("I0", "材料 A", "A.mp4", ("video",)), ("I1", "材料 B", "B.mp4", ("video",))]
    (tmp_path / "media" / "0003_1.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        media_inputs(Case.load(path), path)


def test_key_exclusion_normalizes_and_sees_through_archive(tmp_path: Path) -> None:
    ex = KeyExclusion(("https://www.snopes.com/fact-check/eiffel-tower/",), tmp_path / "log.jsonl")
    assert ex.matches("http://snopes.com/fact-check/eiffel-tower")
    assert ex.matches("https://web.archive.org/web/20160715/https://www.snopes.com/fact-check/eiffel-tower/")
    assert ex.matches("https://snopes.com/fact-check/eiffel-tower/amp")
    assert not ex.matches("https://snopes.com/fact-check/other")
    ex.record("source_open", "u")
    assert json.loads((tmp_path / "log.jsonl").read_text())["stage"] == "source_open"


def _teacher(tmp_path: Path, handler, **kwargs) -> tuple[EndpointTeacher, list[dict]]:
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return handler(len(seen))

    teacher = EndpointTeacher(store=None, output_dir=tmp_path / "requests", endpoint="http://t/v1", model="q122",
                              instruction={"HI": "HI 指令", "GIA": "GIA 指令"}, transport=httpx.MockTransport(wrapped),
                              sleep=lambda s: None, **kwargs)
    return teacher, seen


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content, "reasoning_content": "想"}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 100, "completion_tokens": 7}})


def test_endpoint_teacher_writes_same_layout_as_codex(tmp_path: Path) -> None:
    teacher, seen = _teacher(tmp_path, lambda n: _ok('{"decision": "FINISH", "answer": "x"}'))
    events: list[dict] = []
    teacher.generation_observer = events.append
    out = asyncio.run(teacher.generate_json(system_prompt="学生提示", payload={"global_view": {"task": "t"}}))
    assert out == {"decision": "FINISH", "answer": "x"}
    directory = tmp_path / "requests" / "request_0001"
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    assert request["system_prompt"] == "学生提示\nGIA 指令"           # GIA 调用拼 GIA 的教师指令
    assert request["student_system_prompt"] == "学生提示" and request["teacher_instruction"] == "GIA 指令"
    assert (directory / "response.json").read_text(encoding="utf-8") == '{"decision": "FINISH", "answer": "x"}'
    assert (directory / "reasoning.txt").read_text(encoding="utf-8") == "想"
    transport = json.loads((directory / "transport.json").read_text(encoding="utf-8"))
    assert transport["returncode"] == 0 and transport["usage"] == {"input_tokens": 100, "output_tokens": 7}
    assert seen[0]["messages"][0]["content"] == "学生提示\nGIA 指令" and "chat_template_kwargs" not in seen[0]
    assert events[0]["raw_output"] == '{"decision": "FINISH", "answer": "x"}' and events[0]["workspace_id"] is None
    asyncio.run(teacher.generate_json(system_prompt="S", payload={"workspace_id": "W1-1", "model_view": {}}))
    assert json.loads((tmp_path / "requests" / "request_0002" / "request.json").read_text(encoding="utf-8"))["system_prompt"] == "S\nHI 指令"


def test_endpoint_teacher_retries_then_records_failure(tmp_path: Path) -> None:
    teacher, seen = _teacher(tmp_path, lambda n: httpx.Response(503) if n == 1 else _ok('{"action": "x"}'), thinking=False)
    assert asyncio.run(teacher.generate_json(system_prompt="S", payload={"workspace_id": "W1-1", "model_view": {}})) == {"action": "x"}
    assert len(seen) == 2 and seen[0]["chat_template_kwargs"] == {"enable_thinking": False}

    bad, _ = _teacher(tmp_path / "b", lambda n: httpx.Response(400, text="bad request"))
    with pytest.raises(TeacherTransportError):
        asyncio.run(bad.generate_json(system_prompt="S", payload={"global_view": {}}))
    transport = json.loads((tmp_path / "b" / "requests" / "request_0001" / "transport.json").read_text(encoding="utf-8"))
    assert transport["returncode"] == 1 and "400" in transport["error"]
    assert not (tmp_path / "b" / "requests" / "request_0001" / "response.json").exists()


def test_role_instructions_accept_shared_text() -> None:
    shared = RoleInstructions("共用")
    assert shared.system_prompt("S", {"workspace_id": "W1"}) == "S\n共用" == shared.system_prompt("S", {})
    assert RoleInstructions("").system_prompt("S", {}) == "S"


def _event(i: int, kind: str, payload: dict, actor: str = "W1-1", role: str = "HI") -> dict:
    return {"id": i, "event_type": kind, "actor_id": actor, "actor_role": role, "payload": payload}


def test_link_events_matches_pilot_semantics() -> None:
    rows = [_event(1, "raw_model_output", {"raw_output": '{"action":"bad"}'}),
            _event(2, "raw_model_output", {"raw_output": '{"action":"good"}'}),
            _event(3, "model_output", {"output": {"action": "good"}}),
            _event(4, "observation", {})]
    linked = link_events(rows)
    assert [r["status"] for r in linked] == ["proposal_not_accepted", "executed"]


def _run_dir(tmp_path: Path, outputs: list[str], events: list[dict]) -> Path:
    run = tmp_path / "run"
    for i, text in enumerate(outputs, 1):
        d = run / "requests" / f"request_{i:04d}"
        d.mkdir(parents=True)
        (d / "response.json").write_text(text, encoding="utf-8")
        (d / "request.json").write_text(json.dumps({"payload": {"workspace_id": "W1-1"}}), encoding="utf-8")
        (d / "transport.json").write_text(json.dumps({"returncode": 0, "usage": {"input_tokens": 10, "output_tokens": 2},
                                                      "images": 1, "elapsed_s": 1.5}), encoding="utf-8")
    (run / "trajectory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return run


def test_write_execution_links_and_stats(tmp_path: Path) -> None:
    events = [_event(1, "raw_model_output", {"raw_output": '{"action":"a"}'}), _event(2, "model_output", {"output": {"action": "a"}}),
              _event(3, "workspace_report", {})]
    run = _run_dir(tmp_path, ['{"action":"a"}'], events)
    summary = write_execution_links(run)
    assert summary["statuses"] == {"report_committed": 1}
    assert json.loads((run / "execution_links.json").read_text())[0]["request_id"] == "request_0001"
    stats = run_stats(run)
    assert (stats["teacher_calls"], stats["input_tokens"], stats["output_tokens"], stats["images"], stats["roles"]["HI"]) == (1, 10, 2, 1, 1)
    assert stats["workspace_reports"] == 1


def test_write_execution_links_rejects_mismatch(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, ['{"action":"a"}', '{"action":"b"}'], [_event(1, "raw_model_output", {"raw_output": '{"action":"a"}'})])
    with pytest.raises(LinkError):
        write_execution_links(run)


def test_run_name_is_filesystem_safe(tmp_path: Path) -> None:
    path = _case(tmp_path, "0069", media=0)       # case_id 本身已由 Case 校验，只有轮次标签可能带怪字符
    assert run_name_for(Case.load(path), "r0 test/x", 3) == "videosham_0069_r0-test-x_3"


def test_student_system_prompt_strips_teacher_instruction() -> None:
    from foren_searcher.datagen.runlog import student_system_prompt

    assert student_system_prompt({"system_prompt": "S", "student_system_prompt": "S0"}, {}, "HI") == "S0"
    assert student_system_prompt({"system_prompt": "学生\n指令"}, {"instruction": "指令"}, "GIA") == "学生"
    assert student_system_prompt({"system_prompt": "学生\n甲"}, {"instruction": {"HI": "甲", "GIA": "乙"}}, "HI") == "学生"
    assert student_system_prompt({"system_prompt": "学生\n甲"}, {"instruction": {"HI": "甲", "GIA": "乙"}}, "GIA") == "学生\n甲"
    assert student_system_prompt({"system_prompt": "学生\n指令\n"}, {"instruction": "指令\n"}, "HI") == "学生"
    assert student_system_prompt({"system_prompt": "学生"}, {"instruction": ""}, "HI") == "学生"
