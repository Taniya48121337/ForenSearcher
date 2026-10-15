"""evidb3 冒烟暴露的两处工具层问题（10-05）。

1. 部署不允许上传调查图片时，SEARCH 的动作清单不再列 image / text_image 输入（video_151 七次以图搜图全被拒）。
2. 本机服务报"超出上下文窗口"时，教师减半附图重发，而不是让整个 run 崩掉（video_001 GIA 33,299 > 32,768 token）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from foren_searcher.datagen.teacher import EndpointTeacher, TeacherContextOverflow, TeacherTransportError


def _harness(allow: bool | None):
    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.store import GlobalStore
    from foren_searcher.workspace import WorkspaceHarness

    search = SimpleNamespace() if allow is None else SimpleNamespace(allows_image_uploads=allow)
    return WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=GlobalStore(), search_service=search)  # type: ignore[arg-type]


def _search_action(harness) -> dict:
    return next(a for a in harness.eligible_actions() if a["name"] == "SEARCH")


def _input_types(action: dict) -> set:
    return {b["properties"]["input_type"]["const"] for b in action["schema"]["oneOf"] if "input_type" in b["properties"]}


def test_search_schema_hides_image_inputs_when_uploads_disallowed() -> None:
    hidden = _search_action(_harness(False))
    assert _input_types(hidden) == {"text"}
    assert "not available" in hidden["description"]
    assert any("source" in b["properties"] for b in hidden["schema"]["oneOf"])     # 打开 URL / R 照旧
    for allow in (True, None):                                                     # 允许上传、或旧服务没有该属性：不变
        assert _input_types(_search_action(_harness(allow))) == {"text", "image", "text_image"}


def test_free_search_service_without_reverse_engine_reports_no_uploads() -> None:
    from foren_searcher.search.factory import create_free_search_service
    from foren_searcher.store import GlobalStore

    assert create_free_search_service(store=GlobalStore(), reverse_engine="none").allows_image_uploads is False


class _FakeImages(EndpointTeacher):
    """不读真实图片：附图消息只按 view['image_inputs'] 生成占位内容。"""

    def _image_message(self, view):
        ids = [image["evidence_id"] for image in view.get("image_inputs", [])]
        if not ids:
            return None, []
        content = [{"type": "text", "text": "imgs"}]
        for _ in ids:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}})
        return {"role": "user", "content": content}, ids


OVERFLOW = ('{"error":{"message":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"request (33299 tokens) exceeds the available '
            'context size (32768 tokens), try increasing it\\",\\"type\\":\\"exceed_context_size_error\\"}}"}}')


def _ok() -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": '{"decision": "FINISH", "answer": "1. x"}'},
                                                  "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}})


def _teacher(tmp_path: Path, handler) -> tuple[EndpointTeacher, list]:
    seen: list = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][1]["content"]
        seen.append(sum(1 for item in user if isinstance(item, dict) and item.get("type") == "image_url")
                    if isinstance(user, list) else 0)
        return handler(len(seen))

    teacher = _FakeImages(store=None, output_dir=tmp_path / "requests", endpoint="http://t/v1", model="q",
                          transport=httpx.MockTransport(wrapped), sleep=lambda s: None)
    return teacher, seen


def _payload(n: int) -> dict:
    return {"global_view": {"task": "t", "image_inputs": [{"evidence_id": f"E{i}"} for i in range(1, n + 1)],
                            "image_inputs_omitted": []}}


def test_context_overflow_halves_images_until_it_fits(tmp_path: Path) -> None:
    teacher, seen = _teacher(tmp_path, lambda n: httpx.Response(400, text=OVERFLOW) if n <= 2 else _ok())
    payload = _payload(8)
    assert asyncio.run(teacher.generate_json(system_prompt="S", payload=payload))["decision"] == "FINISH"
    assert seen == [8, 4, 2]
    directory = tmp_path / "requests" / "request_0001"
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    assert request["image_evidence_ids"] == ["E1", "E2"]                       # 落盘 = 最后实际发出的请求
    assert len(list((directory / "actor").iterdir())) == 2
    view = request["payload"]["global_view"]
    assert [i["evidence_id"] for i in view["image_inputs"]] == ["E1", "E2"]
    assert view["image_inputs_omitted"] == ["E5", "E6", "E7", "E8", "E3", "E4"]
    transport = json.loads((directory / "transport.json").read_text(encoding="utf-8"))
    assert transport["returncode"] == 0
    assert [(r["images_before"], r["images_after"]) for r in transport["context_retries"]] == [(8, 4), (4, 2)]
    assert not (tmp_path / "requests" / "request_0002").exists()               # 同一请求目录，不多占序号


def test_context_overflow_without_images_still_fails(tmp_path: Path) -> None:
    teacher, seen = _teacher(tmp_path, lambda n: httpx.Response(400, text=OVERFLOW))
    with pytest.raises(TeacherContextOverflow):
        asyncio.run(teacher.generate_json(system_prompt="S", payload=_payload(2)))
    assert seen == [2, 1, 0]


def test_other_400_is_not_treated_as_overflow(tmp_path: Path) -> None:
    teacher, seen = _teacher(tmp_path, lambda n: httpx.Response(400, text="bad request"))
    with pytest.raises(TeacherTransportError) as info:
        asyncio.run(teacher.generate_json(system_prompt="S", payload=_payload(4)))
    assert not isinstance(info.value, TeacherContextOverflow) and seen == [4]


@pytest.mark.parametrize("endpoint, thinking, expected", [("http://127.0.0.1:11434/v1", False, "none"),
                                                          ("http://127.0.0.1:11434/v1", True, None),
                                                          ("http://gpu-server:8000/v1", False, None)])
def test_no_think_reaches_ollama_as_reasoning_effort(tmp_path: Path, endpoint: str, thinking: bool, expected) -> None:
    """10-05 换 Qwen3.8-27B：Ollama 不认 chat_template_kwargs，关思考要发 reasoning_effort=none；vLLM 照旧。"""

    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok()

    teacher = EndpointTeacher(store=None, output_dir=tmp_path / "r", endpoint=endpoint, model="q", thinking=thinking,
                              transport=httpx.MockTransport(handler), sleep=lambda s: None)
    asyncio.run(teacher.generate_json(system_prompt="S", payload={"global_view": {}}))
    assert seen[0].get("reasoning_effort") == expected
    assert ("chat_template_kwargs" in seen[0]) == (not thinking)


def test_evid27_launcher_profile() -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_launch_evid", root / "scripts" / "_launch_evid.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(root / "scripts"))
    spec.loader.exec_module(module)
    assert {"evid27_full", "evid27_score_sonnet"} <= set(module.JOBS)
    # 10-05 晚起默认 27B；修正优化计划起默认阶段 p1（10-05 晚起英文规格 hi@0.7.0 / gia@0.8.0，轮次 evidq27b）
    assert module.EVID27_ROUND == module.EVID_ROUND == "evidq27b" and module.EVID_SPECS == ("hi@0.7.0", "gia@0.8.0")
    assert module.EVID27_JOBS == "2"
    assert (root / module.EVID27_MODELFILE).read_text(encoding="utf-8").splitlines() == ["FROM qwen3.8:27b", "PARAMETER num_ctx 32768"]
    from foren_searcher.config import load_config

    config = load_config(root / module.EVID27_CONFIG)
    assert config.runtime.gia_turn_timeout_s == 900 and config.runtime.workspace_timeout_s == 3600
    assert config.runtime.max_workspaces_per_round == 3
    for name in ("_evid27_full.cmd", "_evid27_score_sonnet.cmd"):
        (root / "scripts" / name).read_bytes().decode("ascii")
