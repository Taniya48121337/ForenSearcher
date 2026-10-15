"""教师接法：服务器上的 OpenAI 兼容服务（vLLM 上的 Qwen3.5-122B），与 Codex CLI 教师写同一种 requests/ 目录。

runlog / 核验 / 导出读的是 run 目录里的 requests/request_NNNN/{request.json, response.json, transport.json, actor/image_NN.png}，
所以换教师只换"怎么拿到回答"，落盘格式与 experiments/sft_data/cli_generator.py（Codex CLI）逐字段一致：
- request.json：system_prompt（学生提示 + "\\n" + 教师指令）、payload、image_notes、image_evidence_ids
- response.json：模型回答原文（content）；思考内容另存 reasoning.txt（不进训练，留作以后抽 rationale）
- transport.json：returncode（0 = 成功）、elapsed_s、usage{input_tokens, output_tokens}、transport、images、attempts、configured_model

教师指令按角色取：HI 调用拼 specs/hi@x/teacher_instruction.md，GIA 调用拼 specs/gia@x/teacher_instruction.md。
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..local_models import extract_json_object
from ..model_images import ModelImageInputs
from ..policies import active_guards

TEACHER_ENDPOINT_ENV = "FOREN_TEACHER_ENDPOINT"
TEACHER_MODEL_ENV = "FOREN_TEACHER_MODEL"
TEACHER_API_KEY_ENV = "FOREN_TEACHER_API_KEY"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class TeacherTransportError(RuntimeError):
    """教师服务调用失败（连不上 / 非 2xx / 回包没有内容）。已写 transport.json（returncode=1），run 会记为失败。"""


class TeacherContextOverflow(TeacherTransportError):
    """服务端报请求超出上下文窗口（HTTP 400）。generate_json 会减半附图重发；没有附图可减时才让 run 失败。"""


_CONTEXT_OVERFLOW_MARKERS = ("exceed_context_size", "exceeds the available context size", "context length",
                             "maximum context length", "context_length_exceeded")


def _is_ollama(endpoint: str) -> bool:
    return ":11434" in endpoint or os.environ.get("FOREN_TEACHER_BACKEND", "").lower() == "ollama"


def _is_context_overflow(status: int, text: str) -> bool:
    lowered = text.lower()
    return status == 400 and any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


def role_of(payload: dict[str, Any]) -> str:
    return "HI" if payload.get("workspace_id") else "GIA"


def parse_role_models(model: str) -> dict[str, str]:
    """模型名可以按角色分开（10-12，学生 HI / GIA 各挂一套 LoRA，同一个 vLLM 服务）。

    "teacher"                 -> 两个角色都用 teacher
    "HI=hi_r1,GIA=gia_r1"     -> HI 调用发 hi_r1，GIA 调用发 gia_r1
    "HI=hi_r1"                -> 缺的角色报错（避免 GIA 悄悄用底座）
    """

    text = (model or "").strip()
    if "=" not in text:
        return {"HI": text, "GIA": text}
    out: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        role, _, name = part.partition("=")
        role, name = role.strip().upper(), name.strip()
        if role not in ("HI", "GIA") or not name:
            raise ValueError(f"模型名 {model!r} 里的 {part!r} 不是 HI=<名字> 或 GIA=<名字>")
        out[role] = name
    missing = {"HI", "GIA"} - set(out)
    if missing:
        raise ValueError(f"模型名 {model!r} 缺少角色 {sorted(missing)}；按角色写时两个都要给")
    return out


class RoleInstructions:
    """按角色拼接教师指令。instruction 可以是一段文字（两个角色共用，旧 Codex 用法）或 {"HI": ..., "GIA": ...}。"""

    def __init__(self, instruction: str | dict[str, str] = "") -> None:
        self.by_role = dict(instruction) if isinstance(instruction, dict) else {"HI": instruction, "GIA": instruction}

    def system_prompt(self, system_prompt: str, payload: dict[str, Any]) -> str:
        extra = self.by_role.get(role_of(payload), "")
        return system_prompt + ("\n" + extra if extra else "")


def render_for_model(payload: dict[str, Any], renderer: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """按档位渲染送给模型的状态（与本地学生 _generation_messages、导出 build_messages 同一渲染器）。

    renderer=None 时用 payload 里 runtime_constraints.state_renderer。legacy 返回 (payload, None)；
    渲染不进预算时发原视图并在 info 里记下原因（之后靠减图兜底）。修复回合的反馈字段原样带上。
    EndpointTeacher（render_inputs）、Codex CLI 教师与 bridge 重答共用（10-05：教师输入 = 学生输入）。"""

    from ..local_models import _payload_state_renderer
    from ..state_render import RenderError, payload_chars, render_payload

    renderer = renderer or _payload_state_renderer(payload)
    if renderer == "legacy":
        return payload, None
    clean = {key: value for key, value in payload.items() if key not in {"previous_invalid_output", "validation_error"}}
    try:
        rendered = render_payload(clean, renderer)
    except RenderError as exc:
        return payload, {"renderer": renderer, "error": str(exc)[:300], "chars": exc.chars, "budget": exc.budget}
    for key in ("previous_invalid_output", "validation_error"):     # 修复回合的反馈原样带上
        if key in payload:
            rendered[key] = payload[key]
    return rendered, {"renderer": renderer, "raw_chars": payload_chars(payload), "rendered_chars": payload_chars(rendered)}


class EndpointTeacher(ModelImageInputs):
    """OpenAI 兼容 /v1/chat/completions 教师。接口与 CliGenerator 相同：generate_json(system_prompt, payload) -> dict。

    与 Codex CLI 不同，服务端能并行：序号在锁里分配，HTTP 调用本身并发（vLLM 会自己合批）。
    """

    def __init__(self, *, store: Any, output_dir: Path, endpoint: str, model: str,
                 instruction: str | dict[str, str] = "", thinking: bool = True, max_tokens: int = 8192,
                 temperature: float = 0.6, timeout: float = 1800.0, attempts: int = 3, api_key: str | None = None,
                 transport: Any = None, sleep: Callable[[float], None] = time.sleep, render_inputs: bool = False) -> None:
        import httpx

        super().__init__(store=store)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.model_by_role = parse_role_models(model)
        self.instructions = RoleInstructions(instruction)
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.attempts = max(1, attempts)
        self.generation_observer: Callable[[dict[str, Any]], None] | None = None
        self.sequence = 0
        self.lock = asyncio.Lock()
        key = api_key if api_key is not None else os.environ.get(TEACHER_API_KEY_ENV)
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0), transport=transport, trust_env=False)
        self._sleep = sleep
        self._json_mode_unsupported = False        # 服务拒绝 response_format 后本进程不再发
        self._json_schema_unsupported = False      # 服务拒绝 json_schema 后退回 json_object（P0.2）
        # P0.1（10-05）：按 runtime_constraints.state_renderer 渲染送给模型的 payload（与本地学生 _generation_messages 同一渲染器）。
        # 以前 EndpointTeacher 一律发 legacy 全量视图——教师模式下这是有意的（教师看全量、导出时再按档位给学生渲染），
        # 但 EVID 基线用 student 模式让这个端点"当学生"，legacy 在 12 轮循环里顶过 32K（evidb4 ≥346 个回合被削附图）。
        # run_cases 在 --instruction-mode student 或显式 --state-renderer 时打开；request.json 另存 rendered_payload。
        self.render_inputs = render_inputs
        self.model_config = {"model": model, "model_by_role": dict(self.model_by_role), "model_provider": "openai_compatible", "endpoint": self.endpoint,
                             "thinking": thinking, "temperature": temperature, "max_tokens": max_tokens,
                             "render_inputs": render_inputs}

    def model_for(self, payload: dict[str, Any]) -> str:
        return self.model_by_role[role_of(payload)]

    async def generate_json(self, *, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            self.sequence += 1
            sequence = self.sequence
            directory = self.output_dir / f"request_{sequence:04d}"
            directory.mkdir(exist_ok=False)
            # 附图要在锁内取：_image_message 读的是当下的 view，放到线程里会与并发调用交错
            view = payload.get("model_view", payload.get("global_view", {}))
            image_message, image_ids = self._image_message(view)
            request = self._write_request(directory, system_prompt, payload, image_message, image_ids)
        context_retries: list[dict[str, Any]] = []
        while True:
            try:
                output, raw = await asyncio.to_thread(self._call, directory, request, image_message, sequence,
                                                      context_retries)
                break
            except TeacherContextOverflow as exc:
                # 10-05 evidb3 video_001：GIA 第 62 次请求 33,299 token > 本机 32,768 窗口，整个 run 崩掉。
                # 附图按优先级排好（点名读取 > 原件 > 报告引用），减半保留前面的；同一请求目录重写 request.json，
                # 落盘内容始终等于最后实际发出的请求，transport.json 记下每次减图。
                images = list(view.get("image_inputs") or [])
                if not images:
                    raise
                keep = len(images) // 2
                dropped = [image["evidence_id"] for image in images[keep:]]
                context_retries.append({"images_before": len(images), "images_after": keep,
                                        "dropped_evidence_ids": dropped, "error": str(exc)[:300]})
                view["image_inputs"] = images[:keep]
                omitted = list(view.get("image_inputs_omitted") or [])
                view["image_inputs_omitted"] = omitted + [eid for eid in dropped if eid not in omitted]
                view["image_inputs_reduced_for_context"] = True
                async with self.lock:
                    image_message, image_ids = self._image_message(view)
                    shutil.rmtree(directory / "actor", ignore_errors=True)
                    request = self._write_request(directory, system_prompt, payload, image_message, image_ids)
        if image_ids:
            observed = view.setdefault("content_observed_evidence_ids", [])
            observed.extend(eid for eid in image_ids if eid not in observed)
            view["model_observed_image_ids"] = list(image_ids)
        return output

    def _write_request(self, directory: Path, system_prompt: str, payload: dict[str, Any],
                       image_message: dict[str, Any] | None, image_ids: list[str]) -> dict[str, Any]:
        actor_dir = directory / "actor"
        actor_dir.mkdir()
        notes: list[str] = []
        count = 0
        for item in (image_message or {}).get("content", []):
            if item["type"] == "text":
                notes.append(item["text"])
            elif item["type"] == "image_url":
                count += 1
                (actor_dir / f"image_{count:02d}.png").write_bytes(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
        request = {"system_prompt": self.instructions.system_prompt(system_prompt, payload),
                   "student_system_prompt": system_prompt,          # runlog / export 用它和规格比较
                   "teacher_instruction": self.instructions.by_role.get(role_of(payload), ""),
                   "payload": copy.deepcopy(payload), "image_notes": notes, "image_evidence_ids": list(image_ids)}
        rendered, render_info = self._render(payload)
        if render_info:
            # 实际发给模型的是 rendered_payload；payload 仍是未压缩的原视图（runlog / 导出按它重放渲染）
            request["rendered_payload"] = rendered
            request["state_render"] = render_info
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        return request

    def _render(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """render_inputs 打开且档位不是 legacy 时按档位压缩（见 render_for_model）。"""

        if not self.render_inputs:
            return payload, None
        return render_for_model(payload)

    def _messages(self, request: dict[str, Any], image_message: dict[str, Any] | None) -> list[dict[str, Any]]:
        text = ("Return exactly one JSON object conforming to the role's instructions. "
                "Use only the supplied state and attached images; do not invent observations, evidence, or future results.\n\n"
                + (json.dumps(request["rendered_payload"], ensure_ascii=False, separators=(",", ":"))   # 与档位预算同一计法
                   if "rendered_payload" in request else json.dumps(request["payload"], ensure_ascii=False)))
        user: Any = text
        if image_message:
            user = [{"type": "text", "text": text}, *image_message["content"]]
        return [{"role": "system", "content": request["system_prompt"]}, {"role": "user", "content": user}]

    def _call(self, directory: Path, request: dict[str, Any], image_message: dict[str, Any] | None,
              sequence: int, context_retries: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], str]:
        import httpx

        body: dict[str, Any] = {"model": self.model_for(request["payload"]), "messages": self._messages(request, image_message),
                                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if active_guards().json_schema:
            # evidb6：受约束解码下模型会在合法 JSON 里"刷长度"（get_frames 的 times_s 从 0 数到 36570、GIA 无限追加任务），
            # 一次吃满 8192 token、44 秒。决定 / 报告实际不过 1–2K token，评测档封顶 4096
            body["max_tokens"] = min(self.max_tokens, SCHEMA_MAX_TOKENS)
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}        # vLLM / SGLang 认这个
            if _is_ollama(self.endpoint):
                # Ollama 的 OpenAI 兼容接口不认 chat_template_kwargs，关思考要用 reasoning_effort="none"（10-05 换 Qwen3.8-27B）；
                # vLLM 的 reasoning_effort 只收 low/medium/high，所以只对 Ollama 发
                body["reasoning_effort"] = "none"
        guards = active_guards()
        schema = self._output_schema(request["payload"]) if guards.json_schema and not self._json_schema_unsupported else None
        if schema is not None and not self._json_mode_unsupported:
            # P0.2（10-05）：schema 约束解码——HI 动作/报告、GIA 决定各自的 anyOf schema（与本地学生、OpenRouter 同源：
            # local_models._decision_output_schema）。只约束 JSON 外壳与字段，不约束事实内容；服务拒收时退回 json_object。
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "harness_decision", "strict": True, "schema": schema}}
        elif guards.json_mode and not self._json_mode_unsupported:
            # 评测兜底 0（10-05）：JSON 模式只保证输出是合法 JSON（evidb4 里约 230 次语法错），不约束字段；不改输入。
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        data: dict[str, Any] | None = None
        error = ""
        overflow = False
        for attempt in range(1, self.attempts + 1):
            try:
                response = self._client.post(self.endpoint + "/chat/completions", json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "error": error})
            else:
                attempts.append({"attempt": attempt, "status": response.status_code})
                if (body.get("response_format", {}).get("type") == "json_schema"
                        and (response.status_code >= 500 or (response.status_code == 400 and re.search(
                            r"response_format|json_schema|schema|grammar|\bformat\b", response.text, re.IGNORECASE)))):
                    # 服务不支持（或编译不了）这份 schema：本进程以后改发 json_object，这次也立即重发
                    self._json_schema_unsupported = True
                    attempts[-1]["json_schema_rejected"] = response.text[:300]
                    body["response_format"] = {"type": "json_object"}
                    response = self._client.post(self.endpoint + "/chat/completions", json=body, headers=self._headers)
                    attempts.append({"attempt": attempt, "status": response.status_code, "response_format": "json_object"})
                if (response.status_code == 400 and "response_format" in body
                        and re.search(r"response_format|json_object|\bformat\b", response.text, re.IGNORECASE)):
                    # 服务不支持 JSON 模式：去掉再发一次，本进程以后都不发
                    self._json_mode_unsupported = True
                    body.pop("response_format", None)
                    attempts[-1]["json_mode_rejected"] = True
                    response = self._client.post(self.endpoint + "/chat/completions", json=body, headers=self._headers)
                    attempts.append({"attempt": attempt, "status": response.status_code, "json_mode": False})
                if response.status_code < 400:
                    data = response.json()
                    break
                error = f"HTTP {response.status_code}: {response.text[:500]}"
                overflow = _is_context_overflow(response.status_code, response.text)
                if response.status_code not in _RETRY_STATUS:
                    break
            if attempt < self.attempts:
                self._sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
        message = ((data or {}).get("choices") or [{}])[0].get("message") or {}
        raw = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage_in = (data or {}).get("usage") or {}
        usage = {"input_tokens": usage_in.get("prompt_tokens"), "output_tokens": usage_in.get("completion_tokens")}
        ok = data is not None and bool(raw.strip())
        if data is not None and not raw.strip():
            error = "empty content（可能思考用完了 max_tokens）"
        metadata = {"returncode": 0 if ok else 1, "elapsed_s": time.monotonic() - started, "usage": usage,
                    "json_mode": "response_format" in body,
                    "response_format": (body.get("response_format") or {}).get("type"),
                    "external_tool_items": [], "transport": "openai_compatible", "images": len(request["image_evidence_ids"]),
                    "attempts": attempts, "configured_model": self.model_config,
                    "finish_reason": ((data or {}).get("choices") or [{}])[0].get("finish_reason")}
        if error and not ok:
            metadata["error"] = error
        if context_retries:
            metadata["context_retries"] = list(context_retries)
        (directory / "transport.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if reasoning:
            (directory / "reasoning.txt").write_text(reasoning, encoding="utf-8")
        if not ok:
            if overflow and data is None:
                raise TeacherContextOverflow(f"教师调用失败：{error}（{directory}）")
            raise TeacherTransportError(f"教师调用失败：{error}（{directory}）")
        (directory / "response.json").write_text(raw, encoding="utf-8")
        output = extract_json_object(raw)
        if self.generation_observer:
            self.generation_observer({"workspace_id": request["payload"].get("workspace_id"), "model": self.model_for(request["payload"]),
                                      "provider": "openai_compatible", "raw_output": raw,
                                      "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                                      "image_evidence_ids": request["image_evidence_ids"]})
        print(json.dumps({"request": sequence, "role": role_of(request["payload"]),
                          "images": len(request["image_evidence_ids"]), "output": output}, ensure_ascii=False), flush=True)
        return output, raw

    @staticmethod
    def _output_schema(payload: dict[str, Any]) -> dict[str, Any] | None:
        from ..local_models import _decision_output_schema
        from ..policies import guard_state

        try:
            # full_tool_schema：INSPECT 每个操作的 params 都给出字段（简化版的 params 是任意对象，数组没有上限）
            schema = _decision_output_schema(payload, full_tool_schema=True)
        except (KeyError, TypeError, IndexError):      # 自定义动作等形状不规整：不约束，退回 json_object
            return None
        if schema is None:
            return None
        state = guard_state()
        return bounded_schema(schema, payload, state.task_limits if state is not None else {})

    def close(self) -> None:
        self._client.close()


SCHEMA_MAX_TOKENS = 4096
SCHEMA_ARRAY_CAP = 32          # 没写上限的数组（取帧时刻、焦点 E、材料 …）一律封顶；运行时校验本来就不收更长的
_SCHEMA_DROP = ("not", "uniqueItems")      # llama.cpp 的 schema→grammar 不支持；仍由运行时校验


def bounded_schema(schema: dict[str, Any], payload: dict[str, Any], limits: dict[str, int]) -> dict[str, Any]:
    """P0.2 补丁（evidb6 复盘）：让 schema 解码不再"合法地刷长度"，并把运行时一定会拒的分支从语法里拿掉。

    - 所有数组设 maxItems（GIA 的 tasks / materials 用运行时上限，其余 32）——evidb6 里 3 次 GIA 输出因无限追加任务被截断、
      45 次 INVESTIGATE 超过每轮 3 个任务被拒（7 个 run 靠 forced_task_truncation 收尾）；
    - GIA 已不能再开轮（can_create_round=false）时只留 FINISH 分支——evidb6 里 16 次"预算已耗尽仍 INVESTIGATE"被拒；
    - 去掉 grammar 不支持的 not / uniqueItems（运行时校验照旧）。
    这些都只约束输出外壳，不改模型输入、不替模型写内容。
    """

    import copy as _copy

    value = _copy.deepcopy(schema)
    task_cap = limits.get("max_tasks_per_round")
    material_cap = limits.get("max_materials_per_task")

    def visit(node: Any, key: str | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, key)
            return
        if not isinstance(node, dict):
            return
        for drop in _SCHEMA_DROP:
            node.pop(drop, None)
        if node.get("type") == "array" and "maxItems" not in node:
            cap = task_cap if key == "tasks" and task_cap else material_cap if key == "materials" and material_cap else SCHEMA_ARRAY_CAP
            node["maxItems"] = max(cap, int(node.get("minItems", 0)))
        for child_key, child in node.items():
            if child_key == "properties" and isinstance(child, dict):
                for name, sub in child.items():
                    visit(sub, name)
            elif isinstance(child, (dict, list)):
                visit(child, key if child_key in ("items", "anyOf", "oneOf", "allOf") else child_key)

    visit(value)
    view = payload.get("global_view")
    constraints = view.get("runtime_constraints") if isinstance(view, dict) else None
    if (not payload.get("workspace_id") and isinstance(constraints, dict) and constraints.get("can_create_round") is False
            and not constraints.get("round_in_progress") and isinstance(value.get("anyOf"), list)):
        finish = [branch for branch in value["anyOf"]
                  if (branch.get("properties") or {}).get("decision", {}).get("enum") == ["FINISH"]
                  or (branch.get("properties") or {}).get("decision", {}).get("const") == "FINISH"]
        if finish:
            value["anyOf"] = finish
    return value


def spec_instructions(specs_root: str | Path, hi_spec: str, gia_spec: str) -> tuple[dict[str, str], dict[str, str]]:
    """从规格目录读两个角色的教师指令，返回 ({"HI": 文本, "GIA": 文本}, {"hi": 规格名@hash前12位, "gia": ...})。"""

    from .specs import Spec

    root = Path(specs_root)
    hi = Spec.load(root / hi_spec)
    gia = Spec.load(root / gia_spec)
    short = lambda spec: f"{spec.name}#{spec.hash().split(':')[-1][:12]}"   # noqa: E731
    return ({"HI": hi.teacher_instruction.strip(), "GIA": gia.teacher_instruction.strip()},
            {"hi": short(hi), "gia": short(gia)})


__all__ = ["EndpointTeacher", "RoleInstructions", "TeacherTransportError", "parse_role_models", "render_for_model", "role_of", "spec_instructions",
           "TEACHER_ENDPOINT_ENV", "TEACHER_MODEL_ENV"]
