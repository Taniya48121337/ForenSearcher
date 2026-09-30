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
import time
from pathlib import Path
from typing import Any, Callable

from ..local_models import extract_json_object
from ..model_images import ModelImageInputs

TEACHER_ENDPOINT_ENV = "FOREN_TEACHER_ENDPOINT"
TEACHER_MODEL_ENV = "FOREN_TEACHER_MODEL"
TEACHER_API_KEY_ENV = "FOREN_TEACHER_API_KEY"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class TeacherTransportError(RuntimeError):
    """教师服务调用失败（连不上 / 非 2xx / 回包没有内容）。已写 transport.json（returncode=1），run 会记为失败。"""


def role_of(payload: dict[str, Any]) -> str:
    return "HI" if payload.get("workspace_id") else "GIA"


class RoleInstructions:
    """按角色拼接教师指令。instruction 可以是一段文字（两个角色共用，旧 Codex 用法）或 {"HI": ..., "GIA": ...}。"""

    def __init__(self, instruction: str | dict[str, str] = "") -> None:
        self.by_role = dict(instruction) if isinstance(instruction, dict) else {"HI": instruction, "GIA": instruction}

    def system_prompt(self, system_prompt: str, payload: dict[str, Any]) -> str:
        extra = self.by_role.get(role_of(payload), "")
        return system_prompt + ("\n" + extra if extra else "")


class EndpointTeacher(ModelImageInputs):
    """OpenAI 兼容 /v1/chat/completions 教师。接口与 CliGenerator 相同：generate_json(system_prompt, payload) -> dict。

    与 Codex CLI 不同，服务端能并行：序号在锁里分配，HTTP 调用本身并发（vLLM 会自己合批）。
    """

    def __init__(self, *, store: Any, output_dir: Path, endpoint: str, model: str,
                 instruction: str | dict[str, str] = "", thinking: bool = True, max_tokens: int = 8192,
                 temperature: float = 0.6, timeout: float = 1800.0, attempts: int = 3, api_key: str | None = None,
                 transport: Any = None, sleep: Callable[[float], None] = time.sleep) -> None:
        import httpx

        super().__init__(store=store)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
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
        self.model_config = {"model": model, "model_provider": "openai_compatible", "endpoint": self.endpoint,
                             "thinking": thinking, "temperature": temperature, "max_tokens": max_tokens}

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
        output, raw = await asyncio.to_thread(self._call, directory, request, image_message, sequence)
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
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        return request

    def _messages(self, request: dict[str, Any], image_message: dict[str, Any] | None) -> list[dict[str, Any]]:
        text = ("Return exactly one JSON object conforming to the role's instructions. "
                "Use only the supplied state and attached images; do not invent observations, evidence, or future results.\n\n"
                + json.dumps(request["payload"], ensure_ascii=False))
        user: Any = text
        if image_message:
            user = [{"type": "text", "text": text}, *image_message["content"]]
        return [{"role": "system", "content": request["system_prompt"]}, {"role": "user", "content": user}]

    def _call(self, directory: Path, request: dict[str, Any], image_message: dict[str, Any] | None,
              sequence: int) -> tuple[dict[str, Any], str]:
        import httpx

        body: dict[str, Any] = {"model": self.model, "messages": self._messages(request, image_message),
                                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        data: dict[str, Any] | None = None
        error = ""
        for attempt in range(1, self.attempts + 1):
            try:
                response = self._client.post(self.endpoint + "/chat/completions", json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "error": error})
            else:
                attempts.append({"attempt": attempt, "status": response.status_code})
                if response.status_code < 400:
                    data = response.json()
                    break
                error = f"HTTP {response.status_code}: {response.text[:500]}"
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
                    "external_tool_items": [], "transport": "openai_compatible", "images": len(request["image_evidence_ids"]),
                    "attempts": attempts, "configured_model": self.model_config,
                    "finish_reason": ((data or {}).get("choices") or [{}])[0].get("finish_reason")}
        if error and not ok:
            metadata["error"] = error
        (directory / "transport.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if reasoning:
            (directory / "reasoning.txt").write_text(reasoning, encoding="utf-8")
        if not ok:
            raise TeacherTransportError(f"教师调用失败：{error}（{directory}）")
        (directory / "response.json").write_text(raw, encoding="utf-8")
        output = extract_json_object(raw)
        if self.generation_observer:
            self.generation_observer({"workspace_id": request["payload"].get("workspace_id"), "model": self.model,
                                      "provider": "openai_compatible", "raw_output": raw,
                                      "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                                      "image_evidence_ids": request["image_evidence_ids"]})
        print(json.dumps({"request": sequence, "role": role_of(request["payload"]),
                          "images": len(request["image_evidence_ids"]), "output": output}, ensure_ascii=False), flush=True)
        return output, raw

    def close(self) -> None:
        self._client.close()


def spec_instructions(specs_root: str | Path, hi_spec: str, gia_spec: str) -> tuple[dict[str, str], dict[str, str]]:
    """从规格目录读两个角色的教师指令，返回 ({"HI": 文本, "GIA": 文本}, {"hi": 规格名@hash前12位, "gia": ...})。"""

    from .specs import Spec

    root = Path(specs_root)
    hi = Spec.load(root / hi_spec)
    gia = Spec.load(root / gia_spec)
    short = lambda spec: f"{spec.name}#{spec.hash().split(':')[-1][:12]}"   # noqa: E731
    return ({"HI": hi.teacher_instruction.strip(), "GIA": gia.teacher_instruction.strip()},
            {"hi": short(hi), "gia": short(gia)})


__all__ = ["EndpointTeacher", "RoleInstructions", "TeacherTransportError", "role_of", "spec_instructions",
           "TEACHER_ENDPOINT_ENV", "TEACHER_MODEL_ENV"]
