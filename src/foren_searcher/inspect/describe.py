"""Fixed describer VLM for INSPECT comparison images (10-12, from detect/describe.py v2.4).

compare_media / video_structure return side-by-side images (a changed region, a different face, footage only in A).
In evidC27c the comparison had found 205's refrigerator -> vending machine and backpack -> cardboard box edits, but
the HI described them as "a dancer's clothing / prop". The backend therefore adds one short "Visual check of
comparison image k" line per image, written by a vision-language model.

The describer is part of the tool, not of the model being evaluated: it is fixed per deployment so that the same
tool call gives the same observation whichever HI model is running. Resolution order:

1. FOREN_DESCRIBE_ENDPOINT + FOREN_DESCRIBE_MODEL (OpenAI-compatible /v1 endpoint), optional FOREN_DESCRIBE_THINKING=1;
2. otherwise the run's own model endpoint (openai_compatible model config), which is what DETECT v2.4 did;
3. otherwise no describer (the image lines are simply absent).

The chosen identity and its source are recorded in run_config (inspect.describer).
"""

from __future__ import annotations

import base64
import os
from typing import Any

ENDPOINT_ENV, MODEL_ENV, THINKING_ENV = "FOREN_DESCRIBE_ENDPOINT", "FOREN_DESCRIBE_MODEL", "FOREN_DESCRIBE_THINKING"


def _is_ollama(endpoint: str) -> bool:
    return ":11434" in endpoint or os.environ.get("FOREN_TEACHER_BACKEND", "").lower() == "ollama"


class EndpointDescriber:
    def __init__(self, *, endpoint: str, model: str, thinking: bool = False, timeout_s: float = 180.0,
                 max_tokens: int = 160, transport: Any = None, source: str = "explicit"):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.transport = transport
        self.source = source
        self.calls: list[dict] = []

    @classmethod
    def from_env(cls) -> "EndpointDescriber | None":
        endpoint, model = os.environ.get(ENDPOINT_ENV, "").strip(), os.environ.get(MODEL_ENV, "").strip()
        if not endpoint or not model:
            return None
        return cls(endpoint=endpoint, model=model, thinking=os.environ.get(THINKING_ENV, "") in {"1", "true"},
                   source="env")

    @classmethod
    def from_model_config(cls, config: Any) -> "EndpointDescriber | None":
        if not isinstance(config, dict) or config.get("model_provider") != "openai_compatible":
            return None
        endpoint, model = config.get("endpoint"), config.get("model")
        if not isinstance(endpoint, str) or not isinstance(model, str):
            return None
        return cls(endpoint=endpoint, model=model, thinking=bool(config.get("thinking")), source="run_model")

    @classmethod
    def resolve(cls, model_config: Any = None) -> "EndpointDescriber | None":
        return cls.from_env() or cls.from_model_config(model_config)

    def identity(self) -> dict:
        return {"model": self.model, "endpoint": self.endpoint, "source": self.source}

    def describe(self, jpeg: bytes, question: str) -> str:
        """Synchronous (called from the backend worker thread). Returns the answer; raises on HTTP / format errors."""

        import httpx

        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        body: dict[str, Any] = {"model": self.model, "temperature": 0.2, "max_tokens": self.max_tokens,
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": question},
                                    {"type": "image_url", "image_url": {"url": image}}]}]}
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
            if _is_ollama(self.endpoint):
                body["reasoning_effort"] = "none"
        with httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=10.0), transport=self.transport,
                          trust_env=False) as client:
            response = client.post(self.endpoint + "/chat/completions", json=body)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        text = message.get("content") or ""
        if isinstance(text, list):
            text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
        text = str(text)
        if "</think>" in text:                     # a reasoning block left in the content
            text = text.rsplit("</think>", 1)[1]
        text = " ".join(text.split())
        self.calls.append({"question": question[:60], "answer": text[:200]})
        return text


__all__ = ["ENDPOINT_ENV", "EndpointDescriber", "MODEL_ENV", "THINKING_ENV"]
