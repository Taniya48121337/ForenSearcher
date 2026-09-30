"""Use an authenticated Codex CLI as a policy, with tools executed by ForenSearcher."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

from foren_searcher.local_models import extract_json_object
from foren_searcher.model_images import ModelImageInputs


class CliGenerator(ModelImageInputs):
    def __init__(self, *, store: Any, output_dir: Path, instruction: str = "") -> None:
        super().__init__(store=store)
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.generation_observer = None
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.codex = shutil.which("codex")
        if not self.codex:
            raise RuntimeError("Codex CLI is not installed")
        config_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        self.model_config = {key: config.get(key) for key in ("model", "model_provider", "model_reasoning_effort", "profile")}
        # 2026-09-30：Codex 的默认模型会随 config.toml / 客户端更新变化（09-30 变成 gpt-6.1-sol，ChatGPT 账号不支持，
        # 11 个 run 全部 400）。FOREN_CODEX_MODEL 显式指定教师模型，保证与之前的轮次同一个教师；记进 transport.json。
        self.model_override = os.environ.get("FOREN_CODEX_MODEL") or None
        if self.model_override:
            self.model_config = {**self.model_config, "model": self.model_override, "model_override": True}

    async def generate_json(self, *, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            self.sequence += 1
            return await asyncio.to_thread(self._generate, system_prompt, payload, self.sequence)

    def _generate(self, system_prompt: str, payload: dict[str, Any], sequence: int) -> dict[str, Any]:
        directory = self.output_dir / f"request_{sequence:04d}"
        directory.mkdir(exist_ok=False)
        actor_dir = directory / "actor"
        actor_dir.mkdir()
        view = payload.get("model_view", payload.get("global_view", {}))
        image_message, image_ids = self._image_message(view)
        image_notes: list[str] = []
        image_paths: list[Path] = []
        if image_message:
            for item in image_message["content"]:
                if item["type"] == "text":
                    image_notes.append(item["text"])
                elif item["type"] == "image_url":
                    path = actor_dir / f"image_{len(image_paths) + 1:02d}.png"
                    path.write_bytes(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
                    image_paths.append(path)
        request = {
            "system_prompt": system_prompt + ("\n" + self.instruction if self.instruction else ""),
            "payload": copy.deepcopy(payload),
            "image_notes": image_notes,
            "image_evidence_ids": image_ids,
        }
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        prompt = (
            "You are a policy in a separate executable investigation environment. "
            "Use only the supplied state and attached images. Do not use any Codex tools, "
            "shell, filesystem, network or external memory. Do not inspect this directory. "
            "The host executes your chosen action and will request the subsequent decision. "
            "Return exactly one JSON object conforming to the role's instructions. "
            "Do not invent observations, evidence, or future results.\n\n"
            + json.dumps(request, ensure_ascii=False)
        )
        result_path = directory / "response.json"
        command = [self.codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                   "--color", "never", "--json", "-C", str(actor_dir), "-o", str(result_path)]
        if self.model_override:
            command.extend(["-m", self.model_override])
        for path in image_paths:
            command.extend(["--image", str(path)])
        command.extend(["--", "-"])
        started = time.monotonic()
        attempts = []
        for attempt in range(3):
            completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace",
                                       capture_output=True, timeout=600, check=False)
            attempts.append({"attempt": attempt + 1, "returncode": completed.returncode})
            (directory / f"attempt_{attempt + 1:02d}_events.jsonl").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode == 0 or "Selected model is at capacity" not in completed.stdout:
                break
            if attempt < 2:
                time.sleep(15)
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        (directory / "cli_events.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
        (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        tool_items = [e.get("item", {}) for e in events
                      if e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}]
        usage = next((e.get("usage", {}) for e in reversed(events) if e.get("type") == "turn.completed"), {})
        metadata = {"returncode": completed.returncode, "elapsed_s": time.monotonic() - started,
                    "usage": usage, "external_tool_items": tool_items,
                    "transport": "codex_cli_chatgpt", "images": len(image_paths)}
        metadata["attempts"] = attempts
        metadata["configured_model"] = self.model_config
        (directory / "transport.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if completed.returncode or tool_items or not result_path.is_file():
            reason = next((str((e.get("error") or {}).get("message") or e.get("message") or "")
                           for e in reversed(events) if e.get("type") in {"turn.failed", "error"}), "")
            raise RuntimeError(f"Policy transport failed or used out-of-environment tools: {directory}"
                               + (f"；Codex 报错：{reason[:300]}" if reason else ""))
        raw = result_path.read_text(encoding="utf-8")
        output = extract_json_object(raw)
        if image_ids:
            observed = view.setdefault("content_observed_evidence_ids", [])
            observed.extend(eid for eid in image_ids if eid not in observed)
            view["model_observed_image_ids"] = list(image_ids)
        if self.generation_observer:
            self.generation_observer({"workspace_id": payload.get("workspace_id"), "model": self.model_config.get("model") or "configured_codex_model",
                "provider": "codex_cli_chatgpt", "raw_output": raw,
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "image_evidence_ids": image_ids})
        print(json.dumps({"request": sequence, "role": "HI" if payload.get("workspace_id") else "GIA",
                          "images": len(image_ids), "output": output}, ensure_ascii=False), flush=True)
        return output
