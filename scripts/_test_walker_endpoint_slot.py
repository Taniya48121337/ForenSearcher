"""Offline test of OracleGenerator's endpoint slot backend (2026-10-14): a fake vLLM server answers 429 twice, then
rejects json_object once, then returns a JSON answer; a second slot returns empty content and must latch the generator
so the run cannot continue. No GPU, no network, no Codex. Run: python scripts/_test_walker_endpoint_slot.py
(tests/test_walker_endpoint_slot_20261014.py runs it in a subprocess, because it stubs foren_searcher.* in sys.modules)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# --- stub the heavy project imports oracle_generator pulls in (only the endpoint path is under test)
fs = types.ModuleType("foren_searcher")
dg = types.ModuleType("foren_searcher.datagen")
teacher = types.ModuleType("foren_searcher.datagen.teacher")
teacher.render_for_model = lambda payload: (payload, None)
walker = types.ModuleType("foren_searcher.datagen.walker")
walker.Decision = object
walker.decide_gia = walker.decide_hi = walker.resolve_slot_choice = lambda *a, **k: None
verify = types.ModuleType("foren_searcher.datagen.verify")
lang = types.ModuleType("foren_searcher.datagen.verify.lang")
lang.dates_in = lambda *a, **k: []
lang.label_hit = lambda *a, **k: False
local_models = types.ModuleType("foren_searcher.local_models")
local_models.extract_json_object = lambda raw: json.loads(raw)
model_images = types.ModuleType("foren_searcher.model_images")


class ModelImageInputs:
    def __init__(self, *, store=None):
        self.store = store


model_images.ModelImageInputs = ModelImageInputs
for name, mod in {"foren_searcher": fs, "foren_searcher.datagen": dg, "foren_searcher.datagen.teacher": teacher,
                  "foren_searcher.datagen.walker": walker, "foren_searcher.datagen.verify": verify,
                  "foren_searcher.datagen.verify.lang": lang, "foren_searcher.local_models": local_models,
                  "foren_searcher.model_images": model_images}.items():
    sys.modules[name] = mod

CALLS: list[dict] = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS.append(body)
        n = len(CALLS)
        if n <= 2:
            self.send_response(429); self.end_headers(); self.wfile.write(b"Too Many Requests"); return
        if n == 3 and "response_format" in body:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"response_format not supported"}'); return
        if n <= 4:
            out = {"choices": [{"message": {"content": json.dumps({"decision": "FINISH", "answer": "ok [E3]"}),
                                            "reasoning_content": "thought"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        else:   # empty content: thinking ate the budget
            out = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)


server = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
port = server.server_address[1]

os.environ.update({"FOREN_ORACLE_SLOT_ENDPOINT": f"http://127.0.0.1:{port}/v1", "FOREN_ORACLE_SLOT_MODEL": "qwen-teacher",
                   "FOREN_ORACLE_SLOT_ATTEMPTS": "3", "FOREN_ORACLE_SLOT_THINKING": "0"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "sft_data"))
import oracle_generator as og  # noqa: E402

og.time.sleep = lambda s: None          # no real backoff in the test
tmp = Path(tempfile.mkdtemp())
gen = og.OracleGenerator(store=None, output_dir=tmp / "requests")
assert gen.backend == "endpoint" and gen.codex is None, gen.backend
d = tmp / "requests" / "request_0001"; d.mkdir(parents=True); (d / "actor").mkdir()
img = d / "actor" / "image_01.png"; img.write_bytes(b"\x89PNG fake")
slots: list = []
raw = gen._codex(d, d / "actor", "S5_answer", "PROMPT TEXT", [img], slots, {"mock": True})
assert json.loads(raw)["answer"] == "ok [E3]", raw
assert len(CALLS) == 4, len(CALLS)   # 429, 429, 400 (json_object rejected) + immediate resend 200
assert slots[0]["attempts"][-1]["status"] == 200, slots[0]["attempts"]
assert slots[0]["usage"] == {"input_tokens": 100, "output_tokens": 20}
assert (d / "slot_S5_answer_output.txt").read_text() == raw
assert (d / "slot_S5_answer_reasoning.txt").read_text() == "thought"
assert gen._json_mode_unsupported is True
body = CALLS[0]
assert body["model"] == "qwen-teacher" and body["chat_template_kwargs"] == {"enable_thinking": False}
content = body["messages"][0]["content"]
assert content[0] == {"type": "text", "text": "PROMPT TEXT"} and content[1]["type"] == "image_url"
assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
print("first slot: 429 x2 -> json_object rejected -> 200; files + usage ok")

# second slot: empty content every time -> RuntimeError and latch
d2 = tmp / "requests" / "request_0002"; d2.mkdir(); (d2 / "actor").mkdir()
try:
    gen._codex(d2, d2 / "actor", "S4_report", "P", [], slots, {})
except RuntimeError as exc:
    assert "empty content" in str(exc), exc
else:
    raise AssertionError("expected failure")
assert gen.slot_failed and "S4_report" in gen.slot_failed
assert json.loads((d2 / "transport.json").read_text())["returncode"] == 1
try:
    gen._generate("sys", {"model_view": {}}, 3)
except RuntimeError as exc:
    assert "earlier language slot failed" in str(exc), exc
else:
    raise AssertionError("latch did not fire")
print("second slot: empty content -> failed, latch stops the run")
print("calls:", len(CALLS))
