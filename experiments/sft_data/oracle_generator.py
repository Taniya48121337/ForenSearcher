"""OracleGenerator：程序走（walker）、教师填（语言槽）。与 CliGenerator 同接口、同落盘格式（2026-10-12，walker 计划 D1）。

- 每次 generate_json 照 CliGenerator 写 requests/request_NNNN/{request.json, actor/image_NN.png, response.json, transport.json}；
  request.json 的 payload / rendered_payload / 附图与 Codex 教师完全相同，导出、runlog、核验器不用改。
- 决定由 foren_searcher.datagen.walker 按可见状态给出（只读 payload，案卷与标注不传进来）。
- 需要语言的步骤调 Codex 教师槽，输入 = 这一步学生那份渲染后的状态 + 本步附图（教师输入 = 学生输入），外加一句
  "这一步主机已定为报告 / FINISH / 第 2 轮派任务，status 是 X，只能引用这些 E"：
    S4 报告（HI 交报告）、S5 答案（GIA FINISH）、S2 任务文字（GIA 第 2 轮）。
  槽的原始输出存 slot_<name>_*.{json,txt}；程序再强制 task_id / status、过滤不可见的 E，写进 response.json。
- transport.json：transport=oracle_walker，returncode=0，usage/elapsed 只算槽调用（程序步为 0），
  oracle={rule, branch, kind}，slots=[{name, elapsed_s, usage, ...}]。run_stats 的 teacher_calls 仍按请求目录数，
  真正的教师调用数看 slots（compare 脚本里叫 llm_calls）。

环境变量：
  FOREN_ORACLE_SLOT_ENDPOINT  （10-14）服务器上自己部署的教师：OpenAI 兼容地址（…/v1）。设了就不用 Codex；
                            配 FOREN_ORACLE_SLOT_MODEL（vLLM 里的模型名，必填）、FOREN_ORACLE_SLOT_THINKING（默认 1）、
                            FOREN_ORACLE_SLOT_MAX_TOKENS（8192）、FOREN_ORACLE_SLOT_TEMPERATURE（0.3）、
                            FOREN_ORACLE_SLOT_ATTEMPTS（5，429/5xx 指数退避）、FOREN_ORACLE_SLOT_TIMEOUT（900 秒）、
                            FOREN_TEACHER_API_KEY（可选）。
  （10-14 晚）端点是 OpenRouter（…openrouter.ai/api/v1）时：key 取 FOREN_TEACHER_API_KEY，没有就用本机加密保存的
                            OpenRouter key（openrouter_settings）；思考开关用 reasoning.enabled（不发 chat_template_kwargs）；
                            走系统代理；FOREN_ORACLE_SLOT_PROVIDER=逗号分隔的供应商名 可固定供应商。
  FOREN_ORACLE_GT_REVIEW=1  （10-14 晚起默认关）S5 写完后按案卷 GT 提示教师重写。wk11 / wk12 对照：GT 提示不增加核验通过数，
                            GT 类核验意见还会教教师删掉正确内容来躲检查，所以默认不用 GT。
  FOREN_ORACLE_RATIONALE=0  （10-14 晚起默认开）程序定的 HI 动作，rationale 由教师按这一步的状态重写（S0 槽）；
                            关掉则保留程序模板（模板 rationale 在训练集里重复度很高）。
  FOREN_ORACLE_SLOT_EFFORT  槽调用的推理档（默认 medium，计划 §4"全部 medium 档"；只对 Codex 有意义）；
  FOREN_ORACLE_SLOT=mock    不调 Codex，槽输出用模板（只用于本地冒烟，产出不能进训练）；
  FOREN_ORACLE_SLOT=claude  （10-14）教师换成本机 Claude Code CLI（claude -p，走订阅额度）：FOREN_ORACLE_CLAUDE_MODEL
                            （默认 sonnet）、FOREN_ORACLE_CLAUDE_THINKING（思考 token 上限，默认 8000，0 = 不思考）；
                            调用方式与 verify/judges.py 的 ClaudeCliJudge 相同（无工具、不读本机设置、图片 base64）。
  FOREN_ORACLE_ARMS=1       （10-14）S5 对照：同一状态下另写几版答案存 slot_S5_arms.json（d0 首稿不用 GT；b 原方法
                            = GT 词提示重写；g 证据门控提示；n 首稿 + FOREN_ORACLE_BON-1 份重抽、按 GT 择优），
                            轨迹里实际交的是 d0；walker_run.py --arms 再把各版拷成单独的轮次去核验。
  FOREN_ORACLE_CASE         案卷路径（wk5 起）：只用于 S5 写完后的 GT 核对（label_expected / interval_iou / VCSL 标注段），
                            没对上时把"可能漏了什么"反馈给教师重写一次，要求只写证据支持的内容；walker 的决定从不读它。
                            （10-16 r0+）循环内充分性检查也用它（核验器要案卷的必需证据清单）。
  FOREN_INLOOP_SUFFICIENCY=1（10-16 r0+）S4 写完报告、GIA 看到之前，用核验器现成的充分性层（verify.sufficiency，claude-cli
                            sonnet，同一提示、同一 payload → 同一缓存键）核一次 COMPLETED 报告；判 completed_without_finishing
                            major / critical 时 status 改 INCOMPLETE，status_reason = status_rewrite.build_reason 过滤后的 missing
                            + walker 写的可修缺口（walker.inloop_actions），GIA 据此在轮次预算内派第 2 轮；第 2 轮的报告同样核。
                            范围（requirement_ids）：第 1 轮按核验器 plan 单元同一道评判题（judge_plan，结果同样进缓存），
                            第 2 轮按任务文字兜底。缓存写 run 目录的 inloop_verify_cache.jsonl（walker_run 核验前并进主缓存），
                            每次检查记一条到 inloop_sufficiency.json。E / L / C 第 1 轮报告是 INCOMPLETE 但没写跟进时，也补
                            walker 的可修缺口。mock 槽下不跑。
  FOREN_ORACLE_SLOT=slots   （10-15，对话 5a，r0 口径）教师按槽分：槽 → 后端的映射与各后端的 endpoint / model / 思考 / 温度
                            从冻结文件 teacher 一节读（默认 specs/r0_freeze.json，FOREN_TEACHER_FREEZE 可换）：
                            S0 rationale / S1 查询 / S2 任务文本 / S3 充分性 → qwen 后端（服务器 H20 自部署 vLLM，本机 OpenRouter）；
                            S4 报告 / S5 答案（含 S5_answer_revise 核验后只按证据重写）→ strong 后端（API 强模型）。
                            槽名按前缀归槽（S4_report_retry → S4，S5_answer_revise → S5）。所有后端都走同一个 OpenAI 兼容
                            /chat/completions 路径（_endpoint，429 / 5xx 退避同上），不调 Codex / Claude CLI。
                            每次槽调用的记录带 slot、slot_backend、backend_role、model、endpoint_host、usage、cost_usd。
                            服务器上：FOREN_TEACHER_QWEN_ENDPOINT / _MODEL（vLLM 地址与 served name）、FOREN_TEACHER_STRONG_ENDPOINT /
                            _MODEL 覆盖冻结值（字段名见冻结文件 endpoint_env / model_env）；FOREN_TEACHER_BACKUP=qwen,strong|all
                            让该后端整条 run 换成冻结文件 backup 里的模型（429 换备用由批处理整条重跑，不在一条 run 里混用）。
                            （10-15 r0 成本压缩）后端多一个类型：冻结文件 backends.<名>.type = "endpoint"（默认，上面那条
                            /chat/completions 路径）或 "codex"（本机 / 服务器的 Codex CLI，走 ChatGPT 订阅）。strong = codex 时
                            S4 / S5（含 S5_answer_revise）调 `codex exec`（只读沙箱、无工具、-m <model>、reasoning effort 取后端配置，
                            附图 --image，图注写进提示末尾），跨进程串行（文件锁 experiments/sft_data/.codex_slot.lock；--jobs 2 时
                            两个 run 的 Codex 调用排队，FOREN_CODEX_SERIAL=0 可关），429 / usage limit / 满载按 30、60、120、240、480 s
                            退避（总等待 ≤ FOREN_ORACLE_RETRY_MAX_S，默认 900 s），用尽后槽失败、run 停下（报错里带 usage limit，
                            批处理认得出）。槽记录 backend=codex、model（-m 请求的）、model_served（Codex 输出里能找到的模型名，找不到为
                            null）、codex_version、lock_wait_s、usage（input / cached_input / output token）。将来换 CLIProxyAPI 之类
                            OpenAI 兼容端点：只把冻结文件 strong 的 type 改回 endpoint、填 endpoint / model，代码不动。
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

from foren_searcher.datagen.teacher import render_for_model
from foren_searcher.datagen import walker as _walker
from foren_searcher.datagen.walker import Decision, decide_gia, decide_hi, resolve_slot_choice
from foren_searcher.datagen.verify.lang import dates_in as _dates_in, label_hit as _label_hit
from foren_searcher.local_models import extract_json_object
from foren_searcher.model_images import ModelImageInputs

INLOOP_FILE = "inloop_sufficiency.json"          # 10-16 r0+：循环内充分性检查的记录（run 目录）
INLOOP_CACHE = "inloop_verify_cache.jsonl"       # 同上，判官缓存（核验前并进主缓存）
_EREF = re.compile(r"\bE[1-9][0-9]*\b")
_BRACKET = re.compile(r"\[(E[1-9][0-9]*)\]")


MEDIA_RULES = (
            "- If the status is INCOMPLETE, write status_reason as the missing pieces separated by '; '. Each piece names the material "
            "and the time range where it lies, in the form 'Material B 2.267–2.400 s: whether the edge sliver is the person in black is "
            "unverified' or 'audio of Material A: not transcribed'; or says plainly that it cannot be obtained from the given materials "
            "(e.g. 'Material A has no audio track'). Take the times from the state; do not invent ranges.\n"
            "- Each Finding answers one part of model_view.task with what the attached images and the recorded tool observations "
            "actually show; give times in seconds on both sides where the task asks for them, and keep the certainty no higher than "
            "the evidence. Sampled frames prove only the sampled instants.\n"
            "- evidence_ids may contain only these observed E IDs: %s. Cite the derived frames / comparison images / crops that "
            "show the point, not just the source video.\n"
            "- Put precision limits, unexamined ranges and modalities that were not verified into limitations; an unverified "
            "modality is not 'no difference'.\n"
            "- The host notes below say which ranges and audio tracks were or were not examined; reflect each of them in a finding "
            "(with the E that shows it) or a limitation.\n"
            "- When stating times, ranges, areas or durations, copy the values exactly as a tool result or frame label in the state "
            "gives them; do not recompute them through a time mapping.\n"
            "- Never write that a record, tool result, transcript or output is truncated, cut off or not fully shown; if a value is "
            "not visible in the state, simply do not state it.\n"
            "- Refer to people and objects with the task's own wording (e.g. 'the person in black'); when other people or objects "
            "appear in the same frame, say which one you mean, so a reader cannot confuse them.\n"
            "- Describe in a finding only what the attached images or the recorded tool results show; an image that is not attached "
            "in this decision may be cited only for what its label states (its time and source).\n"
            "- A correspondence whose start / end is not established is a status_reason piece of the form 'correspondence A 3.333 s "
            "↔ B 25.880 s: start and end boundaries not established'.\n"
            "- A boundary is answered when a frame just inside shows the thing and a frame just outside does not: write it as lying "
            "between those two times (e.g. 'starts between 2.267 s and 2.300 s') and put the bracket width in limitations; that "
            "part is then answered, not missing. Only a boundary with no such pair of frames is unresolved.\n"
            "- A tool result that reports spans, cuts, speed changes or changed regions answers those parts at the precision it "
            "states; cite it and give that precision instead of calling the part unresolved.\n"
            "- Cite a tool's result record (the first E after 'Result Evidence' in its observation) for what its summary says "
            "(matched share, spans, speeds, cuts, sound); cite a comparison image or frame only for what that picture shows.\n"
            "- When frames of A and B were taken at the same or corresponding times, write a finding that says what each pair "
            "shows side by side, citing the A frame and the B frame.\n"
            "- When a comparison result lists corresponding spans (A range ↔ B range), those spans answer which segments correspond "
            "and their time ranges, at the precision the tool states (about 0.5 s): give each listed span with both ranges, citing "
            "the comparison result. Do not call that part unanswered, or set INCOMPLETE, only because boundaries are known to that "
            "precision; state the precision in limitations.\n"
            "- When a search found the video whose picture matched the given video and a comparison shows it contains the given "
            "footage, identify it as the source video found: give its title and address as its page record shows them, citing that "
            "page E, and cite the comparison result for the picture match. Whether it is the earliest upload is a separate question "
            "for limitations, not a reason to leave the source unidentified.\n"
        ) + "%s"
# 10-15 对话 5''（追加 3）：r0a l101 报告说 "reference photographs are not available"，E7 / E15 就在 state 里；1028_v9 关键的
# compare_media / OCR 记录一个不引、只引采样帧；c_46581202 F3 把"未核实"的句子挂上 [E2, E3, E4]
CONTRACT_RULES = (
            "- A limitation must not say that a record or material is unavailable, missing, not attached or not provided when it is "
            "among the observed E IDs above (for example reference photos that were retrieved, a transcript, an OCR or comparison "
            "record); if such a record was not compared or examined, say that, naming it.\n"
            "- Each statement cites the record that establishes it: the comparison result, the OCR or transcript record, the date "
            "record or the page that states it — not only sampled frames when such a record exists.\n"
            "- A statement asserts what the evidence establishes. What is not verified, not established or could not be determined "
            "goes into limitations, not into a statement with evidence_ids.\n"
        )
SEARCH_RULES = (
            "- If the status is INCOMPLETE, write status_reason as the missing facts separated by '; ', each naming what is "
            "missing and why (e.g. 'earliest publication: no dated copy of the picture was found'; 'where the photo was taken: "
            "no page that carries it was read').\n"
            "- Each Finding answers one part of model_view.task with what the read pages, date records, tool results and "
            "attached images actually show; keep the certainty no higher than the evidence.\n"
            "- evidence_ids may contain only these observed E IDs: %s.\n"
            "- Cite a page only for what that page's text says; a search result that was not read supports nothing beyond its "
            "title. Cite a date record (dates[R] … E) for a date and say what kind of date it is (declared by the page, upload, "
            "archive capture) — it shows 'no later than', not a first publication.\n"
            "- Cite a speech transcript or OCR record for what the given material says or shows in words (quote only the "
            "words shown in this state), and an attached image for what it shows; a reference photo shows only what that "
            "photo shows.\n"
            "- Say plainly whether the claim is accurate, misleading or false as far as the evidence shows, or that it could not "
            "be verified, and put what was not established into limitations.\n"
            "- Never write that a record, tool result, transcript or output is truncated, cut off or not fully shown; if a value is "
            "not visible in the state, simply do not state it.\n"
        ) + "%s"


class OracleGenerator(ModelImageInputs):
    def __init__(self, *, store: Any, output_dir: Path, instruction: str = "") -> None:
        import asyncio

        super().__init__(store=store)
        self.store = store                                  # 10-16 r0+：循环内充分性检查读工作区任务与 F 序号
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.generation_observer = None
        self.sequence = 0
        self.lock = asyncio.Lock()
        self.mock = os.environ.get("FOREN_ORACLE_SLOT", "").lower() == "mock"
        self.effort = os.environ.get("FOREN_ORACLE_SLOT_EFFORT") or "medium"
        # 10-14：服务器上教师是自己部署的模型（vLLM，OpenAI 兼容），不用 Codex。设了 FOREN_ORACLE_SLOT_ENDPOINT 就走端点，
        # 否则走 Codex CLI；mock 两者都不调。
        self.endpoint = (os.environ.get("FOREN_ORACLE_SLOT_ENDPOINT") or "").rstrip("/") or None
        self.endpoint_model = os.environ.get("FOREN_ORACLE_SLOT_MODEL") or None
        slot_kind = os.environ.get("FOREN_ORACLE_SLOT", "").lower()
        self.backend = "mock" if self.mock else ("slots" if slot_kind == "slots" else "endpoint" if self.endpoint
                                                 else "claude" if slot_kind == "claude" else "codex")
        self.codex = shutil.which("codex") if self.backend == "codex" else None
        if self.backend == "codex" and not self.codex:
            raise RuntimeError("Codex CLI is not installed (needed for the language slots; set FOREN_ORACLE_SLOT=slots to use the "
                               "frozen slot table, FOREN_ORACLE_SLOT_ENDPOINT for one deployed teacher, or FOREN_ORACLE_SLOT=mock for smoke tests)")
        self._backends: dict[str, dict[str, Any]] = {}      # 端点后端：名字 → 配置（含 httpx.Client，首次调用时建）
        self.slot_table: dict[str, Any] | None = None
        if self.backend == "slots":
            # 10-15 对话 5a：S0–S3 → 自部署 Qwen，S4/S5 → API 强模型；映射与后端参数都从冻结文件读
            self.slot_table = load_slot_table()
            self._backends = self.slot_table["backends"]
            self.model_override = None
            self.model_config = {"model": self.slot_table["summary"], "model_provider": "slot_table",
                                 "freeze_file": self.slot_table["file"], "freeze_id": self.slot_table.get("freeze_id"),
                                 "slots": {k: {"backend": b, "model": self._backends[b]["model"],
                                               "role": self._backends[b]["role"], "type": self._backends[b]["type"]}
                                           for k, b in self.slot_table["slots"].items()},
                                 "backends": {n: public_backend(c) for n, c in self._backends.items()}}
            for name, cfg in self._backends.items():
                if cfg["type"] == "codex":
                    cfg["codex_bin"] = find_codex(cfg)          # 没装 Codex CLI 就在建 generator 时报错，不等到 S4
                    self.model_config["backends"][name]["codex_version"] = codex_version(cfg["codex_bin"])
        elif self.backend == "endpoint":
            if not self.endpoint_model:
                raise RuntimeError("FOREN_ORACLE_SLOT_MODEL is required with FOREN_ORACLE_SLOT_ENDPOINT (the model name served by vLLM)")
            self.model_override = None
            self.openrouter = "openrouter.ai" in self.endpoint
            providers = [p.strip() for p in (os.environ.get("FOREN_ORACLE_SLOT_PROVIDER") or "").split(",") if p.strip()]
            self.model_config = {"model": self.endpoint_model,
                                 "model_provider": "openrouter" if self.openrouter else "openai_compatible", "endpoint": self.endpoint,
                                 "thinking": _env_flag("FOREN_ORACLE_SLOT_THINKING", True),
                                 "providers": providers or None,
                                 "max_tokens": int(os.environ.get("FOREN_ORACLE_SLOT_MAX_TOKENS") or 16384),
                                 "temperature": float(os.environ.get("FOREN_ORACLE_SLOT_TEMPERATURE") or 0.3),
                                 "attempts": int(os.environ.get("FOREN_ORACLE_SLOT_ATTEMPTS") or 5),
                                 "timeout_s": float(os.environ.get("FOREN_ORACLE_SLOT_TIMEOUT") or 900)}
            # 单端点（10-14 起的老路径）：所有槽同一个后端，参数与以前完全相同
            self._backends = {"endpoint": {**self.model_config, "name": "endpoint", "role": None, "openrouter": self.openrouter,
                                           "api_key": os.environ.get("FOREN_TEACHER_API_KEY") or None,
                                           "reasoning_effort": None, "thinking_off_slots": ["S0"], "trust_env": self.openrouter}}
        elif self.backend == "claude":
            from foren_searcher.datagen.verify.judges import find_claude_cli

            self.claude_cli = find_claude_cli()
            self.model_override = None
            self.model_config = {"model": os.environ.get("FOREN_ORACLE_CLAUDE_MODEL") or "sonnet",
                                 "model_provider": "claude_cli",
                                 "thinking_tokens": int(os.environ.get("FOREN_ORACLE_CLAUDE_THINKING") or 8000),
                                 "timeout_s": float(os.environ.get("FOREN_ORACLE_SLOT_TIMEOUT") or 900)}
            self._claude_workdir = None
        else:
            config_path = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
            config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            self.model_config = {key: config.get(key) for key in ("model", "model_provider", "model_reasoning_effort", "profile")}
            self.model_override = os.environ.get("FOREN_CODEX_MODEL") or None
            if self.model_override:
                self.model_config = {**self.model_config, "model": self.model_override, "model_override": True}
        self.model_config = {**self.model_config, "policy": "oracle_walker", "slot_backend": self.backend,
                             "slot_effort": self.effort, "slot_mock": self.mock}
        self.memories: dict[str, dict[str, Any]] = {}
        self.slot_calls = 0
        # 10-14：某个槽失败（端点 / Codex 重试用尽）后，这条 run 不能再继续：wk10 里 S1 失败 → 工作区 runtime_error →
        # GIA 照 G9_no_reports 直接 FINISH，写出"没找到"的答案，run 还被标成 done 进了导出。记下第一次失败，之后每次
        # generate_json 都直接抛，run_cases 会把 run 标成 incomplete（下次重跑）。
        self.slot_failed: str | None = None
        self.gt_points = _gt_points(os.environ.get("FOREN_ORACLE_CASE"))
        self.arms = os.environ.get("FOREN_ORACLE_ARMS", "").lower() in ("1", "true", "yes", "on")
        self.gt_review = _env_flag("FOREN_ORACLE_GT_REVIEW", False)
        self._captions: list[str] = []
        self._caption_intro = ""
        self.rationale_slot = _env_flag("FOREN_ORACLE_RATIONALE", True) and not self.mock
        self.bon = max(1, int(os.environ.get("FOREN_ORACLE_BON") or 3))
        # 10-16 r0+：循环内充分性检查
        self.inloop = _env_flag("FOREN_INLOOP_SUFFICIENCY", False) and not self.mock
        self.plan_records: dict[int, dict[str, Any]] = {}
        self._inloop_judge_obj: Any = None
        self._inloop_case_obj: Any = None

    async def generate_json(self, *, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        async with self.lock:
            self.sequence += 1
            return await asyncio.to_thread(self._generate, system_prompt, payload, self.sequence)

    # ------------------------------------------------------------------ 主流程
    def _generate(self, system_prompt: str, payload: dict[str, Any], sequence: int) -> dict[str, Any]:
        if self.slot_failed:
            raise RuntimeError(f"an earlier language slot failed, this run cannot continue: {self.slot_failed}")
        started = time.monotonic()
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
        # 10-14 晚：每张附图前放它的说明（E 编号、来源、时刻），与学生看到的附图消息同一种排法（导出 build_messages）。
        # 以前图只按顺序附在长提示末尾，教师得自己把第 k 张图和 image_notes 里第 k 条对上；videosham_1028 的报告因此把
        # A / B 的两张裁剪图说反（E25 / E26），被核验判 wrong_citation
        self._captions = image_notes[1:] if len(image_notes) == len(image_paths) + 1 else []
        self._caption_intro = image_notes[0] if image_notes else ""
        request = {"system_prompt": system_prompt + ("\n" + self.instruction if self.instruction else ""),
                   "payload": copy.deepcopy(payload), "image_notes": image_notes, "image_evidence_ids": image_ids}
        rendered, render_info = render_for_model(payload)
        if render_info:
            request["rendered_payload"] = rendered
            request["state_render"] = render_info
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        base_prompt = _base_prompt(request)

        workspace = payload.get("workspace_id")
        actor = workspace or "GIA"
        memory = self.memories.setdefault(actor, {})
        if workspace:
            # r1（10-17）：检索类案例的族按第 1 轮工作区的任务定，之后各轮沿用（第 2 轮的任务文字是 S2 写的，族词可能不在里面）
            has_search = any(isinstance(a, dict) and a.get("name") == "SEARCH" for a in payload.get("available_actions") or [])
            if str(workspace).startswith("W1-") and getattr(self, "run_family", None) is None:
                self.run_family = _walker.task_family(str(view.get("task") or ""), has_search)
            elif getattr(self, "run_family", None):
                memory.setdefault("run_family", self.run_family)
        decision = (decide_hi(payload, memory, rendered if render_info else None) if workspace
                    else decide_gia(payload, memory))
        slots: list[dict[str, Any]] = []
        if decision.kind == "action":
            output = decision.output
            if workspace and self.rationale_slot and isinstance(output, dict) and output.get("action"):
                output = self._slot_rationale(directory, actor_dir, base_prompt, image_paths, output, rendered if render_info else payload,
                                              slots)
        elif decision.kind == "report":
            output = self._slot_report(directory, actor_dir, base_prompt, image_paths, decision, payload, slots)
            output = self._after_report(directory, decision, payload, rendered if render_info else None, output)
        elif decision.kind == "finish":
            output = self._slot_finish(directory, actor_dir, base_prompt, image_paths, decision, payload, slots)
        elif decision.kind == "investigate_slot":
            output = self._slot_round(directory, actor_dir, base_prompt, image_paths, decision, payload, slots)
        elif decision.kind in ("query_slot", "next_slot"):
            # D2（10-13）：S1 写查询词 / S3 选下一步；结果由 walker 校验成真正的一步，选"交报告"时再走 S4
            decision = self._slot_choose(directory, actor_dir, base_prompt, image_paths, decision, payload, slots)
            if decision.kind == "report":
                output = self._slot_report(directory, actor_dir, base_prompt, image_paths, decision, payload, slots)
                output = self._after_report(directory, decision, payload, rendered if render_info else None, output)
            else:
                output = decision.output
                if workspace and self.rationale_slot and decision.slot.get("substituted"):
                    # 10-15 对话 5''：S3 选的编号不合法、walker 换成了别的一步——理由按实际动作由 S0 重写
                    output = self._slot_rationale(directory, actor_dir, base_prompt, image_paths, output,
                                                  rendered if render_info else payload, slots)
        else:
            raise RuntimeError(f"unknown walker decision kind {decision.kind}")
        if workspace and isinstance(output, dict) and output.get("action"):
            output = self._rationale_target_guard(directory, actor_dir, base_prompt, image_paths, output,
                                                  rendered if render_info else payload, slots)

        if not workspace and isinstance(output, dict) and output.get("decision") == "INVESTIGATE":
            # r0+：循环内充分性检查给第 1 轮工作区定范围时，按核验器 plan 单元同一道评判题（plan_judged.plan_payload）
            rc_view = (payload.get("global_view") or {}).get("runtime_constraints") or {}
            self.plan_records[int(rc_view.get("round_index") or 1)] = {"payload": copy.deepcopy(payload),
                                                                         "output": copy.deepcopy(output)}
        raw = json.dumps(output, ensure_ascii=False)
        (directory / "response.json").write_text(raw, encoding="utf-8")
        usage = {"input_tokens": sum(int((s.get("usage") or {}).get("input_tokens") or 0) for s in slots),
                 "output_tokens": sum(int((s.get("usage") or {}).get("output_tokens") or 0) for s in slots)}
        meta = {"returncode": 0, "transport": "oracle_walker", "elapsed_s": time.monotonic() - started,
                "usage": usage, "images": len(image_paths), "external_tool_items": [],
                "oracle": {k: v for k, v in decision.to_dict().items() if k != "output"},
                "slots": slots, "llm_calls": len(slots), "configured_model": self.model_config}
        (directory / "transport.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if image_ids:
            observed = view.setdefault("content_observed_evidence_ids", [])
            observed.extend(eid for eid in image_ids if eid not in observed)
            view["model_observed_image_ids"] = list(image_ids)
        if self.generation_observer:
            self.generation_observer({"workspace_id": workspace, "model": "oracle_walker", "provider": "oracle_walker",
                                      "raw_output": raw, "input_tokens": usage["input_tokens"],
                                      "output_tokens": usage["output_tokens"], "image_evidence_ids": image_ids})
        print(json.dumps({"request": sequence, "role": "HI" if workspace else "GIA", "rule": decision.rule,
                          "branch": decision.branch, "llm_calls": len(slots), "output": output}, ensure_ascii=False)[:1500], flush=True)
        return output

    # ------------------------------------------------------------------ 教师槽
    def _codex(self, directory: Path, actor_dir: Path, name: str, prompt: str, image_paths: list[Path],
               slots: list[dict[str, Any]], mock: dict[str, Any]) -> str:
        """调一次教师槽，返回原始文本；记一条 slot 记录。名字沿用（各槽都这么调），后端按 self.backend 分发：
        mock 用模板；endpoint 发 OpenAI 兼容 /chat/completions（服务器上自己部署的教师）；codex 跑 Codex CLI。
        失败时记下 self.slot_failed 再抛，这条 run 之后的每一步都会直接抛（见 _generate）。"""

        (directory / f"slot_{name}_prompt_tail.txt").write_text(prompt[-6000:], encoding="utf-8")
        if self.mock:
            raw = json.dumps(mock, ensure_ascii=False)
            (directory / f"slot_{name}_output.txt").write_text(raw, encoding="utf-8")
            slots.append({"name": name, "elapsed_s": 0.0, "usage": {}, "mock": True, "images": len(image_paths)})
            self.slot_calls += 1
            return raw
        try:
            if self.backend == "slots":
                cfg = self.backend_for(name)
                if cfg.get("type") == "codex":
                    return self._codex_slot(directory, actor_dir, name, prompt, image_paths, slots, cfg)
                return self._endpoint(directory, name, prompt, image_paths, slots, cfg)
            if self.backend == "endpoint":
                return self._endpoint(directory, name, prompt, image_paths, slots, self._backends["endpoint"])
            if self.backend == "claude":
                return self._claude(directory, name, prompt, image_paths, slots)
            return self._codex_cli(directory, actor_dir, name, prompt, image_paths, slots)
        except Exception as exc:  # noqa: BLE001 - 任何槽失败都让 run 停下
            self.slot_failed = f"{name} @ {directory}: {str(exc)[:300]}"
            raise

    @property
    def _json_mode_unsupported(self) -> bool:
        """兼容 10-14 的属性名（scripts/_test_walker_endpoint_slot.py 读它）：有后端拒收过 json_object 就是 True。"""

        return any(bool(cfg.get("_json_mode_unsupported")) for cfg in self._backends.values())

    def backend_for(self, name: str) -> dict[str, Any]:
        """槽名 → 冻结文件里映射的后端（S4_report_retry → S4 → strong）。不认识的槽名直接报错，不悄悄落到某个后端。"""

        assert self.slot_table is not None
        key = slot_key(name)
        if key not in self.slot_table["slots"]:
            raise RuntimeError(f"slot {name!r} ({key}) is not in the frozen slot table {sorted(self.slot_table['slots'])}")
        return self._backends[self.slot_table["slots"][key]]

    def _endpoint(self, directory: Path, name: str, prompt: str, image_paths: list[Path], slots: list[dict[str, Any]],
                  cfg: dict[str, Any]) -> str:
        """端点教师（10-14 起；10-15 起按槽分后端）：一条 user 消息 = 槽提示全文 + 本步附图（与 Codex 的 stdin + --image 等价）。
        429 / 5xx / 连接错误按指数退避重试（attempts 次）；服务不认 response_format 时去掉再发；输出为空（思考吃完
        max_tokens）也算失败。落盘与 Codex 后端相同：slot_<name>_output.txt 是原始文本，slot_<name>_attempt_NN.json
        是每次尝试的状态。cfg 是后端配置（单端点老路径只有一个；按槽分时由 backend_for 选）。"""

        import httpx

        if cfg.get("_client") is None:
            key = cfg.get("api_key") or os.environ.get("FOREN_TEACHER_API_KEY")
            headers: dict[str, str] = {}
            if not key and cfg.get("openrouter"):
                from foren_searcher.openrouter_settings import load_api_key

                key = load_api_key()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            if cfg.get("openrouter"):
                headers.update({"HTTP-Referer": "https://github.com/foren-searcher", "X-Title": "ForenSearcher walker teacher"})
            # 本机 / 内网 vLLM 不走代理；OpenRouter 等公网 API 走系统代理（国内机器要经代理才出得去）
            cfg["_client"] = httpx.Client(timeout=httpx.Timeout(float(cfg["timeout_s"]), connect=20.0),
                                          trust_env=bool(cfg.get("trust_env")), headers=headers)
        client = cfg["_client"]
        started = time.monotonic()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        captions = self._captions if len(self._captions) == len(image_paths) else []
        if image_paths and self._caption_intro:
            content.append({"type": "text", "text": self._caption_intro})
        for i, path in enumerate(image_paths):
            if captions:
                content.append({"type": "text", "text": f"Image {i + 1}: {captions[i]}"})
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        body = endpoint_body(cfg, name, content)
        url = str(cfg["endpoint"]).rstrip("/") + "/chat/completions"
        attempts: list[dict[str, Any]] = []
        data: dict[str, Any] | None = None
        error = ""
        for attempt in range(1, int(cfg["attempts"]) + 1):
            try:
                response = client.post(url, json=body)
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "error": error})
            else:
                attempts.append({"attempt": attempt, "status": response.status_code})
                if (response.status_code == 400 and "response_format" in body
                        and re.search(r"response_format|json_object|\bformat\b", response.text, re.IGNORECASE)):
                    cfg["_json_mode_unsupported"] = True
                    body.pop("response_format", None)
                    attempts[-1]["json_mode_rejected"] = True
                    response = client.post(url, json=body)
                    attempts.append({"attempt": attempt, "status": response.status_code, "json_mode": False})
                if response.status_code < 400:
                    try:
                        data = response.json()
                    except ValueError:
                        error = "non-JSON response body"
                        data = None
                    else:
                        message = ((data.get("choices") or [{}])[0].get("message") or {})
                        if str(message.get("content") or "").strip():
                            break
                        error = "empty content（可能思考用完了 max_tokens）"
                        attempts[-1]["finish_reason"] = (data.get("choices") or [{}])[0].get("finish_reason")
                        data = None
                else:
                    error = f"HTTP {response.status_code}: {response.text[:500]}"
                    # 10-15：OpenRouter 按 max_tokens 预占余额，并发时回 402 "...given your current in-flight requests. Retry
                    # after in-flight requests complete"——等一会儿就能过，按限流退避；真余额不足（其他 402）照旧直接失败
                    inflight = response.status_code == 402 and "in-flight" in response.text
                    if response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504} and not inflight:
                        break
            (directory / f"slot_{name}_attempt_{attempt:02d}.json").write_text(
                json.dumps(attempts[-1], ensure_ascii=False), encoding="utf-8")
            if attempt < int(cfg["attempts"]):
                time.sleep(min(90.0, 5.0 * 2 ** (attempt - 1)))
        message = ((data or {}).get("choices") or [{}])[0].get("message") or {}
        raw = str(message.get("content") or "")
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage_in = (data or {}).get("usage") or {}
        record = {"name": name, "elapsed_s": round(time.monotonic() - started, 1),
                  "usage": {"input_tokens": usage_in.get("prompt_tokens"), "output_tokens": usage_in.get("completion_tokens")},
                  "attempts": attempts, "images": len(image_paths),
                  "effort": cfg.get("reasoning_effort") or self.effort, "backend": "endpoint",
                  "slot": slot_key(name), "slot_backend": cfg.get("name"), "backend_role": cfg.get("role"),
                  "model": cfg["model"], "model_served": (data or {}).get("model"), "endpoint_host": _host(cfg["endpoint"]),
                  "provider": (data or {}).get("provider"), "cost_usd": usage_in.get("cost"),
                  "reasoning_tokens": (usage_in.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                  "external_tool_items": []}
        slots.append(record)
        self.slot_calls += 1
        if reasoning:
            (directory / f"slot_{name}_reasoning.txt").write_text(str(reasoning), encoding="utf-8")
        if not raw.strip():
            (directory / "transport.json").write_text(json.dumps({"returncode": 1, "transport": "oracle_walker",
                                                                  "slots": slots, "error": error}, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
            raise RuntimeError(f"oracle slot {name} failed: {directory}；teacher endpoint error: {error[:300]}")
        (directory / f"slot_{name}_output.txt").write_text(raw, encoding="utf-8")
        return raw

    def _codex_slot(self, directory: Path, actor_dir: Path, name: str, prompt: str, image_paths: list[Path],
                    slots: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
        """分槽表里 type=codex 的后端（10-15 r0 成本压缩：S4 / S5 走 ChatGPT 订阅的 Codex CLI）。
        与 _codex_cli（老的整条 run 用 Codex 的路径）相同的只读、无工具调用；不同处：模型 / effort 取后端配置，图注写进提示，
        跨进程串行，429 / usage limit 长退避，槽记录带 slot / backend / model / model_served。"""

        codex = find_codex(cfg)
        started = time.monotonic()
        actor_dir.mkdir(parents=True, exist_ok=True)
        result_path = directory / f"slot_{name}_output.txt"
        effort = cfg.get("reasoning_effort") or self.effort
        captions = self._captions if len(self._captions) == len(image_paths) else []
        tail = ""
        if image_paths and (self._caption_intro or captions):
            tail = "\n\n" + (self._caption_intro + "\n" if self._caption_intro else "")
            tail += "\n".join(f"Image {i + 1}: {c}" for i, c in enumerate(captions))
        command = [codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--json",
                   "-C", str(actor_dir), "-o", str(result_path), "-m", str(cfg["model"])]
        if effort:
            command.extend(["-c", f'model_reasoning_effort="{effort}"'])
        for path in image_paths:
            command.extend(["--image", str(path)])
        command.extend(["--", "-"])
        budget = float(cfg.get("retry_max_s") or os.environ.get("FOREN_ORACLE_RETRY_MAX_S") or 900)
        timeout = float(cfg.get("timeout_s") or 900)
        attempts: list[dict[str, Any]] = []
        waited = 0.0
        completed = None
        lock_wait = 0.0
        transient_left = 1
        for attempt in range(1, 12):
            if result_path.exists():
                result_path.unlink()
            with codex_lock(serial=cfg.get("serial", True)) as wait_s:
                lock_wait += wait_s
                try:
                    completed = subprocess.run(command, input=prompt + tail, text=True, encoding="utf-8", errors="replace",
                                               capture_output=True, timeout=timeout, check=False)
                except subprocess.TimeoutExpired:
                    completed = subprocess.CompletedProcess(command, 124, stdout="", stderr=f"codex exec timed out ({timeout:.0f}s)")
            (directory / f"slot_{name}_attempt_{attempt:02d}_events.jsonl").write_text(completed.stdout or "", encoding="utf-8")
            ok = completed.returncode == 0 and result_path.is_file() and result_path.read_text(encoding="utf-8").strip() != ""
            limited = codex_rate_limited((completed.stdout or "") + " " + (completed.stderr or ""))
            attempts.append({"attempt": attempt, "returncode": completed.returncode, "ok": ok, "rate_limited": limited,
                             "lock_wait_s": round(wait_s, 1)})
            if ok:
                break
            if limited:
                delay = CODEX_BACKOFF[min(len([a for a in attempts if a.get("rate_limited")]) - 1, len(CODEX_BACKOFF) - 1)]
                delay = min(delay, budget - waited)          # 最后一次退避截到总预算（30+60+120+240+450 = 900 s）
                if delay < 1.0:
                    break
            elif transient_left > 0:
                transient_left -= 1
                delay = 15.0
            else:
                break
            attempts[-1]["sleep_s"] = delay
            time.sleep(delay)
            waited += delay
        assert completed is not None
        events = []
        for line in (completed.stdout or "").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        (directory / f"slot_{name}_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        tool_items = [e.get("item", {}) for e in events
                      if e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}]
        usage = next((e.get("usage", {}) for e in reversed(events) if e.get("type") == "turn.completed"), {}) or {}
        record = {"name": name, "elapsed_s": round(time.monotonic() - started, 1),
                  "usage": {"input_tokens": usage.get("input_tokens"), "cached_input_tokens": usage.get("cached_input_tokens"),
                            "output_tokens": usage.get("output_tokens")},
                  "attempts": attempts, "images": len(image_paths), "effort": effort, "backend": "codex",
                  "slot": slot_key(name), "slot_backend": cfg.get("name"), "backend_role": cfg.get("role"),
                  "model": cfg["model"], "model_served": codex_model_served(events, completed.stderr or ""),
                  "model_source": "codex exec -m", "endpoint_host": "codex-cli", "codex_version": codex_version(codex),
                  "lock_wait_s": round(lock_wait, 1), "cost_usd": 0.0, "billing": "chatgpt-subscription",
                  "external_tool_items": tool_items}
        slots.append(record)
        self.slot_calls += 1
        if completed.returncode or tool_items or not result_path.is_file() or not result_path.read_text(encoding="utf-8").strip():
            reason = next((str((e.get("error") or {}).get("message") or e.get("message") or "")
                           for e in reversed(events) if e.get("type") in {"turn.failed", "error"}), "") or (completed.stderr or "")[-300:]
            if attempts and attempts[-1].get("rate_limited"):
                reason = f"Codex usage limit / rate limit (429) after {waited:.0f}s of backoff: {reason}"
            (directory / "transport.json").write_text(json.dumps({"returncode": 1, "transport": "oracle_walker",
                                                                  "slots": slots}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"oracle slot {name} failed: {directory}" + (f"；Codex 报错：{reason[:400]}" if reason else ""))
        return result_path.read_text(encoding="utf-8")

    def _claude(self, directory: Path, name: str, prompt: str, image_paths: list[Path], slots: list[dict[str, Any]]) -> str:
        """Claude Code CLI 教师（10-14）：claude -p，--tools "" 不给工具、--setting-sources "" 不读本机设置 / CLAUDE.md、
        --strict-mcp-config 不连 MCP、空临时目录里跑；一条 user 消息 = 槽提示全文 + 本步附图（base64，>3.5MB 的转 JPEG，
        API 单图上限 5MB）。普通失败退避重试 3 次；订阅额度 / 限流直接抛（消息里带 usage limit，walker_run 认得出来停批）。"""

        import tempfile

        from foren_searcher.datagen.verify.judges import ClaudeCliLimit, _CLI_LIMIT_WORDS

        cfg = self.model_config
        if self._claude_workdir is None:
            self._claude_workdir = tempfile.mkdtemp(prefix="foren_claude_slot_")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        captions = self._captions if len(self._captions) == len(image_paths) else []
        if image_paths and self._caption_intro:
            content.append({"type": "text", "text": self._caption_intro})
        for i, path in enumerate(image_paths):
            if captions:
                content.append({"type": "text", "text": f"Image {i + 1}: {captions[i]}"})
            mime, data = _image_for_claude(path)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        message = json.dumps({"type": "user", "message": {"role": "user", "content": content}}, ensure_ascii=False) + "\n"
        system = ("You fill one language slot of an investigation policy. Use only the state and the images in the user "
                  "message; you have no tools. Follow the slot instructions there and return exactly one JSON object.")
        command = [*self.claude_cli, "-p", "--model", cfg["model"], "--tools", "", "--setting-sources", "",
                   "--strict-mcp-config", "--no-session-persistence", "--input-format", "stream-json",
                   "--output-format", "stream-json", "--verbose", "--system-prompt", system]
        env = dict(os.environ)
        if cfg["thinking_tokens"] > 0:
            env["MAX_THINKING_TOKENS"] = str(cfg["thinking_tokens"])
        else:
            env.pop("MAX_THINKING_TOKENS", None)
        started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        error = ""
        for attempt in range(1, 4):
            try:
                done = subprocess.run(command, input=message, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=cfg["timeout_s"], cwd=self._claude_workdir, env=env)
            except subprocess.TimeoutExpired:
                error = f"claude -p timed out ({cfg['timeout_s']:.0f}s)"
                attempts.append({"attempt": attempt, "error": error})
                continue
            result = None
            for line in done.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if item.get("type") == "result":
                        result = item
            (directory / f"slot_{name}_attempt_{attempt:02d}_events.jsonl").write_text(done.stdout[-200000:], encoding="utf-8")
            ok = bool(result and not result.get("is_error") and result.get("subtype") == "success"
                      and str(result.get("result") or "").strip())
            attempts.append({"attempt": attempt, "returncode": done.returncode, "ok": ok})
            if ok:
                break
            blob = (done.stderr or "") + " " + (json.dumps(result, ensure_ascii=False) if result else done.stdout[-2000:])
            error = blob.strip()[:500]
            if (result or {}).get("api_error_status") == 429 or any(w in blob.lower() for w in _CLI_LIMIT_WORDS):
                slots.append({"name": name, "elapsed_s": round(time.monotonic() - started, 1), "attempts": attempts,
                              "backend": "claude", "model": cfg["model"], "images": len(image_paths)})
                raise RuntimeError(f"oracle slot {name}: Claude usage limit / rate limit: {error[:300]}")
            if attempt < 3:
                time.sleep(20 * attempt)
            result = None
        usage = (result or {}).get("usage") or {}
        model_usage = (result or {}).get("modelUsage") or {}
        record = {"name": name, "elapsed_s": round(time.monotonic() - started, 1),
                  "usage": {"input_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0),
                            "output_tokens": usage.get("output_tokens")},
                  "cost_usd": (result or {}).get("total_cost_usd"), "attempts": attempts, "images": len(image_paths),
                  "backend": "claude", "model": cfg["model"],
                  "model_resolved": max(model_usage, key=lambda k: (model_usage[k] or {}).get("outputTokens", 0)) if model_usage else None,
                  "external_tool_items": []}
        slots.append(record)
        self.slot_calls += 1
        if result is None:
            (directory / "transport.json").write_text(json.dumps({"returncode": 1, "transport": "oracle_walker",
                                                                  "slots": slots, "error": error}, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
            raise RuntimeError(f"oracle slot {name} failed: {directory}；claude -p error: {error[:300]}")
        raw = str(result.get("result") or "")
        (directory / f"slot_{name}_output.txt").write_text(raw, encoding="utf-8")
        return raw

    def _codex_cli(self, directory: Path, actor_dir: Path, name: str, prompt: str, image_paths: list[Path],
                   slots: list[dict[str, Any]]) -> str:
        """调一次 Codex（只读沙箱、无工具）。"""

        started = time.monotonic()
        result_path = directory / f"slot_{name}_output.txt"
        command = [self.codex, "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                   "--color", "never", "--json", "-C", str(actor_dir), "-o", str(result_path)]
        if self.model_override:
            command.extend(["-m", self.model_override])
        if self.effort:
            command.extend(["-c", f'model_reasoning_effort="{self.effort}"'])
        for path in image_paths:
            command.extend(["--image", str(path)])
        command.extend(["--", "-"])
        attempts = []
        completed = None
        # 10-14：429（限额）也重试，退避比"at capacity"长；wk10 三条都死在 429 上
        for attempt in range(4):
            completed = subprocess.run(command, input=prompt, text=True, encoding="utf-8", errors="replace",
                                       capture_output=True, timeout=900, check=False)
            attempts.append({"attempt": attempt + 1, "returncode": completed.returncode})
            (directory / f"slot_{name}_attempt_{attempt + 1:02d}_events.jsonl").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode == 0:
                break
            rate_limited = "429" in completed.stdout and "Too Many Requests" in completed.stdout
            if "Selected model is at capacity" not in completed.stdout and not rate_limited:
                break
            if attempt < 3:
                time.sleep(60 * (attempt + 1) if rate_limited else 15)
        assert completed is not None
        events = []
        for line in completed.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        (directory / f"slot_{name}_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        tool_items = [e.get("item", {}) for e in events
                      if e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}]
        usage = next((e.get("usage", {}) for e in reversed(events) if e.get("type") == "turn.completed"), {})
        record = {"name": name, "elapsed_s": round(time.monotonic() - started, 1), "usage": usage, "attempts": attempts,
                  "images": len(image_paths), "effort": self.effort, "external_tool_items": tool_items}
        slots.append(record)
        self.slot_calls += 1
        if completed.returncode or tool_items or not result_path.is_file():
            reason = next((str((e.get("error") or {}).get("message") or e.get("message") or "")
                           for e in reversed(events) if e.get("type") in {"turn.failed", "error"}), "")
            (directory / "transport.json").write_text(json.dumps({"returncode": 1, "transport": "oracle_walker",
                                                                  "slots": slots}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"oracle slot {name} failed: {directory}" + (f"；Codex 报错：{reason[:300]}" if reason else ""))
        return result_path.read_text(encoding="utf-8")

    def _rationale_target_guard(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path],
                                output: dict[str, Any], shown: dict[str, Any], slots: list[dict[str, Any]]) -> dict[str, Any]:
        """10-15 对话 5''（追加 1）：理由里的 E / R 编号不含动作的对象、却含别的编号 → S0 按实际动作重写一次；仍不符 → 写
        slot_rationale_target.json status=mismatch，export 把这一步放 held（rationale_target_mismatch）。"""

        text = json.dumps(shown, ensure_ascii=False)
        first = rationale_target_check(output, text)
        if first["status"] != "mismatch":
            return output
        log: dict[str, Any] = {"first": {**first, "rationale": output.get("rationale")}}
        retried = output
        if self.rationale_slot:
            feedback = (f"\nThe action's own record is {', '.join(first['targets'])}; a previous rationale talked about "
                        f"{', '.join(first['mentioned'])} instead. The rationale must say why THIS record is the next step, "
                        "naming it.")
            retried = self._slot_rationale(directory, actor_dir, base_prompt, image_paths, output, shown, slots,
                                           feedback=feedback, name="S0_rationale_retry", strict_target=True)
        second = rationale_target_check(retried, text)
        log["retry"] = {**second, "rationale": retried.get("rationale")}
        log["status"] = "mismatch" if second["status"] == "mismatch" else "fixed"
        (directory / "slot_rationale_target.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
        return retried

    def _slot_rationale(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path], output: dict[str, Any],
                        shown: dict[str, Any], slots: list[dict[str, Any]], *, feedback: str = "", name: str = "S0_rationale",
                        strict_target: bool = False) -> dict[str, Any]:
        """S0（10-14 晚）：程序定的 HI 动作，rationale 由教师按这一步学生看到的状态写（模板 rationale 在 wk8/wk9 里 328 行只有
        201 种，同一句出现 35 次）。动作本身不变；教师写的不合格（空、过长、引了状态里没有的编号）就保留模板。"""

        action = {k: v for k, v in output.items() if k != "rationale"}
        suffix = (
            "\n\nHOST DECISION FOR THIS TURN (language slot S0): the next action has already been chosen:\n"
            + json.dumps(action, ensure_ascii=False) +
            "\nWrite only its \"rationale\": one or two sentences (at most 50 words) saying why this is the right next step given "
            "what the state shows now — what is already established, what is still unknown, and what this action will check or "
            "obtain. Refer to the concrete records (E / R IDs, times, names) that appear in the state; describe image content only "
            "where it is the reason for the step, and only what the attached images plainly show. Do not mention the host, rules or "
            "a plan, and do not state results this action has not produced yet. Write in English. "
            'Return only {"rationale": "..."}.') + feedback
        mock = {"rationale": str(output.get("rationale") or "")}
        try:
            raw = self._codex(directory, actor_dir, name, base_prompt + suffix, image_paths, slots, mock)
        except Exception:
            if self.slot_failed:
                raise
            return output
        text = str(extract_json_object(raw).get("rationale") or "").strip()
        visible = set(_EREF.findall(json.dumps(shown, ensure_ascii=False))) | set(re.findall(r"\bR[1-9][0-9]*\b", json.dumps(shown, ensure_ascii=False)))
        cited = set(_EREF.findall(text)) | set(re.findall(r"\bR[1-9][0-9]*\b", text))
        bad = sorted(cited - visible)
        ok = 15 <= len(text) <= 480 and not bad
        if ok and strict_target and rationale_target_check({**output, "rationale": text}, json.dumps(shown, ensure_ascii=False))["status"] == "mismatch":
            ok = False
        record = {"template": output.get("rationale"), "teacher": text, "used": ok, "unknown_ids": bad}
        if name != "S0_rationale":
            record["slot"] = name
        (directory / "slot_S0_rationale.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return {**output, "rationale": text} if ok else output

    def _slot_report(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path], decision: Decision,
                     payload: dict[str, Any], slots: list[dict[str, Any]]) -> dict[str, Any]:
        slot = decision.slot
        status, reason = slot["status"], slot.get("status_reason")
        allowed = list(slot.get("allowed_evidence_ids") or [])
        task_id = decision.output["task_id"]
        parts = list(slot.get("task_parts") or [])
        # 10-14：检索类任务（E 图片出处 / L 拍摄地 / C 核查声明）不用两段视频比对的那组要点（时间、边界、A/B 帧）
        search_task = slot.get("family") in ("E", "L", "C")
        rules = (SEARCH_RULES if search_task else MEDIA_RULES) % (", ".join(allowed) or "(none)", CONTRACT_RULES)
        suffix = (
            "\n\nHOST DECISION FOR THIS TURN (language slot S4): the host has already decided that this turn is the workspace "
            "REPORT, not a tool action. Write the report JSON exactly as the HI instructions define it: "
            '{"task_id": "%s", "status": "%s", %s"findings": [{"statement": "...", "evidence_ids": ["..."], "limitations": ["..."]}], '
            '"selected_evidence_ids": ["..."], "rationale": "..."}.\n'
            "- status: %s.%s Never use COMPLETED when the host set INCOMPLETE. When the host set COMPLETED, change it to INCOMPLETE "
            "only if one of the numbered task parts below has no obtained evidence that answers it: mark that part not_answered in "
            "part_coverage and write status_reason as one piece per such part, naming what the part asks and what is missing.\n"
            "- Missing corroboration is not a missing part and never a reason for INCOMPLETE: the lack of a second or independent "
            "source, of a first-hand, primary, original or official source, or of a visual / architectural match to the claimed "
            "place, event or video goes into the limitations of the finding that answers the part. A part asked 'as far as the "
            "evidence allows' is answered by what the evidence does establish (e.g. the earliest dated copy found, as 'no later "
            "than'), with what stays open in limitations; it is not_answered only when no obtained evidence bears on it. A part "
            "that only repeats the claim's wording or its date ('(posted: …)', '(claim date: …)') is answered by the finding that "
            "checks the claim.\n"
            "%s"
            "%s"
            "- The task has these parts; answer EACH one explicitly in its own finding (statement + cited E), not only in limitations:\n%s"
            "  Add to the JSON a key \"part_coverage\": a list with one entry per part, {\"part\": <number>, \"finding\": <1-based finding "
            "index>} when a finding answers it, or {\"part\": <number>, \"not_answered\": \"<what is missing>\"} when no obtained "
            "evidence answers it (missing corroboration does not count). Do not mark a part answered if the finding only says it is "
            "unknown.\n"
            "Host notes recorded from the tool outputs (facts already in the state, for precision):\n%s\n"
            "Return only the JSON object."
        ) % (task_id, status, '"status_reason": "...", ' if status == "INCOMPLETE" else "", status,
             (f" Use status_reason (you may rephrase, same meaning): {reason}" if reason else ""),
             rules,
             _referent_rule(payload),
             "".join(f"  {i}. {part}\n" for i, part in enumerate(parts, 1)) or "  1. (the task as a whole)\n",
             "\n".join(f"- {n}" for n in slot.get("notes") or []) or "- (none)")
        mock = {"task_id": task_id, "status": status, "findings": [{"statement": "[mock] report placeholder",
                "evidence_ids": allowed[-2:] or allowed, "limitations": list(slot.get("notes") or [])[:2]}],
                "selected_evidence_ids": allowed[-2:], "rationale": "[mock]"}
        if reason:
            mock["status_reason"] = reason
        mock_gap = os.environ.get("FOREN_ORACLE_MOCK_GAP")      # 只用于冒烟：模拟教师把第 1 轮报告降为带时间窗的 INCOMPLETE
        if self.mock and mock_gap and task_id.startswith("W1-"):
            mock.update({"status": "INCOMPLETE", "status_reason": mock_gap})
        raw = self._codex(directory, actor_dir, "S4_report", base_prompt + suffix, image_paths, slots, mock)
        report = extract_json_object(raw)
        status = _final_status(status, report, parts)
        status, reason = _coverage_status(status, reason, report, parts)
        report, dropped = _clean_report(report, task_id=task_id, status=status, reason=reason, allowed=set(allowed))
        if not report["findings"]:
            retry = suffix + ("\nYour previous output cited no allowed E ID in any finding (dropped: %s). Every finding must cite at "
                              "least one of the allowed IDs." % ", ".join(dropped))
            raw = self._codex(directory, actor_dir, "S4_report_retry", base_prompt + retry, image_paths, slots, mock)
            retry_report = extract_json_object(raw)
            status = _final_status(slot["status"], retry_report, parts)
            status, reason = _coverage_status(status, reason, retry_report, parts)
            report, dropped = _clean_report(retry_report, task_id=task_id, status=status, reason=reason, allowed=set(allowed))
        # 10-15 对话 5''（追加 3）：程序校验 CONTRACT_RULES，有问题就把问题反馈给教师重写一次；重写后问题更少（且有 Finding）才换
        kinds = slot.get("evidence_kinds") or {}
        required = list(slot.get("required_findings") or [])     # r1：E 比对 / L 参考图 / C 帧与整段转写，必须写成 Finding
        problems = s4_contract_problems(report, kinds, required) if (kinds or required) else []
        if problems:
            contract: dict[str, Any] = {"first": problems}
            retry = suffix + ("\nYour previous report has these problems; fix them and return the complete JSON object again:\n"
                              + "".join(f"- {p}\n" for p in problems[:8]))
            raw = self._codex(directory, actor_dir, "S4_report_contract", base_prompt + retry, image_paths, slots, mock)
            try:
                again = extract_json_object(raw)
                status2 = _final_status(slot["status"], again, parts)
                status2, reason2 = _coverage_status(status2, slot.get("status_reason"), again, parts)
                report2, dropped2 = _clean_report(again, task_id=task_id, status=status2, reason=reason2, allowed=set(allowed))
            except Exception as exc:  # noqa: BLE001
                report2, dropped2 = None, []
                contract["retry_error"] = f"{type(exc).__name__}: {exc}"[:300]
            if report2 and report2["findings"]:
                left = s4_contract_problems(report2, kinds, required)
                contract["retry"] = left
                if len(left) < len(problems):
                    report, dropped = report2, dropped2
                    contract["used"] = "retry"
            contract.setdefault("used", "first")
            (directory / "slot_S4_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=1), encoding="utf-8")
        # 10-15 对话 5''（追加 2）：walker 写进 status_reason 的可修缺口（未读命中页的网址、没并排看的参考图）教师可能改写掉，
        # GIA 要靠原文认它派第 2 轮——报告仍是 INCOMPLETE 时原样补回
        if report["status"] == "INCOMPLETE":
            for piece in (x.strip() for x in str(slot.get("status_reason") or "").split(";")):
                if _OPEN_GAP.search(piece) and piece not in str(report.get("status_reason") or ""):
                    report["status_reason"] = (str(report.get("status_reason") or "").rstrip("; ") + "; " + piece).strip("; ")
        # 10-15 对话 5''（追加 4）：候选页面 failed ≥ 1/3 时 walker 给的模板句，加在最后一条 Finding 的 limitations 末尾
        for text in slot.get("auto_limitations") or []:
            if report["findings"]:
                _add_limitation(report, text, len(report["findings"]) - 1)
        if dropped:
            (directory / "slot_S4_dropped_ids.json").write_text(json.dumps(dropped), encoding="utf-8")
        return report

    # ------------------------------------------------------------------ 10-16 r0+：循环内充分性检查
    @property
    def run_dir(self) -> Path:
        return self.output_dir.parent

    def _inloop_judge(self) -> Any:
        if self._inloop_judge_obj is None:
            from foren_searcher.datagen.verify.judges import CachedJudge, ClaudeCliJudge

            # 与 walker_run 的核验（complete_verify_local --judge claude-cli，模型别名 sonnet）同一个判官；缓存先写 run 目录，
            # 核验前由 walker_run 并进主缓存（--jobs 2 两个进程同时追加一个大文件不安全）
            self._inloop_judge_obj = CachedJudge(ClaudeCliJudge(model=os.environ.get("FOREN_INLOOP_JUDGE_MODEL") or "sonnet"),
                                                 self.run_dir / INLOOP_CACHE)
        return self._inloop_judge_obj

    def _inloop_case(self) -> tuple[Any, dict[str, Any]]:
        if self._inloop_case_obj is None:
            from foren_searcher.datagen.cases import Case

            path = os.environ.get("FOREN_ORACLE_CASE")
            if not path or not Path(path).is_file():
                self._inloop_case_obj = (None, {})
            else:
                self._inloop_case_obj = (Case.load(Path(path)), json.loads(Path(path).read_text(encoding="utf-8")))
        return self._inloop_case_obj

    def _inloop_scope(self, workspace: str, case: Any, task_text: str, judge: Any) -> tuple[set[str] | None, dict[str, Any]]:
        """核验器的范围：plan 单元评判的 judge_targets（第 1 轮可复现：covered_before 为空、之前没有报告），否则按任务文字兜底
        （scope.requirements_for_workspace 不给 units 时的做法）。第 2 轮的 covered_before 要第 1 轮报告的锚定核验结果，循环里没有，
        所以第 2 轮只能兜底，跑完后的全量核验可能换范围（缓存不命中），统计里单列。"""

        import asyncio
        from types import SimpleNamespace

        from foren_searcher.datagen.verify.plan_judged import judge_plan
        from foren_searcher.datagen.verify.scope import requirements_for_workspace

        m = re.match(r"^W(\d+)-(\d+)$", workspace)
        record = self.plan_records.get(int(m.group(1))) if m else None
        if case is not None and case.requirements and m and record:
            round_n = int(m.group(1))
            gview = record["payload"].get("global_view") or {}
            plan = SimpleNamespace(output=record["output"], view=gview,
                                   round_index=(gview.get("runtime_constraints") or {}).get("round_index"))
            # r1（10-17 负责人定）：第 2 轮也按 plan 评判定范围——covered_before 与 existing_reports 按事后核验同一算法
            # （verify.units.verify_units → check_plan；complete_units 用 log 取本轮之前的报告全文），同一 payload、同一 key
            covered = self._inloop_covered_before(round_n, case, judge) if round_n > 1 else set()
            log = SimpleNamespace(store=self.store) if round_n > 1 else None
            raw = asyncio.run(judge_plan(plan, case, covered, judge, log))
            known = {r.id for r in case.requirements}
            for item in (raw or {}).get("tasks") or []:
                if isinstance(item, dict) and int(item.get("index") or 0) == int(m.group(2)):
                    targets = {str(t) for t in item.get("targets") or [] if str(t) in known or not known}
                    return targets, {"method": "judge_targets", "requirement_ids": sorted(targets)}
        req_ids, note = requirements_for_workspace(case, task_text, run=self.run_dir.name, workspace_id=workspace, units=None)
        return req_ids, {**note, "inloop": "text_overlap" + ("" if record else " (no plan record)")}

    def _inloop_covered_before(self, round_n: int, case: Any, judge: Any) -> set[str]:
        """事后核验里第 round_n 轮 plan 单元的 covered_before：之前各轮工作区里、没被污染的 Finding（三层核验 + 校准后的
        tainted_findings 以外）所引的案卷必需证据支持的要求（verify.hindsight.UsefulEvidence.coverage_by_workspace）。
        之前各轮的报告这里按事后核验同样的参数跑一遍 verify_report（同一判官、同一缓存：事后核验直接命中）。"""

        import asyncio

        from foren_searcher.datagen.verify.calibrate import recalibrate
        from foren_searcher.datagen.verify.context import StoreLookup
        from foren_searcher.datagen.verify.verdict import verify_report

        lookup = StoreLookup(self.store)
        covered: set[str] = set()
        cache = self.__dict__.setdefault("_covered_cache", {})
        for rec in self.store.workspace_report_records():
            mm = re.match(r"^W(\d+)-\d+$", str(rec.task_id))
            if not mm or int(mm.group(1)) >= round_n:
                continue
            if rec.task_id in cache:
                covered |= cache[rec.task_id]
                continue
            try:
                task_text = str(self.store.workspace_task_record(rec.task_id).get("task") or "")
            except KeyError:
                task_text = ""
            req_ids, _ = self._inloop_scope(rec.task_id, case, task_text, judge)
            visible = self._visible_ids(rec.task_id)
            verdict = asyncio.run(verify_report(rec.to_dict(), lookup, subject={"run": self.run_dir.name, "workspace_id": rec.task_id},
                                                case=case, task_text=task_text or None, judge=judge, visible_ids=visible,
                                                requirement_ids=req_ids))
            tainted = set(recalibrate(verdict).tainted_findings)
            reqs: set[str] = set()
            for finding in rec.to_dict().get("findings") or []:
                if str(finding.get("finding_id")) in tainted:
                    continue
                for eid in finding.get("evidence_ids") or []:
                    if not isinstance(eid, str) or not lookup.has(eid):
                        continue
                    text = lookup.get(eid)
                    acquired_by = getattr(self.store.evidence(eid), "acquired_by", None)
                    for req in case.required_evidence:
                        if req.matches(source_url=text.source_url, acquired_by=acquired_by, label=text.label):
                            reqs |= set(req.supports)
            cache[rec.task_id] = reqs
            covered |= reqs
        return covered

    def _visible_ids(self, workspace: str) -> set[str] | None:
        """reverify._visible_ids 的循环内版本：这个工作区各次请求的 model_view 里出现过的 E 编号并集。"""

        ids: set[str] = set()
        found = False
        for rd in sorted((self.output_dir).glob("request_*")):
            try:
                payload = json.loads((rd / "request.json").read_text(encoding="utf-8")).get("payload") or {}
            except (OSError, ValueError):
                continue
            if payload.get("workspace_id") != workspace:
                continue
            found = True
            ids |= set(re.findall(r'"(E\d+)"', json.dumps(payload.get("model_view") or {}, ensure_ascii=False)))   # 同 reverify._EVIDENCE_TOKEN
        return ids if found else None

    def _predicted_store_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """store.commit_workspace_report 会给的样子：F 编号 = finding_sequence 的下一个（独立序列表，校验失败也不复用）。"""

        try:
            with self.store._connect() as connection:          # noqa: SLF001 —— 只读一个序号
                row = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='finding_sequence'").fetchone()
            seq = int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            seq = 0
        findings = [{"finding_id": f"F{seq + i + 1}", "workspace_id": report.get("task_id"), "statement": f["statement"],
                     "evidence_ids": list(f.get("evidence_ids") or []), "limitations": list(f.get("limitations") or [])}
                    for i, f in enumerate(report.get("findings") or [])]
        return {"task_id": report.get("task_id"), "status": report.get("status"), "status_reason": report.get("status_reason"),
                "findings": findings}

    def _after_report(self, directory: Path, decision: Decision, payload: dict[str, Any], rendered: dict[str, Any] | None,
                      report: dict[str, Any]) -> dict[str, Any]:
        """S4 之后、交给 harness（之后 GIA 看到）之前：COMPLETED → 充分性层核一次；E / L / C 第 1 轮的 INCOMPLETE 报告没写跟进
        → 补 walker 的可修缺口。"""

        if not self.inloop or not isinstance(report, dict):
            return report
        import asyncio

        from foren_searcher.datagen.errors import ErrorType, Severity
        from foren_searcher.datagen.status_rewrite import build_reason
        from foren_searcher.datagen.verify.context import StoreLookup
        from foren_searcher.datagen.verify.judges import CachedJudge, ClaudeCliLimit
        from foren_searcher.datagen.verify.sufficiency import (SUFFICIENCY_SYSTEM_PROMPT, sufficiency_payload,
                                                               verify_sufficiency)

        workspace = str(report.get("task_id") or "")
        family = str(decision.slot.get("family") or "")
        hints = decision.slot.get("followup") or {}
        entry: dict[str, Any] = {"request": directory.name, "workspace": workspace, "family": family,
                                 "status_in": report.get("status"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        case, case_dict = self._inloop_case()
        if report.get("status") == "COMPLETED" and case is not None:
            started = time.monotonic()
            judge = self._inloop_judge()
            calls_before = (judge.misses, judge.hits)
            try:
                task_text = str(self.store.workspace_task_record(workspace).get("task") or "")
                req_ids, scope_note = self._inloop_scope(workspace, case, task_text, judge)
                stored = self._predicted_store_report(report)
                # r1（10-17）：与事后核验（reverify / verdict.verify_report）同一份 payload——verifier@0.4.0 的 sufficiency_payload，
                # 带 lookup（证据索引：所引 E 的标题 / 取得方式 / 网址）。没被改写的报告在事后全量核验时直接命中这条缓存
                lookup = StoreLookup(self.store)
                suff_payload = sufficiency_payload(stored, task_text, case=case, requirement_ids=req_ids, lookup=lookup)
                entry.update({"scope": scope_note, "key": CachedJudge.key_for(SUFFICIENCY_SYSTEM_PROMPT, suff_payload, []),
                              "predicted_finding_ids": [f["finding_id"] for f in stored["findings"]]})
                result = asyncio.run(verify_sufficiency(stored, task_text, judge, case=case, requirement_ids=req_ids,
                                                        lookup=lookup))
            except ClaudeCliLimit as exc:
                entry.update({"error": f"claude_cli_limit: {str(exc)[:200]}", "limit_hit": True})
                result = None
            except Exception as exc:  # noqa: BLE001 —— 检查失败不挡 run：报告照原样交，跑完后的全量核验 + 事后改写兜底
                entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                result = None
            entry.update({"judge_new_calls": judge.misses - calls_before[0], "judge_cache_hits": judge.hits - calls_before[1],
                          "elapsed_s": round(time.monotonic() - started, 1)})
            if result is not None:
                raw = result.raw or {}
                severe = [e for e in result.errors if e.type is ErrorType.COMPLETED_WITHOUT_FINISHING
                          and e.severity in (Severity.MAJOR, Severity.CRITICAL)]
                entry.update({"judge_failed": bool(raw.get("_failed")), "errors": [e.to_dict() for e in result.errors],
                              "missing": raw.get("missing"), "status_should_be": raw.get("status_should_be"),
                              "flagged": bool(severe)})
                if severe:
                    visible = json.dumps(rendered or payload, ensure_ascii=False) + "\n" + json.dumps(report, ensure_ascii=False)
                    detail = build_reason(raw, [e.to_dict() for e in severe], visible, case_dict.get("requirements") or [], case_dict)
                    reason = _walker.inloop_status_reason(detail["kept"], hints, family)
                    report = {**report, "status": "INCOMPLETE", "status_reason": reason}
                    entry.update({"rewritten": True, "status_reason": reason, "reason_detail": detail,
                                  "actions": _walker.inloop_actions(detail["kept"], hints, family)})
        if report.get("status") == "INCOMPLETE" and family in ("E", "L", "C") and workspace.startswith("W1-") \
                and not _walker.has_followup_clause(str(report.get("status_reason") or "")):
            reason = str(report.get("status_reason") or "").strip().rstrip(";. ")
            actions = _walker.inloop_actions(_walker._gap_clauses(reason), hints, family)
            if actions:
                report = {**report, "status_reason": "; ".join([reason, *actions]) if reason else "; ".join(actions)}
                entry.update({"followup_appended": actions})
        entry["status_out"] = report.get("status")
        path = self.run_dir / INLOOP_FILE
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"checks": []}
        record["checks"].append(entry)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        return report

    def _slot_finish(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path], decision: Decision,
                     payload: dict[str, Any], slots: list[dict[str, Any]]) -> dict[str, Any]:
        notes = decision.slot.get("notes") or []
        suffix, findings, attached, visible = _s5_suffix(payload, notes)
        cited = [e for e in visible if e not in ("E1", "E2")][-3:]
        mock = {"decision": "FINISH", "answer": "[mock] answer " + " ".join(f"[{e}]" for e in cited), "rationale": "[mock]"}
        raw = self._codex(directory, actor_dir, "S5_answer", base_prompt + suffix, image_paths, slots, mock)
        out = extract_json_object(raw)
        answer = str(out.get("answer") or "")
        draft0 = (dict(out), answer)
        # GT 核对（特权信息只在这里用）：没对上就把"可能漏了什么"反馈给教师重写一次，只许写证据支持的内容
        missed = _gt_misses(self.gt_points, answer)
        gt_log: dict[str, Any] = {"points": [g["kind"] for g in self.gt_points], "missed_first": [m["kind"] for m in missed],
                                  "support": {g["kind"]: _point_support(g, findings) for g in self.gt_points}}

        def review(name: str, hints: list[str]) -> tuple[dict[str, Any], str] | None:
            feedback = "".join(f"- {h}\n" for h in hints)
            retry = suffix + (
                "\n\nREVIEW OF YOUR DRAFT. Your draft answer was:\n" + draft0[1] + "\n\nA reviewer notes these possible gaps:\n"
                + feedback + "Re-check the findings and the attached images. Where they support the point, state it plainly and cite "
                "the E IDs of the supporting findings; where they do not, keep the draft's wording and do not add the point. "
                "Do not mention the reviewer. Return the complete JSON object again.")
            raw2 = self._codex(directory, actor_dir, name, base_prompt + retry, image_paths, slots, mock)
            out2 = extract_json_object(raw2)
            return (out2, str(out2.get("answer") or "")) if str(out2.get("answer") or "").strip() else None

        if missed and not self.mock and (self.gt_review or self.arms):
            got = review("S5_answer_review", [m["hint"] for m in missed])
            if got:
                out, answer = got
            gt_log["missed_after"] = [m["kind"] for m in _gt_misses(self.gt_points, answer)]
        base = (out, answer)
        gt_log["gt_review"] = bool(self.gt_review or self.arms)
        if not self.arms and not self.gt_review:
            out, answer = draft0          # 默认（10-14 晚起）：首稿就是答案，不用 GT；核验后只按证据重写（walker_run）
        if self.arms:
            # 10-14 对照：轨迹里交 d0（首稿，没用 GT）；b = 上面的原方法；g = 证据门控提示；n = 按 GT 从 N 份稿里挑
            arms: dict[str, Any] = {}

            def pack(item: tuple[dict[str, Any], str], **extra: Any) -> dict[str, Any]:
                text, dropped = _filter_answer(item[1], findings, attached, visible)
                return {"answer": text, "rationale": str(item[0].get("rationale") or "")[:400], "dropped_ids": dropped,
                        "gt_missed": [m["kind"] for m in _gt_misses(self.gt_points, text)], **extra}

            arms["d0"] = pack(draft0)
            arms["b"] = pack(base, hinted=bool(missed))
            hints = _gate_hints(self.gt_points, missed, findings)
            gate = review("S5_answer_gate", hints) if hints and not self.mock else None
            arms["g"] = pack(gate or draft0, hinted=bool(hints), hints=hints)
            drafts = [draft0]
            for k in range(1, self.bon):
                rawk = self._codex(directory, actor_dir, f"S5_answer_bon{k}", base_prompt + suffix, image_paths, slots, mock)
                outk = extract_json_object(rawk)
                if str(outk.get("answer") or "").strip():
                    drafts.append((outk, str(outk.get("answer") or "")))
            scores = [len(self.gt_points) - len(_gt_misses(self.gt_points, d[1])) for d in drafts]
            best = max(range(len(drafts)), key=lambda i: (scores[i], -i))      # 同分取最早的（d0 优先）
            arms["n"] = pack(drafts[best], picked=best, scores=scores)
            (directory / "slot_S5_arms.json").write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
            out, answer = draft0
            gt_log["live"] = "d0"
        (directory / "slot_S5_gt_check.json").write_text(json.dumps(gt_log, ensure_ascii=False), encoding="utf-8")
        answer, bad = _filter_answer(answer, findings, attached, visible)
        result = {"decision": "FINISH", "answer": answer, "rationale": str(out.get("rationale") or "")[:400]}
        if bad:
            (directory / "slot_S5_dropped_ids.json").write_text(json.dumps(bad), encoding="utf-8")
        return result

    def _slot_choose(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path], decision: Decision,
                     payload: dict[str, Any], slots: list[dict[str, Any]]) -> Decision:
        slot = decision.slot
        if slot.get("name") == "S1_query" and slot.get("purpose") == "reference_images":
            # 10-13 L 族：以文字搜图片，取声明所说地点 / 地标的参考照片，用来与给定照片当面比对
            suffix = (
                "\n\nHOST DECISION FOR THIS TURN (language slot S1): the host has decided that this turn is a SEARCH for "
                "reference photos of the place or landmark that the claim says the given picture shows (input_type=text, "
                "output_type=image). Write ONE image-search query.\n"
                "- Name that place or landmark as specifically as the claim allows (its name plus town or country), and add "
                "the kind of view only if the claim implies it (for example 'aerial view'). The claim: %s\n"
                "- Use on-screen text read from the picture (the texts listed on the OCR record's observation line) only if "
                "it names the same place.\n"
                "- The query must fetch photos that show what that place looks like (add what the claim says is there only "
                "when the place is known for it). Never write E/R/W IDs, 'the given image', 'fact check', or who the claim "
                "blames.\n"
                'Return only {"query": "...", "rationale": "one sentence (at most 40 words) saying which place the claim names"}.'
            ) % (json.dumps(str(slot.get("claim") or ""), ensure_ascii=False),)
            claim_words = [w for w in re.findall(r"[A-Z][\w'-]+", str(slot.get("claim") or ""))][:5]
            mock = {"query": " ".join(claim_words) or "landmark photo", "rationale": "[mock]"}
            raw = self._codex(directory, actor_dir, "S1_query", base_prompt + suffix, image_paths, slots, mock)
            try:
                ok = bool(str(extract_json_object(raw).get("query") or "").strip())
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                raw = self._codex(directory, actor_dir, "S1_query_retry", base_prompt + suffix
                                  + '\nYour previous output had no usable "query" string. Return the JSON object again.',
                                  image_paths, slots, mock)
        elif slot.get("name") == "S1_query" and slot.get("purpose") == "followup":
            # 10-16 r0+：检索类第 2 轮"再搜来源"——任务文字里列着第 1 轮没答上的点，写一条文字查询去找讲这些点的来源
            suffix = (
                "\n\nHOST DECISION FOR THIS TURN (language slot S1): the host has decided that this turn is a text SEARCH for "
                "web pages (input_type=text, output_type=text) about the open points that your task lists (what the first "
                "round left unanswered). Write ONE search query.\n"
                "- Aim it at the most important open point: name the specific event, place, people, date or publication it "
                "concerns, using words from the claim and from what the pages already read in this state say. The claim: %s\n"
                "- Never write E/R/W IDs, 'the given image', 'fact check', 'first round' or 'open point'; search engines do not "
                "know them.\n"
                'Return only {"query": "...", "rationale": "one sentence (at most 40 words) saying which open point it targets"}.'
            ) % (json.dumps(str(slot.get("claim") or ""), ensure_ascii=False),)
            claim_words = [w for w in re.findall(r"[A-Z][\w'-]+", str(slot.get("claim") or ""))][:6]
            mock = {"query": " ".join(claim_words) or "news report", "rationale": "[mock]"}
            raw = self._codex(directory, actor_dir, "S1_query", base_prompt + suffix, image_paths, slots, mock)
            try:
                ok = bool(str(extract_json_object(raw).get("query") or "").strip())
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                raw = self._codex(directory, actor_dir, "S1_query_retry", base_prompt + suffix
                                  + '\nYour previous output had no usable "query" string. Return the JSON object again.',
                                  image_paths, slots, mock)
        elif slot.get("name") == "S1_query" and slot.get("purpose") == "place_text":
            # r1（10-17，负责人定 (c)）：L 族以图搜之后必做一次文字搜——找讲"这张照片实际拍于哪里"的网页
            suffix = (
                "\n\nHOST DECISION FOR THIS TURN (language slot S1): the host has decided that this turn is a text SEARCH for "
                "web pages (input_type=text, output_type=text) that report where the given picture was actually taken or "
                "show that it is not the place the claim names. Write ONE search query.\n"
                "- Build it from clues visible in this state: text read from the picture (the texts listed on the OCR record's "
                "observation line, copied in their own language and script), words from the titles of the pages that the "
                "reverse image search returned, and place clues; add the claim's key words only when they help. The claim: %s\n"
                "- Earlier queries: %s. The new query must differ from them.\n"
                "- Never write E/R/W IDs, 'the given image', 'reverse image search' or 'fact check'; search engines do not know them.\n"
                'Return only {"query": "...", "rationale": "one sentence (at most 40 words) citing the E or R IDs the clues come from"}.'
            ) % (json.dumps(str(slot.get("claim") or ""), ensure_ascii=False),
                 json.dumps(list(slot.get("previous") or []), ensure_ascii=False))
            claim_words = [w for w in re.findall(r"[A-Z][\w'-]+", str(slot.get("claim") or ""))][:6]
            mock = {"query": " ".join(claim_words + ["photo location"]) or "photo location", "rationale": "[mock]"}
            raw = self._codex(directory, actor_dir, "S1_query", base_prompt + suffix, image_paths, slots, mock)
            try:
                ok = bool(str(extract_json_object(raw).get("query") or "").strip())
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                raw = self._codex(directory, actor_dir, "S1_query_retry", base_prompt + suffix
                                  + '\nYour previous output had no usable "query" string. Return the JSON object again.',
                                  image_paths, slots, mock)
        elif slot.get("name") == "S1_query":
            previous = slot.get("previous") or []
            suffix = (
                "\n\nHOST DECISION FOR THIS TURN (language slot S1): the host has decided that this turn is a SEARCH for "
                "possible original versions of the material video (input_type=%s, output_type=%s). Write ONE search query%s.\n"
                "- Build it only from clues visible in this state: exact on-screen captions or name straps (the texts listed "
                "on the OCR record's observation line: its most frequent texts with the times they appear), "
                "channel logos or watermarks, names, places, events and distinctive phrases that are spoken (see the ASR "
                "record), and what the attached frames show. Prefer the most specific clue (a quoted headline or caption, a "
                "named person plus event plus place).\n"
                "- Never write E/R/W IDs or words such as 'Material A', 'the given video', 'original source'; search engines do "
                "not know them. Do not add 'fact check'.\n"
                "- Earlier queries and what they returned: %s. The new query must differ from them: use another clue, fewer "
                "words, or the language of the on-screen text.\n"
                'Return only {"query": "...", "rationale": "one sentence (at most 40 words) citing the E IDs the clue comes from"}.'
            ) % (slot.get("input_type") or "text", slot.get("output_type") or "video",
                 (f" (it is sent together with the frame {slot['evidence_id']}, so describe the event, not the picture)"
                  if slot.get("input_type") == "text_image" else ""),
                 "; ".join(f"{p['query']!r} → {p['outcome']}" for p in previous) or "(none yet)")
            mock = {"query": " ".join(str((payload.get("model_view") or {}).get("task") or "news video").split()[:6]),
                    "rationale": "[mock]"}
            raw = self._codex(directory, actor_dir, "S1_query", base_prompt + suffix, image_paths, slots, mock)
            try:
                ok = bool(str(extract_json_object(raw).get("query") or "").strip())
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                raw = self._codex(directory, actor_dir, "S1_query_retry", base_prompt + suffix
                                  + '\nYour previous output had no usable "query" string. Return the JSON object again.',
                                  image_paths, slots, mock)
        else:
            unread = slot.get("unread") or []
            offsets = slot.get("offsets") or {}
            options = []
            if "search" in slot.get("choices", []):
                options.append('{"next": "search", "query": "..."} — a new text search for pages about the event in the claim '
                               "(people, places, dates, quoted words; not fact-check sites; different from earlier queries: %s)"
                               % (", ".join(repr(q) for q in slot.get("previous") or []) or "none"))
            if "read" in slot.get("choices", []):
                listing = "; ".join(f"{p['evidence_id']} ({str(p.get('title') or p.get('url') or '')[:90]})" for p in unread)
                more = "; ".join(f"{e} continues at offset {o}" for e, o in offsets.items())
                options.append('{"next": "read", "evidence_id": "E..."} — read one page; unread pages: %s%s'
                               % (listing or "none", (". Pages with more text: " + more) if more else ""))
            if "dates" in slot.get("choices", []):
                undated = "; ".join(f"{p['candidate']} ({str(p.get('title') or p.get('url') or '')[:90]})"
                                    for p in slot.get("undated") or [])
                options.append('{"next": "dates", "candidate": "R..."} — look up when one result was published, uploaded or '
                               "first archived (needed to tell which copy is earliest or whether a source predates the claim); "
                               "results without a date lookup: %s" % (undated or "none"))
            if "report" in slot.get("choices", []):
                options.append('{"next": "report"} — the pages already read answer every part of the task, or nothing '
                               "useful is left to read")
            suffix = (
                "\n\nHOST DECISION FOR THIS TURN (language slot S3): choose the next step of this claim check. Allowed:\n- "
                + "\n- ".join(options) + "\n"
                "Prefer reading a page whose title shows it reports the event itself (a news outlet, an official source) over "
                "aggregators; search again when the pages found do not concern the claim's event. "
                'Return only the chosen JSON object, with an extra "rationale" field (one sentence, at most 40 words, citing '
                "only existing IDs).")
            first = unread[0]["evidence_id"] if unread else None
            undated_first = (slot.get("undated") or [{}])[0].get("candidate") if slot.get("undated") else None
            mock = ({"next": "dates", "candidate": undated_first, "rationale": "[mock]"}
                    if undated_first and "dates" in slot.get("choices", []) and not slot.get("read")
                    else {"next": "read", "evidence_id": first, "rationale": "[mock]"} if first and "read" in slot.get("choices", [])
                    else {"next": "search", "query": "news report", "rationale": "[mock]"} if "search" in slot.get("choices", [])
                    else {"next": "report", "rationale": "[mock]"})
            raw = self._codex(directory, actor_dir, "S3_next", base_prompt + suffix, image_paths, slots, mock)
        try:
            choice = extract_json_object(raw)
        except Exception:  # noqa: BLE001 —— 解析不了按空选择处理，walker 退到保守的一步
            choice = {}
        (directory / f"slot_{slot.get('name')}_choice.json").write_text(json.dumps(choice, ensure_ascii=False), encoding="utf-8")
        resolved = resolve_slot_choice(decision, choice if isinstance(choice, dict) else {})
        resolved.slot = {**resolved.slot, "chosen_by": slot.get("name"), "choice": choice}
        return resolved

    def _slot_round(self, directory: Path, actor_dir: Path, base_prompt: str, image_paths: list[Path], decision: Decision,
                    payload: dict[str, Any], slots: list[dict[str, Any]]) -> dict[str, Any]:
        gaps = decision.slot.get("gaps") or []
        visible = set(_EREF.findall(json.dumps(payload.get("global_view") or {}, ensure_ascii=False)))
        materials = [m for m in decision.slot.get("materials") or []]
        suffix = (
            "\n\nHOST DECISION FOR THIS TURN (language slot S2): the host has decided to open one more round with exactly ONE "
            "workspace task that closes the gap the reports left open. Output exactly "
            '{"decision": "INVESTIGATE", "tasks": [{"task": "...", "materials": ["..."]}], "rationale": "..."}.\n'
            "- Gaps reported as INCOMPLETE: %s\n"
            "- The task states what still has to be established and from which material; do not repeat work the reports "
            "already completed. materials: choose from %s plus any E IDs in this state that the task needs.\n"
            "- The task text must contain this scope line verbatim at its end: %s\n"
            "Return only the JSON object."
        ) % ("; ".join(gaps) or "(unspecified)", ", ".join(materials), decision.slot.get("scope") or "(none)")
        mock = {"decision": "INVESTIGATE", "tasks": [{"task": "[mock] follow-up", "materials": materials}], "rationale": "[mock]"}
        raw = self._codex(directory, actor_dir, "S2_task", base_prompt + suffix, image_paths, slots, mock)
        out = extract_json_object(raw)
        tasks = [t for t in out.get("tasks") or [] if isinstance(t, dict) and t.get("task")][:1]
        if not tasks:
            tasks = [{"task": payload.get("global_view", {}).get("task", ""), "materials": materials}]
        scope = str(decision.slot.get("scope") or "")
        for t in tasks:
            t["materials"] = [m for m in (t.get("materials") or materials)
                              if isinstance(m, str) and (m in visible or m in materials)] or materials
            for m in materials:                       # 原始材料与程序挑的帧必须在
                if m not in t["materials"]:
                    t["materials"].append(m)
            if scope and scope not in t["task"]:      # 跟进范围由程序定；教师没照抄就补在末尾
                t["task"] = t["task"].rstrip() + " " + scope
        return {"decision": "INVESTIGATE", "tasks": tasks, "rationale": str(out.get("rationale") or "")[:400]}


_REFERENT = re.compile(r"\bthe (?:person|man|woman|boy|girl|child|people|player|rider|cyclist|guy|car|vehicle|object|animal|dog)"
                       r"(?: (?:in|wearing|with|holding|on) (?:(?:a|an|the) [a-z]+(?: [a-z]+)?|[a-z]+))?", re.I)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------- 10-15 对话 5a：教师按槽分后端

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = ROOT / "specs" / "r0_freeze.json"
SLOT_KEYS = ("S0", "S1", "S2", "S3", "S4", "S5")
_SLOT_KEY = re.compile(r"^(S[0-9])(?:_|$)")
# 后端缺省值（冻结文件没写的字段）：与 10-14 单端点后端的缺省相同
BACKEND_DEFAULTS: dict[str, Any] = {"type": "endpoint", "thinking": True, "thinking_off_slots": ["S0"], "reasoning_effort": None,
                                    "temperature": 0.3, "max_tokens": 16384, "attempts": 5, "timeout_s": 900, "providers": None}
BACKEND_TYPES = ("endpoint", "codex")
# Codex 429 / usage limit / 满载：30、60、120、240、480 s 退避（wk10 起的口径），总等待上限 FOREN_ORACLE_RETRY_MAX_S（900 s）
CODEX_BACKOFF = (30.0, 60.0, 120.0, 240.0, 480.0)
_CODEX_LIMIT_WORDS = ("429", "too many requests", "rate limit", "rate_limit", "ratelimit", "usage limit", "usage_limit",
                      "at capacity", "quota", "try again in")
_CODEX_VERSION: dict[str, str | None] = {}


def find_codex(cfg: dict[str, Any] | None = None) -> str:
    """Codex CLI 可执行文件：后端配置 codex_bin > FOREN_CODEX_BIN > PATH。找不到直接报错（预检会先报）。"""

    path = (cfg or {}).get("codex_bin") or os.environ.get("FOREN_CODEX_BIN") or shutil.which("codex")
    if not path:
        raise RuntimeError("Codex CLI is not installed or not on PATH (strong backend type=codex; set FOREN_CODEX_BIN, or switch the "
                           "frozen strong backend / FOREN_TEACHER_BACKUP=strong)")
    return str(path)


def codex_version(codex: str) -> str | None:
    if codex not in _CODEX_VERSION:
        try:
            done = subprocess.run([codex, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            _CODEX_VERSION[codex] = (done.stdout or done.stderr).strip().splitlines()[0] if (done.stdout or done.stderr).strip() else None
        except (OSError, subprocess.SubprocessError):
            _CODEX_VERSION[codex] = None
    return _CODEX_VERSION[codex]


def codex_rate_limited(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in _CODEX_LIMIT_WORDS)


def codex_model_served(events: list[dict[str, Any]], stderr: str) -> str | None:
    """Codex 输出里实际用的模型名（事件里的 model 字段或 stderr 的 "model: xxx" 行）；都没有就 None（只记 -m 请求的）。"""

    for event in events:
        for holder in (event, event.get("item") or {}, event.get("session") or {}):
            if isinstance(holder, dict) and isinstance(holder.get("model"), str) and holder.get("model"):
                return holder["model"]
    m = re.search(r"^\s*model:\s*(\S+)", stderr or "", re.MULTILINE)
    return m.group(1) if m else None


class codex_lock:
    """跨进程串行锁（--jobs 2 时两个 run 子进程的 Codex 调用排队）。with 返回等锁秒数。FOREN_CODEX_SERIAL=0 或 serial=False 不加锁。"""

    def __init__(self, serial: bool = True, path: str | Path | None = None) -> None:
        self.enabled = bool(serial) and os.environ.get("FOREN_CODEX_SERIAL", "1") != "0"
        self.path = Path(path or os.environ.get("FOREN_CODEX_LOCK") or (ROOT / "experiments" / "sft_data" / ".codex_slot.lock"))
        self.handle: Any = None

    def __enter__(self) -> float:
        if not self.enabled:
            return 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+b")
        started = time.monotonic()
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return time.monotonic() - started
            except OSError:
                time.sleep(0.5)

    def __exit__(self, *exc: Any) -> None:
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
_PUBLIC_API_HOSTS = ("openrouter.ai", "api.anthropic.com", "api.openai.com")


def slot_key(name: str) -> str:
    """槽调用名 → 槽号：S0_rationale → S0，S4_report_retry → S4，S5_answer_revise / S5_answer_bon1 → S5。"""

    m = _SLOT_KEY.match(str(name))
    return m.group(1) if m else str(name)


def _host(endpoint: str | None) -> str:
    from urllib.parse import urlparse

    if not endpoint:
        return "codex-cli"          # type=codex 的后端没有 HTTP 端点
    return urlparse(str(endpoint)).netloc or str(endpoint)


def normalize_model(model: str | None) -> str:
    """比对模型身份用：去掉 org 前缀、小写、点换成横线（anthropic/claude-sonnet-5.5 ≡ claude-sonnet-5-5；
    vLLM 的 served name Qwen/Qwen3.5-397B-A17B ≡ qwen/qwen3.5-397b-a17b）。"""

    return str(model or "").strip().rsplit("/", 1)[-1].lower().replace(".", "-")


def model_matches(actual: str | None, expected: str | None) -> bool:
    """actual 是否就是 expected 那个模型（服务器 served name 可能多后缀，如 -FP8：按前缀认）。"""

    a, e = normalize_model(actual), normalize_model(expected)
    return bool(a and e) and (a == e or a.startswith(e + "-") or a.startswith(e + ":"))


def load_slot_table(path: str | Path | None = None, *, backup: str | None = None) -> dict[str, Any]:
    """读冻结文件 teacher 一节 → {"file", "freeze_id", "slots": {S0: 后端名}, "backends": {名: 配置}, "summary"}。

    后端配置 = BACKEND_DEFAULTS ← 冻结文件 backends.<名> ←（FOREN_TEACHER_BACKUP 点名时）backup_backends.<名> ← 环境变量覆盖
    （endpoint_env / model_env / api_key_env 写的变量名，服务器上指到 H20 的 vLLM）。每个后端记 role = primary / backup。"""

    path = Path(path or os.environ.get("FOREN_TEACHER_FREEZE") or DEFAULT_FREEZE)
    data = json.loads(path.read_text(encoding="utf-8"))
    teacher = data.get("teacher") or {}
    slots = {str(k): str(v) for k, v in (teacher.get("slots") or {}).items()}
    base = teacher.get("backends") or {}
    # 分槽备用在 backup_backends（teacher.backup 留作 run 级摘要字符串，pack_r0.py 按它核对）
    backups = teacher.get("backup_backends") or (teacher.get("backup") if isinstance(teacher.get("backup"), dict) else {})
    if not slots or not base:
        raise RuntimeError(f"{path}: teacher.slots / teacher.backends missing (freeze file is not in the slot-table format)")
    missing = [k for k in SLOT_KEYS if k not in slots]
    unknown = sorted({b for b in slots.values() if b not in base})
    if missing or unknown:
        raise RuntimeError(f"{path}: slot table incomplete (missing slots {missing}, unknown backends {unknown})")
    wanted = backup if backup is not None else (os.environ.get("FOREN_TEACHER_BACKUP") or "")
    use_backup = {x.strip().lower() for x in wanted.split(",") if x.strip()}
    backends: dict[str, dict[str, Any]] = {}
    for name, conf in base.items():
        role = "backup" if (name.lower() in use_backup or "all" in use_backup) and name in backups else "primary"
        cfg: dict[str, Any] = {**BACKEND_DEFAULTS, **conf, **(backups[name] if role == "backup" else {})}
        if role == "backup" and str(backups[name].get("type") or "endpoint") != str(conf.get("type") or "endpoint"):
            # 备用换了后端类型（strong：codex → OpenRouter Sonnet 端点）：不继承主后端的字段，只用备用自己的配置
            cfg = {**BACKEND_DEFAULTS, **backups[name]}
        overrides = {}
        for field, env_field in (("endpoint", "endpoint_env"), ("model", "model_env"), ("api_key", "api_key_env")):
            env_name = cfg.get(env_field)
            if env_name and os.environ.get(env_name):
                cfg[field] = os.environ[env_name]
                if field != "api_key":
                    overrides[field] = env_name
        cfg["type"] = str(cfg.get("type") or "endpoint").lower()
        if cfg["type"] not in BACKEND_TYPES:
            raise RuntimeError(f"{path}: backend {name!r} has unknown type {cfg['type']!r} (allowed: {', '.join(BACKEND_TYPES)})")
        if not cfg.get("model") or (cfg["type"] == "endpoint" and not cfg.get("endpoint")):
            raise RuntimeError(f"{path}: backend {name!r} needs " + ("endpoint and model" if cfg["type"] == "endpoint" else "model"))
        if cfg["type"] == "codex":
            # Codex CLI 不走 HTTP：endpoint 只是记录用的标识；key 由 Codex 自己的登录（~/.codex/auth.json）管
            cfg["endpoint"] = None
            overrides.pop("endpoint", None)
        cfg["endpoint"] = str(cfg["endpoint"]).rstrip("/") if cfg.get("endpoint") else None
        cfg["openrouter"] = bool(cfg["endpoint"]) and "openrouter.ai" in str(cfg["endpoint"])
        cfg.setdefault("trust_env", bool(cfg["endpoint"]) and any(h in str(cfg["endpoint"]) for h in _PUBLIC_API_HOSTS))
        if not cfg.get("api_key") and "api.anthropic.com" in str(cfg["endpoint"]):
            cfg["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        if not cfg.get("api_key") and "api.openai.com" in str(cfg["endpoint"]):
            cfg["api_key"] = os.environ.get("OPENAI_API_KEY")
        cfg.update({"name": name, "role": role, "env_overrides": overrides or None})
        backends[name] = cfg
    models = []
    for key in SLOT_KEYS:
        model = backends[slots[key]]["model"]
        if model not in models:
            models.append(model)
    return {"file": str(path), "freeze_id": data.get("freeze_id"), "slots": {k: slots[k] for k in SLOT_KEYS},
            "backends": backends, "summary": "+".join(models)}


def public_backend(cfg: dict[str, Any]) -> dict[str, Any]:
    """后端配置里能写进 run_config / transport 的部分（不含 key 与 httpx.Client）。"""

    keep = ("name", "role", "type", "model", "endpoint", "openrouter", "thinking", "thinking_off_slots", "reasoning_effort",
            "temperature", "max_tokens", "attempts", "timeout_s", "providers", "env_overrides", "serial", "retry_max_s")
    return {k: cfg.get(k) for k in keep}


def endpoint_body(cfg: dict[str, Any], name: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    """一次槽调用的 /chat/completions 请求体。

    - temperature 为 null 就不发（gpt-6-astra、Claude 5 系在 OpenRouter / Anthropic 上不收 < 1 的温度）；
    - 思考：OpenRouter 有 reasoning_effort 时发 reasoning.effort，否则 reasoning.enabled（S0 等 thinking_off_slots 关）；
      本机 / 服务器 vLLM 关思考时发 chat_template_kwargs.enable_thinking=false；api.openai.com 发 reasoning_effort；
      api.anthropic.com（OpenAI 兼容层）什么都不发：Claude 5 默认自适应思考，兼容层不认 effort；
    - response_format=json_object，服务拒收过就不再发（cfg._json_mode_unsupported）。"""

    body: dict[str, Any] = {"model": cfg["model"], "messages": [{"role": "user", "content": content}],
                            "max_tokens": int(cfg["max_tokens"])}
    if cfg.get("temperature") is not None:
        body["temperature"] = float(cfg["temperature"])
    # 10-15：S0 rationale 是一两句话，不开思考（wk13 每步多一次带思考的调用，工作区因此跑超时）
    thinking = bool(cfg.get("thinking")) and slot_key(name) not in (cfg.get("thinking_off_slots") or [])
    effort = cfg.get("reasoning_effort")
    endpoint = str(cfg["endpoint"])
    if cfg.get("openrouter"):
        body["reasoning"] = {"effort": effort} if effort else {"enabled": thinking}
        if cfg.get("providers"):
            body["provider"] = {"order": list(cfg["providers"]), "allow_fallbacks": True}
        body["usage"] = {"include": True}
    elif "api.anthropic.com" in endpoint:
        pass
    elif "api.openai.com" in endpoint:
        if effort:
            body["reasoning_effort"] = effort
    elif not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if not cfg.get("_json_mode_unsupported"):
        body["response_format"] = {"type": "json_object"}
    return body


def _referent_rule(payload: dict[str, Any], global_view: bool = False) -> str:
    """任务里点名的对象（如 "the person in black"）：要求逐字使用，别人看不出指的是谁时核验就会对错对象。"""

    view = payload.get("global_view" if global_view else "model_view") or payload.get("model_view") or payload.get("global_view") or {}
    task = str(view.get("task") or "")
    phrases = sorted({m.group(0) for m in _REFERENT.finditer(task) if len(m.group(0).split()) >= 3}, key=len, reverse=True)
    if not phrases:
        return ""
    return (f"- Always write '{phrases[0]}' in full when you mean it (never just 'the person' / 'the subject'); other people or "
            f"objects in the frame must be named so they cannot be mistaken for it.\n")


_GT_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\w*)?\s*(?:–|—|-|to)\s*(\d+(?:\.\d+)?)\s*(?:s\b|sec)", re.I)
_EN = re.compile(r"[A-Za-z]")


def _gt_points(case_path: str | None) -> list[dict[str, Any]]:
    """案卷里程序能核对的 GT 要点：label_expected（别名子串）、interval_iou（区间 IoU）、VCSL 备注里的标注段。"""

    if not case_path or not Path(case_path).is_file():
        return []
    try:
        case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    points: list[dict[str, Any]] = []
    for check in case.get("checks") or []:
        params = check.get("params") or {}
        if check.get("kind") == "label_expected":
            terms = [params.get("label", "")] + list(params.get("aliases") or [])
            english = [t for t in terms if t and _EN.search(t)]
            hint = ("the answer may not state its verdict on the claim plainly; if the evidence supports it, a verdict such as: "
                    if case.get("task_kind") == "claim_check" else "the answer may not say whether the footage shows: ")
            points.append({"kind": "label", "terms": [t.lower() for t in terms if t], "hint": hint + ", ".join(english[:5])})
        elif check.get("kind") == "date_in_range" and params.get("min") and params.get("max"):
            points.append({"kind": "date", "min": str(params["min"]), "max": str(params["max"]),
                           "hint": "the answer may not state the date of the earliest copy that the date lookups and pages "
                                   "support; re-check which source is earliest and give its date"})
        elif check.get("kind") == "entity_in_evidence" and params.get("text"):
            # 10-13 L 族：照片实际拍摄地（提示里不写出地名，只让教师回看已读页面）
            terms = [str(params["text"])] + [str(x) for x in params.get("statement_aliases") or []]
            points.append({"kind": "entity", "terms": [t.lower() for t in terms if t],
                           "hint": "the answer may not say where the picture was actually taken as the pages read state it; "
                                   "re-check those pages and name the place they give, or say that it is not established"})
        elif check.get("kind") == "interval_iou":
            points.append({"kind": "interval", "start": float(params["start"]), "end": float(params["end"]),
                           "min_iou": float(params.get("min_iou", 0.5)),
                           "hint": "the localized time range may not be the one the boundary frames support; re-check which "
                                   "sampled frames match the description and which do not, and give the range they bracket"})
    m = re.search(r"A\s*(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*秒\s*↔\s*B\s*(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*秒",
                  str(case.get("notes") or ""))
    if m and case.get("task_kind") == "copy_detect":
        points.append({"kind": "pair", "a": (float(m.group(1)), float(m.group(2))), "b": (float(m.group(3)), float(m.group(4))),
                       "hint": "the answer may not give the full extent of the main corresponding span on both sides; re-check "
                               "the comparison results for the longest span that corresponds and state its A and B ranges"})
    return points


def _iou(x: tuple[float, float], y: tuple[float, float]) -> float:
    inter = max(0.0, min(x[1], y[1]) - max(x[0], y[0]))
    union = max(x[1], y[1]) - min(x[0], y[0])
    return inter / union if union > 0 else 0.0


def _image_for_claude(path: Path) -> tuple[str, str]:
    """附图给 Claude：API 单图上限 5MB（base64 前约 3.7MB），大于 3.5MB 的转成 JPEG（长边 ≤ 2400）。"""

    data = Path(path).read_bytes()
    mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
    if len(data) > 3_500_000:
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(data)) as img:
                img = img.convert("RGB")
                img.thumbnail((2400, 2400))
                for quality in (88, 75, 60):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= 3_500_000:
                        break
            data, mime = buf.getvalue(), "image/jpeg"
        except Exception:  # noqa: BLE001 - 转不了就原样发，API 会报错，按槽失败处理
            pass
    return mime, base64.b64encode(data).decode("ascii")


# 10-14 S5 对照：GT 要点的"非泄露"描述（门控提示里用；不含 GT 的取值）
_PART_TEXT = {
    "label": "the overall verdict the task asks for",
    "date": "the date of the earliest copy",
    "entity": "where the picture was actually taken",
    "interval": "the localized time range",
    "pair": "the extent of the main corresponding span on both sides",
}


def _point_support(point: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    """哪些 Finding 的文字本身就含这条 GT 要点（同 _gt_misses 的判定）→ 它们引的 E。空 = 学生可见的证据里没有。"""

    out: list[str] = []
    for f in findings:
        if not _gt_misses([point], f["statement"]):
            out.extend(e for e in f["evidence_ids"] if e not in out)
    return out


def _gate_hints(points: list[dict[str, Any]], missed: list[dict[str, Any]], findings: list[dict[str, Any]]) -> list[str]:
    """证据门控提示：漏掉的要点只有在可见 Finding 里确实有时才提示，且只指到 E、不说 GT 是什么。
    裁决标签（label）在 Finding 里多半不会逐字出现：没有支持时只要求"按证据把结论说清楚，或说无法判定"，不给候选词。"""

    hints = []
    for m in missed:
        part = _PART_TEXT.get(m["kind"], "a part of the task")
        support = _point_support(m, findings)
        if support:
            hints.append(f"the findings citing {' '.join('[' + e + ']' for e in support[:4])} bear on {part}; re-read them "
                         "and state plainly what they establish about it")
        elif m["kind"] == "label":
            hints.append(f"the answer may not state {part} plainly; give the verdict the findings support, or say that "
                         "it cannot be determined from them")
    return hints


def _gt_misses(points: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    low = answer.lower()
    ranges = [(float(a), float(b)) for a, b in _GT_RANGE.findall(answer) if float(b) > float(a)]
    missed = []
    for p in points:
        if p["kind"] == "label" and not any(_label_hit(t, answer) for t in p["terms"]):
            missed.append(p)
        elif p["kind"] == "interval" and not any(_iou(r, (p["start"], p["end"])) >= p["min_iou"] for r in ranges):
            missed.append(p)
        elif p["kind"] == "date" and not any(p["min"] <= d <= p["max"] for d in _dates_in(answer)):
            missed.append(p)
        elif p["kind"] == "pair" and not any(_iou(r, p["a"]) >= 0.5 for r in ranges):
            missed.append(p)
        elif p["kind"] == "entity" and not any(t in low for t in p["terms"]):
            missed.append(p)
    return missed


def _base_prompt(request: dict[str, Any]) -> str:
    """教师槽的公共前缀：request.json（payload 换成渲染后的那份）= 学生这一步看到的状态。"""

    shown = {key: value for key, value in request.items() if key not in ("rendered_payload", "state_render")}
    if request.get("state_render"):
        shown["payload"] = request.get("rendered_payload")
    return ("You are a policy in a separate executable investigation environment. "
            "Use only the supplied state and attached images. Do not use any Codex tools, "
            "shell, filesystem, network or external memory. Do not inspect this directory. "
            "The host executes your chosen action and will request the subsequent decision. "
            "Return exactly one JSON object conforming to the role's instructions. "
            "Do not invent observations, evidence, or future results.\n\n"
            + json.dumps(shown, ensure_ascii=False))


def revise_answer(run_dir: Path, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    """核验后重写（10-13）：答案是轨迹最后一步，后面没有决策依赖它。核验判 negative 时，把核验器指出的句子与原因
    反馈给教师重写一次（输入 = 当时那一步的状态 + 附图 + S5 规则），写回 execution_links / response / result；原文件留 .orig。
    只改答案；报告不能这样改（GIA 已按原报告做了决定）。返回改动记录，没找到 FINISH 请求时返回 None。"""

    run_dir = Path(run_dir)
    links_path = run_dir / "execution_links.json"
    links = json.loads(links_path.read_text(encoding="utf-8"))
    finish = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH" and l.get("request_id")]
    if not finish:
        return None
    link = finish[-1]
    rdir = run_dir / "requests" / link["request_id"]
    request = json.loads((rdir / "request.json").read_text(encoding="utf-8"))
    payload = request["payload"]
    transport = json.loads((rdir / "transport.json").read_text(encoding="utf-8")) if (rdir / "transport.json").is_file() else {}
    notes = ((transport.get("oracle") or {}).get("slot") or {}).get("notes") or []
    suffix, findings, attached, visible = _s5_suffix(payload, notes)
    old = dict(link["raw_output"])
    lines = []
    for e in errors:
        span = str(e.get("span") or "").strip()
        lines.append(f"- [{e.get('type')}] " + (f"\"{span[:300]}\" — " if span else "") + str(e.get("detail") or "")[:400])
    prompt = (_base_prompt(request) + suffix + "\n\nREVIEW OF YOUR PREVIOUS ANSWER. Your answer was:\n" + str(old.get("answer") or "")
              + "\n\nAn evidence checker compared each sentence with the evidence it cites and flagged these problems:\n"
              + "\n".join(lines) + "\nRewrite the answer: fix or remove each flagged sentence so that it says only what its cited "
              "findings / images show (re-cite or drop as needed); keep the unflagged sentences unchanged; do not add new claims. "
              "Write in English. Do not mention the checker. Return the complete JSON object again.")
    image_paths = sorted((rdir / "actor").glob("image_*.png"))
    work = rdir / "revise"
    work.mkdir(exist_ok=True)
    generator = OracleGenerator(store=None, output_dir=work)
    notes_img = list(request.get("image_notes") or [])
    if len(notes_img) == len(image_paths) + 1:
        generator._caption_intro, generator._captions = notes_img[0], notes_img[1:]
    slots: list[dict[str, Any]] = []
    raw = generator._codex(work, rdir / "actor", "S5_answer_revise", prompt, image_paths, slots,
                           {"decision": "FINISH", "answer": old.get("answer"), "rationale": "[mock]"})
    out = extract_json_object(raw)
    answer, dropped = _filter_answer(str(out.get("answer") or ""), findings, attached, visible)
    if not answer:
        return {"run": run_dir.name, "request": link["request_id"], "changed": False, "reason": "empty revision"}
    new = {"decision": "FINISH", "answer": answer, "rationale": str(out.get("rationale") or old.get("rationale") or "")[:400]}
    _write_answer(run_dir, new)
    record = {"run": run_dir.name, "request": link["request_id"], "changed": True, "errors": errors, "old": old.get("answer"),
              "new": answer, "dropped_ids": dropped, "slots": slots}
    (rdir / "slot_S5_revise.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _write_answer(run_dir: Path, new: dict[str, Any]) -> bool:
    """把 FINISH 的答案换成 new（execution_links / 该请求的 response.json / result.json），原文件留 .orig（只留第一次的）。"""

    run_dir = Path(run_dir)
    links_path = run_dir / "execution_links.json"
    links = json.loads(links_path.read_text(encoding="utf-8"))
    finish = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH" and l.get("request_id")]
    if not finish:
        return False
    link = finish[-1]
    rdir = run_dir / "requests" / link["request_id"]
    for path in (links_path, rdir / "response.json", run_dir / "result.json"):
        backup = path.with_name(path.name + ".orig")
        if path.is_file() and not backup.exists():
            shutil.copy2(path, backup)
    link["raw_output"] = new
    links_path.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    (rdir / "response.json").write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    if (run_dir / "result.json").is_file():
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        if isinstance(result.get("decision"), dict):
            result["decision"]["answer"] = new["answer"]
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def make_arm_run(run_dir: Path, arm: str, arm_tag: str, live_tag: str) -> dict[str, Any] | None:
    """10-14 对照：把一条带 slot_S5_arms.json 的 run 拷成 <案例>_<arm_tag>_1，答案换成该版。媒体 / 附图用硬链接
    （之后谁都不改它们），json / sqlite 实拷（重写会改 json；核验开 store 前自己会复制）。已存在就跳过。"""

    run_dir = Path(run_dir)
    links = json.loads((run_dir / "execution_links.json").read_text(encoding="utf-8"))
    finish = [l for l in links if (l.get("raw_output") or {}).get("decision") == "FINISH" and l.get("request_id")]
    if not finish:
        return None
    arms_path = run_dir / "requests" / finish[-1]["request_id"] / "slot_S5_arms.json"
    if not arms_path.is_file():
        return None
    arms = json.loads(arms_path.read_text(encoding="utf-8"))
    if arm not in arms:
        return None
    target = run_dir.with_name(run_dir.name.replace(f"_{live_tag}_", f"_{arm_tag}_"))
    if target == run_dir:
        raise ValueError(f"run 名里没有 _{live_tag}_：{run_dir.name}")
    if (target / "result.json").is_file():
        return {"run": target.name, "arm": arm, "created": False}
    if target.exists():
        shutil.rmtree(target)

    def link_or_copy(src: str, dst: str) -> None:
        if Path(src).suffix.lower() in (".json", ".jsonl", ".sqlite3", ".txt", ".orig", ".db", ".wal", ".shm"):
            shutil.copy2(src, dst)
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(run_dir, target, copy_function=link_or_copy)
    item = arms[arm]
    _write_answer(target, {"decision": "FINISH", "answer": item["answer"], "rationale": item.get("rationale") or ""})
    for name in ("execution_links.json.orig", "result.json.orig"):       # 拷来的 .orig 是 d0，留着会挡住"每条只重写一次"
        (target / name).unlink(missing_ok=True)
    (target / "requests" / finish[-1]["request_id"] / "response.json.orig").unlink(missing_ok=True)
    cfg_path = target / "run_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["round"] = arm_tag
        cfg["s5_arm"] = {"arm": arm, "copied_from": run_dir.name}
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run": target.name, "arm": arm, "created": True}


def _s5_suffix(payload: dict[str, Any], notes: list[str]) -> tuple[str, list[dict[str, Any]], list[str], list[str]]:
    """S5 答案槽的提示后缀（生成与核验后重写共用）→ (suffix, findings, attached, visible)。"""

    view = payload.get("global_view") or {}
    visible = sorted(set(_EREF.findall(json.dumps(view, ensure_ascii=False))), key=lambda e: int(e[1:]))
    findings = _collect_findings(view)
    attached = [i.get("evidence_id") for i in view.get("image_inputs") or [] if isinstance(i, dict) and i.get("evidence_id")]
    table = "".join(f"  [{f['task_id']} #{f['n']}] {f['statement'][:600]} — evidence: {', '.join(f['evidence_ids'])}\n"
                    for f in findings) or "  (no findings)\n"
    suffix = (
        "\n\nHOST DECISION FOR THIS TURN (language slot S5): the host has already decided to FINISH now. Output exactly "
        '{"decision": "FINISH", "answer": "...", "rationale": "..."}.\n'
        "- The answer addresses every part of the original task explicitly (including sound / other modalities when the task "
        "mentions them); a part that was not verified is stated as not verified.\n"
        "- Build the answer from these workspace findings (all rounds):\n%s"
        "- Each sentence restates one or more of those findings and ends with the E IDs listed for them, each in its own "
        "brackets, e.g. [E12][E15]. You may drop IDs, never add IDs a restated finding does not list. Images attached to this "
        "decision (%s) may also be cited for what they plainly show.\n"
        "- A finding's summary facts (matched share, spans, speeds, cuts) keep the tool result E it cites; never move them onto "
        "an image E. Cite the result that actually contains the point (an image-lookup result for a candidate pair, not a "
        "comparison that reported nothing).\n"
        "- Do not restate exclusions the task itself sets (e.g. that audio was not assessed when the task says not to assess it).\n"
        "- Do not add objects, colours, positions, motions or other visual details that the restated finding does not state; "
        "do not sharpen a finding's certainty or scope. Refer to people and objects with the task's own wording.\n"
        "- Copy times, ranges, durations and resolutions exactly as the findings state them, attributed to the correct side "
        "(A or B); do not recompute them through a time mapping.\n"
        "- Do not inherit a report's COMPLETED status beyond what its findings show; say plainly what was not verified "
        "(including modalities or ranges that were not examined).\n"
        "- Never mention that records or results are truncated or not fully shown.\n"
        "- State time ranges only as the findings state them; never build a range from the times of sampled frames.\n"
        "- Do not add generic disclaimers the task does not ask about (editing software, AI use, authorship, publisher, "
        "production history); limits go in one short sentence about what was not examined, citing the result that shows "
        "its coverage.\n"
        "%s"
        "- Keep it short: at most about 200 words, conclusions first; do not replay every check.\n%s"
        "Return only the JSON object."
    ) % (table, ", ".join(attached) or "none", _referent_rule(payload, global_view=True),
         "".join(f"- {n}\n" for n in notes))
    return suffix, findings, attached, visible


def _filter_answer(answer: str, findings: list[dict[str, Any]], attached: list[str], visible: list[str]) -> tuple[str, list[str]]:
    """答案里只留 Finding 列出的 E 与本次附图；其余 E 删掉。"""

    allowed = {e for f in findings for e in f["evidence_ids"]} | set(attached)
    if not findings:
        allowed = set(visible)
    bad = sorted({e for e in _EREF.findall(answer) if e not in allowed or e not in set(visible)}, key=lambda e: int(e[1:]))
    for e in bad:
        answer = re.sub(r"\[" + e + r"\]", "", answer)
        answer = re.sub(r"\b" + e + r"\b", "", answer)
    return answer.strip(), bad


def _collect_findings(view: dict[str, Any]) -> list[dict[str, Any]]:
    """GIA 状态里能看到的所有 Finding（latest_reports 与记忆检索结果里的），去重。"""

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(node: Any, task_id: str) -> None:
        if isinstance(node, dict):
            tid = str(node.get("task_id") or node.get("workspace_id") or task_id or "") or "?"
            if isinstance(node.get("statement"), str) and isinstance(node.get("evidence_ids"), list):
                key = node["statement"].strip()[:200]
                if key and key not in seen:
                    seen.add(key)
                    n = 1 + sum(1 for f in out if f["task_id"] == tid)
                    out.append({"task_id": tid, "n": n, "statement": node["statement"].strip(),
                                "evidence_ids": [e for e in node["evidence_ids"] if isinstance(e, str)]})
                return
            for value in node.values():
                walk(value, tid)
        elif isinstance(node, list):
            for value in node:
                walk(value, task_id)
        elif isinstance(node, str) and '"statement"' in node and node.lstrip().startswith(("{", "[")):
            try:
                walk(json.loads(node), task_id)
            except ValueError:
                pass

    walk(view.get("latest_reports") or [], "")
    walk(view.get("history_index") or [], "")
    walk({k: v for k, v in view.items() if k not in ("latest_reports", "history_index")}, "")
    return out


def _coverage_status(status: str, reason: str | None, report: dict[str, Any], parts: list[str]) -> tuple[str, str | None]:
    """逐条子要求：教师标了 not_answered、或某条没有对应 Finding，COMPLETED 就降为 INCOMPLETE，原因写明哪一条。
    10-15（B6）：not_answered 写的是佐证缺口（第二独立来源 / 一手来源 / 视觉对应），或那一条只是声明原文 / 日期的碎片，
    不算缺口——佐证缺口写进 limitations，状态不降。"""

    if status != "COMPLETED" or not parts:
        return status, reason
    coverage = report.get("part_coverage") if isinstance(report.get("part_coverage"), list) else []
    n_findings = len([f for f in report.get("findings") or [] if isinstance(f, dict)])
    answered: set[int] = {i for i, part in enumerate(parts, 1) if not _required_part(part)}
    gaps: list[str] = []
    for entry in coverage:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(entry.get("part"))
        except (TypeError, ValueError):
            continue
        if entry.get("not_answered"):
            if number in answered or _CORROBORATION.search(str(entry["not_answered"])):
                answered.add(number)
                _add_limitation(report, str(entry["not_answered"]),
                                _finding_for(report, number, parts[number - 1] if 0 < number <= len(parts) else ""))
                continue
            gaps.append(f"part {number} ({parts[number - 1][:80] if 0 < number <= len(parts) else '?'}): {str(entry['not_answered'])[:160]}")
        else:
            try:
                index = int(entry.get("finding"))
            except (TypeError, ValueError):
                continue
            if 1 <= index <= n_findings:
                answered.add(number)
    missing = [i for i in range(1, len(parts) + 1) if i not in answered and not any(g.startswith(f"part {i} ") for g in gaps)]
    gaps += [f"part {i} ({parts[i - 1][:80]}) has no finding" for i in missing]
    if not gaps:
        return status, reason
    return "INCOMPLETE", "not every part of the task is answered by the evidence: " + "; ".join(gaps)


# 佐证缺口（B6，r0a 20 份被 S4 降级的报告里的主要理由）：缺第二 / 独立来源、一手 / 原始 / 官方来源、视觉 / 建筑对应
_CORROBORATION = re.compile(
    r"\b(?:second|another|additional|other|separate)\s+(?:independent\s+)?(?:source|report|outlet|confirmation)s?\b"
    r"|\bindependent(?:ly)?\b|\bcorroborat|\bfirst[- ]hand\b|\bprimary\s+source"
    r"|\boriginal\s+(?:source|attribution|official\s+statement|statement|post)\b|\bofficial\s+(?:statement|confirmation|source)"
    r"|\bvisual(?:ly)?\s+(?:or\s+documentary\s+)?(?:correspondence|match|comparison|compared|identification|confirmation)"
    r"|\bnot\s+visually\s+compared|\b(?:architectural|landmark)\s+(?:match|correspondence|identification)"
    r"|\bcorrespondence\s+with\s+(?:an?|the)\s+(?:external|described)", re.I)
_CLAIM_DATE = re.compile(r"\((?:posted|claim date):", re.I)
_PART_VERB = re.compile(r"^(?:state|give|check|cite|investigate|find|determine|describe|say|explain|fact-check|compare|answer|"
                        r"verify|establish|identify|report|use|inspect|look|examine|when|whether|where|how|what|which|scope)\b", re.I)
_STOP = frozenset(("that this with from were have been which there their does only given material materials established "
                   "establish establishes verified evidence obtained available whether where when what could cannot about into "
                   "than them they these those such also part task claim source sources picture image photograph photo video "
                   "shows show state find check give cite remain remains").split())


def _required_part(part: str) -> bool:
    """task_parts 里真正要回答的一项；把声明原文拆碎了的片段（'Military Veterans (posted: 2015-08-04)'、'(claim date: …)'）不算。"""

    text = part.strip().lstrip('"“')
    return not (text.startswith("(") or (_CLAIM_DATE.search(text) and not _PART_VERB.match(text)))


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def _finding_for(report: dict[str, Any], number: int | None, text: str = "") -> int | None:
    """part_coverage 里回答第 number 条的 Finding（0 起）；没有就取与 text 用词最接近的 Finding。"""

    findings = [f for f in report.get("findings") or [] if isinstance(f, dict)]
    for entry in report.get("part_coverage") or []:
        if isinstance(entry, dict) and str(entry.get("part")) == str(number) and not entry.get("not_answered"):
            try:
                index = int(entry.get("finding")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(findings):
                return index
    words = _stems(text)
    scored = [(len(words & _stems(str(f.get("statement") or ""))), i) for i, f in enumerate(findings)]
    return max(scored)[1] if scored and max(scored)[0] > 0 else None


def _add_limitation(report: dict[str, Any], text: str, index: int | None) -> None:
    findings = [f for f in report.get("findings") or [] if isinstance(f, dict)]
    if not findings or not text.strip():
        return
    target = findings[index] if index is not None and 0 <= index < len(findings) else findings[-1]
    limits = target.get("limitations") if isinstance(target.get("limitations"), list) else []
    if text.strip() not in limits:
        limits.append(text.strip())
    target["limitations"] = limits


def _part_gaps(report: dict[str, Any], parts: list[str]) -> dict[int, str]:
    """part_coverage 里能算缺口的 not_answered：编号对得上一条要回答的 task_part，且写的不是佐证缺口。"""

    gaps: dict[int, str] = {}
    for entry in report.get("part_coverage") or []:
        if not isinstance(entry, dict) or not str(entry.get("not_answered") or "").strip():
            continue
        try:
            number = int(entry.get("part"))
        except (TypeError, ValueError):
            continue
        text = str(entry["not_answered"]).strip()
        if 0 < number <= len(parts) and _required_part(parts[number - 1]) and not _CORROBORATION.search(text):
            gaps[number] = text
    return gaps


def _final_status(host_status: str, report: dict[str, Any], parts: list[str] | None = None) -> str:
    """程序定的 status 为准；教师看了证据后可以把 COMPLETED 降为 INCOMPLETE（要写 status_reason），不能反过来。
    10-15（B6）：降级只在 status_reason 对得上某条 task_part（任务要求的具体项没有证据回答，part_coverage 里标了
    not_answered）时采纳，status_reason 只留对得上的段；佐证缺口与对不上任何一条的段移进 limitations（report 原地改写）。
    一条都对不上就维持 COMPLETED。"""

    if host_status != "COMPLETED" or report.get("status") != "INCOMPLETE" or not str(report.get("status_reason") or "").strip():
        return host_status
    parts = list(parts or []) or ["(the task as a whole)"]
    gaps = _part_gaps(report, parts)
    kept: list[str] = []
    for piece in (p.strip() for p in str(report["status_reason"]).split(";")):
        if not piece:
            continue
        if not _CORROBORATION.search(piece) and any(len(_stems(piece) & _stems(parts[n - 1] + " " + text)) >= 2
                                                    for n, text in gaps.items()):
            kept.append(piece)
        else:
            _add_limitation(report, piece, _finding_for(report, None, piece))
    if gaps and not kept:
        kept = [f"{parts[n - 1][:80].rstrip(' ,.')}: {text}" for n, text in sorted(gaps.items())]
    if gaps:
        report["status_reason"] = "; ".join(kept)
        return "INCOMPLETE"
    report["status"] = host_status
    report.pop("status_reason", None)
    return host_status


# 10-15 对话 5''（追加 3）：S4 报告的程序校验（提示词里的 CONTRACT_RULES 对应的三条）
_UNAVAILABLE = re.compile(r"\b(?:not|n't|never)\s+(?:been\s+)?(?:available|attached|provided|supplied|included|shown|present)\b"
                          r"|\bunavailable\b|\bno\s+(?:reference\s+)?(?:photo|image|picture|photograph|transcript|ocr)s?\s+(?:is|are|was|were)\s+"
                          r"(?:available|attached|provided)", re.I)
_UNESTABLISHED = re.compile(r"\bnot\s+(?:been\s+|yet\s+|independently\s+)?(?:verified|established|determined|confirmed)\b"
                            r"|\b(?:could|can)\s*not\s+be\s+(?:determined|established|verified|confirmed)|\bcannot\s+be\s+"
                            r"(?:determined|established|verified|confirmed)|\bunverified\b|\bundetermined\b"
                            r"|\bremains?\s+(?:unknown|undetermined|unclear|unestablished)", re.I)
_OPEN_GAP = re.compile(r"candidate pages that match the claim were not examined: https?://|reference photos .{0,80}were not viewed "
                       r"side by side|fetch the main image of https?://", re.I)
_CLAIM_VERDICT = re.compile(r"^(?:therefore,?\s+|so\s+|overall,?\s+)?the claim(?:'s|’s)?\b", re.I)
_KIND_WORDS = {
    "reference": (re.compile(r"\breference\s+(?:photo|image|picture|photograph)s?|\bphotos?\s+of\s+the\s+(?:claimed|named)", re.I),
                  lambda k: k.get("image") and str(k.get("acquired_by") or "").startswith("SEARCH")),
    "transcript": (re.compile(r"\btranscri", re.I), lambda k: k.get("acquired_by") == "INSPECT/asr"),
    "ocr": (re.compile(r"\bOCR\b|on-screen text|caption text", re.I), lambda k: k.get("acquired_by") == "INSPECT/ocr"),
    "comparison": (re.compile(r"\bcompar(?:ison|e_media)\b", re.I), lambda k: k.get("acquired_by") == "INSPECT/compare_media"),
}
# 结论里提到这些，就该引对应的工具记录（有这种记录时），不能只引采样帧
_ESTABLISH_BY = {
    "INSPECT/compare_media": re.compile(r"\b(?:match(?:es|ed|ing)?|correspond\w*|identical|differ\w*|speed\w*|slow\w*|cut|"
                                        r"segment|span|changed region|not in B|not in A)\b", re.I),
    "INSPECT/ocr": re.compile(r"\b(?:reads?|text|caption|written|wording|lorem)\b", re.I),
    "SEARCH/dates": re.compile(r"\b(?:archiv\w*|published|upload\w*|dated|no later than)\b", re.I),
}


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.;!?])\s+", text or "") if x.strip()]


def required_finding_problems(report: dict[str, Any], required: list[dict[str, Any]]) -> list[str]:
    """r1（10-17）：walker 要求必须写成 Finding 的结论（E：每条图片比对记录；L：参考图比对的结论；C：帧里有没有声明说的事、
    整段转写说了什么）。一项算写了 = 某个 Finding 引了其中任一 E（ids 为空则不要求引用）、且 statement 或 limitations 里有
    对应的词。"""

    problems: list[str] = []
    findings = [f for f in report.get("findings") or [] if isinstance(f, dict)]
    for item in required or []:
        ids = set(item.get("ids") or [])
        words = re.compile(str(item.get("words") or "."), re.I)
        ok = False
        for f in findings:
            cited = set(f.get("evidence_ids") or [])
            text = " ".join([str(f.get("statement") or ""), *[str(x) for x in f.get("limitations") or []]])
            if (not ids or cited & ids) and words.search(text):
                ok = True
                break
        if not ok:
            problems.append(str(item.get("message") or "a required finding is missing"))
    return problems


def s4_contract_problems(report: dict[str, Any], kinds: dict[str, dict[str, Any]],
                         required: list[dict[str, Any]] | None = None) -> list[str]:
    """报告违反 CONTRACT_RULES 的地方（给教师的反馈句，空 = 没问题）：
    (a) limitation 说某条已在可引 E 里的记录 / 某类已取得的材料"不可用 / 没附上"；
    (b1) statement 里有"未核实 / 未确定"的句子（这类内容该进 limitations，不挂引用）；
    (b2) statement 讲的是比对 / OCR / 日期这类工具结果，有这种记录却只引了采样帧；
    (c)（r1）walker 要求的结论没写成 Finding（required_finding_problems）。"""

    problems: list[str] = required_finding_problems(report, required or [])
    have = {name: sorted(e for e, k in kinds.items() if test(k)) for name, (_, test) in _KIND_WORDS.items()}
    for i, f in enumerate(report.get("findings") or [], 1):
        if not isinstance(f, dict):
            continue
        for limit in f.get("limitations") or []:
            for sent in _sentences(str(limit)):
                if not _UNAVAILABLE.search(sent):
                    continue
                named = [e for e in _EREF.findall(sent) if e in kinds]
                kind_hit = [name for name, (pat, _) in _KIND_WORDS.items() if pat.search(sent) and have[name]]
                if named or kind_hit:
                    ids = named or [e for name in kind_hit for e in have[name]][:4]
                    problems.append(f"finding {i} limitation says {sent[:120]!r}, but {', '.join(ids)} "
                                    f"{'is' if len(ids) == 1 else 'are'} in the observed records; say what was or was not "
                                    "compared or examined instead")
        statement = str(f.get("statement") or "")
        for sent in _sentences(statement):
            # 对声明本身的判词（"The claim … could not be verified"）是 SEARCH_RULES 要的回答形式，不算
            if _UNESTABLISHED.search(sent) and not _CLAIM_VERDICT.match(sent):
                problems.append(f"finding {i} statement contains {sent[:120]!r}: what is not established belongs in limitations, "
                                "not in a cited statement")
                break
        cited = [e for e in f.get("evidence_ids") or [] if e in kinds]
        if cited and all(str((kinds.get(e) or {}).get("acquired_by")) == "INSPECT/get_frames" for e in cited):
            for source, pat in _ESTABLISH_BY.items():
                records = [e for e, k in kinds.items() if k.get("acquired_by") == source]
                if records and pat.search(statement):
                    problems.append(f"finding {i} cites only sampled frames ({', '.join(cited)}) for what the "
                                    f"{source.split('/')[-1]} record {', '.join(records[:3])} establishes; cite that record")
                    break
    return problems


# 10-15 对话 5''（追加 1）：理由与动作对象一致性。r0a fmnv_c_RXR0B1SxiU0 request_0006/0007：动作读 E14 / E15，理由两次都说
# "read the ASR transcript (E11)"；averimatec_e17 request_0009：动作读 E4，理由讲 E6。
_RID = re.compile(r"\b[ER][1-9][0-9]*\b")
_PAGE_OF = re.compile(r"\[(R[1-9][0-9]*)\][^\"\n]{0,600}?page=(?:ready|partial)\((E[1-9][0-9]*)")


def action_targets(output: dict[str, Any]) -> set[str]:
    """动作参数里指名的单条记录：READ_RECORD 的 record_id（focus 也算——L3 并排看：读参考图、同时附上给定照片）、
    SEARCH 的 source（日期查询 / 打开某个候选）。INSPECT 的对象是整段素材（E1 / E2），理由引比对结果、帧来说明为什么取帧
    是正常的（r0a 30 条这类"命中"全是误报），不判；文字查询、以图搜、QUERY_MEMORY 也不判。"""

    args = output.get("arguments") or {}
    out: set[str] = set()
    if output.get("action") == "READ_RECORD":
        out.add(str(args.get("record_id") or ""))
        out |= {str(e) for e in output.get("focus_evidence_ids") or []}
    elif output.get("action") == "SEARCH" and args.get("source"):
        out.add(str(args["source"]))
    return {x for x in out if _RID.fullmatch(x)}


def rationale_target_check(output: dict[str, Any], shown_text: str) -> dict[str, Any]:
    """理由里提到的 E / R 编号不含动作的对象、却含别的编号 → mismatch。检索结果里同一条的候选 R 与页面 E 视为同一对象
    （读 E12 时理由说 R5 的页面、查 R5 的日期时理由说它的页面 E12，都算对）。没有对象的动作、理由里没提编号的不判。"""

    targets = action_targets(output)
    mentioned = set(_RID.findall(str(output.get("rationale") or "")))
    if not targets or not mentioned:
        return {"status": "skip", "targets": sorted(targets), "mentioned": sorted(mentioned)}
    same: dict[str, set[str]] = {}
    for r, e in _PAGE_OF.findall(shown_text.replace('\\"', '"')):
        same.setdefault(r, set()).add(e)
        same.setdefault(e, set()).add(r)
    accepted = set(targets)
    for t in targets:
        accepted |= same.get(t, set())
    status = "ok" if mentioned & accepted else "mismatch"
    return {"status": status, "targets": sorted(targets), "accepted": sorted(accepted), "mentioned": sorted(mentioned)}


def _clean_report(report: dict[str, Any], *, task_id: str, status: str, reason: str | None,
                  allowed: set[str]) -> tuple[dict[str, Any], list[str]]:
    dropped: list[str] = []
    findings = []
    for f in report.get("findings") or []:
        if not isinstance(f, dict) or not str(f.get("statement") or "").strip():
            continue
        ids = []
        for e in f.get("evidence_ids") or []:
            if e in allowed and e not in ids:
                ids.append(e)
            elif e not in allowed:
                dropped.append(str(e))
        if not ids:
            continue
        findings.append({"statement": str(f["statement"]).strip(), "evidence_ids": ids,
                         "limitations": [str(x) for x in f.get("limitations") or [] if str(x).strip()]})
    selected = []
    for f in findings:
        selected += [e for e in f["evidence_ids"] if e not in selected]
    out: dict[str, Any] = {"task_id": task_id, "status": status, "findings": findings, "selected_evidence_ids": selected,
                           "rationale": str(report.get("rationale") or "")[:400]}
    if status == "INCOMPLETE":
        out["status_reason"] = str(report.get("status_reason") or reason or "key evidence could not be obtained")
    return out, dropped
