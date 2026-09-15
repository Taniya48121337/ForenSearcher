"""两个本地模型策略的受控端到端冒烟测试。

这里不访问真实搜索站点，而是用固定 Provider/Opener 提供可验证证据；真实 GIA 和 HI
仍需完成“分派 Workspace → 调工具 → 写报告 → GIA 终结”的完整链路。因此测试失败
能定位模型策略与系统接口是否真正对齐，而不会受公网波动干扰。
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, ImageDraw

from .inspect.factory import create_inspect_service
from .local_models import LocalPolicyBundle, create_qwen_local_policies
from .gia import InitialMedia, InvestigationRuntime
from .search.contracts import (
    CandidateBatch,
    CandidateSource,
    OpenedMaterial,
    OpenedSource,
    SearchQueryRequest,
)
from .search.policy import SearchPolicy
from .search.service import SearchService
from .store import GlobalStore


class _SmokeSearchProvider:
    """始终返回一个固定网页候选的离线搜索 Provider。"""

    async def search(
        self,
        request: SearchQueryRequest,
        *,
        offset: int,
        limit: int,
        image_path: Path | None = None,
    ) -> CandidateBatch:
        """按 offset/limit 模拟分页，忽略查询内容和图片。"""

        del request, image_path
        candidates = (
            CandidateSource(
                "https://example.test/foren-searcher-smoke",
                search_title="ForenSearcher controlled smoke source",
                search_snippet="Controlled evidence for the local end-to-end run.",
                provider="local-smoke",
                engine="fixture",
                provider_rank=1,
                candidate_type="page",
            ),
        )
        return CandidateBatch(
            "local-smoke",
            "fixture",
            candidates[offset : offset + limit],
            raw_response={"controlled": True},
        )


class _SmokeSourceOpener:
    """把固定候选打开成含明确日期的 HTML Evidence。"""

    async def open(self, candidate: CandidateSource, request: object) -> OpenedSource:
        """返回可被 READ_RECORD 和最终答案核验的确定性材料。"""

        del request
        statement = (
            "Controlled source statement: the ForenSearcher local-model smoke "
            "scenario is dated 2026-09-03."
        )
        material = OpenedMaterial(
            kind="page",
            status="ready",
            content=statement,
            artifact_bytes=f"<p>{statement}</p>".encode(),
            media_type="text/html",
            suffix=".html",
            modalities=("text",),
            label="Controlled smoke statement dated 2026-09-03",
            locator="paragraph 1",
            acquired_by="SEARCH/open/local-smoke",
            production_method="source_excerpt",
        )
        return OpenedSource(candidate.source_url, (material,))


def _smoke_image() -> bytes:
    """生成一张简单红色矩形 PNG，供 INSPECT crop 链路使用。"""

    image = Image.new("RGB", (320, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 40, 240, 140), fill=(220, 30, 30))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _successful_smoke_action(observation: dict[str, Any]) -> bool:
    """判断一条 HI observation 是否代表所需动作真实成功。"""

    action = observation.get("action")
    details = observation.get("details")
    if not isinstance(details, dict):
        return False
    if action == "SEARCH":
        return details.get("stop_reason") == "target_reached" and bool(
            details.get("results")
        )
    if action == "INSPECT":
        return details.get("status") in {"ready", "partial"}
    if action == "READ_RECORD":
        return (
            details.get("record_type") == "evidence"
            and isinstance(details.get("acquired_by"), str)
            and details["acquired_by"].startswith("SEARCH")
        )
    return False


async def _run_controlled_case(
    *,
    policies: LocalPolicyBundle,
    output_dir: str | Path,
    include_inspect: bool,
    natural: bool = False,
) -> dict[str, Any]:
    """使用已加载策略运行一个相互隔离的受控调查。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # 每个场景使用独立 Store，防止上一场景缓存让动作“虚假成功”。
    store = GlobalStore(destination / "store.sqlite3", destination / "artifacts")
    search = SearchService(
        store=store,
        provider=_SmokeSearchProvider(),
        opener=_SmokeSourceOpener(),  # type: ignore[arg-type]
        policy=SearchPolicy(
            target_sources=1,
            maximum_candidates=1,
            maximum_provider_requests=1,
            source_concurrency=1,
        ),
    )
    inspect = create_inspect_service(store=store)
    required = {"SEARCH", "READ_RECORD"}
    media: tuple[InitialMedia, ...]
    if natural:
        # 不向模型泄露动作名或参数，用于验证 HI 能从证据缺口自主选择外部检索。
        required = {"SEARCH"}
        task = (
            "核查材料中声称的发布日期。请寻找公开来源进行验证，"
            "给出能够确认的日期、依据和必要限制。"
        )
        media = ()
        scenario = "natural-language-search"
    elif include_inspect:
        # 多模态场景还要求真实执行一次 INSPECT crop。
        required.add("INSPECT")
        task = (
            "完成一次受控端到端核验。必须让一个 Workspace 依次完成："
            "对原始图像调用一次 INSPECT crop，参数 box=[0,0,1,1]；"
            "调用一次 SEARCH，使用文字查询 ForenSearcher smoke evidence，"
            "output_type=text；调用 READ_RECORD 读取 SEARCH 取得的文本 Evidence；"
            "最后提交有 Evidence 引用的报告。最终答案必须引用报告采用的 Evidence ID。"
        )
        media = (
            InitialMedia(
                source_ref="I0",
                media_type="image/png",
                modalities=("image",),
                label="原始红色矩形测试图像",
                original_filename="smoke.png",
                data=_smoke_image(),
            ),
        )
        scenario = "multimodal-tool-chain"
    else:
        task = (
            "完成一次纯文本受控端到端核验。必须让一个 Workspace 依次完成："
            "调用一次 SEARCH，使用文字查询 ForenSearcher smoke evidence，"
            "output_type=text；调用 READ_RECORD 读取 SEARCH 取得的文本 Evidence；"
            "最后提交有 Evidence 引用的报告。最终答案必须引用报告采用的 Evidence ID。"
        )
        media = ()
        scenario = "text-search-read"
    try:
        async with InvestigationRuntime(
            store=store,
            gia_policy=policies.gia_policy,
            hi_policy=policies.hi_policy,
            search_service=search,
            inspect_service=inspect,
            max_rounds=2,
            max_gia_turns=5,
            max_workspace_turns=10,
            max_concurrency=1,
        ) as runtime:
            result = await runtime.run_new(
                task=task,
                claim_materials={
                    "C0": {
                        "claim": "受控来源给出的日期是 2026-09-03。",
                        "scope": "local-model end-to-end smoke",
                    }
                },
                media=media,
            )
        events = store.trajectory_events(actor_role="HI")
        # 不只相信最终答案；从审计轨迹确认模型实际完成了每个必需动作。
        actions = tuple(
            event["payload"]["observation"].get("action")
            for event in events
            if event["event_type"] == "observation"
            and isinstance(event["payload"].get("observation"), dict)
            and event["payload"]["observation"].get("action")
            in {"SEARCH", "INSPECT", "READ_RECORD"}
            and _successful_smoke_action(event["payload"]["observation"])
        )
        missing = sorted(required.difference(actions))
        if missing:
            raise RuntimeError(
                "local model smoke did not execute required actions: " + ", ".join(missing)
            )
        evidence = store.evidence_records()
        searched = [item for item in evidence if item.acquired_by.startswith("SEARCH")]
        if not searched or not any(
            item.evidence_id in {value.evidence_id for value in result.cited_evidence}
            for item in searched
        ):
            raise RuntimeError("final answer did not cite SEARCH-derived Evidence")
        # 最终答案既要引用 SEARCH E，也要保留来源中唯一可核验日期。
        if "2026-09-03" not in result.decision.answer:
            raise RuntimeError("final answer did not preserve the verified smoke date")
        health = store.healthcheck()
        if health.status != "ok":
            raise RuntimeError(f"Store health check failed: {health.to_dict()}")
        return {
            "status": "passed",
            "scenario": scenario,
            "answer": result.decision.answer,
            "cited_evidence_ids": [item.evidence_id for item in result.cited_evidence],
            "actions": list(actions),
            "evidence_count": len(evidence),
            "rounds": len(store.round_records()),
            "store_health": health.to_dict(),
            "output_dir": str(destination.resolve()),
        }
    finally:
        # runtime 正常会关闭服务；异常发生在进入上下文前时由这里补偿关闭。
        if not getattr(search, "_closed", False):
            await search.aclose()
        if not getattr(inspect, "_closed", False):
            await inspect.aclose()


async def run_local_model_smoke(
    *, model_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """让两个本地模型跑通单个多模态控制场景。"""

    policies = create_qwen_local_policies(model_root, max_new_tokens=1_024)
    try:
        await policies.preload()
        result = await _run_controlled_case(
            policies=policies,
            output_dir=output_dir,
            include_inspect=True,
        )
        result["model_metrics"] = {
            "gia": policies.gia_model.metrics().to_dict(),
            "hi": policies.hi_model.metrics().to_dict(),
        }
        return result
    finally:
        await policies.aclose()


async def run_local_model_smoke_suite(
    *, model_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """复用已加载权重，分别记录三个隔离场景的结果。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policies = create_qwen_local_policies(model_root, max_new_tokens=1_024)
    try:
        await policies.preload()
        # 场景数据隔离，但模型权重复用，以节约加载时间和内存抖动。
        scenarios = []
        for name, include_inspect, natural in (
            ("multimodal-tool-chain", True, False),
            ("text-search-read", False, False),
            ("natural-language-search", False, True),
        ):
            started = perf_counter()
            try:
                scenario = await _run_controlled_case(
                    policies=policies, output_dir=destination / name,
                    include_inspect=include_inspect, natural=natural,
                )
            except Exception as exc:
                scenario = {
                    "status": "failed", "scenario": name,
                    "output_dir": str((destination / name).resolve()),
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            scenario["elapsed_seconds"] = perf_counter() - started
            scenarios.append(scenario)
        return {
            "status": "passed" if all(item["status"] == "passed" for item in scenarios) else "failed",
            "scenarios": scenarios,
            "model_metrics": {
                "gia": policies.gia_model.metrics().to_dict(),
                "hi": policies.hi_model.metrics().to_dict(),
            },
            "output_dir": str(destination.resolve()),
        }
    finally:
        await policies.aclose()


def main() -> None:
    """解析 CLI 参数、运行冒烟测试并写入 summary.json。"""

    parser = argparse.ArgumentParser(
        description="Run the controlled ForenSearcher local-model smoke test"
    )
    parser.add_argument("--model-root", default=".models")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run multimodal, text-only, and natural-language end-to-end scenarios",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        Path("outputs") / "local-model-smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run = run_local_model_smoke_suite if args.suite else run_local_model_smoke
    result = asyncio.run(run(model_root=args.model_root, output_dir=output_dir))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    # 终端打印便于 CI 读取，文件副本便于人工复核。
    Path(result["output_dir"], "summary.json").write_text(rendered, encoding="utf-8")
    print(rendered)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
