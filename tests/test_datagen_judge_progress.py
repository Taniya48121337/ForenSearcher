"""核验模型并发与进度：verify_anchored 的并发上限、结果顺序、CachedJudge 同 key 去重、进度计数。全部用假模型。"""

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen.verify import DictLookup
from foren_searcher.datagen.verify.anchored import PAIR_SYSTEM_PROMPTS, count_pair_calls, judge_concurrency, verify_anchored
from foren_searcher.datagen.verify.context import EvidenceText
from foren_searcher.datagen.verify.judges import CachedJudge, CallableJudge, ProgressJudge, judge_summary

SOURCES = {f"E{i}": f"Source number {i} says the bridge opened on 2023-05-0{i % 9 + 1} with {i * 10} cars." for i in range(1, 9)}


def make_lookup() -> DictLookup:
    table = {eid: EvidenceText(eid, text, source_url=f"https://example.org/{eid}") for eid, text in SOURCES.items()}
    table["E9"] = EvidenceText("E9", "", modalities=("image",), label="无像素的图片")     # needs_pixels，不调模型
    return DictLookup(table)


def make_report() -> dict[str, Any]:
    return {"findings": [
        {"finding_id": "F1", "statement": "Bridge opened with 10 cars.", "evidence_ids": ["E1", "E2", "E3", "E9"]},
        {"finding_id": "F2", "statement": "Bridge opened with 40 cars.", "evidence_ids": ["E4", "E5"]},
        {"finding_id": "F3", "statement": "Nothing here.", "evidence_ids": ["E6", "E7", "E8", "E1"]},
        {"finding_id": "F4", "statement": "Only a picture.", "evidence_ids": ["E9"]},
    ]}


class FakeModel:
    """记录并发峰值；pair 调用按来源给确定的支持/反驳，coverage 调用给确定的 unsupported。E7 永远坏输出。"""

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.calls: list[str] = []

    async def __call__(self, system_prompt: str, payload: dict[str, Any], images: list[Path]) -> dict[str, Any] | str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            # 让后发的调用先完成，打乱完成顺序，检验结果仍按原顺序拼回
            source = payload.get("source") or {}
            eid = source.get("evidence_id", "")
            await asyncio.sleep(self.delay * (1 + (hash(eid) % 3)))
            if system_prompt in PAIR_SYSTEM_PROMPTS:
                self.calls.append(f"pair:{payload['statement']}:{eid}")
                if eid == "E7":
                    return "not json at all"
                n = int(eid[1:])
                content = source["content"]
                quote = content[content.index("with"):]
                if n % 2:
                    return {"supported_parts": [{"part": f"{n * 10} cars", "quote": quote}], "contradicted_parts": [],
                            "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": eid}
                return {"supported_parts": [], "contradicted_parts": [{"part": "cars", "quote": quote, "hint": "sloppy_numeric"}],
                        "source_reports_claim": False, "statement_asserts_fact": True, "scope_overreach": False, "notes": eid}
            self.calls.append(f"coverage:{payload['statement']}")
            return {"unsupported_parts": ["opened"], "grounded": False, "notes": "cov"}
        finally:
            self.active -= 1


def run(coro):
    return asyncio.run(coro)


def serial_result(monkeypatch):
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "1")
    model = FakeModel()
    result = run(verify_anchored(make_report(), make_lookup(), CallableJudge(model)))
    assert model.peak == 1
    return result


def test_concurrency_env(monkeypatch):
    monkeypatch.delenv("FOREN_JUDGE_CONCURRENCY", raising=False)
    assert judge_concurrency() == 4
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "7")
    assert judge_concurrency() == 7
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "abc")
    assert judge_concurrency() == 4


def test_verify_anchored_concurrent_peak_and_order(monkeypatch):
    serial = serial_result(monkeypatch)
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "4")
    model = FakeModel()
    result = run(verify_anchored(make_report(), make_lookup(), CallableJudge(model)))
    assert 1 < model.peak <= 4
    # 产物与串行逐字一致：judgments / pending / coverage / errors
    assert json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) == json.dumps(serial.to_dict(), ensure_ascii=False, sort_keys=True)
    assert [(j.finding_id, j.evidence_id) for j in result.judgments] == [
        ("F1", "E1"), ("F1", "E2"), ("F1", "E3"), ("F1", "E9"), ("F2", "E4"), ("F2", "E5"),
        ("F3", "E6"), ("F3", "E7"), ("F3", "E8"), ("F3", "E1"), ("F4", "E9")]
    assert [(p["finding"], p.get("evidence_id"), p["reason"]) for p in result.pending] == [
        ("F1", "E9", "image_without_pixels"), ("F3", "E7", "judge_output_invalid"), ("F4", "E9", "image_without_pixels")]
    # 失败原因跟着那一条走（并发下不串）
    assert "LocalModelOutputError" in result.pending[1]["error"]
    assert [c["finding_id"] for c in result.coverage] == ["F1", "F2"]      # F3 有失败的 E → 不汇总；F4 没有可判 E
    # coverage 在该 Finding 的所有 pair 之后
    for fid, statement in (("F1", "Bridge opened with 10 cars."), ("F2", "Bridge opened with 40 cars.")):
        cov = model.calls.index(f"coverage:{statement}")
        pairs = [i for i, c in enumerate(model.calls) if c.startswith(f"pair:{statement}:")]
        assert pairs and max(pairs) < cov
    # 估算 12 = F1 3+1、F2 2+1、F3 4+1；实际 F3 因 E7 失败不做汇总（-1）但 E7 坏输出重试一次（+1），也是 12
    assert count_pair_calls(make_report(), make_lookup()) == 12 and len(model.calls) == 12


def test_concurrency_limit_two(monkeypatch):
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "2")
    model = FakeModel()
    run(verify_anchored(make_report(), make_lookup(), CallableJudge(model)))
    assert model.peak == 2


def test_semaphore_survives_multiple_event_loops(monkeypatch):
    # CLI 每个 run 调一次 asyncio.run；信号量不能绑死在第一个循环上
    monkeypatch.setenv("FOREN_JUDGE_CONCURRENCY", "2")
    for _ in range(3):
        model = FakeModel()
        run(verify_anchored(make_report(), make_lookup(), CallableJudge(model)))
        assert model.peak == 2


def test_cached_judge_dedupes_concurrent_misses(tmp_path):
    calls = []

    async def slow(system_prompt, payload, images):
        calls.append(payload["q"])
        await asyncio.sleep(0.02)
        return {"answer": payload["q"]}

    stream = io.StringIO()
    cached = CachedJudge(CallableJudge(slow), tmp_path / "cache.jsonl", progress=True, stream=stream)

    async def main():
        same = [cached.judge(system_prompt="s", payload={"q": "a"}, images=[]) for _ in range(5)]
        other = [cached.judge(system_prompt="s", payload={"q": "b"}, images=[])]
        return await asyncio.gather(*same, *other)

    answers = run(main())
    assert answers == [{"answer": "a"}] * 5 + [{"answer": "b"}]
    assert sorted(calls) == ["a", "b"]                         # 同 key 并发未命中只调一次
    assert (cached.misses, cached.hits, cached.failed, cached.calls) == (2, 4, 0, 6)
    lines = (tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2                                     # 缓存文件不重复写
    # 再跑一遍：全部命中文件缓存
    reloaded = CachedJudge(CallableJudge(slow), tmp_path / "cache.jsonl")
    run(reloaded.judge(system_prompt="s", payload={"q": "a"}, images=[]))
    assert (reloaded.hits, reloaded.misses, len(calls)) == (1, 0, 2)
    assert stream.getvalue().startswith("[judge] ")             # 第一条实际调用完成就打了进度
    assert "cached" in judge_summary(cached)


def test_cached_judge_failure_propagates_and_retries(tmp_path):
    attempts = []

    async def flaky(system_prompt, payload, images):
        attempts.append(1)
        await asyncio.sleep(0.01)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    cached = CachedJudge(CallableJudge(flaky), tmp_path / "c.jsonl")

    async def main():
        results = await asyncio.gather(*(cached.judge(system_prompt="s", payload={}, images=[]) for _ in range(3)),
                                       return_exceptions=True)
        again = await cached.judge(system_prompt="s", payload={}, images=[])
        return results, again

    results, again = run(main())
    assert all(isinstance(r, RuntimeError) for r in results)   # 在途失败传给等待者，不挂起
    assert again == {"ok": True} and len(attempts) == 2
    assert (cached.failed, cached.misses, cached.hits) == (1, 1, 2)
    assert not (tmp_path / "c.jsonl").read_text(encoding="utf-8").count("boom")


def test_progress_judge_counts_and_throttle(monkeypatch):
    monkeypatch.setenv("FOREN_JUDGE_PROGRESS_SECONDS", "0")

    async def answer(system_prompt, payload, images):
        await asyncio.sleep(0.005)
        return {"x": 1}

    stream = io.StringIO()
    judge = ProgressJudge(CallableJudge(answer), progress=True, stream=stream)

    async def main():
        await asyncio.gather(*(judge.judge(system_prompt="s", payload={"i": i}, images=[]) for i in range(6)))

    run(main())
    assert (judge.calls, judge.misses, judge.hits, judge.in_flight) == (6, 6, 0, 0)
    assert judge.last_seconds is not None and judge.busy_seconds >= 6 * 0.004
    assert len(stream.getvalue().splitlines()) == 6            # 节流 0 秒：每次实际调用一行
    assert judge.summary().startswith("[judge] 6 calls (0 cached) | last ")
    monkeypatch.setenv("FOREN_JUDGE_PROGRESS_SECONDS", "3600")
    stream2 = io.StringIO()
    judge2 = ProgressJudge(CallableJudge(answer), progress=True, stream=stream2)
    run(main2(judge2))
    assert len(stream2.getvalue().splitlines()) == 1           # 节流：一小时内只打第一行


async def main2(judge):
    await asyncio.gather(*(judge.judge(system_prompt="s", payload={"i": i}, images=[]) for i in range(5)))


def test_openrouter_judge_preloads_once_under_concurrency():
    from foren_searcher.datagen.verify.judges import OpenRouterJudge

    class FakeClient:
        model = "m"
        reasoning_effort = "low"

        def __init__(self):
            self.preloads = 0

        async def preload(self):
            self.preloads += 1
            await asyncio.sleep(0.01)

        async def complete(self, messages, **kwargs):
            await asyncio.sleep(0.01)
            return {"text": '{"ok": true}'}

    client = FakeClient()
    judge = OpenRouterJudge(client)

    async def main():
        return await asyncio.gather(*(judge.judge(system_prompt="s", payload={"i": i}, images=[]) for i in range(4)))

    assert run(main()) == [{"ok": True}] * 4
    assert client.preloads == 1


# ------------------------------------------------------------------ 重试与退避（09-25：并发下限流 / 瞬时错误让 pair 失败 → 注入句没人判）

@pytest.fixture
def sleeps(monkeypatch):
    """把 asyncio.sleep 换成只记时长、立即返回（退避不真等）。"""

    from foren_searcher.datagen.verify import anchored

    recorded: list[float] = []

    async def fake_sleep(delay, result=None):
        recorded.append(delay)
        return result

    monkeypatch.setattr(anchored.asyncio, "sleep", fake_sleep)
    return recorded


def test_call_retries_with_backoff_then_succeeds(sleeps):
    from foren_searcher.datagen.verify.anchored import LAST_CALL_ERROR, _call

    attempts = []

    async def flaky(system_prompt, payload, images):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RuntimeError("429")
        return {"ok": True}

    assert run(_call(CallableJudge(flaky), "s", {}, [])) == {"ok": True}
    assert len(attempts) == 3 and LAST_CALL_ERROR == {}
    assert len(sleeps) == 2 and 1.0 <= sleeps[0] < 2.0 and 3.0 <= sleeps[1] < 4.0     # 1s、3s + 0–1s 抖动


def test_call_gives_up_after_four_attempts_with_error_in_pending(sleeps):
    calls = []

    async def always_429(system_prompt, payload, images):
        calls.append(payload.get("source", {}).get("evidence_id"))
        raise RuntimeError("429 Too Many Requests from provider")

    report = {"findings": [{"finding_id": "F1", "statement": "Bridge opened with 10 cars.", "evidence_ids": ["E1"]}]}
    result = run(verify_anchored(report, make_lookup(), CallableJudge(always_429)))
    assert len(calls) == 4 and result.judgments[0].verdict == "failed" and result.coverage == []
    [pending] = result.pending
    assert pending["reason"] == "judge_output_invalid" and pending["attempts"] == 4
    assert pending["error"] == {"RuntimeError": "429 Too Many Requests from provider"}
    assert pending["last_error"] == "RuntimeError: 429 Too Many Requests from provider"
    assert len(sleeps) == 3 and [int(s) for s in sleeps] == [1, 3, 9]


def test_client_exhausted_rate_limit_is_not_retried_again(sleeps):
    # OpenRouterClient.complete 已按 retry_at 等过、用尽才抛：_call 不叠第二层等待
    from foren_searcher.datagen.verify.anchored import _call_checked
    from foren_searcher.openrouter import OpenRouterRateLimitError

    calls = []

    async def exhausted(system_prompt, payload, images):
        calls.append(1)
        raise OpenRouterRateLimitError("OpenRouter HTTP 429; automatic retries exhausted (2 retries)", request_sent=True)

    raw, failure = run(_call_checked(CallableJudge(exhausted), "s", {}, []))
    assert raw is None and len(calls) == 1 and failure.attempts == 1 and sleeps == []
    assert "OpenRouterRateLimitError" in failure.last_error


def test_progress_judge_reports_last_failure():
    async def boom(system_prompt, payload, images):
        raise RuntimeError("503 upstream provider unavailable " + "x" * 200)

    judge = ProgressJudge(CallableJudge(boom))
    with pytest.raises(RuntimeError):
        run(judge.judge(system_prompt="s", payload={}, images=[]))
    assert judge.failed == 1 and judge.last_failure is not None and len(judge.last_failure) == 80
    assert judge.last_failure.startswith("RuntimeError: 503 upstream")
    assert "1 failed" in judge.summary() and "last fail: RuntimeError: 503" in judge.summary()
    assert judge.stats()["last_failure"] == judge.last_failure


class BudgetExceeded(RuntimeError):
    """与 openrouter.BudgetExceeded 同名的假异常：_call_checked 按类名识别。"""


def test_budget_exceeded_raises_judge_budget_exhausted(sleeps):
    from foren_searcher.datagen.verify.anchored import JudgeBudgetExhausted, _call_checked

    calls = []

    async def broke(system_prompt, payload, images):
        calls.append(1)
        raise BudgetExceeded("OpenRouter cumulative budget exhausted: $40.0000; accounted $39.9143, next reservation $0.0864")

    with pytest.raises(JudgeBudgetExhausted, match=r"accounted \$39.9143"):
        run(_call_checked(CallableJudge(broke), "s", {}, []))
    assert len(calls) == 1 and sleeps == []      # 不重试、不退避
    # 真实的 openrouter.BudgetExceeded 也一样
    from foren_searcher.openrouter import BudgetExceeded as RealBudgetExceeded

    async def real(system_prompt, payload, images):
        raise RealBudgetExceeded("no credit")

    with pytest.raises(JudgeBudgetExhausted):
        run(_call_checked(CallableJudge(real), "s", {}, []))


def test_progress_judge_summary_shows_ledger_spend():
    class Ledger:
        def summary(self, *, request_ids=None):
            return {"budget_usd": 40.0, "cumulative_accounted_usd": 12.345}

    class Inner:
        client = type("Client", (), {"ledger": Ledger()})()

        async def judge(self, **kw):
            return {}

    assert "spent $12.35/$40.00" in CachedJudge(Inner(), Path("/nonexistent/cache.jsonl")).summary()
    assert "spent" not in ProgressJudge(CallableJudge(lambda *a: None)).summary()


def _budget_calibration(after_rows: int):
    from test_datagen_judged_units import CASES, RUN, VERDICTS, FakeJudge, _replay_from_base

    from foren_searcher.datagen.cases import load_cases
    from foren_searcher.datagen.verify.calibrate_recall import run_calibration
    from foren_searcher.datagen.verify.verdict import Verdict

    lines: list[str] = []
    done = {"rows": 0}

    def say(line: str) -> None:
        lines.append(line)
        if line.startswith("  ") and any(m in line for m in ("命中", "漏报", "未定")):
            done["rows"] += 1

    class Broke(FakeJudge):
        async def judge(self, *, system_prompt, payload, images, schema=None):
            if done["rows"] >= after_rows:
                raise BudgetExceeded("OpenRouter cumulative budget exhausted: $40.0000")
            return await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)

    base = [Verdict.from_dict(json.loads(l)) for l in VERDICTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = run(run_calibration([RUN], load_cases(CASES), base, Broke(_replay_from_base(base)), judge_name="fake",
                                  kinds=["append_clause", "swap_citation"], per_kind=1, seed=3, say=say))
    return summary, lines


def _needs_run():
    from test_datagen_judged_units import needs_run

    return needs_run


@_needs_run()
def test_calibration_aborts_on_budget_at_first_injection():
    summary, lines = _budget_calibration(0)
    assert summary["aborted"] == "budget_exhausted" and "budget exhausted" in summary["aborted_message"]
    assert summary["rows"] == [] and summary["n_injections"] == 0
    assert any(l.startswith("!!! OpenRouter 预算用尽") and "budget_usd" in l for l in lines)


@_needs_run()
def test_calibration_keeps_completed_rows_on_budget():
    full, _ = _budget_calibration(10 ** 6)
    assert "aborted" not in full and full["n_injections"] >= 2
    summary, lines = _budget_calibration(1)
    assert summary["aborted"] == "budget_exhausted"
    assert summary["n_injections"] == 1 and summary["rows"][0] == full["rows"][0]
    assert lines[-1].startswith("!!! OpenRouter 预算用尽")
