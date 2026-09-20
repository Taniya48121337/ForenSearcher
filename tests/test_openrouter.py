"""支付边界、密钥隔离与真实 Harness 消息的离线验证。"""

import asyncio
import base64
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from PIL import Image

from foren_searcher.demo import capability_report
from foren_searcher.openrouter import (
    BudgetExceeded, OpenRouterClient, OpenRouterError, OpenRouterInspectBackend,
    OpenRouterJsonGenerator, UsageLedger,
)
from foren_searcher.openrouter_settings import DEFAULT_MODEL, load_api_key, save_settings
from foren_searcher.inspect.backends.base import MediaProfileData, PreparedMedia
from foren_searcher.inspect.contracts import VisualQaParams
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.local_models import LocalModelOutputError
from foren_searcher.store import GlobalStore


MODEL = {"id": DEFAULT_MODEL, "context_length": 262144,
         "supported_parameters": ["reasoning", "response_format", "structured_outputs"],
         "architecture": {"input_modalities": ["text", "image"]},
         "pricing": {"prompt": "0.0000001", "completion": "0.00000036"}}


@pytest.fixture
def retry_clock(monkeypatch):
    import foren_searcher.openrouter as remote
    clock, waits = [100.0], []
    monkeypatch.setattr(remote, "monotonic", lambda: clock[0])
    async def sleep(delay):
        waits.append(delay)
        clock[0] += delay
        await asyncio.sleep(0)
    monkeypatch.setattr(remote, "_retry_sleep", sleep)
    return clock, waits


@pytest.mark.parametrize('prompt,completion,accepted', [
    ('0.00000099', '0.00000149', True),
    ('9.9e-7', '1.49e-6', True),
    ('0.0000009900000000000000000000001', '0.00000149', False),
    ('0.00000099', '0.000001490000000000000000000001', False),
])
def test_decimal_price_caps_accept_exact_rate_and_reject_any_excess(make_client, prompt, completion, accepted):
    client = make_client()
    client.input_price_cap = .99
    client.output_price_cap = 1.49
    try:
        if accepted:
            client._check_prices({'prompt': prompt, 'completion': completion})
        else:
            with pytest.raises(OpenRouterError, match='unit-price caps'):
                client._check_prices({'prompt': prompt, 'completion': completion})
    finally:
        client._close()


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-for-output")
    def create(handler=None, budget=1.0, reasoning_effort="low"):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"model": DEFAULT_MODEL, "budget_usd": budget, "reasoning_effort": reasoning_effort}), encoding="utf-8")
        def route(request):
            assert request.url.host == "openrouter.ai"
            if request.method == "GET":
                if request.url.path.endswith('/endpoints'):
                    return httpx.Response(200, json={'data': {'endpoints': [
                        {'tag': 'deepinfra', 'quantization': 'fp8', 'pricing': MODEL['pricing'],
                         'supported_parameters': MODEL['supported_parameters']}]}})
                return httpx.Response(200, json={"data": [MODEL] if request.url.path.endswith("/models") else {"limit_remaining": 1}})
            if handler:
                return handler(request)
            return httpx.Response(200, json={"id": "generation-example", "model": DEFAULT_MODEL, "provider": "example",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"decision":"FINISH","answer":"unknown"}'}}],
                "usage": {"cost": 0.002, "prompt_tokens": 30, "completion_tokens": 20}})
        return OpenRouterClient(config, transport=httpx.MockTransport(route))
    return create


@pytest.mark.asyncio
async def test_restored_usage_keeps_unknown_reservation_and_includes_late_cost(make_client):
    first = make_client()
    await first.preload()
    await first.complete([{'role':'user', 'content':'test'}], role='HI')
    pending = first.ledger.reserve(.15, model=first.model, role='GIA')
    first._request_ids.add(pending)
    previous = first.usage_report()
    await first.aclose()
    second = make_client()
    try:
        second.restore_usage(previous)
        await second.preload()
        await second.complete([{'role':'user', 'content':'test'}], role='GIA')
        usage = second.usage_report()
        assert usage['request_count'] == 3
        assert usage['actual_cost_usd'] == pytest.approx(.004)
        assert usage['unsettled_reserved_usd'] == pytest.approx(.15)
        second.ledger.settle(pending, {'usage':{'cost':.003}})
        second.restore_usage(previous)
        assert second.usage_report()['actual_cost_usd'] == pytest.approx(.007)
        assert second.usage_report()['unsettled_reserved_usd'] == 0
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_restore_rejects_different_ledger_and_mismatched_legacy_request(make_client, tmp_path):
    client = make_client()
    try:
        identifier = client.ledger.reserve(.1, model=client.model, role='GIA')
        previous = client.ledger.summary(request_ids={identifier})
        with pytest.raises(ValueError, match='original usage ledger'):
            client.restore_usage({**previous, 'ledger_path':str(tmp_path / 'another.sqlite3')})
        previous['requests'][0]['model'] = 'another model'
        with pytest.raises(ValueError, match='does not belong'):
            client.restore_usage(previous)
        assert client.usage_report()['request_count'] == 0
    finally:
        await client.aclose()


def test_concurrent_and_reopened_budget_reservations(tmp_path):
    path = tmp_path / "usage.sqlite3"
    ledger = UsageLedger(path, 0.1)
    def reserve(_):
        try:
            return UsageLedger(path, 0.1).reserve(0.06, model="model", role="HI")
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(reserve, range(8)))
    assert sum(value is not None for value in ids) == 1
    request_id = next(value for value in ids if value is not None)
    ledger.settle(request_id, {"usage": {"cost": 0.001}})
    UsageLedger(path, 0.1).reserve(0.06, model="model", role="HI")
    assert ledger.summary()["cumulative_accounted_usd"] == pytest.approx(0.061)


def test_reconciliation_preserves_unknown_details_and_caps_concurrent_new_spend(tmp_path):
    path = tmp_path / "usage.sqlite3"
    ledger = UsageLedger(path, 10)
    known = ledger.reserve(8.4, model="model", role="GIA")
    ledger.settle(known, {"usage": {"cost": 8.4}})
    unknown = ledger.reserve(1.54, model="model", role="HI")
    ledger.settle(unknown, {"id": "gen-late"})
    before = ledger.summary()["requests"]
    ledger.reconcile(through_request_id=unknown, spend_usd=8.44, note="User confirmed cumulative spend")
    report = UsageLedger(path, 10).summary()
    assert report["requests"] == before
    assert report["remaining_usd"] == pytest.approx(1.56)
    assert report["unsettled_reserved_usd"] == 1.54
    assert report["active_budget_reserved_usd"] == 0
    def reserve(_):
        try:
            return UsageLedger(path, 10).reserve(.8, model="model", role="HI")
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(reserve, range(8)))
    assert sum(value is not None for value in ids) == 1
    current = next(value for value in ids if value is not None)
    report = ledger.summary(request_ids={current})
    assert report["request_count"] == 1
    assert report["active_budget_reserved_usd"] == .8
    assert report["cumulative_accounted_usd"] == pytest.approx(9.24)
    ledger.settle(current, {"usage": {"cost": .02}})
    ledger.settle(unknown, {"usage": {"cost": .01}})
    assert ledger.summary()["cumulative_accounted_usd"] == pytest.approx(8.46)
    # 历史已知费用超出对账总额时不能被对账记录掩盖。
    ledger.settle(unknown, {"usage": {"cost": 1.59}})
    assert ledger.summary()["cumulative_accounted_usd"] == pytest.approx(10.01)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(.001, model="model", role="GIA")


def test_reconciliation_rejects_stale_boundary_and_cannot_be_applied_twice(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite3", 10)
    first = ledger.reserve(.1, model="model", role="HI")
    last = ledger.reserve(.1, model="model", role="HI")
    with pytest.raises(ValueError, match="boundary changed"):
        ledger.reconcile(through_request_id=first, spend_usd=.1, note="Confirmed total")
    assert ledger.summary()["reconciliation"] is None
    ledger.reconcile(through_request_id=last, spend_usd=.1, note="Confirmed total")
    ledger.reserve(.1, model="model", role="HI")
    with pytest.raises(ValueError, match="already been reconciled"):
        ledger.reconcile(through_request_id=last, spend_usd=.1, note="Confirmed total")
    assert ledger.summary()["cumulative_accounted_usd"] == pytest.approx(.2)


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), True, "8.44", .01])
def test_reconciliation_rejects_invalid_or_understated_spend(tmp_path, amount):
    ledger = UsageLedger(tmp_path / "usage.sqlite3", 10)
    request_id = ledger.reserve(.1, model="model", role="HI")
    ledger.settle(request_id, {"usage": {"cost": .02}})
    before = ledger.summary()
    with pytest.raises(ValueError):
        ledger.reconcile(through_request_id=request_id, spend_usd=amount, note="Confirmed total")
    assert ledger.summary() == before


@pytest.mark.parametrize("cost", [None, -1, float("nan"), float("inf"), True, "0"])
def test_uncertain_usage_does_not_release_reservation(tmp_path, cost):
    ledger = UsageLedger(tmp_path / "usage.sqlite3", 0.1)
    request_id = ledger.reserve(0.06, model="model", role="HI")
    ledger.settle(request_id, {"usage": {"cost": cost}})
    assert ledger.summary()["unsettled_reserved_usd"] == 0.06
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.06, model="model", role="GIA")


@pytest.mark.parametrize("cost", [None, .004])
def test_rejection_does_not_overwrite_an_existing_usage_response(tmp_path, cost):
    ledger = UsageLedger(tmp_path / "usage.sqlite3", .1)
    request_id = ledger.reserve(.06, model="model", role="HI")
    ledger.settle(request_id, {"usage": {"cost": cost}})
    before = ledger.summary()
    ledger.reject(request_id)
    assert ledger.summary() == before


@pytest.mark.asyncio
async def test_usage_contains_actual_cost_and_no_key(make_client):
    client = make_client()
    await client.preload()
    await client.complete([{"role": "user", "content": "one"}], role="GIA")
    other = make_client()
    await other.preload()
    await other.complete([{"role": "user", "content": "two"}], role="HI")
    report = client.usage_report()
    assert report["request_count"] == 1
    assert report["actual_cost_usd"] == 0.002
    assert report["cumulative_accounted_usd"] == 0.004
    assert "test-secret" not in json.dumps(report)
    await client.aclose()
    await other.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('reset,retry,expected', [
    ('1788652800000', '60', 'retry_after_seconds=60'),
    ('1788652800', 'Sun, 06 Sep 2026 00:00:00 GMT', 'retry_after=2026-09-06T00:00:00+00:00'),
])
async def test_single_attempt_rate_limit_preserves_headers_and_budget(make_client, reset, retry, expected):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'X-RateLimit-Limit': '100', 'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': reset, 'Retry-After': retry}, text='test-secret-not-for-output')
    client = make_client(handler)
    await client.preload()
    with pytest.raises(OpenRouterError) as caught:
        await asyncio.to_thread(client._complete, [], 'HI', False)
    message = str(caught.value)
    assert 'limit=100; remaining=0; reset=2026-09-06T00:00:00+00:00' in message
    assert expected in message and 'test-secret' not in message
    assert len(calls) == 1
    assert client.usage_report()['unsettled_reserved_usd'] > 0
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', ['test-secret-not-for-output', '9' * 90])
async def test_invalid_rate_limit_headers_do_not_enter_error_message(make_client, value):
    client = make_client(lambda request: httpx.Response(429,
        headers={k:value for k in ('x-ratelimit-limit', 'x-ratelimit-remaining', 'x-ratelimit-reset', 'retry-after')}))
    await client.preload()
    with pytest.raises(OpenRouterError) as caught:
        await asyncio.to_thread(client._complete, [], 'HI', False)
    assert str(caught.value) == 'OpenRouter HTTP 429'
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "redirect", "server", "truncated"])
async def test_failed_paid_call_is_not_retried_or_assumed_free(make_client, failure):
    calls = []
    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("test-secret-not-for-output", request=request)
        if failure == "truncated":
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "partial"}}], "usage": {"cost": 0.003}})
        return httpx.Response(302 if failure == "redirect" else 500,
                              headers={"Location": "https://example.org/leak"}, text="test-secret-not-for-output")
    client = make_client(handler)
    await client.preload()
    with pytest.raises(OpenRouterError) as caught:
        await client.complete([{"role": "user", "content": "request"}], role="HI")
    assert "test-secret" not in str(caught.value)
    assert len(calls) == 1
    usage = client.usage_report()
    assert usage["actual_cost_usd"] == (0.003 if failure == "truncated" else 0)
    assert (usage["unsettled_reserved_usd"] > 0) == (failure != "truncated")
    await client.aclose()


@pytest.mark.asyncio
async def test_bad_request_releases_only_its_reservation_and_allows_corrected_request(make_client):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(400, headers={"X-Generation-Id":"gen-rejected-schema"},
                json={"error":{"code":400,"message":"test-secret-not-for-output",
                               "metadata":{"provider_name":"Mistral","raw":"Invalid structured output syntax"}}})
        return httpx.Response(200, json={"choices":[{"finish_reason":"stop","message":{"content":"ok"}}],
                                        "usage":{"cost":.002}})

    client = make_client(handler)
    await client.preload()
    unknown = client.ledger.reserve(.05, model="earlier-model", role="HI")
    paid = client.ledger.reserve(.05, model="earlier-model", role="HI")
    client.ledger.settle(paid, {"usage":{"cost":.004}})
    with pytest.raises(OpenRouterError, match="HTTP 400") as caught:
        await client.complete([], role="GIA")
    assert "test-secret" not in str(caught.value)
    assert len(calls) == 1
    report = client.usage_report()
    assert report["unsettled_reserved_usd"] == 0
    rejected = report["requests"][0]
    assert rejected["state"] == "rejected" and rejected["cost_usd"] == 0
    assert rejected["generation_id"] == "gen-rejected-schema"
    assert (await client.complete([], role="GIA"))["text"] == "ok"
    assert len(calls) == 2
    rows = {row["id"]:row for row in client.ledger.summary()["requests"]}
    assert rows[unknown]["cost_usd"] is None and rows[unknown]["reserved_usd"] == .05
    assert rows[paid]["cost_usd"] == .004
    assert client.ledger.summary()["cumulative_accounted_usd"] == pytest.approx(.056)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", [400, 404])
@pytest.mark.parametrize("cost", [None, .004])
async def test_success_http_with_error_body_preserves_actual_or_unknown_cost(make_client, error_code, cost):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"error":{"code":error_code,"message":"test-secret-not-for-output"},
                                        "usage":{"cost":cost}})
    client = make_client(handler)
    await client.preload()
    with pytest.raises(OpenRouterError, match="completion failed") as caught:
        await client.complete([], role="HI")
    assert "test-secret" not in str(caught.value) and len(calls) == 1
    report = client.usage_report()
    row = report["requests"][0]
    assert row["cost_usd"] == cost and row["state"] != "rejected"
    assert report["unsettled_reserved_usd"] == (row["reserved_usd"] if cost is None else 0)
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_inference_route_stops_the_session_before_more_reservations(make_client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, text="test-secret-not-for-output")

    client = make_client(handler)
    await client.preload()
    prior = client.ledger.reserve(.05, model="earlier-model", role="HI")
    with pytest.raises(OpenRouterError, match="HTTP 404"):
        await client.complete([], role="INSPECT/visual_qa")
    before = client.usage_report()
    for role in ("HI", "GIA", "INSPECT/visual_qa"):
        with pytest.raises(OpenRouterError, match="start a new run") as caught:
            await client.complete([], role=role)
        assert "test-secret" not in str(caught.value)
    assert len(calls) == 1
    assert client.usage_report() == before
    assert before["unsettled_reserved_usd"] == 0
    assert before["requests"][0]["state"] == "rejected"
    assert before["requests"][0]["cost_usd"] == 0
    assert client.ledger.summary()["unsettled_reserved_usd"] == .05
    assert client._request("GET", "key")["data"]["limit_remaining"] == 1
    await client.aclose()
    other = make_client()
    await other.preload()
    await other.complete([], role="GIA")
    assert other.usage_report()["actual_cost_usd"] == .002
    assert other.ledger.summary()["unsettled_reserved_usd"] == .05
    assert other.ledger.summary()["requests"][0]["id"] == prior
    await other.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500])
async def test_transient_http_failure_does_not_disable_the_session(make_client, monkeypatch, status):
    import foren_searcher.openrouter as remote
    clock = [100.0]
    monkeypatch.setattr(remote, "monotonic", lambda: clock[0])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="test-secret-not-for-output")

    client = make_client(handler)
    await client.preload()
    for _ in range(2):
        with pytest.raises(OpenRouterError, match=f"HTTP {status}"):
            await asyncio.to_thread(client._complete, [], "HI", False)
        if status == 429:
            clock[0] += 30
    assert len(calls) == 2
    assert client.usage_report()["request_count"] == 2
    assert client.usage_report()["unsettled_reserved_usd"] > 0
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,seconds", [({}, 30), ({"Retry-After": "90"}, 90)])
async def test_rate_limit_stops_followup_reservations_and_allows_recovery(make_client, monkeypatch, headers, seconds):
    import foren_searcher.openrouter as remote
    clock = [100.0]
    monkeypatch.setattr(remote, "monotonic", lambda: clock[0])
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers=headers, text="test-secret-not-for-output")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                                        "usage": {"cost": .002}})

    client = make_client(handler)
    await client.preload()
    prior = client.ledger.reserve(.03, model="earlier-model", role="HI")
    with pytest.raises(OpenRouterError, match="HTTP 429"):
        await asyncio.to_thread(client._complete, [], "INSPECT/visual_qa", False)
    before = client.usage_report()
    assert before["request_count"] == 1 and before["unsettled_reserved_usd"] > 0
    for role in ("INSPECT/visual_qa", "HI", "GIA"):
        with pytest.raises(OpenRouterError, match="no request was sent or budget reserved") as caught:
            await asyncio.to_thread(client._complete, [], role, False)
        assert "test-secret" not in str(caught.value)
    assert client.usage_report() == before and len(calls) == 1
    assert client._request("GET", "key")["data"]["limit_remaining"] == 1
    other = make_client()
    await other.preload()
    assert (await other.complete([], role="GIA"))["text"]
    await other.aclose()
    clock[0] += seconds - .1
    with pytest.raises(OpenRouterError, match="retry after 1 seconds"):
        await asyncio.to_thread(client._complete, [], "HI", False)
    clock[0] += .1
    assert (await asyncio.to_thread(client._complete, [], "HI", False))["text"] == "ok"
    report = client.usage_report()
    assert len(calls) == report["request_count"] == 2
    assert report["actual_cost_usd"] == .002
    assert report["unsettled_reserved_usd"] == before["unsettled_reserved_usd"]
    assert client.ledger.summary()["requests"][0]["id"] == prior
    assert client.ledger.summary()["requests"][0]["cost_usd"] is None
    await client.aclose()


@pytest.mark.parametrize("headers,expected", [
    ({"Retry-After": "Tue, 08 Sep 2026 00:02:00 GMT"}, 120),
    ({"X-RateLimit-Reset": "1788825720"}, 120),
    ({"X-RateLimit-Reset": "1788825720000"}, 120),
    ({"Retry-After": "10", "X-RateLimit-Reset": "1788825720"}, 120),
    ({"Retry-After": "0"}, 0),
    ({"Retry-After": "Mon, 07 Sep 2026 23:59:00 GMT"}, 0),
    ({"Retry-After": "test-secret", "X-RateLimit-Reset": "9" * 90}, 30),
])
def test_rate_limit_wait_uses_server_time_formats(headers, expected):
    from datetime import UTC, datetime
    from foren_searcher.openrouter import _rate_limit_delay
    assert _rate_limit_delay(httpx.Headers(headers), datetime(2026, 9, 8, tzinfo=UTC)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("cost", [None, .004])
async def test_rate_limit_in_success_body_keeps_billing_and_defers_new_calls(make_client, monkeypatch, cost):
    import foren_searcher.openrouter as remote
    monkeypatch.setattr(remote, "monotonic", lambda: 100.0)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, headers={"Retry-After": "90"}, json={"error": {"code": 429}, "usage": {"cost": cost}})

    client = make_client(handler)
    await client.preload()
    with pytest.raises(OpenRouterError, match="completion failed"):
        await asyncio.to_thread(client._complete, [], "HI", False)
    before = client.usage_report()
    with pytest.raises(OpenRouterError, match="retry after 90 seconds; no request was sent or budget reserved"):
        await asyncio.to_thread(client._complete, [], "HI", False)
    assert client.usage_report() == before and len(calls) == 1
    assert before["requests"][0]["cost_usd"] == cost
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_read_only_record_does_not_disable_inference(make_client):
    client = make_client()
    await client.preload()
    original = client._http

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                                       "usage": {"cost": .002}})

    client._http = httpx.Client(base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(handler))
    original.close()
    with pytest.raises(OpenRouterError, match="HTTP 404"):
        client._request("GET", "generation", params={"id": "missing-generation"})
    assert (await client.complete([], role="GIA"))["text"] == "ok"
    assert client.usage_report()["request_count"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_budget_stops_before_network(make_client):
    client = make_client(lambda request: pytest.fail("paid request should not be sent"), budget=0.001)
    await client.preload()
    with pytest.raises(BudgetExceeded):
        await client.complete([], role="GIA")
    assert client.usage_report()["request_count"] == 0
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
async def test_unsent_model_request_releases_only_its_reservation_after_bounded_retry(make_client, error):
    calls = []
    def handler(request):
        calls.append(request)
        raise error("test-secret-not-for-output", request=request)
    client = make_client(handler)
    await client.preload()
    prior = client.ledger.reserve(.05, model="previous", role="HI")
    try:
        with pytest.raises(OpenRouterError, match="request was not sent after two connection attempts") as caught:
            await client.complete([], role="GIA")
        assert "test-secret" not in str(caught.value)
        assert len(calls) == 2
        row, = client.usage_report()["requests"]
        assert row["state"] == "rejected" and row["cost_usd"] == 0
        previous = next(r for r in client.ledger.summary()["requests"] if r["id"] == prior)
        assert previous["cost_usd"] is None and previous["reserved_usd"] == .05
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_real_refused_connection_does_not_consume_next_inference_budget(make_client):
    import socket
    client = make_client(budget=.06)
    await client.preload()
    original = client._http
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        # 占用但不监听，实际 TCP 连接无法建立；不接触付费服务。
        port = endpoint.getsockname()[1]
        client._http = httpx.Client(base_url=f"http://127.0.0.1:{port}/", timeout=.25, trust_env=False)
        try:
            with pytest.raises(OpenRouterError, match="request was not sent after two connection attempts"):
                await client.complete([], role="GIA")
        finally:
            client._http.close()
            client._http = original
    try:
        assert client.usage_report()["cumulative_accounted_usd"] == 0
        await client.complete([], role="GIA")
        assert client.usage_report()["actual_cost_usd"] == .002
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
async def test_unsent_retry_preserves_payload_and_settles_one_request(make_client, error):
    calls = []
    def handler(request):
        calls.append(request.content)
        if len(calls) == 1:
            raise error("temporary connection failure", request=request)
        return httpx.Response(200, json={"id": "gen-reconnected", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"decision":"FINISH","answer":"done"}'}}],
            "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    try:
        result = await client.complete([{"role": "user", "content": "original task"}], role="GIA")
        assert result["generation_id"] == "gen-reconnected"
        assert len(calls) == 2 and calls[0] == calls[1]
        usage = client.usage_report()
        row, = usage["requests"]
        assert row["generation_id"] == "gen-reconnected" and row["cost_usd"] == .002
        assert usage["unsettled_reserved_usd"] == 0
        assert usage["cumulative_accounted_usd"] == .002
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_uncertain_failure_after_unsent_retry_retains_reservation(make_client):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("temporary connection failure", request=request)
        raise httpx.ReadTimeout("test-secret-not-for-output", request=request)
    client = make_client(handler)
    await client.preload()
    try:
        with pytest.raises(OpenRouterError, match="billing may be uncertain") as caught:
            await client.complete([], role="HI")
        assert "test-secret" not in str(caught.value) and len(calls) == 2
        row, = client.usage_report()["requests"]
        assert row["cost_usd"] is None
        assert client.usage_report()["unsettled_reserved_usd"] == row["reserved_usd"] > 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadError, httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout])
async def test_request_io_failure_keeps_uncertain_reservation_without_retry(make_client, error):
    calls = []
    def handler(request):
        calls.append(request)
        raise error("test-secret-not-for-output", request=request)
    client = make_client(handler)
    await client.preload()
    try:
        with pytest.raises(OpenRouterError, match="billing may be uncertain"):
            await client.complete([], role="GIA")
        row, = client.usage_report()["requests"]
        assert row["cost_usd"] is None
        assert client.usage_report()["unsettled_reserved_usd"] == row["reserved_usd"] > 0
        assert len(calls) == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_connection_error_after_response_starts_does_not_release_cost(make_client):
    class BrokenResponse(httpx.SyncByteStream):
        def __iter__(self):
            yield b" "
            raise httpx.ConnectError("test-secret-not-for-output")
    client = make_client(lambda request: httpx.Response(200, stream=BrokenResponse(),
        headers={"X-Generation-Id":"gen-response-started"}))
    await client.preload()
    try:
        with pytest.raises(OpenRouterError, match="billing may be uncertain"):
            await client.complete([], role="GIA")
        row, = client.usage_report()["requests"]
        assert row["generation_id"] == "gen-response-started" and row["cost_usd"] is None
        assert client.usage_report()["unsettled_reserved_usd"] == row["reserved_usd"] > 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_preload_retries_only_read_only_transport_failure(make_client):
    client = make_client()
    original = client._http
    gets = []
    def route(request):
        assert request.method == "GET"
        gets.append(request.url.path)
        if len(gets) == 1:
            raise httpx.ConnectError("temporary network error", request=request)
        return httpx.Response(200, json={"data": [MODEL] if request.url.path.endswith("/models") else {"limit_remaining": 1}})
    client._http = httpx.Client(base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(route))
    original.close()
    await client.preload()
    assert gets == ["/api/v1/models", "/api/v1/models", "/api/v1/key"]
    assert client.usage_report()["request_count"] == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_heartbeat_bytes_cannot_extend_paid_call_indefinitely(make_client, monkeypatch):
    import foren_searcher.openrouter as remote
    elapsed = [0.0]
    monkeypatch.setattr(remote, "monotonic", lambda: elapsed[0])
    class Heartbeat(httpx.SyncByteStream):
        closed = False
        chunks = 0
        def __iter__(self):
            for _ in range(100):
                elapsed[0] += 20
                self.chunks += 1
                yield b" "
        def close(self):
            self.closed = True
    stream = Heartbeat()
    client = make_client(lambda request: httpx.Response(200, stream=stream,
        headers={"X-Generation-Id": "gen-heartbeat-timeout"}))
    await client.preload()
    with pytest.raises(OpenRouterError, match="deadline"):
        await client.complete([], role="GIA")
    assert stream.closed and stream.chunks == 10
    assert client.usage_report()["request_count"] == 1
    assert client.usage_report()["unsettled_reserved_usd"] > 0
    assert client.usage_report()["requests"][0]["generation_id"] == "gen-heartbeat-timeout"
    await client.aclose()
    assert client._closed and client._active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read_timeout", "invalid_json", "http_500", "missing_usage"])
async def test_generation_header_survives_incomplete_response_without_releasing_cost(make_client, failure):
    calls = []

    class BrokenResponse(httpx.SyncByteStream):
        def __iter__(self):
            yield b" "
            raise httpx.ReadTimeout("test-secret-not-for-output")

    def handler(request):
        calls.append(request)
        headers = {"X-Generation-Id": "gen-retained-123", "X-Unrelated": "test-secret-not-for-output"}
        if failure == "read_timeout":
            return httpx.Response(200, headers=headers, stream=BrokenResponse())
        if failure == "invalid_json":
            return httpx.Response(200, headers=headers, content=b"incomplete")
        if failure == "http_500":
            return httpx.Response(500, headers=headers)
        return httpx.Response(200, headers=headers, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    client = make_client(handler)
    await client.preload()
    if failure == "missing_usage":
        assert (await client.complete([], role="HI"))["text"] == "ok"
    else:
        with pytest.raises(OpenRouterError) as caught:
            await client.complete([], role="HI")
        assert "test-secret" not in str(caught.value)
    usage = client.usage_report()
    row = usage["requests"][0]
    assert row["generation_id"] == "gen-retained-123"
    assert row["cost_usd"] is None and usage["unsettled_reserved_usd"] == row["reserved_usd"]
    assert len(calls) == 1 and "ledger_request_id" not in json.loads(calls[0].content)
    assert "test-secret" not in json.dumps(usage)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["", "test-secret-not-for-output", "gen-" + "x" * 201, "gen-invalid/url"])
async def test_invalid_generation_header_is_not_persisted(make_client, header):
    client = make_client(lambda request: httpx.Response(500, headers={"X-Generation-Id": header}))
    await client.preload()
    with pytest.raises(OpenRouterError):
        await client.complete([], role="GIA")
    assert client.usage_report()["requests"][0]["generation_id"] is None
    await client.aclose()


def test_record_generation_header_does_not_overwrite_existing_identity_or_billing(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.sqlite3", 1)
    pending = ledger.reserve(.1, model="model", role="HI")
    settled = ledger.reserve(.1, model="model", role="GIA")
    rejected = ledger.reserve(.1, model="model", role="HI")
    ledger.record_generation_id(pending, "gen-original")
    ledger.record_generation_id(pending, "gen-other")
    ledger.settle(settled, {"id": "gen-settled", "usage": {"cost": .01}})
    ledger.record_generation_id(settled, "gen-wrong")
    ledger.reject(rejected)
    ledger.record_generation_id(rejected, "gen-rejected")
    rows = ledger.summary()["requests"]
    assert [row["generation_id"] for row in rows] == ["gen-original", "gen-settled", None]
    assert [row["cost_usd"] for row in rows] == [None, .01, 0]


@pytest.mark.asyncio
async def test_generation_header_is_bound_to_each_concurrent_inference_request(make_client):
    barrier = threading.Barrier(2)
    def handler(request):
        label = json.loads(request.content)["messages"][0]["content"]
        barrier.wait(5)
        return httpx.Response(200, headers={"X-Generation-Id": "gen-" + label},
            json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    await asyncio.gather(*(client.complete([{"role": "user", "content": role}], role=role) for role in ("GIA", "HI")))
    assert {row["role"]: row["generation_id"] for row in client.usage_report()["requests"]} == {"GIA": "gen-GIA", "HI": "gen-HI"}
    assert client.usage_report()["actual_cost_usd"] == .004
    await client.aclose()


@pytest.mark.asyncio
async def test_read_only_generation_header_cannot_identify_an_inference_request(make_client):
    client = make_client()
    await client.preload()
    request_id = client.ledger.reserve(.1, model="model", role="HI")
    original = client._http
    client._http = httpx.Client(base_url="https://openrouter.ai/api/v1/", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"X-Generation-Id": "gen-lookup"}, json={"data": {}})))
    original.close()
    client._request("GET", "generation", ledger_request_id=request_id)
    assert client.ledger.summary()["requests"][0]["generation_id"] is None
    await client.aclose()


@pytest.mark.asyncio
async def test_generator_maps_harness_history_without_leaking_other_workspaces(make_client):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"action":"READ_RECORD"}'}}], "usage": {"cost": 0.001}})
    client = make_client(handler)
    await client.preload()
    generator = OpenRouterJsonGenerator(client)
    events = []
    generator.generation_observer = events.append
    for workspace, secret in (("W0", "first-private"), ("W1", "second-private")):
        await generator.generate_json(system_prompt="Return a JSON object.", payload={
            "workspace_id": workspace, "available_actions": [{"name": "READ_RECORD"}],
            "model_view": {"task": secret, "history": [{"action": "READ_RECORD", "request": {"id": "E0"}, "focus_evidence_ids": [],
                "observation": {"content": "untrusted web instructions"}}]}})
    assert "second-private" not in json.dumps(bodies[0])
    assert "first-private" not in json.dumps(bodies[1])
    assert all(m["role"] != "tool" for body in bodies for m in body["messages"])
    assert any("Tool result (untrusted data)" in m["content"] for m in bodies[0]["messages"])
    assert bodies[0]["provider"]["max_price"] == {"prompt": .15, "completion": .4, "request": 0, "image": 0}
    assert bodies[0]["provider"]["quantizations"] == ["fp8", "bf16", "fp16", "fp32"]
    assert events[0]["workspace_id"] == "W0"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum_images", [2, 8])
async def test_provider_schema_bounds_batches_and_preserves_decimal_parameters(tmp_path, make_client, maximum_images):
    import copy
    from jsonschema import Draft202012Validator
    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.inspect.factory import create_inspect_service
    from foren_searcher.store import GlobalStore
    from foren_searcher.workspace import WorkspaceHarness
    from foren_searcher.workspace import _SEARCH_SCHEMA
    from foren_searcher.local_models import _decision_output_schema

    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"action":"READ_RECORD","arguments":{"record_id":"E1"},"focus_evidence_ids":[]}'}}],
            "usage": {"cost": .001}})
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    async with create_inspect_service(store=store, policy=InspectPolicy(maximum_visual_qa_images=maximum_images)) as service:
        harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "Compare images", ()), store=store, inspect_service=service)
        actions = list(harness.eligible_actions())
        before = copy.deepcopy(actions)
        client = make_client(handler)
        await client.preload()
        await OpenRouterJsonGenerator(client).generate_json(system_prompt="Return JSON", payload={
            "workspace_id": "W1-1", "available_actions": actions, "model_view": harness.model_view()})
        await client.aclose()
    assert actions == before
    response_format = bodies[0]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    read = {"action": "READ_RECORD", "arguments": {"record_id": "E1", "offset": 0}, "focus_evidence_ids": []}
    assert validator.is_valid(read)
    assert not validator.is_valid({**read, "arguments": {}})
    assert not validator.is_valid({**read, "arguments": {"record_id": "E1", "offset": -1}})
    search = {"action": "SEARCH", "arguments": {"input_type": "text", "query": "Voyager NASA", "output_type": "text"}, "focus_evidence_ids": []}
    assert not validator.is_valid(search)
    search_validator = Draft202012Validator(_decision_output_schema({"workspace_id": "W1-1",
        "available_actions": [{"name": "SEARCH", "schema": _SEARCH_SCHEMA}]}))
    assert search_validator.is_valid(search)
    assert not search_validator.is_valid({**search, "arguments": {}})
    assert search_validator.is_valid({**search, "arguments": {"source": "https://science.nasa.gov/", "output_type": "text"}})
    action = {"action": "INSPECT", "arguments": {"targets": [f"E{i}" for i in range(1, maximum_images + 1)],
        "request": {"operation": "visual_qa", "params": {"question": "Compare images"}}}, "focus_evidence_ids": []}
    assert validator.is_valid(action)
    inspect_schema = next(item['schema'] for item in actions if item['name'] == 'INSPECT')
    emitted_inspect = next(branch for branch in schema['anyOf']
        if branch.get('properties', {}).get('action', {}).get('enum') == ['INSPECT'])
    original_operations = inspect_schema['oneOf']
    emitted_operations = emitted_inspect['properties']['arguments']['oneOf']
    assert len(original_operations) == len(emitted_operations)
    for original, emitted in zip(original_operations, emitted_operations):
        if original['properties']['request']['properties']['operation']['const'] != 'visual_qa':
            assert emitted == original
        else:
            assert original['properties']['targets']['uniqueItems'] is True
            assert 'uniqueItems' not in emitted['properties']['targets']
            assert emitted['properties']['request'] == original['properties']['request']
    missing_question = copy.deepcopy(action)
    missing_question['arguments']['request']['params'] = {}
    assert not validator.is_valid(missing_question)
    align = {'action':'INSPECT','arguments':{'targets':['E1','E2'],
        'request':{'operation':'align_media','params':{'mode':'visual','range_a_s':[.1,2.7]}}},
        'focus_evidence_ids':[]}
    assert validator.is_valid(align)
    align['arguments']['request']['params'] = {}
    assert not validator.is_valid(align)
    too_many = copy.deepcopy(action)
    too_many["arguments"]["targets"].append("E999")
    assert not validator.is_valid(too_many)
    combined_ids = copy.deepcopy(action)
    combined_ids["arguments"]["targets"] = ["E1, E2"]
    assert not validator.is_valid(combined_ids)
    fractional_times = copy.deepcopy(action)
    fractional_times["arguments"] = {"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [.3, .8, 1.4]}}}
    assert validator.is_valid(fractional_times)
    no_focus = {k: v for k, v in action.items() if k != "focus_evidence_ids"}
    assert not validator.is_valid(no_focus)
    report = {"task_id": "W1-1", "status": "COMPLETED", "findings": [], "selected_evidence_ids": []}
    assert not validator.is_valid(report)
    assert validator.is_valid({**report, "status": "INCOMPLETE", "status_reason": "Required evidence is unavailable."})
    report = {**report, "findings": [{"statement": "Visible content", "evidence_ids": ["E1"], "limitations": []}],
              "selected_evidence_ids": ["E1"]}
    assert validator.is_valid(report)
    assert not validator.is_valid({**report, "findings": [{**report["findings"][0], "evidence_ids": []}]})
    report["status_reason"] = "missing evidence"
    assert not validator.is_valid(report)
    report["status"] = "INCOMPLETE"
    assert validator.is_valid(report)


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('duplicate', [False, True])
async def test_provider_decoder_compatibility_keeps_runtime_target_uniqueness(make_client, monkeypatch, native, duplicate):
    import copy
    from foren_searcher.inspect.contracts import inspect_action_schema
    from foren_searcher.local_models import TransformersHiPolicy

    if native:
        monkeypatch.setitem(MODEL, 'supported_parameters', ['reasoning','tools','tool_choice'])
    targets = ['E1', 'E1' if duplicate else 'E2']
    output = {'action':'INSPECT','arguments':{'targets':targets,
        'request':{'operation':'visual_qa','params':{'question':'Compare the images'}}},'focus_evidence_ids':[]}
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        schema = (body['tools'][0]['function']['parameters'] if native
                  else body['response_format']['json_schema']['schema'])
        if 'uniqueItems' in json.dumps(schema):
            return httpx.Response(200,json={'error':{'code':502,'message':'unsupported schema'},'usage':{'cost':0}})
        if native:
            assert body['tool_choice']['function']['name']=='submit_harness_decision'
            message = {'tool_calls':[{'id':'call-test','type':'function','function':{
                'name':'submit_harness_decision','arguments':json.dumps({'decision':output})}}]}
        else:
            message = {'content':json.dumps(output)}
        return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls' if native else 'stop',
            'message':message}],'usage':{'cost':.001}})
    client = make_client(handler)
    actions = ({'name':'INSPECT','schema':inspect_action_schema()},)
    original = copy.deepcopy(actions)
    generator = OpenRouterJsonGenerator(client)
    events = []
    generator.generation_observer = events.append
    try:
        await client.preload()
        call = TransformersHiPolicy(generator,policy_attempts=1).decide(workspace_id='W1-1',
            model_view={'task_id':'W1-1','task':'Compare supplied images',
                        'evidence_view':[{'evidence_id':'E1'},{'evidence_id':'E2'}]},actions=actions)
        if duplicate:
            with pytest.raises(LocalModelOutputError,match='duplicates|duplicate'):
                await call
        else:
            assert await call == output
        assert actions == original
        assert 'uniqueItems' in bodies[0]['messages'][1]['content']
        assert len(events)==len(bodies)==1
        assert ('native_tool_call' in events[0]) == native
        assert client.usage_report()['actual_cost_usd']==.001
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['multiple','wrong_name','truncated','bad_json','extra_field','not_object'])
async def test_native_decision_rejects_invalid_envelopes_after_recording_cost(make_client, monkeypatch, case):
    monkeypatch.setitem(MODEL,'supported_parameters',['reasoning','tools','tool_choice'])
    function = {'name':'submit_harness_decision','arguments':json.dumps({'decision':{'decision':'FINISH','answer':'unknown'}})}
    call = {'type':'function','id':'call-test','function':function}
    if case=='wrong_name': function['name']='SEARCH'
    if case=='bad_json': function['arguments']='{"decision":'
    if case=='extra_field': function['arguments']='{"decision":{},"extra":true}'
    if case=='not_object': function['arguments']='{"decision":[]}'
    def handler(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'length' if case=='truncated' else 'tool_calls',
            'message':{'tool_calls':[call,call] if case=='multiple' else [call]}}],'usage':{'cost':.002}})
    client=make_client(handler)
    try:
        await client.preload()
        with pytest.raises(OpenRouterError,match='no action was executed'):
            await client.complete([{'role':'user','content':'Investigate'}],role='GIA',output_schema={'type':'object'})
        assert client.usage_report()['actual_cost_usd']==.002
        assert client.usage_report()['request_count']==1
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [[], ["reasoning"]])
@pytest.mark.parametrize("effort", ["low", "none"])
async def test_completion_respects_catalog_reasoning_support(make_client, supported, effort):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"cost": .001}})
    client = make_client(handler, reasoning_effort=effort)
    await client.preload()
    client.metadata["supported_parameters"] = supported
    await client.complete([], role="HI")
    assert ("reasoning" in bodies[0]) == bool(supported)
    if supported:
        assert bodies[0]["reasoning"] == {"effort": effort}
    assert client.usage_report()["reasoning_effort"] == (effort if supported else None)
    assert bodies[0]["provider"]["require_parameters"] is True
    assert client.usage_report()["actual_cost_usd"] == .001
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported,format_type", [
    (["structured_outputs", "response_format"], "json_schema"),
    (["response_format"], "json_object"),
    ([], None),
])
async def test_output_format_uses_declared_model_capability(make_client, supported, format_type):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"cost": .001}})
    client = make_client(handler)
    await client.preload()
    client.metadata["supported_parameters"] = supported
    await client.complete([], role="GIA", json_output=True,
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert bodies[0].get("response_format", {}).get("type") == format_type
    assert bodies[0]["provider"]["require_parameters"] is True
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('endpoints,sampling', [
    ([['reasoning','structured_outputs']], {}),
    ([['reasoning','structured_outputs','temperature']], {'temperature':1.0}),
    ([['reasoning','structured_outputs','temperature','top_p']], {'temperature':1.0,'top_p':.95}),
    ([['reasoning','structured_outputs','temperature'],['reasoning','response_format','top_p']], {'temperature':1.0}),
    ([['reasoning','tools','tool_choice','temperature'],['reasoning','top_p']], {'temperature':1.0}),
])
async def test_sampling_defaults_share_a_compatible_endpoint_with_decision_format(make_client,endpoints,sampling):
    bodies=[]
    supported=[set(values) for values in endpoints]
    def handler(request):
        body=json.loads(request.content)
        bodies.append(body)
        required={p for p in ('reasoning','temperature','top_p') if p in body}
        if 'tools' in body:
            required.update(('tools','tool_choice'))
            choice={'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'call-one','type':'function',
                'function':{'name':'submit_harness_decision','arguments':'{"decision":{"ok":true}}'}}]}}
        else:
            required.add('structured_outputs' if body['response_format']['type']=='json_schema' else 'response_format')
            choice={'finish_reason':'stop','message':{'content':'{"ok":true}'}}
        if not any(required<=parameters for parameters in supported):
            return httpx.Response(404,json={'error':{'message':'No endpoint supports every requested parameter'}})
        return httpx.Response(200,json={'choices':[choice],'usage':{'cost':.001}})
    client=make_client(handler)
    try:
        await client.preload()
        client._endpoint_parameters=tuple(frozenset(values) for values in endpoints)
        result=await client.complete([],role='HI',output_schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']})
        assert json.loads(result['text'])=={'ok':True}
        assert len(bodies)==1
        assert {k:bodies[0][k] for k in ('temperature','top_p') if k in bodies[0]}==sampling
        assert bodies[0]['provider']['require_parameters'] is True
        assert client.usage_report()['unsettled_reserved_usd']==0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_unconstrained_json_preserves_gia_reference_validation(make_client):
    from foren_searcher.local_models import TransformersGiaPolicy
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content":
            '{"decision":"INVESTIGATE","tasks":[{"task":"Read source","materials":["E999"]}]}'}}], "usage": {"cost": .001}})
    client = make_client(handler)
    client.structured_outputs = False
    try:
        await client.preload()
        with pytest.raises(LocalModelOutputError, match="E999"):
            await TransformersGiaPolicy(OpenRouterJsonGenerator(client), policy_attempts=1).decide(
                model_view={"task": "Read the actual source", "original_materials": [{"reference_id": "C0"}]},
                history=(), actions=())
        assert len(bodies) == 1 and "response_format" not in bodies[0]
        assert "tools" not in bodies[0]
        assert bodies[0]["provider"]["require_parameters"] is True
        assert client.usage_report()["structured_outputs"] is False
        assert client.usage_report()["actual_cost_usd"] == .001
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('reference', ['E1', 'E999'])
async def test_finish_quote_recovery_preserves_text_citations_and_single_charge(make_client, reference):
    from foren_searcher.local_models import TransformersGiaPolicy
    raw = '{"decision":"FINISH","answer":"配文称"日期已确认"，正文为\\"date unknown\\"。' + reference + '\\n尚不能核实。"}'
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': raw}}],
                                          'usage': {'cost': .001}})
    client = make_client(handler)
    client.structured_outputs = False
    generator = OpenRouterJsonGenerator(client)
    observed = []
    generator.generation_observer = observed.append
    try:
        await client.preload()
        operation = TransformersGiaPolicy(generator, policy_attempts=1).decide(
            model_view={'task': '核查日期', 'current_materials': [{'evidence_id': 'E1'}],
                        'latest_reports': [{'run_status': 'reported', 'report': {'status': 'INCOMPLETE'}}]},
            history=(), actions=())
        if reference == 'E999':
            with pytest.raises(LocalModelOutputError, match='E999'):
                await operation
        else:
            assert await operation == {'decision': 'FINISH', 'answer': '配文称"日期已确认"，正文为"date unknown"。E1\n尚不能核实。'}
        assert observed[0]['raw_output'] == raw
        assert len(bodies) == 1 and 'response_format' not in bodies[0]
        assert client.usage_report()['actual_cost_usd'] == .001
    finally:
        await client.aclose()


@pytest.mark.parametrize('raw', [
    '{"decision":"FINISH","answer":"未完整返回',
    '{"decision":"FINISH","answer":"未完整返回"',
    '{"decision":"FINISH","answer":"配文称"未知"","action":"SEARCH"}',
    '{"action":"SEARCH","arguments":{"query":"配文称"未知""}}',
    '{"decision":"INVESTIGATE","tasks":[{"task":"配文称"未知"","materials":[]}]}',
    '{"task_id":"W1-1","status":"COMPLETED","findings":[]}',
])
def test_finish_text_recovery_does_not_complete_or_change_action_objects(raw):
    from foren_searcher.openrouter import _recover_finish_text
    assert _recover_finish_text(raw) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("effort,json_supported,has_report,invalid,format_type", [
    ("low", True, True, False, "json_schema"),
    ("low", True, True, True, "json_schema"),
    ("none", True, True, False, "json_schema"),
    ("low", False, True, False, "json_schema"),
    ("low", True, False, False, "json_schema"),
])
async def test_gia_summary_json_preserves_policy_validation(make_client, effort, json_supported, has_report, invalid, format_type):
    from foren_searcher.local_models import TransformersGiaPolicy

    bodies = []
    decision = {"decision": "INVESTIGATE", "tasks": [{"task": "核验材料", "materials": ["E999" if invalid else "C0"]}]}
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}],
                                         "usage": {"cost": .001}})
    client = make_client(handler, reasoning_effort=effort)
    try:
        await client.preload()
        client.metadata["supported_parameters"] = ["reasoning", "structured_outputs", *(["response_format"] if json_supported else [])]
        policy = TransformersGiaPolicy(OpenRouterJsonGenerator(client), policy_attempts=1)
        report = {"run_status": "reported", "report": {"status": "COMPLETED", "findings": [
            {"statement": "已有材料内容", "evidence_ids": ["E1"], "limitations": []}]}}
        call = policy.decide(model_view={"task": "核验材料", "original_materials": [{"reference_id": "C0"}],
                                        "latest_reports": [report] if has_report else []}, history=(), actions=())
        if invalid:
            with pytest.raises(LocalModelOutputError, match="E999"):
                await call
        else:
            assert await call == decision
        assert len(bodies) == 1
        assert bodies[0]["response_format"]["type"] == format_type
        assert bodies[0]["reasoning"] == {"effort": effort}
        assert bodies[0]["provider"]["require_parameters"] is True
        assert bodies[0]["provider"]["max_price"] == {"prompt": .15, "completion": .4, "request": 0, "image": 0}
        assert client.usage_report()["actual_cost_usd"] == .001
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["runtime_error", "reported"])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("structured", [False, True])
async def test_gia_without_findings_can_finish_in_json_after_workspace_outcome(make_client, status, invalid, structured):
    from foren_searcher.local_models import TransformersGiaPolicy
    bodies = []
    decision = {"decision": "FINISH", "answer": "工具结果 " + ("E999" if invalid else "E1") + " 尚不足以确定结论。"}
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}],
                                         "usage": {"cost": .001}})
    client = make_client(handler)
    client.structured_outputs = structured
    try:
        await client.preload()
        outcome = {"task_id": "W1-1", "run_status": status}
        if status == "reported":
            outcome["report"] = {"status": "INCOMPLETE", "findings": [], "selected_evidence_ids": []}
        policy = TransformersGiaPolicy(OpenRouterJsonGenerator(client), policy_attempts=1)
        call = policy.decide(model_view={"task": "核验材料", "current_materials": [{"evidence_id": "E1"}],
            "latest_reports": [outcome], "runtime_constraints": {"can_create_round": True}}, history=(), actions=())
        if invalid:
            with pytest.raises(LocalModelOutputError, match="E999"):
                await call
        else:
            assert await call == decision
        assert len(bodies) == 1
        if structured:
            fmt = bodies[0]['response_format']
            assert fmt['type'] == 'json_schema'
            from jsonschema import Draft202012Validator
            validator = Draft202012Validator(fmt['json_schema']['schema'])
            validator.validate(decision)
            validator.validate({'decision': 'INVESTIGATE', 'tasks': [{'task':'核查来源', 'materials':['E1']}]})
        else:
            assert 'response_format' not in bodies[0] and 'tools' not in bodies[0]
        assert bodies[0]["provider"]["require_parameters"] is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_zero_price_route_preserves_paid_reservations_and_disallows_price_increase(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-for-output")
    config = tmp_path/'free.json'
    config.write_text(json.dumps({"model": "example/free", "budget_usd": .1,
        "input_price_cap_per_million": 0, "output_price_cap_per_million": 0}), encoding='utf-8')
    ledger = UsageLedger(tmp_path/'openrouter-usage.sqlite3', .1)
    ledger.reserve(.1, model="paid/model", role="HI")
    bodies = []
    model = {**MODEL, "id": "example/free", "pricing": {"prompt": "0", "completion": "0"}}
    def handler(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={"data": [model]})
        if request.url.path.endswith('/key'):
            return httpx.Response(200, json={"data": {"limit_remaining": 1}})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"cost": 0}})
    client = OpenRouterClient(config, transport=httpx.MockTransport(handler))
    await client.preload()
    await client.complete([], role="GIA")
    assert bodies[0]["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0, "image": 0}
    assert client.usage_report()["requests"][0]["reserved_usd"] == 0
    assert ledger.summary()["cumulative_accounted_usd"] == .1
    assert ledger.summary()["unsettled_reserved_usd"] == .1
    with pytest.raises(BudgetExceeded):
        ledger.reserve(.001, model="paid/model", role="HI")
    model["pricing"] = {"prompt": "0.0000001", "completion": "0"}
    with pytest.raises(OpenRouterError, match="unit-price caps"):
        await client.preload()
    assert len(bodies) == 1
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,precision,extra_price,eligible", [
    (["fp4"], "nvfp4", {}, True),
    (["fp4"], "mxfp4", {}, True),
    (["fp8"], "mxfp8", {}, True),
    (["nvfp4"], "nvfp4", {}, True),
    (["mxfp4"], "mxfp4", {}, True),
    (["mxfp8"], "mxfp8", {}, True),
    (["nvfp4", "fp8"], "mxfp8", {}, True),
    (["nvfp4"], "fp4", {}, False),
    (["nvfp4"], "mxfp4", {}, False),
    (["mxfp4"], "nvfp4", {}, False),
    (["mxfp8"], "fp8", {}, False),
    (["fp8", "bf16"], "nvfp4", {}, False),
    (["fp4"], "unknown", {}, False),
    (["fp4"], ["nvfp4"], {}, False),
    (["fp4"], None, {}, False),
    (["fp4"], "nvfp4", {"prompt": "0.0000001"}, False),
    (["fp4"], "nvfp4", {"request": "0.01"}, False),
])
async def test_endpoint_precision_families_preserve_exact_filters_and_zero_price(
    tmp_path, monkeypatch, requested, precision, extra_price, eligible
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-for-output")
    config = tmp_path / "free.json"
    config.write_text(json.dumps({"model": "example/free", "budget_usd": .1,
        "providers": ["selected"], "quantizations": requested,
        "input_price_cap_per_million": 0, "output_price_cap_per_million": 0}), encoding="utf-8")
    ledger = UsageLedger(tmp_path / "openrouter-usage.sqlite3", .1)
    ledger.reserve(.075, model="previous/paid", role="HI")
    model = {**MODEL, "id": "example/free", "pricing": {"prompt": "0", "completion": "0"}}
    endpoint = {"tag": "selected/variant", "quantization": precision,
        "pricing": {**model["pricing"], **extra_price}, "supported_parameters": MODEL["supported_parameters"]}
    bodies = []

    def route(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [model]})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json={"data": {"endpoints": [endpoint]}})
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit_remaining": 1}})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"cost": 0}})

    client = OpenRouterClient(config, transport=httpx.MockTransport(route))
    try:
        if eligible:
            await client.preload()
            assert (await client.complete([], role="GIA"))["text"] == "ok"
            assert len(bodies) == 1
            assert bodies[0]["provider"]["quantizations"] == requested
            assert bodies[0]["provider"]["only"] == ["selected"]
            assert bodies[0]["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0, "image": 0}
        else:
            with pytest.raises(OpenRouterError, match="precision settings"):
                await client.preload()
            assert not bodies
            assert client.usage_report()["request_count"] == 0
        assert ledger.summary()["unsettled_reserved_usd"] == .075
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['eligible', 'other_provider', 'wrong_precision', 'higher_price', 'missing_price', 'invalid_price', 'extra_fee'])
async def test_selected_endpoint_price_is_checked_when_catalog_quote_exceeds_cap(make_client, case):
    client = make_client(budget=.1)
    client.providers = ['selected']
    client.input_price_cap = .043
    client.output_price_cap = .221
    prior_id = client.ledger.reserve(.07, model='previous/paid', role='HI')
    endpoint = {'tag': 'selected/fp8', 'quantization': 'fp8', 'context_length': 131072,
                'pricing': {'prompt': '0.000000042', 'completion': '0.00000022'}}
    if case == 'other_provider': endpoint['tag'] = 'elsewhere/fp8'
    if case == 'wrong_precision': endpoint['quantization'] = 'fp4'
    if case == 'higher_price': endpoint['pricing']['completion'] = '0.000001'
    if case == 'missing_price': endpoint['pricing'].pop('prompt')
    if case == 'invalid_price': endpoint['pricing']['prompt'] = 'nan'
    if case == 'extra_fee': endpoint['pricing']['request'] = '0.01'
    calls = []
    def route(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [MODEL]})
        if request.url.path.endswith('/endpoints'):
            return httpx.Response(200, json={'data': {'endpoints': [endpoint]}})
        if request.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {'limit_remaining': 1}})
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}], 'usage': {'cost': .00001}})
    client._http.close()
    client._http = httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=httpx.MockTransport(route))
    try:
        if case == 'eligible':
            await client.preload()
            await client.complete([], role='GIA')
            assert calls[0]['provider']['only'] == ['selected']
            assert calls[0]['provider']['max_price'] == {'prompt': .043, 'completion': .221, 'request': 0, 'image': 0}
            assert calls[0]['provider']['quantizations'] == ['fp8', 'bf16', 'fp16', 'fp32']
            request = client.usage_report()['requests'][0]
            assert client.metadata['context_length'] == 131072
            assert request['reserved_usd'] == pytest.approx((131072 * .043 + 8192 * .221) / 1_000_000)
        else:
            with pytest.raises(OpenRouterError):
                await client.preload()
            assert not calls
            assert client.usage_report()['request_count'] == 0
        prior = next(r for r in client.ledger.summary()['requests'] if r['id'] == prior_id)
        assert prior['reserved_usd'] == .07 and prior['cost_usd'] is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('case,expected', [
    ('plain', None), ('json', 'json_object'), ('schema', 'json_schema'),
    ('split', None), ('expensive_schema', None), ('fp4_schema', None), ('other_schema', None),
])
async def test_selected_endpoint_parameters_must_fit_one_eligible_route(make_client, case, expected):
    client = make_client()
    client.providers = ['selected']
    endpoint = {'tag': 'selected/fp8', 'quantization': 'fp8', 'pricing': dict(MODEL['pricing']),
                'supported_parameters': ['reasoning']}
    endpoints = [endpoint]
    if case == 'json': endpoint['supported_parameters'].append('response_format')
    if case == 'schema': endpoint['supported_parameters'].append('structured_outputs')
    if case in {'split', 'expensive_schema', 'fp4_schema', 'other_schema'}:
        other = {**endpoint, 'pricing': dict(MODEL['pricing']),
                 'supported_parameters': ['reasoning', 'structured_outputs']}
        if case == 'split': other['supported_parameters'] = ['structured_outputs']
        if case == 'expensive_schema': other['pricing']['completion'] = '0.000001'
        if case == 'fp4_schema': other['quantization'] = 'fp4'
        if case == 'other_schema': other['tag'] = 'elsewhere'
        endpoints.append(other)
    bodies = []

    def route(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [MODEL]})
        if request.url.path.endswith('/endpoints'):
            return httpx.Response(200, json={'data': {'endpoints': endpoints}})
        if request.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {'limit_remaining': 1}})
        body = json.loads(request.content)
        bodies.append(body)
        assert body.get('response_format', {}).get('type') == expected
        assert body['reasoning'] == {'effort': 'low'}
        assert body['provider']['only'] == ['selected']
        assert body['provider']['require_parameters'] is True
        assert body['provider']['quantizations'] == ['fp8', 'bf16', 'fp16', 'fp32']
        assert body['provider']['max_price'] == {'prompt': .15, 'completion': .4, 'request': 0, 'image': 0}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{"value": 1}'}}],
                                         'usage': {'cost': .001}})

    client._http.close()
    client._http = httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=httpx.MockTransport(route))
    try:
        await client.preload()
        value = await client.complete([], role='GIA', json_output=True, output_schema={'type': 'object'})
        assert json.loads(value['text']) == {'value': 1}
        assert len(bodies) == 1 and client.usage_report()['actual_cost_usd'] == .001
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('contexts,view_length,reservation_length', [
    ([131072], 131072, 131072),
    ([131072, 262144], 131072, 262144),
    ([None], 262144, 262144),
    ([True, 0, -1, '131072'], 262144, 262144),
    ([524288], 262144, 262144),
])
async def test_selected_context_limits_view_but_reserves_for_every_eligible_route(
    make_client, contexts, view_length, reservation_length
):
    client = make_client()
    client.providers = ['selected']
    endpoint = {'tag': 'selected/fp8', 'quantization': 'fp8', 'pricing': dict(MODEL['pricing']),
                'supported_parameters': ['reasoning']}
    endpoints = [{**endpoint, 'context_length': value} for value in contexts]
    endpoints += [
        {**endpoint, 'tag': 'other/fp8', 'context_length': 1024},
        {**endpoint, 'quantization': 'fp4', 'context_length': 2048},
        {**endpoint, 'pricing': {'prompt': '1', 'completion': '1'}, 'context_length': 4096},
    ]
    def route(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [MODEL]})
        if request.url.path.endswith('/endpoints'):
            return httpx.Response(200, json={'data': {'endpoints': endpoints}})
        if request.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {'limit_remaining': 1}})
        assert json.loads(request.content)['provider']['only'] == ['selected']
        # 缺失回执时保留实际预留，检查费用上限而非只检查元数据。
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}]})
    client._http.close()
    client._http = httpx.Client(base_url='https://openrouter.ai/api/v1/', transport=httpx.MockTransport(route))
    try:
        await client.preload()
        assert client.metadata['context_length'] == view_length
        await client.complete([], role='GIA')
        reserved = (reservation_length * .15 + 8192 * .4) / 1_000_000
        report = client.usage_report()
        assert report['unsettled_reserved_usd'] == pytest.approx(reserved)
        assert report['requests'][0]['cost_usd'] is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("repaired", [True, False])
async def test_complete_non_json_answer_gets_one_billed_format_repair(make_client, repaired):
    bodies = []
    valid = {"decision": "FINISH", "answer": "Evidence is insufficient."}
    def handler(request):
        bodies.append(json.loads(request.content))
        content = json.dumps(valid) if repaired and len(bodies) == 2 else "I need more evidence."
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    generator = OpenRouterJsonGenerator(client)
    outputs = []
    generator.generation_observer = outputs.append
    call = generator.generate_json(system_prompt="Respond with one JSON object.",
        payload={"available_actions": [], "global_view": {"task": "Check the evidence."}})
    if repaired:
        assert await call == valid
    else:
        with pytest.raises(LocalModelOutputError):
            await call
    assert len(bodies) == len(outputs) == client.usage_report()["request_count"] == 2
    assert client.usage_report()["actual_cost_usd"] == .004
    assert bodies[1]["messages"][:-2] == bodies[0]["messages"]
    assert bodies[1]["messages"][-2] == {"role": "assistant", "content": "I need more evidence."}
    assert bodies[1]["response_format"] == bodies[0]["response_format"]
    await client.aclose()


@pytest.mark.asyncio
async def test_format_repair_does_not_repeat_failed_paid_network_request(make_client):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503)
    client = make_client(handler)
    await client.preload()
    with pytest.raises(OpenRouterError, match="HTTP 503"):
        await OpenRouterJsonGenerator(client).generate_json(system_prompt="Respond with JSON.", payload={})
    assert len(calls) == client.usage_report()["request_count"] == 1
    assert client.usage_report()["unsettled_reserved_usd"] > 0
    await client.aclose()


@pytest.mark.asyncio
async def test_malformed_multitask_decision_retries_whole_request_without_dispatching_child(make_client):
    malformed = '{"decision":"INVESTIGATE","tasks":[{"task":"核对"正式验证"","materials":["E1"]},{"task":"核对日期","materials":[]}]}'
    valid = {"decision": "INVESTIGATE", "tasks": [
        {"task": '核对"正式验证"', "materials": ["E1"]},
        {"task": "核对日期", "materials": []}]}
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        content = malformed if len(bodies) == 1 else json.dumps(valid, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    generator = OpenRouterJsonGenerator(client)
    outputs = []
    generator.generation_observer = outputs.append
    try:
        decision = await generator.generate_json(system_prompt="Respond with one JSON object.",
            payload={"available_actions": [], "global_view": {"task": "并行完成两项调查"}})
        assert decision == valid
        assert len(bodies) == len(outputs) == 2
        assert outputs[0]["raw_output"] == malformed
        assert bodies[1]["messages"][-2]["content"] == malformed
        assert "JSON syntax error:" in bodies[1]["messages"][-1]["content"]
        assert "column" in bodies[1]["messages"][-1]["content"]
        assert client.usage_report()["actual_cost_usd"] == .004
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_complete_report_missing_array_closure_preserves_content_without_regeneration(make_client):
    valid = {"task_id": "W1-1", "status": "COMPLETED", "findings": [
        {"statement": "Observed content", "evidence_ids": ["E1"], "limitations": ["Only a sample"]}],
        "selected_evidence_ids": ["E1"]}
    raw = json.dumps(valid).replace('"Only a sample"]}', '"Only a sample"}')
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        content = raw if len(bodies) == 1 else json.dumps(valid)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    try:
        generator = OpenRouterJsonGenerator(client)
        outputs = []
        generator.generation_observer = outputs.append
        result = await generator.generate_json(system_prompt="Return the report.",
            payload={"workspace_id": "W1-1", "model_view": {}, "available_actions": []})
        assert result == valid
        assert len(bodies) == len(outputs) == 1
        assert outputs[0]["raw_output"] == raw
        assert client.usage_report()["actual_cost_usd"] == .002
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_id", [None, "W1-1"])
@pytest.mark.parametrize("first_content", ['', '{"decision":"FINISH","answer":"partial"}'])
async def test_token_limit_retries_without_accepting_even_parseable_partial_output(make_client, workspace_id, first_content):
    bodies, outputs = [], []
    valid = {"decision": "FINISH", "answer": "complete"}
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "length" if len(bodies) == 1 else "stop",
            "message": {"content": first_content if len(bodies) == 1 else json.dumps(valid)}}],
            "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    generator = OpenRouterJsonGenerator(client)
    generator.generation_observer = outputs.append
    try:
        assert await generator.generate_json(system_prompt="Return a JSON decision.",
            payload={"workspace_id": workspace_id}) == valid
        assert len(bodies) == len(outputs) == 2
        assert outputs[0]["raw_output"] == first_content
        assert "token limit" in outputs[0]["output_error"]
        assert "token limit" in bodies[1]["messages"][-1]["content"]
        assert bodies[1]["max_tokens"] == bodies[0]["max_tokens"]
        usage = client.usage_report()
        assert usage["actual_cost_usd"] == .004
        assert usage["unsettled_reserved_usd"] == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_repeated_token_limit_returns_recoverable_policy_error_with_bounded_spend(make_client):
    calls, outputs = [], []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
            "message": {"content": '{"decision":"FINISH","answer":"partial"}'}}], "usage": {"cost": .002}})
    client = make_client(handler)
    await client.preload()
    generator = OpenRouterJsonGenerator(client)
    generator.generation_observer = outputs.append
    try:
        from foren_searcher.policies import PolicyOutputError
        with pytest.raises(PolicyOutputError, match="token limit"):
            await generator.generate_json(system_prompt="Return JSON.", payload={})
        assert len(calls) == len(outputs) == 2
        assert all("token limit" in output["output_error"] for output in outputs)
        assert client.usage_report()["actual_cost_usd"] == .004
        assert client.usage_report()["unsettled_reserved_usd"] == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mandatory_reasoning_is_checked_before_paid_requests(make_client, monkeypatch):
    monkeypatch.setitem(MODEL, "reasoning", {"mandatory": True})
    client = make_client(reasoning_effort="none")
    with pytest.raises(OpenRouterError, match="requires reasoning"):
        await client.preload()
    assert client.usage_report()["request_count"] == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_selection_preserves_budget_and_sends_no_cross_provider_fallback(tmp_path, make_client):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"cost": .001}})
    client = make_client(handler)
    client.providers = ["deepinfra"]
    await client.preload()
    await client.complete([], role="GIA")
    assert bodies[0]["provider"]["only"] == ["deepinfra"]
    assert bodies[0]["provider"]["max_price"]["completion"] == .4
    assert client.usage_report()["routing"]["providers"] == ["deepinfra"]
    await client.aclose()


@pytest.mark.asyncio
async def test_vision_transmits_only_selected_bounded_images_and_source_times(tmp_path, make_client):
    bodies = []
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "E2: a cat; E8: a cup."}}], "usage": {"cost": 0.002}})
    client = make_client(handler)
    await client.preload()
    path = tmp_path / "frame.png"
    Image.new("RGB", (2000, 1000), "red").save(path)
    media = tuple(PreparedMedia(e, path, "image", MediaProfileData("image"),
                  json.dumps({"locator": {"time_s": t}, "derived_from": ["E0"]})) for e, t in (("E2", 3.1), ("E8", 1.2)))
    backend = OpenRouterInspectBackend(client)
    # 与真实 dispatcher 一样，INSPECT 方法在独立线程的事件循环运行。
    result = await asyncio.to_thread(lambda: asyncio.run(backend.visual_qa(media, VisualQaParams("Compare the frames"), InspectPolicy(maximum_model_image_pixels=10000))))
    images = [item for item in bodies[0]["messages"][1]["content"] if item["type"] == "image_url"]
    assert len(images) == 2
    for item in images:
        picture = Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1])))
        assert picture.width * picture.height <= 10000
        assert item["image_url"]["url"].startswith("data:image/png;base64,")
        assert picture.getpixel((0, 0)) == (255, 0, 0)
    assert result.result["image_evidence_ids"] == ["E2", "E8"]
    assert "3.1" in result.result["answer"] and "1.2" in result.result["answer"]
    assert "data:image" not in json.dumps(result.details)
    await backend.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_visual_service_recovers_after_bounded_rate_limit_retry(tmp_path, make_client, retry_clock):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="test-secret-not-for-output")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "A red rectangle."}}],
                                        "usage": {"cost": .002}})

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    source = store.commit_ingested_evidence(source_ref="I0", media_type="image/png", modalities=("image",),
        label="image", original_filename="image.png", data=buffer.getvalue())
    client = make_client(handler)
    await client.preload()
    service = create_inspect_service(store=store, backend=OpenRouterInspectBackend(client))
    request = {"targets": [source.evidence_id], "request": {"operation": "visual_qa",
        "params": {"question": "Describe the visible colors."}}}
    try:
        response = await service.execute(request, workspace_id="W1-1")
        assert response.status == "ready" and response.result_evidence_id
        assert json.loads(store.evidence(response.result_evidence_id).content)["answer"]
        assert len(calls) == client.usage_report()["request_count"] == 2
        assert client.usage_report()["unsettled_reserved_usd"] > 0
    finally:
        await service.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_native_reasoning_is_returned_separately_from_final_decision(make_client):
    def handler(request):
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"action":"READ_RECORD","arguments":{"record_id":"E1"},"focus_evidence_ids":[]}',
            'reasoning': 'Private observation before selecting the next action.'}}], 'usage': {'cost': .002}})
    client = make_client(handler)
    try:
        await client.preload()
        result = await client.complete([], role='HI')
        assert result['reasoning'] == 'Private observation before selecting the next action.'
        assert 'Private observation' not in result['text']
        assert client.usage_report()['actual_cost_usd'] == .002
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("identical", [False, True])
async def test_vision_service_binds_distinct_pixels_to_selected_evidence_in_order(tmp_path, make_client, filtered, identical):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        answer = "Visible solid colors." + (' The label reads "CREW ONLY". The context is unknown.' if filtered else '')
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer}}], "usage": {"cost": 0}})

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = []
    colors = [(230, 15, 20), (20, 210, 30), (15, 20, 220), (250, 240, 20)]
    if identical:
        colors[2] = colors[0]
    for index, color in enumerate(colors):
        buffer = io.BytesIO()
        Image.new("RGB", (200, 100), color).save(buffer, format="PNG")
        parent = store.commit_ingested_evidence(source_ref=f"I{index}", media_type="image/png", modalities=("image",),
            label="image", original_filename="image.png", data=buffer.getvalue())
        frame = store.commit_derived_evidence(derived_from=(parent.evidence_id,), modalities=("image",),
            label="frame", content="frame pixels", locator=json.dumps({"time_s": index / 10}), acquired_by="INSPECT/get_frames",
            discovered_by="W1-1", production_method="derived_media", artifact_bytes=buffer.getvalue(),
            media_type="image/png", suffix=".png", role="video_frame")
        evidence.append((frame, parent, color))
    client = make_client(handler)
    await client.preload()
    backend = OpenRouterInspectBackend(client)
    service = create_inspect_service(store=store, backend=backend,
        policy=InspectPolicy(maximum_model_image_pixels=10000))
    selected = [evidence[2], evidence[0], evidence[1]]
    targets = [item.evidence_id for item, _, _ in selected]
    try:
        response = await service.execute({"targets": targets, "request": {"operation": "visual_qa",
            "params": {"question": "Describe each supplied image."}}}, workspace_id="W1-2")
        assert response.status == "ready"
        parts = bodies[0]["messages"][1]["content"]
        assert len(parts) == len(selected) * 2 + 1
        identity_note = targets[0] + '、' + targets[1] + ' 的输入图像完全相同'
        assert (identity_note in parts[-1]['text']) == identical
        assert ('本次视觉输入一致性检查' in parts[-1]['text']) == identical
        if identical:
            note = parts[-1]['text'].split('本次视觉输入一致性检查：', 1)[1]
            assert targets[2] not in note
            assert '不证明采样时刻之间没有变化' in note
        for index, (item, parent, color) in enumerate(selected):
            label, image = parts[index * 2:index * 2 + 2]
            prefix = f"Evidence {item.evidence_id}; source locator: "
            assert label["text"].startswith(prefix)
            assert json.loads(label["text"][len(prefix):]) == {
                "derived_from": [parent.evidence_id], "locator": json.loads(item.locator)}
            picture = Image.open(io.BytesIO(base64.b64decode(image["image_url"]["url"].split(",", 1)[1])))
            assert picture.width * picture.height <= 10000
            assert max(abs(actual - expected) for actual, expected in zip(picture.getpixel((0, 0)), color)) <= 3
        saved = store.evidence(response.result_evidence_id)
        assert list(saved.derived_from) == targets
        assert (identity_note in json.loads(saved.content)['answer']) == identical
        assert json.loads(saved.content)["image_evidence_ids"] == targets
        assert any("相关句子未保留" in value for value in json.loads(saved.content)["limitations"]) == filtered
        if filtered:
            assert "The context is unknown." in json.loads(saved.content)["answer"]
        assert "CREW ONLY" not in saved.content
        assert evidence[3][0].evidence_id not in json.dumps(bodies[0]["messages"])
    finally:
        await service.aclose()
        await backend.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_cancellation_waits_for_inflight_billing_on_close(make_client):
    entered, release = threading.Event(), threading.Event()
    def handler(request):
        entered.set()
        assert release.wait(5)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"cost": 0.004}})
    client = make_client(handler)
    await client.preload()
    task = asyncio.create_task(client.complete([], role="GIA"))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    close = asyncio.create_task(client.aclose())
    await asyncio.sleep(0.02)
    assert not close.done()
    release.set()
    await close
    assert client.usage_report()["actual_cost_usd"] == 0.004


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_settings_encrypt_key_and_preserve_ledger(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = tmp_path / "config.json"
    save_settings(config, key="dummy-test-secret", model=DEFAULT_MODEL, budget_usd=1)
    assert "dummy-test-secret" not in config.read_text()
    assert load_api_key(config) == "dummy-test-secret"
    ledger = UsageLedger(tmp_path / "openrouter-usage.sqlite3", 1)
    ledger.reserve(.01, model=DEFAULT_MODEL, role="GIA")
    save_settings(config, key="", model="openai/gpt-oss-120b", visual_model=DEFAULT_MODEL, budget_usd=.5)
    assert load_api_key(config) == "dummy-test-secret"
    saved = json.loads(config.read_text())
    assert saved["model"] == "openai/gpt-oss-120b" and saved["visual_model"] == DEFAULT_MODEL
    assert ledger.summary()["request_count"] == 1


def test_remote_capabilities_do_not_require_local_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-for-output")
    harness = tmp_path / "harness/runtime"
    harness.mkdir(parents=True)
    (harness / "tool_executor.py").touch()
    result = capability_report(model_root=tmp_path, harness_root=tmp_path,
                               model_backend="openrouter", openrouter_config=tmp_path / "config.json")
    assert result["configured_for_text_demo"] is True
    assert result["model_backend"] == "openrouter"
    assert "test-secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_unified_preload_rejects_a_text_only_model_without_spending(tmp_path, monkeypatch):
    import foren_searcher.demo as demo
    import foren_searcher.openrouter as remote
    from foren_searcher.workspace import WorkspaceHarness
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-not-for-output")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "text/controller", "visual_model": "text/controller", "budget_usd": .1}))
    clients = []
    def create(config, *, purpose="shared"):
        def route(request):
            assert request.method == "GET"
            return httpx.Response(200, json={"data": [{**MODEL, "id": "text/controller",
                "architecture": {"input_modalities": ["text"]}}] if request.url.path.endswith("/models")
                else {"limit_remaining": 1}})
        client = OpenRouterClient(config, purpose=purpose, transport=httpx.MockTransport(route))
        clients.append(client)
        return client
    monkeypatch.setattr(remote, "OpenRouterClient", create)
    monkeypatch.setattr(demo, "harness1_workspace_type", lambda root: WorkspaceHarness)
    class Search:
        closed = False
        async def aclose(self):
            self.closed = True
    search = Search()
    monkeypatch.setattr(demo, "_create_search", lambda *args: search)
    harness = tmp_path / "harness/runtime"
    harness.mkdir(parents=True)
    (harness / "tool_executor.py").touch()
    with pytest.raises(demo.DemoRunError):
        await demo.run_demo(task="Describe the supplied image", model_root=tmp_path, harness_root=tmp_path,
            strict_semantics=False, output_dir=tmp_path / "result", model_backend="openrouter", openrouter_config=config)
    result = json.loads((tmp_path / "result/result.json").read_text(encoding="utf-8"))
    assert "image-capable" in result["error"]["message"]
    assert len(clients) == 1 and clients[0]._closed and search.closed
    assert not clients[0].metadata
    assert result["model_usage"]["request_count"] == result["model_usage"]["actual_cost_usd"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("route_missing,legacy_visual_setting", [(False, False), (True, False), (False, True)])
async def test_demo_remote_path_preserves_evidence_and_accounts_all_roles(tmp_path, make_client, monkeypatch, route_missing, legacy_visual_setting):
    import foren_searcher.demo as demo
    import foren_searcher.openrouter as remote
    from foren_searcher.workspace import WorkspaceHarness
    outputs = iter([
        {"decision": "INVESTIGATE", "tasks": [{"task": "Describe the image", "materials": ["E1"]}]},
        {"task_id": "W1-1", "status": "COMPLETED", "findings": [{"statement": "The image is red.", "evidence_ids": ["E1"], "limitations": []}], "selected_evidence_ids": ["E1"]},
        {"decision": "FINISH", "answer": "The image is red (E1)."},
    ])
    calls = []
    def handler(request):
        calls.append(request)
        value = next(outputs)
        if route_missing and len(calls) == 2:
            return httpx.Response(404, text="test-secret-not-for-output")
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": value if isinstance(value, str) else json.dumps(value)}}], "usage": {"cost": .001}})
    client = make_client(handler)
    if legacy_visual_setting:
        config = tmp_path / "config.json"
        settings = json.loads(config.read_text())
        settings.update(visual_model="google/gemma-4-31b-it", visual_settings={"reasoning_effort": "low"})
        config.write_text(json.dumps(settings))
    monkeypatch.setattr(remote, "OpenRouterClient", lambda config: client)
    monkeypatch.setattr(demo, "harness1_workspace_type", lambda root: WorkspaceHarness)
    class Search:
        closed = False
        async def aclose(self):
            self.closed = True
    search = Search()
    monkeypatch.setattr(demo, "_create_search", lambda *args: search)
    harness = tmp_path / "harness/runtime"
    harness.mkdir(parents=True)
    (harness / "tool_executor.py").touch()
    path = tmp_path / "red.png"
    Image.new("RGB", (32, 32), "red").save(path)
    run = demo.run_demo(task="Describe the image", media_paths=(str(path),),
        model_root=tmp_path, harness_root=tmp_path, strict_semantics=False, output_dir=tmp_path / "result",
        model_backend="openrouter", openrouter_config=tmp_path / "config.json")
    if route_missing:
        from foren_searcher.store import GlobalStore
        with pytest.raises(demo.DemoRunError):
            await run
        result = json.loads((tmp_path / "result/result.json").read_text(encoding="utf-8"))
        assert result["decision"] is None and "start a new run" in result["error"]["message"]
        assert len(calls) == result["model_usage"]["request_count"] == 2
        assert result["model_usage"]["actual_cost_usd"] == .001
        assert result["model_usage"]["unsettled_reserved_usd"] == 0
        store = GlobalStore(tmp_path / "result/investigation.sqlite3", tmp_path / "result/artifacts")
        assert store.healthcheck().to_dict()["status"] == "ok"
        assert store.evidence("E1").label == "red.png"
        assert (tmp_path / "result/trajectory.jsonl").stat().st_size > 0
    else:
        result = await run
        assert result["decision"]["answer"] == "The image is red (E1)."
        assert result["store_health"]["status"] == "ok"
        assert result["model_usage"]["actual_cost_usd"] == .003
    assert {row["role"] for row in result["model_usage"]["requests"]} == {"GIA", "HI"}
    assert search.closed and client._closed
    bodies = [json.loads(request.content) for request in calls]
    assert all(body["model"] == DEFAULT_MODEL for body in bodies)
    # 第 1 轮还没有任何报告，GIA 依旧要看到用户给的原件像素（DESIGN 2.3）。
    gia_first = [part for message in bodies[0]["messages"] if isinstance(message["content"], list)
                 for part in message["content"] if part["type"] == "image_url"]
    assert len(gia_first) == 1
    with Image.open(io.BytesIO(base64.b64decode(gia_first[0]["image_url"]["url"].split(",", 1)[1]))) as original:
        red, green, blue = original.getpixel((16, 16))
        assert red > 240 and green < 15 and blue < 15
    hi_body = bodies[1]
    image_parts = [part for message in hi_body["messages"] if isinstance(message["content"], list)
                   for part in message["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 1 and "global_view" not in json.dumps(hi_body)
    with Image.open(io.BytesIO(base64.b64decode(image_parts[0]["image_url"]["url"].split(",", 1)[1]))) as picture:
        red, green, blue = picture.getpixel((16, 16))
        assert red > 240 and green < 15 and blue < 15
    assert "visual_qa" not in json.dumps(hi_body) and "audio_qa" not in json.dumps(hi_body)
    if not route_missing:
        gia_images = [part for message in bodies[2]["messages"] if isinstance(message["content"], list)
                      for part in message["content"] if part["type"] == "image_url"]
        assert gia_images == image_parts
        events = [json.loads(line) for line in (tmp_path / "result/trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
        inputs = [e['payload']['model_view'] for e in events if e['actor_role'] == 'GIA' and e['event_type'] == 'model_input']
        assert [image['evidence_id'] for image in inputs[-1]['image_inputs']] == ['E1']
        usage = client.usage_report(client, client)
        assert usage["request_count"] == 3 and len(usage["models"]) == 1
        assert usage["actual_cost_usd"] == usage["cumulative_accounted_usd"] == .003
        with pytest.raises(BudgetExceeded):
            client.ledger.reserve(.998, model=client.model, role="HI")
    assert "test-secret" not in (tmp_path / "result/result.json").read_text(encoding="utf-8")

@pytest.mark.asyncio
@pytest.mark.parametrize('kind,cost', [('http',None),('body',None),('body',.004)])
async def test_async_rate_limit_retries_identical_request_and_preserves_each_bill(make_client,retry_clock,kind,cost):
    calls=[];events=[]
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:
            return httpx.Response(429 if kind=='http' else 200,headers={'Retry-After':'5'},
                json={'error':{'code':429},'usage':{'cost':cost}})
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'recovered'}}],
            'usage':{'cost':.002}})
    client=make_client(handler);await client.preload()
    try:
        result=await client.complete([{'role':'user','content':'same step'}],role='HI',retry_observer=events.append)
        assert result['text']=='recovered' and calls[0]==calls[1]
        assert retry_clock[1]==[5] and [e['state'] for e in events]==['waiting','retrying']
        report=client.usage_report()
        assert report['request_count']==2 and report['requests'][0]['cost_usd']==cost
        assert report['actual_cost_usd']==pytest.approx((cost or 0)+.002)
        assert bool(report['unsettled_reserved_usd'])==(cost is None)
    finally:await client.aclose()


@pytest.mark.asyncio
async def test_persistent_rate_limit_has_bounded_attempts_and_keeps_reservations(make_client,retry_clock):
    calls=[];events=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(429,text='test-secret-not-for-output')
    client=make_client(handler);await client.preload()
    try:
        with pytest.raises(OpenRouterError,match='automatic retries exhausted') as caught:
            await client.complete([],role='GIA',retry_observer=events.append)
        assert 'test-secret' not in str(caught.value)
        assert len(calls)==client.usage_report()['request_count']==3
        assert retry_clock[1]==[30,30]
        assert all(r['cost_usd'] is None for r in client.usage_report()['requests'])
        assert events[-1]['state']=='exhausted'
    finally:await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('seconds,requests,waits',[('600',1,[]),('100',2,[100])])
async def test_long_rate_limit_never_retries_before_server_time_or_beyond_wait_cap(make_client,retry_clock,seconds,requests,waits):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(429,headers={'Retry-After':seconds})
    client=make_client(handler);await client.preload()
    try:
        with pytest.raises(OpenRouterError,match='wait exceeds 180 seconds'):
            await client.complete([],role='HI')
        assert len(calls)==requests and retry_clock[1]==waits
    finally:await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_retry_checks_budget_before_any_second_post(make_client,retry_clock):
    calls=[]
    def handler(request):
        calls.append(request)
        client.ledger.budget_usd=client.usage_report()['unsettled_reserved_usd']*1.5
        return httpx.Response(429)
    client=make_client(handler);await client.preload()
    try:
        with pytest.raises(BudgetExceeded):await client.complete([],role='HI')
        assert len(calls)==client.usage_report()['request_count']==1
        assert retry_clock[1]==[30]
    finally:await client.aclose()


@pytest.mark.asyncio
async def test_cancel_during_rate_limit_wait_prevents_all_future_requests(make_client,monkeypatch):
    import foren_searcher.openrouter as remote
    waiting=asyncio.Event();events=[];calls=[]
    async def sleep(delay):
        waiting.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(remote,'_retry_sleep',sleep)
    def handler(request):
        calls.append(request)
        return httpx.Response(429)
    client=make_client(handler);await client.preload()
    task=asyncio.create_task(client.complete([],role='HI',retry_observer=events.append))
    try:
        await asyncio.wait_for(waiting.wait(),2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert len(calls)==client.usage_report()['request_count']==1
        assert events[-1]['state']=='cancelled' and client.usage_report()['unsettled_reserved_usd']>0
    finally:await client.aclose()


@pytest.mark.asyncio
async def test_gia_and_hi_wait_on_shared_cooldown_without_reserving_while_waiting(make_client,monkeypatch):
    import foren_searcher.openrouter as remote
    clock=[100.0];waiting=[];both=asyncio.Event();release=asyncio.Event();calls=[]
    monkeypatch.setattr(remote,'monotonic',lambda:clock[0])
    async def sleep(delay):
        waiting.append(delay)
        if len(waiting)==2:both.set()
        await release.wait()
    monkeypatch.setattr(remote,'_retry_sleep',sleep)
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:return httpx.Response(429)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'ok'}}],'usage':{'cost':.002}})
    client=make_client(handler);await client.preload()
    hi=asyncio.create_task(client.complete([{'role':'user','content':'private HI task'}],role='HI'))
    while not waiting:await asyncio.sleep(.001)
    gia=asyncio.create_task(client.complete([{'role':'user','content':'GIA task'}],role='GIA'))
    try:
        await asyncio.wait_for(both.wait(),2)
        assert len(calls)==client.usage_report()['request_count']==1
        clock[0]+=30;release.set()
        assert all(r['text']=='ok' for r in await asyncio.gather(hi,gia))
        assert len(calls)==3 and waiting==[30,30]
        assert sorted(c['messages'][0]['content'] for c in calls)==['GIA task','private HI task','private HI task']
    finally:
        hi.cancel();gia.cancel();await asyncio.gather(hi,gia,return_exceptions=True)
        await client.aclose()


@pytest.mark.asyncio
async def test_generator_records_real_retry_events_with_correct_workspace(make_client,retry_clock,tmp_path):
    calls=[]
    def handler(request):
        calls.append(request)
        if len(calls)==1:return httpx.Response(429)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"status":"COMPLETED"}'}}],'usage':{'cost':.002}})
    store=GlobalStore(tmp_path/'retry.sqlite3',tmp_path/'retry-artifacts')
    client=make_client(handler);await client.preload()
    try:
        generator=OpenRouterJsonGenerator(client,store=store)
        result=await generator.generate_json(system_prompt='test',payload={'workspace_id':'W1-2','model_view':{}})
        events=store.trajectory_events()
        assert result=={'status':'COMPLETED'}
        assert [e['event_type'] for e in events]==['model_retry','model_retry']
        assert all(e['actor_id']=='W1-2' and e['actor_role']=='HI' for e in events)
    finally:await client.aclose()
