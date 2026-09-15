import asyncio
import json
from collections import Counter

import pytest

from foren_searcher.contracts import WorkspaceTask
from foren_searcher.local_models import TransformersHiPolicy
from foren_searcher.local_models import _normalize_hi_output, _validate_hi_output
from foren_searcher.openrouter import OpenRouterJsonGenerator
from foren_searcher.policies import OutputRepairLimit
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness, WorkspaceRunner
from test_framework_runtime import add_text_evidence


class Client:
    def __init__(self, respond):
        self.respond = respond
        self.calls = Counter()

    async def complete(self, messages, **kwargs):
        actor = json.loads(messages[1]["content"])["workspace_id"]
        self.calls[actor] += 1
        await asyncio.sleep(0)
        output = self.respond(actor, self.calls[actor])
        return {"text": output if isinstance(output, str) else json.dumps(output),
                "model": "test", "provider": "test", "generation_id": "test", "usage": {}}


def make_report(actor, evidence):
    return {"task_id": actor, "status": "COMPLETED", "findings": [
        {"statement": "Recorded value is 12.", "evidence_ids": [evidence], "limitations": []}],
        "selected_evidence_ids": [evidence]}


@pytest.mark.asyncio
async def test_parse_role_and_outer_retries_share_one_failure_budget():
    store = GlobalStore()
    source = add_text_evidence(store, "Recorded value is 12.")
    outputs = ["not JSON", make_report("W99-1", source.evidence_id), "not JSON", {"unexpected": True}]
    client = Client(lambda actor, count: outputs[count - 1])
    policy = TransformersHiPolicy(OpenRouterJsonGenerator(client))
    runner = WorkspaceRunner(harness=WorkspaceHarness(store=store,
        task=WorkspaceTask("W1-1", "Read the value", (source.evidence_id,))), policy=policy)
    with pytest.raises(OutputRepairLimit, match="4 invalid outputs"):
        await runner.run()
    assert client.calls["W1-1"] == 4
    assert store.workspace_report_records() == ()


@pytest.mark.asyncio
async def test_completed_actions_reset_repair_budget_and_parallel_workspaces_are_independent():
    store = GlobalStore()
    source = add_text_evidence(store, "Recorded value is 12.")
    def respond(actor, count):
        if count in (1, 2, 3, 5, 6, 7):
            return "not JSON"
        if count == 4:
            return {"action": "READ_RECORD", "arguments": {"record_id": source.evidence_id},
                    "focus_evidence_ids": [source.evidence_id]}
        return make_report(actor, source.evidence_id)
    client = Client(respond)
    policy = TransformersHiPolicy(OpenRouterJsonGenerator(client))
    results = await asyncio.gather(*(WorkspaceRunner(harness=WorkspaceHarness(store=store,
        task=WorkspaceTask(actor, "Read the value", (source.evidence_id,))), policy=policy).run()
        for actor in ("W1-1", "W1-2")))
    assert all(result.status == "COMPLETED" for result in results)
    assert dict(client.calls) == {"W1-1": 8, "W1-2": 8}


@pytest.mark.asyncio
async def test_report_provenance_survives_card_truncation_without_expanding_model_input(tmp_path):
    store = GlobalStore(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    source = add_text_evidence(store, "Long source")
    excerpt = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=("text",),
        label="Excerpt", content="Recorded value is 12.", locator="paragraph:2", acquired_by="TEST", discovered_by="W1-1")
    raw = make_report("W1-1", excerpt.evidence_id)
    raw["findings"][0]["statement"] = f"Source {source.evidence_id}, excerpt {excerpt.evidence_id}: recorded value is 12."
    raw["findings"][0]["evidence_ids"].insert(0, source.evidence_id)
    client = Client(lambda actor, count: raw)
    generator = OpenRouterJsonGenerator(client, store=store)
    model_view = {"task_id": "W1-1", "direct_visual_input": True,
        "materials": {source.evidence_id: source.card().to_dict()}, "evidence_view": [],
        "content_observed_evidence_ids": [excerpt.evidence_id]}
    result = await TransformersHiPolicy(generator).decide(workspace_id="W1-1", model_view=model_view, actions=())
    assert result["findings"][0]["evidence_ids"] == [excerpt.evidence_id]
    assert "evidence_provenance" not in model_view
    assert model_view["content_observed_evidence_ids"] == [excerpt.evidence_id]
    assert client.calls["W1-1"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("focus_position", ["omitted", "explicit", "nested"])
async def test_missing_focus_update_preserves_current_focus_without_repeating_the_action(focus_position):
    store = GlobalStore()
    source = add_text_evidence(store, "Recorded value is 12.")
    action = {"action": "READ_RECORD", "arguments": {"record_id": source.evidence_id}}
    if focus_position == "explicit":
        action["focus_evidence_ids"] = []
    elif focus_position == "nested":
        action["arguments"]["focus_evidence_ids"] = []
    client = Client(lambda actor, count: action if count == 1 else make_report(actor, source.evidence_id))
    harness = WorkspaceHarness(store=store, task=WorkspaceTask("W1-1", "Read the value", (source.evidence_id,)))
    harness.memory.focus_evidence_ids = [source.evidence_id]
    result = await WorkspaceRunner(harness=harness, policy=TransformersHiPolicy(OpenRouterJsonGenerator(client))).run()
    assert result.status == "COMPLETED"
    assert client.calls["W1-1"] == 2
    assert len(harness.memory.history) == 1
    assert harness.memory.history[0].focus_evidence_ids == ((source.evidence_id,) if focus_position == "omitted" else ())
    assert ("focus_evidence_ids" in action) is (focus_position == "explicit")
    assert ("focus_evidence_ids" in action["arguments"]) is (focus_position == "nested")


def test_missing_focus_does_not_repair_missing_arguments_or_allow_an_unknown_action():
    view = {"focus_evidence_ids": ["E1"], "materials": {"E1": {}}}
    for raw in ({"action": "READ_RECORD"}, {"action": "EXECUTE_SHELL", "arguments": {}}):
        output = _normalize_hi_output(raw, "W1-1", view)
        with pytest.raises(ValueError):
            _validate_hi_output(output, workspace_id="W1-1", model_view=view, actions=())


def test_conflicting_nested_focus_is_not_silently_chosen():
    from foren_searcher.inspect.contracts import parse_inspect_call
    raw = {"action": "INSPECT", "arguments": {"targets": ["E1"],
           "request": {"operation": "get_frames", "params": {"times_s": [0.4, 0.5]}},
           "focus_evidence_ids": []}}
    normalized = _normalize_hi_output(raw, "W1-1", {"focus_evidence_ids": ["E1"]})
    parse_inspect_call(normalized["arguments"])
    assert normalized["focus_evidence_ids"] == []
    raw["focus_evidence_ids"] = ["E1"]
    with pytest.raises(ValueError):
        parse_inspect_call(_normalize_hi_output(raw, "W1-1", {})["arguments"])
