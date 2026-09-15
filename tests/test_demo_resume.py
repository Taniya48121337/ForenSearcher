import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import foren_searcher.demo as demo
from foren_searcher.contracts import Finding, WorkspaceReport, WorkspaceTask
from foren_searcher.memory import GlobalViewBuilder, MemoryPolicy
from foren_searcher.store import GlobalStore
from foren_searcher.workspace import WorkspaceHarness
from test_framework_runtime import add_text_evidence


def interrupted(root):
    store = GlobalStore(root / 'investigation.sqlite3', root / 'artifacts')
    evidence = add_text_evidence(store, 'There are 18 devices.')
    task = 'Count the devices.'
    originals = (evidence.evidence_id,)
    store.ensure_investigation(task=task, original_material_ids=originals)
    view = GlobalViewBuilder(store, MemoryPolicy()).build(task=task, original_material_ids=originals, latest_round_index=None)
    workspace = WorkspaceTask('W1-1', task, originals)
    store.start_round(1, gia_input=view.to_dict(), gia_decision={'decision':'INVESTIGATE',
        'tasks':[{'task':task, 'materials':list(originals)}]}, tasks=(workspace,))
    store.add_workspace_report(WorkspaceReport('W1-1', 'COMPLETED', (
        Finding('F1', 'W1-1', 'There are 18 devices.', originals),), originals))
    store.set_workspace_run_status('W1-1', 'reported')
    store.finish_round(1)
    previous = {'task':task, 'status':'failed', 'decision':None, 'error':{'type':'TimeoutError','message':'GIA timeout'}}
    (root / 'result.json').write_text(json.dumps(previous), encoding='utf-8')
    return store, task


@pytest.mark.asyncio
async def test_resume_after_gia_failure_reuses_completed_hi_and_original_evidence(tmp_path, monkeypatch):
    store, task = interrupted(tmp_path)
    calls = []
    class Policy:
        async def decide(self, **kwargs):
            calls.append(kwargs)
            assert len(kwargs['model_view']['latest_reports']) == 1
            return {'decision':'FINISH', 'answer':'There are 18 devices (E1).'}
    class NeverHi:
        async def decide(self, **kwargs):
            raise AssertionError('Completed HI must not run again')
    class Policies:
        gia_model = SimpleNamespace()
        hi_model = SimpleNamespace()
        gia_policy = Policy()
        hi_policy = NeverHi()
        async def preload(self): pass
        async def aclose(self): pass
    monkeypatch.setattr(demo, 'capability_report', lambda **kwargs: {'configured_for_text_demo':True})
    monkeypatch.setattr(demo, 'harness1_workspace_type', lambda root: WorkspaceHarness)
    monkeypatch.setattr(demo, '_create_search', lambda *args: None)
    monkeypatch.setattr(demo, 'create_demo_policies', lambda root, **kwargs: Policies())
    value = await demo.run_demo(task=task, output_dir=tmp_path, resume=True,
        model_backend='local', harness_root=tmp_path, strict_semantics=False)
    assert value['decision']['answer'] == 'There are 18 devices (E1).'
    assert len(calls) == len(store.evidence_records()) == len(store.round_records()) == 1
    assert store.trajectory_events(event_type='investigation_resume')[0]['payload']['previous_error']['type'] == 'TimeoutError'
    monkeypatch.setattr(demo, 'capability_report', lambda **kwargs: pytest.fail('Finished result must not load models'))
    assert await demo.run_demo(task=task, output_dir=tmp_path, resume=True, model_backend='local') == value
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['task', 'media'])
async def test_resume_rejects_changed_input_without_overwriting_saved_result(tmp_path, change):
    store, task = interrupted(tmp_path)
    before = (tmp_path / 'result.json').read_bytes()
    with pytest.raises(demo.DemoConfigurationError):
        await demo.run_demo(task='Another question' if change == 'task' else task,
            media_paths=('new.pdf',) if change == 'media' else (), output_dir=tmp_path, resume=True)
    assert (tmp_path / 'result.json').read_bytes() == before
    assert not store.trajectory_events(event_type='investigation_resume')


def test_investigation_lock_excludes_live_process_and_releases_for_resume(tmp_path):
    script = '''from pathlib import Path
import sys
from foren_searcher.demo import _demo_run_lock, DemoConfigurationError
try:
    with _demo_run_lock(Path(sys.argv[1])): pass
except DemoConfigurationError:
    sys.exit(17)
'''
    command = [sys.executable, '-c', script, str(tmp_path)]
    with demo._demo_run_lock(tmp_path):
        assert subprocess.run(command, capture_output=True).returncode == 17
    assert subprocess.run(command, capture_output=True).returncode == 0
