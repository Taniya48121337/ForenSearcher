import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from foren_searcher.inspect.backends.base import BackendOutcome, MediaProfileData, PreparedMedia
from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.contracts import parse_inspect_call
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


class ModelBackend(LocalInspectBackend):
    def __init__(self, fail=False):
        super().__init__()
        self.fail, self.calls = fail, 0

    async def locate_frames(self, *args):
        self.calls += 1
        self._clip = (object(), object(), 'cuda', 'clip')
        # 后端可以在线程内再使用工作线程；共享锁不能替换其内部模型锁。
        def nested():
            with self._model_lock:
                return True
        assert await asyncio.wait_for(asyncio.to_thread(nested), 3)
        if self.fail:
            raise ValueError('model failed')
        return BackendOutcome({'results': []})

    async def get_frames(self, *args):
        return BackendOutcome({'frames': []})


def request(operation):
    params = {'query': {'type': 'text', 'text': 'red'}} if operation == 'locate_frames' else {'count': 2}
    return parse_inspect_call({'targets': ['E1'], 'request': {'operation': operation, 'params': params}})


@pytest.mark.asyncio
@pytest.mark.parametrize('fail', [False, True])
async def test_gpu_operations_release_their_cache_and_can_run_again(tmp_path, monkeypatch, fail):
    monkeypatch.setenv('FOREN_CLIP_MODEL', 'clip')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = ModelBackend(fail)
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    async with create_inspect_service(store=store, backend=backend) as service:
        dispatcher = service.dispatcher
        dispatcher.coordinate_local_models(threading.RLock())
        media = (PreparedMedia('E1', Path('unused'), 'video', MediaProfileData('video', duration_s=2)),)
        for _ in range(2):
            if fail:
                with pytest.raises(ValueError, match='model failed'):
                    await dispatcher.execute(request('locate_frames'), media, policy=service.policy)
            else:
                await dispatcher.execute(request('locate_frames'), media, policy=service.policy)
            assert backend._clip is None
        assert backend.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_cpu_work_remains_parallel_and_cancelled_gpu_waiter_does_not_start(tmp_path, monkeypatch, cancel):
    monkeypatch.setenv('FOREN_CLIP_MODEL', 'clip')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = ModelBackend()
    store = GlobalStore(tmp_path / 'state.sqlite3', tmp_path / 'artifacts')
    lock, held, release = threading.RLock(), threading.Event(), threading.Event()
    def agent_generation():
        with lock:
            held.set()
            assert release.wait(5)
    async with create_inspect_service(store=store, backend=backend) as service:
        dispatcher = service.dispatcher
        dispatcher.coordinate_local_models(lock)
        owner = asyncio.create_task(asyncio.to_thread(agent_generation))
        assert await asyncio.to_thread(held.wait, 3)
        media = (PreparedMedia('E1', Path('unused'), 'video', MediaProfileData('video', duration_s=2)),)
        pending = asyncio.create_task(dispatcher.execute(request('locate_frames'), media, policy=service.policy))
        try:
            result = await asyncio.wait_for(dispatcher.execute(request('get_frames'), media, policy=service.policy), 3)
            assert result.result == {'frames': []} and backend.calls == 0
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        finally:
            release.set()
            await owner
        if not cancel:
            await pending
        await dispatcher.aclose()
        assert backend.calls == (0 if cancel else 1)
        assert backend._clip is None


@pytest.mark.parametrize('whisper_device', ['cpu', 'cuda'])
def test_gpu_release_preserves_cpu_models_and_ocr(monkeypatch, whisper_device):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = LocalInspectBackend()
    backend._clip = (object(), object(), 'cuda', 'clip')
    backend._grounding = (object(), object(), 'cuda:0', 'grounding')
    backend._audio_model = (object(), object(), 'cuda', 'audio', 'family')
    backend._vlm = (object(), object(), 'cpu', 'vlm')
    backend._whisper = (object(), whisper_device, 'whisper')
    backend._ocr_engine = object()
    vlm, whisper, ocr = backend._vlm, backend._whisper, backend._ocr_engine
    backend.release_gpu_models()
    assert backend._clip is backend._grounding is backend._audio_model is None
    assert backend._vlm is vlm and backend._ocr_engine is ocr
    assert backend._whisper is (whisper if whisper_device == 'cpu' else None)
    backend.release_gpu_models()


@pytest.mark.parametrize('resolved_device', ['cpu', 'cuda'])
def test_whisper_auto_uses_the_resolved_device_for_scheduling_and_release(monkeypatch, resolved_device):
    monkeypatch.setenv('FOREN_WHISPER_MODEL', 'whisper')
    monkeypatch.setenv('FOREN_WHISPER_DEVICE', 'auto')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = LocalInspectBackend()
    assert backend.uses_gpu_model('asr')
    cache = (SimpleNamespace(model=SimpleNamespace(device=resolved_device)), 'auto', 'whisper')
    backend._whisper = cache
    assert backend.uses_gpu_model('asr') == (resolved_device == 'cuda')
    backend.release_gpu_models()
    assert backend._whisper is (cache if resolved_device == 'cpu' else None)
