import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


pytestmark = [pytest.mark.live, pytest.mark.asr]


@pytest.mark.asyncio
async def test_real_asr_does_not_invent_speech_from_silence_tones_or_noise(tmp_path):
    if not os.environ.get('FOREN_WHISPER_MODEL'):
        pytest.skip('Set FOREN_WHISPER_MODEL to a local faster-whisper model')
    rate = 16000
    silence = np.zeros(rate*5, dtype=np.int16)
    tones = silence.copy()
    for index, frequency in enumerate((220, 440, 880)):
        t = np.arange(rate//3)/rate
        tones[(index+1)*rate:(index+1)*rate+len(t)] = np.sin(t*2*np.pi*frequency)*8000
    noise = np.random.default_rng(784).normal(0, 2000, rate*5).clip(-32768,32767).astype(np.int16)
    store = GlobalStore(tmp_path/'store.sqlite3',tmp_path/'artifacts')
    async with create_inspect_service(store=store) as service:
        for index, samples in enumerate((silence, tones, noise)):
            path = tmp_path/f'clip-{index}.wav'
            with wave.open(str(path),'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
                stream.writeframes(samples.astype('<i2').tobytes())
            evidence = store.commit_ingested_evidence(source_ref=f'I{index}',modalities=('audio',),media_type='audio/wav',
                label=path.name,path=path,original_filename=path.name)
            response = await service.execute({'targets':[evidence.evidence_id],'request':{'operation':'asr','params':{'language':'en'}}},workspace_id='W1-1')
            assert response.status == 'ready'
            assert response.result.segments == ()
            saved = json.loads(store.evidence(response.result_evidence_id).content)
            assert saved['vad_filter'] is True and saved['speech_duration_s'] == 0
            assert saved['coverage']['processed_ranges'] == [[0,5]]
