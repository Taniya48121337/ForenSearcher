import json

import cv2
import numpy as np
import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore
from test_temporal_alignment import write_video


@pytest.mark.asyncio
@pytest.mark.parametrize("fps_b", [5, 10, 15])
async def test_tempo_change_preserves_moving_detail_against_repeated_background(tmp_path, fps_b):
    rng = np.random.default_rng(193)
    background = cv2.resize(rng.integers(0, 180, (15, 20, 3), dtype=np.uint8), (160, 120))
    frames = []
    for i in range(180):
        frame = background.copy()
        x, y = int(65 + 45*np.sin(i*.06)), int(45 + 30*np.cos(i*.09))
        cv2.rectangle(frame, (x, y), (x+35, y+25), (240, 240, 240), -1)
        cv2.putText(frame, str(i), (x+2, y+18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        frames.append(frame)
    paths = (tmp_path/'a.avi', tmp_path/'b.avi')
    write_video(paths[0], frames, 10)
    write_video(paths[1], frames[30:150], fps_b)
    store = GlobalStore(tmp_path/'store.sqlite3', tmp_path/'artifacts')
    media = [store.commit_ingested_evidence(source_ref=f'I{i}',media_type='video/x-msvideo',modalities=('video',),
        label=p.name,path=p,original_filename=p.name).evidence_id for i,p in enumerate(paths)]
    async with create_inspect_service(store=store) as service:
        result = await service.execute({'targets':media,'request':{'operation':'align_media','params':{'mode':'visual'}}},workspace_id='W1-1')
    assert result.status == 'ready'
    saved = json.loads(store.evidence(result.result_evidence_id).content)
    best = saved['matches'][0]
    assert best['range_a_s'] == pytest.approx([3,15], abs=.3)
    assert best['range_b_s'] == pytest.approx([0,120/fps_b], abs=.3)
    assert best['time_mapping']['scale'] == pytest.approx(10/fps_b, abs=.03)
    assert best['time_mapping']['offset_s'] == pytest.approx(-30/fps_b, abs=.2)
    assert best['time_mapping']['playback_rate_b_relative_to_a'] == pytest.approx(fps_b/10, abs=.03)
