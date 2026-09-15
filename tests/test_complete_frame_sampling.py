from dataclasses import replace
from fractions import Fraction

import av
import numpy as np
import pytest

from foren_searcher.inspect.backends.base import PreparedMedia
from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.contracts import GetFramesParams
from foren_searcher.inspect.policy import InspectPolicy


def make_vfr(path, codec='mpeg4'):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream(codec, rate=20)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        stream.time_base = Fraction(1, 20)
        for index, pts in enumerate([0, 5, 10, 18, 28, 33, 48]):
            pixels = np.full((64, 96, 3), (index * 30, 40, 200 - index * 20), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 20)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.asyncio
@pytest.mark.parametrize('codec', ['mpeg4', 'libx264'])
@pytest.mark.parametrize('count', [7, 8, 12])
async def test_sufficient_budget_returns_every_actual_vfr_frame(tmp_path, codec, count):
    path = tmp_path / 'vfr.mp4'
    make_vfr(path, codec)
    backend = LocalInspectBackend()
    profile = await backend.probe(path, 'video')
    assert profile.frame_count == 7
    media = PreparedMedia('E1', path, 'video', profile)
    with av.open(str(path)) as source:
        times = tuple(float(frame.pts * frame.time_base) for frame in source.decode(video=0))
    result = await backend.get_frames((media,), GetFramesParams(count=count), InspectPolicy())
    assert tuple(row['time_s'] for row in result.result['frames']) == times
    reference = backend._decode(media, times, 'INSPECT/get_frames')
    assert [(a.data, a.locator) for a in result.artifacts] == [(a.data, a.locator) for a in reference.artifacts]
    assert [issue.code for issue in result.issues] == (['COUNT_REDUCED'] if count > 7 else [])


@pytest.mark.asyncio
async def test_complete_sampling_respects_half_open_range(tmp_path):
    path = tmp_path / 'vfr.mp4'
    make_vfr(path)
    backend = LocalInspectBackend()
    media = PreparedMedia('E1', path, 'video', await backend.probe(path, 'video'))
    result = await backend.get_frames((media,), GetFramesParams(count=8, range_s=(.25, 1.4)), InspectPolicy())
    assert [a.locator['time_s'] for a in result.artifacts] == [.25, .5, .9]
    assert all(.25 <= a.locator['time_s'] < 1.4 for a in result.artifacts)


@pytest.mark.asyncio
async def test_explicit_times_keep_order_and_nearest_frame_selection(tmp_path, monkeypatch):
    path = tmp_path / 'vfr.mp4'
    make_vfr(path)
    backend = LocalInspectBackend()
    media = PreparedMedia('E1', path, 'video', await backend.probe(path, 'video'))
    def unexpected(*args):
        pytest.fail('explicit timestamps must not trigger a full-frame scan')
    monkeypatch.setattr(backend, '_complete_frame_times', unexpected)
    times = (1.7, .3, .31, 0.)
    result = await backend.get_frames((media,), GetFramesParams(times_s=times), InspectPolicy())
    reference = backend._decode(media, times, 'INSPECT/get_frames')
    assert [(a.data, a.locator) for a in result.artifacts] == [(a.data, a.locator) for a in reference.artifacts]
    assert 'these selections returned 3 distinct frames' in str(result.issues[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('declared_count', [None, 0, 3, 100])
async def test_unavailable_or_inconsistent_frame_count_preserves_sampling(tmp_path, declared_count):
    path = tmp_path / 'vfr.mp4'
    make_vfr(path)
    backend = LocalInspectBackend()
    profile = replace(await backend.probe(path, 'video'), frame_count=declared_count)
    media = PreparedMedia('E1', path, 'video', profile)
    params, policy = GetFramesParams(count=4), InspectPolicy()
    reference = backend._decode(media, backend._frame_times(profile, params, policy), 'INSPECT/get_frames')
    result = await backend.get_frames((media,), params, policy)
    assert [(a.data, a.locator) for a in result.artifacts] == [(a.data, a.locator) for a in reference.artifacts]
    assert len(result.artifacts) <= 4


@pytest.mark.asyncio
async def test_incorrect_frame_count_scan_stops_after_budget_plus_one(tmp_path, monkeypatch):
    path = tmp_path / 'vfr.mp4'
    make_vfr(path)
    backend = LocalInspectBackend()
    profile = replace(await backend.probe(path, 'video'), frame_count=1)
    media = PreparedMedia('E1', path, 'video', profile)
    real_open, decoded = av.open, []
    class CountingSource:
        def __init__(self, *args, **kwargs):
            self.source = real_open(*args, **kwargs)
            self.streams = self.source.streams
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.source.close()
        def decode(self, stream):
            for frame in self.source.decode(stream):
                decoded.append(frame.pts)
                yield frame
    monkeypatch.setattr(av, 'open', CountingSource)
    assert backend._complete_frame_times(media, 4, None) is None
    assert len(decoded) == 5
