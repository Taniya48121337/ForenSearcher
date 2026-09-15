from fractions import Fraction

import av
import numpy as np
import pytest

from foren_searcher.inspect.backends.base import MediaProfileData, PreparedMedia
from foren_searcher.inspect.backends.local import LocalInspectBackend
from inspect_helpers import make_video


class CountingVideo:
    def __init__(self, container, counts):
        self.container, self.counts = container, counts

    def __enter__(self):
        self.container.__enter__()
        return self

    def __exit__(self, *args):
        return self.container.__exit__(*args)

    @property
    def streams(self):
        return self.container.streams

    def seek(self, *args, **kwargs):
        self.counts['seeks'] += 1
        return self.container.seek(*args, **kwargs)

    def decode(self, *args, **kwargs):
        for frame in self.container.decode(*args, **kwargs):
            self.counts['frames'] += 1
            yield frame


@pytest.mark.parametrize('codec', ['mpeg4', 'libx264'])
@pytest.mark.parametrize('requests,span,expected,seeks', [
    ((.01, .02, .125, .2, .24), None, [0., .25], 1),
    ((.2, .01, .24, .125), None, [.25, 0.], 1),
    ((.26, .3, .4, .49, 1.26, 1.3), None, [.25, .5, 1.25], 2),
    ((.26, .3, .4, .49, .7, .74), (.25, .75), [.25, .5], 2),
])
def test_dense_frame_requests_reuse_decoded_neighbors(tmp_path, monkeypatch, codec, requests, span, expected, seeks):
    path = tmp_path / 'scene.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream(codec, rate=4)
        stream.width, stream.height, stream.pix_fmt = 96, 64, 'yuv420p'
        for i in range(12):
            pixels = np.full((64, 96, 3), (i * 17, 20, 200 - i * 11), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    counts = {'seeks': 0, 'frames': 0}
    real_open = av.open
    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: CountingVideo(real_open(*args, **kwargs), counts))
    media = PreparedMedia('E1', path, 'video', MediaProfileData('video', duration_s=3.))
    backend = LocalInspectBackend()
    result = backend._decode(media, requests, 'INSPECT/get_frames', range_s=span)
    assert not result.issues
    assert [row['time_s'] for row in result.result['frames']] == expected
    assert counts['seeks'] == seeks
    assert all(item.parent_evidence_id == 'E1' for item in result.artifacts)
    assert [item.locator['time_s'] for item in result.artifacts] == expected
    # 单个时间点独立取帧，核对实际像素、持续时长和首次请求对应的 handle。
    reference = []
    seen = set()
    for index, requested in enumerate(requests):
        single = backend._decode(media, (requested,), 'INSPECT/get_frames', range_s=span)
        artifact = single.artifacts[0]
        if artifact.locator['time_s'] not in seen:
            seen.add(artifact.locator['time_s'])
            reference.append((f'frame:{index}', artifact.data, artifact.locator))
    assert [(item.handle, item.data, item.locator) for item in result.artifacts] == reference


def test_failed_frame_encoding_can_be_retried_in_the_same_request(tmp_path, monkeypatch):
    import foren_searcher.inspect.backends.local as local

    path = tmp_path / 'retry.avi'
    make_video(path, fps=4., frames=8)
    real_png = local._png
    encoded = []
    def encode(frame):
        encoded.append(True)
        if len(encoded) == 1:
            raise ValueError('temporary conversion failure')
        return real_png(frame)
    monkeypatch.setattr(local, '_png', encode)
    counts = {'seeks': 0, 'frames': 0}
    real_open = av.open
    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: CountingVideo(real_open(*args, **kwargs), counts))
    media = PreparedMedia('E1', path, 'video', MediaProfileData('video', duration_s=2.))
    result = LocalInspectBackend()._decode(media, (.2, .21, .22), 'INSPECT/get_frames')
    assert len(result.issues) == 1
    assert result.issues[0].locator == {'requested_time_s': .2}
    assert len(result.artifacts) == 1
    assert result.artifacts[0].handle == 'frame:1'
    assert result.artifacts[0].locator['time_s'] == .25
    assert counts['seeks'] == 2


def test_seek_failure_invalidates_the_previous_window(tmp_path, monkeypatch):
    path = tmp_path / 'seek.avi'
    make_video(path, fps=4., frames=8)
    counts = {'seeks': 0, 'frames': 0}
    class FailingVideo(CountingVideo):
        def seek(self, *args, **kwargs):
            if self.counts['seeks'] == 1:
                self.counts['seeks'] += 1
                raise ValueError('temporary seek failure')
            return super().seek(*args, **kwargs)
    real_open = av.open
    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: FailingVideo(real_open(*args, **kwargs), counts))
    media = PreparedMedia('E1', path, 'video', MediaProfileData('video', duration_s=2.))
    result = LocalInspectBackend()._decode(media, (.01, .3, .02, .2), 'INSPECT/get_frames')
    assert len(result.issues) == 1 and result.issues[0].locator == {'requested_time_s': .3}
    assert [item.locator['time_s'] for item in result.artifacts] == [0., .25]
    assert counts['seeks'] == 3
