"""真实编码的 HLS 分片经过 HTTP 获取、无损封装和 INSPECT 解码。"""

import asyncio
from fractions import Fraction
import json
from pathlib import Path
import re

import av
import httpx
import numpy as np
import pytest

from foren_searcher.inspect import acquisition
from foren_searcher.inspect.acquisition import DirectMediaAcquirer
from foren_searcher.inspect.backends.base import BackendFailure
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.hls import HlsDownloader
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.search.openers.http import SafeHttpClient
from foren_searcher.store import GlobalStore


def encode_hls(root: Path, *, fragmented=False, single_file=False, audio=False, only_audio=False):
    root.mkdir(parents=True, exist_ok=True)
    options = {"hls_time":"1", "hls_list_size":"0", "hls_playlist_type":"vod"}
    if fragmented:
        options["hls_segment_type"] = "fmp4"
        options["hls_fmp4_init_filename"] = (root/'init.mp4').as_posix()
    if single_file:
        options["hls_flags"] = "single_file"
    with av.open(str(root/'index.m3u8'), "w", format="hls", options=options) as output:
        video = output.add_stream("libx264", rate=10) if not only_audio else None
        if video is not None:
            video.width, video.height, video.pix_fmt = 96, 64, "yuv420p"
            video.options = {"preset":"ultrafast", "g":"10", "sc_threshold":"0"}
        sound = output.add_stream("aac", rate=16000) if audio or only_audio else None
        if sound is not None:
            sound.layout = "mono"
        for index in range(40):
            color = (220, 20, 20) if index < 20 else (20, 20, 220)
            pixels = np.full((64, 96, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            if video is not None:
                for packet in video.encode(frame):
                    output.mux(packet)
            if sound is not None:
                samples = (0.2 * np.sin(np.arange(index*1600, (index+1)*1600) * 2*np.pi*440/16000)).astype(np.float32)[None, :]
                chunk = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                chunk.sample_rate = 16000
                chunk.pts, chunk.time_base = index*1600, Fraction(1, 16000)
                for packet in sound.encode(chunk):
                    output.mux(packet)
        for stream in [item for item in (video,sound) if item is not None]:
            for packet in stream.encode(None):
                output.mux(packet)
    files = {"/"+path.name:path.read_bytes() for path in root.iterdir() if path.is_file()}
    files['/index.m3u8'] = files['/index.m3u8'].replace((root/'init.mp4').as_posix().encode(), b'init.mp4')
    return files


def mock_http(files, requests, *, wrong_range=False):
    def handler(request):
        requests.append(str(request.url))
        body = files.get(request.url.path)
        if body is None:
            return httpx.Response(404)
        window = request.headers.get("range")
        if window:
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", window).groups())
            if wrong_range:
                return httpx.Response(200, content=body)
            return httpx.Response(206, headers={"Content-Range":f"bytes {start}-{end}/{len(body)}"}, content=body[start:end+1])
        return httpx.Response(200, content=body)
    return SafeHttpClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['external_audio', 'discontinuity'])
async def test_hls_preserves_renditions_and_segment_timeline(tmp_path, monkeypatch, case):
    first = encode_hls(tmp_path/'one')
    second = encode_hls(tmp_path/'two', only_audio=case=='external_audio')
    files = {**{'/one'+path:body for path,body in first.items()}, **{'/two'+path:body for path,body in second.items()}}
    if case == 'external_audio':
        files['/master.m3u8'] = b'#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="main",DEFAULT=YES,URI="two/index.m3u8"\n#EXT-X-STREAM-INF:BANDWIDTH=100000,AUDIO="audio"\none/index.m3u8\n'
        expected_duration, expected_frames = 4.0, 40
    else:
        lines = ['#EXTM3U','#EXT-X-TARGETDURATION:1','#EXT-X-PLAYLIST-TYPE:VOD']
        for prefix,values in [('one',first),('two',second)]:
            if prefix=='two': lines.append('#EXT-X-DISCONTINUITY')
            for path in sorted(path for path in values if path.endswith('.ts')):
                lines.extend(['#EXTINF:1,',prefix+path])
        lines.append('#EXT-X-ENDLIST')
        files['/master.m3u8'] = ('\n'.join(lines)+'\n').encode()
        expected_duration, expected_frames = 8.0, 80
    requests = []
    monkeypatch.setattr(acquisition,'SafeHttpClient',lambda *args,**kwargs:mock_http(files,requests))
    acquirer = DirectMediaAcquirer()
    try:
        result = await acquirer.acquire('https://93.184.216.34/master.m3u8',destination=tmp_path/'result.mkv',policy=InspectPolicy())
        with av.open(str(result.path)) as video:
            assert len(video.streams.audio) == (1 if case=='external_audio' else 0)
            assert len(video.streams.video) == 1
            frames = list(video.decode(video=0))
            assert len(frames) == expected_frames
            assert abs(frames[-1].time - frames[0].time - (expected_duration-.1)) < .15
            assert all(right.time > left.time for left,right in zip(frames,frames[1:]))
    finally:
        await acquirer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(('fragmented','single_file','audio'), [(False,False,False),(False,False,True),(True,False,True),(True,True,True)])
async def test_hls_video_is_acquired_decoded_and_reused(tmp_path, monkeypatch, fragmented, single_file, audio):
    files = encode_hls(tmp_path/'source', fragmented=fragmented, single_file=single_file, audio=audio)
    files['/master.m3u8'] = b'#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100000,RESOLUTION=96x64\nindex.m3u8\n'
    requests = []
    monkeypatch.setattr(acquisition, "SafeHttpClient", lambda *args, **kwargs: mock_http(files, requests))
    store = GlobalStore(tmp_path/'case.sqlite3', tmp_path/'artifacts')
    rid, _ = store.get_or_create_candidate(identity_key='video:hls', source_url='https://93.184.216.34/watch', candidate_type='video')
    metadata = {'extractor_key':'HTML5MediaEmbed','formats':[{'url':'https://93.184.216.34/index.m3u8',
        'manifest_url':'https://93.184.216.34/master.m3u8','protocol':'m3u8_native','vcodec':'avc1','acodec':'mp4a'}]}
    page = store.commit_material_evidence(candidate_id=rid, material_kind='page', material_status='ready',
        artifact_bytes=json.dumps(metadata).encode(), media_type='application/json', suffix='.json',
        modalities=('text',), label='public HLS video', content='video metadata', locator='fields',
        acquired_by='SEARCH/open/yt-dlp', discovered_by='W1-1')
    service = create_inspect_service(store=store, policy=InspectPolicy(maximum_file_bytes=2*1024**2))
    call = {'targets':[page.evidence_id], 'request':{'operation':'get_frames','params':{'times_s':[0.5,2.5]}}}
    try:
        response = await service.execute(call, workspace_id='W1-1')
        assert response.status == 'ready', response.to_dict()
        video = store.candidate_evidence(rid, 'video')[0][2]
        assert any(card.acquired_by=='INSPECT/media_profile' and card.derived_from==(video.evidence_id,) for card in response.evidence)
        with av.open(str(store.artifact_path(video.artifact_ref))) as container:
            assert bool(container.streams.audio) is audio
            assert abs(container.duration/av.time_base - 4.0) < .2
        for frame, channel in zip(response.result.frames, [0,2], strict=True):
            with av.open(str(store.artifact_path(store.evidence(frame.evidence_id).artifact_ref))) as container:
                pixels = next(container.decode(video=0)).to_ndarray(format='rgb24')
                assert pixels[...,channel].mean() > 180
        count = len(requests)
        reused = await service.execute(call, workspace_id='W1-2')
        assert reused.to_dict() == response.to_dict()
        assert len(requests) == count
        assert list(store.incoming_root.iterdir()) == []
        assert store.healthcheck().status == 'ok'
    finally:
        await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['private','local_file','encrypted','live','range_ignored','oversize'])
async def test_hls_failures_do_not_publish_partial_media(tmp_path, monkeypatch, kind):
    segment = 'clip.ts'
    extra = ''
    end = '#EXT-X-ENDLIST\n'
    if kind == 'private': segment = 'http://127.0.0.1/private'
    if kind == 'local_file': segment = 'file:///C:/private.txt'
    if kind == 'encrypted': extra = '#EXT-X-KEY:METHOD=AES-128,URI="key"\n'
    if kind == 'live': end = ''
    if kind == 'range_ignored': extra = '#EXT-X-BYTERANGE:5@0\n'
    files = {'/index.m3u8':('#EXTM3U\n#EXT-X-TARGETDURATION:1\n'+extra+'#EXTINF:1,\n'+segment+'\n'+end).encode(),
             '/clip.ts':b'x'*2048}
    requests = []
    monkeypatch.setattr(acquisition, 'SafeHttpClient', lambda *args, **kwargs: mock_http(files, requests, wrong_range=kind=='range_ignored'))
    output = tmp_path/'result.bin'
    acquirer = DirectMediaAcquirer()
    try:
        with pytest.raises(BackendFailure):
            await acquirer.acquire('https://93.184.216.34/index.m3u8', destination=output,
                policy=InspectPolicy(maximum_file_bytes=1024 if kind=='oversize' else 10000))
        assert not output.exists()
        assert not list(tmp_path.glob('hls-*'))
        if kind in {'private','local_file','encrypted','live'}:
            assert all('/index.m3u8' in url for url in requests)
    finally:
        await acquirer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['timeout','cancel'])
async def test_hls_cancellation_closes_downloads_and_cleans_files(tmp_path, monkeypatch, mode):
    started, closed = asyncio.Event(), asyncio.Event()
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            yield b'partial segment'
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                asyncio.current_task().cancel()
                raise
        async def aclose(self):
            closed.set()
    def handler(request):
        if request.url.path.endswith('.m3u8'):
            return httpx.Response(200,content=b'#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nclip.ts\n#EXT-X-ENDLIST\n')
        return httpx.Response(200,stream=SlowBody())
    monkeypatch.setattr(acquisition,'SafeHttpClient',lambda *args,**kwargs:SafeHttpClient(transport=httpx.MockTransport(handler)))
    acquirer = DirectMediaAcquirer()
    output = tmp_path/'result.mkv'
    task = asyncio.create_task(acquirer.acquire('https://93.184.216.34/index.m3u8',destination=output,
        policy=InspectPolicy(acquisition_timeout_s=.1 if mode=='timeout' else 10)))
    try:
        await asyncio.wait_for(started.wait(),1)
        if mode=='cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            with pytest.raises(BackendFailure) as error: await task
            assert error.value.code=='TIMEOUT'
        assert closed.is_set()
        assert not output.exists() and not list(tmp_path.glob('hls-*'))
    finally:
        await acquirer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('limiting_field', ['maximum_file_bytes', 'maximum_download_bytes'])
async def test_parallel_hls_downloads_share_one_byte_budget(tmp_path, monkeypatch, limiting_field):
    files = {'/index.m3u8':b'#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\none.ts\n#EXTINF:1,\ntwo.ts\n#EXT-X-ENDLIST\n',
             '/one.ts':b'x'*600,'/two.ts':b'y'*600}
    requests = []
    monkeypatch.setattr(acquisition,'SafeHttpClient',lambda *args,**kwargs:mock_http(files,requests))
    acquirer = DirectMediaAcquirer()
    try:
        with pytest.raises(BackendFailure) as error:
            await acquirer.acquire('https://93.184.216.34/index.m3u8',destination=tmp_path/'result.mkv',
                policy=InspectPolicy(**{limiting_field:1024}))
        assert error.value.code=='RESOURCE_LIMIT'
        assert not (tmp_path/'result.mkv').exists() and not list(tmp_path.glob('hls-*'))
    finally:
        await acquirer.aclose()
