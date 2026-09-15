"""视频提取器的后续请求、跳转、按需读取和取消使用同一网络边界。"""

import asyncio
import json
from http.cookiejar import CookieJar

import httpx
import pytest

from foren_searcher.search.contracts import CandidateSource, OpenSourceRequest, SearchFailure
from foren_searcher.search.openers.http import SafeHttpClient
from foren_searcher.search.openers.video import VideoSourceOpener
from foren_searcher.search.openers.ytdlp_http import extract_metadata
from foren_searcher.search.policy import SearchPolicy
from yt_dlp.networking import Request
from yt_dlp.networking.exceptions import HTTPError


PUBLIC = 'https://93.184.216.34/'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['hls', 'redirect_private', 'embedded_private'])
async def test_real_ytdlp_uses_guarded_transport_for_embedded_media(kind):
    requests = []
    manifest = 'http://127.0.0.1/private.m3u8' if kind == 'embedded_private' else PUBLIC+'stream.m3u8'
    def handler(request):
        requests.append(str(request.url))
        if request.url.path == '/watch':
            if kind == 'redirect_private':
                return httpx.Response(302, headers={'Location':'http://127.0.0.1/private'})
            return httpx.Response(200, headers={'Content-Type':'text/html'},
                text=f'<html><head><title>Public video</title></head><body><video controls src="{manifest}"></video></body></html>')
        if request.url.path == '/stream.m3u8':
            return httpx.Response(200, headers={'Content-Type':'application/vnd.apple.mpegurl'},
                text='#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nsegment.ts\n#EXT-X-ENDLIST\n')
        raise AssertionError(f'unexpected outgoing request: {request.url}')
    http = SafeHttpClient(transport=httpx.MockTransport(handler))
    try:
        call = VideoSourceOpener(http).open(CandidateSource(PUBLIC+'watch',candidate_type='video'),
            OpenSourceRequest(PUBLIC+'watch','video'))
        if kind == 'hls':
            result = await call
            metadata = json.loads(result.materials[0].artifact_bytes)
            assert metadata['formats'] and any(item['protocol'].startswith('m3u8') for item in metadata['formats'])
            assert PUBLIC+'stream.m3u8' in requests
        else:
            with pytest.raises(SearchFailure) as error:
                await call
            assert error.value.error.code == 'ACCESS_RESTRICTED'
        assert all('127.0.0.1' not in url for url in requests)
    finally:
        await http.aclose()


class Extractor:
    def __init__(self, options): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


@pytest.mark.asyncio
async def test_large_direct_video_metadata_only_consumes_requested_prefix():
    seen = {'chunks':0, 'closed':False}
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(256):
                seen['chunks'] += 1
                yield b'x'*4096
        async def aclose(self): seen['closed'] = True
    class Reader(Extractor):
        def extract_info(self, url, **kwargs):
            with self.urlopen(url) as response:
                assert response.read(512) == b'x'*512
            return {'id':'video','webpage_url':url}
    http = SafeHttpClient(transport=httpx.MockTransport(lambda request:httpx.Response(200,
        headers={'Content-Type':'video/mp4','Content-Length':str(512*1024**2)},stream=Body())))
    try:
        result = await extract_metadata(http,http.policy,Reader,{},PUBLIC+'large.mp4')
        assert result['id']=='video'
        assert seen['chunks'] <= 4 and seen['closed']
    finally:
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['timeout','cancel'])
async def test_metadata_stream_is_closed_and_worker_stops_on_interruption(mode):
    started, closed = asyncio.Event(), asyncio.Event()
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b''
        async def aclose(self): closed.set()
    class Reader(Extractor):
        def extract_info(self,url,**kwargs):
            with self.urlopen(url) as response:
                response.read()
            raise AssertionError('interrupted read returned successfully')
    policy = SearchPolicy(source_timeout_s=.2 if mode=='timeout' else 5)
    http = SafeHttpClient(policy,transport=httpx.MockTransport(lambda request:httpx.Response(200,stream=Body())))
    task = asyncio.create_task(extract_metadata(http,policy,Reader,{},PUBLIC+'stream'))
    try:
        await asyncio.wait_for(started.wait(),1)
        if mode=='cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
        else:
            with pytest.raises(SearchFailure) as error: await task
            assert error.value.error.code=='TIMEOUT'
        await asyncio.wait_for(closed.wait(),1)
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_metadata_read_budget_is_shared_between_subrequests():
    class Reader(Extractor):
        def extract_info(self,url,**kwargs):
            for path in ['one','two']:
                with self.urlopen(PUBLIC+path) as response: response.read()
            return {}
    policy = SearchPolicy(maximum_html_bytes=1000, maximum_document_bytes=1024)
    http = SafeHttpClient(policy,transport=httpx.MockTransport(lambda request:httpx.Response(200,content=b'x'*600)))
    try:
        with pytest.raises(SearchFailure) as error:
            await extract_metadata(http,policy,Reader,{},PUBLIC)
        assert error.value.error.code=='UNSUPPORTED_FORMAT'
    finally:
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [302,307])
async def test_metadata_redirects_preserve_method_rules_and_strip_cross_origin_credentials(status):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path=='/start':
            return httpx.Response(status,headers={'Location':'https://1.1.1.1/end'})
        return httpx.Response(200,content=b'ok')
    http = SafeHttpClient(transport=httpx.MockTransport(handler))
    try:
        async with http.metadata_stream(PUBLIC+'start',method='POST',data=b'query',
                headers={'Authorization':'Bearer fixture','Cookie':'fixture=1'}) as response:
            assert await response.aread()==b'ok'
        assert requests[1].method==('GET' if status==302 else 'POST')
        assert requests[1].content==(b'' if status==302 else b'query')
        assert 'authorization' not in requests[1].headers and 'cookie' not in requests[1].headers
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_extractor_post_cookies_and_http_error_fallback_are_preserved():
    requests = []
    class Reader(Extractor):
        def __init__(self, options):
            self.cookiejar = CookieJar()
        def extract_info(self,url,**kwargs):
            with self.urlopen(url) as response: response.read()
            try:
                self.urlopen(Request(PUBLIC+'api', method='POST', data=b'{"query":"public"}',
                    headers={'Content-Type':'application/json'}))
            except HTTPError as error:
                assert error.status==403
                assert error.response.read()==b'{"retry":"public endpoint"}'
                error.response.close()
            with self.urlopen(PUBLIC+'fallback') as response:
                return json.loads(response.read())
    def handler(request):
        requests.append(request)
        if request.url.path=='/start':
            return httpx.Response(200,headers=[('Set-Cookie','one=1; Path=/'),('Set-Cookie','two=2; Path=/')],content=b'ok')
        if request.url.path=='/api':
            assert request.method=='POST' and request.content==b'{"query":"public"}'
            assert 'one=1' in request.headers['cookie'] and 'two=2' in request.headers['cookie']
            return httpx.Response(403,content=b'{"retry":"public endpoint"}')
        return httpx.Response(200,json={'id':'video'})
    http = SafeHttpClient(transport=httpx.MockTransport(handler))
    try:
        result = await extract_metadata(http,http.policy,Reader,{},PUBLIC+'start')
        assert result['id']=='video' and len(requests)==3
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_metadata_connection_to_private_peer_is_rejected():
    class Connection:
        def get_extra_info(self,name): return ('127.0.0.1',80)
    http = SafeHttpClient(transport=httpx.MockTransport(lambda request:httpx.Response(200,content=b'secret',
        extensions={'network_stream':Connection()})))
    try:
        with pytest.raises(SearchFailure) as error:
            async with http.metadata_stream(PUBLIC) as response:
                await response.aread()
        assert error.value.error.code=='ACCESS_RESTRICTED'
    finally:
        await http.aclose()
