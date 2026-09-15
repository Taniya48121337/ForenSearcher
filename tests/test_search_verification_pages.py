from __future__ import annotations

import httpx
import pytest

from foren_searcher.search.contracts import CandidateSource
from foren_searcher.search.extractors.html import extract_html
from foren_searcher.search.openers.document import DocumentSourceOpener
from foren_searcher.search.openers.http import SafeHttpClient
from foren_searcher.search.policy import SearchPolicy
from foren_searcher.search.providers.fixture import FixtureSearchProvider
from foren_searcher.search.service import SearchService
from foren_searcher.store import GlobalStore


def challenge_page(pronoun="you're"):
    return (f'<html><head><title>Public video</title></head><body><h1>Verify to continue</h1>'
        f'<p>To continue, please confirm that {pronoun} a human (and not a spambot).</p>'
        '<h2>Checking if the site connection is secure</h2></body></html>').encode()


@pytest.mark.parametrize('pronoun', ["you're", 'you’re', 'you are'])
def test_human_confirmation_page_is_restricted(pronoun):
    result = extract_html(challenge_page(pronoun), base_url='https://example.com/video')
    assert result.soft_error == 'ACCESS_RESTRICTED'


@pytest.mark.parametrize('hidden', [False, True])
def test_discussing_or_hiding_a_confirmation_prompt_is_not_a_restriction(hidden):
    example = challenge_page().decode().split('<body>')[1].split('</body>')[0]
    body = (f'<div hidden>{example}</div><article><h1>Research article</h1><p>Readable findings.</p></article>'
        if hidden else f'<article><h1>Research article</h1><p>The following illustrates an access prompt.</p>{example}<p>Readable findings.</p></article>')
    result = extract_html(f'<html><body>{body}</body></html>'.encode(), base_url='https://example.com/article')
    assert result.soft_error is None
    assert 'Readable findings.' in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(('dynamic', 'missing'), [(False, False), (True, False), (True, True)])
async def test_failed_page_is_not_shared_evidence_and_search_continues(dynamic, missing):
    blocked = b'<html><body>Page Not Found</body></html>' if missing else challenge_page()
    shell = b'<html><body><script>loadPage()</script></body></html>'
    article = b'<html><body><article><h1>Readable source</h1><p>Actual source body.</p></article></body></html>'
    opened = []

    def handler(request):
        opened.append(request.url.path)
        body = (shell if dynamic else blocked) if request.url.path == '/blocked' else article
        return httpx.Response(200, content=body, headers={'content-type': 'text/html'})

    class Renderer:
        calls = 0

        async def render(self, url):
            self.calls += 1
            return blocked

    renderer = Renderer()
    policy = SearchPolicy(initial_candidates=1, maximum_candidates=2, target_sources=1)
    http = SafeHttpClient(policy, transport=httpx.MockTransport(handler))
    store = GlobalStore()
    provider = FixtureSearchProvider(tuple(CandidateSource('https://93.184.216.34' + path,
        candidate_type='page') for path in ('/blocked', '/source')))
    service = SearchService(store=store, provider=provider,
        opener=DocumentSourceOpener(http, policy, renderer), policy=policy, close_callback=http.aclose)
    async with service:
        response = await service.execute({'input_type': 'text', 'query': 'public video', 'output_type': 'text'},
            workspace_id='W1-1')
    assert response.stop_reason == 'target_reached', response.to_dict()
    assert opened == ['/blocked', '/source']
    assert renderer.calls == int(dynamic)
    denied, readable = response.results
    failed = denied.materials[0]
    assert failed.status == 'failed' and not failed.evidence
    assert failed.error.code == ('NOT_FOUND' if missing else 'ACCESS_RESTRICTED')
    assert not store.candidate_evidence(denied.candidate_id)
    assert readable.materials[0].status == 'ready'
    evidence = store.evidence(readable.materials[0].evidence[0].evidence_id)
    assert 'Actual source body.' in evidence.content
