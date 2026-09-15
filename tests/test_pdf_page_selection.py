import json

import fitz
from PIL import Image
import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


@pytest.mark.asyncio
@pytest.mark.parametrize('total,params,expected', [(1, {}, 1), (4, {'page_range': [3, 3]}, 3),
    (5, {'page_range': [2, 4], 'count': 1}, 3), (4, {'count': 1}, 2)])
async def test_single_page_selection_preserves_number_range_and_actual_pixels(tmp_path, total, params, expected):
    source = tmp_path / 'pages.pdf'
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    with fitz.open() as doc:
        for number in range(total):
            page = doc.new_page(width=100, height=100)
            page.draw_rect(page.rect, color=colors[number], fill=colors[number])
            page.insert_text((10, 75), f'Page {number+1}')
        doc.save(source)
    store = GlobalStore(tmp_path / 'store.sqlite3', tmp_path / 'artifacts')
    media = store.commit_ingested_evidence(source_ref='I0', modalities=('text',), media_type='application/pdf',
        label='pages.pdf', path=source, original_filename='pages.pdf')
    async with create_inspect_service(store=store) as service:
        result = await service.execute({'targets': [media.evidence_id], 'request': {'operation': 'get_pages', 'params': params}}, workspace_id='W1-1')
    assert result.status == 'ready'
    page, = result.result.pages
    assert page.page_number == expected
    evidence = store.evidence(page.evidence_id)
    assert json.loads(evidence.locator)['page_number'] == expected
    with Image.open(store.artifact(evidence.artifact_ref).path) as rendered:
        assert rendered.getpixel((20, 20)) == tuple(value*255 for value in colors[expected-1])
