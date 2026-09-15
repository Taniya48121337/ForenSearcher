import base64
import io
import json

import pytest
from PIL import Image,ImageDraw

from foren_searcher.model_images import ModelImageInputs
from foren_searcher.store import GlobalStore
from inspect_helpers import ingest


@pytest.mark.parametrize('count,size', [
    (1, (1414, 1060)), (2, (1086, 814)), (3, (886, 665)),
    (4, (768, 576)), (8, (768, 576)),
])
def test_image_count_shares_detail_without_changing_evidence_or_bulk_pixels(tmp_path, count, size):
    original=Image.new('RGB',(320,240),'white')
    draw=ImageDraw.Draw(original)
    draw.line((20,200,280,40),fill='blue',width=1)
    draw.text((25,30),'56 / 52',fill='green')
    buffer=io.BytesIO();original.save(buffer,format='PNG');data=buffer.getvalue()
    store=GlobalStore(tmp_path/'state.sqlite3',tmp_path/'artifacts')
    sources=[ingest(store,data,source=f'I{i}',filename=f'image-{i}.png',
                   media_type='image/png',modality='image') for i in range(count)]
    view={'image_inputs':[{'evidence_id':e.evidence_id,'locator':e.locator,'derived_from':[],
                           'width':320,'height':240} for e in sources]}
    message,identifiers=ModelImageInputs(store=store)._image_message(view)
    assert identifiers==[e.evidence_id for e in sources]
    expected=original.resize(size,Image.Resampling.LANCZOS)
    for index,part in enumerate(message['content']):
        if part['type']!='image_url':continue
        caption=json.loads(message['content'][index-1]['text'])
        with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))) as supplied:
            assert supplied.size==size and supplied.tobytes()==expected.tobytes()
            assert supplied.width*supplied.height<=1_500_000
            assert (caption['model_image_width'],caption['model_image_height'])==size
        assert (caption['width'],caption['height'])==(320,240)
    for source in sources:assert store.artifact_path(source.artifact_ref).read_bytes()==data


def test_empty_image_selection_has_no_payload(tmp_path):
    store=GlobalStore(tmp_path/'state.sqlite3',tmp_path/'artifacts')
    assert ModelImageInputs(store=store)._image_message({'image_inputs':[]})==(None,[])
