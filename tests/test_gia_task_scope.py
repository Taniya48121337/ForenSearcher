from copy import deepcopy

import pytest

from foren_searcher.local_models import TransformersGiaPolicy, _normalize_gia_output
from test_local_models import SequenceGenerator


def scenario():
    task = '第一段为A，第二段为B。核查B是否截取自A，给出证据和不确定性。'
    view = {'task': task, 'original_materials': [{'evidence_id': 'E1'}, {'evidence_id': 'E2'}]}
    whole = {'task': task.replace('为A', '为A（E1）').replace('为B', '为B（E2）'), 'materials': ['E1', 'E2']}
    output = {'decision': 'INVESTIGATE', 'tasks': [whole,
        {'task': '检查A中是否有重复场景，判断能否仅凭相似画面确定唯一对应位置；给出时间与证据。', 'materials': ['E1']}]}
    return view, output


@pytest.mark.asyncio
async def test_full_question_preserves_complementary_check_on_shared_materials():
    view, output = scenario()
    original = deepcopy(output)
    generator = SequenceGenerator([output])
    result = await TransformersGiaPolicy(generator).decide(model_view=view, history=(), actions=())
    assert result == original
    assert output == original
    assert len(generator.calls) == 1


@pytest.mark.parametrize('variation', ['subquestions', 'missing_material', 'new_material', 'unknown_reference', 'followup', 'different_claim'])
def test_scope_normalization_does_not_merge_unproven_relationships(variation):
    view, output = scenario()
    if variation == 'subquestions':
        output['tasks'].pop(0)
    elif variation == 'missing_material':
        output['tasks'][0]['materials'] = ['E1']
    elif variation == 'new_material':
        view['current_materials'] = [{'evidence_id': 'E3'}]
        output['tasks'][1]['materials'].append('E3')
    elif variation == 'unknown_reference':
        output['tasks'][1]['materials'].append('E999')
    elif variation == 'followup':
        view['latest_reports'] = [{'run_status': 'reported'}]
    else:
        output['tasks'][0]['task'] = output['tasks'][0]['task'].replace('是否截取自', '必定截取自')
    assert _normalize_gia_output(output, view, ()) == output
