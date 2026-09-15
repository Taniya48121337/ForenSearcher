import pytest

from foren_searcher.local_models import TransformersGiaPolicy, _validate_gia_output


def view():
    return {'task':'Report the measured relation and uncertainty.', 'latest_reports':[{
        'run_status':'reported', 'report':{'status':'COMPLETED','findings':[
            {'finding_id':'F1','statement':'Speed is 1.5x.', 'evidence_ids':['E2'],
             'limitations':['Boundary uncertainty has not been measured.']}]}}]}


@pytest.mark.parametrize('citation', ['（F1 限制）', '(F1 limitation)', '（F1、F2 限定条件）', '[F1]', '(参见F1)',
    '（F3，E2）', '(E2, F1)', '[F1; E2]', '[E2/F1]', '（依据F1、E2）', '(see E2, F1)', '（证据：E2、F3）'])
def test_finish_reports_explicit_local_finding_citations_for_model_correction(citation):
    with pytest.raises(ValueError, match='local Finding citations'):
        _validate_gia_output({'decision':'FINISH','answer':f'Speed is 1.5x (E2). The boundary is uncertain {citation}.'},
                             view(), (), ())


@pytest.mark.parametrize('answer', ['F1 score is 0.81 (E2).', 'The reported metric (F1) is 0.81 (E2).',
                                  'The identifier is `F1` (E2).', 'Speed is 1.5x; boundary uncertainty is unknown (E2).',
                                  '原文写道「[F1] 或（F1 限制）」。（E2）',
                                  'The text reads "(F1 limitation)" (E2).',
                                  '文档中的代码为 `[F1]`。（E2）',
                                  '使用两个指标（F1，F2）。（E2）', '原文为「（F3，E2）」。（E2）',
                                  '文档中的混合编号为 `[E2/F1]`。（E2）', '证据支持此结论（E2、E2）。'])
def test_metric_names_and_literal_identifiers_are_not_local_finding_citations(answer):
    _validate_gia_output({'decision':'FINISH','answer':answer}, view(), (), ())


@pytest.mark.asyncio
@pytest.mark.parametrize('citation', ['(F1 limitation)', '（F1，E2）'])
async def test_same_gia_model_corrects_citations_with_original_answer_and_evidence(citation):
    first = {'decision':'FINISH','answer':f'Speed is 1.5x (E2). Boundary error is ±0.1s {citation}.'}
    corrected = {'decision':'FINISH','answer':'Speed is 1.5x; boundary uncertainty is unmeasured (E2).'}
    class Model:
        def __init__(self): self.calls = []
        async def generate_json(self, *, system_prompt, payload):
            self.calls.append(payload)
            if len(self.calls) == 1: return first
            assert payload['previous_invalid_output'] == first
            assert payload['global_view'] == view()
            assert 'Do not mechanically replace F with E' in payload['validation_error']
            return corrected
    generator = Model()
    result = await TransformersGiaPolicy(generator).decide(model_view=view(), history=(), actions=())
    assert result == corrected and len(generator.calls) == 2
    assert first['answer'].endswith(citation + '.')
