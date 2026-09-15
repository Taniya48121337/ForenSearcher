import pytest
import json
from jsonschema import Draft202012Validator

from foren_searcher.local_models import _decision_output_schema
from foren_searcher.search.contracts import parse_search_request
from foren_searcher.workspace import _SEARCH_SCHEMA


@pytest.mark.parametrize(('input_type', 'text', 'valid'), [
    ('text', 'John F Kennedy inaugural address', True),
    ('text', '  肯尼迪 就职演讲  ', True),
    ('text', 'first line\nsecond line', True),
    ('text', 'x' * 500, True),
    ('text', 'x' * 501, False),
    ('text', '   ', False),
    ('text', '', False),
    ('text_image', 'Apollo launch original footage', True),
    ('text_image', 'x' * 501, False),
    ('text_image', '\t\n', False),
])
def test_search_decoder_preserves_complete_queries_and_runtime_limits(input_type, text, valid):
    package = pytest.importorskip('lmformatenforcer')
    from foren_searcher.local_models import _create_output_enforcer

    query = text if input_type == 'text' else {'evidence_id': 'E12', 'text': text}
    args = {'input_type': input_type, 'query': query, 'output_type': 'text'}
    value = {'action': 'SEARCH', 'arguments': args, 'focus_evidence_ids': []}
    rendered = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    chars = sorted(set(''.join(chr(i) for i in range(32, 127)) + rendered))
    eos = len(chars)
    data = package.TokenEnforcerTokenizerData([(i, c, False) for i, c in enumerate(chars)],
        lambda ids: ''.join(chars[i] for i in ids), eos, False, eos + 1)
    schema = _decision_output_schema({'workspace_id': 'W1-1',
        'available_actions': [{'name': 'SEARCH', 'schema': _SEARCH_SCHEMA}]})
    enforcer = _create_output_enforcer(data, schema)
    prefix = [eos]
    for char in rendered:
        token = chars.index(char)
        if token not in enforcer.get_allowed_tokens(prefix).allowed_tokens:
            accepted = False
            break
        prefix.append(token)
    else:
        accepted = eos in enforcer.get_allowed_tokens(prefix).allowed_tokens
    if valid:
        assert accepted
    elif not text or len(text) > 500:
        assert not accepted
    assert Draft202012Validator(schema).is_valid(value) == valid
    assert Draft202012Validator(_SEARCH_SCHEMA).is_valid(args) == valid
    if valid:
        assert parse_search_request(args).to_dict() == args
    else:
        with pytest.raises(ValueError):
            parse_search_request(args)


@pytest.mark.parametrize('language', ['en', 'zh-Hans'])
def test_subtitle_language_pattern_allows_complete_language_codes(language):
    package = pytest.importorskip('lmformatenforcer')
    args = {'source': 'https://example.com/video', 'output_type': 'video',
            'include': ['subtitles'], 'subtitle_language': language}
    schema = _SEARCH_SCHEMA['oneOf'][-1]
    parser = package.JsonSchemaParser(schema)
    for char in json.dumps(args, separators=(',', ':')):
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()
    assert parse_search_request(args).to_dict() == args


@pytest.mark.parametrize('output_type', ['text', 'image', 'video'])
def test_search_query_schema_preserves_legal_input_output_combinations(output_type):
    validator = Draft202012Validator(_SEARCH_SCHEMA)
    for input_type, query in [('text', 'official source'), ('image', 'E12'),
                             ('text_image', {'evidence_id': 'E12', 'text': 'official source'})]:
        value = {'input_type': input_type, 'query': query, 'output_type': output_type}
        assert validator.is_valid(value)
        assert parse_search_request(value).to_dict() == value


@pytest.mark.parametrize(('input_type', 'query'), [
    ('image', {'evidence_id': 'E1', 'text': 'reverse image search'}),
    ('image', 'reverse image search'),
    ('text', {'evidence_id': 'E1', 'text': 'official source'}),
    ('text_image', 'E1'),
    ('text_image', {'evidence_id': 'E1'}),
    ('text_image', {'evidence_id': 'R1', 'text': 'official source'}),
    ('text', ' '),
    ('text', 'x' * 501),
    ('text_image', {'evidence_id': 'E1', 'text': 'x' * 501}),
])
def test_search_action_and_output_schemas_reject_invalid_query_shapes(input_type, query):
    args = {'input_type': input_type, 'query': query, 'output_type': 'text'}
    with pytest.raises(ValueError):
        parse_search_request(args)
    assert not Draft202012Validator(_SEARCH_SCHEMA).is_valid(args)
    schema = _decision_output_schema({'workspace_id': 'W1-1',
        'available_actions': [{'name': 'SEARCH', 'schema': _SEARCH_SCHEMA}]})
    assert not Draft202012Validator(schema).is_valid({
        'action': 'SEARCH', 'arguments': args, 'focus_evidence_ids': []})


@pytest.mark.parametrize('source', ['https://example.com/source', 'R12'])
@pytest.mark.parametrize('output_type', ['text', 'image', 'video'])
def test_source_open_schemas_agree_with_runtime_subtitle_options(source, output_type):
    validator = Draft202012Validator(_SEARCH_SCHEMA)
    decision_validator = Draft202012Validator(_decision_output_schema({'workspace_id': 'W1-1',
        'available_actions': [{'name': 'SEARCH', 'schema': _SEARCH_SCHEMA}]}))
    for include in (None, [], ['subtitles'], ['subtitles', 'subtitles'], ['thumbnail']):
        for language in ({}, {'subtitle_language': None}, {'subtitle_language': ''},
                         {'subtitle_language': ' '}, {'subtitle_language': 'en'}):
            value = {'source': source, 'output_type': output_type, **language}
            if include is not None:
                value['include'] = include
            try:
                parsed = parse_search_request(value)
            except ValueError:
                valid = False
            else:
                valid = True
                assert validator.is_valid(parsed.to_dict())
            assert validator.is_valid(value) == valid, value
            assert decision_validator.is_valid({'action': 'SEARCH', 'arguments': value,
                'focus_evidence_ids': []}) == valid, value
