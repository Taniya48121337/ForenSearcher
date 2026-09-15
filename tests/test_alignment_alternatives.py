from foren_searcher.inspect.contracts import (
    AlignMediaResult, AlignmentAnchor, AlignmentMatch, AlignmentSideCoverage,
)


def match(a, b, *, padded_b=None):
    return AlignmentMatch(a, padded_b or b, .9, (
        AlignmentAnchor(a[0], b[0], None, None),
        AlignmentAnchor(a[1], b[1], None, None),
    ))


def result(*matches):
    coverage = AlignmentSideCoverage((0, 100), 'frames', 480, ())
    return AlignMediaResult('visual', coverage, coverage, matches).to_dict()


def test_full_correspondence_and_partial_repeat_are_alternatives():
    value = result(match((2, 10), (0, 8)), match((84, 87), (1, 4)))
    relations = value['candidate_relationships']
    assert len(value['matches']) == 2
    assert len(relations) == 1
    assert relations[0]['match_indices'] == [0, 1]
    assert relations[0]['overlap_in_b_s'] == [1, 4]
    assert relations[0]['relationship'] == 'overlapping_candidates'


def test_padded_edges_of_genuine_splice_do_not_create_false_overlap():
    value = result(match((6, 11.9), (0, 5.9), padded_b=(0, 6.1)),
                   match((15, 23.9), (6, 14.9), padded_b=(5.9, 15)))
    assert value['candidate_relationships'] == []


def test_static_duplicate_candidates_remain_visible():
    value = result(match((0, 3.9), (0, 3.9)), match((4, 7.9), (0, 3.9)),
                   match((8, 9.9), (0, 1.9)))
    assert len(value['matches']) == 3
    assert [r['match_indices'] for r in value['candidate_relationships']] == [[0, 1], [0, 2], [1, 2]]


def test_single_anchor_does_not_establish_overlapping_time_span():
    single = AlignmentMatch((0, 2), (0, 2), .9, (AlignmentAnchor(1, 1, None, None),))
    assert result(single, match((10, 12), (0, 2)))['candidate_relationships'] == []
