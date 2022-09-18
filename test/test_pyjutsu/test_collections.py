import pytest
from pyjutsu.collection_utils import circular


def test_circular():
    ll = [1, 2, 3]
    assert circular(ll)[4] == 2
    assert [*circular(ll)[4:5]] == [2]
    assert [*circular(ll)[4:6]] == [2, 3]

    # TODO: This still fails and I'm to tired and now's too late.
    assert [*circular(ll)[2:5]] == [3, 1, 2]
    assert [*circular(ll)[2:7]] == [3, 1, 2, 3, 1]
    assert [*circular(ll)[2:7:2]] == [3, 2, 1]

    ll_single = [1]
    assert [*circular(ll_single)[0:5:2]] == [1, 1, 1]

    circular(ll)[4] = 44
    assert ll[1] == 44

