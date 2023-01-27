import pytest
from wordview.mwes.mwe_utils import get_ngrams


def test_get_ngrams_int_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=5, n=2)


def test_get_ngrams_int_list_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=[1, 2, 3], n=2)


def test_get_ngrams_mixed_list_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=[1, 2, 'test'], n=2)
