import pytest
from wordview.mwes.mwe_utils import get_ngrams, is_alphanumeric_latinscript_multigram


def test_get_ngrams_int_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=5, n=2)


def test_get_ngrams_int_list_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=[1, 2, 3], n=2)


def test_get_ngrams_mixed_list_input():
    with pytest.raises(TypeError):
        get_ngrams(sentence=[1, 2, 'test'], n=2)


class TestIsAlphanumericLatinscriptMultigram:

    def test_is_alphanumeric_latinscript_multigram_matches_alphabetic_bigram(self):
        match = is_alphanumeric_latinscript_multigram("ab")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_matches_numeric_bigram(self):
        match = is_alphanumeric_latinscript_multigram("01")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_matches_alphanumeric_bigram(self):
        match = is_alphanumeric_latinscript_multigram("a0")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_matches_numeralphabetic_bigram(self):
        match = is_alphanumeric_latinscript_multigram("0a")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_matches_largest_alphanumeric_unique_letter_ngram(self):
        match = is_alphanumeric_latinscript_multigram("abcdefghijklmnopqrstuvwxyz0123456789")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_alphabetical_unigram(self):
        match = is_alphanumeric_latinscript_multigram("a")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_numeric_unigram(self):
        match = is_alphanumeric_latinscript_multigram("0")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_symbol_unigram(self):
        match = is_alphanumeric_latinscript_multigram("%")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_matches_ngram_with_symbol_after_second_char(self):
        match = is_alphanumeric_latinscript_multigram("abcd$efg")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_symbol_before_third_char(self):
        match = is_alphanumeric_latinscript_multigram("a$bcdefg")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_matches_ngram_with_hyphen_after_second_char(self):
        match = is_alphanumeric_latinscript_multigram("abcd-efg")
        assert match is not None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_hyphen_before_third_char(self):
        match = is_alphanumeric_latinscript_multigram("a-bcdefg")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_greek_characters(self):
        match = is_alphanumeric_latinscript_multigram("αβγ")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_hebrew_characters(self):
        match = is_alphanumeric_latinscript_multigram("אָלֶף")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_arabic_characters(self):
        match = is_alphanumeric_latinscript_multigram("أَلِف")
        assert match is None


    def test_is_alphanumeric_latinscript_multigram_does_not_match_ngram_with_russian_cyrillic_characters(self):
        match = is_alphanumeric_latinscript_multigram("азъ")
        assert match is None
