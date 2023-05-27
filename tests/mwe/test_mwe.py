import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from wordview.mwes.mwe import MWE


@pytest.fixture
def dummy_text_pandas():
    text = [
        "new   york  is the capital     of new  york    state", 
        "new york is called the big apple", 
        "i am going to the big apple",
        ]
    return pd.DataFrame(data = text, columns = ["text"])


@pytest.fixture
def dummy_text_pandas_with_a_noun_compund():
    text = [
        "this sentence has a noun compound",
        ]
    return pd.DataFrame(data = text, columns = ["text"])


dummy_pos_tags_with_noun_compund = [("this", "XXX"),("sentence", "XXX"),("has", "XXX"),("a", "XXX"),("noun", "NN"),("compound", "NN")]


@pytest.fixture
def dummy_text_pandas_with_no_noun_compund():
    text = [
        "no sequence of nouns in this one",
        ]
    return pd.DataFrame(data = text, columns = ["text"])


dummy_pos_tags_without_noun_compund = [("no", "XXX"),("sequence", "XXX"),("of", "XXX"),("nouns", "XXX"),("in", "XXX"),("this", "XXX"),("one", "XXX")]


class TestMweInitialisation:

    def test_mwe_does_not_tokenize_text_with_multiple_whitespaces(self, dummy_text_pandas):
        mwe = MWE(df = dummy_text_pandas, text_column = "text", tokenize=False)
        assert mwe.df["text"][0] == "new   york  is the capital     of new  york    state"


    def test_mwe_tokenizes_text_with_multiple_whitespaces(self, dummy_text_pandas):
        mwe = MWE(df = dummy_text_pandas, text_column = "text", tokenize=True)
        assert mwe.df["text"][0] == "new york is the capital of new york state"


    def test_mwe_with_wrong_mwe_type_raises_value_error(self, dummy_text_pandas):
        with pytest.raises(ValueError):
            mwe = MWE(df = dummy_text_pandas, text_column = "text", mwe_types = ["XXX"])


    def test_mwe_with_empty_mwe_type_raises_value_error(self, dummy_text_pandas):
        with pytest.raises(ValueError):
            mwe = MWE(df = dummy_text_pandas, text_column = "text", mwe_types = [])


    def test_mwe_with_non_list_mwe_type_raises_type_error(self, dummy_text_pandas):
        with pytest.raises(TypeError):
            mwe = MWE(df = dummy_text_pandas, text_column = "text", mwe_types = "NC")


class TestMweCounter:

    @patch("wordview.mwes.mwe.get_pos_tags", MagicMock(return_value = dummy_pos_tags_with_noun_compund))
    def test_mwe_if_nc_present_returns_counts_with_nc(self, dummy_text_pandas_with_a_noun_compund):
        mwe = MWE(df = dummy_text_pandas_with_a_noun_compund, text_column = "text", tokenize=True, mwe_types = ["NC"])
        counts = mwe.get_counts()
        assert counts["NC"] == {"noun compound": 1}


    @patch("wordview.mwes.mwe.get_pos_tags", MagicMock(return_value=dummy_pos_tags_without_noun_compund))
    def test_mwe_if_no_nc_returns_empty_mwe_counts(self, dummy_text_pandas_with_no_noun_compund):
        mwe = MWE(df = dummy_text_pandas_with_no_noun_compund, text_column = "text", tokenize=True, mwe_types = ["NC"])
        counts = mwe.get_counts()
        assert counts["NC"] == {}


@pytest.mark.xfail
def test_mwe_build_counts(dummy_text_pandas_with_no_noun_compund):
    mwe = MWE(df = dummy_text_pandas_with_no_noun_compund, text_column = "text", tokenize=True, mwe_types = ["NC"])
    counts = mwe.get_counts()
    assert False
