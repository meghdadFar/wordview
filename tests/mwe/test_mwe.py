import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from wordview.mwes.mwe import MWE, MWEExtractor
import nltk


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


@pytest.fixture
def tagged_sentence_fixture():
    sentence = "The very quick brown fox swiftly jumps over the lazy dog that is extremely lazy while John Doe attentively watches the lazy dog."
    tokens = nltk.word_tokenize(sentence)
    return tokens


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


class TestHigherOrderMWEExtraction:

    def test_extract_higher_order_mwes_wrong_type_tokens(self):
        tokens = "this is a string"
        pattern = "NP: {<DT>?<JJ>*<NN>}"
        with pytest.raises(TypeError):
            mwe_extractor = MWEExtractor(tokens, pattern)   
    
    def test_extract_higher_order_mwes_empty_tokens(self):
        tokens = []
        pattern = "NP: {<DT>?<JJ>*<NN>}"
        with pytest.raises(ValueError):
            mwe_extractor = MWEExtractor(tokens, pattern)   

    def test_extract_higher_order_mwes_wrong_type_pattern(self, tagged_sentence_fixture):
        pattern = 1
        with pytest.raises(TypeError):
            mwe_extractor = MWEExtractor(tagged_sentence_fixture, pattern)

    def test_extract_higher_order_mwes_empty_pattern(self, tagged_sentence_fixture):
        pattern = ""
        with pytest.raises(ValueError):
            mwe_extractor = MWEExtractor(tagged_sentence_fixture, pattern)

    def test_extract_higher_order_mwes_incorrect_pattern(self, tagged_sentence_fixture):
        pattern = "{<DT>?<JJ>*<NN>}"
        mwe_extractor = MWEExtractor(tagged_sentence_fixture, pattern)
        with pytest.raises(ValueError):
            mwe_extractor.extract_mwe_candidates()
    
    def test_extract_higher_order_mwes_single_pattern(self, tagged_sentence_fixture):
        pattern = "NP: {<DT>?<JJ>+<NN>}"
        mwe_extractor = MWEExtractor(tagged_sentence_fixture, pattern)
        expected = {'NP': {'quick brown': 1, 'the lazy dog': 2}}
        actual = mwe_extractor.extract_mwe_candidates()
        assert expected == actual

    def test_extract_higher_order_mwes_multi_pattern(self, tagged_sentence_fixture):
        pattern = """NP: {<DT>?<JJ>+<NN>} # Noun phrase
        PROPN: {<NNP>+} # Proper noun
        ADJP: {<RB|RBR|RBS>*<JJ>} # Adjective phrase
        ADVP: {<RB.*>+<VB.*><RB.*>*} # Adverb phrase"""
        expected = {
            'NP': {'quick brown': 1, 'the lazy dog': 2},
            'PROPN': {'John Doe': 1},
            'ADJP': {'extremely lazy': 1},
            'ADVP': {'swiftly jumps': 1, 'attentively watches': 1}
        }
        mwe_extractor = MWEExtractor(tagged_sentence_fixture, pattern)
        actual = mwe_extractor.extract_mwe_candidates()
        assert expected == actual


@pytest.mark.xfail
def test_mwe_build_counts(dummy_text_pandas_with_no_noun_compund):
    mwe = MWE(df = dummy_text_pandas_with_no_noun_compund, text_column = "text", tokenize=True, mwe_types = ["NC"])
    counts = mwe.get_counts()
    assert False
