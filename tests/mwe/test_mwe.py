import pytest
import pandas as pd
from wordview.mwes.mwe import MWE


@pytest.fixture
def dummy_text_panda():
    text = ["new   york  is the capital     of new  york    state", "new york is called the big apple", "i am going to the big apple"]
    return pd.DataFrame(data = text, columns = ["text"])


def test_mwe_does_not_tokenize_text_with_multiple_whitespaces(dummy_text_panda):
    mwe = MWE(df = dummy_text_panda, text_column = "text", tokenize=False)
    assert mwe.df["text"][0] == "new   york  is the capital     of new  york    state"


def test_mwe_tokenizes_text_with_multiple_whitespaces(dummy_text_panda):
    mwe = MWE(df = dummy_text_panda, text_column = "text", tokenize=True)
    assert mwe.df["text"][0] == "new york is the capital of new york state"
