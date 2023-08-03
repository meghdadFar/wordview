import pytest
import pandas as pd
from wordview.mwes.mwe import PMICalculator


@pytest.fixture
def ngram_counts_dict():
    return {
        'coffee': 100,
        'shop': 150,
        'coffee shop': 80,
        'swimming': 50,
        'pool': 60,
        'swimming pool': 40,
        'give': 200,
        'a': 500,
        'speech': 45,
        'give a': 150,
        'a speech': 40,
        'give a speech': 35,
        'take': 100,
        'deep': 30,
        'breath': 25,
        'take a': 80,
        'a deep': 25,
        'deep breath': 20,
        'take a deep': 18,
        'a deep breath': 17,
        'take a deep breath': 15
    }


class TestPMICalculator:
    def test_compute_association(self, ngram_counts_dict):
        calculator = PMICalculator(ngram_count_source=ngram_counts_dict)
        ngram = 'coffee shop'
        pmi_value = calculator.compute_association(ngram)
        assert pmi_value == pytest.approx(3.24, abs=0.1)
    
    def test_compute_association_with_zero_count(self, ngram_counts_dict):
        calculator = PMICalculator(ngram_count_source=ngram_counts_dict)
        ngram = 'horseback riding'
        pmi_value = calculator.compute_association(ngram)
        assert pmi_value == float('-inf')
