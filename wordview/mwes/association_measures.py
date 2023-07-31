import math
import pickle
from itertools import combinations

class PMICalculator:
    
    def __init__(self, ngram_count_source=None, ngram_count_file_path=None):
        
        if ngram_count_source:
            self.counts = ngram_count_source
        elif ngram_count_file_path:
            self.counts = self._load_ngram_counts(ngram_count_file_path)
        else:
            raise ValueError("Either count_source or count_file_path must be provided.")
        
        self.total_count = sum(self.counts.values())

    def _load_ngram_counts(self, count_file_path):
        with open(count_file_path, 'rb') as f:
            counts = pickle.load(f)
        return counts

    def _prob(self, ngram):
        return self.counts.get(ngram, 0) / self.total_count

    def compute_pmi(self, ngram_candidate):
        words = ngram_candidate.split()

        p_denominator = 1
        for word in words:
            p_denominator *= self._prob(word)

        p_ngram = self._prob(ngram_candidate)

        if p_denominator == 0 or p_ngram == 0:
            return float('-inf')
        else:
            return math.log(p_ngram / p_denominator, 2)


if __name__ == "__main__":
    counts = {
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

    calculator = PMICalculator(ngram_count_source=counts)

    ngram = 'coffee shop'
    pmi_value = calculator.compute_pmi(ngram)
    print(f"PMI of '{ngram}': {pmi_value}")
