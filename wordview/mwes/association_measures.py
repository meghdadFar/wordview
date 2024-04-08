import json
import math


class PMICalculator:
    """Calculates the Pointwise Mutual Information (PMI) of an n-gram candidate."""

    def __init__(self, ngram_count_source: dict):
        """Initializes a new instance of PMICalculator class.

        Args:
            ngram_count_source: A dictionary containing ngram counts.

        Returns:
            None
        """
        self.counts = ngram_count_source
        self.total_count = sum(self.counts.values())

    def _prob(self, ngram) -> float:
        return self.counts.get(ngram, 0) / self.total_count

    def compute_association(self, ngram: str) -> float:
        """Computes the association measure (AM)  --currently only in terms of
        PMI, of an n-gram candidate.

        Args:
            ngram: A string containing the n-gram candidate.

        Returns:
            The association measure of the n-gram candidate.

        """
        words = ngram.split()

        p_denominator = 1.0
        for word in words:
            p_denominator *= self._prob(word)

        p_ngram = self._prob(ngram)

        if p_denominator == 0 or p_ngram == 0:
            return float("-inf")
        else:
            return math.log(p_ngram / p_denominator, 2)
