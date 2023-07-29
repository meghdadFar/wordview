import pandas as pd
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from typing import List, Dict

class DataFrameReader:
    def __init__(self, dataframe, column_name):
        if column_name not in dataframe.columns:
            raise ValueError(f"'{column_name}' not found in the dataframe.")
        self.data = dataframe[column_name]

    def get_sentences(self):
        for text in self.data:
            # Split the text into individual sentences
            sentences = sent_tokenize(text)
            for sentence in sentences:
                yield sentence


class NgramExtractor:
    def __init__(self, dataframe, column_name):
        self.reader = DataFrameReader(dataframe, column_name)
        self.ngram_counts = defaultdict(int)

    def extract_ngrams(self):
        for sentence in self.reader.get_sentences():
            # Tokenize and remove punctuations
            tokens = [word for word in word_tokenize(sentence) if word not in string.punctuation]
            
            for n in range(1, 5):
                for ngram in ngrams(tokens, n):
                    # Convert the n-gram tuple to a space-separated string
                    ngram_str = ' '.join(ngram)
                    self.ngram_counts[ngram_str] += 1

    def get_ngram_counts(self) -> Dict[str, int]:
        """Returns a dictionary of n-grams and their counts.
        
        Args:
            None
        Returns:
            ngram_counts (Dict[str, int]): A dictionary of n-grams and their counts.
        """
        return dict(self.ngram_counts)

if __name__ == "__main__":
    df = pd.DataFrame({
        'text': [
            'This is a sample sentence. Here is another one!',
            'Another sample sentence here. And yet another one.',
            'Yet another sample! This continues.'
        ]
    })

    extractor = NgramExtractor(df, 'text')
    extractor.extract_ngrams()

    ngram_counts = extractor.get_ngram_counts()
    print(ngram_counts)
    # for ngram, count in ngram_counts.items():
        # print(f"{ngram}: {count}")
