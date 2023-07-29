import pandas as pd
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize

class DataFrameReader:
    def __init__(self, dataframe, column_name):
        if column_name not in dataframe.columns:
            raise ValueError(f"'{column_name}' not found in the dataframe.")
        self.data = dataframe[column_name]

    def get_sentences(self):
        for text in self.data:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                yield sentence

class NgramExtractor:
    def __init__(self, dataframe, column_name):
        self.reader = DataFrameReader(dataframe, column_name)
        self.ngram_counts = defaultdict(int)

    def extract_ngrams(self):
        for sentence in self.reader.get_sentences():
            tokens = sentence.split()
            for n in range(1, 5):
                for ngram in ngrams(tokens, n):
                    ngram_str = ' '.join(ngram)
                    self.ngram_counts[ngram_str] += 1

    def get_ngram_counts(self):
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
    for ngram, count in ngram_counts.items():
        print(f"{ngram}: {count}")
