import pandas as pd
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize

class NgramExtractor:
    def __init__(self):
        self.ngram_counts = defaultdict(int)

    def extract_from_sentence(self, sentence):
        tokens = sentence.split()
        for n in range(1, 5):
            for ngram in ngrams(tokens, n):
                self.ngram_counts[ngram] += 1

    def get_ngrams(self):
        return dict(self.ngram_counts)


class DataFrameReader:
    def __init__(self, dataframe, column_name):
        if column_name not in dataframe.columns:
            raise ValueError(f"'{column_name}' not found in the dataframe.")
        self.data = dataframe[column_name]
        self.extractor = NgramExtractor()

    def process_sentences(self):
        for text in self.data:
            # Split the text into individual sentences
            sentences = sent_tokenize(text)
            for sentence in sentences:
                self.extractor.extract_from_sentence(sentence)

    def get_ngram_counts(self):
        return self.extractor.get_ngrams()


if __name__ == "__main__":
    df = pd.DataFrame({
        'text': [
            'This is a sample sentence. Here is another one! And another one. And another one.',
            'Another sample sentence here. And yet another one.',
            'Yet another sample! This continues.'
        ]
    })

    reader = DataFrameReader(df, 'text')
    reader.process_sentences()

    ngram_counts = reader.get_ngram_counts()
    for ngram, count in ngram_counts.items():
        print(f"{ngram}: {count}")
