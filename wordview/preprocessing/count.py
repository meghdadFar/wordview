import pandas as pd
from collections import defaultdict
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from typing import List, Dict, Optional
import json
import os
import datetime

class DataFrameReader:
    """Reads a dataframe column and returns sentences."""
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
    """Extracts n-grams from a dataframe.
    
    Example:
    >>> df = pd.DataFrame({
    ...     'text': [
    ...         'This is a sample sentence. Here is another one!',
    ...         'Another sample sentence here. And yet another one.',
    ...         'Yet another sample! This continues.'
    ...     ]
    ... })
    >>> extractor = NgramExtractor(df, 'text')
    >>> extractor.extract_ngrams()
    >>> ngram_counts = extractor.get_ngram_counts()
    >>> print(ngram_counts)
    """
    def __init__(self, dataframe, column_name):
        """Initializes a new instance of NgramExtractor class.
        
        Args:
            dataframe (pandas.DataFrame): A dataframe containing the text column.
            column_name (str): The name of the text column.

        Returns:
            None
        """
        self.reader = DataFrameReader(dataframe, column_name)
        self.ngram_counts = defaultdict(int)

    def extract_ngrams(self, n: int = 4):
        """Extracts n-grams from the text column of the dataframe.
        
        Args:
            n (int): The maximum number of words in an n-gram. Defaults to 4.
        
        Returns:
            None
        """
        for sentence in self.reader.get_sentences():
            tokens = [word for word in word_tokenize(sentence) if word not in string.punctuation]
            
            for i in range(1, n+1):
                for ngram in ngrams(tokens, i):
                    # Convert the n-gram tuple to a space-separated string
                    ngram_str = ' '.join(ngram)
                    self.ngram_counts[ngram_str] += 1

    def get_ngram_counts(self,
                         ngram_count_file_path: Optional[str]=None,
                         overwrite: bool = True) -> Dict[str, int]:
        """Returns a dictionary of n-grams and their counts.
        
        Args:
            None
        
        Returns:
            ngram_counts (Dict[str, int]): A dictionary of n-grams and their counts.
            E.g. {'This': 2, 'is': 2, 'a': 1, 'sample': 3}
        """
        directory = os.path.dirname(ngram_count_file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        if os.path.exists(ngram_count_file_path):
            if not overwrite:
                base, ext = os.path.splitext(file_path)
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                file_path = f"{base}_{timestamp}{ext}"
            else:
                file_path = ngram_count_file_path
        
        with open(file_path, 'w') as file:
            json.dump(self.ngram_counts,
                      file,
                      ensure_ascii=False,
                      indent=4)

        return dict(self.ngram_counts)

if __name__ == "__main__":
    df = pd.DataFrame({
        'text': [
            'This is a sample sentence. Here is another one!',
            'Another sample sentence here. And yet another one.',
            'Yet another sample! This continues.'
        ]
    })
    imdb_train = pd.read_csv('data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])

    extractor = NgramExtractor(imdb_train, 'text')
    extractor.extract_ngrams()

    ngram_counts = extractor.get_ngram_counts()
    print(ngram_counts)
    # for ngram, count in ngram_counts.items():
        # print(f"{ngram}: {count}")
