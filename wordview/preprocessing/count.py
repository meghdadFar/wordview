import json
import os
import string
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from wordview import logger
from wordview.io.dataframe_reader import DataFrameReader


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

    def __init__(self, dataframe: pd.DataFrame, column_name: str):
        """Initializes a new instance of NgramExtractor class.

        Args:
            dataframe (pd.DataFrame): A pandas dataframe.
            column_name (str): The name of the text column where corpus resides.

        Returns:
            None
        """
        self.reader = DataFrameReader(dataframe, column_name)
        self.ngram_counts: Dict[str, int] = defaultdict(int)

    def extract_ngrams(self, n: int = 4):
        """Extracts n-grams from the text column of the dataframe.

        Args:
            n (int): The maximum number of words in an n-gram. Defaults to 4.

        Returns:
            None
        """
        for sentence in self.reader.get_sentences():
            try:
                tokens = [
                    word
                    for word in word_tokenize(sentence)
                    if word not in string.punctuation
                ]
            except Exception as E:
                logger.warning(
                    f"Could not word tokenize sentence: {sentence}.\
                               \n{E}.\
                               \nSkipping this sentence."
                )
                continue

            for i in range(1, n + 1):
                for ngram in ngrams(tokens, i):
                    # Convert the n-gram tuple to a space-separated string
                    ngram_str = " ".join(ngram)
                    self.ngram_counts[ngram_str] += 1

    def get_ngram_counts(
        self, ngram_count_file_path: Optional[str] = None, overwrite: bool = True
    ) -> Dict[str, int]:
        """Returns a dictionary of n-grams and their counts.

        Args:
            ngram_count_file_path (Optional[str]): The path to the file where n-gram counts will be saved.
                                                    If not provided, the n-gram counts will not be saved to file.
            overwrite (bool): Whether to overwrite the file if it already exists. Defaults to True.

        Returns:
            ngram_counts (Dict[str, int]): A dictionary of n-grams and their counts.
                                           E.g. {'This': 2, 'is': 2, 'a': 1, 'sample': 3}
        """
        if ngram_count_file_path:
            directory = os.path.dirname(ngram_count_file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            file_path = ngram_count_file_path
            if os.path.exists(file_path):
                if not overwrite:
                    base, ext = os.path.splitext(file_path)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_path = f"{base}_{timestamp}{ext}"

            with open(file_path, "w") as file:
                json.dump(self.ngram_counts, file, ensure_ascii=False, indent=4)

        return dict(self.ngram_counts)


if __name__ == "__main__":
    imdb_train = pd.read_csv("data/IMDB_Dataset_sample.csv")
    extractor = NgramExtractor(imdb_train, "review")
    extractor.extract_ngrams()
    extractor.get_ngram_counts(ngram_count_file_path="data/ngram_counts.json")
