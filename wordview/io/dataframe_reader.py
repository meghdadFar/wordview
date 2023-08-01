from wordview import logger
from nltk.tokenize import sent_tokenize


class DataFrameReader:
    """Reads a dataframe column and returns sentences."""
    def __init__(self, dataframe, column_name):
        if column_name not in dataframe.columns:
            raise ValueError(f"'{column_name}' not found in the dataframe.")
        self.data = dataframe[column_name]

    def get_sentences(self):
        for text in self.data:
            try:
                sentences = sent_tokenize(text)
            except Exception as E:
                logger.warning(f'Could not sentence tokenize text: {text}')
                logger.warning(E)
                continue
            for sentence in sentences:
                yield sentence