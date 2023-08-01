from wordview import logger
from nltk.tokenize import sent_tokenize


class DataFrameReader:
    """Reads a dataframe column and returns sentences."""
    def __init__(self, dataframe, column_name):
        if column_name not in dataframe.columns:
            raise ValueError(f"'{column_name}' not found in the dataframe.")
        self.data = dataframe[column_name]

    def get_sentences(self):
        """Returns a generator of sentences from the dataframe column.
        An nltk.sent_tokenize is applied to each row of the dataframe column 
        to extract sentences from the text at each row of the specified column i.e. `column_name`.

        Args:
            None
        Returns:
            A generator of sentences.
        """
        for text in self.data:
            try:
                sentences = sent_tokenize(text)
            except Exception as E:
                logger.warning(f'Could not sentence tokenize text: {text}')
                logger.warning(E)
                continue
            for sentence in sentences:
                yield sentence