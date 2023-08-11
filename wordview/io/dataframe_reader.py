from nltk.tokenize import sent_tokenize

from bin.nltk_resources import check_nltk_resources
from wordview import logger

check_nltk_resources()


class DataFrameReader:
    """Reads a dataframe column and returns sentences."""

    def __init__(self, dataframe, column_name):
        """Initializes the DataFrameReader object.

        Args:
            dataframe: A pandas dataframe.
            column_name: A string representing the column name in the dataframe where corpus resides.

        Returns:
            None
        """
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
                logger.warning(f"Could not sentence tokenize text: {text}")
                logger.warning(E)
                continue
            for sentence in sentences:
                yield sentence
