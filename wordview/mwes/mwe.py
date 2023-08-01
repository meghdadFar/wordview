import json
from collections import Counter
from typing import Dict, Optional, List

import pandas
import string
import tqdm
from nltk import RegexpParser, word_tokenize, sent_tokenize


from wordview import logger
# from wordview.mwes.am import calculate_am
from wordview.mwes.mwe_utils import get_pos_tags, is_alphanumeric_latinscript_multigram
from wordview.mwes.patterns import ENPatterns, DEPatterns
from wordview.mwes.association_measures import PMICalculator
from wordview.io.dataframe_reader import DataFrameReader



class MWEFromTokens:
    """Extract MWE candidates from a list of tokens based on a given pattern."""

    def __init__(self,
                 association_measure: PMICalculator,
                 pattern: str) -> None:
        """Initializes a new instance of MWEExtractor class.

        Args:
            pattern: A string pattern to match against the tokens. The pattern must be a string of the following form.

        Examples of user-defined patterns:
        - NP: {<DT>?<JJ>*<NN>} # Noun phrase
        - VP: {<MD>?<VB.*><NP|PP|CLAUSE>+$} # Verb phrase
        - PP: {<IN><NP>} # Prepositional phrase

        You can use multiple and/or nested patterns, separated by a newline character:
        pattern = '''
        NP: {<DT>?<JJ>*<NN>} # Noun phrase
        PROPN: {<NNP>+} # Proper noun
        ADJP: {<RB|RBR|RBS>*<JJ>} # Adjective phrase
        ADVP: {<RB.*>+<VB.*><RB.*>*} # Adverb phrase
        '''

        In this case, patterns of a clause are executed in order.  An earlier
        pattern may introduce a chunk boundary that prevents a later pattern from executing.
        """
        self.pattern = pattern
        self.association_measure = association_measure

    def _extract_mwe_candidates(self,
                                tokens: list[str]) -> dict:
        """
        Extract variable-length MWE from tokenized input, using a user-defined POS regex pattern.

        Args:
            tokens (list[str]): A list of tokens from which mwe candidates are to be extracted.

        Returns:
            match_counter (dict[str, dict[str, int]]): A counter dictionary with count of matched strings, grouped by pattern label.
                                                    An empty list if none were found.
        """
        def validate_input() -> None:
            if not isinstance(tokens, list):
                raise TypeError(
                    f'Input argument "tokens" must be a list of string. Currently it is of type {type(self.tokens)} \
                    with a value of: {self.tokens}.'
                )
            if len(tokens) == 0:
                raise ValueError(
                    'Input argument "tokens" must be a non-empty list of string.'
                )
            if not isinstance(self.pattern, str):
                raise TypeError(
                    f'Input argument "pattern" must be a string. Currently it is of type {type(self.pattern)} \
                    with a value of: {self.pattern}.'
                )
            if len(self.pattern) == 0:
                raise ValueError(
                    'Input argument "pattern" must be a non-zero length string.'
                )
        validate_input()
        tagged_tokens: list[tuple[str, str]] = get_pos_tags(tokens)
        parser = RegexpParser(self.pattern)
        parsed_tokens = parser.parse(tagged_tokens)

        labels: list[str] = [
            rule.split(":")[0].strip() for rule in self.pattern.split("\n") if rule
        ]

        matches: dict[str, set] = {label: set() for label in labels}

        for subtree in parsed_tokens.subtrees():
            label = subtree.label()
            if label in matches:
                matches[label].add(
                    " ".join(word for (word, tag) in subtree.leaves())
                )
        return matches
    
    def extract_mwes(self,
                     tokens: list[str],
                     threshold: float = 1.0):
        mwes: dict[str, dict[str, float]] = {}
        for mwe_type, candidate_set in self._extract_mwe_candidates(tokens=tokens).items():
            if mwe_type not in mwes:
                mwes[mwe_type] = {}
            for mwe_candidate in candidate_set:
                association = self.association_measure.compute_association(mwe_candidate)
                if association > threshold:
                    mwes[mwe_type][mwe_candidate] = association
        return mwes


class MWEFromCorpus:
    def __init__(self,
                 corpus: pandas.DataFrame,
                 text_column: str,
                 ngram_count_source=None,
                 ngram_count_file_path=None,
                 language: str = "EN") -> None:
        
        # Specify the language
        self.language = language.upper()
        self.mwes = {}
        self.reader = DataFrameReader(corpus, text_column)

        # Specify MWE patterns
        mwe_patterns: str = ""
        if language == "EN":
            for _, value in ENPatterns().patterns.items():
                for v in value:
                    mwe_patterns += (v+"\n")
        elif language == "DE":
            for _, value in DEPatterns().patterns.items():
                for v in value:
                    mwe_patterns += (v+"\n")
        else:
            raise ValueError("Language not supported. Use 'EN' for English or 'DE' for German.")
        
        # Create an MWE extractor object
        mwe_extractor = MWEFromTokens(association_measure=PMICalculator(ngram_count_source=ngram_count_source,
                                                                        ngram_count_file_path=ngram_count_file_path),
                                        pattern=mwe_patterns)
        for sentence in self.reader.get_sentences():
            try:
                tokens = [word for word in word_tokenize(sentence) if word not in string.punctuation]
            except Exception as E:
                logger.warning(f'Could not word tokenize sentence: {sentence}.\
                               \n{E}.\
                               \nSkipping this sentence.')
                continue
            if tokens:
                returned_dict = mwe_extractor.extract_mwes(tokens=tokens)
                for key, inner_dict in returned_dict.items():
                    if key not in self.mwes:
                        self.mwes[key] = {}
                    self.mwes[key].update(inner_dict)


if __name__ == "__main__":
    import pandas as pd
    imdb_corpus = pd.read_csv('data/IMDB_Dataset_sample.csv').sample(100)
    mwe_from_corpus = MWEFromCorpus(imdb_corpus, 'review',
                  ngram_count_file_path='data/ngram_counts.json',
                  language='EN')
    formatted_data = json.dumps(mwe_from_corpus.mwes, indent=4)
    print(formatted_data)
    