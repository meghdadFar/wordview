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
from wordview.mwes.patterns import EnMWEPatterns, DeMWEPatterns
from wordview.mwes.association_measures import PMICalculator
from wordview.io.dataframe_reader import DataFrameReader



class MWEPatternAssociation:
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
    
    def measure_candidate_association(self,
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


class MWE:
    def __init__(self,
                 corpus: pandas.DataFrame,
                 text_column: str,
                 ngram_count_source=None,
                 ngram_count_file_path=None,
                 language: str = "EN", 
                 custom_pattern: Optional[str] = None,
                 only_custom_pattern: bool = False) -> None:
        """Initializes a new instance of MWE class.
        
        Args:
            corpus: A pandas DataFrame containing the corpus.
            text_column: The name of the column containing the text  (corpus).
            ngram_count_source: A dictionary containing ngram counts.
            ngram_count_file_path: A path to a json file containing ngram counts.
            language: The language of the corpus. Currently only 'EN' and 'DE' are supported.
            custom_pattern: pattern: A string pattern to match against the tokens. The pattern must be a string of the following form.
                Examples of user-defined patterns:
                NP: {<DT>?<JJ>*<NN>} # Noun phrase
                You can use multiple and/or nested patterns, separated by a newline character e.g.:
                custom_pattern = '''
                NP: {<DT>?<JJ>*<NN>} # Noun phrase
                VP: {<MD>?<VB.*><NP|PP|CLAUSE>+$} # Verb phrase
                PROPN: {<NNP>+} # Proper noun
                ADJP: {<RB|RBR|RBS>*<JJ>} # Adjective phrase
                ADVP: {<RB.*>+<VB.*><RB.*>*} # Adverb phrase'''
            only_custom_pattern: If True, only the custom pattern will be used to extract MWEs, otherwise, the default patterns will be used as well.

            Returns:
                None
        """
        self.language = language.upper()
        self.mwes = {}
        self.reader = DataFrameReader(corpus, text_column)
        self.mwe_extractor = None

        mwe_patterns: str = ""
        if language == "EN":
            for _, value in EnMWEPatterns().patterns.items():
                for v in value:
                    mwe_patterns += (v+"\n")
        elif language == "DE":
            for _, value in DeMWEPatterns().patterns.items():
                for v in value:
                    mwe_patterns += (v+"\n")
        else:
            raise ValueError("Language not supported. Use 'EN' for English or 'DE' for German.")
        
        if custom_pattern:
            if only_custom_pattern:
                mwe_patterns = custom_pattern
            else:
                mwe_patterns += ("\n"+custom_pattern)
        
        # Create an MWEPatternAssociation extractor object
        self.mwe_extractor = MWEPatternAssociation(association_measure=PMICalculator(ngram_count_source=ngram_count_source,
                                                                        ngram_count_file_path=ngram_count_file_path),
                                        pattern=mwe_patterns)
        
    def extract_mwes(self) -> dict[str, dict[str, float]]:
        for sentence in self.reader.get_sentences():
            try:
                tokens = [word for word in word_tokenize(sentence) if word not in string.punctuation]
            except Exception as E:
                logger.warning(f'Could not word tokenize sentence: {sentence}.\
                            \n{E}.\
                            \nSkipping this sentence.')
                continue
            if tokens:
                returned_dict = self.mwe_extractor.measure_candidate_association(tokens=tokens)
                for key, inner_dict in returned_dict.items():
                    if key not in self.mwes:
                        self.mwes[key] = {}
                    self.mwes[key].update(inner_dict)
        return self.mwes

if __name__ == "__main__":
    import pandas as pd
    imdb_corpus = pd.read_csv('data/IMDB_Dataset_sample.csv').sample(500)
    mwe_from_corpus = MWE(imdb_corpus, 'review',
                  ngram_count_file_path='data/ngram_counts.json',
                  language='EN')
    json.dump(mwe_from_corpus.extract_mwes(), open('data/mwes.json', 'w'), indent=4)
    
    