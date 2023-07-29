import json
from collections import Counter
from typing import Dict, Optional, List

import pandas
import tqdm
from nltk import RegexpParser, word_tokenize

from wordview import logger
from wordview.mwes.am import calculate_am
from wordview.mwes.mwe_utils import get_pos_tags, is_alphanumeric_latinscript_multigram
from wordview.mwes.patterns import ENPatterns, DEPatterns


class MWEFromSentence:
    def __init__(self, sentence: str,
                 language: str = "EN") -> None:
        self.language = language.upper()
        self.mwe_patterns: str = ""
        self.mwe_candidates = None
        if language == "EN":
            for _, value in ENPatterns().patterns.items():
                for v in value:
                    self.mwe_patterns += (v+"\n")
        elif language == "DE":
            for _, value in DEPatterns().patterns.items():
                for v in value:
                    self.mwe_patterns += (v+"\n")
        else:
            raise ValueError("Language not supported. Use 'EN' for English or 'DE' for German.")
        
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        # Chunking
        self.mwe_candidates = MWEExtractor(tokens, self.mwe_patterns).extract_mwe_candidates()
        # TODO Measure association & return MWEs
        return None
        

class MWEExtractor:
    """Extract MWE candidates from a list of tokens based on a given pattern."""

    def __init__(self, tokens: list[str], pattern: str) -> None:
        """Initializes a new instance of HigherOrderMWEExtractor class.

        Args:
            tokens: A list of tokens.
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
        self.tokens = tokens
        self.pattern = pattern
        self._validate_input()

    def _validate_input(self) -> None:
        if not isinstance(self.tokens, list):
            raise TypeError(
                f'Input argument "tokens" must be a list of string. Currently it is of type {type(self.tokens)} \
                with a value of: {self.tokens}.'
            )
        if len(self.tokens) == 0:
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

    def extract_mwe_candidates(self) -> dict:
        """
        Extract variable-length MWE from tokenized input, using a user-defined POS regex pattern.

        Args:
            None

        Returns:
            match_counter (dict[str, dict[str, int]]): A counter dictionary with count of matched strings, grouped by pattern label.
                                                    An empty list if none were found.
        """
        tagged_tokens: list[tuple[str, str]] = get_pos_tags(self.tokens)
        parser = RegexpParser(self.pattern)
        parsed_tokens = parser.parse(tagged_tokens)

        labels: list[str] = [
            rule.split(":")[0].strip() for rule in self.pattern.split("\n") if rule
        ]

        matches: dict[str, list[str]] = {label: [] for label in labels}

        for subtree in parsed_tokens.subtrees():
            label = subtree.label()
            if label in matches:
                matches[label].append(
                    " ".join(word for (word, tag) in subtree.leaves())
                )

        matches_counter: dict[str, dict[str, int]] = {
            label: dict(Counter(match_list)) for label, match_list in matches.items()
        }
        return matches_counter

if __name__ == "__main__":
    sentence = "I will take a walk and give a speech. The coffee shop near the swimming pool sells red apples."
    print(MWEFromSentence(sentence, language="EN").mwe_candidates)