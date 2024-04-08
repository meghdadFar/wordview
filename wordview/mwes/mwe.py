import re
import string
import threading
from re import Match
from typing import Optional

import nltk
import pandas
from flask import Flask, jsonify, request, send_from_directory
from nltk import RegexpParser, word_tokenize
from openai import OpenAI
from tabulate import tabulate  # type: ignore
from tqdm import tqdm

from bin.nltk_resources import check_nltk_resources
from wordview import logger
from wordview.io.dataframe_reader import DataFrameReader
from wordview.mwes.association_measures import PMICalculator
from wordview.mwes.patterns import DeMWEPatterns, EnMWEPatterns


# TODO use this function to ensure that the tokens are alphanumeric
def is_alphanumeric_latinscript_multigram(word: str) -> Optional[Match[str]]:
    match: Optional[Match] = re.match("[a-zA-Z0-9]{2,}", word)
    return match


check_nltk_resources()


class MWE:
    """Extract MWEs of typeS:
    LVC, VPC, Noun Compounds, Adjective Compounds, and custom patterns from a text corpus.
    """

    def __init__(
        self,
        df: pandas.DataFrame,
        text_column: str,
        ngram_count_source=None,
        ngram_count_file_path=None,
        language: str = "EN",
        custom_patterns: Optional[str] = None,
        only_custom_patterns: bool = False,
    ) -> None:
        """Initializes a new instance of MWE class.

        Args:
            df: A pandas DataFrame containing the text corpus.
            text_column: The name of the column containing the text.
            ngram_count_source: A dictionary containing ngram counts.
            ngram_count_file_path: A path to a json file containing ngram counts.
            language: The language of the corpus. Currently only 'EN' and 'DE' are supported. Defaults = 'EN'.
            custom_pattern: A string pattern to match against the tokens. The pattern must be a string of the following form.
                Examples of user-defined patterns:
                NP: {<DT>?<JJ>*<NN>} # Noun phrase
                You can use multiple and/or nested patterns, separated by a newline character e.g.:
                custom_pattern = '''
                VP: {<MD>?<VB.*><NP|PP|CLAUSE>+$} # Verb phrase
                PROPN: {<NNP>+} # Proper noun
                ADJP: {<RB|RBR|RBS>*<JJ>} # Adjective phrase
                ADVP: {<RB.*>+<VB.*><RB.*>*} # Adverb phrase'''
            only_custom_pattern: If True, only the custom pattern will be used to extract MWEs, otherwise, the default patterns will be used as well.

            Returns:
                None
        """
        self.language = language.upper()
        self.mwes: dict[str, dict[str, float]] = {}
        self.reader = DataFrameReader(df, text_column)
        self.mwe_extractor = None

        mwe_patterns: str = ""
        if language == "EN":
            for _, value in EnMWEPatterns().patterns.items():
                for v in value:
                    mwe_patterns += v + "\n"
        elif language == "DE":
            for _, value in DeMWEPatterns().patterns.items():
                for v in value:
                    mwe_patterns += v + "\n"
        else:
            raise ValueError(
                "Language not supported. Use 'EN' for English or 'DE' for German."
            )

        if custom_patterns:
            if only_custom_patterns:
                if not isinstance(only_custom_patterns, bool):
                    raise TypeError(
                        f"only_custom_patterns argument must be a boolean. Currently it is of type {type(only_custom_patterns)} \
                        with a value of: {custom_patterns}."
                    )
                mwe_patterns = custom_patterns
            else:
                mwe_patterns += "\n" + custom_patterns

        # Create an MWEPatternAssociation extractor object
        self.mwe_extractor = MWEPatternAssociation(
            association_measure=PMICalculator(
                ngram_count_source=ngram_count_source,
                ngram_count_file_path=ngram_count_file_path,
            ),
            custom_pattern=mwe_patterns,
        )

    def chat(self, api_key: str = ""):
        """Chat with OpenAI's latest model about MWEs .
        Access the chat UI in your localhost under http://127.0.0.1:5001/

        Args:
            api_key: OpenAI API key.

        Returns:
            None
        """
        self.api_key = api_key
        self.chat_client = OpenAI(api_key=api_key)
        base_content = f"""Answer any questions about the Multiword Expressions (MWEs) that extracted from the uploaded text corpus by Wordview and are presented in the following MWEs dictionary.
        \n\n
        ------------------------------
        MWEs dictionary:
        ------------------------------
        {self.mwes}
        \n\n
        Important Points:\n
        - Answer the questions without including "According/based on to MWEs dictionary".\n
        - The format of the above dictionary is as follows:\n
            "MWE Type": "MWE instance 1": "Association measure", "MWE instance 2": "Association measure", ...\n
        - There could be other custom types in which case you should just mention the dictionary key.\n
        - Depending on a parameter N set by the user, each MWE type contains at most N instances. But it can contain less or even 0.
        """
        chat_history = [
            {"role": "system", "content": base_content},
        ]
        app = Flask(__name__, static_folder="path_to_your_ui_folder")

        @app.route("/")
        def index():
            return send_from_directory("../chat_ui", "chat.html")

        @app.route("/chat", methods=["POST"])
        def chat():
            user_input = request.json["message"]
            chat_history.append({"role": "user", "content": user_input})
            response = (
                self.chat_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=chat_history,
                )
                .choices[0]
                .message.content
            )
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"reply": response})

        def run():
            app.run(port=5001)

        flask_thread = threading.Thread(target=run)
        flask_thread.start()

    def extract_mwes(
        self,
        sort: bool = True,
        top_n: Optional[int] = None,
    ) -> dict[str, dict[str, float]]:
        """Extract MWEs from the text corpus and add them to self.mwes.

        Args:
            sort: If True, the MWEs will be sorted in descending order of association measure.
            top_n: If provided, only the top n MWEs will be returned.

        Returns:
            None.
        """
        for sentence in tqdm(self.reader.get_sentences()):
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
            if tokens:
                returned_dict = self.mwe_extractor._measure_candidate_association(  # type: ignore
                    tokens=tokens
                )
                for key, inner_dict in returned_dict.items():
                    if key not in self.mwes:
                        self.mwes[key] = {}
                    self.mwes[key].update(inner_dict)

        if sort:
            for key, inner_dict in self.mwes.items():
                tmp = sorted(inner_dict.items(), key=lambda item: item[1], reverse=True)
                if top_n:
                    tmp = tmp[:top_n]
                self.mwes[key] = dict(tmp)

        return self.mwes

    def print_mwe_table(self):
        """Prints a table of MWEs and their association measures.

        Args:
            None

        Returns:
            None
        """
        sub_tables = []
        for section, values in self.mwes.items():
            if values:
                formatted_values = {k: "{:.2f}".format(v) for k, v in values.items()}
                headers = [section, "Association"]
                table_data = list(formatted_values.items())
                table_str = tabulate(
                    table_data, headers=headers, tablefmt="double_outline"
                )
                sub_tables.append(table_str)
        final_table = "\n\n".join(sub_tables)
        print(final_table)


class MWEPatternAssociation:
    """Extract MWE candidates from a list of tokens based on a given pattern."""

    def __init__(self, association_measure: PMICalculator, custom_pattern: str) -> None:
        """Initializes a new instance of MWEExtractor class.

        Args:
            association_measure: An instance of an association measure class.
            custom_pattern: A string pattern to match against the tokens.
                     See the examples of the user-defined patterns below.

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
        self.pattern = custom_pattern
        self.association_measure = association_measure

    def _extract_mwe_candidates(self, tokens: list[str]) -> dict:
        """
        Extract variable-length MWE from tokenized input, using a user-defined POS regex pattern.

        Args:
            tokens (list[str]): A list of tokens from which mwe candidates are to be extracted.

        Returns:
            match_counter (dict[str, dict[str, int]]): A counter dictionary with count of matched strings, grouped by pattern label.
                                                    An empty list if none were found.
        """

        def get_pos_tags(tokens: list[str]) -> list[tuple[str, str]]:
            pos_tags = nltk.pos_tag(tokens)
            return pos_tags

        def validate_input() -> None:
            """Validate input argument `tokens`."""
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
                matches[label].add(" ".join(word for (word, tag) in subtree.leaves()))
        return matches

    def _measure_candidate_association(self, tokens: list[str], threshold: float = 1.0):
        """Measure the association of MWE candidates.

        Args:
            tokens: A list of tokens from which mwe candidates are to be extracted.
            threshold: A threshold value for the association measure. Only MWEs with an association measure above this threshold will be returned.

        Returns:
            A dictionary containing the MWEs and their association measures.
        """
        mwes: dict[str, dict[str, float]] = {}
        for mwe_type, candidate_set in self._extract_mwe_candidates(
            tokens=tokens
        ).items():
            if mwe_type not in mwes:
                mwes[mwe_type] = {}
            for mwe_candidate in candidate_set:
                association = self.association_measure.compute_association(
                    mwe_candidate
                )
                if association > threshold:
                    mwes[mwe_type][mwe_candidate] = association
        return mwes
