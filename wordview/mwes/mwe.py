from collections import Counter
import re
import pandas
import json
from pathlib import Path
from nltk import word_tokenize
import tqdm
from wordview.mwes.am import calculate_am
from wordview.mwes.mwe_utils import get_pos_tags, is_alphanumeric_latinscript_multigram
from wordview import logger


class MWE(object):
    def __init__(
        self,
        df: pandas.DataFrame,
        text_column: str,
        mwe_types: list[str] = ["NC"],
        tokenize=False,
    ) -> None:
        """Provide functionalities for unsupervised extraction of MWEs through association measures.

        Args:
            df: DataFrame with a text_column that contains the corpus.
            text_column: Specifies the column of DataFrame where text data resides.
            mwe_types: Types of MWEs to be extracted. Supports: NC for Noun-Noun and JNC for Adjective-Noun compounds. Example: ['NC', 'JNC'].

        Returns:
            None
        """
        self.df = df
        self.text_column = text_column
        if not mwe_types:
            raise ValueError(f"mwe_types is empty.")
        if not isinstance(mwe_types, list):
            raise ValueError(f"mwe_types is not a list.")
        for mt in mwe_types:
            if mt not in ["NC", "JNC"]:
                raise ValueError(f"{mt} type is not recognized.")
        self.mwe_types = mwe_types
        if tokenize:
            logger.info(
                '"tokenize" flag set to True. This might lead to a slow instantiation.'
            )
            self.df[text_column] = self.df[text_column].apply(self._tokenize)
        else:
            self._check_tokenized()

    def _tokenize(self, x): #REFACTOR: Move to utils
        """Helper function to tokenize and join the results with a space.

        Args:
            x:

        Returns:
            None
        """
        return " ".join(word_tokenize(x))

    def _check_tokenized(self) -> None: #REFACTOR: Move to utils
        """Helper function to check if the content of text_column is tokenized.

        Args:
            None

        Returns:
            None
        """
        if self.df[self.text_column].shape[0] > 200:
            tests = self.df[self.text_column].sample(n=200).tolist()
        else:
            tests = self.df[self.text_column].sample(frac=0.8).tolist()
        num_pass = 0
        for t in tests:
            try:
                if " ".join(word_tokenize(t)) == t:
                    num_pass += 1
            except Exception as E:
                logger.error(f"Could not tokenize and join tokens in {t}: \n {E} ")

        if float(num_pass) / float(len(tests)) < 0.8:
            logger.warning(
                f"It seems that the content of {self.text_column} in the input data frame is not (fully) tokenized.\nThis can lead to poor results. Consider re-instantiating your MWE instance with 'tokenize' flag set to True.\nNote that this might lead to a slower instantiation."
            )

    def build_counts(self, counts_filename: str = None) -> None:
        """Create various count files to be used by downstream methods
        by calling snlp.mwes.counter.get_counts.

        Args:
            counts_filename: Filename for storing counts.

        Returns:
            None (when no counts_filename is provided) otherwise res: Dictionary of counts.

        """
        logger.info("Creating counts...")
        res = self.get_counts(
            df=self.df, text_column=self.text_column, mwe_types=self.mwe_types
        )
        if not counts_filename:
            return res
        else:
            try:
                with open(counts_filename, "w") as file:
                    json.dump(res, file)
            except Exception as e:
                logger.error(e)
                raise e

    def extract_mwes(
        self,
        am: str = "pmi",
        mwes_filename: str = None,
        counts_filename: str = None,
        counts: dict = None,
    ) -> dict:
        """
        Args:
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            am: The association measure to be used. Can be any of [pmi, npmi]
            mwes_filename: File for storing MWEs. Defaults to None.
            counts_filename: File to read counts from.

        Returns:
            Dictionary of MWEs.
        """
        if counts:
            count_data = counts
        else:
            try:
                with open(counts_filename, "r") as file:
                    count_data = json.load(file)
            except Exception as e:
                logger.error(e)
                logger.error(
                    "Counts must be provided either via input argument `counts` or `counts_filename`. Argument `counts` is not specified and it seems like there was an error reading the counts from `counts_filename`."
                )
                raise e

        logger.info(f"Extracting {self.mwe_types} based on {am}")
        mwe_am_dict = calculate_am(
            count_data=count_data, am=am, mwe_types=self.mwe_types
        )
        if mwes_filename:
            try:
                with open(mwes_filename, "w") as file:
                    json.dump(mwe_am_dict, file)
            except Exception as e:
                logger.error(e)
                raise e
            finally:
                return mwe_am_dict
        else:
            return mwe_am_dict


    def get_counts(self) -> dict:
        """Read a corpus in pandas.DataFrame format and generates all counts necessary for calculating AMs.

        Args:
            df (pandas.DataFrame): DataFrame with input data, which contains a column with text content
                                from which compounds and their counts are extracted.
            text_column: Name of the column the contains the text content.
            mwe_types: Types of MWEs. Can be any of [NC, JNC]

        Returns:
            res: Dictionary of mwe_types to dictionary of individual mwe within that type and their count.
                E.g. {'NC':{'climate change': 10, 'brain drain': 3}, 'JNC': {'black sheep': 3, 'red flag': 2}}
        """
        res = {}
        for mt in self.mwe_types:
            res[mt] = {}
        res["WORDS"] = {}
        for sent in tqdm.tqdm(self.df[self.text_column]):
            tokens = sent.split(" ")
            word_count_dict = Counter(tokens)
            for k, v in word_count_dict.items(): #REFACTOR into its own method. 
                if k in res["WORDS"]:
                    res["WORDS"][k] += v
                else:
                    res["WORDS"][k] = v
            for mt in self.mwe_types:
                mwes_count_dic = self.extract_mwes_from_sent(tokens, mwe_type=mt) #REFACTOR into its own method.
                for k, v in mwes_count_dic.items():
                    if k in res[mt]:
                        res[mt][k] += v
                    else:
                        res[mt][k] = v
        return res


    def extract_mwes_from_sent(self, tokens: list[str], mwe_type: str) -> dict:
        """Extract two-word noun compounds from tokenized input.

        Args:
            tokens: A tokenized sentence, i.e. list of tokens.
            type: Type of MWE. Any of ['NC', 'JNC'].

        Returns:
            mwes_count_dic: Dictionary of compounds to their count.
        """
        if not isinstance(tokens, list):
            raise TypeError(
                f'Input argument "tokens" must be a list of string. Currently it is of type {type(tokens)} \
                with a value of: {tokens}.'
            )
        if len(tokens) == 0:
            return
        mwes = []
        postag_tokens: list[tuple[str, str]] = get_pos_tags(tokens)
        w1_pos_tags = []
        w2_pos_tags = []
        if mwe_type == "NC":
            w1_pos_tags = ["NN", "NNS"]
            w2_pos_tags = ["NN", "NNS"]
        elif mwe_type == "JNC":
            w1_pos_tags = ["JJ"]
            w2_pos_tags = ["NN", "NNS"]
        for i in range(len(postag_tokens) - 1):
            w1 = postag_tokens[i]
            if w1[1] not in w1_pos_tags:
                continue
            else:
                w2 = postag_tokens[i + 1]
                if not is_alphanumeric_latinscript_multigram(w1[0]) or not is_alphanumeric_latinscript_multigram(w2[0]):
                    continue
                if w2[1] in w2_pos_tags:
                    if i + 2 < len(postag_tokens):
                        w3 = postag_tokens[i + 2]
                        if w3 not in ["NN", "NNS"]:
                            mwes.append(w1[0] + " " + w2[0])
                    else:
                        mwes.append(w1[0] + " " + w2[0])
        mwes_count_dic = Counter(mwes)
        return mwes_count_dic
