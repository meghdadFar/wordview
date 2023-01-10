import os
import sys
import pandas
import json
from pathlib import Path
from typing import List, Dict
from nltk import word_tokenize
from wordview.mwes.am import calculate_am
from wordview.mwes.mwe_utils import replace_mwes, get_counts
from wordview import logger


class MWE(object):
    def __init__(
        self,
        df: pandas.DataFrame,
        text_column: str,
        mwe_types: List[str] = ["NC"],
        output_dir: str = "tmp",
        tokenize=False,
    ) -> None:
        """Provide functionalities for unsupervised extraction of MWEs through association measures.

        Args:
            df: DataFrame with a text_column that contains the corpus.
            text_column: Specifies the column of DataFrame where text data resides.
            mwe_types: Types of MWEs to be extracted. Supports: NC for Noun-Noun and JNC for Adjective-Noun compounds. Example: ['NC', 'JNC'].
            
            # TODO rm below
            # output_dir: Output directory where counts and MWEs are stored.
            # count_dir: Directory where count_file is sotred.
            # count_file: File in which counts are sotred.
            # mwe_dir: Directory where mwe_file is sotred.
            # mwe_file: File in which MWEs are sotred.
            
            tokenize: Tokenize the content of `text_column`.
            
            #TODO
            Remove all dir stuff and replace them with the following params:
            When `build_counts()` receives an input arg filename, it stores the counts, otherwise return counts.
            
            When `extract_mwes()` receives an input arg: count filename, it reads the counts there, 
            otherwise it runs `build_counts()`. When `extract_mwes()` receives a MWE filename argument,
            it stores MWEs there as json. Otherwise, it returns the corresponding dictionary. 



        Returns:
            None
        """
        self.df = df
        self.text_column = text_column
        for mt in mwe_types:
            if mt not in ["NC", "JNC"]:
                raise ValueError(f"{mt} type is not recognized.")
        self.mwe_types = mwe_types
        
        # self.output_dir = output_dir
        # self.count_dir = os.path.join(self.output_dir, "counts")
        # self.count_file = os.path.join(self.count_dir, "count_data.json")
        # self.mwe_dir = os.path.join(self.output_dir, "mwes")
        # self.mwe_file = os.path.join(self.mwe_dir, "mwe_data.json")

        # Path(self.output_dir).mkdir(exist_ok=True)

        if tokenize:
            logger.info(
                '"tokenize" flag set to True. This might lead to a slow instantiation.'
            )
            self.df[text_column] = self.df[text_column].apply(self._tokenize)
        else:
            self._check_tokenized()

    def _tokenize(self, x):
        """Helper function to tokenize and join the results with a space.

        Args:
            x:

        Returns:
            None
        """
        return " ".join(word_tokenize(x))

    def _check_tokenized(self) -> None:
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
        res = get_counts(
            df=self.df, text_column=self.text_column, mwe_types=self.mwe_types
        )
        # Directory
        # try:
        #     Path(self.count_dir).mkdir(exist_ok=True)
        # except Exception as e:
        #     logger.error(e)
        #     raise e
        # File
        if not counts_filename:
            return res
            # try:
            #     with open(self.count_file, "w") as file:
            #         json.dump(res, file)
            # except Exception as e:
            #     logger.error(e)
            #     raise e
        else:
            try:
                with open(counts_filename, "w") as file:
                    json.dump(res, file)
            except Exception as e:
                logger.error(e)
                raise e

    def extract_mwes(self, am: str = "pmi", mwes_filename: str = None, counts_filename: str = None, counts: Dict = None) -> None:
        """
        Args:
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            am: The association measure to be used. Can be any of [pmi, npmi]
            mwes_filename: File for storing MWEs. Defaults to None.
            counts_filename: File to read counts from.

        Returns:
            None (when no mwes_filename is provided) otherwise mwe_am_dict: Dictionary of MWEs.
        """
        if counts:
            count_data = counts
        else:
            try:
                with open(counts_filename, "r") as file:
                    count_data = json.load(file)
            except Exception as e:
                logger.error(e)
                logger.error("Counts must be provided either via input argument `counts` or `counts_filename`. Argument `counts` is not specified and it seems like there was an error reading the counts from `counts_filename`.")
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
        else:
            return mwe_am_dict


if __name__ == "__main__":
    import pandas as pd
    from wordview.mwes.mwe_utils import replace_mwes

    imdb_train = pd.read_csv(
        "data/imdb_train_sample.tsv", sep="\t", names=["label", "text"]
    )
    mwe = MWE(df=imdb_train, mwe_types=["NC", "JNC"], text_column="text", tokenize=True)
    mwe.build_counts()
    mwe.extract_mwes(am="npmi")
    new_df = replace_mwes(
        path_to_mwes="tmp/mwes/mwe_data.json",
        mwe_types=["NC", "JNC"],
        df=imdb_train,
        text_column="text",
    )
    new_df.to_csv("tmp/new_df.csv", sep="\t")
