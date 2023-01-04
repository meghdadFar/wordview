import os
import sys
import pandas
import json
from pathlib import Path
from typing import List
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
        """Provide functionalities around MWEs, for unsupervised extraction of MWEs from text and replacing
        them in the corpus.

        Args:
            df: DataFrame with a text_column that contains the corpus.
            text_col: Specifies the column of DataFrame that contains the corpus. 'text_column' must contain tokenized text.
            mwe_types: Types of MWEs. Can be a list containing any of ['NC', 'JNC'].
            output_dir: Output directory where counts, MWEs and corpus with replaced MWEs are stored.
            count_dir: Directory where count_file is sotred.
            count_file: File in which counts are sotred.
            mwe_dir: Directory where mwe_file is sotred.
            mwe_file: File in which MWEs are sotred.
            tokenize: Tokenize the content of 'text_column'.

        Returns:
            None
        """
        self.df = df
        self.text_col = text_column
        for mt in mwe_types:
            if mt not in ["NC", "JNC"]:
                raise ValueError(f"{mt} type is not recognized.")
        self.mwe_types = mwe_types

        self.output_dir = output_dir
        self.count_dir = os.path.join(self.output_dir, "counts")
        self.count_file = os.path.join(self.count_dir, "count_data.json")
        self.mwe_dir = os.path.join(self.output_dir, "mwes")
        self.mwe_file = os.path.join(self.mwe_dir, "mwe_data.json")
        
        Path(self.output_dir).mkdir(exist_ok=True)

        if tokenize:
            logger.info('"tokenize" flag set to True. This might lead to a slow instantiation.')
            self.df[text_column] = self.df[text_column].apply(self._tokenize)
        else:
            self._check_tokenized()

    def _tokenize(self, x):
        """Helper function to tokenize and join the results with a space.

        Args:
            None

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
        if self.df[self.text_col].shape[0] > 200:
            tests = self.df[self.text_col].sample(n=200).tolist()
        else:
            tests = self.df[self.text_col].sample(frac=0.8).tolist()
        num_pass = 0
        for t in tests:
            try:
                if " ".join(word_tokenize(t)) == t:
                    num_pass += 1
            except Exception as E:
                logger.error(f'Could not tokenize and join tokens in {t}: \n {E} ')

        if float(num_pass) / float(len(tests)) < 0.8:
            logger.warning(
                f"It seems that the content of {self.text_col} in the input data frame is not (fully) tokenized.\nThis can lead to poor results. Consider re-instantiating your MWE instance with 'tokenize' flag set to True.\nNote that this might lead to a slower instantiation."
            )

    def build_counts(self, file_name: str=None) -> None:
        """Create various count files to be used by downstream methods 
        by calling snlp.mwes.counter.get_counts.

        Args:
            None

        Returns:
            None
        """
        logger.info("Creating counts...")
        res = get_counts(df=self.df, text_column=self.text_col, mwe_types=self.mwe_types)
        # Directory
        try:
            Path(self.count_dir).mkdir(exist_ok=True)
        except Exception as e:
            logger.error(e)
            raise e
        # File
        if not file_name:
            try:
                with open(self.count_file, "w") as file:
                    json.dump(res, file)
            except Exception as e:
                logger.error(e)
                raise e
        else:
            try:            
                with open(file_name, "w") as file:
                    json.dump(res, file)
            except Exception as e:
                logger.error(e)
                raise e
            self.count_file = file_name

    def extract_mwes(self, am: str = "pmi", file_name: str=None) -> None:
        """
        Args:
            mwe_types: Types of MWEs. Can be any of [NC, JNC]
            am: The association measure to be used. Can be any of [pmi, npmi]

        Returns:
            None
        """
        with open(self.count_file, "r") as file:
            count_data = json.load(file)
        logger.info(f"Extracting {self.mwe_types} based on {am}")
        mwe_am_dict = calculate_am(count_data=count_data, am=am, mwe_types=self.mwe_types)
        # Dir
        try:
            Path(self.mwe_dir).mkdir(exist_ok=True)
        except Exception as e:
            logger.error(e)
            raise e
        # File
        if not file_name:
            try:
                with open(self.mwe_file, "w") as file:
                    json.dump(mwe_am_dict, file)
            except Exception as e:
                logger.error(e)
                raise e
        else:
            try:
                with open(file_name, "w") as file:
                    json.dump(mwe_am_dict, file)
            except Exception as e:
                logger.error(e)
                raise e


if __name__ == "__main__":
    import pandas as pd
    from wordview.mwes.mwe_utils import replace_mwes

    imdb_train = pd.read_csv("data/imdb_train_sample.tsv", sep="\t", names=["label", "text"])
    mwe = MWE(df=imdb_train, mwe_types=["NC", "JNC"], text_column="text", tokenize=True)
    mwe.build_counts()
    mwe.extract_mwes(am="npmi")
    new_df = replace_mwes(
        path_to_mwes="tmp/mwes/mwe_data.json", mwe_types=["NC", "JNC"], df=imdb_train, text_column="text"
    )
    new_df.to_csv("tmp/new_df.csv", sep="\t")
