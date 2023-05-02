from typing import Dict, List, Optional
import json
import re
from re import Match
import sys
from collections import Counter


import nltk
import pandas
from re import Match
import tqdm

from wordview import logger


def hyphenate_mwes(
    path_to_mwes: str,
    mwe_types: List[str],
    df: pandas.DataFrame,
    text_column: str,
    am_threshold: float = 0.7,
    only_mwes: bool = False,
    lower_case: bool = False,
) -> pandas.DataFrame:
    """Hyphenates the mwes in the corpus so that they are treated as a single token by downstream applications.

    Args:
        path_to_mwes: Path to a json file that contains a dictionary of MWE type for each type,
                    unique MWEs to their count. E.g. {'NC': {'mwe1': 10}}.
        mwe_types: Types of MWEs to be replaced. Can be any of [NC, JNC].
        df: DataFrame comprising training data with a tokenized text column.
        text_column: Text (content) column of df.
        am_threshold: MWEs with an am greater than or equal to this threshold are selected for replacement.
        only_mwes: Whether or not keep only MWEs and drop the rest of the text.
        lower_case: Whether or not lowercase the sentence before replacing MWEs.

    Returns:
        df (pandas.FataFrame)
    """
    try:
        with open(path_to_mwes, "r") as file:
            mwe_type_mwe_am = json.load(file)
    except Exception as e:
        raise e
    good_mwes = set()
    for t in mwe_types:
        logger.info(f"Replacing MWEs of type {t} in the corpus.")
        if t not in mwe_type_mwe_am:
            logger.error(
                f"MWEs of type {t} do not exist. Make sure the file containing MWEs in {path_to_mwes} includes type {t}"
            )
            sys.exit(1)
        pmi_sorted_dict = mwe_type_mwe_am[t]
        logger.info(f"Number of all MWEs of type {t}: {len(pmi_sorted_dict)}")
        for k, v in pmi_sorted_dict.items():
            if v >= am_threshold:
                good_mwes.add(k)
            else:
                break
        logger.info(
            "Number of MWEs to be replaced in corpus based on the association threshold: %d"
            % len(good_mwes)
        )

    logger.info("Replacing compounds in text")
    new_text = []
    for sent in tqdm.tqdm(df[text_column]):
        sent = sent.lower() if lower_case else sent
        bigrams = get_ngrams(sent, 2)
        if only_mwes:
            tmp = ""
            for bg in bigrams:
                if bg in good_mwes:
                    tmp += bg.split(" ")[0] + "-" + bg.split(" ")[1] + " "
            if tmp != "":
                sent = tmp.strip()
        else:
            for bg in bigrams:
                if bg in good_mwes:
                    sent = re.sub(bg, bg.split(" ")[0] + "-" + bg.split(" ")[1], sent)
        new_text.append(sent)
    df[text_column] = new_text
    return df


def get_ngrams(sentence: str, n: int) -> List[str]:
    """Extracts n-grams from sentence.

    Args:
        sentence: Input sentence from which n-grams are to be extracted.
        n: Size of n-grams.

    Returns:
        ngrams: List of extracted n-grams.
    """
    if not isinstance(sentence, str):
        raise TypeError(
            f"Input argument 'sentence' must be of type str. You have provided an input of type: {type(sentence)}."
        )
    ngrams: List[str] = []
    try:
        tokens = sentence.split(" ")
    except Exception as E:
        logger.error(E)
        logger.error(
            f'Input "{sentence}" cannot be spilitted around space. No n-gram is extracted.'
        )
        return ngrams
    for i in range(len(tokens) - n + 1):
        ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def get_pos_tags(tokens: list[str]) -> list[tuple[str, str]]:
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


def is_alphanumeric_latinscript_multigram(word: str) -> Optional[Match]:
    match: Match = re.match("[a-zA-Z0-9]{2,}", word)
    return match
