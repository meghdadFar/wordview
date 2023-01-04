from typing import Dict, List
import json
import sys
import pandas
import re
import tqdm
from wordview import logger
import nltk
from collections import Counter


def replace_mwes(
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
        logger.info(f'Replacing MWEs of type {t} in the corpus.')
        if t not in mwe_type_mwe_am:
            logger.error(f'MWEs of type {t} do not exist. Make sure the file containing MWEs in {path_to_mwes} includes type {t}')
            sys.exit(1)
        pmi_sorted_dict = mwe_type_mwe_am[t]
        logger.info(f"Number of all MWEs of type {t}: {len(pmi_sorted_dict)}")
        for k, v in pmi_sorted_dict.items():
            if v >= am_threshold:
                good_mwes.add(k)
            else:
                break
        logger.info("Number of MWEs to be replaced in corpus based on the association threshold: %d" % len(good_mwes))

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


def get_ngrams(sentence: str, n: int) -> List:
    """Extracts n-grams from sentence.

    Args:
        sentence: Input sentence from which n-grams are to be extracted.
        n: Size of n-grams.

    Returns:
        ngrams: List of extracted n-grams.
    """
    ngrams = []
    try:
        tokens = sentence.split(" ")
    except Exception as E:
        logger.error(E)
        logger.error(f'Input "{sentence}" cannot be spilitted around space. No n-gram is extracted.')
        return ngrams
    for i in range(len(tokens) - n + 1):
        ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def get_counts(df: pandas.DataFrame, text_column: str, mwe_types: List[str]) -> dict:
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
    for mt in mwe_types:
        res[mt] = {}
    res["WORDS"] = {}
    for sent in tqdm.tqdm(df[text_column]):
        tokens = sent.split(" ")
        word_count_dict = Counter(tokens)
        for k, v in word_count_dict.items():
            if k in res["WORDS"]:
                res["WORDS"][k] += v
            else:
                res["WORDS"][k] = v
        for mt in mwe_types:
            mwes_count_dic = extract_mwes_from_sent(tokens, mwe_type=mt)
            for k, v in mwes_count_dic.items():
                if k in res[mt]:
                    res[mt][k] += v
                else:
                    res[mt][k] = v
    return res


def extract_mwes_from_sent(tokens: List[str], mwe_type: str) -> Dict:
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
    postag_tokens = nltk.pos_tag(tokens)
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
            if not re.match("[a-zA-Z0-9]{2,}", w1[0]) or not re.match("[a-zA-Z0-9]{2,}", w2[0]):
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
