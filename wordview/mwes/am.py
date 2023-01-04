from typing import Dict, List
import math


def calculate_pmi(
    compound_dict: dict, word_dic: dict, num_compound: int, num_words: int, normalize: bool = False
) -> Dict[str, float]:
    """Calculate Pointwise Mutual Information between the two words of every word pair in nn_dict.

    Args:
        compound_dict: Dictionary of compounds and their count.
        word_dic: Dictionary of words and their count.
        num_compound: Number of compounds.
        num_words: Number of words.
        normalize: Whether or not normalize the pmi score. Normalized pmi is referred to as npmi.

    Returns:
        sorted_compound_dict: Dictionary of compounds and their pmi/npmi values, sorted wrt their pmi/npmi.
    """
    tmp_compound_dict = compound_dict
    for compound, count in tmp_compound_dict.items():
        w1w2 = compound.split(" ")
        # To filter out compounds that are rare/unique because of strange/misspelled component words.
        if float(word_dic[w1w2[0]]) > 10 and float(word_dic[w1w2[1]]) > 10:
            p_of_c = float(count) / float(num_words)
            p_of_h = float(word_dic[w1w2[0]]) / float(num_words)
            p_of_m = float(word_dic[w1w2[1]]) / float(num_words)
            pmi = math.log(p_of_c / (p_of_h * p_of_m))
            if not normalize:
                tmp_compound_dict[compound] = round(pmi, 2)
            else:
                npmi = pmi / float(-math.log(p_of_c))
                tmp_compound_dict[compound] = round(npmi, 2)
        else:
            tmp_compound_dict[compound] = 0.0
    sorted_compound_dict = dict(sorted(tmp_compound_dict.items(), key=lambda e: e[1], reverse=True))
    return sorted_compound_dict


def calculate_am(count_data: dict, am: str, mwe_types: List[str]) -> Dict[str, Dict]:
    """Read the counts from path_to_counts and for each compound calculates the measure specified by am.

    Args:
        count_data: A dictionary that contains different MWE types and their counts.
        am: Association measure to be used in order to extract MWEs. Can be any of [pmi, npmi]
        mwe_types: Types of MWEs. Can be any of [NC, JNC].

    Returns:
        res: Dictionary of MWE type to their individual MWE to its score dictionary.
    """
    res = {}
    num_words = sum(count_data["WORDS"].values())
    if am == "pmi":
        for mt in mwe_types:
            compound_dict_tmp = calculate_pmi(
                compound_dict=count_data[mt],
                word_dic=count_data["WORDS"],
                num_compound=sum(count_data[mt].values()),
                num_words=num_words,
                normalize=False,
            )
            res[mt] = compound_dict_tmp
    elif am == "npmi":
        for mt in mwe_types:
            compound_dict_tmp = calculate_pmi(
                compound_dict=count_data[mt],
                word_dic=count_data["WORDS"],
                num_compound=sum(count_data[mt].values()),
                num_words=num_words,
                normalize=True,
            )
            res[mt] = compound_dict_tmp
    return res
