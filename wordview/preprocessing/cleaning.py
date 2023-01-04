import re
from nltk import word_tokenize
from typing import Set, Dict


def clean_text(
    text: str,
    keep_pattern: str = "[a-zA-Z0-9!.,?]",
    drop_patterns: Set[str]=set([]),
    replace: Dict = {},
    maxlen: int = 15,
    lower=False
) -> str:
    """Tokenize and clean text, by matching it against keep_pattern and droping and replacing provided patterns.

    Args:
        text (str): Input text.
        keep_pattern (str): Allowed patterns e.g. [a-zA-Z]. Defaults to "[a-zA-Z0-9!.,?]".
        drop_patterns (set): Set of patterns that should be dropeed from text.
        replace (dict): Dictionary of to_be_replaced_pattern: replaced_with. E.g. {[0-9]+: NUM}
        maxlen (int): Maximum length of a token. Defaults to 15.
        lower (bool): Whether or nor lowercase the text at the end.


    Returns:
        out_text (string): Tokenized and cleaned up text with respect to all above criteria.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if len(text) == 0:
        raise ValueError("Input must be a non empty string.")

    # Drop unwanted tokens: Replace them with space, then replace resulting \s{2, } with one space
    for d in drop_patterns:  # d = e.g. <br/>
        if re.search(d, text):
            text = re.sub(d, " ", text)
    text = re.sub("\s{2,}", " ", text)

    for k, v in replace.items():
        if re.search(k, text):
            text = re.sub(k, v, text)

    tokens = word_tokenize(text)
    out_tokens = []
    for t in tokens:
        if len(t) < maxlen:
            if re.match(keep_pattern, t):
                out_tokens.append(t)
    out_text = " ".join(out_tokens)
    if lower:
        out_text = out_text.lower()
    return out_text
