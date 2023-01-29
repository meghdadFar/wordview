import re
from typing import Dict, Set

from nltk import word_tokenize


def clean_text(
    text: str,
    keep_pattern: str = "[a-zA-Z0-9!.,?]",
    drop_patterns: Set[str] = set([]),
    replace: Dict = {},
    remove_emojis=False,
    remove_blanks=False,
    maxlen: int = 15,
    lower=False,
) -> str:
    """Tokenize and clean text, by matching it against keep_pattern and droping and replacing provided patterns.

    Args:
        text (str): Input text.
        keep_pattern (str): Allowed patterns e.g. [a-zA-Z]. Defaults to "[a-zA-Z0-9!.,?]".
        drop_patterns (set): Set of patterns that should be dropeed from text.
        replace (dict): Dictionary of to_be_replaced_pattern: replaced_with. E.g. {[0-9]+: NUM}
        remove_emojis:‌ Whether or not to remove emojis. Defaults to False.
        remove_blanks: Relevant for text scraped from Web or HTML tags. Remove excessive blank lines and whitespaces. Defaults to False.
        maxlen (int): Maximum length of a token. Defaults to 15.
        lower (bool): Whether or nor lowercase the text at the end.

    Returns:
        out_text (string): Tokenized and cleaned up text with respect to all above criteria.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if len(text) == 0:
        raise ValueError("Input must be a non empty string.")

    if remove_emojis:
        text = remove_emojis(text)

    if remove_blanks:
        text = re.sub(" {2,}", " ", text)
        text = re.sub("\n{2,}", "\n", text)

    # Drop unwanted tokens: Replace them with space, then replace resulting \s{2, } with one space
    for d in drop_patterns:  # d = e.g. <br/>
        if re.search(d, text):
            text = re.sub(d, " ", text)
    text = re.sub(" {2,}", " ", text)

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


def remove_emojis(text: str) -> str:
    """Remove Emojis from `text`.

    Args:
        text: Input text to remove emojis from.

    Returns:
        text with emojis removed.
    """
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", text)
