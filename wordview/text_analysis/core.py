import string
import time
from statistics import median
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
import pandas
import pandas as pd
import plotly
import plotly.graph_objs as go
from langdetect import detect
from nltk.corpus import stopwords
from plotly.subplots import make_subplots
from tqdm import tqdm
from wordcloud import WordCloud, get_single_color_func

from wordview import logger


def plotly_wordcloud(token_count_dic: dict, **kwargs) -> plotly.graph_objects.Scattergl:
    """Create a world cloud trace for plotly.

    Args:
        token_count_dic (Dict): Dictionary of token to its count
        **kwargs:

    Returns:
        trace (plotly.graph_objects.Scatter)
    """
    wc_settings: Dict = kwargs.get(
        "wc_settings", {"color": "deepskyblue", "max_words": 100}
    )
    wc = WordCloud(
        color_func=get_single_color_func(wc_settings["color"]),
        max_words=wc_settings["max_words"],
    )
    wc.generate_from_frequencies(token_count_dic)
    word_list = []
    rel_freq_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []
    for (word, rel_freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        rel_freq_list.append(rel_freq)
        freq_list.append(token_count_dic[word])
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
    # get the relative occurrence frequencies
    new_freq_list = []
    for i in rel_freq_list:
        i_tmp = round(i * 100, 4)
        i_tmp = (
            i_tmp if i_tmp > 1 else 1
        )  # Plotly textfont.size in go.Scatter throws exception for values below 1.
        new_freq_list.append(i_tmp)
    try:
        trace = go.Scattergl(
            x=x,
            y=y,
            textfont=dict(size=new_freq_list, color=color_list),
            hoverinfo="text",
            hovertext=["{0}: {1}".format(w, f) for w, f in zip(word_list, freq_list)],
            mode="text",
            text=word_list,
        )
        return trace
    except Exception as E:
        logger.error(
            f"While creating the word cloud, plotly.go returned the following error \
         \n{E}\nfor relative frequencies: {rel_freq_list}\nthat were mapped to {new_freq_list}"
        )


def generate_label_plots(
    df: pandas.DataFrame, label_cols: List[Tuple]
) -> plotly.graph_objects.Figure:
    """Generate histogram and bar plots for the labels in label_cols.

    Args:
        figure (plotly.graph_objs.Figure): Figure object in which the plots are created.
        df (Pandas DataFrame): DataFrame that contains labels specified in label_cols.
        label_cols (list): list of tuples in the form of [('label_1', 'categorical/numerical'),
                           ('label_2', 'categorical/numerical'), ...]

    Returns:
        plotly.graph_objects.Figure
    """

    if len(label_cols) == 1:
        # with titles
        # figure = make_subplots(rows=1, cols=1,subplot_titles=("Plot 1"))
        # w/o titles
        figure = make_subplots(rows=1, cols=1)
        lab_trace1 = label_plot(
            df, label_col=label_cols[0][0], label_type=label_cols[0][1]
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.update_yaxes(title_text="Count", row=2, col=2)
    elif len(label_cols) == 2:
        # with titles
        # figure = make_subplots(rows=1, cols=2,subplot_titles=("Plot 1", "Plot 2"))
        # w/o titles
        figure = make_subplots(rows=1, cols=2)
        lab_trace1 = label_plot(
            df, label_col=label_cols[0][0], label_type=label_cols[0][1]
        )
        lab_trace2 = label_plot(
            df, label_col=label_cols[1][0], label_type=label_cols[1][1]
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.append_trace(lab_trace2, 1, 2)
    elif len(label_cols) == 3:
        figure = make_subplots(rows=1, cols=3)
        lab_trace1 = label_plot(
            df, label_col=label_cols[0][0], label_type=label_cols[0][1]
        )
        lab_trace2 = label_plot(
            df, label_col=label_cols[1][0], label_type=label_cols[1][1]
        )
        lab_trace3 = label_plot(
            df, label_col=label_cols[2][0], label_type=label_cols[2][1]
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.append_trace(lab_trace2, 1, 2)
        figure.append_trace(lab_trace3, 1, 3)
    elif len(label_cols) == 4:
        figure = make_subplots(rows=2, cols=2)
        lab_trace1 = label_plot(
            df, label_col=label_cols[0][0], label_type=label_cols[0][1]
        )
        lab_trace2 = label_plot(
            df, label_col=label_cols[1][0], label_type=label_cols[1][1]
        )
        lab_trace3 = label_plot(
            df, label_col=label_cols[2][0], label_type=label_cols[2][1]
        )
        lab_trace4 = label_plot(
            df, label_col=label_cols[3][0], label_type=label_cols[3][1]
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.append_trace(lab_trace2, 1, 2)
        figure.append_trace(lab_trace3, 2, 1)
        figure.append_trace(lab_trace4, 2, 2)
    return figure


def label_plot(
    df: pandas.DataFrame, label_col: str, label_type: str
) -> plotly.graph_objects.Histogram:
    """Create a plot for label_col in df, wrt to label_type.

    Args:
        df (Pandas DataFrame): DataFrame that contains label_col.
        label_col (str): Name of the label column in df that must be plotted.
        label_type (str): Represents the type of label and consequently specifies the type of plot.
                             It can be "numerical" or "categorical".

    Returns:
        trace (plotly.graph_objects.Histogram)
    """
    if label_type == "categorical":
        values = df[label_col].unique().tolist()  # ['pos', 'neg', 'neutral']
        counts = df[label_col].value_counts()  # 1212323
        x = []
        y = []
        for v in values:
            x.append(v)
            y.append(counts[v])
        trace = go.Bar(x=x, y=y, name=label_col)
    elif label_type == "numerical":
        trace = go.Histogram(
            x=df[label_col],
            name=label_col,
            marker=dict(line=dict(width=0.5, color="white")),
        )
    else:
        raise ValueError(
            'label_col input argument must be set to either "categorical" or "numerical".'
        )
    return trace


def do_txt_analysis(
    df: pandas.DataFrame,
    text_col: str,
    language: str = "english",
    skip_stopwords_punc: bool = True,
) -> Any:
    """Generate analysis report and eitherr renders the report via Plotly show api or saves it offline to html.

    Args:
        df (pandas.DataFrame): DataFrame that contains text and labels.
        text_col (str): Name of the column that contains a tokenized text content.
        language (str): Language of the text in df[text_col]
        skip_stopwords_punc (bool): Whether or not skip stopwords and punctuations in the analysis. Default: True

    Returns:
        TxtAnalysisFields
    """

    global ftmodel

    def update_count(items_dic: dict, items: List[str]) -> None:
        """Update the corresponding count for each key in  items_dic. w.r.t. terms in items.

        Args:
            items_dic (dict): Dictionary mapping keys to their count
            items (list): List of tokens

        Returns:
            None
        """
        for t in items:
            if t in items_dic:
                items_dic[t] += 1
            else:
                items_dic[t] = 1

    def get_pos(tagged_tokens: List[Tuple[str, str]], goal_pos: str) -> List:
        """Extracts goal_pos POS tags from tagged_tokens.

        Args:
            tagged_tokens (List[Tuple(str, str)]): Contains terms and ther pos tags. E.g.
                                                   [('cat', 'NN'), ('sat', 'VBD'), ('on', 'IN'), ('mat', 'NN')]
            goal_pos (str): Pos tag to look for in tagged_tokens

        Returns:
            res (List(str)): List of tokens with goal_pos pos tag
        """
        res = []
        for pt in tagged_tokens:
            if pt[1].startswith(goal_pos):
                res.append(pt[0])
        return res

    stop_words = set(stopwords.words(language))
    punctuations = set(string.punctuation)

    doc_lengths = []
    token_to_count_dict: Dict[str, int] = {}
    NNs: Dict[str, int] = {}
    JJs: Dict[str, int] = {}
    Vs: Dict[str, int] = {}
    languages = set()

    logger.info("Processing text in %s column of the input DataFrame..." % text_col)
    for text in tqdm(df[text_col]):
        ls = detect(text).upper()
        languages.update([ls])
        try:
            tokens = text.lower().split(" ")
            doc_lengths.append(len(tokens))
            if skip_stopwords_punc:
                tokens = [
                    t for t in tokens if t not in stop_words and t not in punctuations
                ]
                update_count(token_to_count_dict, tokens)

        except Exception as e:
            logger.warning(
                "Processing entry --- %s --- lead to exception: %s" % (text, e.args[0])
            )
            continue

        postag_tokens = nltk.pos_tag(tokens)
        nouns = get_pos(postag_tokens, "NN")
        update_count(NNs, nouns)
        verbs = get_pos(postag_tokens, "VB")
        update_count(Vs, verbs)
        adjectives = get_pos(postag_tokens, "JJ")
        update_count(JJs, adjectives)

    freq_df = pd.DataFrame(
        {"tokens": token_to_count_dict.keys(), "count": token_to_count_dict.values()}
    )
    freq_df["proportion"] = freq_df["count"] / freq_df["count"].sum()
    vocab_size = freq_df.shape[0]
    n_tokens = freq_df["count"].sum()
    s = 1
    denom = np.sum(1 / (np.arange(1, n_tokens + 1) ** s))

    def classic_zipf(k, s=s, denom=denom):
        num = 1 / k**s
        res = num / denom
        return res

    logger.info("Calculating Empirical and Theoretical Zipf values...")
    freq_df["position"] = freq_df.index + 1
    start = time.time()
    freq_df["predicted_proportion"] = freq_df["position"].apply(classic_zipf)
    end = time.time()
    logger.info(
        f"Time to measure predicted proportion for {freq_df.shape[0]} rows: {end - start}"
    )

    x = np.log(freq_df["position"].values)
    y_emperical = np.log(freq_df["count"])
    y_theoritical = np.log(freq_df["predicted_proportion"] * n_tokens)

    return TxtAnalysisFields(
        doc_lengths=doc_lengths,
        zipf_x=x,
        zipf_y_emp=y_emperical,
        zipf_y_theory=y_theoritical,
        languages=languages,
        type_count=vocab_size,
        token_count=n_tokens,
        doc_count=len(doc_lengths),
        median_doc_len=median(doc_lengths),
        nns=NNs,
        jjs=JJs,
        vs=Vs,
        token_to_count_dict=token_to_count_dict,
    )


class TxtAnalysisFields:
    def __init__(
        self,
        doc_lengths,
        zipf_x,
        zipf_y_emp,
        zipf_y_theory,
        languages,
        type_count,
        token_count,
        doc_count,
        median_doc_len,
        nns,
        jjs,
        vs,
        token_to_count_dict,
    ):
        self.doc_lengths = doc_lengths
        self.zipf_x = zipf_x
        self.zipf_y_emp = zipf_y_emp
        self.zipf_y_theory = zipf_y_theory
        self.languages = languages
        self.type_count = type_count
        self.token_count = token_count
        self.doc_count = doc_count
        self.median_doc_len = median_doc_len
        self.nns = nns
        self.jjs = jjs
        self.vs = vs
        self.token_to_count_dict = token_to_count_dict
