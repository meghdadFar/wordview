import string
import time
from statistics import median
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from plotly.subplots import make_subplots
from tqdm import tqdm
from wordcloud import WordCloud, get_single_color_func

from bin.nltk_resources import check_nltk_resources
from wordview import logger

check_nltk_resources()


def plotly_barplot(
    token_count_dic: dict, plot_settings: Dict = {}
) -> plotly.graph_objects.Bar:
    """
    Create a bar plot trace for plotly.

    Args:
        token_count_dic (Dict): Dictionary of token to its count
        **kwargs:

    Returns:
        trace (plotly.graph_objects.Bar)
    """
    max_words = plot_settings.get("max_words", 100)
    trace = go.Bar(
        x=sorted(
            list(token_count_dic.keys()), key=lambda x: token_count_dic[x], reverse=True
        )[:max_words],
        y=sorted(list(token_count_dic.values()), reverse=True)[:max_words],
    )
    return trace


def plotly_wordcloud(
    token_count_dic: dict, plot_settings: Dict = {}
) -> plotly.graph_objects.Scattergl:
    """Create a world cloud trace for plotly.

    Args:
        token_count_dic (Dict): Dictionary of token to its count
        **kwargs:

    Returns:
        trace (plotly.graph_objects.Scatter)
    """
    wc = WordCloud(
        color_func=get_single_color_func(plot_settings.get("color", "deepskyblue")),
        max_words=plot_settings.get("max_words", 100),
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
    df: pd.DataFrame, label_cols: List[Tuple], layout_settings: Dict = {}
) -> plotly.graph_objects.Figure:
    """Generate histogram and bar plots for the labels in label_cols.

    Args:
        df (Pandas DataFrame): DataFrame that contains labels specified in label_cols.
        label_cols (list): list of tuples in the form of [('label_1', 'categorical/numerical'),
                           ('label_2', 'categorical/numerical'), ...]
        **kwargs: Additional arguments to pass to plotly.Figure.update_layout(). For more details
                  see https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout

    Returns:
        plotly.graph_objects.Figure
    """

    if len(label_cols) == 1:
        # with titles
        figure = make_subplots(
            rows=1, cols=1, subplot_titles=([label_cols[0][0].capitalize()])
        )
        # w/o titles
        # figure = make_subplots(rows=1, cols=1)
        lab_trace1 = label_plot(
            df, label_col=label_cols[0][0], label_type=label_cols[0][1]
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.update_yaxes(title_text="Count", row=2, col=2)
    elif len(label_cols) == 2:
        # with titles
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                [label_cols[0][0].capitalize(), label_cols[1][0].capitalize()]
            ),
        )
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
        [label_cols[0][0].capitalize(), label_cols[1][0].capitalize()]
        figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                [
                    label_cols[0][0].capitalize(),
                    label_cols[1][0].capitalize(),
                    label_cols[2][0].capitalize(),
                ]
            ),
        )
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
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                [
                    label_cols[0][0].capitalize(),
                    label_cols[1][0].capitalize(),
                    label_cols[2][0].capitalize(),
                    label_cols[3][0].capitalize(),
                ]
            ),
        )
        lab_trace1 = label_plot(
            df,
            label_col=label_cols[0][0],
            label_type=label_cols[0][1],
        )
        lab_trace2 = label_plot(
            df,
            label_col=label_cols[1][0],
            label_type=label_cols[1][1],
        )
        lab_trace3 = label_plot(
            df,
            label_col=label_cols[2][0],
            label_type=label_cols[2][1],
        )
        lab_trace4 = label_plot(
            df,
            label_col=label_cols[3][0],
            label_type=label_cols[3][1],
        )
        figure.append_trace(lab_trace1, 1, 1)
        figure.append_trace(lab_trace2, 1, 2)
        figure.append_trace(lab_trace3, 2, 1)
        figure.append_trace(lab_trace4, 2, 2)
    figure.update_layout(layout_settings)
    return figure


def label_plot(
    df: pd.DataFrame, label_col: str, label_type: str
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
        values = df[label_col].unique().tolist()  # E.g. ['pos', 'neg', 'neutral']
        counts = df[label_col].value_counts()  # E.g. 1212323
        x = []
        y = []
        for v in values:
            x.append(v)
            y.append(counts[v])
        trace = go.Bar(
            x=x,
            y=y,
            name=label_col,
            showlegend=False,
            marker=dict(
                color=y, coloraxis="coloraxis", line=dict(width=0.5, color="white")
            ),
        )
    elif label_type == "numerical":
        trace = go.Histogram(
            x=df[label_col],
            name=label_col,
            showlegend=False,
            marker=dict(
                color=df[label_col],
                coloraxis="coloraxis",
                line=dict(width=0.5, color="white"),
            ),
        )
    else:
        raise ValueError(
            'label_col input argument must be set to either "categorical" or "numerical".'
        )
    return trace


def do_txt_analysis(
    df: pd.DataFrame,
    text_col: str,
    pos_tags: set,
    language: str = "english",
    skip_stopwords_punc: bool = True,
) -> Any:
    """Generate analysis report and eitherr renders the report via Plotly show api or saves it offline to html.

    Args:
        df (pd.DataFrame): DataFrame that contains text and labels.
        pos_tags (set): Set of POS tags to look for in the text.
        text_col (str): Name of the column that contains a tokenized text content.
        language (str): Language of the text in df[text_col]
        skip_stopwords_punc (bool): Whether or not skip stopwords and punctuations in the analysis. Default: True

    Returns:
        TxtAnalysisFields
    """

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
    sentence_lengths = []
    token_to_count_dict: Dict[str, int] = {}
    word_count_by_pos: Dict[str, dict] = {pos: {} for pos in pos_tags}
    languages = set()

    logger.info("Processing text in %s column of the input DataFrame..." % text_col)
    for text in tqdm(df[text_col]):
        ls = detect(text).upper()
        languages.update([ls])
        try:
            doc_len = 0
            doc_tokens = []
            sentences = sent_tokenize(text.lower())
            for sentence in sentences:
                sentence_tokens = word_tokenize(sentence)
                sentence_lengths.append(len(sentence_tokens))
                doc_len += len(sentence_tokens)
                doc_tokens.extend(sentence_tokens)
            doc_lengths.append(doc_len)
            if skip_stopwords_punc:
                doc_tokens = [
                    t
                    for t in doc_tokens
                    if t not in stop_words and t not in punctuations
                ]
                update_count(token_to_count_dict, doc_tokens)

        except Exception as e:
            logger.warning(
                "Processing entry --- %s --- lead to exception: %s" % (text, e.args[0])
            )
            continue

        postag_tokens = nltk.pos_tag(doc_tokens)

        for pos in pos_tags:
            pos_items = get_pos(postag_tokens, pos)
            update_count(word_count_by_pos[pos], pos_items)

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
        sentence_lengths=sentence_lengths,
        zipf_x=x,
        zipf_y_emp=y_emperical,
        zipf_y_theory=y_theoritical,
        languages=languages,
        type_count=vocab_size,
        token_count=n_tokens,
        doc_count=len(doc_lengths),
        median_doc_len=median(doc_lengths),
        word_count_by_pos=word_count_by_pos,
        token_to_count_dict=token_to_count_dict,
    )


class TxtAnalysisFields:
    def __init__(
        self,
        doc_lengths,
        sentence_lengths,
        zipf_x,
        zipf_y_emp,
        zipf_y_theory,
        languages,
        type_count,
        token_count,
        doc_count,
        median_doc_len,
        word_count_by_pos,
        token_to_count_dict,
    ):
        self.doc_lengths = doc_lengths
        self.sentence_lengths = sentence_lengths
        self.zipf_x = zipf_x
        self.zipf_y_emp = zipf_y_emp
        self.zipf_y_theory = zipf_y_theory
        self.languages = languages
        self.type_count = type_count
        self.token_count = token_count
        self.doc_count = doc_count
        self.median_doc_len = median_doc_len
        self.word_count_by_pos = word_count_by_pos
        self.token_to_count_dict = token_to_count_dict
