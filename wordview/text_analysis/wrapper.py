from typing import Dict, List, Set, Tuple

import pandas
import plotly.figure_factory as ff
import plotly.graph_objs as go
from tabulate import tabulate  # type: ignore

from wordview.text_analysis.core import (
    do_txt_analysis,
    generate_label_plots,
    plotly_wordcloud,
)


class TextStatsPlots:
    """
    Represents Text Statistics and Plots.
    """

    def __init__(
        self,
        df: pandas.DataFrame,
        text_column: str,
        distributions: Set = {"doc_len", "word_frequency_zipf"},
        pos_tags: Set = {"NN", "VB", "JJ"},
    ) -> None:
        """Initialize a new TextStatsPlots object with the given arguments.

        Args:
            df (pandas.DataFrame): DataFrame with a text_column that contains the corpus.
            text_column (str): Specifies the column of DataFrame where text data resides.
            distributions (Set): Set of distribution types to generate and plot. Available distributions are:
                doc_len: document lengths, word_frequency_zipf: Zipfian word frequency distribution.
                Default = {"doc_len", "word_frequency_zipf"}.
            pos_tags (Set): = Set of POS tags for which world cloud is shown. Default = {"NN", "VB", "JJ"}.

        Returns:
            None
        """
        self.df = df
        self.analysis = do_txt_analysis(df=self.df, text_col=text_column)
        self.distributions = distributions
        self.dist_plots = self.create_dist_plots()
        self.pos_tags = pos_tags
        self.pos_plots = self.create_pos_plots()
        self.languages = self.analysis.languages
        self.type_count = self.analysis.type_count
        self.token_count = self.analysis.token_count
        self.num_docs = self.analysis.doc_count
        self.median_doc_len = self.analysis.median_doc_len
        self.num_nns = len(self.analysis.nns)
        self.num_jjs = len(self.analysis.jjs)
        self.num_vbs = len(self.analysis.vs)

    def create_dist_plots(self) -> Dict[str, go.Figure]:
        """Create distribution plots for items in `self.distributions`.

        Args:
            None

        Returns:
            Dictionary of distribution names to plotly go.Figure objects for that distribution.
        """
        res = {}
        if "doc_len" in self.distributions:
            fig_doc_len_dist = ff.create_distplot(
                [self.analysis.doc_lengths], group_labels=["distplot"], colors=["blue"]
            )
            res["doc_len"] = fig_doc_len_dist

        if "word_frequency_zipf" in self.distributions:
            fig_w_freq = go.Figure()
            # Alternative nice color scales that go together:
            # Plotly3
            # ice
            fig_w_freq.add_trace(
                go.Scattergl(
                    x=self.analysis.zipf_x,
                    y=self.analysis.zipf_y_emp,
                    mode="markers",
                    marker=dict(
                        color=self.analysis.zipf_x,
                        colorscale="Tealgrn",
                    ),
                )
            )
            fig_w_freq.add_trace(
                go.Scattergl(
                    x=self.analysis.zipf_x,
                    y=self.analysis.zipf_y_theory,
                    mode="markers",
                    marker=dict(color=self.analysis.zipf_x, colorscale="Reds"),
                )
            )

            res["word_frequency_zipf"] = fig_w_freq

        dist_plot_setup = {
            # 'paper_bgcolor': '#007A78',
            "showlegend": False
        }
        for _, fig in res.items():
            fig.update_layout(dist_plot_setup)

        return res

    def create_pos_plots(self) -> Dict[str, go.Figure]:
        """Create plots for the POS tags specified in items in `self.pos_tags`.

        Args:
            None

        Returns:
            Dictionary of POS tags to plotly go.Figure objects.

        """
        word_cloud_setup = {
            # 'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            # 'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            "showlegend": False,
            "xaxis_showgrid": False,
            "yaxis_showgrid": False,
            "xaxis_zeroline": False,
            "yaxis_zeroline": False,
            "yaxis_visible": False,
            "yaxis_showticklabels": False,
            "xaxis_visible": False,
            "xaxis_showticklabels": False,
        }
        res = {}
        if "NN" in self.pos_tags:
            res["noun_cloud"] = go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.nns)
            )
        if "JJ" in self.pos_tags:
            res["adj_cloud"] = go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.jjs)
            )
        if "VB" in self.pos_tags:
            res["verb_cloud"] = go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.vs)
            )

        for _, fig in res.items():
            fig.update_layout(word_cloud_setup)

        return res

    def show_distplot(self, plot: str):
        self.dist_plots[plot].show()

    def show_word_clouds(self, type: str):
        if type == "all":
            raise ValueError(
                'Type "all" is not supported. Consider changing it to: NN, JJ, or VB.'
            )
        if type == "NN":
            self.pos_plots["noun_cloud"].show()
        if type == "JJ":
            self.pos_plots["adj_cloud"].show()
        if type == "VB":
            self.pos_plots["verb_cloud"].show()

    def show_stats(self) -> None:
        """Print dataset statistics, including:
        Language/s
        Number of unique words
        Number of all words
        Number of documents
        Median document length
        Number of nouns
        Number of adjectives
        Number of verbs.
        """
        table = tabulate(
            [
                ["Language/s", ", ".join(self.languages)],
                ["Unique Words", f"{self.type_count:,d}"],
                ["All Words", f"{self.token_count:,d}"],
                ["Documents", f"{self.num_docs:,d}"],
                ["Median Doc Length", self.median_doc_len],
                ["Nouns", f"{self.num_nns:,d}"],
                ["Adjectives", f"{self.num_jjs:,d}"],
                ["Verbs", f"{self.num_vbs:,d}"],
            ],
            tablefmt="simple_grid",
        )
        print(table)

    def show_insights(self):
        "Show topics, MWEs, clusters,"
        raise NotImplementedError


class LabelStatsPlots:
    """
    Represents Label Statistics and Plots.
    """

    def __init__(
        self,
        df: pandas.DataFrame,
        label_columns: List[Tuple],
    ) -> None:
        """Initialize a new LabelStatsPlots object with the given arguments.

        Args:
            df (pandas.DataFrame): DataFrame with one or more label column/s.
            label_columns (List): List of tuples (column_name, label_type) that specify a label column and its type (categorical or numerical).

        Returns:
            None
        """
        self.df = df
        self.labels_fig = generate_label_plots(self.df, label_cols=label_columns)

    def show_label_plots(self) -> None:
        """Renders label plots.

        Args:
            None

        Returns:
            None
        """
        self.labels_fig.show()
