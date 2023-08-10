from typing import Any, Dict, List, Set, Tuple

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
        self.pos_tags = pos_tags
        self.languages = self.analysis.languages
        self.type_count = self.analysis.type_count
        self.token_count = self.analysis.token_count
        self.num_docs = self.analysis.doc_count
        self.median_doc_len = self.analysis.median_doc_len
        self.num_nns = len(self.analysis.nns)
        self.num_jjs = len(self.analysis.jjs)
        self.num_vbs = len(self.analysis.vs)

    def show_distplot(
        self,
        distribution: str,
        layout_settings: Dict[str, str] = {},
        plot_settings: Dict[str, str] = {},
    ) -> None:
        """Shows distribution plots for `dist`.

        Args:
            dist (str): The distribution for which the plot is to be shown.
                        Can be either of: doc_len" or "word_frequency_zipf.
            **kwargs: Additional arguments to be passed to self._create_dist_plots and then plotly figure factory.
                      For available settings see: https://plotly.com/python/reference/layout/

        Returns:
            None
        """
        if distribution not in self.distributions:
            raise ValueError(
                f"Invalid distribution. Available distributions are: {self.distributions}"
            )
        if distribution == "doc_len":
            self._create_doc_len_plot(layout_settings, plot_settings).show()
        elif distribution == "word_frequency_zipf":
            self._create_word_freq_zipf_plot(layout_settings, plot_settings).show()

    def _create_doc_len_plot(
        self, layout_settings: Dict[str, Any] = {}, plot_settings: Dict[str, str] = {}
    ) -> go.Figure:
        res = ff.create_distplot(
            [self.analysis.doc_lengths],
            group_labels=["distplot"],
            colors=[plot_settings.get("color", "blue")],
        )
        tmp_layout_settings = layout_settings
        tmp_layout_settings.update({"showlegend": False})
        res.update_layout(tmp_layout_settings)
        return res

    def _create_word_freq_zipf_plot(
        self, layout_settings: Dict[str, Any] = {}, plot_settings: Dict[str, str] = {}
    ) -> go.Figure:
        res = go.Figure()
        res.add_trace(
            go.Scattergl(
                x=self.analysis.zipf_x,
                y=self.analysis.zipf_y_emp,
                mode=plot_settings.get("mode", "markers"),
                marker=dict(
                    color=self.analysis.zipf_x,
                    colorscale=plot_settings.get(
                        "emperical_zipf_colorscale", "Tealgrn"
                    ),
                ),
            )
        )
        res.add_trace(
            go.Scattergl(
                x=self.analysis.zipf_x,
                y=self.analysis.zipf_y_theory,
                mode=plot_settings.get("mode", "markers"),
                marker=dict(
                    color=self.analysis.zipf_x,
                    colorscale=plot_settings.get("theoritical_zipf_colorscale", "Reds"),
                ),
            )
        )
        tmp_layout_settings = layout_settings
        tmp_layout_settings.update({"showlegend": False})
        res.update_layout(tmp_layout_settings)
        return res

    def _create_pos_plots(
        self,
        pos: str,
        **kwargs,
    ) -> go.Figure:
        """Create plots for the POS tags specified in items in `self.pos_tags`.

        Args:
            pos (str): The POS tag for which the plot is to be shown.

        Returns:
            Dictionary of POS tags to plotly go.Figure objects.

        """
        word_cloud_plot_mandatory_settings = {
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
        plot_settings = kwargs.get("plot_settings", {})
        plot_settings = {**word_cloud_plot_mandatory_settings, **plot_settings}
        if pos == "NN" and "NN" in self.pos_tags:
            return go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.nns, **kwargs)
            ).update_layout(plot_settings)
        elif pos == "JJ" and "JJ" in self.pos_tags:
            return go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.jjs, **kwargs)
            ).update_layout(plot_settings)
        elif pos == "VB" and "VB" in self.pos_tags:
            return go.Figure(
                plotly_wordcloud(token_count_dic=self.analysis.vs, **kwargs)
            ).update_layout(plot_settings)
        else:
            raise ValueError(
                f"Invalid value for pos: {pos}. Valid values are: {self.pos_tags}"
            )

    def show_word_clouds(self, pos: str, **kwargs) -> None:
        """Shows POS word clouds.

        Args:
            pos (str): Type of POS. Can be any of: [NN, JJ, VB].
            **kwargs: Keyword arguments to be passed to self._create_pos_plots() and wordview.text_analysis.core.plotly_wordcloud().
            This includes:
            - plot_settings: Dictionary of form: for self._create_pos_plots(). For available settings see: https://plotly.com/python/reference/layout/.
            - wc_settings: Dictionary of form: {"color": "<color>", "max_words": int} for core.plotly_wordcloud(). Accepted values are color strings as usable by PIL/Pillow.

        Returns:
            None
        """
        self._create_pos_plots(pos=pos, **kwargs).show()

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
        self.label_columns = label_columns

    def show_label_plots(self, **kwargs) -> None:
        """Renders label plots for columns specified in `self.label_columns`.

        Args:
            **kwargs: Additional arguments to be passed to generate_label_plots() to be used by plotly.Figure.update_layout(). For more details
                  see https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html#plotly.graph_objects.Figure.update_layout

        Returns:
            None
        """
        generate_label_plots(self.df, label_cols=self.label_columns, **kwargs).show()
