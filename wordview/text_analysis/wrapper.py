import threading
from typing import Any, Tuple

import pandas
import plotly.figure_factory as ff
import plotly.graph_objs as go
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
from tabulate import tabulate  # type: ignore

from wordview.text_analysis.core import (
    do_txt_analysis,
    generate_label_plots,
    plotly_barplot,
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
        distributions: set = {"doc_len", "word_frequency_zipf", "sentence_len"},
        pos_tags: set = {
            "CC",
            "CD",
            "DT",
            "EX",
            "FW",
            "IN",
            "JJ",
            "JJR",
            "JJS",
            "LS",
            "MD",
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "PDT",
            "POS",
            "PRP",
            "PRP$",
            "RB",
            "RBR",
            "RBS",
            "RP",
            "SYM",
            "TO",
            "UH",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "WDT",
            "WP",
            "WP$",
            "WRB",
        },
    ) -> None:
        """Initialize a new TextStatsPlots object with the given arguments.

        Args:
            df: DataFrame with a text_column that contains the text corpus.
            text_column: Specifies the column of DataFrame where text data resides.
            distributions: set of distribution types to generate and plot. Available distributions are: \n
                `doc_len`: Document lengths \n
                `sentence_len`: Sentence lengths \n
                `word_frequency_zipf`: Zipfian word frequency distribution. \n
                Default = ``{'doc_len', 'word_frequency_zipf'}`` \n
            pos_tags: A set of target POS tags for downstream analysis. \n
                Default = ``{'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
                'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'}`` \n

        Returns:
            None
        """
        self.df = df
        self.analysis = do_txt_analysis(
            df=self.df, text_col=text_column, pos_tags=pos_tags
        )
        self.distributions = distributions
        self.pos_tags = pos_tags
        self.languages = self.analysis.languages
        self.type_count = self.analysis.type_count
        self.token_count = self.analysis.token_count
        self.num_docs = self.analysis.doc_count
        self.median_doc_len = self.analysis.median_doc_len
        self.pos_counts = {
            k: len(v) for k, v in self.analysis.word_count_by_pos.items()
        }

    def chat(self, api_key: str = ""):
        """Chat with OpenAI's latest model about the results of Wordview's text analysis.

        Args:
            api_key: OpenAI API key.

        Returns:
            None
        """
        self.api_key = api_key
        self.chat_client = OpenAI(api_key=api_key)
        base_content = f"""Answer any questions about text and corpus analysis based on the following dictionary of Wordview Analysis.
        \n\n
        ------------------------------
        Wordview Analysis:
        ------------------------------
        {self.return_stats()}
        \n\n
        In your answers, avoid using According to the Wordview Analysis dictionary.
        In this integration, you are Wordview.
        """
        chat_history = [
            {"role": "system", "content": base_content},
        ]
        app = Flask(__name__, static_folder="path_to_your_ui_folder")

        @app.route("/")
        def index():
            return send_from_directory("chat", "chat.html")

        @app.route("/chat", methods=["POST"])
        def chat():
            user_input = request.json["message"]
            chat_history.append({"role": "user", "content": user_input})
            print(f"User: {user_input}")
            response = (
                self.chat_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=chat_history,
                )
                .choices[0]
                .message.content
            )
            print(f"Wordview: {response}")
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"reply": response})

        def run():
            app.run(port=5000)

        flask_thread = threading.Thread(target=run)
        flask_thread.start()

        # while True:
        #     user_prompt = input("You: ")
        #     chat_history.append({"role": "user", "content": user_prompt})
        #     response = (
        #         self.chat_client.chat.completions.create(
        #             model="gpt-3.5-turbo",
        #             messages=chat_history,
        #         )
        #         .choices[0]
        #         .message.content
        #     )
        #     print(f"Wordview: {response}")
        #     chat_history.append({"role": "assistant", "content": response})

    def show_distplot(
        self,
        distribution: str,
        layout_settings: dict[str, str] = {},
        plot_settings: dict[str, str] = {},
    ) -> None:
        """Shows distribution plots for `distribution`.

        Args:
            distribution: The distribution for which the plot is to be shown. Available distributions are: \n
                `doc_len`: document lengths \n
                `word_frequency_zipf`: Zipfian word frequency distribution. \n
            layout_settings: To customize the plot layout. For example: \n

                .. code-block:: python

                    layout_settings = {'plot_bgcolor':'rgba(245, 245, 245, 1)',
                                       'paper_bgcolor': 'rgba(255, 255, 255, 1)',
                                       'hovermode': 'y'}

                For a full list of possible options, see:
                https://plotly.com/python/reference/layout/

            plot_settings: A dictionary of form: ``{"<plot_setting>": "<value>"}`` for each one of the supported plots, in order to customize the plot colors and other attributes.
                For example, for `word_frequency_zipf` and `doc_len` plots, you can, respectively pass:

                .. code-block:: python

                    plot_settings = {'theoritical_zipf_colorscale': 'Reds',
                                     'emperical_zipf_colorscale': 'Greens',
                                     'mode': 'markers'}

                    plot_settings = {'color': 'blue',
                                     'showlegend': False}

                You can pass all the attributes for different available distribution plots at once, but not all of them are supported across all plots. The supported attributes will be extracted and used for each distribution type.

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
        elif distribution == "sentence_len":
            self._create_sentence_len_plot(layout_settings, plot_settings).show()

    def _create_doc_len_plot(
        self, layout_settings: dict[str, Any] = {}, plot_settings: dict[str, str] = {}
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
        self, layout_settings: dict[str, Any] = {}, plot_settings: dict[str, str] = {}
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

    def _create_sentence_len_plot(
        self, layout_settings: dict[str, Any] = {}, plot_settings: dict[str, str] = {}
    ) -> go.Figure:
        res = ff.create_distplot(
            [self.analysis.sentence_lengths],
            group_labels=["distplot"],
            colors=[plot_settings.get("color", "midnightblue")],
        )
        tmp_layout_settings = layout_settings
        tmp_layout_settings.update({"showlegend": False})
        res.update_layout(tmp_layout_settings)
        return res

    def _create_pos_plots(
        self,
        pos: str,
        option: str,
        layout_settings: dict[str, Any] = {},
        plot_settings: dict[str, Any] = {},
    ) -> go.Figure:
        word_cloud_layout_fixed_settings = {
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

        bar_plot_layout_fixed_settings = {
            "xaxis_title": "Word",
            "yaxis_title": "Count",
            "xaxis_dtick": 1,
        }

        if pos in self.pos_tags and pos in self.pos_counts:
            if option == "word_cloud":
                layout_settings.update(word_cloud_layout_fixed_settings)
                return go.Figure(
                    plotly_wordcloud(
                        self.analysis.word_count_by_pos[pos], plot_settings
                    )
                ).update_layout(layout_settings)
            elif option == "bar_plot":
                layout_settings.update(bar_plot_layout_fixed_settings)
                return go.Figure(
                    plotly_barplot(self.analysis.word_count_by_pos[pos], plot_settings)
                ).update_layout(layout_settings)
        else:
            raise ValueError(
                f"Invalid value for pos: {pos}. Valid values are: {self.pos_tags}"
            )

    def show_word_clouds(
        self,
        pos: str,
        layout_settings: dict[str, Any] = {},
        plot_settings: dict[str, str] = {},
    ) -> None:
        """Shows POS word clouds.

        Args:
            pos: Type of POS. Can be any of the Penn POS tags
            (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).
            layout_settings: To customize the plot layout. For example:
                layout_settings = {'plot_bgcolor':'rgba(245, 245, 245, 1)',
                   'paper_bgcolor': 'rgba(255, 255, 255, 1)',
                   'hovermode': 'y'
                    }
            plot_settings = To customize the plot colors and other attributes. For example:
                {'color': 'darkgreen',
                    'max_words': 200}

        Returns:
            None
        """
        self._create_pos_plots(pos, "word_cloud", layout_settings, plot_settings).show()

    def show_bar_plots(
        self,
        pos: str,
        layout_settings: dict[str, Any] = {},
        plot_settings: dict[str, str] = {},
    ) -> None:
        """
        Shows POS bar plots.

        Args:
            pos: Type of POS. Can be any of the Penn POS tags
            (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).
            layout_settings: To customize the plot layout. For example:
                layout_settings = {'plot_bgcolor':'rgba(245, 245, 245, 1)',
                   'paper_bgcolor': 'rgba(255, 255, 255, 1)',
                   'hovermode': 'y'
                    }
            plot_settings = To customize the plot colors and other attributes. For example:
                {'color': 'darkgreen',
                    'max_words': 200}
        """

        self._create_pos_plots(pos, "bar_plot", layout_settings, plot_settings).show()

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
                ["Nouns", f"{self.pos_counts['NN']:,d}"],
                ["Adjectives", f"{self.pos_counts['JJ']:,d}"],
                ["Verbs", f"{self.pos_counts['VB']:,d}"],
                ["Proper Nouns", f"{self.pos_counts['NNP']:,d}"],
                ["Adverbs", f"{self.pos_counts['RB']:,d}"],
            ],
            tablefmt="simple_grid",
        )
        print(table)

    def return_stats(self) -> dict[str, Any]:
        """Returns dataset statistics, including:
        Language/s
        Number of unique words
        Number of all words
        Number of documents
        Median document length
        Number of nouns
        Number of adjectives
        Number of verbs.
        """
        return {
            "Language/s": ", ".join(self.languages),
            "Unique Words": f"{self.type_count:,d}",
            "All Words": f"{self.token_count:,d}",
            "Documents": f"{self.num_docs:,d}",
            "Median Doc Length": self.median_doc_len,
            "Nouns": f"{self.pos_counts['NN']:,d}",
            "Adjectives": f"{self.pos_counts['JJ']:,d}",
            "Verbs": f"{self.pos_counts['VB']:,d}",
            "Proper Nouns": f"{self.pos_counts['NNP']:,d}",
            "Adverbs": f"{self.pos_counts['RB']:,d}",
        }

    def show_insights(self):
        """Prints insights about the dataset."""
        raise NotImplementedError


class LabelStatsPlots:
    """
    Represents Label Statistics and Plots.
    """

    def __init__(
        self,
        df: pandas.DataFrame,
        label_columns: list[Tuple],
    ) -> None:
        """Initialize a new LabelStatsPlots object with the given arguments.

        Args:
            df: DataFrame with one or more label column/s.
            label_columns: list of tuples (column_name, label_type) that specify a label column and its type (categorical or numerical).

        Returns:
            None
        """
        self.df = df
        self.label_columns = label_columns

    def show_label_plots(self, layout_settings: dict[str, Any] = {}) -> None:
        """Renders label plots for columns specified in `self.label_columns`.

        Args:
            layout_settings: To customize the plot layout.
            For example: layout_settings ={'plot_bgcolor':'rgba(245, 245, 245, 1)',
                   'paper_bgcolor': 'rgba(255, 255, 255, 1)',
                   'hovermode': 'y',
                   'coloraxis': {'colorscale': 'peach'},
                   'coloraxis_showscale':True
                  }
            See here for a list of named color scales:
            https://plotly.com/python/builtin-colorscales/

        Returns:
            None
        """
        generate_label_plots(self.df, self.label_columns, layout_settings).show()
