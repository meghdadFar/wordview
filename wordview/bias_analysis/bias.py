import string
from typing import Dict, Optional, Union

import plotly.graph_objects as go
from nltk import word_tokenize
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from tabulate import tabulate  # type: ignore
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from bin.nltk_resources import check_nltk_resources
from wordview import logger
from wordview.bias_analysis import bias_terms
from wordview.io.dataframe_reader import DataFrameReader

check_nltk_resources()


class BiasDetector:
    def __init__(self, df, text_column):
        # bert-base-multilingual-uncased-sentiment supports English, Dutch, German, French, Spanish, and Italian.
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.sentiment_tokenizer = BertTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.embedding_model = SentenceTransformer(
            "distiluse-base-multilingual-cased-v2"
        )
        self.reader = DataFrameReader(df, text_column)
        self.biases = {}

    def _calculate_association(self, sentence, category_terms):
        sentiments = []
        for term in category_terms:
            inputs = self.sentiment_tokenizer.encode_plus(
                f"{term} {sentence}",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            output = self.sentiment_model(**inputs)
            sentiment_class = output[0].argmax(1).item()
            sentiments.append(sentiment_class)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return avg_sentiment

    def _detect_bias_category(self, bias_category):
        biases = {}
        for category_type, category_terms in bias_category.items():
            category_type_avg_sentiment = 0
            n = 0
            for sentence in tqdm(self.reader.get_sentences()):
                try:
                    tokens = [
                        word.lower()
                        for word in word_tokenize(sentence)
                        # TODO support languages other than English
                        if word not in string.punctuation
                    ]
                except Exception as E:
                    logger.warning(
                        f"Could not word tokenize sentence: {sentence}.\
                                \n{E}.\
                                \nSkipping this sentence."
                    )
                    continue
                if tokens:
                    intersection = set(tokens).intersection(
                        [cat_term.lower() for cat_term in category_terms]
                    )
                    if intersection:
                        association_result = self._calculate_association(
                            category_terms=intersection, sentence=sentence
                        )
                        category_type_avg_sentiment += association_result
                        n += 1
            category_type_avg_sentiment = (
                (category_type_avg_sentiment / n) if n > 0 else "-inf"
            )
            biases[category_type] = category_type_avg_sentiment
        return biases

    def show_bias_plot(
        self,
        colorscale: Optional[Union[str, list[list]]] = None,
        layout_settings: Optional[dict] = None,
        font_settings: Dict = {
            "colorbar_tick_font": {"size": 16},
            "colorbar_title_font": {"size": 18},
            "bias_category_xaxis_font": {"size": 16},
            "title_font": {"size": 20},
            "category_titles": {"size": 18},
        },
    ):
        """
        Displays a plotly heatmap of the bias scores for each category.

        Args:
            colorscale: The colorscale to use for the heatmap.
                If not provided, the default colorscale is used.
                You can define a custom colorscale by providing a list of lists of the form:
                    cyan_scopecolorscale = [
                        [0.0, "#E0FFFF"],  # Lightest Cyan
                        [0.25, "#B3E4E4"],  # Lighter Cyan
                        [0.5, "#66C2C2"],   # Neutral Cyan
                        [0.75, "#339999"],  # Darker Cyan
                        [1.0, "#006666"],   # Darkest Cyan
                    ]
                Or you can use one of the built-in colorscales by providing a string.
                Example of available colorscales are:'aggrnyl', 'agsunset', 'algae', 'amp',
                'armyrose', 'balance', 'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl',
                'brbg','brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl'

                You can reverse a colorscale by appending an '_r' to it, e.g.
                'algae_r'
                # See here for a full list:
                # https://plotly.com/python/builtin-colorscales/

            layout_settings: A dictionary of layout settings to apply to the plot.
                If not provided, the default layout settings are used.
                Example:
                    layout_settings = {'plot_bgcolor':'rgba(245, 245, 245, 1)',
                    'paper_bgcolor': 'rgba(255, 255, 255, 1)',
                    'hovermode': 'y'
                    }
                For a full list of possible options, see:
                https://plotly.com/python/reference/layout/

            font_settings: A dictionary of font sizes for color bar, tick, title and subtitle fonts.


        """
        categories = list(self.biases.keys())
        fig = make_subplots(
            rows=len(categories),
            cols=1,
            subplot_titles=[cat.capitalize() for cat in categories],
        )
        if colorscale is None:
            scopecolorscale = [
                [0.0, "#8B0000"],  # Very Negative
                [0.25, "#FF4500"],  # Negative
                [0.5, "yellow"],  # Neutral
                [0.75, "#ADFF2F"],  # Positive
                [1.0, "#006400"],  # Very Positive
            ]
        elif isinstance(colorscale, str) or isinstance(colorscale, list):
            scopecolorscale = colorscale  # type: ignore
        else:
            raise ValueError(
                f"Invalid colorscale value: {colorscale}.\
                    \nMust be a string or list of lists."
            )

        for index, (category, sub_data) in enumerate(self.biases.items()):
            labels = list(sub_data.keys())
            labels = [label.capitalize() for label in labels]
            values = [val if val != "-inf" else None for val in sub_data.values()]

            heatmap = go.Heatmap(
                z=[values],
                x=labels,
                zmin=0,
                zmax=4,
                colorscale=scopecolorscale,
                showscale=True,
                colorbar_title="Bias",
                colorbar=dict(
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=[
                        "Very Negative",
                        "Negative",
                        "Neutral",
                        "Positive",
                        "Very Positive",
                    ],
                    tickfont=dict(size=16),
                    titlefont=dict(size=18),
                ),
                zauto=False,  # Prevents auto scaling
            )
            fig.add_trace(heatmap, row=index + 1, col=1)
            fig.update_yaxes(showgrid=False, showticklabels=False)
            fig.update_xaxes(showgrid=False, tickfont=dict(size=16))
        if layout_settings is not None:
            fig.update_layout(layout_settings)
        else:
            fig.update_layout(
                title="Bias Scores Across Categories",
                # title_font_size=20,  # Increase title font size
                title_font=dict(size=20),  # Increase title font size
                title_x=0.5,  # Center main title
                width=1000,  # Fixed width
                height=250
                * len(categories),  # Adjust height based on number of categories
                plot_bgcolor="white",  # Set background color to white
            )
        # Increase font size of subplot titles.
        fig.update_annotations(font=dict(size=18))
        fig.show()

    def print_bias_table(self):
        def sentiment_to_bias(val):
            if val == "-inf":
                return "Unknown"
            elif val == 0:
                return "Very Negative"
            elif val == 1:
                return "Negative"
            elif val == 2:
                return "Neutral"
            elif val == 3:
                return "Positive"
            elif val == 4:
                return "Very Positive"

        sub_tables = []
        for section, values in self.biases.items():
            if values:
                formatted_values = {
                    k.capitalize(): sentiment_to_bias(v) for k, v in values.items()
                }
                headers = [section.capitalize(), "Bias"]
                table_data = list(formatted_values.items())
                table_str = tabulate(
                    table_data, headers=headers, tablefmt="double_outline"
                )
                sub_tables.append(table_str)
        final_table = "\n\n".join(sub_tables)
        print(final_table)

    def detect_bias(self, language="en"):
        gender_categories = bias_terms.get_terms(language, "gender")
        racial_categories = bias_terms.get_terms(language, "racial")
        religion_categories = bias_terms.get_terms(language, "religion")

        self.biases = {
            "gender": self._detect_bias_category(bias_category=gender_categories),
            "racial": self._detect_bias_category(bias_category=racial_categories),
            "religion": self._detect_bias_category(bias_category=religion_categories),
        }
        return self.biases
