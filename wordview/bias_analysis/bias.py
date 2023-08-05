import string

import bias_terms
import plotly.graph_objects as go
from nltk import word_tokenize
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from tabulate import tabulate  # type: ignore
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

from wordview import logger
from wordview.io.dataframe_reader import DataFrameReader


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
                        print(
                            f"> Category type: {category_type}\n\tcategory terms: {category_terms}\n\tsentence: {sentence}\n\tassoociation: {association_result}\n--------------------"
                        )
                        category_type_avg_sentiment += association_result
                        n += 1
            category_type_avg_sentiment = (
                (category_type_avg_sentiment / n) if n > 0 else "-inf"
            )
            biases[category_type] = category_type_avg_sentiment
        return biases

    def show_plot(self):
        categories = list(self.biases.keys())
        fig = make_subplots(
            rows=len(categories),
            cols=1,
            subplot_titles=[cat.capitalize() for cat in categories],
        )
        colorscale = [
            [0.0, "#8B0000"],  # Very Negative
            [0.25, "#FF4500"],  # Negative
            [0.5, "yellow"],  # Neutral
            [0.75, "#ADFF2F"],  # Positive
            [1.0, "#006400"],  # Very Positive
        ]
        for index, (category, sub_data) in enumerate(self.biases.items()):
            labels = list(sub_data.keys())
            labels = [label.capitalize() for label in labels]
            values = [val if val != "-inf" else None for val in sub_data.values()]

            heatmap = go.Heatmap(
                z=[values],
                x=labels,
                zmin=0,
                zmax=4,
                colorscale=colorscale,
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
                ),
                zauto=False,  # Prevents auto scaling
            )
            fig.add_trace(heatmap, row=index + 1, col=1)
            fig.update_yaxes(showgrid=False, showticklabels=False)
            fig.update_xaxes(
                showgrid=False,
            )
        fig.update_layout(
            title="Bias Scores Across Categories",
            title_font_size=18,  # Increase title font size
            title_x=0.5,  # Center main title
            width=1200,  # Fixed width
            height=300 * len(categories),  # Adjust height based on number of categories
            plot_bgcolor="white",  # Set background color to white
        )
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


if __name__ == "__main__":
    import json

    import pandas as pd

    biased_df = pd.DataFrame(
        {
            "text": [
                # Gender
                "He is always so successful because men are naturally gifted.",
                "She is too emotional to lead; women cannot handle pressure.",
                # Racial
                "Asians are brilliant at mathematics.",
                "Black people are not fit for this kind of job.",
                "Indian food is too spicy, and their traditions are too complex to understand.",
                "Latinos are one of the many ethnic groups in the region.",  # neutral
                # Religious
                "Christians are the only ones who know the true path to salvation.",
                "Muslims cannot be trusted in our community.",
                "Atheists often have a logical and evidence-based approach to understanding the world.",
            ]
        }
    )

    detector = BiasDetector(biased_df, "text")
    results_en = detector.detect_bias()
    print(json.dumps(results_en, indent=4))
    detector.print_bias_table()
    detector.show_plot()
