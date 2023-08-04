import string

import bias_terms
import pandas as pd
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
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
            # print(f" >> Processing term '{term}' in text '{sentence}' - sentiment: {sentiment_class}")
            sentiments.append(sentiment_class)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return avg_sentiment

    def detect_bias(self, language="en"):
        gender_categories = bias_terms.get_terms(language, "gender")
        racial_categories = bias_terms.get_terms(language, "racial")
        religion_categories = bias_terms.get_terms(language, "religion")

        return {
            "gender": self.detect_bias_category(bias_category=gender_categories),
            "racial": self.detect_bias_category(bias_category=racial_categories),
            "religion": self.detect_bias_category(bias_category=religion_categories),
        }

    def detect_bias_category(self, bias_category):
        biases = {}
        for category_type, category_terms in bias_category.items():
            category_type_avg_sentiment = 0
            n = 0
            for sentence in tqdm(self.reader.get_sentences()):
                try:
                    tokens = [
                        word
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
                    intersection = set(tokens).intersection(category_terms)
                    if intersection:
                        category_type_avg_sentiment += self._calculate_association(
                            category_terms=intersection, sentence=sentence
                        )
                        n += 1
            category_type_avg_sentiment = (
                (category_type_avg_sentiment / n) if n > 0 else "-inf"
            )
            biases[category_type] = category_type_avg_sentiment

        return biases


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "text": [
                "man is talented",
                "she can be hard working",
                "he is always reliable",
            ]
        }
    )
    detector = BiasDetector(df, "text")
    results_en = detector.detect_bias(language="en")
    import json

    print(json.dumps(results_en, indent=4))
