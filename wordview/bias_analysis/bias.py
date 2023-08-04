import string

import bias_terms
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

    def _detect_bias_category(self, bias_category):
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

    def detect_bias(self, language="en"):
        gender_categories = bias_terms.get_terms(language, "gender")
        racial_categories = bias_terms.get_terms(language, "racial")
        religion_categories = bias_terms.get_terms(language, "religion")

        return {
            "gender": self._detect_bias_category(bias_category=gender_categories),
            "racial": self._detect_bias_category(bias_category=racial_categories),
            "religion": self._detect_bias_category(bias_category=religion_categories),
        }


if __name__ == "__main__":
    import json

    import pandas as pd

    gender_bias_df = pd.DataFrame(
        {
            "texts": [
                "She is too emotional to lead the team.",
                "Men can't be trusted with house chores.",
                "Women aren't suited for technical roles.",
                "He's a man, so he wouldn't understand.",
            ]
        }
    )

    religious_bias_df = pd.DataFrame(
        {
            "texts": [
                "Muslims are too conservative.",
                "Christians are always pushing their beliefs.",
                "Jews are too focused on money.",
                "Buddhists are detached from reality.",
            ]
        }
    )

    racial_bias_df = pd.DataFrame(
        {
            "texts": [
                "Asians aren't innovative thinkers.",
                "Black people can't be trusted.",
                "White people lack cultural depth.",
                "Hispanics are too lazy to work hard.",
            ]
        }
    )

    no_bias_df = pd.DataFrame(
        {
            "texts": [
                "The sky is blue today.",
                "Cats are often considered independent animals.",
                "Mountains are breathtaking natural formations.",
                "Coffee helps many people start their day.",
            ]
        }
    )

    detector = BiasDetector(no_bias_df, "texts")
    results_en = detector.detect_bias(language="en")
    print(json.dumps(results_en, indent=4))
