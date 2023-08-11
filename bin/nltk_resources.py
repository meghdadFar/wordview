import nltk
import os
from wordview import logger


def check_nltk_resources():
    nltk_data_path = os.path.expanduser('~/nltk_data/')
    
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }

    for path, package in resources.items():
        if not os.path.exists(os.path.join(nltk_data_path, path)):
            logger.info(f"Downloading NLTK resource: {package}")
            nltk.download(package)
        else:
            pass