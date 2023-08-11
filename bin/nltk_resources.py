import nltk


def check_nltk_resources():
    resources = ['punkt',
                 'stopwords',
                 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res)
