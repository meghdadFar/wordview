#!/usr/bin/env python
import nltk

def download_req():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')