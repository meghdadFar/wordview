# wordview

[![PyPI version](https://badge.fury.io/py/wordview.svg?&kill_cache=1)](https://badge.fury.io/py/wordview)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


WORDVIEW is a Python package for primarily text analysis. It, moreover, provides a number of unsupervised models for Information Extraction and Preprocessing. See section [Features](#Features) for different functionalities.

WORDVIEW is open-source and free. We, however, developed a Dashboard version of WORDVIEW based on Plotly, for non-developers. See WORDVIEW-DASH page here.


# Features
- [Text Analysis](#text-analysis)
- Information Extraction
  - [Extraction of Multiword Expressions (statistically idiomatic aka collocations)](#extraction-of-multiword-expressions)
  - Extraction of Non-compositional Multiword Expressions (semantically idiomatic e.g. *red tape* and *brain drain*)
  - Anomalies
- General
  - Entropy Calculation for Natural Language Entropy has a wide range of applications in NLP. See how it can be used to improve the quality of conversational AI [[1]](#1) and text summarization [[2]](#2).
- [Text Cleaning](#text-cleaning)
  - [Identification and filtering of Statistically Redundant Words](#identification-of-statistically-redundant-words)
  - [Auto Text Cleaning](#auto-text-cleaning)


# Usage

Install the package:

`pip install wordview`

To demo different functionalities, let's first load a test dataset.

```python
import pandas as pd
imdb_train = pd.read_csv('data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])
imdb_train.head()

  label                                               text
0   neg  well , i rented this movie and found out it re...
1   pos  you know , this movie is n't that great , but ...
2   pos  a heartwarming film . the usual superb acting ...
3   pos  i did n't expect to like this film as much as ...
4   pos  i could n't help but feel that this could have...
```


## **Text Analysis**

To analyze the text, you can use `TextStatsPlots` class. 

```python
from wordview.text_analysis import TextStatsPlots
ta = TextStatsPlots(df=imdb_train, text_column='text')
```
Use the `show_stats` method to see an overview of different statistics.

```python
ta.show_stats()
┌───────────────────┬─────────┐
│ Language/s        │ EN      │
├───────────────────┼─────────┤
│ Unique Words      │ 48,791  │
├───────────────────┼─────────┤
│ All Words         │ 666,898 │
├───────────────────┼─────────┤
│ Documents         │ 5,000   │
├───────────────────┼─────────┤
│ Median Doc Length │ 211.0   │
├───────────────────┼─────────┤
│ Nouns             │ 28,482  │
├───────────────────┼─────────┤
│ Adjectives        │ 19,519  │
├───────────────────┼─────────┤
│ Verbs             │ 15,241  │
└───────────────────┴─────────┘
```
You can also look into different distributions using `show_distplot` method.
For instance, the distribution of document lengths:
```python
ta.show_distplot(plot='doc_len')
```
![annotation1](/figs/doclen.png)
Or, the distribution of words:
```python
ta.show_distplot(plot='word_frequency_zipf')
```
![annotation1](/figs/wordszipf.png)

You can moreover, see different part of speech tags in corresponding word clouds: 
```python
# To see verbs
ta.show_word_clouds(type="VB")
# To see nouns
ta.show_word_clouds(type="NN")
# To see adjectives
ta.show_word_clouds(type="JJ")
```
![annotation1](/figs/verbs.png)
![annotation1](/figs/nouns.png)
![annotation1](/figs/adjectives.png)

## **Extraction of Multiword Expressions**

Multiword Expressions (MWEs) are phrases that behave as a single semantic unit E.g. *swimming pool* and *climate change*. You can use `wordview` to identify different types of MWEs in your text leveraging statistical measures such as *PMI* and *NPMI*. See a worked example below.

```python
from wordview.mwes import MWE
mwe = MWE(df=imdb_train, mwe_types=["NC", "JNC"], text_column='text')

# Creating counts is a time consuming procedure
# Hence, you can run it once and store the counts, by providing counts_filename argument.
mwe.build_counts(counts_filename='tmp/counts.json')

# One counts are created, extraction of MWEs is fast and can be carried out with different parameters
# Use the optional `mwes_filename` parameter to store the extracted MWEs.
mwe.extract_mwes(counts_filename='tmp/counts.json', mwes_filename='tmp/mwes.json')

```


**Notes**
- If the text in `text_column` is partly tokenized or not tokenized at all, this issue is recognized at instantiation time and shows you a warning. If you already know that your text is not tokenized, you can run the same instantiation with flag `tokenize=True`. 
- `mwes.json` contains dictionary of mwe types e.g. (*NC, JNC*) and their association scores (e.g. *PMI*).
- The MWEs in this json file are sorted with respect to their `am` score.

Top 5 examples from imdb review corpus:

```
NOUN-NOUN COMPOUNDS (NC)
-------------------
jet li
clint eastwood
monty python
kung fu
blade runner


ADJECTIVE-NOUN COMPOUNDS (JNC)
------------------------
spinal tap
martial arts
citizen kane
facial expressions
global warming
```

A practical use of extracting MWEs is to treat them as a single unit. Research shows that when MWEs are treated as a single token, they can improve the performance of downstream applications such as classification and NER. Using `replace_mwes` function, you can hyphenate the extracted MWEs in the corpus (global warming --> global-warming). This will force downstream tokenizers to treat them as a single token. A worked example can be seen below:

```python
from wordview.mwes import hyphenate_mwes
new_df = hyphenate_mwes(path_to_mwes='tmp/mwes.json', mwe_types=['NC', 'JNC'], df=imdb_train, text_column='text')
```

## **Identification of Statistically Redundant Words**

Redundant words carry little value and can exacerbate the results of many NLP tasks. To solve this issue, traditionally, a pre-defined list of words, called stop words was defined and removed from the data. However, creating such a list is not optimal because in addition to being a rule-based and manual approach which does not generalize well, one has to assume that there is a universal list of stop words that represents highly low entropy words for all corpora, which is a very strong assumption and not necessarily a true assumption in many cases.

To solve this issue, one can use a purely statistical solution which is completely automatic and does not make any universal assumption. It focuses only on the corpus at hand. Words can be represented with various statistics. For instance, they can be represented by their term frequency (tf) or inverse document frequency (idf). It can be then interpreted that terms with anomalous (very high or very low) statistics carry little value and can be discarded.
WORDVIEW enables you to identify such terms in an automatic fashion. The solution might seem complex behind the scene, as it firsts needs calculate certain statistics, gaussanize the distribution of the specified statistics (i.e. tf or ifd), and then identify the terms with anomalous values on the gaussanized distribution by looking at their z-score. However, the API is easy and convenient to use. The example below shows how you can use this API:

```python
from wordview.preprocessing import RedunWords

rw = RedunWords(imdb_train["text"], method='idf')
```

Let the program automatically identify a set of redundant words:

```python
red_words = rw.get_redundant_terms()
```


Alternatively, you can manually set cut-off threshold for the specified score, by setting the manual Flag to True and specifying lower and upper cut-off thresholds. 
```python
red_words = rw.get_redundant_terms(manual=True, manual_thresholds: dict={'lower_threshold':1, 'upper_threshold': 8})
```

In order to get a better understanding of the distribution of the scores before setting the thresholds, you can run `show_plot()` method from `RedunWords` class to see this distribution:

```python
rw.show_plot()
```

When red_words is ready, you can filter the corpus:

```python
# text must be a list of words
res = " ".join([t for t in text if t not in redundant_terms])
```

## **Text Cleaning**

*wordview* implements an easy to use and powerful function for cleaning up the text (`clean_text`). 
Using, `clean_text`, you can choose what pattern to accept via `keep_pattern` argument, 
what pattern to drop via `drop_patterns` argument, and what pattern to replace via `replace` argument. You can also specify the maximum length of tokens. 
Let's use [Stanford's IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) as an example. A sample of this data can be found in `resources/data/imdb_train_sample.tsv`.


```python
from wordview.preprocessing import clean_text

imdb_train = pd.read_csv('data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])

# Let's only keep alphanumeric tokens as well as important punctuation marks:
keep_pattern='^[a-zA-Z0-9!.,?\';:$/_-]+$'

# In this corpus, one can frequently see HTML tags such as `< br / >`. So let's drop them:
drop_patterns={'< br / >'}

# By skimming throw the text one can frequently see many patterns such as !!! or ???. Let's replace them:
replace={'!!!':'!', '\?\?\?':'?'}

# Finally, let's set the maximum length of a token to 15:
maxlen=15

# Pass the set keyword arguments to the apply:
imdb_train.text = imdb_train.text.apply(clean_text, args=(), keep_pattern=keep_pattern, replace=replace, maxlen=maxlen)
```

Note that `clean_text` returns tokenized text. 


# Contributions

```bash
# Create and activate a virtual env
python -m venv VENV
source VENV/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run app
python wordview/dashapp/index.py
```


## References
<a id="1">[1]</a> R. Csaky et al. - Improving Neural Conversational Models with Entropy-Based Data Filtering - In Proceedings of ACL 2019 - Florence, Italy.

<a id="2">[2]</a> Maxime Peyrard - A Simple Theoretical Model of Importance for Summarization - In Proceedings of ACL 2019 - Florence, Italy.

## **Auto Text Cleaning**
One of the first obstacles that any NLP practitioner faces is the tedious, demotivating, and confusing task of cleaning up the text. Unlike images that come in a perfect-for-ml vector format which is the same across the globe and input sources, text comes in all forms, formats, styles, languages, and structures. Before being able to get value out of it, any NLP expert has gone through the painful and tedious task of text cleaning. Even worse, there is no universal recipe for this. That is to say, should I lowercase this text? Shall I replace numbers with a place holder or should remove them altogether? How about stop words?... Each person can carry out a subset of the mentioned example steps, and arrive at a different cleaned text. How can we compare models then? A slight change in the often overlooked cleaning step can lead to inconsistency in our experiments. 