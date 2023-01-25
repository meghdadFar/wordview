# wordview

[![PyPI version](https://badge.fury.io/py/wordview.svg?&kill_cache=1)](https://badge.fury.io/py/wordview)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)


WORDVIEW is a Python package primarily for text analysis. It, moreover, provides a number of unsupervised models for Information Extraction and Preprocessing. See section [Features](#Features) for more details.

WORDVIEW is open-source and free. We, however, developed a Dashboard version of WORDVIEW based on Plotly, for non-developers. See WORDVIEW-DASH page for more details.


# Features
- [Text Analysis](#text-analysis)
  - [Overview](#overview)
  - [Word Distributions](#distributions)
  - [Part of Speech Tags (POS)](#part-of-speech-tags)
  - [Topics]()
  - [Clusters]()
  - [Labels]()
- Information Extraction
  - [Multiword Expressions](#extraction-of-multiword-expressions)
  - [Statistically Redundant Words](#identification-of-statistically-redundant-words)
  - [Anomalies](#anomalies) (Planned)
- [Text Cleaning](#text-cleaning)
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
### Overview
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

### Word Distributions
You can look into different distributions using `show_distplot` method.
For instance, the distribution of document lengths:
```python
ta.show_distplot(plot='doc_len')
```
![doclen](/figs/doclen.png)

Or, the Zipf distribution of words:

```python
ta.show_distplot(plot='word_frequency_zipf')
```
![zipf](/figs/wordszipf.png)

See [this excellent article](https://medium.com/@_init_/using-zipfs-law-to-improve-neural-language-models-4c3d66e6d2f6) to learn how you can use Zipf’s law to Improve NLP models.

### Part of Speech Tags
You can moreover, see different part of speech tags in corresponding word clouds: 
```python
# To see verbs
ta.show_word_clouds(type="VB")
# To see nouns
ta.show_word_clouds(type="NN")
# To see adjectives
ta.show_word_clouds(type="JJ")
```
![verbs](/figs/verbs.png)
![nouns](/figs/nouns.png)
![adjs](/figs/adjectives.png)

### Labels
*wordview* provides basic statistics and analysis for labels in labeled datasets. To use this feature, you can use `LabelStatsPlots` which supports up to 4 independent labels that can be either categorical or numerical.

```python
from wordview.text_analysis import LabelStatsPlots

# In addition to the original label which is located in the column `label`, of the dataframe, 
# for illustration purpose, let's create two random labels:
imdb_train['numerical_label'] = np.random.randint(1, 500, imdb_train.shape[0])
imdb_train['new_label'] = random.choices(['a', 'b', 'c', 'd'], [0.2, 0.5, 0.8, 0.9], k=imdb_train.shape[0])
imdb_train['numerical_labe2'] = np.random.randint(1, 500, imdb_train.shape[0])

lsp = LabelStatsPlots(df=imdb_train, label_columns=[('label', 'categorical'),
                                                    ('label2', 'categorical'),
                                                    ('numerical_label', 'numerical'),
                                                    ('numerical_label2', 'numerical')
                                                   ])

lsp.show_label_plots()
```
![labels](/figs/labels.png)


## Multiword Expressions

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

See the following worked example, to see how you can access and use MWEs.

```python
from tabulate import tabulate
import json

with open('tmp/mwes.json') as json_file:
    mwes_dict = json.load(json_file)

nc_association = {k: v for k, v in mwes_dict['NC'].items()}
top_nc_association_table = [[k, v] for k,v in nc_association.items()][:10]
print(tabulate(top_nc_table, tablefmt="simple_grid"))

╔══════════════════╦═══════╗
║ busby berkeley   ║ 11.2  ║
║ burgess meredith ║ 11.13 ║
║ bruno mattei     ║ 10.92 ║
║ monty python     ║ 10.69 ║
║ ki aag           ║ 10.65 ║
║ denise richards  ║ 10.63 ║
║ guinea pig       ║ 10.52 ║
║ blade runner     ║ 10.48 ║
║ domino principle ║ 10.44 ║
║ quantum physics  ║ 10.38 ║
╚══════════════════╩═══════╝
```

Notice how the name of actors and shows such as `busby berkeley`, `burgess meredith`, and `monty python` as well other other multi-word concepts such as `quantum physics` and `guinea pig` are captured, without the need for any labeled data and supervised model which can add value by saving much costs and speed things up, in certain situations.


One common use of extracting MWEs is to treat them as a single unit. Research shows that when MWEs are treated as a single token, they performance of downstream applications such as classification and NER increases. Using `hyphenate_mwes` function, you can hyphenate the extracted MWEs in the corpus (global warming --> global-warming). This will force downstream tokenizers to treat them as a single token. Here is a worked example:

```python
from wordview.mwes import hyphenate_mwes
mwe_hyphenated_df = hyphenate_mwes(path_to_mwes='tmp/mwes.json', mwe_types=['NC', 'JNC'], df=imdb_train, text_column='text')
```

## Anomalies
Sometimes, anomalies find their way into the data and tamper with the quality of the downstream ML model. For instance, a classifier that is trained to classify input documents into N known classes, does not know what to do with an anomalous document, hence, it places it into one of those classes that can be completely wrong. Anomaly detection, in this example, allows us to identify and discard anomalies before running the classifier. On the other hand, sometimes anomalies the most interesting part of our data and those are the ones that we are looking for.

## **Statistically Redundant Words**

Redundant words carry little value and can exacerbate the results of many NLP tasks. To solve this issue, one common approach is to filter out a pre-defined set of words, called stop words. Stop words however, are usually universal static sets. They mostly vary only across languages. `wordview` provides a more dynamic solution that is specific to each individual data set, by employing purely statistical methods to identify redundant words with little value. The solution might seem complex behind the scene, as it first calculates certain statistics, gaussanize the distribution of those specified statistics (i.e. tf or ifd), and then identify the terms with anomalous values on the gaussanized distribution by looking at their z-score. However, the API is easy and convenient to use. The example below shows how you can use this API:

```python
from wordview.preprocessing import RedunTerms
rt = RedunTerms(imdb_train["text"], method='idf')

# Let the method automatically identify a set of redundant terms
redundant_terms = rt.get_redundant_terms()

#  Manually set cut-off threshold for the specified score
redundant_terms_manual = rt.get_redundant_terms(manual=True, manual_thresholds: dict={'lower_threshold':1, 'upper_threshold': 8})
```

When choosing the manual approach, to get a better understanding of the distribution of the scores before setting the thresholds, you can run `show_plot()` method from `RedunTerms` class to see this distribution:

```python
rt.show_plot()
```

## **Text Cleaning**

Cleaning up the text can be a tedious task, but for most NLP applications we almost always need some degree of text cleaning. *WORDVIEW* offers easy to use functionalities for cleaning up the text (`clean_text`). For instance, you can choose what pattern to accept via `keep_pattern` argument, what pattern to drop via `drop_patterns` argument, and what pattern to replace via `replace` argument. Or you can specify the max length of allowed tokens to filter out very long sequences that are often noise. See the docs to learn more about other parameters of `clean_text`. Here is a worked example:


```python
from wordview.preprocessing import clean_text

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

**Note** `clean_text` returns tokenized text. 


# Contributions

Thank you for contributing to wordview! We and the users of this repo appreciate your efforts! If spot a problem or you have a feature request or you wanted to suggest an improvement, please create an issue. Please first search the existing open and closed issues [here](https://github.com/meghdadFar/wordview/issues). If a related issue already exists, you can add your comment and avoid creating duplicate or very similar issues. If you come across an issue that you would like to work on, feel free to [open a PR](#pull-request-pr) for it.

## Branches
To begin contributing, clone the repository and make sure you are on `main` branch. Then create your own branch.

```bash
# Clone the repo
git clone git@github.com:meghdadFar/wordview.git

# Get the latest updates, if you have previously cloned wordview.
git pull

# Create a new branch
git checkout -b BRANCH_NAME
```

Please try to name your branch such that the name clarifies the purpose of your branch, to some extent. We commonly use hyphenated branch names. For instance, if you are developing an anomaly detection functionality based on a normal distribution, a good branch name can be `normal-dist-anomaly-detection`.

## Environment Setup

We use [`Poetry`](https://pypi.org/project/poetry/) to manage dependencies and packaging. Follow these steps to set up your dev environment:

```bash
python -m venv venv

source venv/bin/activate

pip install poetry

# Disable Poetry's environment creation, since we already have created one
poetry config virtualenvs.create false
```
Use Poetry to install dependencies:

```bash
poetry install
```
By default, dependencies across all non-optional groups are install. See [Poetry documentation](https://python-poetry.org/docs/managing-dependencies/) for more details and for instructions on how to define optional dependency groups.

## Quality Checks

To ensure a high quality in terms of readability, complying with PEP standards, and static type checking, we use `black`, `flake8`, `mypy` and `isort`. These tools are part of dev dependencies and hence they are installed when you [set up your dev environment](#environment-setup). To use them, change directory to project home where corresponding configuration files (`mypy.ini`, `.flake8`) live and then simply run them as follows.

```bash

black <PATH_TO_NEW/CHANGED_CODE>

mypy <PATH_TO_NEW/CHANGED_CODE>

flake8 <PATH_TO_NEW/CHANGED_CODE>

isort <PATH_TO_NEW/CHANGED_CODE>
```
Commit the changes and push to remote. We run all the above in GitHub checks. So if you don't take these steps, GitHub checks will fail preventing you from [merging your PR](#pull-request-pr).

## Testing
`wordview` primary testing is carried out via unittests. We use [Pytest](https://docs.pytest.org/). Please include your for any functionality that you provide inside the [test](./tests/) directory. See [this test module]() for a minimal example of unittesting with Pytest.


## Pull Request (PR)
Once your work is complete, you can make a pull request. Remember to link your pull request to an issue by using a supported keyword in the pull request's description or in a commit message. E.g. "closes #issue_number", "resolves #issue_number", or "fixes #issue_number". See [this page](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) for more details.

Once your PR is submitted, a maintainer will review your PR. They may ask questions or suggest changes either using [suggested changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments.

Once all the comments and changes are resolved, your PR will be merged. 🥳🥳

Thank you for your contribution! We are really excited to have your work integrated in wordview!
