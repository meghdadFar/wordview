
.. image:: https://img.shields.io/pypi/v/wordview
   :alt: PyPI - Version
.. image:: https://img.shields.io/pypi/dm/wordview
   :alt: PyPI - Downloads
.. image:: https://img.shields.io/librariesio/release/pypi/wordview
   :alt: Dependencies
.. image:: https://img.shields.io/pypi/l/wordview
   :alt: License

Wordview
########
Wordview is a Python package for Exploratory Data Analysis of text and provides
many statistics about your data in the form of plots, tables, and descriptions
allowing you to have both a high-level and detailed overview of your data.
It has functions to analyze explicit text elements such as words, n-grams, POS tags,
and multi-word expressions, as well as implicit elements such as clusters, anomalies, and biases.
Full documentation is available at `Wordview’s documentation page <https://meghdadfar.github.io/wordview/>`__.

.. image:: sphinx-docs/figs/cover.png
   :alt: Wordview Cover
   :width: 100%
   :align: center

Usage
*****

Install the package via ``pip``:

``pip install wordview``

The following sections present a high-level description of Wordview's features and functionalities.
For details, usage, tutorials, and worked examples see
the `documentation page <https://meghdadfar.github.io/wordview/>`__.

Text Analysis
*************
Using this feature, you can gain a comprehensive overview of your text data in terms of various statistics,
plots, and distributions. It enables a rapid understanding of the underlying patterns present in your dataset. 
You can see, for instance, what languages were used in your corpus, the average document lengths
(in terms of tokens), how many documents and words are in your corpus, various part-of-speech tags, and more.
You can also look at different distributions, plots, and word clouds to gain valuable insights into your text corpus.
Worldview uses Plotly interactive plots, with many intriguing features such as zooming,
panning, selection, hovering, and screenshots.

.. image:: sphinx-docs/figs/textanalysiscover.png
   :alt: Text Analysis Cover
   :width: 100%
   :align: center

Analysis of Labels
******************
In NLP, the proper analysis and understanding of labels within datasets can provide valuable insights for some of downstream tasks,
ensuring that models are trained on balanced and representative set of labels.
Wordview calculates an array of statistics tailored for labeled datasets. It provides a comprehensive overview of the distribution of labels,
the frequency of each label, and the distribution of labels across different categories.

.. image:: sphinx-docs/figs/labels_peach.png
   :width: 100%
   :align: center

Extraction & Analysis of Multiword Expressions
**********************************************
Multiword Expressions (MWEs) are phrases that can be treated as a single semantic unit, e.g., *swimming pool* and *climate change*. They can offer great insights into natural language data and have many practical applications, including machine translation, topic modeling, named entity recognition, terminology extraction, profanity detection, and more.
At the high level, we define MWEs as phrases whose components co-occur more than expected by chance and identify MWEs using precisely this property, which is modeled by statistical association measures such as PMI, and NPMI.
Wordview's MWE features is one of the most powerful, comprehensive, and easy-to-use tools that are available for the extraction of MWEs.

.. raw:: html

   <div style="text-align: center;">
       <img src="sphinx-docs/figs/mwescover.png" alt="MWWsImage" style="width: 70%; height: auto;">
   </div>

Bias Analysis
**************
In the rapidly evolving realm of Natural Language Processing (NLP), downstream models can be as fair and unbiased as the data on which they are trained. Wordview's bias analysis module is designed to help ensure that underlying training datasets are devoid of explicit negative biases related to categories such as gender, race, and religion.
By identifying and rectifying these biases, Wordview attempts to help with the creation of more inclusive, fair, and unbiased NLP applications.
Bias analysis is currently based on sentiment-analysis and a predefined set of categories, but we are working hard to extend it and make it better in many ways.

.. raw:: html

   <div style="text-align: center;">
       <img src="sphinx-docs/figs/biascover.png" alt="BiasImage">
   </div>


Analysis of Anomalies and Outliers
**********************************
Anomalies and outliers have wide applications in Machine Learning. While in
some cases, you can capture them and remove them from the data to improve the
performance of a downstream ML model, in other cases, they become the data points
of interest where we endeavor to find them in order to shed light into our data.
Wordview offers several anomaly and outlier detection functions.


Cluster Analysis
****************
Clustering can be used to identify different groups of documents with similar information, in an unsupervised fashion.
Despite it's ability to provide valuable insights into your data, you do not need labeled data for clustering.



Chat with Wordview
******************
Worldview integrates GPT-3.5-Turbo to enable the users to interact with their data and get insights from it via Natural Language.

.. raw:: html

   <div style="text-align: center;">
       <img src="sphinx-docs/figs/chat_stats.png" alt="ChatImage" style="width: 70%; height: auto;">
   </div>

.. raw:: html

   <div style="text-align: center;">
       <img src="sphinx-docs/figs/chat_mwe.png" alt="ChatImage" style="width: 70%; height: auto;">
   </div>

Utilities
*********
Wordview offers a number of utility functions that you can use for common pre and post processing tasks in NLP.

Contributing
############
We are just getting started with Wordview and are looking to make Wordview a go-to solution for anyone who loves NLP and knows and appreciates the actual value of data and data analysis. But that requires help from the community. So, we are looking forward to seeing you join Wordview as a collaborator.
You can visit the `contributing page <CONTRIBUTING.rst>`__ for detailed instructions about how you can contribute to Wordview.

