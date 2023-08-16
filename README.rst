
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
Wordview is a Python package for Exploratory Data Analysis of text and provides many statistics about your data in the form of plots, tables, and descriptions allowing you to have both a high-level and detailed overview of your data.
It has functions to analyze explicit text elements such as words, n-grams, POS tags, and multi-word expressions, as well as implicit elements such as clusters, anomalies, and biases.
Wordview's Python API is open-source and available under the `MIT license <https://en.wikipedia.org/wiki/MIT_License>`__.

|cover|

Usage
*****

Install the package via ``pip``:

``pip install wordview``

To explore various features and functionalities, consult the documentation pages. The following sections
present a high-level description of Wordview's features and functionalities. For details, tutorials and worked examples, corresponding 
documentation pages are linked in each section.


Text Analysis
*************
Using this feature, you can gain a comprehensive overview of your text data in terms of various statistics, plots, and distributions.
It enables a rapid understanding of the underlying patterns present in your dataset.
By visually representing the data's nuances, this feature can aid in making informed decisions for downstream applications.
It's a step forward in ensuring that you have a grasp on the intricacies of your data before delving deeper into more complex tasks.
See `text analysis documentation pages <./docs/source/textstats.rst>`__  for usage and examples.

.. image:: docs/figs/textanalysiscover.png
   :alt: Text Analysis Cover
   :width: 100%
   :align: center

Analysis of Labels
******************
In the realm of Natural Language Processing (NLP), the proper analysis and understanding of labels within datasets can provide valuable insights, ensuring that models are trained on balanced and representative data.
Recognizing this, Wordview is engineered to compute an array of statistics tailored for labeled datasets.
These statistics cater to both document and sequence levels, providing a holistic view of the dataset's structure.
By diving deep into the intricacies of the labels, Wordview offers an enriched perspective, helping researchers and practitioners identify
potential biases, discrepancies, or areas of interest,
which are essential for creating robust and effective models.
See `label analysis documentation pages <./docs/source/labels.rst>`__ for usage and examples.

.. image:: docs/figs/labels_peach.png
   :width: 100%
   :align: center

Extraction & Analysis of Multiword Expressions
**********************************************
Multiword Expressions (MWEs) are phrases that can be treated as a single
semantic unit. E.g. *swimming pool* and *climate change*. MWEs have
application in different areas including: parsing, language models,
language generation, terminology extraction, and topic models. Wordview can extract different types of MWEs from text.
See `MWEs documentation page <./docs/source/mwes.rst>`__ for usage and examples.

.. raw:: html

   <div style="text-align: center;">
       <img src="docs/figs/mwescover.png" alt="MWWsImage" style="width: 70%; height: auto;">
   </div>

Bias Analysis
**************
In the rapidly evolving realm of Natural Language Processing (NLP), downstream models are as unbiased and fair as the data on which they are trained.
Wordview Bias Analysis module is designed to assist in the rigorous task of ensuring that underlying training datasets are devoid of explicit negative biases related to categories such as gender, race, and religion.
By identifying and rectifying these biases, Wordview attempts to pave the way for the creation of more inclusive, fair, and unbiased NLP applications, leading to better user experiences and more equitable technology.
See the `bias analysis documentation page <./docs/source/bias.rst>`__ for usage and examples.

.. raw:: html

   <div style="text-align: center;">
       <img src="docs/figs/biascover.png" alt="BiasImage">
   </div>


Analysis of Anomalies and Outliers
**********************************
Anomalies and outliers have wide applications in Machine Learning. While in
some cases, you can capture them and remove them from the data to improve the
performance of a downstream ML model, in other cases, they become the data points
of interest where we endeavor to find them in order to shed light into our data.

Wordview offers several anomaly and outlier detection functions.
See `anomalies documentation page <./docs/source/anomalies.rst>`__ for usage and examples.


Cluster Analysis
****************
Clustering can be used to identify different groups of documents with similar information, in an unsupervised fashion.
Despite it's ability to provide valuable insights into your data, you do not need labeled data for clustering. See
`wordview`'s `clustering documentation page <./docs/source/clustering.rst>`__ for usage and examples.


Utilities
#########

Wordview offers a number of utility functions that you can use for common pre and post processing tasks in NLP. 
See `utilities documentation page <./docs/source/utilities.rst>`__ for usage and examples.

Contributing
############

Thank you for contributing to wordview! We and the users of this repo
appreciate your efforts! You can visit the `contributing page <CONTRIBUTING.rst>`__ for detailed instructions about how you can contribute to Wordview.

.. |cover| image:: docs/figs/cover.png
