Version 1.2.0
-------------
- Support for all Penn POS tags
- Bar plots for POS tags (in addition to wordclouds)
- Remove deprecated fasttext model.


Version 1.1.2
-------------
- Automatic check and download of NLTK missing resources. 
- Rm CI step for downloading NLTK resources.
- Facilitate configuration of plots for Text & Label Analysis plots, by creating new and more clear arguments.


Version 1.1.1
-------------
- Fix minor bugs in bias analysis.
- Improve fonts and minor details in bias analysis plots.


Version 1.1.0
-------------
- Add bias detection and analysis feature (based on sentiment analysis)
- Include 3 bias categories: race, religion, and gender.
- Include an initial set of key terms for each bias category.
- Add function to visualize bias analysis in a plot.
- Add function to visualize bias analysis in a tables.


Version 1.0.0
-------------
- Complete refactoring and upgrading of the MWE module.
- Support for extracting variable length MWEs given a custom user syntactic patterns of POS tags.
- Predefined patterns for extracting Light Verb Constructions (LVCs), 2-3 word Noun Compounds, 2-3 Adjective Noun Compounds, and 2-3 Verb Noun Compounds, and Verb Particle Constructions (VPCs).
- Refactoring of the Association Measure module.
- Move DataFrame reader to a separate preprocessing module so that it can support all modules easier.
- Add support for extracting ngrams for MWE and also ngram analysis.


Version 0.4.2
-------------
- Better encapsulation.
- Overall improvement and fix of several inconsistencies in docstring.
- Allow quite a few plot configurations via kwargs.
- Rm old code from the demo notebook.
- Change cover.
- Optimize figure creation.

Version 0.4.1
-------------
- Update precommit hooks mypy and black versions

Version 0.4.0
-------------
- Support for extracting variable length MWE given a user pattern of POS tags.


Version 0.3.7
-------------
- Change newline encoding.
- To support multiline in GitHub release body.


Version 0.3.6
-------------
- Test description.
- Test description2.
- Test description3.
- Test description4.

Version 0.3.5
-------------
- Test description.
- Test description2.

Version 0.3.4
-------------
- Update awk sed parser to correctly read release body. 

Version 0.3.3
-------------
- Fix missing multiline description in GitHub release using printf.

Version 0.3.2
-------------
- Fix missing multiline description in GitHub release.

Version 0.3.1
-------------
- Add action for CD.
- Publish to PyPi and GitHub Releases on bump version.
- Improve the CD workflow to ensure checks are passed and the merge is successful.
- Check for release notes, otherwise do not publish. 

Version 0.2.4
-------------
- Improve MWE functionalities.
- Fix fasttext issues.
- Remove support for Python 11 (for now).

Version 0.2.3
-------------
- Make POS wordclouds configurable.
- Remove pos wordclouds and distplots as fields, and allow access to them via function call, for an improved data encapsulation.

Version 0.2.2
-------------
- Upgrade wordcloud version to latest to avoid build failure.


Version 0.2.1
-------------
- Upgrade pandas and scikit-learn versions

Version 0.2.0
-------------

- Major refactoring with a semi-stable features (see below) and their documentations.
- Exploratory Data Analysis.
- Doc level Label Analysis.
- Clustering.
- Preprocessing functions.
- Partial MWEs.
- Tets.


Version 0.1.0
-------------

- Initial release with major Exploratory Data Analysis, MWEs, and Preprocessing features.
- Initial documentations.
