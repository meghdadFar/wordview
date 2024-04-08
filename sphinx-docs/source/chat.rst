Chat with Wordview
##################

Worldview integrates GPT-3.5-Turbo to enable the users to interact with their data and get insights from it via Natural Language.
See examples of chatting with different components of Wordview below.


Chat with TextStatsPlots
~~~~~~~~~~~~~~~~~~~~~~~~

After allowing Wordview to process and analyze your corpus, you can 
call the `chat` method to interact with the data and get insights from it via Natural Language.

.. code:: python

   import json

   import pandas as pd

   from wordview.text_analysis import TextStatsPlots
   imdb_df = pd.read_csv("data/IMDB_Dataset_sample_5k.csv")
   with open("your_secrets_dir/openai_api_key.json", "r") as f:
      credentials = json.load(f)

   tsp = TextStatsPlots(df=imdb_df, text_column="review")
   tsp.chat(api_key=credentials.get("openai_api_key"))

The chat UI is available under http://127.0.0.1:5000/


Chat with MWEs
~~~~~~~~~~~~~~

After allowing Wordview to extract MWEs, you can call the `chat` method to get insights from this extraction through Natural Language.

.. code:: python

   import json

   import pandas as pd

   from wordview.mwe_extraction import MWEs
   from wordview.preprocessing import NgramExtractor

   imdb_df = pd.read_csv("data/IMDB_Dataset_sample_5k.csv")
   with open("your_secrets_dir/openai_api_key.json", "r") as f:
      credentials = json.load(f)

   extractor = NgramExtractor(imdb_df, "review")
   extractor.extract_ngrams()
   extractor.get_ngram_counts(ngram_count_file_path="ngram_counts.json")

   mwe_obj = MWE(imdb_df, 'review',
                ngram_count_file_path='ngram_counts.json',
                language='EN',
                custom_patterns="NP: {<DT>?<JJ>*<NN>}",
                only_custom_patterns=False,
                )
    mwe_obj.extract_mwes(sort=True, top_n=10)
    mwe_obj.chat(api_key=credentials.get("openai_api_key"))

The chat UI for MWEs is available under http://127.0.0.1:5001/


|chat|

.. |chat| image:: ../figs/chat.png