Analysis & Extraction of Multiword Expressions (MWEs)
#####################################################

Multiword Expressions (MWEs) are phrases that can be treated as a single
semantic unit. E.g. *swimming pool* and *climate change*. MWEs have
application in different areas including: parsing, language generation,
language modeling, terminology extraction, and topic models.

Wordview can extract different types of MWEs from a text corpus in any of the supported languages. Wordview by default extracts the following types of MWEs:
Light Verb Constructions (LVCs), 2 and 3 word Noun Compounds (NCs), 2 and 3 word Adjective-Noun Compounds (ANCs), and Verb-Noun Compounds (VNCs).
However, you can specify other types of MWEs you want to extract using the `custom_pattern` argument. For more details, see the 
the documentation.

.. code:: python

   # First we need to extract ngrams from the corpus
   # If this was not done previously, e.g. when running other functions of Wordview, 
   # you can do it as follows:
   from wordview.preprocessing import NgramExtractor
   import pandas as pd
   imdb_corpus = pd.read_csv("data/IMDB_Dataset_sample.csv")
   extractor = NgramExtractor(imdb_corpus, "review")
   extractor.extract_ngrams()
   extractor.get_ngram_counts(ngram_count_file_path="data/ngram_counts.json")
   
   # Now we can extract MWEs
   from wordview.mwes import MWE
   import json
   mwe_obj = MWE(imdb_corpus, 'review',
                  ngram_count_file_path='data/ngram_counts.json',
                  language='EN', 
                  custom_patterns="NP: {<DT>?<JJ>*<NN>}",
                  only_custom_patterns=False,
                  )
    mwes = mwe_obj.extract_mwes(sort=True, top_n=10)
    json.dump(mwes, open('data/mwes.json', 'w'), indent=4)
    

The above returns the results in a dictionary, that in this example we stored in `mwes.json` file.
You can also return the result in a table:

.. code-block:: python

    mwe_obj.print_mwe_table()
    ╔═════════════════════════╦═══════════════╗
    ║ LVC                     ║   Association ║
    ╠═════════════════════════╬═══════════════╣
    ║ SHOOT the binding       ║         26.02 ║
    ║ achieve this elusive    ║         24.7  ║
    ║ manipulate the wildlife ║         24.44 ║
    ║ offset the darker       ║         24.02 ║
    ║ remove the bindings     ║         24.02 ║
    ║ Wish that Anthony       ║         23.9  ║
    ║ Add some French         ║         23.5  ║
    ║ grab a beer             ║         22.82 ║
    ║ steal the 42            ║         22.5  ║
    ║ invoke the spirit       ║         22.12 ║
    ╚═════════════════════════╩═══════════════╝
    ╔══════════════════════╦═══════════════╗
    ║ NC2                  ║   Association ║
    ╠══════════════════════╬═══════════════╣
    ║ gordon willis        ║         20.74 ║
    ║ Smoking Barrels      ║         20.74 ║
    ║ sadahiv amrapurkar   ║         20.74 ║
    ║ nihilism nothingness ║         20.74 ║
    ║ tomato sauce         ║         20.74 ║
    ║ Picket Fences        ║         20.74 ║
    ║ deja vu              ║         19.74 ║
    ║ cargo bay            ║         19.74 ║
    ║ zoo souvenir         ║         19.16 ║
    ║ cake frosting        ║         19.16 ║
    ╚══════════════════════╩═══════════════╝
    ╔══════════════════════════════╦═══════════════╗
    ║ ANC2                         ║   Association ║
    ╠══════════════════════════════╬═══════════════╣
    ║ bite-sized chunks            ║         20.74 ║
    ║ lizardly snouts              ║         20.74 ║
    ║ behind-the-scenes featurette ║         20.74 ║
    ║ hidebound conservatives      ║         20.74 ║
    ║ judicious pruning            ║         20.74 ║
    ║ substantial gauge            ║         19.74 ║
    ║ haggish airheads             ║         19.74 ║
    ║ global warming               ║         19.74 ║
    ║ Ukrainian flags              ║         19.16 ║
    ║ well-lit sights              ║         19.16 ║
    ╚══════════════════════════════╩═══════════════╝
    ╔═══════════════╦═══════════════╗
    ║ VPC           ║   Association ║
    ╠═══════════════╬═══════════════╣
    ║ upside down   ║         12.67 ║
    ║ Stay away     ║         12.49 ║
    ║ put together. ║         11.62 ║
    ║ sit through   ║         10.93 ║
    ║ ratchet up    ║         10.83 ║
    ║ shoot'em up   ║         10.83 ║
    ║ rip off       ║         10.72 ║
    ║ hunt down     ║         10.67 ║
    ║ screw up      ║         10.41 ║
    ║ scorch out    ║         10.4  ║
    ╚═══════════════╩═══════════════╝
    ╔══════════════╦═══════════════╗
    ║ NP           ║   Association ║
    ╠══════════════╬═══════════════╣
    ║ every penny  ║         12.78 ║
    ║ THE END      ║         12.07 ║
    ║ A JOKE       ║         11.79 ║
    ║ A LOT        ║         11.05 ║
    ║ Either way   ║         11.03 ║
    ║ An absolute  ║         10.72 ║
    ║ half hour    ║         10.65 ║
    ║ no qualms    ║         10.47 ║
    ║ every cliche ║         10.46 ║
    ║ another user ║         10.37 ║
    ╚══════════════╩═══════════════╝

Notice how many interesting entities are captured,
without the need for any labeled data and supervised model.
This can speed things up and save much costs in certain applications.


