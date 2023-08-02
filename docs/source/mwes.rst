Multiword Expressions (MWEs)
############################

Multiword Expressions (MWEs) are phrases that can be treated as a single
semantic unit. E.g. *swimming pool* and *climate change*. MWEs have
application in different areas including: parsing, language generation,
language modeling, terminology extraction, and topic models.

Wordview can extract different types of MWEs from a text corpus in any of the supported languages. Wordview by default extracts the following types of MWEs:
Light Verb Constructions (LVCs), 2 and 3 word Noun Compounds (NCs), 2 and 3 word Adjective-Noun Compounds (ANCs), and Verb-Noun Compounds (VNCs).
However, you can specify other types of MWEs you want to extract using the `custom_pattern` argument. For more details, see the 
the documentation.

.. code:: python

   from wordview.mwes import MWE
   import pandas as pd
   import json

   imdb_corpus = pd.read_csv('data/IMDB_Dataset_sample.csv').sample(500)
   mwe_obj = MWE(imdb_corpus, 'review',
                  ngram_count_file_path='data/ngram_counts.json',
                  language='EN', 
                  custom_patterns="NP: {<DT>?<JJ>*<NN>}",
                  only_custom_patterns=False,
                  )
    mwes = mwe_obj.extract_mwes(sort=True, top_n=10)
    json.dump(mwes, open('data/mwes.json', 'w'), indent=4)
    

The above returns the results in a dictionary, that in this example we stored in `mwes.json` file.
You can also return the result in a nice tabulate table for a better overview:

..  code::python

mwe_obj.print_mwe_table()

╔═════════════════════════╦═══════════════╗
║ LVC                     ║   Association ║
╠═════════════════════════╬═══════════════╣
║ SHOOT the binding       ║       26.0248 ║
║ achieve this elusive    ║       24.7009 ║
║ manipulate the wildlife ║       24.4398 ║
║ offset the darker       ║       24.0248 ║
║ remove the bindings     ║       24.0248 ║
║ Wish that Anthony       ║       23.8985 ║
║ Add some French         ║       23.5016 ║
║ grab a beer             ║       22.8247 ║
║ steal the 42            ║       22.5012 ║
║ invoke the spirit       ║       22.1179 ║
╚═════════════════════════╩═══════════════╝

╔══════════════════════╦═══════════════╗
║ NC2                  ║   Association ║
╠══════════════════════╬═══════════════╣
║ tomato sauce         ║        20.74  ║
║ sadahiv amrapurkar   ║        20.74  ║
║ nihilism nothingness ║        20.74  ║
║ Picket Fences        ║        20.74  ║
║ gordon willis        ║        20.74  ║
║ Smoking Barrels      ║        20.74  ║
║ cargo bay            ║        19.74  ║
║ deja vu              ║        19.74  ║
║ cake frosting        ║        19.155 ║
║ zoo souvenir         ║        19.155 ║
╚══════════════════════╩═══════════════╝

╔══════════════════════════════╦═══════════════╗
║ ANC2                         ║   Association ║
╠══════════════════════════════╬═══════════════╣
║ lizardly snouts              ║        20.74  ║
║ bite-sized chunks            ║        20.74  ║
║ behind-the-scenes featurette ║        20.74  ║
║ hidebound conservatives      ║        20.74  ║
║ judicious pruning            ║        20.74  ║
║ haggish airheads             ║        19.74  ║
║ global warming               ║        19.74  ║
║ substantial gauge            ║        19.74  ║
║ unfinished château           ║        19.155 ║
║ Ukrainian flags              ║        19.155 ║
╚══════════════════════════════╩═══════════════╝

╔═══════════════╦═══════════════╗
║ VPC           ║   Association ║
╠═══════════════╬═══════════════╣
║ upside down   ║       12.6739 ║
║ Stay away     ║       12.4897 ║
║ put together. ║       11.6159 ║
║ sit through   ║       10.9329 ║
║ ratchet up    ║       10.8286 ║
║ shoot'em up   ║       10.8286 ║
║ rip off       ║       10.7192 ║
║ hunt down     ║       10.6739 ║
║ screw up      ║       10.4136 ║
║ carve out     ║       10.4035 ║
╚═══════════════╩═══════════════╝

╔══════════════╦═══════════════╗
║ NP           ║   Association ║
╠══════════════╬═══════════════╣
║ every penny  ║       12.78   ║
║ THE END      ║       12.0676 ║
║ A JOKE       ║       11.7858 ║
║ A LOT        ║       11.0488 ║
║ Either way   ║       11.0335 ║
║ An absolute  ║       10.7176 ║
║ half hour    ║       10.6477 ║
║ no qualms    ║       10.4685 ║
║ every cliche ║       10.4581 ║
║ another user ║       10.3682 ║
╚══════════════╩═══════════════╝


Notice meany interesting entities are captured,
without the need for any labeled data and supervised model. This can
speed things up and save much costs in certain applications.


