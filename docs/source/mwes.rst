Multiword Expressions
#####################

Multiword Expressions (MWEs) are phrases that can be treated as a single
semantic unit. E.g. *swimming pool* and *climate change*. MWEs have
application in different areas including: parsing, language models,
language generation, terminology extraction, and topic models.
``wordview`` can extract different types of MWEs in your text.

.. code:: python

   from wordview.mwes import MWE

   # NC: NOUN-NOUN MWEs e.g. climate change
   # JNC: ADJECTIVE-NOUN MWEs e.g. big shot
   mwe = MWE(df=imdb_train, mwe_types=["NC", "JNC"], text_column='text')

   # build_counts method --that creates word occurrence counts, is time consuming.
   # Hence, you can run it once and store the counts, by the setting the
   # counts_filename argument.
   mwe.build_counts(counts_filename='tmp/counts.json')

   # Once the counts are created, extraction of MWEs is fast and can be carried out
   # with different parameters.
   # If the optional mwes_filename parameter is set, the extracted MWEs
   # will be stored in the corresponding file.
   mwes_dict = mwe.extract_mwes(counts_filename='tmp/counts.json')
   mwes_nc = {k: v for k, v in mwes_dict['NC'].items()}
   top_mwes_nc = [[k, v] for k,v in mwes_nc.items()][:10]
   print(tabulate(top_mwes_nc, tablefmt="double_outline"))

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

Notice how show and actor names such as ``busby berkeley``,
``burgess meredith``, and ``monty python`` as well other multi-word
concepts such as ``quantum physics`` and ``guinea pig`` are captured,
without the need for any labeled data and supervised model. This can
speed things up and save much costs in certain situations.


