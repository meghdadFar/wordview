Text Cleaning
#############

Cleaning up the text can be a tedious task, but for
most NLP applications we almost always need some degree of it.
*wordview* offers easy to use functionalities for filtering noise,
customized definition of noise, and cleaning up the text from it. For
instance, you can choose what pattern to accept via ``keep_pattern``
argument, what pattern to drop via ``drop_patterns`` argument, and what
pattern to replace via ``replace`` argument. Or you can specify the max
length of allowed tokens to filter out very long sequences that are
often noise. See the docs to learn more about other parameters of
``clean_text``. Here is a worked example:

.. code:: python

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

**Note** ``clean_text`` returns tokenized text.


Hyphenating MWEs
################

An important use of extracting MWEs is to treat them as a single token. 
Research shows that when fixed expressions are treated as a single token rather than the sum of their components, 
they can improve the performance of downstream applications such as classification and NER. 
Using the `hyphenate_mwes` function, you can replace the extracted expressions in the corpus 
with their hyphenated version (global warming --> global-warming) so that they are considered 
a single token by downstream applications. A worked example can be seen below:

.. code:: python

    from snlp.mwes import hyphenate_mwes
    new_df = hyphenate_mwes(path_to_mwes='tmp/mwes/mwe_data.json',
                            mwe_types=['NC', 'JNC'],
                            df=imdb_train,
                            text_column='text')
    new_df.to_csv('tmp/new_df.csv', sep='\t')

