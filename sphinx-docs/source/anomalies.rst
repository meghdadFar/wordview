Anomalies & Outliers
####################

Sometimes, anomalies find their way into the data and tamper with the
quality of the downstream ML model. For instance, a classifier that is
trained to classify input documents into N known classes, does not know
what to do with an anomalous document, hence, it places it into one of
those classes that can be completely wrong. Anomaly detection, in this
example, allows us to identify and discard anomalies before running the
classifier. On the other hand, sometimes anomalies the most interesting
part of our data and those are the ones that we are looking for.
You can use ``wordview`` to identify anomalies in your data. For instance,
you can use ``NormalDistAnomalies`` to identify anomalies based on (the normalized)
distribution of your data. See a worked example below. 

.. code:: python

   from wordview.anomaly import NormalDistAnomalies
   from sklearn.feature_extraction.text import TfidfVectorizer
   
   # Create a score for words.
   # It can be e.g. word frequency 
   tsp = TextStatsPlots(df=imdb_train, text_column='text')
   token_score_dict = tsp.analysis.token_to_count_dict
   # or it can be the inverse document frequency (IDF)
   vectorizer = TfidfVectorizer(min_df=1)
   X = vectorizer.fit_transform(imdb_train["text"])
   idf = vectorizer.idf_
   token_score_dict = dict(zip(vectorizer.get_feature_names(), idf))
   
   # Use NormalDistAnomalies to identify anomalies.
   nda = NormalDistAnomalies(items=token_score_dict)
   nda.anomalous_items()