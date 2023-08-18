Cluster Analysis
################

Clustering can be used to identify different groups of documents with similar information, in an unsupervised fashion.
Despite it's ability to provide valuable insights into your data, you do not need labeled data for clustering. ``wordview`` provide 
a clustering API with options across vectorization models and clustering algorithms to ensure that the user has access to a wide
experiment space in order to find the most suitable model for their purpose. See the following example to learn how you can get started with
clustering in ``wordview``.

.. code:: python

    from wordview.clustering import Cluster
    docs = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey"]
    
    cl = Cluster(documents=docs, vector_model="transformer")
    cl.cluster(clustering_algorithm="kmeans", n_clusters=3)
    print(cl.clusters)



