import pytest
from wordview.clustering import Cluster


def test_tfidf_agglomerative_good_cluster_sizes(documents):
    cl = Cluster(documents=documents, vector_model="tfidf")
    good_cluster_sizes = [1, 5]
    for n in good_cluster_sizes:
        cl.cluster(clustering_algorithm="AgglomerativeClustering", n_clusters=n)
        assert len(cl.clusters) == n


def test_tfidf_agglomerative_zero_cluster_sizes(documents):
    cl = Cluster(documents=documents, vector_model="tfidf")
    with pytest.raises(ValueError):
        cl.cluster(clustering_algorithm="AgglomerativeClustering", n_clusters=0)


def test_tfidf_kmeans_good_cluster_sizes(documents):
    cl = Cluster(documents=documents, vector_model="tfidf")
    good_cluster_sizes = [1, 5]
    for n in good_cluster_sizes:
        cl.cluster(clustering_algorithm="kmeans", n_clusters=n)
        assert len(cl.clusters) == n


def test_tfidf_kmeans_zero_cluster_sizes(documents):
    cl = Cluster(documents=documents, vector_model="tfidf")
    # sklearn.cluster.KMeans does not raise value error but an OverflowError for cluster size of 0.
    with pytest.raises(Exception):
        cl.cluster(clustering_algorithm="kmeans", n_clusters=0)


@pytest.fixture
def documents():
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
    return docs
