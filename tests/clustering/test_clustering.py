import pytest
from wordview.clustering import Cluster


def test_tfidf_agglomerative_clusters(documents, clusters):
    cl = Cluster(documents=documents, vector_model="tfidf")
    cl.cluster(clustering_algorithm="AgglomerativeClustering")
    assert cl.clusters == clusters


def test_transformer_agglomerative_clusters(documents, clusters):
    cl = Cluster(documents=documents, vector_model="transformer")
    cl.cluster(clustering_algorithm="AgglomerativeClustering")
    assert cl.clusters == clusters


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


@pytest.fixture
def clusters():
    clusts = {0: ['The generation of random binary unordered trees',
                'The intersection graph of paths in trees',
                'Graph minors IV Widths of trees and well quasi ordering',
                'Graph minors A survey'],
            1: ['Human machine interface for lab abc computer applications',
                'The EPS user interface management system',
                'System and human system engineering testing of EPS'],
            2: ['A survey of user opinion of computer system response time',
                'Relation of user perceived response time to error measurement']
            }
    return clusts
