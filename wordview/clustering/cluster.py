from typing import Iterable, Union

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Cluster:
    def __init__(self, documents: Iterable, vector_model: str) -> None:
        self.documents = documents
        self.vector_model = vector_model
        self.vectors = self._vectorize()

    def _vectorize(self):
        if self.vector_model == "transformer":
            vectorizer = SentenceTransformer("all-MiniLM-L6-v2")
            vectors = vectorizer.encode(self.documents)
        elif self.vector_model == "tfidf":
            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform(self.documents)
        else:
            raise ValueError(f"Vector model {self.vector_model} is not supported.")
        return vectors

    def cluster(self, clustering_algorithm: str):
        if clustering_algorithm == "AgglomerativeClustering":
            clust_model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.5
            )
            clust_model.fit(self.vectors)
            clust_labels = clust_model.labels_
        else:
            raise ValueError(
                f"Clustering algorithm {clustering_algorithm} is not supported."
            )
        return clust_labels


if __name__ == "__main__":

    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    # Perform kmean clustering
    # clustering_model = AgglomerativeClustering(
    #     n_clusters=None, distance_threshold=1.5
    # )  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    # clustering_model.fit(corpus_embeddings.toarray())
    # cluster_assignment = clustering_model.labels_

    # clustered_sentences = {}
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     if cluster_id not in clustered_sentences:
    #         clustered_sentences[cluster_id] = []

    #     clustered_sentences[cluster_id].append(documents[sentence_id])

    # for i, cluster in clustered_sentences.items():
    #     print("Cluster ", i + 1)
    #     print(cluster)
    #     print("")

    # true_k = 2
    # model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    # model.fit(X)

    # print("Top terms per cluster:")
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(true_k):
    #     print(f"Cluster {i}")
    #     for ind in order_centroids[i, :10]:
    #         print(f"\tCluster {terms[ind]}")
    #     print
