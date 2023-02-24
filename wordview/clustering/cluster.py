from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


class Cluster:
    def __init__(self, documents: List[str], vector_model: str) -> None:
        self.documents = documents
        self.vector_model = vector_model
        self.vectors = self._vectorize()
        self.clusters: Dict[str, List] = {}

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
            clust_model.fit(self.vectors.toarray())
            clust_labels = clust_model.labels_
        else:
            raise ValueError(
                f"Clustering algorithm {clustering_algorithm} is not supported."
            )
        clusters: Dict[str, List] = {k: [] for k in clust_labels}
        for i in range(len(self.documents)):
            clusters[clust_labels[i]].append(self.documents[i])
        self.clusters = clusters
