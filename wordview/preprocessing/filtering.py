import pandas as pd
import plotly.express as px
from scipy.stats import zscore
from wordview import logger
from wordview import gaussianize
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Set, Dict


class RedunWords(object):
    def __init__(self, documents, method="idf"):
        """"Class to identify and represent a set of redundant words.

        Args:
            documents: An iterable which yields str.
            method: Method of creating the set of redundant words. Can be either of ['idf'].

        Returns:
            None
        """
        self.documents = documents
        self.method = method
    
    def _get_scores(self) -> Dict:
        """Helper method to generate statistical scores using the technique specified by self.method arg. 

        Args:
            None

        Returns:
            token_score_dict: Dictionary of tokens to their score
        """
        token_score_dict = {}
        if self.method == 'idf':
            vectorizer = TfidfVectorizer(min_df=1)
            X = vectorizer.fit_transform(self.documents)
            idf = vectorizer.idf_
            token_score_dict = dict(zip(vectorizer.get_feature_names(), idf))
        else:
            raise ValueError(f'Currently, the only available method is idf but you have specified {self.method}')
        return token_score_dict

    def get_redundant_terms(self, z=3, manual=False, manual_thresholds: dict={'lower_threshold':-1, 'upper_threshold': -1}):
        """Create a filter set by identifying words with anomalous statistics.

        Args:
            documents: An iterable which yields either str, unicode or file objects.
            manual: Whether or not select redundant words using manual thresholds.
            z: Words with a Z-score above this value are considered redundant. Default is 3.
            l_idf: Lower cut-off threshold for IDF (inclusive). Used only when 
            u_idf: Upper cut-off threshold for IDF (inclusive).

        Returns:
            redundant_word_set (set): Set of redundant words.
        """
        token_score_dict = self._get_scores()
        if not manual:
            redundant_word_set = self._redundant_terms_zscore(token_score_dict=token_score_dict, z_value=z)
        else:
            redundant_word_set = self._redundant_terms(token_score_dict=token_score_dict,
                                            lower_threshold=manual_thresholds['lower_threshold'],
                                            upper_threshold=manual_thresholds['upper_threshold'])
        return redundant_word_set

    def _redundant_terms(self, token_score_dict, lower_threshold, upper_threshold) -> Set[str]:
        """Identify redundant terms by looking at manual thresholds i.e. lower_threshold, upper_threshold. 
        
        Args:
            token_metric_dict: Dictionary of tokens to their metric.
            lower_threshold: Lower cut-off threshold (inclusive).
            upper_threshold: Upper cut-off threshold (inclusive).

        Returns:
            redundant_terms: Set of redundant terms.
        """
        sl = sorted(token_score_dict.items(), key=lambda kv: kv[1])
        smallest_idf = sl[0][1]
        largest_idf = sl[len(sl) - 1][1]
        if lower_threshold < smallest_idf:
            raise ValueError(
                f'Idf values are between {smallest_idf:.2f} and {largest_idf:.2f}. You have set the lower_idf to {lower_threshold:.2f}. Update the values accordingly, \
                or consider switching the manual flag back to False.'
            )
        if upper_threshold > largest_idf:
            raise ValueError(
                f'Idf values are between {smallest_idf:.2f} and {largest_idf:.2f}. You have set the upper_idf to {upper_threshold:.2f}. Update the values accordingly, \
                or consider switching the manual flag back to False.'
            )
        redundant_terms = set()
        for i in range(len(sl)):
            if sl[i][1] < lower_threshold:
                redundant_terms.add(sl[i][0])
            if sl[i][1] >= upper_threshold:
                redundant_terms.add(sl[i][0])
        return redundant_terms

    def _redundant_terms_zscore(self, token_score_dict, z_value):
        """Identify redundant terms by looking at the provided Z-score.

        Args:
            token_metric_dict: Dictionary of tokens to their metric.
            z_value: Words with a Z-score above this value are considered redundant.

        Returns:
            redundant_terms: Set of redundant terms.
        """
        idf_df = pd.DataFrame(token_score_dict.items(), columns=["token", "idf"])
        # Gaussianize idf
        g = gaussianize.Gaussianize(strategy="brute")
        g.fit(idf_df["idf"])
        idf_guassian = g.transform(idf_df["idf"])
        idf_df["idf_gaussianized"] = idf_guassian
        # Calculate z score
        z = zscore(idf_df["idf_gaussianized"])
        idf_df["zscore"] = z
        redundant_terms = set()
        for i in range(len(idf_df)):
            if idf_df.iloc[i]["zscore"] <= -z_value or idf_df.iloc[i]["zscore"] >= z_value:
                redundant_terms.add(idf_df.iloc[i]["token"])
        return redundant_terms

    def show_plot(self) -> None:
        """ Create a histogram for the scores to help identify the cut-off threshold.

        Args:
            scores: 
        Returns:
            None
        """
        token_score_dict = self._get_scores()
        scores = [v for k,v in token_score_dict.items()]
        # l = len(scores)
        # print(l)
        # print(scores[l-100:l])
        score_name = f'{self.method} scores'
        scores_df = pd.DataFrame(scores, columns=[score_name])
        fig = px.histogram(scores_df, x=score_name, marginal='rug', color_discrete_sequence=['magenta'])
        fig.show()


if __name__ == "__main__":
    imdb_train = pd.read_csv('resources/data/imdb_train_sample.tsv', sep='\t', names=['label', 'text'])
    rw = RedunWords(imdb_train["text"])
    rw.show_plot()
    red_words = rw.get_redundant_terms()
    print(red_words)


# def filter_text(text, filter_set):
#     """
#     Args:
#         text (list): tokenized text
#         filterset (set): Set of filter words
#     """
#     if not isinstance(text, list):
#         raise TypeError("Input must be a list.")
#     if len(text) == 0:
#         raise ValueError("Input must be a non empty list.")
#     res = " ".join([t for t in text if t not in filter_set])
#     return res



# def save_filterset_tofile(filter_set, path):
#     with open(path, "w") as f:
#         for word in filter_set:
#             f.write(word + "\n")
#     f.close()


# def create_filterset_map(p2_raw_train_df, output_dir, zs=[3]):
#     original_df = pd.read_csv(p2_raw_train_df, sep="\t", names=["label", "text"])
#     wf = RedunWords()
#     z_map = {}
#     for z in zs:
#         filter_words = wf.get_redundant_terms(original_df.text, method="automatic", z=z)
#         save_filterset_tofile(filter_words, os.path.join(output_dir, "z" + str(z) + "_filterset.txt"))
#         z_map[z] = filter_words
#     return z_map


# def create_filterset(p2_raw_train_df, output_dir):
#     original_df = pd.read_csv(p2_raw_train_df, sep="\t", names=["label", "text"])
#     wf = RedunWords()
#     filter_words = wf.get_redundant_terms(original_df.text, method="manual", z=None)
#     save_filterset_tofile(filter_words, os.path.join(output_dir, "manual_filterset.txt"))
#     return filter_words
