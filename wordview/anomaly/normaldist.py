from typing import Dict, Set

import pandas as pd
import plotly.express as px
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer

from wordview import gaussianize


class NormalDistAnomalies(object):
    def __init__(self, items: Dict, method="idf"):
        """Identify anomalies on a normal distribution.

        Args:
            items: A dictionary of items and their representative value, such as count, idf, etc.
                    The representative value can also be a vector.
            method: Method of creating the set of redundant words. Can be either of ['idf'].
        Returns:
            None
        """
        self.items = items

    def anomalous_items(
        self,
        manual: bool = False,
        z: int = 3,
        manual_thresholds: Dict = {"lower_threshold": -1, "upper_threshold": -1},
    ):
        """Identify anomalous items in `self.items`.

        Args:
            manual: Whether or not select redundant words using manual thresholds.
                    When set to True, manual_thresholds should be specified.
            z: Items with a z-score above this value are considered anomalous. Used only when manual is False.  Default is 3.
            l_idf: Lower cut-off threshold for  items (inclusive). Used only when manual is True.
            u_idf: Upper cut-off threshold for items (inclusive). Used only when manual is True.

        Returns:

        """
        if not manual:
            anomalous_set = self._anomalous_items_zscore(
                item_score_dict=self.items, z_value=z
            )
        else:
            anomalous_set = self._anomalous_items_manual(
                item_score_dict=self.items,
                lower_threshold=manual_thresholds["lower_threshold"],
                upper_threshold=manual_thresholds["upper_threshold"],
            )
        return anomalous_set

    def _anomalous_items_manual(
        self, item_score_dict, lower_threshold, upper_threshold
    ) -> Set[str]:
        """Identify anomalous items by looking at manual thresholds i.e.
            lower_threshold, upper_threshold.

        Args:
            item_score_dict: Dictionary of items to their score.
            lower_threshold: Lower cut-off threshold (inclusive).
            upper_threshold: Upper cut-off threshold (inclusive).

        Returns:
            Set of anomalous items.
        """
        sl = sorted(item_score_dict.items(), key=lambda kv: kv[1])
        smallest_idf = sl[0][1]
        largest_idf = sl[len(sl) - 1][1]
        if lower_threshold < smallest_idf:
            raise ValueError(
                f"Idf values are between {smallest_idf:.2f} and {largest_idf:.2f}. \
                    You have set the lower_idf to {lower_threshold:.2f}. Update the values accordingly, \
                        or consider switching the manual flag back to False."
            )
        if upper_threshold > largest_idf:
            raise ValueError(
                f"Idf values are between {smallest_idf:.2f} and {largest_idf:.2f}. \
                You have set the upper_idf to {upper_threshold:.2f}. Update the values accordingly, \
                or consider switching the manual flag back to False."
            )
        anomalous_items = set()
        for i in range(len(sl)):
            if sl[i][1] < lower_threshold:
                anomalous_items.add(sl[i][0])
            if sl[i][1] >= upper_threshold:
                anomalous_items.add(sl[i][0])
        return anomalous_items

    def _anomalous_items_zscore(self, item_score_dict, z_value):
        """Identify anomalous items by looking at 'z_value'.

        Args:
            item_score_dict: Dictionary of items to their score.
            z_value: Items that are away this many standard deviation
                    from mean are considered anomalous.

        Returns:
            Set of anomalous items.
        """
        item_score_df = pd.DataFrame(item_score_dict.items(), columns=["item", "score"])
        g = gaussianize.Gaussianize(strategy="brute")
        g.fit(item_score_df["item"])
        idf_guassian = g.transform(item_score_df["item"])
        item_score_df["score_gaussianized"] = idf_guassian
        z = zscore(item_score_df["score_gaussianized"])
        item_score_df["zscore"] = z
        anomalies_set = set()
        for i in range(len(item_score_df)):
            if (
                item_score_df.iloc[i]["zscore"] <= -z_value
                or item_score_df.iloc[i]["zscore"] >= z_value
            ):
                anomalies_set.add(item_score_df.iloc[i]["item"])
        return anomalies_set

    def show_plot(self) -> None:
        """Create a distribution plot for the scores to help manually
            identify cut-off thresholds.

        Args:
            scores:
        Returns:
            None
        """
        scores = [v for _, v in self.items.items()]
        score_name = "scores"
        scores_df = pd.DataFrame(scores, columns=[score_name])
        fig = px.histogram(
            scores_df, x=score_name, marginal="rug", color_discrete_sequence=["magenta"]
        )
        fig.show()
