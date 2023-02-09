from typing import Dict, Set

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import zscore

from wordview import gaussianize


class NormalDistAnomalies(object):
    def __init__(self, items: Dict, val_name: str = "representative_value"):
        """Identify anomalies on a normal distribution.

        Args:
            items: A dictionary of items and their representative value, such as word_count, idf, etc.
            val_name: Name of the value in the above dictionary. i.e. word_count, idf, etc. Defaults to `representative_value`.

        Returns:
            None
        """
        # self.items = items
        self.val_name = val_name
        self.item_value_df = pd.DataFrame(
            items.items(), columns=["item", self.val_name]
        )

    def anomalous_items(
        self,
        manual: bool = False,
        z: int = 3,
        manual_thresholds: Dict = {"lower_threshold": -1, "upper_threshold": -1},
    ) -> Set[str]:
        """Identify anomalous items in `self.items`.

        Args:
            manual: Whether or not select redundant words using manual thresholds.
                    When set to True, manual_thresholds should be specified.
            z: Items with a z-score above this value are considered anomalous. Used only when manual is False.  Default is 3.
            l_idf: Lower cut-off threshold for  items (inclusive). Used only when manual is True.
            u_idf: Upper cut-off threshold for items (inclusive). Used only when manual is True.

        Returns:
            Set of anomalous items.
        """
        if not manual:
            anomalous_set = self._anomalous_items_zscore(z_value=z)
        else:
            anomalous_set = self._anomalous_items_manual(
                lower_threshold=manual_thresholds["lower_threshold"],
                upper_threshold=manual_thresholds["upper_threshold"],
            )
        return anomalous_set

    def _anomalous_items_manual(
        self, lower_threshold: int, upper_threshold: int
    ) -> Set[str]:
        """Identify anomalous items by looking at manual thresholds i.e.
            lower_threshold, upper_threshold.

        Args:
            lower_threshold: Lower cut-off threshold (not inclusive).
            upper_threshold: Upper cut-off threshold (inclusive).

        Returns:
            Set of anomalous items.
        """
        sl = self.item_value_df.sort_values(by=[self.val_name])[self.val_name].to_list()
        smallest_value = sl[0]
        largest_value = sl[-1]
        if lower_threshold < smallest_value:
            raise ValueError(
                f"Idf values are between {smallest_value:.2f} and {largest_value:.2f}. \
                    You have set the lower_idf to {lower_threshold:.2f}. Update the values accordingly, \
                        or consider switching the manual flag back to False."
            )
        if upper_threshold > largest_value:
            raise ValueError(
                f"Idf values are between {smallest_value:.2f} and {largest_value:.2f}. \
                You have set the upper_idf to {upper_threshold:.2f}. Update the values accordingly, \
                or consider switching the manual flag back to False."
            )

        anomalous_items = set(
            self.item_value_df[
                self.item_value_df[self.val_name] < lower_threshold
                or self.item_value_df[self.val_name] > upper_threshold
            ]["item"].to_list()
        )
        return anomalous_items

    def _anomalous_items_zscore(self, z_value: int) -> Set[str]:
        """Identify anomalous items by looking at 'z_value'.

        Args:
            z_value: Items that are away this many standard deviation
                    from mean are considered anomalous.

        Returns:
            Set of anomalous items.
        """
        g = gaussianize.Gaussianize(strategy="brute")
        g.fit(self.item_value_df["item"])
        guassian_score = g.transform(self.item_value_df["item"])
        self.item_value_df["guassian_score"] = guassian_score
        z = zscore(self.item_value_df["guassian_score"])
        self.item_value_df["zscore"] = z
        anomalies_set = set()
        for i in range(len(self.item_value_df)):
            if (
                self.item_value_df.iloc[i]["zscore"] <= -z_value
                or self.item_value_df.iloc[i]["zscore"] >= z_value
            ):
                anomalies_set.add(self.item_value_df.iloc[i]["item"])
        return anomalies_set

    def show_plot(self, type: str = "default") -> None:
        """Create a distribution plot for the representative value to help manually
            identify cut-off thresholds.

        Args:
            type: Type of the data that is plotted. It can be `default` or `normal`. Defaults to `default`.

        Returns:
            None
        """
        if type == "default":
            x_value = self.val_name
        elif type == "normal":
            x_value = "guassian_score"

        fig = px.histogram(
            self.item_value_df,
            x=x_value,
            marginal="rug",
            color_discrete_sequence=["teal"],
        )
        fig.show()
