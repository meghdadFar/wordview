from typing import Dict, Iterable, Set

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import norm, zscore

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
        prob: float = 0.001,
    ) -> Set[str]:
        """Identify anomalous items in `self.items`.

        Args:
            manual: Whether or not select redundant words using manual thresholds.
                    When set to True, manual_thresholds should be specified.
            z: Items with a z-score above this value are considered anomalous. Used only when manual is False.  Default is 3.
            prob: Probability threshold below which items are considered anomalous.


        Returns:
            An alphabetically sorted set of anomalous items.
        """
        self.item_value_df["guassian_values"] = self.gaussianize_values(
            self.item_value_df[self.val_name]
        )
        if manual:
            anomalous_set = self._anomalous_items_manual(prob=prob)
        else:
            anomalous_set = self._anomalous_items_zscore(z_value=z)
        return set(sorted(anomalous_set))

    def _anomalous_items_manual(self, prob: float) -> Set[str]:
        """Identifies anomalous with respect to the input argument `prob` that specifies
        the probability threshold below which the items are considered anomalous.

        Args:
            prob: Probability threshold below which items are considered anomalous.

        Returns:
            Set of anomalous items.

        """
        mean = np.mean(self.item_value_df["guassian_values"])
        std = np.std(self.item_value_df["guassian_values"])
        dist = norm(loc=mean, scale=std)
        self.item_value_df["normal_prob"] = self.item_value_df["guassian_values"].apply(
            lambda x: dist.pdf(x)
        )
        anomalous_items = set(
            self.item_value_df[self.item_value_df["normal_prob"] < prob][
                "item"
            ].to_list()
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
        z = zscore(self.item_value_df["guassian_values"])
        self.item_value_df["zscore"] = z
        anomalies_set = set()
        for i in range(len(self.item_value_df)):
            if (
                self.item_value_df.iloc[i]["zscore"] <= -z_value
                or self.item_value_df.iloc[i]["zscore"] >= z_value
            ):
                anomalies_set.add(self.item_value_df.iloc[i]["item"])
        return anomalies_set

    def gaussianize_values(self, values: Iterable[float]) -> npt.NDArray:
        """Gaussianize input values using the brute strategy.

        Args:
            values: Iterable containing numerical values.

        Returns:
            numpy.NDArray containing the gaussianized distribution of `values`.
        """
        g = gaussianize.Gaussianize(strategy="brute").fit(values)
        return g.transform(values)

    def show_plot(self, type: str = "default") -> None:
        """Create a distribution plot for the representative value to help manually
            identify cut-off thresholds.

        Args:
            type: Type of the data that is plotted. It can be `default` or `normal`. Defaults to `default`.

        Returns:
            None
        """
        if type == "default":
            x = self.item_value_df[self.val_name].to_list()
            group_labels = [
                "Original values",
            ]
            colors = ["slategray"]
            curve_type = "kde"
        elif type == "normal":
            x = self.item_value_df["guassian_values"].to_list()
            group_labels = [
                "Guassianized values",
            ]
            colors = ["magenta"]
            curve_type = "normal"
        fig = ff.create_distplot(
            [x],
            group_labels,
            bin_size=0.5,
            curve_type=curve_type,  # override default 'kde'
            colors=colors,
        )
        fig.show()


if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer

    imdb_train = pd.read_csv(
        "data/imdb_train_sample.tsv", sep="\t", names=["label", "text"]
    )
    imdb_train = imdb_train[:100]
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(imdb_train["text"])
    idf = vectorizer.idf_
    token_score_dict = dict(zip(vectorizer.get_feature_names(), idf))
    nda = NormalDistAnomalies(items=token_score_dict)
    print(nda.anomalous_items(manual=False))
