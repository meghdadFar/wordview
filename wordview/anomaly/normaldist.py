from typing import Dict, Iterable, Set

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.figure_factory as ff
from scipy.stats import norm, shapiro, zscore

from wordview import logger
from wordview.anomaly import gaussianize


class NormalDistAnomalies(object):
    def __init__(
        self,
        items: pd.DataFrame,
        val_name: str = "representative_value",
        gaussianization_strategy: str = "brute",
    ):
        """Identify anomalies on a normal distribution.

        Args:
            items: A data frame with items and their representative value, such as word_count, idf, etc.
            val_name: Name of the value in the above data frame. i.e. word_count, idf, etc. Defaults to `representative_value`.
            gaussianization_strategy: Strategy for gaussianization. Can be any of lambert, brute, or boxcox. Defaults = `brute`.

        Returns:
            None
        """
        self.items = items
        self.val_name = val_name
        self.gaussianization_strategy = gaussianization_strategy

        # Gaussianize values
        g_results = self._gaussianize_values(
            self.items[self.val_name], strategy=self.gaussianization_strategy
        )
        if shapiro(g_results).pvalue > 0.05:
            self.items["gaussian_values"] = g_results
            # Calculate normal prob of gaussianized values
            dist = norm(
                loc=np.mean(self.items["gaussian_values"]),
                scale=np.std(self.items["gaussian_values"]),
            )
            self.items["normal_prob"] = self.items["gaussian_values"].apply(
                lambda x: dist.pdf(x)
            )
        else:
            logger.error(
                "The provided values cannot be gaussianized. Please consider using another anomaly detection method."
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
            z: Items with a z-score above this value are considered anomalous. Used only when manual is False.  Default = 3.
            prob: Probability threshold below which items are considered anomalous.

        Returns:
            An alphabetically sorted set of anomalous items.
        """
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
        anomalous_items = set(
            self.items[self.items["normal_prob"] < prob]["item"].to_list()
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
        z = zscore(self.items["gaussian_values"])
        self.items["zscore"] = z
        anomalies_set = set()
        for i in range(len(self.items)):
            if (
                self.items.iloc[i]["zscore"] <= -z_value
                or self.items.iloc[i]["zscore"] >= z_value
            ):
                anomalies_set.add(self.items.iloc[i]["item"])
        return anomalies_set

    def _gaussianize_values(
        self, values: Iterable[float], strategy: str
    ) -> npt.NDArray:
        """Gaussianize input values using the brute strategy.

        Args:
            values: Iterable containing numerical values.
            strategy: Strategy for gaussianization. Can be any of lambert, brute, or boxcox.

        Returns:
            numpy.NDArray containing the gaussianized distribution of `values`.
        """
        g = gaussianize.Gaussianize(strategy=strategy)
        g.fit(values)
        res = g.transform(values)
        return res
