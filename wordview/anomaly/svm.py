from typing import Dict, List, Set

import pandas as pd
from sklearn.svm import OneClassSVM


class SVMAnomalies:
    def __init__(
        self,
        items: pd.DataFrame,
        val_names: List = [],
        nu: float = 0.05,
        kernel: str = "rbf",
    ):
        """Identify anomalies using One-Class SVM.

        Args:
            items: A data frame with items and their representative values, such as word_count, idf, etc.
            val_names: Name of the values in the above data frame. i.e. word_count, idf, etc. Defaults to an empty list (all columns).
            nu: Parameter for the one-class SVM model. Defaults to 0.05.
            kernel: Specifies the kernel type to be used in the SVM model. Defaults to `rbf`.

        Returns:
            None
        """
        self.items = items
        self.val_names = val_names if val_names else items.columns[1:]
        self.model = OneClassSVM(nu=nu, kernel=kernel)
        self.items["anomaly"] = self._detect_anomalies()

    def _detect_anomalies(self) -> List[int]:
        """Detect anomalies in `self.items`."""

        self.model.fit(self.items[self.val_names])
        return self.model.predict(self.items[self.val_names])

    def anomalous_items(self) -> Set[str]:
        """Identify anomalous items in `self.items`.

        Returns:
            An alphabetically sorted set of anomalous items.
        """
        return set(sorted(self.items[self.items["anomaly"] == -1]["item"]))
