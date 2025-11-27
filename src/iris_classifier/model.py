from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@dataclass
class IrisModelConfig:
    test_size: float = 0.2
    random_state: int = 42


class IrisClassifier:
    def __init__(self, config: IrisModelConfig | None = None):
        self.config = config or IrisModelConfig()
        self.model = LogisticRegression(max_iter=200)

    def train(self, df: pd.DataFrame) -> float:
        """
        Train the model on the iris data frame.
        Returns the test accuracy.
        """
        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for new samples.
        """
        return self.model.predict(X)