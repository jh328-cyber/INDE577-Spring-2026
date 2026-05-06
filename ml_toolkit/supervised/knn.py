"""
K-Nearest Neighbors algorithms for classification and regression.

Both classes use the brute-force method: at prediction time the distances
from each query point to every training point are computed, and the labels
of the ``k`` nearest neighbors are aggregated.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _pairwise_distances(
    X1: np.ndarray, X2: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Return distance matrix of shape (len(X1), len(X2))."""
    if metric == "euclidean":
        diff = X1[:, None, :] - X2[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
    elif metric == "manhattan":
        diff = X1[:, None, :] - X2[None, :, :]
        return np.abs(diff).sum(axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class _KNNBase:
    """Shared logic for KNN classifier and regressor."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ) -> None:
        """Initialize the KNN base with n_neighbors and a distance metric."""
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Memorize the training data; KNN is a lazy learner so no model is trained at fit time."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot exceed n_samples="
                f"{X.shape[0]}"
            )
        self.X_train_ = X
        self.y_train_ = y
        return self

    def _kneighbors_indices(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        D = _pairwise_distances(X, self.X_train_, self.metric)
        return np.argsort(D, axis=1)[:, : self.n_neighbors]


class KNNClassifier(_KNNBase):
    """K-Nearest Neighbors classifier with majority vote."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by majority vote among the k nearest neighbors."""
        X = np.asarray(X, dtype=float)
        idx = self._kneighbors_indices(X)
        neighbor_labels = self.y_train_[idx]
        # Majority vote per row
        preds = []
        for row in neighbor_labels:
            values, counts = np.unique(row, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the KNN classifier on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class KNNRegressor(_KNNBase):
    """K-Nearest Neighbors regressor: predicts the mean of neighbor targets."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for X by averaging the targets of the k nearest neighbors."""
        X = np.asarray(X, dtype=float)
        idx = self._kneighbors_indices(X)
        return self.y_train_[idx].mean(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R^2 score of the KNN regressor on (X, y)."""
        from ml_toolkit.metrics import r2_score
        return r2_score(y, self.predict(X))
