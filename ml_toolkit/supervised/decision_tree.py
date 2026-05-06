"""
Decision tree classifier (CART variant) using Gini impurity.

The tree is grown greedily: at each node, the split that maximally reduces
the weighted Gini impurity of its children is chosen. Growth stops when one
of ``max_depth``, ``min_samples_split`` or class purity is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Node:
    """Internal tree node."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    prediction: Optional[int] = None  # set when this is a leaf


def _gini(y: np.ndarray) -> float:
    """Compute Gini impurity of a label vector."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return float(1.0 - np.sum(p ** 2))


class DecisionTreeClassifier:
    """A simple decision tree classifier using Gini impurity (CART).

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_state : int, optional
        Currently unused but kept for API compatibility.

    Attributes
    ----------
    root_ : _Node
        Root node of the trained tree.
    n_classes_ : int
        Number of unique classes seen in fit().
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        random_state: int | None = None,
    ) -> None:
        """Initialize the decision tree classifier with the given hyperparameters."""
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {min_samples_split}"
            )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root_: _Node | None = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Grow a CART tree on (X, y) using Gini impurity as the split criterion."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self.n_classes_ = len(np.unique(y))
        self.root_ = self._grow(X, y, depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        # Stopping conditions: all same class, max depth, too few samples.
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return _Node(prediction=self._majority_class(y))

        best = self._best_split(X, y)
        if best is None:
            return _Node(prediction=self._majority_class(y))

        feature, threshold = best
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return _Node(prediction=self._majority_class(y))

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Return (feature_index, threshold) of the best split, or None."""
        n_samples, n_features = X.shape
        parent_impurity = _gini(y)
        best_gain = 0.0
        best_split = None

        for feature in range(n_features):
            # Use unique sorted values as candidate thresholds (midpoints
            # between adjacent unique values would be slightly nicer but
            # this is simpler and works well in practice).
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                child_impurity = (
                    len(left) * _gini(left) + len(right) * _gini(right)
                ) / n_samples
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, float(t))
        return best_split

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by routing each row down the trained tree."""
        if self.root_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the tree on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
