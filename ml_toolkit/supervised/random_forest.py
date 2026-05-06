"""
Random Forest classifier built on top of ``DecisionTreeClassifier``.

A random forest is a bagging ensemble of decision trees: each tree is
trained on a different bootstrap sample of the rows and a random subset
of the columns. Predictions are made by majority vote.

The two sources of randomness — bootstrap rows and random feature
subsetting — make the trees less correlated, which usually drives the
ensemble's variance well below that of any single tree.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_toolkit.supervised.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Random forest classifier with bootstrap rows and random feature subsets.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    max_depth : int, default=10
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    max_features : {"sqrt", "log2"} or int or None, default="sqrt"
        Number of features randomly drawn for each tree to consider.
        ``"sqrt"`` uses ``floor(sqrt(n_features))`` (the standard default).
    bootstrap : bool, default=True
        Whether to draw each tree's training rows with replacement
        (standard bagging). When ``False``, every tree sees every row.
    random_state : int, optional
        Seed for both bootstrap row sampling and feature subset sampling,
        making the entire forest reproducible.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The trained tree estimators.
    feature_indices_ : list of np.ndarray
        For each tree, the column indices of the features it was trained on.
    classes_ : np.ndarray
        Sorted class labels seen in ``fit``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: Literal["sqrt", "log2"] | int | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the random forest with the given hyperparameters."""
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {min_samples_split}"
            )
        if isinstance(max_features, int) and max_features <= 0:
            raise ValueError(
                f"max_features as int must be positive, got {max_features}"
            )
        if (
            max_features is not None
            and not isinstance(max_features, int)
            and max_features not in {"sqrt", "log2"}
        ):
            raise ValueError(
                f"max_features must be 'sqrt', 'log2', a positive int, or None; "
                f"got {max_features!r}"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: list[DecisionTreeClassifier] = []
        self.feature_indices_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(np.floor(np.sqrt(n_features))))
        if self.max_features == "log2":
            return max(1, int(np.floor(np.log2(n_features))))
        # The validation in __init__ guarantees we never reach here.
        raise ValueError(f"Unknown max_features setting: {self.max_features!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Train each tree on a bootstrap sample and a random feature subset of (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_feat_per_tree = self._resolve_max_features(n_features)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.feature_indices_ = []

        for _ in range(self.n_estimators):
            # Bootstrap rows
            if self.bootstrap:
                row_idx = rng.integers(0, n_samples, size=n_samples)
            else:
                row_idx = np.arange(n_samples)
            # Random feature subset (no replacement)
            feat_idx = rng.choice(n_features, size=n_feat_per_tree, replace=False)

            X_sub = X[row_idx][:, feat_idx]
            y_sub = y[row_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X_sub, y_sub)

            self.trees_.append(tree)
            self.feature_indices_.append(feat_idx)

        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by majority vote across the forest."""
        if not self.trees_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)

        # Collect predictions from each tree, indexed by its feature subset.
        # Shape after stacking: (n_estimators, n_samples).
        all_preds = np.array(
            [
                tree.predict(X[:, feat_idx])
                for tree, feat_idx in zip(self.trees_, self.feature_indices_)
            ]
        )

        # Majority vote across trees, per sample.
        n_samples = X.shape[0]
        out = np.empty(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            values, counts = np.unique(all_preds[:, i], return_counts=True)
            out[i] = values[np.argmax(counts)]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class vote frequencies as soft-probability estimates."""
        if not self.trees_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        all_preds = np.array(
            [
                tree.predict(X[:, feat_idx])
                for tree, feat_idx in zip(self.trees_, self.feature_indices_)
            ]
        )
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.mean(all_preds[:, i] == cls)
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the forest on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
