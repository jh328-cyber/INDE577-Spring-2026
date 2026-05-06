"""
Utility functions for data preparation.

Includes train/test splitting and basic scalers, all implemented from scratch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify: np.ndarray | bool | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split. Must lie in
        the open interval (0, 1).
    random_state : int, optional
        Seed for the random number generator. Pass an int for reproducible
        output.
    stratify : np.ndarray or bool, optional
        If provided as an array, this must be a class-label array (typically
        ``y``) and the split will preserve the proportion of each class in
        both the train and test sets. ``stratify=True`` is shorthand for
        ``stratify=y``. If ``None`` (the default), splitting is purely random.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train and test splits of X and y.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have inconsistent lengths, if ``test_size`` is
        outside (0, 1), or if ``stratify`` is given but its length does not
        match ``y``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y)}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    rng = np.random.default_rng(random_state)
    n_samples = len(X)

    if stratify is None:
        indices = rng.permutation(n_samples)
        n_test = int(round(n_samples * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    else:
        # Stratified split: keep class proportions in both train and test.
        if isinstance(stratify, bool):
            if not stratify:
                # stratify=False is treated like None
                indices = rng.permutation(n_samples)
                n_test = int(round(n_samples * test_size))
                test_idx = indices[:n_test]
                train_idx = indices[n_test:]
                return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            strata = y
        else:
            strata = np.asarray(stratify)
        if len(strata) != n_samples:
            raise ValueError(
                f"stratify length {len(strata)} does not match y length {n_samples}"
            )
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for cls in np.unique(strata):
            cls_indices = np.where(strata == cls)[0]
            cls_indices = rng.permutation(cls_indices)
            n_cls_test = int(round(len(cls_indices) * test_size))
            test_parts.append(cls_indices[:n_cls_test])
            train_parts.append(cls_indices[n_cls_test:])
        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        # Shuffle once more so the train/test sets are not class-ordered
        test_idx = rng.permutation(test_idx)
        train_idx = rng.permutation(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The transformation is ``z = (x - mean) / std``, computed per feature on
    the training data.

    Attributes
    ----------
    mean_ : np.ndarray
        Per-feature mean computed on the training data.
    scale_ : np.ndarray
        Per-feature standard deviation (with zeros replaced by one to avoid
        division by zero).
    """

    def __init__(self) -> None:
        """Initialize a standard scaler."""
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute the per-feature mean and standard deviation from X."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Replace zero std with 1 to avoid division by zero (constant features)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize X using the previously fitted statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the standardized array in one step."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features to the range [0, 1].

    The transformation is ``z = (x - min) / (max - min)``.
    """

    def __init__(self) -> None:
        """Initialize a min-max scaler with a target output range."""
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute the per-feature min and max from X."""
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        rng = X.max(axis=0) - self.min_
        self.range_ = np.where(rng == 0, 1.0, rng)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to the configured output range using the fitted min/max."""
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the scaled array in one step."""
        return self.fit(X).transform(X)
