"""
Gaussian Naive Bayes classifier.

Models P(x_j | y) as a Gaussian per feature per class, and computes the
posterior using log-probabilities for numerical stability.
"""

from __future__ import annotations

import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Small value added to variances to avoid division by zero.

    Attributes
    ----------
    classes_ : np.ndarray
        Sorted unique class labels.
    class_priors_ : np.ndarray
        Prior probability of each class (proportional to class frequency).
    means_ : np.ndarray
        Per-class, per-feature mean. Shape (n_classes, n_features).
    variances_ : np.ndarray
        Per-class, per-feature variance. Shape (n_classes, n_features).
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """Initialize Gaussian Naive Bayes with a variance smoothing constant."""
        if var_smoothing < 0:
            raise ValueError(
                f"var_smoothing must be non-negative, got {var_smoothing}"
            )
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """Estimate per-class means, variances, and class priors from training data."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.class_priors_ = np.zeros(n_classes)

        # Add a tiny smoothing term proportional to the largest variance,
        # mirroring scikit-learn's behaviour to handle zero-variance features.
        epsilon = self.var_smoothing * X.var(axis=0).max() if X.size > 0 else 0.0

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.variances_[i] = X_c.var(axis=0) + epsilon
            self.class_priors_[i] = X_c.shape[0] / X.shape[0]
        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Return log P(x | y) + log P(y) for each class."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_post = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            mean = self.means_[i]
            var = self.variances_[i]
            log_prior = np.log(self.class_priors_[i])
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X - mean) ** 2) / var, axis=1
            )
            log_post[:, i] = log_prior + log_likelihood
        return log_post

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the highest posterior probability for each row of X."""
        X = np.asarray(X, dtype=float)
        log_post = self._log_likelihood(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class posterior probabilities for each row of X (rows sum to 1)."""
        X = np.asarray(X, dtype=float)
        log_post = self._log_likelihood(X)
        # Subtract max for numerical stability before exponentiating.
        log_post -= log_post.max(axis=1, keepdims=True)
        post = np.exp(log_post)
        return post / post.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the model on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
