"""
Rosenblatt's perceptron for binary classification.

The perceptron is a single linear neuron with a sign activation. Given a
training sample (x, y) with y in {-1, +1}, the update rule is:
    if y * (w . x + b) <= 0:
        w += lr * y * x
        b += lr * y
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    """Rosenblatt perceptron for binary classification.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Step size for the perceptron updates.
    n_iter : int, default=100
        Maximum number of passes (epochs) over the training data.
    random_state : int, optional
        Random seed used to shuffle samples each epoch.

    Attributes
    ----------
    coef_ : np.ndarray
        Weight vector.
    intercept_ : float
        Bias term.
    errors_ : list of int
        Number of misclassifications per epoch (useful for diagnostics).
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        n_iter: int = 100,
        random_state: int | None = None,
    ) -> None:
        """Initialize the perceptron with the given hyperparameters."""
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.errors_: list[int] = []
        self._classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Run perceptron learning on (X, y); converges in finite steps if the data are linearly separable."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"Perceptron only supports binary classification, got "
                f"{len(classes)} classes."
            )
        self._classes_ = classes
        # Map labels to {-1, +1} internally
        y_mapped = np.where(y == classes[0], -1, 1).astype(float)

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.errors_ = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_iter):
            order = rng.permutation(n_samples)
            errors = 0
            for i in order:
                xi, yi = X[i], y_mapped[i]
                if yi * (xi @ self.coef_ + self.intercept_) <= 0:
                    self.coef_ += self.learning_rate * yi * xi
                    self.intercept_ += self.learning_rate * yi
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for X using the current weight vector."""
        if self.coef_ is None or self._classes_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        scores = X @ self.coef_ + self.intercept_
        return np.where(scores >= 0, self._classes_[1], self._classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the perceptron on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
