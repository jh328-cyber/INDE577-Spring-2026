"""
Binary logistic regression trained with gradient descent.

The model predicts P(y=1 | x) = sigmoid(w . x + b), and is trained by
minimizing the average binary cross-entropy loss with optional L2
regularization.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class LogisticRegression:
    """Binary logistic regression with gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size for gradient descent.
    n_iter : int, default=1000
        Number of training iterations.
    alpha : float, default=0.0
        L2 regularization strength.
    fit_intercept : bool, default=True
        Whether to learn a bias term.
    threshold : float, default=0.5
        Threshold applied to predicted probabilities to obtain class labels.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned coefficient vector.
    intercept_ : float
        Learned bias term.
    loss_history_ : list of float
        Cross-entropy loss at every iteration.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iter: int = 1000,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the logistic regression model with the given hyperparameters."""
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.threshold = threshold

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the binary logistic regression model on (X, y) by gradient descent."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        unique = np.unique(y)
        if not set(unique.tolist()).issubset({0.0, 1.0}):
            raise ValueError(
                f"LogisticRegression expects binary 0/1 labels, got {unique}"
            )

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_ = []
        eps = 1e-12

        for _ in range(self.n_iter):
            z = X @ self.coef_ + self.intercept_
            p = _sigmoid(z)
            error = p - y
            grad_w = (X.T @ error) / n_samples + self.alpha * self.coef_
            grad_b = error.mean()

            self.coef_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b

            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            if self.alpha > 0:
                loss += 0.5 * self.alpha * float(np.sum(self.coef_ ** 2))
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | X) for each row of X."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        p1 = _sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for X using a 0.5 probability threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the model on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
