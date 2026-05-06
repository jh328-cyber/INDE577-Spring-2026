"""
Linear Support Vector Machine (soft-margin) trained with sub-gradient descent.

The model learns a hyperplane ``w . x + b = 0`` that separates the two
classes with the largest possible margin, while allowing some training
points to lie inside the margin (the "soft" part). Training minimizes the
regularized hinge loss

    L(w, b) = (1/2) * ||w||^2 + C * mean(max(0, 1 - y_i (w . x_i + b)))

with sub-gradient descent (PEGASOS-style updates). Labels are accepted
either as ``{0, 1}`` or ``{-1, +1}`` and are converted internally to
``{-1, +1}``.
"""

from __future__ import annotations

import numpy as np


class LinearSVC:
    """Soft-margin linear Support Vector Classifier.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength on the hinge loss term. Larger
        ``C`` means a smaller margin but fewer training-set violations.
    learning_rate : float, default=0.01
        Step size for sub-gradient descent.
    n_iter : int, default=1000
        Number of full-batch sub-gradient iterations.
    fit_intercept : bool, default=True
        Whether to learn a bias term ``b``.
    random_state : int, optional
        Currently unused; kept for API consistency.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned weight vector of shape ``(n_features,)``.
    intercept_ : float
        Learned bias term (0 if ``fit_intercept=False``).
    classes_ : np.ndarray
        The two original class labels, sorted, as seen in ``fit``.
    loss_history_ : list of float
        Regularized hinge loss at every iteration.
    """

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.01,
        n_iter: int = 1000,
        fit_intercept: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the linear SVM with the given hyperparameters."""
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")
        self.C = C
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVC":
        """Fit the linear SVM on (X, y) by minimizing the regularized hinge loss."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"LinearSVC supports binary classification only, "
                f"got {len(self.classes_)} classes: {self.classes_}"
            )

        # Map original labels to +/- 1 internally. The convention is:
        # smaller label -> -1, larger label -> +1.
        y_signed = np.where(y == self.classes_[0], -1.0, 1.0)

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iter):
            margins = y_signed * (X @ self.coef_ + self.intercept_)
            # Indicator: 1 where the constraint is violated (margin < 1).
            violating = margins < 1

            # Sub-gradient of (1/2 ||w||^2 + C * mean(hinge)) w.r.t. w
            grad_w = self.coef_ - self.C * (
                X[violating].T @ y_signed[violating]
            ) / n_samples
            # Sub-gradient w.r.t. b
            grad_b = -self.C * y_signed[violating].sum() / n_samples

            self.coef_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b

            hinge = np.maximum(0.0, 1.0 - margins).mean()
            loss = 0.5 * float(self.coef_ @ self.coef_) + self.C * float(hinge)
            self.loss_history_.append(loss)

        return self

    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed distance ``w . x + b`` for each sample."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (in the original label space)."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the SVM on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
