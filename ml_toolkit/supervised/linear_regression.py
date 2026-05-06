"""
Linear regression with optional L1 (lasso) or L2 (ridge) regularization.

Both closed-form (for OLS / ridge) and gradient-descent solvers are
available. The default solver is gradient descent so that the same code
path supports unregularized, ridge, and lasso regression.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


class LinearRegression:
    """Linear regression with optional L1 / L2 regularization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iter : int, default=1000
        Number of gradient descent iterations.
    penalty : {"none", "l2", "l1"}, default="none"
        Type of regularization to apply.
    alpha : float, default=0.0
        Regularization strength. Only used when ``penalty != "none"``.
    fit_intercept : bool, default=True
        Whether to learn a bias term.
    solver : {"gd", "closed_form"}, default="gd"
        Optimization method. ``"closed_form"`` is supported only for
        ``penalty in {"none", "l2"}``.

    Attributes
    ----------
    coef_ : np.ndarray
        Learned coefficients of shape (n_features,).
    intercept_ : float
        Learned bias term (0 if ``fit_intercept=False``).
    loss_history_ : list of float
        Mean-squared-error loss at every iteration (only for the gradient
        descent solver).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iter: int = 1000,
        penalty: Literal["none", "l2", "l1"] = "none",
        alpha: float = 0.0,
        fit_intercept: bool = True,
        solver: Literal["gd", "closed_form"] = "gd",
    ) -> None:
        """Initialize the linear regression model with the given hyperparameters."""
        if penalty not in {"none", "l2", "l1"}:
            raise ValueError(f"Unknown penalty: {penalty}")
        if solver not in {"gd", "closed_form"}:
            raise ValueError(f"Unknown solver: {solver}")
        if solver == "closed_form" and penalty == "l1":
            raise ValueError("L1 penalty is not supported with closed-form solver.")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the linear regression model on (X, y)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.solver == "closed_form":
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)
        return self

    def _fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X
        n_features = X_aug.shape[1]
        if self.penalty == "l2" and self.alpha > 0:
            reg = self.alpha * np.eye(n_features)
            if self.fit_intercept:
                reg[0, 0] = 0.0  # do not regularize the intercept
            theta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        else:
            theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y
            grad_w = (X.T @ error) / n_samples
            grad_b = error.mean()

            if self.penalty == "l2":
                grad_w = grad_w + self.alpha * self.coef_
            elif self.penalty == "l1":
                grad_w = grad_w + self.alpha * np.sign(self.coef_)

            self.coef_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * grad_b

            self.loss_history_.append(float(np.mean(error ** 2)))

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X using the fitted coefficients."""
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R^2 on the given data."""
        from ml_toolkit.metrics import r2_score
        return r2_score(y, self.predict(X))
