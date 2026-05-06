"""
Principal Component Analysis via eigendecomposition of the covariance matrix.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis (PCA) via eigendecomposition.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If ``None``, keep all components.

    Attributes
    ----------
    components_ : np.ndarray
        Principal axes in feature space, shape ``(n_components, n_features)``.
    mean_ : np.ndarray
        Per-feature mean estimated from the training data.
    explained_variance_ : np.ndarray
        Variance explained by each selected component (the eigenvalues).
    explained_variance_ratio_ : np.ndarray
        ``explained_variance_`` divided by total variance.
    """

    def __init__(self, n_components: int | None = None) -> None:
        """Initialize PCA with the desired number of components."""
        if n_components is not None and n_components <= 0:
            raise ValueError(
                f"n_components must be positive, got {n_components}"
            )
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Compute the principal components of X via SVD on the centered data matrix."""
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Use SVD: more numerically stable than eig on the covariance matrix.
        # X_centered = U S V^T -> components are rows of V^T.
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Variance explained = S^2 / (n_samples - 1)
        explained_variance = (S ** 2) / max(n_samples - 1, 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = (
            explained_variance / total_var if total_var > 0 else explained_variance
        )

        n_comp = self.n_components or min(n_samples, n_features)
        if n_comp > Vt.shape[0]:
            raise ValueError(
                f"n_components={n_comp} cannot exceed min(n_samples, n_features)"
                f"={Vt.shape[0]}"
            )
        self.components_ = Vt[:n_comp]
        self.explained_variance_ = explained_variance[:n_comp]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_comp]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the previously fitted principal components."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the components on X and return the projected data in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """Map data from component space back to the original feature space."""
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_proj = np.asarray(X_proj, dtype=float)
        return X_proj @ self.components_ + self.mean_
