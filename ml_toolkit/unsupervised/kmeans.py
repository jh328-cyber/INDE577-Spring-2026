"""
K-Means clustering with k-means++ initialization.
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """K-Means clustering using Lloyd's algorithm with k-means++ init.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of Lloyd's iterations.
    tol : float, default=1e-4
        Stop when the squared change in centroids is below this threshold.
    n_init : int, default=10
        Number of restarts with different random centroid seeds. The result
        with the lowest inertia is kept.
    random_state : int, optional
        Seed for centroid initialization.

    Attributes
    ----------
    cluster_centers_ : np.ndarray
        Final centroids of shape (n_clusters, n_features).
    labels_ : np.ndarray
        Cluster index assigned to each training sample.
    inertia_ : float
        Sum of squared distances from each sample to its assigned centroid.
    n_iter_ : int
        Number of iterations actually performed in the best run.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int | None = None,
    ) -> None:
        """Initialize the k-means model with the given number of clusters and hyperparameters."""
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if n_init <= 0:
            raise ValueError(f"n_init must be positive, got {n_init}")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    def _init_centers_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++ initialization."""
        n_samples = X.shape[0]
        # Choose first center uniformly at random
        first = rng.integers(0, n_samples)
        centers = [X[first]]
        for _ in range(1, self.n_clusters):
            dists = np.min(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centers]),
                axis=0,
            )
            total = dists.sum()
            if total == 0:
                # All remaining points are duplicates of existing centers;
                # fall back to a random pick.
                idx = rng.integers(0, n_samples)
            else:
                probs = dists / total
                idx = rng.choice(n_samples, p=probs)
            centers.append(X[idx])
        return np.array(centers)

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        centers = self._init_centers_pp(X, rng)
        for it in range(1, self.max_iter + 1):
            # Assignment step: distance from each point to each center
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update step: each new center is the mean of its assigned points.
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() == 0:
                    # Empty cluster: re-seed at the farthest point.
                    farthest = np.argmax(np.min(dists, axis=1))
                    new_centers[k] = X[farthest]
                else:
                    new_centers[k] = X[mask].mean(axis=0)

            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            if shift < self.tol:
                break

        # Final inertia
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.sum(np.min(dists, axis=1) ** 2))
        return centers, labels, inertia, it

    def fit(self, X: np.ndarray) -> "KMeans":
        """Run k-means on X with multiple restarts and keep the best (lowest-inertia) run."""
        X = np.asarray(X, dtype=float)
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} must be >= n_clusters={self.n_clusters}"
            )
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_iter = 0
        for run in range(self.n_init):
            run_seed = rng.integers(0, 2**31 - 1)
            run_rng = np.random.default_rng(int(run_seed))
            centers, labels, inertia, n_iter = self._single_run(X, run_rng)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_iter = n_iter
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of X to the index of its nearest learned centroid."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model on X and return the cluster assignment for each row in one step."""
        return self.fit(X).labels_
