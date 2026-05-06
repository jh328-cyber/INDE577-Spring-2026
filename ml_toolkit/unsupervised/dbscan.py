"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise.

A point is a *core point* if it has at least ``min_samples`` neighbors
within distance ``eps``. Two core points within ``eps`` of each other are
in the same cluster, and reachable non-core points are also assigned to
that cluster. Unreachable points get the label ``-1`` (noise).
"""

from __future__ import annotations

import numpy as np


class DBSCAN:
    """DBSCAN clustering.

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius.
    min_samples : int, default=5
        Minimum number of points (including self) within ``eps`` for a
        point to be classified as a core point.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster index for each sample (``-1`` denotes noise).
    core_sample_indices_ : np.ndarray
        Indices of the points that were classified as core points.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        """Initialize DBSCAN with the given eps neighborhood radius and min_samples."""
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Run DBSCAN on X; populates labels_ where -1 marks noise points."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        # Pairwise Euclidean distances
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        neighborhoods = [np.where(row <= self.eps)[0] for row in D]
        is_core = np.array([len(nb) >= self.min_samples for nb in neighborhoods])

        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i] or not is_core[i]:
                continue
            # Start a new cluster from this core point and expand by BFS.
            queue = [i]
            visited[i] = True
            while queue:
                j = queue.pop()
                labels[j] = cluster_id
                if is_core[j]:
                    for k in neighborhoods[j]:
                        if not visited[k]:
                            visited[k] = True
                            queue.append(k)
                        elif labels[k] == -1:
                            # A noise neighbor of a core point becomes a
                            # border point in this cluster.
                            labels[k] = cluster_id
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN on X and return the cluster labels in one step."""
        return self.fit(X).labels_
