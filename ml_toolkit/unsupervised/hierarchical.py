"""
Agglomerative hierarchical clustering with single, complete or average
linkage. Implementation is O(n^3) and intended for small datasets only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _update_distance(
    D: np.ndarray,
    i: int,
    j: int,
    linkage: str,
    sizes: np.ndarray,
) -> None:
    """Merge cluster j into cluster i in-place; mark j as inactive (np.inf)."""
    n = D.shape[0]
    for k in range(n):
        if k == i or k == j:
            continue
        if linkage == "single":
            new = min(D[i, k], D[j, k])
        elif linkage == "complete":
            new = max(D[i, k], D[j, k])
        elif linkage == "average":
            new = (sizes[i] * D[i, k] + sizes[j] * D[j, k]) / (sizes[i] + sizes[j])
        else:
            raise ValueError(f"Unknown linkage: {linkage}")
        D[i, k] = D[k, i] = new
    # Mark cluster j as removed by setting its row/col to infinity.
    D[j, :] = np.inf
    D[:, j] = np.inf
    D[i, i] = np.inf  # ignore diagonal


class HierarchicalClustering:
    """Agglomerative hierarchical clustering.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to return.
    linkage : {"single", "complete", "average"}, default="average"
        Linkage strategy used to compute the distance between clusters.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster index for each input sample.
    linkage_matrix_ : np.ndarray
        Merge history of shape ``(n_samples - 1, 4)`` in the same format that
        ``scipy.cluster.hierarchy.dendrogram`` expects: each row is
        ``[cluster_a, cluster_b, merge_distance, new_cluster_size]``. Cluster
        ids 0..n-1 refer to original observations; ids n..2n-2 refer to
        clusters formed by previous merges.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal["single", "complete", "average"] = "average",
    ) -> None:
        """Initialize agglomerative clustering with the given number of clusters and linkage rule."""
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        if linkage not in {"single", "complete", "average"}:
            raise ValueError(f"Unknown linkage: {linkage}")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_: np.ndarray | None = None
        self.linkage_matrix_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "HierarchicalClustering":
        """Build the agglomerative merge tree on X and cut it to produce n_clusters clusters."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                f"n_clusters={self.n_clusters} cannot exceed n_samples={n}"
            )

        # Initialize: each point is its own cluster.
        cluster_assignment = np.arange(n)
        sizes = np.ones(n)
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        active = n

        # Track merges in scipy linkage-matrix format. cluster_id[i] is the
        # current scipy id (0..n-1 for leaves; n..2n-2 for merged) of the
        # cluster currently *living at* slot i.
        cluster_id = np.arange(n)
        merges: list[list[float]] = []
        next_id = n

        # We always run merges until everything is one cluster so that the
        # full dendrogram is available, then derive labels by cutting at the
        # appropriate height for ``n_clusters``.
        while active > 1:
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if i > j:
                i, j = j, i
            d = float(D[i, j])
            new_size = sizes[i] + sizes[j]
            merges.append([float(cluster_id[i]), float(cluster_id[j]), d, float(new_size)])
            _update_distance(D, i, j, self.linkage, sizes)
            cluster_assignment[cluster_assignment == j] = i
            sizes[i] = new_size
            sizes[j] = 0
            cluster_id[i] = next_id
            next_id += 1
            active -= 1
            # Save the labels right when we cross n_clusters
            if active == self.n_clusters:
                snapshot = cluster_assignment.copy()

        # Re-number cluster ids at the snapshot to be contiguous 0..n_clusters-1
        unique_clusters = np.unique(snapshot)
        relabel = {c: i for i, c in enumerate(unique_clusters)}
        self.labels_ = np.array([relabel[c] for c in snapshot])
        self.linkage_matrix_ = np.array(merges)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model on X and return the cluster labels in one step."""
        return self.fit(X).labels_
