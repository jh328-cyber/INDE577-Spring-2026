"""
Evaluation metrics for regression, classification, and clustering.

These metrics are computed from raw NumPy arrays and mirror the names and
behaviour of the equivalents in scikit-learn so that results are easy to
cross-check.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error: ``mean((y_true - y_pred) ** 2)``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error: ``mean(|y_true - y_pred|)``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares around the mean of ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        # Constant target: return 0 if predictions are also constant equal,
        # otherwise something is off; convention is to return 0.
        return 0.0
    return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the proportion of correctly predicted labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | list | None = None,
) -> np.ndarray:
    """Return a confusion matrix C such that ``C[i, j]`` is the number of
    observations of true class ``labels[i]`` predicted as class ``labels[j]``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer (or string) class labels.
    y_pred : np.ndarray
        Predicted class labels of the same shape.
    labels : array-like, optional
        Ordered list of class labels to index the rows and columns of the
        matrix. If ``None`` (default), the union of unique values appearing
        in ``y_true`` and ``y_pred`` is used in sorted order.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape ``(n_labels, n_labels)``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(labels.tolist())}
    for t, p in zip(y_true, y_pred):
        # Skip points whose label is not in ``labels``
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int | str | None = None,
    labels: np.ndarray | list | None = None,
) -> Tuple[float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall and F1-score.

    Two modes:

    1. **Binary** — pass ``positive_label`` and the function returns three
       scalars (precision, recall, F1) for that class.
    2. **Multi-class** — pass ``labels`` (or pass neither, in which case the
       function discovers the labels from the data) and the function returns
       three NumPy arrays of length ``len(labels)``, one entry per class.

    Returns
    -------
    (precision, recall, f1) : tuple of float or tuple of np.ndarray
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if positive_label is not None:
        # Binary mode
        tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
        fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
        fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return float(precision), float(recall), float(f1)

    # Multi-class mode (per-label vectors)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    p = np.zeros(len(labels))
    r = np.zeros(len(labels))
    f = np.zeros(len(labels))
    for i, c in enumerate(labels):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if (p[i] + r[i]) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean silhouette coefficient over all samples.

    The silhouette of a point i is ``(b - a) / max(a, b)`` where ``a`` is the
    mean intra-cluster distance and ``b`` is the mean nearest-cluster
    distance. Values close to 1 indicate well-separated clusters; values
    close to 0 indicate overlapping clusters.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Silhouette requires at least 2 clusters.")

    # Pairwise Euclidean distances
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    sil = np.zeros(n)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False  # exclude self
        if own_mask.sum() == 0:
            sil[i] = 0.0
            continue
        a = dist[i, own_mask].mean()
        b_candidates = []
        for c in unique_labels:
            if c == own:
                continue
            mask = labels == c
            if mask.sum() == 0:
                continue
            b_candidates.append(dist[i, mask].mean())
        b = min(b_candidates)
        sil[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(sil.mean())
