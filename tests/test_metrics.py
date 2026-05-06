"""Tests for ml_toolkit.metrics."""

import numpy as np
import pytest

from ml_toolkit.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_f1,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------
class TestRegressionMetrics:
    def test_mse_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_mse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert mean_squared_error(y_true, y_pred) == pytest.approx(1.0 / 3.0)

    def test_mse_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_squared_error(np.array([1, 2]), np.array([1, 2, 3]))

    def test_rmse_is_sqrt_of_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            np.sqrt(1.0 / 3.0)
        )

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)

    def test_r2_perfect_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_predict_mean_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y, y.mean())
        assert r2_score(y, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
class TestClassificationMetrics:
    def test_accuracy_all_correct(self):
        y = np.array([0, 1, 1, 0])
        assert accuracy_score(y, y) == 1.0

    def test_accuracy_known_value(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)

    def test_accuracy_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy_score(np.array([0, 1]), np.array([0]))

    def test_confusion_matrix_known(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred)
        np.testing.assert_array_equal(cm, np.array([[1, 1], [1, 1]]))

    def test_precision_recall_f1_simple(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        # TP=2, FP=1, FN=1
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label=1)
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_precision_recall_no_positives_returns_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label=1)
        assert p == 0.0 and r == 0.0 and f1 == 0.0

    def test_confusion_matrix_with_explicit_labels(self):
        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 2, 2, 0, 1])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        # Row=true class, col=predicted class
        expected = np.array([
            [1, 0, 0],  # true 0: 1 predicted as 0
            [0, 1, 1],  # true 1: 1 predicted as 1, 1 as 2
            [1, 0, 1],  # true 2: 1 predicted as 0, 1 as 2
        ])
        np.testing.assert_array_equal(cm, expected)

    def test_precision_recall_f1_multiclass(self):
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 0])  # only one error
        p, r, f = precision_recall_f1(y_true, y_pred, labels=[0, 1, 2])
        # All three should be arrays of length 3
        assert p.shape == (3,) and r.shape == (3,) and f.shape == (3,)
        # Class 1 is perfect -> precision=recall=f1=1
        assert p[1] == pytest.approx(1.0)
        assert r[1] == pytest.approx(1.0)
        assert f[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------
class TestClusteringMetrics:
    def test_silhouette_well_separated(self):
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
        labels = np.array([0, 0, 1, 1])
        score = silhouette_score(X, labels)
        # Two extremely separated clusters should be near 1
        assert score > 0.9

    def test_silhouette_single_cluster_raises(self):
        X = np.array([[0, 0], [1, 1]], dtype=float)
        labels = np.array([0, 0])
        with pytest.raises(ValueError):
            silhouette_score(X, labels)
