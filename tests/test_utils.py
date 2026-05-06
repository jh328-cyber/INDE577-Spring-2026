"""Tests for ml_toolkit.utils."""

import numpy as np
import pytest

from ml_toolkit.utils import MinMaxScaler, StandardScaler, train_test_split


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------
class TestTrainTestSplit:
    def test_returns_correct_sizes(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        assert X_tr.shape == (40, 2)
        assert X_te.shape == (10, 2)
        assert y_tr.shape == (40,)
        assert y_te.shape == (10,)

    def test_reproducible_with_seed(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        a = train_test_split(X, y, test_size=0.2, random_state=42)
        b = train_test_split(X, y, test_size=0.2, random_state=42)
        for arr_a, arr_b in zip(a, b):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_no_overlap_between_train_and_test(self):
        X = np.arange(100).reshape(50, 2)
        y = np.arange(50)
        X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.3, random_state=0)
        train_rows = {tuple(r) for r in X_tr}
        test_rows = {tuple(r) for r in X_te}
        assert train_rows.isdisjoint(test_rows)

    def test_invalid_test_size_raises(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.0)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=1.0)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=-0.1)

    def test_inconsistent_lengths_raises(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(5)
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=0.2)

    def test_stratify_preserves_class_proportions(self):
        # 60 class-0, 30 class-1, 10 class-2 -> 6:3:1 ratio
        y = np.array([0] * 60 + [1] * 30 + [2] * 10)
        X = np.arange(len(y)).reshape(-1, 1)
        _, _, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )
        # Each split should hold the 6:3:1 ratio (within rounding)
        for arr in (y_tr, y_te):
            counts = np.bincount(arr, minlength=3)
            ratios = counts / counts.sum()
            np.testing.assert_allclose(ratios, [0.6, 0.3, 0.1], atol=0.05)

    def test_stratify_length_mismatch_raises(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        with pytest.raises(ValueError, match="stratify"):
            train_test_split(X, y, stratify=np.arange(5))


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------
class TestStandardScaler:
    def test_zero_mean_unit_variance(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-10)

    def test_constant_feature_no_division_error(self):
        X = np.column_stack([np.ones(10), np.arange(10)])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)  # must not raise
        assert np.all(np.isfinite(Xs))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------
class TestMinMaxScaler:
    def test_range_zero_to_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        np.testing.assert_allclose(Xs.min(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xs.max(axis=0), 1.0, atol=1e-10)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            MinMaxScaler().transform(np.zeros((3, 2)))
