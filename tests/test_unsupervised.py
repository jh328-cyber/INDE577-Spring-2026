"""Tests for ml_toolkit.unsupervised algorithms."""

import numpy as np
import pytest

from ml_toolkit.unsupervised import DBSCAN, PCA, HierarchicalClustering, KMeans


def _make_three_blobs(n_per_cluster=30, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[-5, -5], [5, 5], [-5, 5]])
    X = np.vstack(
        [rng.normal(loc=c, scale=0.4, size=(n_per_cluster, 2)) for c in centers]
    )
    return X, centers


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------
class TestKMeans:
    def test_recovers_three_blobs(self):
        X, centers = _make_three_blobs(seed=0)
        model = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        # The order of centers is arbitrary; check that each true center
        # is close to *some* recovered center.
        for c in centers:
            dists = np.linalg.norm(model.cluster_centers_ - c, axis=1)
            assert dists.min() < 1.0

    def test_predict_consistent_with_labels(self):
        X, _ = _make_three_blobs(seed=1)
        model = KMeans(n_clusters=3, n_init=5, random_state=0).fit(X)
        np.testing.assert_array_equal(model.predict(X), model.labels_)

    def test_inertia_is_non_negative(self):
        X, _ = _make_three_blobs(seed=2)
        model = KMeans(n_clusters=3, n_init=3, random_state=0).fit(X)
        assert model.inertia_ >= 0

    def test_invalid_n_clusters_raises(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)

    def test_more_clusters_than_samples_raises(self):
        X = np.random.default_rng(0).normal(size=(3, 2))
        with pytest.raises(ValueError):
            KMeans(n_clusters=5).fit(X)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
class TestPCA:
    def test_explained_variance_ratio_sums_to_one(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 4))
        pca = PCA(n_components=4).fit(X)
        assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-9)

    def test_components_are_orthonormal(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 3))
        pca = PCA(n_components=3).fit(X)
        gram = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-9)

    def test_inverse_transform_reconstructs_full_rank(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 4))
        pca = PCA(n_components=4).fit(X)
        X_back = pca.inverse_transform(pca.transform(X))
        np.testing.assert_allclose(X_back, X, atol=1e-9)

    def test_invalid_n_components_raises(self):
        with pytest.raises(ValueError):
            PCA(n_components=0)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
class TestDBSCAN:
    def test_finds_three_clusters_on_well_separated_blobs(self):
        X, _ = _make_three_blobs(seed=3)
        model = DBSCAN(eps=1.0, min_samples=4).fit(X)
        # Excluding noise, we should have at most 3 clusters and at least 2.
        unique = set(model.labels_) - {-1}
        assert 2 <= len(unique) <= 3

    def test_noise_for_isolated_points(self):
        X = np.array([[0, 0], [0.1, 0], [0, 0.1], [10, 10]], dtype=float)
        model = DBSCAN(eps=0.5, min_samples=2).fit(X)
        # Last point is isolated and must be flagged as noise.
        assert model.labels_[-1] == -1

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError):
            DBSCAN(eps=0)

    def test_invalid_min_samples_raises(self):
        with pytest.raises(ValueError):
            DBSCAN(min_samples=0)


# ---------------------------------------------------------------------------
# HierarchicalClustering
# ---------------------------------------------------------------------------
class TestHierarchical:
    def test_recovers_three_blobs(self):
        X, _ = _make_three_blobs(seed=4)
        model = HierarchicalClustering(n_clusters=3, linkage="average").fit(X)
        assert len(np.unique(model.labels_)) == 3

    def test_invalid_linkage_raises(self):
        with pytest.raises(ValueError):
            HierarchicalClustering(linkage="ward")  # not supported

    def test_too_many_clusters_raises(self):
        X = np.random.default_rng(0).normal(size=(5, 2))
        with pytest.raises(ValueError):
            HierarchicalClustering(n_clusters=10).fit(X)

    def test_linkage_matrix_shape_and_monotone(self):
        X, _ = _make_three_blobs(seed=7)
        n = X.shape[0]
        model = HierarchicalClustering(n_clusters=3, linkage="complete").fit(X)
        Z = model.linkage_matrix_
        # n-1 merges, 4 columns each
        assert Z.shape == (n - 1, 4)
        # Distances should be non-decreasing for complete linkage
        distances = Z[:, 2]
        assert np.all(np.diff(distances) >= -1e-9)
        # Cluster sizes in last column should be increasing in expectation
        assert Z[-1, 3] == n  # final merge contains all points
