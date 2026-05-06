"""Tests for ml_toolkit.supervised algorithms."""

import numpy as np
import pytest

from ml_toolkit.supervised import (
    DecisionTreeClassifier,
    GaussianNB,
    KNNClassifier,
    KNNRegressor,
    LinearRegression,
    LinearSVC,
    LogisticRegression,
    MLPClassifier,
    Perceptron,
    RandomForestClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_linear_data(n=100, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_w = np.array([2.0, -1.5])
    true_b = 0.5
    y = X @ true_w + true_b + rng.normal(scale=noise, size=n)
    return X, y, true_w, true_b


def _make_blobs(n_per_class=50, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=[-2, -2], scale=0.5, size=(n_per_class, 2))
    X1 = rng.normal(loc=[2, 2], scale=0.5, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


# ---------------------------------------------------------------------------
# LinearRegression
# ---------------------------------------------------------------------------
class TestLinearRegression:
    def test_recovers_true_coefficients_closed_form(self):
        X, y, true_w, true_b = _make_linear_data(n=200, noise=0.01)
        model = LinearRegression(solver="closed_form").fit(X, y)
        np.testing.assert_allclose(model.coef_, true_w, atol=0.05)
        assert abs(model.intercept_ - true_b) < 0.05

    def test_gd_converges_close_to_closed_form(self):
        X, y, _, _ = _make_linear_data(n=200, noise=0.01)
        gd = LinearRegression(solver="gd", learning_rate=0.05, n_iter=2000).fit(X, y)
        cf = LinearRegression(solver="closed_form").fit(X, y)
        np.testing.assert_allclose(gd.coef_, cf.coef_, atol=0.05)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LinearRegression().predict(np.zeros((3, 2)))

    def test_invalid_penalty_raises(self):
        with pytest.raises(ValueError):
            LinearRegression(penalty="elasticnet")  # not supported

    def test_l1_with_closed_form_raises(self):
        with pytest.raises(ValueError):
            LinearRegression(penalty="l1", solver="closed_form")

    def test_score_equals_r2(self):
        X, y, _, _ = _make_linear_data(n=100, noise=0.01)
        model = LinearRegression(solver="closed_form").fit(X, y)
        assert model.score(X, y) > 0.99


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------
class TestLogisticRegression:
    def test_high_accuracy_on_linearly_separable_blobs(self):
        X, y = _make_blobs(seed=0)
        model = LogisticRegression(learning_rate=0.1, n_iter=500).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_predict_proba_sums_to_one(self):
        X, y = _make_blobs(seed=0)
        model = LogisticRegression(n_iter=200).fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_non_binary_labels_raise(self):
        X = np.zeros((6, 2))
        y = np.array([0, 1, 2, 0, 1, 2])
        with pytest.raises(ValueError):
            LogisticRegression().fit(X, y)


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------
class TestPerceptron:
    def test_separates_blobs_perfectly(self):
        X, y = _make_blobs(seed=1)
        model = Perceptron(n_iter=50, random_state=0).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_more_than_two_classes_raises(self):
        X = np.zeros((9, 2))
        y = np.array([0, 1, 2] * 3)
        with pytest.raises(ValueError):
            Perceptron().fit(X, y)


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------
class TestKNN:
    def test_classifier_perfect_on_separable_blobs(self):
        X, y = _make_blobs(seed=2)
        model = KNNClassifier(n_neighbors=5).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_regressor_recovers_simple_function(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(200, 1))
        y = (X[:, 0] ** 2)
        model = KNNRegressor(n_neighbors=3).fit(X, y)
        # Test points well inside the training range
        X_test = np.array([[-2.0], [0.0], [2.0]])
        preds = model.predict(X_test)
        np.testing.assert_allclose(preds, [4.0, 0.0, 4.0], atol=1.0)

    def test_n_neighbors_larger_than_n_samples_raises(self):
        X, y = _make_blobs(n_per_class=2, seed=0)
        with pytest.raises(ValueError):
            KNNClassifier(n_neighbors=100).fit(X, y)

    def test_invalid_n_neighbors_raises(self):
        with pytest.raises(ValueError):
            KNNClassifier(n_neighbors=0)


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------
class TestDecisionTree:
    def test_perfect_fit_on_training_data(self):
        X, y = _make_blobs(seed=3)
        model = DecisionTreeClassifier(max_depth=10).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(max_depth=0)

    def test_invalid_min_samples_split_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(min_samples_split=1)


# ---------------------------------------------------------------------------
# Gaussian Naive Bayes
# ---------------------------------------------------------------------------
class TestGaussianNB:
    def test_high_accuracy_on_blobs(self):
        X, y = _make_blobs(seed=4)
        model = GaussianNB().fit(X, y)
        assert model.score(X, y) > 0.95

    def test_predict_proba_sums_to_one(self):
        X, y = _make_blobs(seed=4)
        model = GaussianNB().fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# MLP / Neural Network
# ---------------------------------------------------------------------------
class TestMLP:
    def test_learns_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        model = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            learning_rate=0.1,
            n_iter=2000,
            batch_size=4,
            random_state=0,
        ).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_invalid_hidden_size_raises(self):
        with pytest.raises(ValueError):
            MLPClassifier(hidden_layer_sizes=(0,))

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            MLPClassifier(activation="tanh")


# ---------------------------------------------------------------------------
# LinearSVC
# ---------------------------------------------------------------------------
class TestLinearSVC:
    def test_separates_blobs_perfectly(self):
        X, y = _make_blobs(seed=10)
        model = LinearSVC(C=1.0, learning_rate=0.01, n_iter=500).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_decision_function_sign_matches_predict(self):
        X, y = _make_blobs(seed=11)
        model = LinearSVC(n_iter=300).fit(X, y)
        scores = model.decision_function(X)
        preds = model.predict(X)
        # Where score >= 0, prediction should be the larger class (1);
        # where score < 0, the smaller class (0).
        np.testing.assert_array_equal(preds[scores >= 0], 1)
        np.testing.assert_array_equal(preds[scores < 0], 0)

    def test_accepts_pm1_labels(self):
        X, y = _make_blobs(seed=12)
        # Convert {0, 1} -> {-1, +1}
        y_pm = np.where(y == 0, -1, 1)
        model = LinearSVC(n_iter=300).fit(X, y_pm)
        # Predicted labels must be in the original {-1, +1} space.
        preds = model.predict(X)
        assert set(np.unique(preds).tolist()).issubset({-1, 1})
        assert model.score(X, y_pm) > 0.95

    def test_loss_history_decreases(self):
        X, y = _make_blobs(seed=13)
        model = LinearSVC(learning_rate=0.01, n_iter=200).fit(X, y)
        # The loss should generally trend downward; require the second half
        # of the history to be lower on average than the first half.
        history = np.array(model.loss_history_)
        assert history[100:].mean() < history[:100].mean()

    def test_more_than_two_classes_raises(self):
        X = np.zeros((9, 2))
        y = np.array([0, 1, 2] * 3)
        with pytest.raises(ValueError):
            LinearSVC().fit(X, y)

    def test_invalid_C_raises(self):
        with pytest.raises(ValueError):
            LinearSVC(C=0)
        with pytest.raises(ValueError):
            LinearSVC(C=-1)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LinearSVC().predict(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# RandomForestClassifier
# ---------------------------------------------------------------------------
class TestRandomForest:
    def test_high_accuracy_on_blobs(self):
        X, y = _make_blobs(seed=20)
        model = RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=0
        ).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_predict_proba_sums_to_one(self):
        X, y = _make_blobs(seed=21)
        model = RandomForestClassifier(
            n_estimators=10, random_state=0
        ).fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_generalizes_well_on_held_out_data(self):
        # The forest should achieve reasonable held-out accuracy on a
        # noisy linearly-separable problem.
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(200, 5))
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_test = rng.normal(size=(200, 5))
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

        forest = RandomForestClassifier(
            n_estimators=25, max_depth=8, max_features=None, random_state=0
        ).fit(X_train, y_train)
        # With all features available, the forest should easily beat 70%.
        assert forest.score(X_test, y_test) > 0.7

    def test_reproducible_with_seed(self):
        X, y = _make_blobs(seed=22)
        a = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
        b = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_invalid_n_estimators_raises(self):
        with pytest.raises(ValueError):
            RandomForestClassifier(n_estimators=0)

    def test_invalid_max_features_raises(self):
        with pytest.raises(ValueError):
            RandomForestClassifier(max_features="foo")
        with pytest.raises(ValueError):
            RandomForestClassifier(max_features=0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RandomForestClassifier().predict(np.zeros((3, 2)))
