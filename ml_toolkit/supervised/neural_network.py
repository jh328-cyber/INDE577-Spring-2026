"""
Multi-layer perceptron (feed-forward neural network) for classification.

Implements forward and backward propagation by hand using NumPy. Supports
arbitrary hidden layer sizes, ReLU or sigmoid activations on the hidden
layers, softmax + cross-entropy on the output, and full-batch / mini-batch
gradient descent.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1 - s)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


_ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
}


class MLPClassifier:
    """Multi-layer perceptron classifier trained with mini-batch SGD.

    The network is ``input -> [hidden_layers] -> softmax``. The hidden
    activation is applied after every hidden linear layer.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int, default=(32,)
        Sizes of the hidden layers.
    activation : {"relu", "sigmoid"}, default="relu"
        Hidden-layer activation function.
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iter : int, default=200
        Number of training epochs.
    batch_size : int, default=32
        Mini-batch size for SGD.
    random_state : int, optional
        Seed for parameter initialization and shuffling.

    Attributes
    ----------
    weights_ : list of np.ndarray
        Learned weight matrices, one per layer.
    biases_ : list of np.ndarray
        Learned bias vectors.
    classes_ : np.ndarray
        Sorted class labels seen in fit().
    loss_history_ : list of float
        Training cross-entropy loss after each epoch.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (32,),
        activation: Literal["relu", "sigmoid"] = "relu",
        learning_rate: float = 0.01,
        n_iter: int = 200,
        batch_size: int = 32,
        random_state: int | None = None,
    ) -> None:
        """Initialize the MLP classifier with the given architecture and hyperparameters."""
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        if any(h <= 0 for h in hidden_layer_sizes):
            raise ValueError("All hidden_layer_sizes must be positive integers.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    # ------------------------------------------------------------------
    def _init_params(self, n_features: int, n_classes: int) -> None:
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features, *self.hidden_layer_sizes, n_classes]
        self.weights_ = []
        self.biases_ = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialization works well for ReLU; it's also acceptable for sigmoid.
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0.0, scale, size=(layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            self.weights_.append(W)
            self.biases_.append(b)

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        Y = np.zeros((len(y), n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for row, label in enumerate(y):
            Y[row, class_to_idx[label]] = 1.0
        return Y

    def _forward(self, X: np.ndarray):
        """Run forward pass; return a list of pre-activations and activations."""
        act_fn, _ = _ACTIVATIONS[self.activation]
        a = X
        cache = {"a": [X], "z": []}
        # All hidden layers
        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            z = a @ W + b
            a = act_fn(z)
            cache["z"].append(z)
            cache["a"].append(a)
        # Output layer: linear -> softmax
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        a_out = _softmax(z_out)
        cache["z"].append(z_out)
        cache["a"].append(a_out)
        return cache

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train the MLP on (X, y) using mini-batch SGD and backpropagation."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self._init_params(n_features, n_classes)
        Y = self._one_hot(y)

        rng = np.random.default_rng(self.random_state)
        eps = 1e-12
        self.loss_history_ = []
        _, act_grad = _ACTIVATIONS[self.activation]

        for epoch in range(self.n_iter):
            # Shuffle training data each epoch
            perm = rng.permutation(n_samples)
            X_shuf, Y_shuf = X[perm], Y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                Xb, Yb = X_shuf[start:end], Y_shuf[start:end]
                cache = self._forward(Xb)
                a_out = cache["a"][-1]

                # ---- Backprop ----
                # Gradient at output (softmax + cross-entropy).
                m = len(Xb)
                delta = (a_out - Yb) / m  # shape (m, n_classes)

                grads_W = [None] * len(self.weights_)
                grads_b = [None] * len(self.biases_)
                # Output layer
                grads_W[-1] = cache["a"][-2].T @ delta
                grads_b[-1] = delta.sum(axis=0)
                # Hidden layers (back to front)
                for layer in range(len(self.weights_) - 2, -1, -1):
                    delta = (delta @ self.weights_[layer + 1].T) * act_grad(
                        cache["z"][layer]
                    )
                    grads_W[layer] = cache["a"][layer].T @ delta
                    grads_b[layer] = delta.sum(axis=0)

                # Gradient step
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.learning_rate * grads_W[i]
                    self.biases_[i] -= self.learning_rate * grads_b[i]

            # Track full-data training loss for diagnostics
            full_out = self._forward(X)["a"][-1]
            loss = -np.mean(np.sum(Y * np.log(full_out + eps), axis=1))
            self.loss_history_.append(float(loss))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probabilities for each row of X via a forward pass."""
        if not self.weights_:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return self._forward(X)["a"][-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X by taking the argmax over class probabilities."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the classification accuracy of the network on (X, y)."""
        from ml_toolkit.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
