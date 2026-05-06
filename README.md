# ml_toolkit — A From-Scratch Machine Learning Library

[![CI](https://github.com/jh328-cyber/INDE577-Spring-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/jh328-cyber/INDE577-Spring-2026/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Final project for **CMOR 438 / INDE 577 — Data Science & Machine Learning** (Spring 2026)
> Instructor: Dr. Randy R. Davila — Rice University

This repository contains a complete, from-scratch Python implementation of the supervised and
unsupervised learning algorithms taught in the course, plus a companion **examples** directory
of Jupyter notebooks that perform end-to-end data analysis on real datasets.

Every algorithm is implemented in pure NumPy. **No scikit-learn estimator is used as a
backbone** — scikit-learn appears only as a sanity reference in the example notebooks.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [What's implemented](#whats-implemented)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Running the tests](#running-the-tests)
6. [Continuous integration](#continuous-integration)
7. [Examples](#examples)
8. [Design notes](#design-notes)
9. [License](#license)

---

## Repository layout

```
INDE577-Spring-2026/
├── ml_toolkit/                  # The Python package itself
│   ├── __init__.py
│   ├── metrics.py               # Regression, classification, clustering metrics
│   ├── utils.py                 # train_test_split, StandardScaler, MinMaxScaler
│   ├── supervised/
│   │   ├── linear_regression.py # OLS / Ridge / Lasso (gradient descent + closed form)
│   │   ├── logistic_regression.py
│   │   ├── perceptron.py
│   │   ├── knn.py               # KNNClassifier and KNNRegressor
│   │   ├── decision_tree.py     # CART with Gini impurity
│   │   ├── random_forest.py     # Bagging ensemble of decision trees
│   │   ├── naive_bayes.py       # Gaussian Naive Bayes
│   │   ├── svm.py               # Linear soft-margin SVM (hinge loss)
│   │   └── neural_network.py    # MLPClassifier with backprop
│   └── unsupervised/
│       ├── kmeans.py            # k-means with k-means++ init
│       ├── pca.py               # PCA via SVD
│       ├── dbscan.py
│       └── hierarchical.py      # Single / complete / average linkage
├── tests/                       # Pytest suite (83 tests, 92% coverage)
├── examples/                    # Jupyter notebooks: one per algorithm
│   ├── 01_linear_regression/
│   ├── 02_logistic_regression/
│   └── ...
├── .github/workflows/ci.yml     # GitHub Actions CI (lint + tests on 3.10, 3.11, 3.12)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## What's implemented

### Supervised learning

| Algorithm | Class | File | Highlights |
|---|---|---|---|
| Linear regression | `LinearRegression` | `supervised/linear_regression.py` | Gradient descent **and** closed-form, optional L1/L2 |
| Logistic regression | `LogisticRegression` | `supervised/logistic_regression.py` | Numerically stable sigmoid, optional L2 |
| Perceptron | `Perceptron` | `supervised/perceptron.py` | Online updates, early stopping when separable |
| k-Nearest Neighbors | `KNNClassifier`, `KNNRegressor` | `supervised/knn.py` | Brute-force, Euclidean and Manhattan |
| Decision tree | `DecisionTreeClassifier` | `supervised/decision_tree.py` | CART with Gini impurity |
| Random forest | `RandomForestClassifier` | `supervised/random_forest.py` | Bagging + random feature subspaces, built on top of the decision tree |
| Gaussian Naive Bayes | `GaussianNB` | `supervised/naive_bayes.py` | Log-space scoring for numerical stability |
| Linear SVM | `LinearSVC` | `supervised/svm.py` | Soft-margin, sub-gradient descent on hinge loss + L2 |
| Multi-layer perceptron | `MLPClassifier` | `supervised/neural_network.py` | Manual backprop, ReLU/sigmoid, mini-batch SGD |

### Unsupervised learning

| Algorithm | Class | File | Highlights |
|---|---|---|---|
| k-Means | `KMeans` | `unsupervised/kmeans.py` | k-means++ init, multiple restarts |
| PCA | `PCA` | `unsupervised/pca.py` | SVD-based for numerical stability |
| DBSCAN | `DBSCAN` | `unsupervised/dbscan.py` | BFS expansion, noise label `-1` |
| Hierarchical | `HierarchicalClustering` | `unsupervised/hierarchical.py` | Single / complete / average linkage |

### Evaluation metrics (`ml_toolkit.metrics`)

- Regression: `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `r2_score`
- Classification: `accuracy_score`, `confusion_matrix`, `precision_recall_f1`
- Clustering: `silhouette_score`

### Utilities (`ml_toolkit.utils`)

- `train_test_split(X, y, test_size, random_state)`
- `StandardScaler`, `MinMaxScaler`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jh328-cyber/INDE577-Spring-2026.git
cd ml_toolkit

# (Recommended) create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install in editable mode with development extras
pip install -e ".[dev,examples]"
```

The package itself only requires **Python ≥ 3.10** and **NumPy ≥ 1.24**.

---

## Quick start

```python
import numpy as np
from ml_toolkit.supervised import LogisticRegression
from ml_toolkit.utils import StandardScaler, train_test_split
from ml_toolkit.metrics import accuracy_score, confusion_matrix

# Toy data
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split, scale, fit, evaluate
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

model = LogisticRegression(learning_rate=0.1, n_iter=500).fit(X_tr, y_tr)
print("Accuracy:", accuracy_score(y_te, model.predict(X_te)))
print("Confusion matrix:\n", confusion_matrix(y_te, model.predict(X_te)))
```

---

## Running the tests

```bash
pytest tests/ -v                              # run everything
pytest tests/test_supervised.py -v            # run one module
pytest tests/ --cov=ml_toolkit                # with coverage
```

The suite covers every public class with:

- **Functional checks** — does the algorithm recover known parameters / labels?
- **Edge cases** — empty inputs, mismatched shapes, invalid hyperparameters.
- **Exception handling** — every public method that has preconditions has at least one
  test that asserts it raises `ValueError` / `RuntimeError` when those preconditions are
  violated.

All tests are run automatically by GitHub Actions (see below).

---

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`. For every
combination of Python 3.10 / 3.11 / 3.12 it:

1. Checks out the code.
2. Installs the package and its dev dependencies.
3. Runs `ruff` for static analysis (non-blocking).
4. Runs the full pytest suite with coverage.
5. Uploads the coverage report as a build artifact.

The badge at the top of this README always reflects the latest run — green means every
algorithm and every test passes on every supported Python version.

---

## Examples

The `examples/` directory contains one Jupyter notebook per algorithm. Each notebook
follows the same template, in line with the analysis emphasis Dr. Davila highlighted in
class:

1. **Dataset description** — what is in it, where it comes from, what's the prediction target.
2. **Exploratory data analysis** — distributions, missing values, pairwise correlations.
3. **Hypothesis** — what we expect this algorithm to do well and what we expect to go wrong.
4. **Preprocessing** — train/test split, scaling, encoding.
5. **Model** — fit on train, predict on test.
6. **Evaluation** — task-appropriate metrics + visualization (confusion matrix, residuals,
   learning curve, etc.).
7. **Discussion** — what the metrics actually mean for this dataset, where the model
   succeeds and where it fails, and how that connects back to the assumptions of the
   algorithm.

See [`examples/README.md`](examples/README.md) for a full index.

---

## Design notes

A few intentional design choices worth pointing out:

- **scikit-learn-compatible API.** Every estimator exposes `fit`, `predict` (and
  `predict_proba` / `transform` where appropriate), so notebooks can switch between
  `ml_toolkit` and scikit-learn just by changing the import line — handy for sanity checks.
- **Numerical stability.** `LogisticRegression` and `MLPClassifier` use a piecewise sigmoid
  and softmax-with-max-subtraction. PCA uses SVD instead of computing the covariance
  matrix explicitly.
- **Pure NumPy.** No `for` loops over samples in any forward pass; every algorithm uses
  vectorized linear algebra wherever possible. The only sample-by-sample loop in the
  package is the perceptron's online update, which is part of the algorithm definition.
- **Reproducibility.** Every stochastic algorithm (perceptron shuffling, MLP weight init,
  k-means restart) accepts a `random_state` parameter and uses `numpy.random.default_rng`.

---

## License

MIT — see `LICENSE` (or feel free to use this code as a study reference).
