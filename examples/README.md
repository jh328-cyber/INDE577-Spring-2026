# Examples — Algorithm Demonstrations

This directory hosts one Jupyter notebook per algorithm. Each notebook applies an
algorithm from `ml_toolkit` to a real (or realistic) dataset and walks through a full
data-analysis pipeline, **with a strong emphasis on interpretation rather than code**.

Every notebook follows the same seven-section template, in line with how Dr. Davila has
asked us to think about the project ("the analysis is what matters"):

1. **Dataset description** — what is in it, where it comes from, what we're predicting.
2. **Exploratory data analysis** — distributions, missing values, correlations.
3. **Hypothesis** — what we expect to happen and *why*, in terms of the algorithm's
   assumptions.
4. **Preprocessing** — train/test split, scaling, encoding.
5. **Model fitting** — `ml_toolkit` first, scikit-learn as a sanity reference.
6. **Evaluation on the test set** — task-appropriate metrics plus a visual diagnostic.
7. **Discussion** — does the result match the hypothesis? What does each metric *mean*
   for this dataset? Where does the model fail and why?

## Index

### Supervised

| Notebook | Algorithm | Dataset | Task type |
|---|---|---|---|
| [01_linear_regression](01_linear_regression/) | Linear regression | California Housing | Regression |
| [02_logistic_regression](02_logistic_regression/) | Logistic regression | Breast Cancer Wisconsin | Binary classification |
| [03_knn](03_knn/) | K-Nearest Neighbors | Wine | Multi-class classification |
| [04_decision_tree](04_decision_tree/) | Decision tree | Iris | Multi-class classification |
| [05_neural_network](05_neural_network/) | MLP | Digits (8x8) | Multi-class classification |
| [10_naive_bayes](10_naive_bayes/) | Gaussian NB | Iris | Multi-class classification |
| [12_svm](12_svm/) | Linear SVM | Breast Cancer Wisconsin | Binary classification |
| [13_random_forest](13_random_forest/) | Random forest | Wine | Multi-class classification |
| [14_perceptron](14_perceptron/) | Perceptron | Synthetic 2-D blobs | Binary classification (structural demo) |

### Unsupervised

| Notebook | Algorithm | Dataset | Task type |
|---|---|---|---|
| [06_kmeans](06_kmeans/) | k-means | Iris (features only) | Clustering |
| [07_pca](07_pca/) | PCA | Digits | Dimensionality reduction + visualization |
| [08_dbscan](08_dbscan/) | DBSCAN | Synthetic moons / blobs | Density-based clustering |
| [09_hierarchical](09_hierarchical/) | Agglomerative | Iris | Clustering |

### Bonus — relational / graph data

| Notebook | Topic | Dataset | Idea |
|---|---|---|---|
| [11_graph_bonus](11_graph_bonus/) | Similarity graph + connected components | Iris (re-used) | Bridges feature-vector ML with graph-data ML — converts a vector dataset into a similarity graph and recovers community structure with a graph-native algorithm |

## How to run

```bash
# From the repository root
pip install -e ".[examples]"
jupyter lab examples/
```

Each notebook is self-contained — you can run them in any order.
