# 06 — K-Means on Iris

**Notebook:** [`kmeans_iris.ipynb`](kmeans_iris.ipynb)

Unsupervised clustering on Iris with k chosen via **both** the elbow method (inertia
vs. k) and the silhouette score, then a 2-D PCA projection used to visualise the
recovered clusters against the (held-out) species labels.

## Why this dataset?

Iris has known ground-truth labels we can compare against — but only after fitting,
so the algorithm doesn't peek. It also exposes a known surprise: silhouette score
prefers **k=2** even though the data has three biological species, because two of
them overlap in feature space and look like a single denser blob to a distance-based
score.

## Key takeaways

- Elbow plot points to k=3, silhouette plot points to k=2 — disagreement is the
  interesting result, not a bug.
- A best-permutation matching shows the clusters recover species at ~81% accuracy.
- PCA-2D projection makes the versicolor/virginica overlap visible directly.
- Discussion: silhouette and ground-truth-recovery measure *different things*, and
  unsupervised algorithms cannot in principle agree with biology unless biology
  matches density.
