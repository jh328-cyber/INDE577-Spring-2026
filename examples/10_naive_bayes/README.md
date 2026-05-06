# 10 — Gaussian Naive Bayes on Iris

**Notebook:** [`naive_bayes_iris.ipynb`](naive_bayes_iris.ipynb)

The repo's only generative model. The notebook starts by *checking the model
assumption* (per-class Gaussianity of features) before fitting, then connects
predictive performance to whether that assumption holds.

## Why this dataset?

Iris features are roughly normal within each species, which is the exact regime where
GNB is supposed to shine. It also lets us compare GNB head-to-head against the same
features used by the decision-tree and hierarchical notebooks.

## Key takeaways

- ~91% test accuracy — same ballpark as KNN, decision tree, logistic regression.
- The learned per-class Gaussians are themselves interpretable; plotted directly.
- The 'naive' independence assumption double-counts correlated features, producing
  over-confident probabilities — predictions OK, calibration not.
- Best applications: text (bag-of-words), tiny datasets, fast baselines.
