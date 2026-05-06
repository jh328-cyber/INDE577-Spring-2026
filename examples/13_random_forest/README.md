# 13 — Random Forest on Wine Recognition

**Notebook:** [`random_forest_wine.ipynb`](random_forest_wine.ipynb)

A multi-class classification demo that contrasts a single decision tree (notebook 04)
with an ensemble of 50 trees on the same algorithmic family. The analysis emphasizes
*why* and *when* an ensemble helps, not just *that* it helps.

## Why this dataset?

Wine has 178 samples, 13 chemical features, and 3 well-separated cultivars — small
enough to make the variance-reduction effect of bagging easy to see, but real enough
that the result is informative.

## Key takeaways

- Single-tree test accuracy ≈ 91% → 50-tree forest test accuracy ≈ 98%.
- Plotting test accuracy as a function of `n_estimators` shows the textbook
  diminishing-returns curve: most of the gain comes from the first 10–20 trees.
- Our implementation uses **per-tree feature subsetting** (random subspace method)
  rather than per-split feature subsetting; the comparison against scikit-learn's
  `RandomForestClassifier` confirms this simpler variant captures most of the
  benefit on this dataset.
- Trees are scale-invariant, so standardization is *not* needed (a key advantage
  over linear/SVM models).
