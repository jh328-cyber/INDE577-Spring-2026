# 04 — Decision Tree on Iris

**Notebook:** [`decision_tree_iris.ipynb`](decision_tree_iris.ipynb)

The point of this notebook is **interpretability**. Iris is small and well-known so the
splits the tree learns can be checked directly against botanical knowledge. The
notebook also runs a depth sweep that makes overfitting visible as a literal gap on a
chart.

## Why this dataset?

Iris is small (150 rows), perfectly clean, and the discriminative features are
well-known to be petal length and petal width. That makes it the ideal sanity check
for a from-scratch CART implementation — if the tree's first split *isn't* on a petal
feature, the implementation is wrong.

## Key takeaways

- Tree's top split: petal length ≤ 1.9 cm — matches the EDA prediction.
- Test accuracy plateaus at depth 3; deeper trees overfit (train→100%, test flat).
- Errors concentrate on the versicolor/virginica boundary, as expected.
- No feature scaling needed for trees — splits are scale-invariant.
