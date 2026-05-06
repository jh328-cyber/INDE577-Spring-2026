# 14 — Perceptron on Synthetic 2-D Blobs

**Notebook:** [`perceptron_blobs.ipynb`](perceptron_blobs.ipynb)

A *structural* demonstration of Rosenblatt's perceptron on two synthetic
datasets — one linearly separable, one not. The point of the notebook is to
show the **Perceptron Convergence Theorem** in action: the algorithm finishes
in finite time when a separating hyperplane exists, and it oscillates forever
when one doesn't.

## Why this dataset?

Real data is almost always *somewhere* between fully separable and fully
overlapping, which makes it impossible to demonstrate the theorem cleanly.
Synthetic 2-D Gaussian blobs let us *control* the separability by changing
`cluster_std` while keeping every other dimension of the experiment fixed,
so the contrast in the convergence curve is unambiguous and reproducible.

## Key takeaways

- **Separable** (cluster_std=0.5): converges in **2 epochs** (1 mistake → 0
  mistakes → early stop), 100% test accuracy.
- **Overlap** (cluster_std=1.2): runs all 50 epochs, mistake count oscillates
  between ~12 and ~24, 70.0% test accuracy.
- The `errors_` per-epoch curve is the textbook visualisation of the
  convergence theorem: separable → drops to zero, non-separable → never does.
- Why the perceptron stalls on non-separable data is *not* an optimization
  bug — it is a property of the model class. Logistic regression and SVM
  fix it by penalising margin instead of mistakes; the multi-layer
  perceptron fixes it by changing the model class.
