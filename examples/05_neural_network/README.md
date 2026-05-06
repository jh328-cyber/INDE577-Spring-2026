# 05 — Multi-Layer Perceptron on Handwritten Digits (8×8)

**Notebook:** [`mlp_digits.ipynb`](mlp_digits.ipynb)

A from-scratch MLP — manual forward pass, manual backpropagation, mini-batch SGD —
trained on the 8×8 handwritten digits dataset. The point of the notebook is to
demonstrate that an implementation built from one-line linear-algebra primitives can
reach ~97% accuracy, and to interpret the loss curve and the misclassified examples.

## Why this dataset?

It's the smallest realistic image-classification benchmark. 1797 examples × 64 features
fits in memory trivially and a 64-unit hidden layer trains to convergence in under 30
seconds of pure-Python NumPy. Big enough to be non-trivial, small enough to debug.

## Key takeaways

- ~95–98% test accuracy with one 64-unit hidden ReLU layer.
- Loss curve has the classic sharp-drop-then-asymptote shape.
- Misclassified digits are *humanly* ambiguous — a useful kind of error.
- Discussion explains why this network size is sensible (3 weights per example).
