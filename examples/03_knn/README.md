# 03 — K-Nearest Neighbors on Wine

**Notebook:** [`knn_wine.ipynb`](knn_wine.ipynb)

A multi-class demo that uses the Wine dataset to make one specific lesson concrete:
**KNN is only as good as its distance metric, and feature scaling is part of the
metric**. The notebook trains two identical KNN models — one on raw features, one on
standardized features — and shows the accuracy gap.

## Why this dataset?

Wine is small (178×13), balanced across three classes, and has features whose ranges
span four orders of magnitude. That last property is exactly what's needed to
demonstrate the scaling failure mode visibly, so the comparison shows real impact
rather than rounding noise.

## Key takeaways

- Raw KNN: 66.7% accuracy. Standardized KNN: 93.3% accuracy.
- 5-fold CV on the training set picks **k=1** (mean CV accuracy 0.970); accuracy is highest near k=1 and degrades steadily as k grows past ~10.
- The 3 test errors all fall on class 1 (the broadest cluster in the EDA), confirming the predicted boundary fragility.
- Discussion notes the O(n × d) per-query cost — KNN doesn't scale.
