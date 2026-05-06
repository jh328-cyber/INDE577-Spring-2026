# 02 — Logistic Regression on Breast Cancer Wisconsin

**Notebook:** [`logistic_regression_breast_cancer.ipynb`](logistic_regression_breast_cancer.ipynb)

This notebook is the **template** for every other algorithm notebook in this directory:
data description → EDA → hypothesis → preprocessing → model → evaluation → discussion.

## Why this dataset?

The Breast Cancer Wisconsin (Diagnostic) dataset is a well-studied binary classification
problem with a clinical interpretation, which forces us to think about *which* metric
matters (recall on the malignant class), not just overall accuracy. It is small enough
(569 samples, 30 features) that a from-scratch gradient-descent logistic regression
trains in well under a second.

## Key takeaways

- A simple linear classifier reaches **96.5% test accuracy** (110 / 114).
- The decision rule is meaningful: the strongest features all measure size and
  irregularity of the cell nuclei, and they correlate negatively with the benign label.
- Aggregate accuracy hides the metric we actually care about clinically — false
  negatives on malignant samples — and the analysis discusses how to adjust the
  decision threshold to address that.
