# 12 — Linear SVM on Breast Cancer Wisconsin

**Notebook:** [`svm_breast_cancer.ipynb`](svm_breast_cancer.ipynb)

A binary classification demo that contrasts a linear Support Vector Machine
with the logistic regression from notebook 02 on identical data. The point of
the analysis is **not** to declare a winner — SVM lands at 94.7% test accuracy, logistic regression at 96.5%, well within each other's noise band — but to demonstrate how the two losses produce different decision geometry.

## Why this dataset?

Re-using Breast Cancer Wisconsin lets us hold the data fixed and vary only the
*algorithm*. This is the cleanest way to see what "maximizing the margin"
actually means in practice.

## Key takeaways

- SVM test accuracy ≈ 94.7%, comparable to logistic regression.
- The two models **disagree on a tiny number of test points (~2 / 114)**, and
  those points lie *near the SVM's decision boundary* (mean |decision score|
  ≈ 0.18 on disagreements vs. ≈ 1.24 on agreements).
- This empirically shows the difference between hinge loss (zero contribution
  far from the margin) and cross-entropy (every point always pulls a little).
- Implementation note: standardization is mandatory because the SVM regularizer
  `½‖w‖²` is scale-sensitive.
