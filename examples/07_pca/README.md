# 07 — Principal Component Analysis on Digits

**Notebook:** [`pca_digits.ipynb`](pca_digits.ipynb)

PCA on the 8×8 handwritten digits dataset. The notebook answers two questions:
how many components carry most of the variance, and does a 2-D projection separate
digit classes that the algorithm never saw labels for? It also visualizes the
principal components themselves as 8×8 images.

## Why this dataset?

The digits are 64-dimensional, which is high enough that compression is meaningful but
low enough that we can plot every component. The fact that PCA — an unsupervised
method — produces a 2-D projection that visibly separates the (held-out) labels is the
clearest possible demonstration of why PCA is useful as a feature-extraction step.

## Key takeaways

- 21 components capture 80% of the variance; 40 capture 95%.
- 2-D projection separates 0 and 6 cleanly; 1 and 7 overlap, as predicted.
- The principal components themselves look like recognisable digit-stroke patterns
  when reshaped back to 8×8.
- PCA is linear; non-linear methods (t-SNE, UMAP) are the natural follow-up.
